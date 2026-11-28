"""Uncertainty-gated C++17 2-ply neural lookahead search for Orbit Wars.

Combines ``CppFastOrbitSim`` resident batch rollouts, ``evaluate_settled_shared_physics``
(which credits in-flight captures at arrival to eliminate short-horizon hold bias),
surgical borderline/tactical candidate generation, a single batched mid-ply neural
replan, Stage-2 rotational leaf refinement, and an asymmetric override gate so noisy
leaf estimates never overturn high-confidence ``8x D4`` greedy policy decisions.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orbit_wars_bot.fast_sim_cpp import HAS_ORBIT_SIM_CPP, CppFastOrbitSim
from orbit_wars_bot.physics import DEFAULT_EPISODE_STEPS, DEFAULT_SHIP_SPEED
from orbit_wars_bot.planet_transformer import (
    EDGE_FEAT_DIM,
    PLANET_FEAT_DIM,
    TIMELINE_HORIZON,
    EncodedPlanetState,
    SearchPlanetTransformerAgent,
    _obs_get,
    apply_d4_planet_feats,
    apply_d4_timeline_feats,
    build_shared_planet_physics,
    extract_planet_features,
)


def evaluate_settled_shared_physics(
    shared_physics: Dict[str, Any],
    player: int,
    *,
    max_steps: int = DEFAULT_EPISODE_STEPS,
    settle_ticks: int = 12,
) -> float:
    """Zero-hold-bias settled tactical value in ``[-1.0, 1.0]`` from ``shared_physics``.

    Projects each planet forward ``t_settle`` ticks along the precomputed C++ timeline
    (resolving all in-flight arrivals <= ``t_settle`` and skipping expired comets), then
    credits only surviving in-flight planet-bound fleets with ``eta > t_settle``.
    """
    fut_owner = shared_physics["fut_owner"]
    fut_ships = shared_physics["fut_ships"]
    tr_valid = shared_physics["tr_valid"]
    pprod = shared_physics["pprod"]
    is_comet_arr = shared_physics["is_comet_arr"]
    step = int(shared_physics["step"])

    n = int(shared_physics["n"])
    if n == 0 or fut_owner.shape[1] == 0:
        return 0.0

    horizon_avail = int(fut_owner.shape[1]) - 1
    rem_episode = max(0, int(max_steps) - step)
    t_settle = max(0, min(int(settle_ticks), horizon_avail, rem_episode))

    ships_by_p: Dict[int, float] = {}
    prod_by_p: Dict[int, float] = {}
    planets_by_p: Dict[int, int] = {}

    fo_col = fut_owner[:, t_settle]
    fs_col = fut_ships[:, t_settle]
    tv_col = tr_valid[:, t_settle]
    for i in range(n):
        if not bool(tv_col[i]):
            continue
        owner = int(fo_col[i])
        if owner < 0:
            continue
        ships_by_p[owner] = ships_by_p.get(owner, 0.0) + max(0.0, float(fs_col[i]))
        eff_prod = float(pprod[i]) * (0.35 if bool(is_comet_arr[i]) else 1.0)
        prod_by_p[owner] = prod_by_p.get(owner, 0.0) + eff_prod
        planets_by_p[owner] = planets_by_p.get(owner, 0) + 1

    # Add surviving planet-bound fleets arriving strictly after t_settle
    for arr_list in shared_physics["arrivals"].values():
        for a in arr_list:
            if a.eta > t_settle and a.owner >= 0:
                ships_by_p[a.owner] = ships_by_p.get(a.owner, 0.0) + float(a.ships)

    my_ships = ships_by_p.get(player, 0.0)
    opp_ids = [pid for pid, s in ships_by_p.items() if pid != player and s > 0.0]

    if my_ships <= 0.0 and opp_ids:
        return -1.0
    if not opp_ids and my_ships > 0.0:
        return 1.0
    if my_ships <= 0.0 and not opp_ids:
        return 0.0

    remaining = max(0, int(max_steps) - (step + t_settle))
    prod_horizon = min(36.0, float(remaining) * 0.65)

    def _power(pid: int) -> float:
        return (
            ships_by_p.get(pid, 0.0)
            + prod_by_p.get(pid, 0.0) * prod_horizon
            + planets_by_p.get(pid, 0) * 2.5
        )

    my_power = _power(player)
    opp_powers = [_power(pid) for pid in opp_ids]
    max_opp = max(opp_powers)
    if len(opp_powers) == 1:
        ref_opp = max_opp
    else:
        mean_opp = sum(opp_powers) / len(opp_powers)
        ref_opp = 0.72 * max_opp + 0.28 * mean_opp

    denom = max(10.0, my_power + ref_opp)
    advantage = (my_power - ref_opp) / denom
    return float(math.tanh(2.4 * advantage))


def _terminal_value(winner_pid: int, player: int) -> float:
    if winner_pid == player:
        return 1.0
    if winner_pid < 0:
        return 0.0
    return -1.0


class OrbitSearchAgent(SearchPlanetTransformerAgent):
    """Uncertainty-gated C++17 2-ply neural-guided online search agent."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.53,
        use_rot_tta: bool = True,
        use_d4_tta: bool = True,
        force_all_in: bool = True,
        rollout_horizon: int = 10,
        replan_step: int = 4,
        time_budget_sec: float = 0.21,
        max_candidates: int = 4,
        cliff_band: float = 0.07,
        alt_target_ratio: float = 0.45,
        neural_value_weight: float = 0.90,
        prior_bonus: float = 0.05,
        override_margin: float = 0.08,
        settle_ticks: int = 12,
        leaf_rot_refine: bool = True,
        joint_cliff_delta: float = 0.08,
        symmetric_value: bool = True,
        opp_launch_threshold: Optional[float] = None,
        use_cpp_sim: bool = True,
        device: str = "cpu",
    ) -> None:
        r_step = max(1, min(int(replan_step), max(2, int(rollout_horizon)) - 1))
        super().__init__(
            checkpoint_path=checkpoint_path,
            launch_threshold=launch_threshold,
            use_rot_tta=use_rot_tta,
            use_d4_tta=use_d4_tta,
            force_all_in=force_all_in,
            neural_mid_replan=True,
            rollout_horizon=rollout_horizon,
            time_budget_sec=time_budget_sec,
            neural_value_weight=neural_value_weight,
            replan_steps=(r_step,),
            max_candidates=max_candidates,
            prior_bonus=prior_bonus,
            symmetric_value=symmetric_value,
            opp_rot_tta=False,
            opp_launch_threshold=opp_launch_threshold,
            trajectory_val_blend=0.0,
            leaf_rot_tta_top_k=0,
            device=device,
        )
        self.replan_step = r_step
        self.cliff_band = float(cliff_band)
        self.alt_target_ratio = float(alt_target_ratio)
        self.override_margin = float(override_margin)
        self.settle_ticks = max(1, int(settle_ticks))
        self.leaf_rot_refine = bool(leaf_rot_refine)
        self.joint_cliff_delta = float(joint_cliff_delta)
        self.use_cpp_sim = bool(use_cpp_sim and HAS_ORBIT_SIM_CPP)

    def _build_surgical_candidates(
        self,
        enc: EncodedPlanetState,
        probs: np.ndarray,
        mean_fracs: np.ndarray,
        max_speed: float,
        shared_physics: Optional[Dict[str, Any]],
    ) -> Tuple[List[List[List[float]]], List[float], List[Optional[Dict[int, int]]], bool]:
        """Build greedy [0] plus uncertainty-gated surgical tactical/borderline plans."""
        n = len(enc.planet_ids)
        expiring_set = set(enc.expiring_comet_indices)
        thr = self.launch_threshold

        greedy_mv = self._decode_plan(
            enc,
            probs,
            mean_fracs,
            launch_threshold=thr,
            force_all_in=self.force_all_in,
            max_speed=max_speed,
        )
        greedy_lp = self._plan_log_prior(enc, probs, greedy_mv, overrides=None, launch_threshold=thr)

        candidates: List[List[List[float]]] = [greedy_mv]
        log_priors: List[float] = [greedy_lp]
        cand_overrides: List[Optional[Dict[int, int]]] = [None]
        seen_keys = {self._plan_key(greedy_mv)}

        proposals: List[Tuple[float, Dict[int, int]]] = []
        best_eta = shared_physics.get("best_eta") if shared_physics is not None else None
        has_futures = len(enc.futures) == n

        # 1. Urgent defensive rescue for highest-value owned planet flipping within 1..14 ticks
        if best_eta is not None and has_futures:
            greedy_dst: Dict[int, int] = {}
            for i in range(n):
                if enc.owned_mask[i] and np.any(enc.legal_edges[i]) and (1.0 - float(probs[i, 0]) >= thr):
                    ljs = np.flatnonzero(enc.legal_edges[i])
                    greedy_dst[i] = int(ljs[int(np.argmax(probs[i, 1 + ljs]))])

            best_rescue: Optional[Tuple[float, int, int]] = None
            for d_idx in range(n):
                if not enc.owned_mask[d_idx]:
                    continue
                fut_d = enc.futures[d_idx]
                flip_t = None
                for t_step in range(1, min(15, len(fut_d))):
                    if fut_d[t_step][0] != enc.player:
                        flip_t = t_step
                        break
                if flip_t is None:
                    continue
                val_d = float(enc.planet_rows[d_idx][6]) * 20.0 + float(enc.planet_rows[d_idx][5])
                for s_idx in range(n):
                    if s_idx == d_idx or (not enc.owned_mask[s_idx]) or (s_idx in expiring_set):
                        continue
                    if not enc.legal_edges[s_idx, d_idx] or greedy_dst.get(s_idx) == d_idx:
                        continue
                    eta_sd = int(best_eta[s_idx, d_idx])
                    if 1 <= eta_sd <= flip_t:
                        sc = val_d + float(enc.planet_rows[s_idx][5]) * 0.5
                        if best_rescue is None or sc > best_rescue[0]:
                            best_rescue = (sc, s_idx, d_idx)
            if best_rescue is not None:
                proposals.append((1e6 + best_rescue[0], {best_rescue[1]: 1 + best_rescue[2]}))

            # 4. Synchronized 2-planet focus-fire strike onto a contested non-friendly planet
            best_sync: Optional[Tuple[float, int, int, int]] = None
            for i1, dst in greedy_dst.items():
                if int(enc.planet_rows[dst][1]) == enc.player:
                    continue
                eta1 = int(best_eta[i1, dst])
                if eta1 <= 0:
                    continue
                sh1 = float(enc.planet_rows[i1][5])
                dst_prod = float(enc.planet_rows[dst][6])
                for i2 in range(n):
                    if i2 == i1 or (not enc.owned_mask[i2]) or (i2 in expiring_set):
                        continue
                    if not enc.legal_edges[i2, dst] or greedy_dst.get(i2) == dst:
                        continue
                    eta2 = int(best_eta[i2, dst])
                    if eta2 <= 0 or abs(eta2 - eta1) > 3:
                        continue
                    p_l2 = 1.0 - float(probs[i2, 0])
                    p_d2 = float(probs[i2, 1 + dst])
                    if p_l2 >= thr - 0.12 or p_d2 >= 0.10:
                        sc = (dst_prod + 1.0) * (sh1 + float(enc.planet_rows[i2][5]))
                        if best_sync is None or sc > best_sync[0]:
                            best_sync = (sc, i1, i2, dst)
            if best_sync is not None:
                proposals.append((1e5 + best_sync[0], {best_sync[1]: 1 + best_sync[3], best_sync[2]: 1 + best_sync[3]}))

        # 2 & 3. Borderline Launch<->Hold flips and credible 2nd-target switches
        single_flips: List[Tuple[float, int, int]] = []
        joint_launch_ov: Dict[int, int] = {}
        joint_hold_ov: Dict[int, int] = {}
        joint_launch_w = 0.0
        joint_hold_w = 0.0
        for i in range(n):
            if not enc.owned_mask[i] or i in expiring_set:
                continue
            legal_js = np.flatnonzero(enc.legal_edges[i])
            if len(legal_js) == 0:
                continue
            avail = float(enc.planet_rows[i][5])
            prod = float(enc.planet_rows[i][6])
            if avail <= 0.0:
                continue
            weight = avail + 3.0 * prod
            p_launch = 1.0 - float(probs[i, 0])
            order = legal_js[np.argsort(-probs[i, 1 + legal_js])]
            j1 = int(order[0])

            margin = abs(p_launch - thr)
            if margin <= self.cliff_band:
                alt_cls = 0 if p_launch >= thr else (1 + j1)
                sc = weight * (self.cliff_band + 0.02 - margin)
                single_flips.append((sc, i, alt_cls))
                proposals.append((sc, {i: alt_cls}))

            if self.joint_cliff_delta > 0.0:
                if (thr - self.joint_cliff_delta) <= p_launch < thr:
                    joint_launch_ov[i] = 1 + j1
                    joint_launch_w += weight
                elif thr <= p_launch <= (thr + self.joint_cliff_delta):
                    joint_hold_ov[i] = 0
                    joint_hold_w += weight

            if p_launch >= thr and len(order) >= 2:
                j2 = int(order[1])
                p1 = float(probs[i, 1 + j1])
                p2 = float(probs[i, 1 + j2])
                if p2 >= max(0.08, self.alt_target_ratio * p1):
                    sc2 = weight * p2
                    single_flips.append((sc2, i, 1 + j2))
                    proposals.append((sc2, {i: 1 + j2}))

        if joint_launch_ov:
            proposals.append((5e4 + joint_launch_w, joint_launch_ov))
        if joint_hold_ov:
            proposals.append((5e4 + joint_hold_w, joint_hold_ov))

        # 5. Top 2-planet pair flip if >= 2 distinct borderline planets exist
        single_flips.sort(key=lambda x: x[0], reverse=True)
        top_distinct: List[Tuple[int, int]] = []
        seen_p: set[int] = set()
        for _, p_idx, alt_cls in single_flips:
            if p_idx not in seen_p:
                seen_p.add(p_idx)
                top_distinct.append((p_idx, alt_cls))
            if len(top_distinct) >= 2:
                break
        if len(top_distinct) >= 2:
            p0, c0 = top_distinct[0]
            p1, c1 = top_distinct[1]
            proposals.append((single_flips[0][0] * 0.9, {p0: c0, p1: c1}))

        if not proposals:
            return candidates, log_priors, cand_overrides, False

        proposals.sort(key=lambda x: x[0], reverse=True)
        cap_k = min(self.max_candidates, 3 if n > 28 else self.max_candidates)

        for _, ov in proposals:
            mv = self._decode_plan(
                enc,
                probs,
                mean_fracs,
                launch_threshold=thr,
                force_all_in=self.force_all_in,
                max_speed=max_speed,
                overrides=ov,
            )
            key = self._plan_key(mv)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            lp = self._plan_log_prior(enc, probs, mv, overrides=ov, launch_threshold=thr)
            candidates.append(mv)
            log_priors.append(lp)
            cand_overrides.append(ov)
            if len(candidates) >= cap_k:
                break

        triggered = len(candidates) > 1
        return candidates, log_priors, cand_overrides, triggered

    def _init_sims(self, obs: Any, config: Any) -> Tuple[Any, bool]:
        if self.use_cpp_sim:
            return CppFastOrbitSim.from_observation(obs, config), True
        from orbit_wars_bot.fast_sim import FastOrbitSim

        return FastOrbitSim.from_observation(obs, config, use_cpp=False), False

    @staticmethod
    def _build_shared(sim: Any, player: int, max_speed: float, is_cpp: bool) -> Dict[str, Any]:
        if is_cpp:
            return sim.build_shared_physics(include_py_tracks=False)
        return build_shared_planet_physics(sim.to_observation(player=player), max_speed=max_speed)

    @staticmethod
    def _step_sims_burst(sims: Sequence[Any], acts_per_sim: Sequence[Any], n_empty_after: int, is_cpp: bool) -> None:
        if is_cpp:
            CppFastOrbitSim.step_batch(sims, acts_per_sim, n_empty_after=n_empty_after)
            return
        for sim, acts in zip(sims, acts_per_sim):
            if sim.done:
                continue
            sim.step(acts)
            if n_empty_after > 0 and not sim.done:
                n_agents = sim.num_agents
                empty = [[] for _ in range(n_agents)]
                for _ in range(n_empty_after):
                    if sim.done:
                        break
                    sim.step(empty)

    def _run_leaf_rot_refine(
        self,
        hero_opps: Dict[int, List[Tuple[EncodedPlanetState, int]]],
        finalists: Tuple[int, int],
        k_rots: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run batched rotated value_only=True evals across {0, best_alt} reusing edge_feats."""
        slots: List[Tuple[int, int]] = []
        pf_l: List[np.ndarray] = []
        tf_l: List[np.ndarray] = []
        ef_l: List[np.ndarray] = []
        le_l: List[np.ndarray] = []
        ns_l: List[int] = []

        for idx_c in finalists:
            for l_enc, sign in hero_opps.get(idx_c, ()):
                n_l = len(l_enc.planet_ids)
                if n_l == 0:
                    continue
                tf_base = l_enc.timeline_feats[..., : self.timeline_dim]
                for k_rot in k_rots:
                    pf_l.append(apply_d4_planet_feats(l_enc.planet_feats, k_rot=k_rot, flip_x=False))
                    tf_l.append(apply_d4_timeline_feats(tf_base, k_rot=k_rot, flip_x=False))
                    ef_l.append(l_enc.edge_feats)
                    le_l.append(l_enc.legal_edges)
                    ns_l.append(n_l)
                    slots.append((idx_c, sign))

        val_sum = np.zeros((2,), dtype=np.float64)
        val_cnt = np.zeros((2,), dtype=np.int32)
        if not slots:
            return val_sum, val_cnt

        b_tot = len(slots)
        max_nl = max(ns_l)
        if all(x == max_nl for x in ns_l):
            pf_t = torch.from_numpy(np.stack(pf_l, axis=0)).to(self.device)
            tf_t = torch.from_numpy(np.stack(tf_l, axis=0)).to(self.device)
            ef_t = torch.from_numpy(np.stack(ef_l, axis=0)).to(self.device)
            le_t = torch.from_numpy(np.stack(le_l, axis=0)).to(self.device)
            pm_t = torch.ones((b_tot, max_nl), dtype=torch.bool, device=self.device)
        else:
            pf_np = np.zeros((b_tot, max_nl, PLANET_FEAT_DIM), dtype=np.float32)
            tf_np = np.zeros((b_tot, max_nl, TIMELINE_HORIZON, self.timeline_dim), dtype=np.float32)
            ef_np = np.zeros((b_tot, max_nl, max_nl, EDGE_FEAT_DIM), dtype=np.float32)
            le_np = np.zeros((b_tot, max_nl, max_nl), dtype=bool)
            pm_np = np.zeros((b_tot, max_nl), dtype=bool)
            for ib, n_l in enumerate(ns_l):
                pf_np[ib, :n_l] = pf_l[ib]
                tf_np[ib, :n_l] = tf_l[ib]
                ef_np[ib, :n_l, :n_l] = ef_l[ib]
                le_np[ib, :n_l, :n_l] = le_l[ib]
                pm_np[ib, :n_l] = True
            pf_t = torch.from_numpy(pf_np).to(self.device)
            tf_t = torch.from_numpy(tf_np).to(self.device)
            ef_t = torch.from_numpy(ef_np).to(self.device)
            le_t = torch.from_numpy(le_np).to(self.device)
            pm_t = torch.from_numpy(pm_np).to(self.device)

        _, _, vals_t = self.model(
            pf_t, tf_t, ef_t, pm_t, legal_edges=le_t, compute_frac=False, value_only=True
        )
        pos_map = {finalists[0]: 0, finalists[1]: 1}
        for (idx_c, sign), v_raw in zip(slots, vals_t.cpu().numpy()):
            pos = pos_map[idx_c]
            val_sum[pos] += float(sign) * float(v_raw)
            val_cnt[pos] += 1
        return val_sum, val_cnt

    @torch.no_grad()
    def act_with_search_info(self, obs: Any, config: Any = None) -> Dict[str, Any]:
        """Run C++ 2-ply lookahead search and return chosen moves plus search metadata."""
        t_start = time.perf_counter()
        deadline = t_start + self.time_budget_sec
        max_speed = float(_obs_get(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config else DEFAULT_SHIP_SPEED

        player = int(_obs_get(obs, "player", 0))
        shared0 = build_shared_planet_physics(obs, max_speed=max_speed)
        enc = extract_planet_features(
            obs,
            player=player,
            max_speed=max_speed,
            feature_version=self.feature_version,
            shared_physics=shared0,
        )
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return {
                "moves": [],
                "chosen_idx": 0,
                "triggered": False,
                "candidates": [[]],
                "cand_overrides": [None],
                "scores": [0.0],
                "v_totals": [0.0],
                "log_priors": [0.0],
                "root_probs": np.zeros((n, n + 1), dtype=np.float32),
                "root_val": 0.0,
                "enc": enc,
            }

        probs, mean_fracs, root_val = self.predict_probs_and_value(enc, use_rot_tta=self.use_rot_tta)
        dt_root = time.perf_counter() - t_start
        candidates, log_priors, cand_overrides, triggered = self._build_surgical_candidates(
            enc, probs, mean_fracs, max_speed=max_speed, shared_physics=shared0
        )

        # Dynamic guard: if greedy root pass left too little budget for 2-ply search, return greedy immediately
        min_rem_needed = max(0.055, dt_root * 1.15)
        if (not triggered) or len(candidates) <= 1 or (time.perf_counter() >= deadline - min_rem_needed):
            return {
                "moves": candidates[0],
                "chosen_idx": 0,
                "triggered": bool(triggered),
                "candidates": candidates,
                "cand_overrides": cand_overrides,
                "scores": [0.0] * len(candidates),
                "v_totals": [float(root_val)] * len(candidates),
                "log_priors": log_priors,
                "root_probs": probs,
                "root_val": float(root_val),
                "enc": enc,
            }

        base_sim, is_cpp = self._init_sims(obs, config)
        n_players = base_sim.num_agents
        k_cand = len(candidates)

        # Compute turn-0 opponent responses ONCE reusing shared0
        opp_moves_t0: Dict[int, List[List[float]]] = {}
        opp_ids_active: List[int] = []
        opp_encs_t0: List[EncodedPlanetState] = []
        for opp_id in range(n_players):
            if opp_id == player:
                continue
            opp_enc = extract_planet_features(
                obs,
                player=opp_id,
                max_speed=max_speed,
                feature_version=self.feature_version,
                shared_physics=shared0,
            )
            if np.any(opp_enc.owned_mask):
                opp_ids_active.append(opp_id)
                opp_encs_t0.append(opp_enc)
            else:
                opp_moves_t0[opp_id] = []

        if opp_encs_t0:
            o_probs_l, o_fracs_l, _ = self.predict_batch(
                opp_encs_t0, compute_frac=not self.force_all_in, value_only=False
            )
            for opp_id, opp_enc, o_p, o_f in zip(opp_ids_active, opp_encs_t0, o_probs_l, o_fracs_l):
                opp_moves_t0[opp_id] = self._decode_plan(
                    opp_enc,
                    o_p,
                    o_f,
                    launch_threshold=self.opp_launch_threshold,
                    force_all_in=self.force_all_in,
                    max_speed=max_speed,
                )

        sims = [base_sim.clone() for _ in range(k_cand)]
        acts_t0: List[List[List[List[float]]]] = []
        for plan in candidates:
            ap = [[] for _ in range(n_players)]
            ap[player] = plan
            for opp_id, omv in opp_moves_t0.items():
                ap[opp_id] = omv
            acts_t0.append(ap)

        # Advance t=0 -> replan_step in a single C++ batch burst
        self._step_sims_burst(sims, acts_t0, n_empty_after=self.replan_step - 1, is_cpp=is_cpp)

        use_sym = self.symmetric_value and (n_players == 2)
        need_nn_leaf = self.neural_value_weight > 1e-6
        est_replan_sec = max(0.025, dt_root * 0.65 * (k_cand / 4.0) * (2 if n_players == 2 else 1) / 2.0)
        est_leaf_sec = (max(0.020, est_replan_sec * (0.85 if use_sym else 0.50))) if need_nn_leaf else 0.004

        rem_ticks = self.rollout_horizon - self.replan_step
        if rem_ticks > 0:
            if time.perf_counter() + est_replan_sec + est_leaf_sec < deadline:
                req_meta: List[Tuple[int, int, EncodedPlanetState]] = []
                step_acts_all: List[List[List[List[float]]]] = [
                    [[] for _ in range(n_players)] for _ in range(k_cand)
                ]
                for idx_c, sim in enumerate(sims):
                    if sim.done:
                        continue
                    sh = self._build_shared(sim, player, max_speed, is_cpp)
                    for pid_cur in range(n_players):
                        m_enc = extract_planet_features(
                            None,
                            player=pid_cur,
                            max_speed=max_speed,
                            feature_version=self.feature_version,
                            shared_physics=sh,
                        )
                        if np.any(m_enc.owned_mask):
                            req_meta.append((idx_c, pid_cur, m_enc))
                if req_meta:
                    m_probs_l, m_fracs_l, _ = self.predict_batch(
                        [m[2] for m in req_meta],
                        compute_frac=not self.force_all_in,
                        value_only=False,
                    )
                    for (idx_c, pid_cur, m_enc), m_p, m_f in zip(req_meta, m_probs_l, m_fracs_l):
                        step_acts_all[idx_c][pid_cur] = self._decode_plan(
                            m_enc,
                            m_p,
                            m_f,
                            launch_threshold=self.launch_threshold if pid_cur == player else self.opp_launch_threshold,
                            force_all_in=self.force_all_in,
                            max_speed=max_speed,
                        )
                self._step_sims_burst(sims, step_acts_all, n_empty_after=rem_ticks - 1, is_cpp=is_cpp)
            else:
                empty_all = [[[] for _ in range(n_players)] for _ in range(k_cand)]
                self._step_sims_burst(sims, empty_all, n_empty_after=rem_ticks - 1, is_cpp=is_cpp)

        # Leaf evaluation at t = rollout_horizon
        h_vals = np.zeros((k_cand,), dtype=np.float64)
        leaf_encs: List[EncodedPlanetState] = []
        leaf_slots: List[Tuple[int, int]] = []
        hero_opps: Dict[int, List[Tuple[EncodedPlanetState, int]]] = {}
        nn_val_sum = np.zeros((k_cand,), dtype=np.float64)
        nn_val_cnt = np.zeros((k_cand,), dtype=np.int32)

        for idx_c, sim in enumerate(sims):
            if sim.done:
                continue
            sh_leaf = self._build_shared(sim, player, max_speed, is_cpp)
            h_vals[idx_c] = evaluate_settled_shared_physics(
                sh_leaf,
                player,
                max_steps=base_sim.max_steps,
                settle_ticks=self.settle_ticks,
            )
            if need_nn_leaf:
                hero_enc = extract_planet_features(
                    None,
                    player=player,
                    max_speed=max_speed,
                    feature_version=self.feature_version,
                    shared_physics=sh_leaf,
                )
                leaf_encs.append(hero_enc)
                leaf_slots.append((idx_c, 1))
                hero_opps[idx_c] = [(hero_enc, 1)]
                if use_sym:
                    opp_enc = extract_planet_features(
                        None,
                        player=1 - player,
                        max_speed=max_speed,
                        feature_version=self.feature_version,
                        shared_physics=sh_leaf,
                    )
                    leaf_encs.append(opp_enc)
                    leaf_slots.append((idx_c, -1))
                    hero_opps[idx_c].append((opp_enc, -1))

        if need_nn_leaf and leaf_encs and time.perf_counter() < deadline:
            t_lv = time.perf_counter()
            _, _, leaf_vals = self.predict_batch(leaf_encs, compute_frac=False, value_only=True)
            dt_leaf_batch = time.perf_counter() - t_lv
            for (idx_c, sign), v_raw in zip(leaf_slots, leaf_vals):
                nn_val_sum[idx_c] += float(sign) * float(v_raw)
                nn_val_cnt[idx_c] += 1
        else:
            dt_leaf_batch = est_leaf_sec

        base_lp = log_priors[0]
        v_totals = [0.0] * k_cand
        scores = [0.0] * k_cand

        def _update_score(idx_c: int) -> None:
            sim = sims[idx_c]
            if sim.done:
                v_tot = _terminal_value(sim.winner(), player)
            elif need_nn_leaf and nn_val_cnt[idx_c] > 0:
                nn_eff = float(nn_val_sum[idx_c] / nn_val_cnt[idx_c])
                v_tot = self.neural_value_weight * nn_eff + (1.0 - self.neural_value_weight) * float(h_vals[idx_c])
            else:
                v_tot = float(h_vals[idx_c])
            v_totals[idx_c] = v_tot
            scores[idx_c] = v_tot + self.prior_bonus * (log_priors[idx_c] - base_lp)

        for idx_c in range(k_cand):
            _update_score(idx_c)

        best_alt = 1 + int(np.argmax(scores[1:]))

        # Stage-2 Rotational Variance-Cancellation on {0, best_alt}
        if (
            self.leaf_rot_refine
            and need_nn_leaf
            and scores[best_alt] > scores[0] + 0.5 * self.override_margin
            and (0 in hero_opps)
            and (best_alt in hero_opps)
            and (not sims[0].done)
            and (not sims[best_alt].done)
        ):
            per_slot_cost = dt_leaf_batch / max(1, len(leaf_slots))
            n_pair_slots = len(hero_opps[0]) + len(hero_opps[best_alt])
            rem_sec = deadline - time.perf_counter()
            if rem_sec >= per_slot_cost * n_pair_slots * 3.0 + 0.004:
                k_rots: Tuple[int, ...] = (1, 2, 3)
            elif rem_sec >= per_slot_cost * n_pair_slots + 0.004:
                k_rots = (2,)
            else:
                k_rots = ()

            if k_rots:
                add_sum, add_cnt = self._run_leaf_rot_refine(hero_opps, (0, best_alt), k_rots)
                for pos, idx_c in enumerate((0, best_alt)):
                    nn_val_sum[idx_c] += add_sum[pos]
                    nn_val_cnt[idx_c] += int(add_cnt[pos])
                    _update_score(idx_c)

        # Asymmetric Override Gate: keep high-confidence greedy [0] unless best_alt beats it cleanly
        chosen_idx = best_alt if scores[best_alt] >= scores[0] + self.override_margin else 0
        return {
            "moves": candidates[chosen_idx],
            "chosen_idx": chosen_idx,
            "triggered": True,
            "candidates": candidates,
            "cand_overrides": cand_overrides,
            "scores": scores,
            "v_totals": v_totals,
            "log_priors": log_priors,
            "root_probs": probs,
            "root_val": float(root_val),
            "enc": enc,
        }

    @torch.no_grad()
    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        return self.act_with_search_info(obs, config)["moves"]


def create_orbit_search_agent(
    checkpoint_path: str | Path,
    **kwargs: Any,
):
    """Factory returning a Kaggle ``agent(obs, config)`` callable powered by ``OrbitSearchAgent``."""
    agent_obj = OrbitSearchAgent(checkpoint_path=checkpoint_path, **kwargs)

    def _kaggle_agent(obs: Any, config: Any = None) -> List[List[float]]:
        return agent_obj.act(obs, config)

    return _kaggle_agent
