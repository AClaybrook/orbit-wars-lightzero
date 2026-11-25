#!/usr/bin/env python3
"""Gen 8 Orbit Wars trainer combining all 3 Top-10 ROI tricks:

1. **Full Soft-Logit Knowledge Distillation (`KL(q_hober || p_ours)`)**:
   Queries `#2 Hober Malloc`'s local JAX policy (`rotation_tta=True` = exact full 2P `1800.1`
   probabilities!) directly in Python to obtain the full soft distribution
   `q_i(HOLD | target_0..N-1)` for every owned planet — ~10x richer signal than 1-hot argmax!
2. **Auxiliary Future-State World-Model Prediction (`#7 Audun` trick)**:
   Trains a lightweight training-only head (`aux_future_head`, zero inference cost) to predict
   every planet's true post-action owner & ship count 10 ticks into the future (`t + 10`).
3. **Dedicated 2P Specialist + All-In Execution (`#5 TonyK` & `#2 SimJeg` tricks)**:
   Trains purely on high-signal 2P boards (Hober-vs-Hober + on-policy student-vs-Hober DAgger)
   for maximum 2P sharpness against `#2 Hober Malloc (1800.1)`.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
HOBER_DIR = REPO_ROOT / "external" / "opponents" / "simjeg_hober_malloc"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orbit_wars_bot.dml_utils import (
    build_optimizer,
    cpu_state_dict,
    one_hot_no_scatter,
    resolve_train_device,
)
from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.planet_transformer import (
    MAX_PLANETS,
    EncodedPlanetState,
    PlanetTimelineTransformer,
    apply_d4_planet_feats,
    apply_d4_timeline_feats,
    extract_planet_features,
    pad_encoded_state,
)


class HoberSoftTeacher:
    """In-process JAX query wrapper for `#2 Hober Malloc`'s exact soft action probabilities."""

    def __init__(self, *, rotation_tta: bool = True, opponent_tta: bool = False) -> None:
        if str(HOBER_DIR) not in sys.path:
            sys.path.insert(0, str(HOBER_DIR))
        os.environ["OW_MODEL_PATH"] = str(HOBER_DIR / "checkpoint" / "weight.npz")
        os.environ["OW_ROTATION_TTA"] = "1" if rotation_tta else "0"
        os.environ["OW_OPPONENT_TTA"] = "1" if opponent_tta else "0"
        os.environ["OW_MAX_TIMESTEPS"] = "20"

        from ow_runtime.policy_server import (
            Policy,
            jax_runtime,
            prepare_request_inputs,
            scatter_compact_outputs,
        )

        self._prepare = prepare_request_inputs
        self._scatter = scatter_compact_outputs
        self._jax, self._jnp, self._model_fn = jax_runtime()
        self.policy = Policy()

    def _build_hober_states(self, obs: Dict[str, Any], config: Any, horizon: int = 20) -> List[List[Any]]:
        """Run an exact 20-step no-action FastOrbitSim rollout to build Hober Malloc's `states` table."""
        sim = FastOrbitSim.from_observation(obs, config)
        all_pids = sorted(int(p[0]) for p in sim.planets)
        owners_map: Dict[int, List[int]] = {pid: [] for pid in all_pids}
        ships_map: Dict[int, List[int]] = {pid: [] for pid in all_pids}
        valid_map: Dict[int, List[bool]] = {pid: [] for pid in all_pids}
        empty_acts = [[] for _ in range(sim.num_agents)]

        for t in range(horizon):
            by_id = {int(p[0]): p for p in sim.planets}
            for pid in all_pids:
                p = by_id.get(pid)
                if p is not None:
                    owners_map[pid].append(int(p[1]))
                    ships_map[pid].append(int(p[5]))
                    valid_map[pid].append(True)
                else:
                    owners_map[pid].append(-1)
                    ships_map[pid].append(0)
                    valid_map[pid].append(False)
            if t + 1 < horizon:
                sim.step(empty_acts)

        return [[pid, owners_map[pid], ships_map[pid], valid_map[pid]] for pid in all_pids]

    def query_soft_probs(
        self,
        obs: Dict[str, Any],
        enc: EncodedPlanetState,
        config: Any = None,
    ) -> np.ndarray:
        """Return exact `(MAX_PLANETS, MAX_PLANETS + 1)` soft teacher probabilities aligned with `enc`."""
        n = len(enc.planet_ids)
        soft = np.zeros((MAX_PLANETS, MAX_PLANETS + 1), dtype=np.float32)
        soft[:, 0] = 1.0
        if n == 0 or not np.any(enc.owned_mask):
            return soft

        # Build legal candidate pairs from enc.legal_edges (ETA <= 20, sun-clear)
        candidates: List[List[int]] = []
        for i in range(n):
            if not enc.owned_mask[i]:
                continue
            src_pid = int(enc.planet_ids[i])
            for j in range(n):
                if enc.legal_edges[i, j]:
                    candidates.append([src_pid, int(enc.planet_ids[j])])
        if not candidates:
            return soft

        req = {
            "id": 1,
            "player": int(enc.player),
            "observation": obs,
            "states": self._build_hober_states(obs, config, horizon=20),
            "candidates": candidates,
        }
        prepared = self._prepare(
            req,
            rotation_tta=self.policy.rotation_tta,
            opponent_tta=self.policy.opponent_tta,
            max_timesteps=self.policy.max_timesteps,
        )
        if prepared.get("status") != "OK":
            return soft

        x = prepared["x"]
        model_body_slots = int(x.shape[1])
        batch_size = int(prepared["batch_size"])
        out = self._model_fn(
            self.policy.params,
            self._jnp.asarray(x, dtype=self._jnp.bfloat16),
            time_mask=self._jnp.asarray(prepared["time_mask"]),
            body_mask=self._jnp.asarray(prepared["body_mask"]),
            launch_mask=self._jnp.asarray(prepared["launch_mask"]),
            target_mask=self._jnp.asarray(prepared["target_mask"]),
        )
        launch_logits = out.launch_logits.astype(self._jnp.float32)
        target_logits = out.target_logits.astype(self._jnp.float32)
        if batch_size > 1:
            launch_logits = launch_logits.reshape((1, batch_size, model_body_slots)).mean(axis=1)
            target_logits = target_logits.reshape((1, batch_size, model_body_slots, model_body_slots)).mean(axis=1)
        else:
            launch_logits = launch_logits.reshape((1, model_body_slots))
            target_logits = target_logits.reshape((1, model_body_slots, model_body_slots))

        launch_prob = np.asarray(self._jax.nn.sigmoid(launch_logits), dtype=np.float32)[0]
        target_logits_np = np.asarray(target_logits, dtype=np.float32)[0]
        lp_full, tl_full = self._scatter(
            launch_prob, target_logits_np, prepared["active_indices"], prepared["n_bodies"]
        )

        hober_indices = prepared["indices"]  # planet_id -> hober_slot_idx
        tmask_2d = np.asarray(prepared["full_target_mask"][0], dtype=bool)
        id_to_enc_idx = {int(pid): idx for idx, pid in enumerate(enc.planet_ids)}

        for i in range(n):
            if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                continue
            src_pid = int(enc.planet_ids[i])
            h_src = hober_indices.get(src_pid)
            if h_src is None or h_src < 0 or h_src >= prepared["n_bodies"]:
                continue

            row_mask = tmask_2d[h_src]
            if not bool(row_mask.any()):
                continue
            row_logits = tl_full[h_src]
            masked = np.where(row_mask, row_logits, -np.inf)
            mx = float(np.max(masked))
            if not np.isfinite(mx):
                continue
            exps = np.where(row_mask, np.exp(masked - mx), 0.0)
            denom = float(np.sum(exps))
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            t_probs = exps / denom

            p_launch = float(np.clip(lp_full[h_src], 1e-4, 1.0 - 1e-4))
            soft[i, :] = 0.0
            soft[i, 0] = 1.0 - p_launch

            # Scatter target probabilities onto our exact enc planet indices
            mapped_sum = 0.0
            for dst_pid, h_dst in hober_indices.items():
                if 0 <= h_dst < prepared["n_bodies"] and row_mask[h_dst]:
                    j = id_to_enc_idx.get(int(dst_pid))
                    if j is not None and enc.legal_edges[i, j]:
                        p_j = float(t_probs[h_dst])
                        soft[i, 1 + j] = p_launch * p_j
                        mapped_sum += p_j
            # Renormalize cleanly in case a comet slot wasn't mapped
            row_tot = float(soft[i, : n + 1].sum())
            if row_tot > 0.0:
                soft[i, : n + 1] /= row_tot
            else:
                soft[i, :] = 0.0
                soft[i, 0] = 1.0

        return soft


def _extract_aux_future_targets(
    enc_t: EncodedPlanetState,
    obs_future: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build true `t + 10` auxiliary world-model targets aligned to `enc_t.planet_ids`."""
    aux_owner = np.full((MAX_PLANETS,), 2, dtype=np.int64)  # 0=mine, 1=enemy, 2=neutral
    aux_ships = np.zeros((MAX_PLANETS,), dtype=np.float32)
    aux_valid = np.zeros((MAX_PLANETS,), dtype=bool)
    if obs_future is None:
        return aux_owner, aux_ships, aux_valid

    fut_by_id = {int(row[0]): (int(row[1]), int(row[5])) for row in obs_future.get("planets", [])}
    player = int(enc_t.player)
    for i, pid in enumerate(enc_t.planet_ids):
        rec = fut_by_id.get(int(pid))
        if rec is None:
            continue
        owner, ships = rec
        aux_owner[i] = 0 if owner == player else (2 if owner < 0 else 1)
        aux_ships[i] = float(np.log1p(max(0, ships)) / 6.0)
        aux_valid[i] = True
    return aux_owner, aux_ships, aux_valid


_WORKER_CACHE: Dict[str, Any] = {}


def _collect_soft_kd_worker(
    task: Tuple[int, bool, Optional[str], str, int, int, int, int, int, bool, int, int]
) -> Dict[str, Any]:
    from kaggle_environments import make

    from scripts.evaluate import resolve_agent

    torch.set_num_threads(1)
    (
        seed,
        is_dagger,
        student_checkpoint,
        student_spec_prefix,
        sample_stride,
        dense_start_tick,
        dense_end_tick,
        future_horizon,
        fast_rollout_opp,
        record_opp_slot,
        max_collect_tick,
        feature_version,
    ) = task

    # Instantiate full-TTA HoberSoftTeacher first so its env defaults do not overwrite
    # the rollout opponent env vars configured immediately below.
    if "teacher" not in _WORKER_CACHE:
        _WORKER_CACHE["teacher"] = HoberSoftTeacher(rotation_tta=True, opponent_tta=False)

    if fast_rollout_opp:
        os.environ["OW_ROTATION_TTA"] = "0"
        os.environ["OW_OPPONENT_TTA"] = "0"
        os.environ["ORBIT_WARS_BC_ALL_IN_ROTATION_TTA"] = "0"
    else:
        os.environ["OW_ROTATION_TTA"] = "1"
        os.environ["OW_OPPONENT_TTA"] = "0"
        os.environ["ORBIT_WARS_BC_ALL_IN_ROTATION_TTA"] = "1"

    if "boss_agent" not in _WORKER_CACHE:
        _WORKER_CACHE["boss_agent"] = resolve_agent("hober_malloc")
    stu_key = f"stu:{student_spec_prefix}{student_checkpoint}"
    if is_dagger and student_checkpoint and stu_key not in _WORKER_CACHE:
        _WORKER_CACHE[stu_key] = resolve_agent(f"{student_spec_prefix}{student_checkpoint}")

    boss_agent = _WORKER_CACHE["boss_agent"]
    teacher = _WORKER_CACHE["teacher"]
    student_agent = _WORKER_CACHE.get(stu_key) if (is_dagger and student_checkpoint) else None

    random.seed(seed)
    np.random.seed(seed)
    env = make("orbit_wars", configuration={"seed": seed, "actTimeout": 30, "agentTimeout": 60}, debug=False)
    if is_dagger and student_agent is not None:
        stu_slot = seed % 2
        roster = [student_agent, boss_agent] if stu_slot == 0 else [boss_agent, student_agent]
        record_slots = [stu_slot, 1 - stu_slot] if record_opp_slot else [stu_slot]
    else:
        stu_slot = -1
        roster = [boss_agent, boss_agent]
        record_slots = [0, 1]

    steps = env.run(roster)
    n_ticks = len(steps) - 1
    rewards = [steps[-1][0]["reward"], steps[-1][1]["reward"]]

    pf_l, tf_l, ef_l, le_l, om_l, pm_l = [], [], [], [], [], []
    sp_l, vt_l, ao_l, as_l, av_l = [], [], [], [], []
    agree_launch = tot_dec = 0

    for slot in record_slots:
        r_self = rewards[slot] or 0
        r_opp = rewards[1 - slot] or 0
        val_target = 1.0 if r_self > r_opp else (-1.0 if r_self < r_opp else 0.0)
        is_counter_opp = is_dagger and stu_slot >= 0 and slot != stu_slot

        tick_limit = min(n_ticks, max_collect_tick) if max_collect_tick > 0 else n_ticks
        for t in range(0, tick_limit):
            obs_t = steps[t][slot]["observation"]
            has_comets = bool(obs_t.get("comets"))
            in_dense = dense_start_tick <= t <= dense_end_tick
            if is_counter_opp:
                keep = ((in_dense or has_comets) and (t % 2 == 0)) or (t % max(1, sample_stride * 2) == 0)
            else:
                stride = 1 if (is_dagger and val_target < 0 and sample_stride > 1) else sample_stride
                keep = in_dense or has_comets or (t % stride == 0)
            if not keep:
                continue
            obs_t_copy = dict(obs_t)
            obs_t_copy["player"] = slot
            obs_t_copy["step"] = int(steps[t][0]["observation"].get("step", t))
            enc = extract_planet_features(obs_t_copy, player=slot, feature_version=feature_version)
            if not enc.planet_ids or not np.any(enc.owned_mask):
                continue

            soft_probs = teacher.query_soft_probs(obs_t_copy, enc, env.configuration)
            pad = pad_encoded_state(enc)
            obs_fut = steps[t + future_horizon][slot]["observation"] if (t + future_horizon < len(steps)) else None
            ao, ash, av = _extract_aux_future_targets(enc, obs_fut)

            if not is_dagger:
                actual_acts = steps[t + 1][slot].get("action") or []
                actual_srcs = {int(m[0]) for m in actual_acts}
                for i, pid in enumerate(enc.planet_ids):
                    if enc.owned_mask[i] and np.any(enc.legal_edges[i]):
                        pred_launch = (1.0 - float(soft_probs[i, 0])) >= 0.53
                        actual_launch = int(pid) in actual_srcs
                        tot_dec += 1
                        if pred_launch == actual_launch:
                            agree_launch += 1

            pf_l.append(pad["planet_feats"])
            tf_l.append(pad["timeline_feats"])
            ef_l.append(pad["edge_feats"])
            le_l.append(pad["legal_edges"])
            om_l.append(pad["owned_mask"])
            pm_l.append(pad["pad_mask"])
            sp_l.append(soft_probs.astype(np.float16))
            vt_l.append(val_target)
            ao_l.append(ao)
            as_l.append(ash)
            av_l.append(av)

    return {
        "planet_feats": np.stack(pf_l).astype(np.float32),
        "timeline_feats": np.stack(tf_l).astype(np.float32),
        "edge_feats": np.stack(ef_l).astype(np.float32),
        "legal_edges": np.stack(le_l),
        "owned_mask": np.stack(om_l),
        "pad_mask": np.stack(pm_l),
        "soft_probs": np.stack(sp_l).astype(np.float16),
        "val_targets": np.array(vt_l, dtype=np.float32),
        "aux_owner": np.stack(ao_l).astype(np.int64),
        "aux_ships": np.stack(as_l).astype(np.float32),
        "aux_valid": np.stack(av_l),
        "agree_launch": agree_launch,
        "tot_dec": tot_dec,
    }


def collect_soft_kd_dataset(
    *,
    hober_games: int = 2,
    dagger_games: int = 16,
    student_checkpoint: Optional[str] = None,
    student_spec_prefix: str = "pt_d4_allin:",
    seed: int = 12000,
    sample_stride: int = 3,
    dense_start_tick: int = 22,
    dense_end_tick: int = 78,
    future_horizon: int = 10,
    workers: int = 1,
    full_boss_rollout: bool = False,
    record_opp_slot: bool = False,
    max_collect_tick: int = 0,
    feature_version: int = 1,
) -> Dict[str, np.ndarray]:
    """Collect 2P games with exact Full-TTA `#2 Hober Malloc` soft probabilities + `t+10` future targets."""
    import multiprocessing as mp

    fast_opp = 0 if full_boss_rollout else 1
    tasks: List[Tuple[int, bool, Optional[str], str, int, int, int, int, int, bool, int, int]] = []
    if student_checkpoint:
        for g in range(dagger_games):
            tasks.append((
                seed + 1000 + g,
                True,
                student_checkpoint,
                student_spec_prefix,
                sample_stride,
                dense_start_tick,
                dense_end_tick,
                future_horizon,
                fast_opp,
                record_opp_slot,
                max_collect_tick,
                feature_version,
            ))
    for g in range(hober_games):
        tasks.append((
            seed + g,
            False,
            None,
            student_spec_prefix,
            sample_stride,
            dense_start_tick,
            dense_end_tick,
            future_horizon,
            fast_opp,
            record_opp_slot,
            max_collect_tick,
            feature_version,
        ))

    t0 = time.time()
    parts: List[Dict[str, Any]] = []
    tot_states = agree_launch = tot_dec = 0
    print(f"[SoftKD Collect] Running {len(tasks)} 2P games (workers={workers}, exact full-TTA soft probs)...", flush=True)

    if workers <= 1:
        for idx, task in enumerate(tasks, 1):
            res = _collect_soft_kd_worker(task)
            parts.append(res)
            tot_states += len(res["val_targets"])
            agree_launch += res["agree_launch"]
            tot_dec += res["tot_dec"]
            pstr = f" | parity={100.0*agree_launch/max(1,tot_dec):.1f}%" if tot_dec else ""
            print(f"  [SoftKD Collect] {idx}/{len(tasks)} ({time.time()-t0:.1f}s) | states={tot_states}{pstr}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for idx, res in enumerate(pool.imap_unordered(_collect_soft_kd_worker, tasks), 1):
                parts.append(res)
                tot_states += len(res["val_targets"])
                agree_launch += res["agree_launch"]
                tot_dec += res["tot_dec"]
                pstr = f" | parity={100.0*agree_launch/max(1,tot_dec):.1f}%" if tot_dec else ""
                print(f"  [SoftKD Collect] {idx}/{len(tasks)} ({time.time()-t0:.1f}s) | states={tot_states}{pstr}", flush=True)

    keys = [k for k in parts[0].keys() if k not in ("agree_launch", "tot_dec")]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def _trim_soft_kd_batch_n(
    pf: torch.Tensor,
    tf: torch.Tensor,
    ef: torch.Tensor,
    le: torch.Tensor,
    om: torch.Tensor,
    pm: torch.Tensor,
    sp: torch.Tensor,
    ao: torch.Tensor,
    ash: torch.Tensor,
    av: torch.Tensor,
):
    max_n = max(2, int(pm.sum(dim=1).max().item()))
    if max_n < pf.shape[1]:
        pf = pf[:, :max_n]
        tf = tf[:, :max_n]
        ef = ef[:, :max_n, :max_n]
        le = le[:, :max_n, :max_n]
        om = om[:, :max_n]
        pm = pm[:, :max_n]
        sp = sp[:, :max_n, : max_n + 1]
        ao = ao[:, :max_n]
        ash = ash[:, :max_n]
        av = av[:, :max_n]
    return pf, tf, ef, le, om, pm, sp, ao, ash, av


def compute_soft_kd_loss(
    model: PlanetTimelineTransformer,
    pf: torch.Tensor,
    tf: torch.Tensor,
    ef: torch.Tensor,
    le: torch.Tensor,
    om: torch.Tensor,
    pm: torch.Tensor,
    soft_probs: torch.Tensor,
    vtar: torch.Tensor,
    aux_owner: torch.Tensor,
    aux_ships: torch.Tensor,
    aux_valid: torch.Tensor,
    *,
    aux_weight: float = 0.20,
    value_weight: float = 0.25,
    launch_boost: float = 1.25,
    midgame_weight: float = 1.0,
    comet_weight: float = 1.0,
    hold_boost: float = 1.0,
    launch_bce_weight: float = 0.0,
    return_tensor_metrics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute Soft-KL policy distillation + Auxiliary t+10 world-model + Value loss.

    Gen 9/10 extensions (default 1.0 / False for backward compatibility):
      - ``midgame_weight``: multiplies per-board KL weight during ``t in [25, 75]``.
      - ``comet_weight``: multiplies per-board KL weight whenever comets are active.
      - ``hold_boost``: boosts KL weight on midgame decisions where teacher prefers HOLD
        (``q_hold >= 0.65``), curbing impulsive all-in garrison shuffles.
      - ``launch_bce_weight``: explicit binary launch-vs-hold KL calibration weight.
      - ``return_tensor_metrics``: returns detached device scalar tensors without ``.item()``
        so GPU/DirectML training loops avoid mid-epoch CPU sync stalls.
    """
    out = model(pf, tf, ef, pm, return_aux=True, legal_edges=le)
    if len(out) == 4:
        logits, fracs, val, aux_pred = out
    else:
        logits, fracs, val = out
        aux_pred = None

    B, N, _ = logits.shape
    hold_ok = torch.ones((B, N, 1), dtype=torch.bool, device=logits.device)
    act_mask = torch.cat([hold_ok, le], dim=-1)
    logits = logits.masked_fill(~act_mask, -1e9)
    log_p = F.log_softmax(logits, dim=-1)

    valid_dec = om & le.any(dim=-1)
    valid_f = valid_dec.float()
    valid_tot_t = valid_f.sum()

    # Per-board phase & comet emphasis derived losslessly from planet_feats
    # col 16 = step / 500.0, col 6 = is_comet
    steps = pf[:, 0, 16] * 500.0  # (B,)
    is_mid = (steps >= 24.5) & (steps <= 75.5)
    has_comet = ((pf[:, :, 6] > 0.5) & pm).any(dim=-1)
    board_w = torch.ones((B,), dtype=torch.float32, device=logits.device)
    if midgame_weight != 1.0:
        board_w = torch.where(is_mid, board_w * float(midgame_weight), board_w)
    if comet_weight != 1.0:
        board_w = torch.where(has_comet, board_w * float(comet_weight), board_w)
    node_board_w = board_w.unsqueeze(1).expand(B, N)
    node_is_mid = is_mid.unsqueeze(1).expand(B, N)

    q_all = soft_probs.float()
    q_all = q_all / q_all.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    # Exact KL(q || p) = sum q * (log q - log p)
    kl_per = (q_all * (torch.log(q_all.clamp_min(1e-6)) - log_p)).sum(dim=-1)
    # Per-decision weighting: launch boost, midgame hold calibration, phase & comet weights
    q_hold = q_all[..., 0]
    q_launch = 1.0 - q_hold
    w = (1.0 + (launch_boost - 1.0) * q_launch) * node_board_w
    if hold_boost != 1.0:
        mid_hold = node_is_mid & (q_hold >= 0.65)
        w = torch.where(mid_hold, w * float(hold_boost), w)
    w_masked = w * valid_f
    kd_loss = (kl_per * w_masked).sum() / w_masked.sum().clamp_min(1.0)

    with torch.no_grad():
        pred_cls = log_p.argmax(dim=-1)
        teacher_cls = q_all.argmax(dim=-1)
        all_hits_t = ((pred_cls == teacher_cls).float() * valid_f).sum()
        lm_f = (valid_dec & (teacher_cls > 0)).float()
        launch_tot_t = lm_f.sum()
        launch_hits_t = ((pred_cls == teacher_cls).float() * lm_f).sum()
        tgt_pred = 1 + log_p[..., 1:].argmax(dim=-1)
        tgt_hits_t = ((tgt_pred == teacher_cls).float() * lm_f).sum()
        q_l_bin = (1.0 - q_hold) >= 0.53
        p_l_bin = (1.0 - log_p[..., 0].exp()) >= 0.53
        l_tp_t = ((q_l_bin & p_l_bin).float() * valid_f).sum()
        l_fp_t = (((~q_l_bin) & p_l_bin).float() * valid_f).sum()
        l_fn_t = ((q_l_bin & (~p_l_bin)).float() * valid_f).sum()

    # All-in calibration for fraction head on teacher launches (Hober Malloc always launches 100%)
    q_top_cls = q_all.argmax(dim=-1)
    lmask_f = (valid_dec & (q_top_cls > 0)).float()
    dst_idx = (q_top_cls - 1).clamp(0, N - 1)
    dst_oh = one_hot_no_scatter(dst_idx, N, dtype=fracs.dtype)
    f_pred = (fracs * dst_oh).sum(dim=-1)
    frac_loss = (F.smooth_l1_loss(f_pred, torch.ones_like(f_pred), reduction="none") * lmask_f).sum() / lmask_f.sum().clamp_min(1.0)

    # Auxiliary t+10 world-model loss (#7 Audun trick)
    av_f = aux_valid.float()
    av_tot_t = av_f.sum()
    if aux_pred is not None:
        av_denom = av_tot_t.clamp_min(1.0)
        ao_logp = F.log_softmax(aux_pred[..., :3], dim=-1)
        ao_oh = one_hot_no_scatter(aux_owner.clamp(0, 2), 3, dtype=ao_logp.dtype)
        ao_ce = -(ao_logp * ao_oh).sum(dim=-1)
        ao_loss = (ao_ce * av_f).sum() / av_denom
        as_loss = (F.smooth_l1_loss(aux_pred[..., 3], aux_ships, reduction="none") * av_f).sum() / av_denom
        aux_loss = ao_loss + as_loss
        with torch.no_grad():
            aux_hits_t = ((aux_pred[..., :3].argmax(dim=-1) == aux_owner).float() * av_f).sum()
    else:
        aux_loss = logits.sum() * 0.0
        aux_hits_t = logits.new_zeros(())
        av_tot_t = logits.new_zeros(())

    val_loss = F.mse_loss(val, vtar)
    with torch.no_grad():
        vsign_hits_t = ((val >= 0) == (vtar >= 0)).float().sum()
        vsign_tot_t = torch.tensor(float(B), dtype=torch.float32, device=logits.device)

    total = kd_loss + 0.15 * frac_loss + aux_weight * aux_loss + value_weight * val_loss
    if launch_bce_weight > 0.0:
        q_l = (1.0 - q_hold).clamp(1e-6, 1.0 - 1e-6)
        p_h = log_p[..., 0].exp().clamp(1e-6, 1.0 - 1e-6)
        p_l = (1.0 - p_h).clamp(1e-6, 1.0 - 1e-6)
        launch_kl_per = q_l * (torch.log(q_l) - torch.log(p_l)) + (1.0 - q_l) * (torch.log(1.0 - q_l) - torch.log(p_h))
        lw_masked = valid_f * node_board_w
        launch_bce_loss = (launch_kl_per * lw_masked).sum() / lw_masked.sum().clamp_min(1.0)
        total = total + float(launch_bce_weight) * launch_bce_loss

    if return_tensor_metrics:
        return total, {
            "loss": total.detach(),
            "kl": kd_loss.detach(),
            "aux": aux_loss.detach(),
            "aux_hits": aux_hits_t,
            "aux_tot": av_tot_t,
            "all_hits": all_hits_t,
            "all_tot": valid_tot_t,
            "launch_hits": launch_hits_t,
            "tgt_hits": tgt_hits_t,
            "launch_tot": launch_tot_t,
            "vsign_hits": vsign_hits_t,
            "vsign_tot": vsign_tot_t,
            "l_tp": l_tp_t,
            "l_fp": l_fp_t,
            "l_fn": l_fn_t,
        }

    valid_cnt = float(valid_tot_t.item())
    lm_cnt = float(launch_tot_t.item())
    av_cnt = float(av_tot_t.item())
    return total, {
        "loss": float(total.item()),
        "kl": float(kd_loss.item()),
        "aux": float(aux_loss.item()),
        "aux_acc": (float(aux_hits_t.item()) / av_cnt) if av_cnt > 0.0 else 0.0,
        "acc_all": (float(all_hits_t.item()) / valid_cnt) if valid_cnt > 0.0 else 0.0,
        "acc_launch": (float(launch_hits_t.item()) / lm_cnt) if lm_cnt > 0.0 else 0.0,
        "acc_tgt": (float(tgt_hits_t.item()) / lm_cnt) if lm_cnt > 0.0 else 0.0,
        "vsign": float(vsign_hits_t.item()) / max(1.0, float(B)),
        "l_tp": float(l_tp_t.item()),
        "l_fp": float(l_fp_t.item()),
        "l_fn": float(l_fn_t.item()),
    }


def train_soft_kd(
    data: Dict[str, np.ndarray],
    *,
    resume_from: Optional[str] = None,
    out_path: Path,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 3.5e-4,
    aux_weight: float = 0.20,
    value_weight: float = 0.25,
    weight_decay: float = 1e-4,
    launch_boost: float = 1.25,
    midgame_weight: float = 1.0,
    comet_weight: float = 1.0,
    hold_boost: float = 1.0,
    launch_bce_weight: float = 0.0,
    feature_version: int = 1,
    arch_version: int = 1,
    contiguous_val: bool = False,
    seed: int = 42,
    device: str = "cpu",
    train_threads: int = 4,
    d_model: int = 192,
    n_heads: int = 6,
    n_layers: int = 6,
    dim_ff: int = 384,
) -> Path:
    torch.set_num_threads(max(1, int(train_threads)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_dev, use_dml = resolve_train_device(device)

    n = len(data["val_targets"])
    if contiguous_val and n >= 10:
        n_val = max(1, int(0.10 * n))
        tr_idx = np.arange(0, n - n_val)
        val_idx = np.arange(n - n_val, n)
    else:
        perm = np.random.permutation(n)
        n_val = max(1, int(0.12 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

    full_ds = TensorDataset(
        torch.from_numpy(data["planet_feats"]),
        torch.from_numpy(data["timeline_feats"]),
        torch.from_numpy(data["edge_feats"]),
        torch.from_numpy(data["legal_edges"]),
        torch.from_numpy(data["owned_mask"]),
        torch.from_numpy(data["pad_mask"]),
        torch.from_numpy(data["soft_probs"]),
        torch.from_numpy(data["val_targets"]),
        torch.from_numpy(data["aux_owner"]),
        torch.from_numpy(data["aux_ships"]),
        torch.from_numpy(data["aux_valid"]),
    )

    tr_loader = DataLoader(Subset(full_ds, tr_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx.tolist()), batch_size=batch_size, shuffle=False)

    dev_tag = "dml" if use_dml else str(train_dev)
    if resume_from and Path(resume_from).is_file():
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        cfg = dict(ckpt.get("model_config", {}))
        cfg["use_aux_future"] = True
        model = PlanetTimelineTransformer(**cfg)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        init_desc = f"Resumed {resume_from} (new aux keys={len(missing)}, unexpected={len(unexpected)})"
    else:
        tl_dim = int(data["timeline_feats"].shape[-1])
        model = PlanetTimelineTransformer(
            timeline_dim=tl_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            use_aux_future=True,
            feature_version=feature_version,
            arch_version=arch_version,
        )
        init_desc = f"Fresh init (d={d_model}, h={n_heads}, L={n_layers}, ff={dim_ff}, fv={feature_version}, av={arch_version})"

    model = model.to(train_dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[SoftKD Train] {init_desc} -> {n_params/1e6:.2f}M params (device={dev_tag}) | "
        f"train={len(tr_idx)} val={len(val_idx)} | epochs={epochs} batch={batch_size} lr={lr} | "
        f"mid_w={midgame_weight} comet_w={comet_weight} hold_b={hold_boost} launch_b={launch_boost} lbce={launch_bce_weight}",
        flush=True,
    )

    opt = build_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, use_dml=use_dml)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    best_kl = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    acc_keys = (
        "kl",
        "aux_hits",
        "aux_tot",
        "all_hits",
        "all_tot",
        "launch_hits",
        "tgt_hits",
        "launch_tot",
        "vsign_hits",
        "vsign_tot",
        "l_tp",
        "l_fp",
        "l_fn",
    )

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_kl_acc = torch.zeros((), device=train_dev)
        tr_steps = 0
        for pf, tf, ef, le, om, pm, sp, vt, ao, ash, av in tr_loader:
            pf, tf, ef, le, om, pm, sp, ao, ash, av = _trim_soft_kd_batch_n(
                pf, tf, ef, le, om, pm, sp, ao, ash, av
            )
            k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
            pf_a = torch.from_numpy(apply_d4_planet_feats(pf.float().numpy(), k_rot=k_rot, flip_x=flip_x)).to(train_dev)
            tf_a = torch.from_numpy(apply_d4_timeline_feats(tf.float().numpy(), k_rot=k_rot, flip_x=flip_x)).to(train_dev)
            ef, sp, vt, ash = (t.float().to(train_dev) for t in (ef, sp, vt, ash))
            le, om, pm, ao, av = (t.to(train_dev) for t in (le, om, pm, ao, av))
            opt.zero_grad(set_to_none=True)
            loss, m = compute_soft_kd_loss(
                model, pf_a, tf_a, ef, le, om, pm, sp, vt, ao, ash, av,
                aux_weight=aux_weight,
                value_weight=value_weight,
                launch_boost=launch_boost,
                midgame_weight=midgame_weight,
                comet_weight=comet_weight,
                hold_boost=hold_boost,
                launch_bce_weight=launch_bce_weight,
                return_tensor_metrics=True,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=not use_dml)
            opt.step()
            tr_kl_acc = tr_kl_acc + m["kl"]
            tr_steps += 1
        sched.step()

        model.eval()
        val_acc = {k: torch.zeros((), device=train_dev) for k in acc_keys}
        val_steps = 0
        with torch.no_grad():
            for pf, tf, ef, le, om, pm, sp, vt, ao, ash, av in val_loader:
                pf, tf, ef, le, om, pm, sp, ao, ash, av = _trim_soft_kd_batch_n(
                    pf, tf, ef, le, om, pm, sp, ao, ash, av
                )
                pf, tf, ef, sp, vt, ash = (t.float().to(train_dev) for t in (pf, tf, ef, sp, vt, ash))
                le, om, pm, ao, av = (t.to(train_dev) for t in (le, om, pm, ao, av))
                _, m = compute_soft_kd_loss(
                    model, pf, tf, ef, le, om, pm, sp, vt, ao, ash, av,
                    aux_weight=aux_weight,
                    value_weight=value_weight,
                    launch_boost=launch_boost,
                    midgame_weight=midgame_weight,
                    comet_weight=comet_weight,
                    hold_boost=hold_boost,
                    launch_bce_weight=launch_bce_weight,
                    return_tensor_metrics=True,
                )
                for k in acc_keys:
                    val_acc[k] = val_acc[k] + m[k]
                val_steps += 1

        tr_kl = float(tr_kl_acc.item()) / max(1, tr_steps)
        v_kl = float(val_acc["kl"].item()) / max(1, val_steps)
        all_tot = max(1.0, float(val_acc["all_tot"].item()))
        launch_tot = max(1.0, float(val_acc["launch_tot"].item()))
        aux_tot = max(1.0, float(val_acc["aux_tot"].item()))
        vsign_tot = max(1.0, float(val_acc["vsign_tot"].item()))

        v_all = float(val_acc["all_hits"].item()) / all_tot
        v_launch = float(val_acc["launch_hits"].item()) / launch_tot
        v_tgt = float(val_acc["tgt_hits"].item()) / launch_tot
        v_aux = float(val_acc["aux_hits"].item()) / aux_tot
        v_sign = float(val_acc["vsign_hits"].item()) / vsign_tot
        l_tp = float(val_acc["l_tp"].item())
        l_fp = float(val_acc["l_fp"].item())
        l_fn = float(val_acc["l_fn"].item())
        l_prec = l_tp / max(1.0, l_tp + l_fp)
        l_rec = l_tp / max(1.0, l_tp + l_fn)
        l_f1 = (2.0 * l_prec * l_rec / max(1e-9, l_prec + l_rec)) if (l_prec + l_rec) > 0.0 else 0.0

        if v_kl < best_kl:
            best_kl = v_kl
            torch.save(
                {
                    "model_config": model.config,
                    "model_state_dict": cpu_state_dict(model),
                    "epoch": ep,
                    "val_kl": v_kl,
                    "val_acc_all": v_all,
                    "val_acc_launch": v_launch,
                    "val_acc_target": v_tgt,
                    "val_l_f1": l_f1,
                    "val_aux_owner_acc": v_aux,
                    "val_sign": v_sign,
                },
                out_path,
            )
        print(
            f"  ep {ep:2d}/{epochs} ({time.time()-t0:5.1f}s, {time.time()-ep_t0:4.1f}s/ep) | "
            f"tr_kl={tr_kl:.3f} val_kl={v_kl:.3f} | "
            f"val_all={100*v_all:5.1f}% l_f1={100*l_f1:5.1f}% (p={100*l_prec:4.1f}% r={100*l_rec:4.1f}%) "
            f"val_launch={100*v_launch:5.1f}% val_tgt={100*v_tgt:5.1f}% aux_own={100*v_aux:5.1f}% "
            f"vsign={100*v_sign:5.1f}%",
            flush=True,
        )
    print(f"[SoftKD Train] Saved best 2P Specialist -> {out_path} ({out_path.stat().st_size/1024:.1f} KB)", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gen 8/9 Soft-Logit KD + Aux Future World Model + Midgame Calibration.")
    parser.add_argument("--dataset-cache", required=True, help="Path to .npz soft-KD dataset cache.")
    parser.add_argument("--hober-games", type=int, default=16)
    parser.add_argument("--dagger-games", type=int, default=20)
    parser.add_argument("--student-checkpoint", default="checkpoints/planet_transformer_gen8_2p_softkd.pt")
    parser.add_argument("--student-spec-prefix", default="pt_d4_allin:")
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--dense-start-tick", type=int, default=22)
    parser.add_argument("--dense-end-tick", type=int, default=78)
    parser.add_argument("--max-collect-tick", type=int, default=0)
    parser.add_argument("--full-boss-rollout", action="store_true")
    parser.add_argument("--record-opp-slot", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--append", action="store_true", help="Collect new games and append onto existing dataset-cache.")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume", default="checkpoints/planet_transformer_gen8_2p_softkd.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--launch-boost", type=float, default=1.0)
    parser.add_argument("--midgame-weight", type=float, default=2.5)
    parser.add_argument("--comet-weight", type=float, default=2.0)
    parser.add_argument("--hold-boost", type=float, default=1.25)
    parser.add_argument("--launch-bce-weight", type=float, default=0.0)
    parser.add_argument("--feature-version", type=int, default=1)
    parser.add_argument("--arch-version", type=int, default=1)
    parser.add_argument("--contiguous-val", action="store_true")
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--device", choices=["cpu", "dml", "cuda", "auto"], default="cpu")
    parser.add_argument("--train-threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dim-ff", type=int, default=384)
    parser.add_argument("--out", default="checkpoints/planet_transformer_gen9_softkd.pt")
    args = parser.parse_args(argv)

    cache = Path(args.dataset_cache)
    existing: Optional[Dict[str, np.ndarray]] = None
    if cache.is_file():
        print(f"[SoftKD] Loading cached dataset from {cache}...", flush=True)
        with np.load(cache) as npz:
            existing = {k: npz[k] for k in npz.files}
        print(f"[SoftKD] Loaded {len(existing['val_targets'])} soft-KD 2P boards.", flush=True)

    if existing is None or args.append:
        fresh = collect_soft_kd_dataset(
            hober_games=args.hober_games,
            dagger_games=args.dagger_games,
            student_checkpoint=args.student_checkpoint,
            student_spec_prefix=args.student_spec_prefix,
            seed=args.seed,
            sample_stride=args.sample_stride,
            dense_start_tick=args.dense_start_tick,
            dense_end_tick=args.dense_end_tick,
            workers=args.workers,
            full_boss_rollout=args.full_boss_rollout,
            record_opp_slot=args.record_opp_slot,
            max_collect_tick=args.max_collect_tick,
            feature_version=args.feature_version,
        )
        if existing is not None:
            data = {k: np.concatenate([existing[k], fresh[k]], axis=0) for k in existing}
        else:
            data = fresh
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
        print(f"[SoftKD] Cached {len(data['val_targets'])} total boards -> {cache} ({cache.stat().st_size/1e6:.2f} MB)", flush=True)
    else:
        data = existing

    if args.collect_only:
        return 0

    train_soft_kd(
        data,
        resume_from=(args.resume or None),
        out_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aux_weight=args.aux_weight,
        value_weight=args.value_weight,
        weight_decay=args.weight_decay,
        launch_boost=args.launch_boost,
        midgame_weight=args.midgame_weight,
        comet_weight=args.comet_weight,
        hold_boost=args.hold_boost,
        launch_bce_weight=args.launch_bce_weight,
        feature_version=args.feature_version,
        arch_version=args.arch_version,
        contiguous_val=args.contiguous_val,
        seed=args.seed,
        device=args.device,
        train_threads=args.train_threads,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
