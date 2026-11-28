#!/usr/bin/env python3
"""Self-Play League PPO + Dual-KL + Aux World-Model trainer for ``PlanetTimelineTransformer``.

Upgraded Gen 5 -> Gen 10 GPU-scale DirectML trainer incorporating:
- **Zero-Fallback DirectML Execution**: ``DMLAdamW`` (``--device cpu|dml|cuda|auto``),
  broadcast one-hot (``one_hot_no_scatter``), dense masked PPO/KL/Aux losses, and
  ``foreach=False`` gradient clipping on DirectML (eliminating CPU fallbacks and
  DirectML scatter errors).
- **Paired-Seed Mirror Rollouts + On-Policy D4 Symmetry**: Plays both player seats
  on each map seed with rollout-time D4 symmetries so ``ppo_update`` maintains exact
  importance ratios (``ratio == 1.0`` on iteration start).
- **Persistent Worker Pool & File-Cached Checkpoints**: Eliminates per-iteration
  state-dict IPC overhead across spawn workers and parallelizes validation games.
- **Dual-KL Anchoring + Ground-Truth ``t+10`` Aux World Model**: Combines a slow EMA
  teacher KL with a light frozen-init anchor KL and live ``t+10`` planet ownership /
  ship-count world-model supervision during RL.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orbit_wars_bot.dml_utils import (  # noqa: E402
    apply_d4_batch_gpu,
    build_optimizer,
    cpu_state_dict,
    one_hot_no_scatter,
    resolve_train_device,
    snap_bucket_batches,
)
from orbit_wars_bot.fast_sim import FastOrbitSim  # noqa: E402
from orbit_wars_bot.fast_sim_cpp import HAS_ORBIT_SIM_CPP, solve_intercept_cpp  # noqa: E402
from orbit_wars_bot.features import evaluate_state  # noqa: E402
from orbit_wars_bot.heuristic_bot import heuristic_moves  # noqa: E402
from orbit_wars_bot.physics import DEFAULT_SHIP_SPEED, solve_intercept  # noqa: E402
from orbit_wars_bot.planet_transformer import (  # noqa: E402
    MAX_PLANETS,
    MAX_ROUTE_ETA,
    EncodedPlanetState,
    PlanetTimelineTransformer,
    apply_d4_planet_feats,
    apply_d4_timeline_feats,
    build_shared_planet_physics,
    extract_planet_features,
    pad_encoded_state,
)

# Cached external opponent callable and model caches per worker process
_SUBHAM_AGENT: Any = None
_WORKER_HOBER: Any = None
_WORKER_POLICY_CACHE: Dict[str, Any] = {}
_WORKER_OPP_CACHE: Dict[str, PlanetTimelineTransformer] = {}
_HOBER_SENTINEL = "__HOBER__"


def _get_worker_hober():
    global _WORKER_HOBER
    if _WORKER_HOBER is None:
        from orbit_wars_bot.hober_sparring import FastHoberSparringModel

        _WORKER_HOBER = FastHoberSparringModel()
    return _WORKER_HOBER


def _get_subham_agent():
    global _SUBHAM_AGENT
    if _SUBHAM_AGENT is None:
        from evaluate import resolve_agent

        _SUBHAM_AGENT = resolve_agent("subham_v14")
    return _SUBHAM_AGENT


def _load_worker_model(ckpt_path: str) -> PlanetTimelineTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = PlanetTimelineTransformer(**cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def _get_worker_policy(ckpt_path: str, version: int) -> PlanetTimelineTransformer:
    cached = _WORKER_POLICY_CACHE.get("model")
    if (
        cached is not None
        and _WORKER_POLICY_CACHE.get("path") == ckpt_path
        and _WORKER_POLICY_CACHE.get("version") == version
    ):
        return cached
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    if cached is None or _WORKER_POLICY_CACHE.get("cfg") != cfg:
        cached = PlanetTimelineTransformer(**cfg)
    cached.load_state_dict(ckpt["model_state_dict"], strict=True)
    cached.eval()
    _WORKER_POLICY_CACHE["path"] = ckpt_path
    _WORKER_POLICY_CACHE["version"] = version
    _WORKER_POLICY_CACHE["cfg"] = cfg
    _WORKER_POLICY_CACHE["model"] = cached
    return cached


def _get_worker_opp(opp_path: Optional[str]) -> Optional[PlanetTimelineTransformer]:
    if opp_path is None:
        return None
    model = _WORKER_OPP_CACHE.get(opp_path)
    if model is None:
        model = _load_worker_model(opp_path)
        _WORKER_OPP_CACHE[opp_path] = model
    return model


def _decode_sampled_actions(
    enc: EncodedPlanetState,
    action_cls: np.ndarray,
    fracs: np.ndarray,
    *,
    max_speed: float = DEFAULT_SHIP_SPEED,
    force_all_in: bool = False,
) -> List[List[float]]:
    """Convert sampled per-planet action indices (0=HOLD, 1+j=target j) into Kaggle moves."""
    n = len(enc.planet_ids)
    expiring_set = set(enc.expiring_comet_indices)
    moves: List[List[float]] = []

    for i in range(n):
        if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
            continue
        cls = int(action_cls[i])
        if cls <= 0:
            continue
        dst_j = cls - 1
        if dst_j < 0 or dst_j >= n or not enc.legal_edges[i, dst_j]:
            continue

        src_row = enc.planet_rows[i]
        dst_row = enc.planet_rows[dst_j]
        avail = int(src_row[5])
        if avail <= 0:
            continue

        forced = i in expiring_set
        f = float(fracs[i, dst_j])
        send_ships = avail if (force_all_in or forced or f >= 0.88) else max(1, min(avail, int(round(avail * f))))

        if (
            HAS_ORBIT_SIM_CPP
            and enc.tr_x is not None
            and enc.tr_y is not None
            and enc.planet_radii is not None
        ):
            sol = solve_intercept_cpp(
                i,
                dst_j,
                send_ships,
                enc.tr_x,
                enc.tr_y,
                enc.planet_radii,
                max_eta=MAX_ROUTE_ETA + 4,
                max_speed=max_speed,
            )
        else:
            sol = solve_intercept(
                src_row,
                dst_row,
                send_ships,
                enc.planet_rows,
                enc.current_step,
                enc.angular_velocity,
                enc.initial_by_id,
                comet_lookup=enc.comet_lookup,
                tracks=enc.tracks,
                max_eta=MAX_ROUTE_ETA + 4,
                max_speed=max_speed,
            )
        if sol is not None:
            moves.append([int(src_row[0]), float(sol.angle), int(send_ships)])
    return moves


@torch.no_grad()
def _policy_step(
    model: PlanetTimelineTransformer,
    enc: EncodedPlanetState,
    *,
    sample: bool = True,
    temperature: float = 0.85,
    launch_threshold: float = 0.50,
    force_all_in: bool = False,
    rng: np.random.Generator | None = None,
    k_rot: int = 0,
    flip_x: bool = False,
) -> Tuple[List[List[float]], np.ndarray, np.ndarray, float]:
    """Run one forward pass and return ``(moves, action_cls[MAX_PLANETS], old_logp[MAX_PLANETS], value)``."""
    n = len(enc.planet_ids)
    action_cls = np.zeros((MAX_PLANETS,), dtype=np.int64)
    old_logp = np.zeros((MAX_PLANETS,), dtype=np.float32)
    if n == 0 or not np.any(enc.owned_mask):
        return [], action_cls, old_logp, 0.0

    pf_np = (
        apply_d4_planet_feats(enc.planet_feats, k_rot=k_rot, flip_x=flip_x)
        if (k_rot != 0 or flip_x)
        else enc.planet_feats
    )
    tf_raw = enc.timeline_feats[..., : getattr(model, "timeline_dim", 8)]
    tf_np = (
        apply_d4_timeline_feats(tf_raw, k_rot=k_rot, flip_x=flip_x)
        if (k_rot != 0 or flip_x)
        else tf_raw
    )
    pf = torch.from_numpy(pf_np[None])
    tf = torch.from_numpy(tf_np[None])
    ef = torch.from_numpy(enc.edge_feats[None])
    pm = torch.ones((1, n), dtype=torch.bool)
    legal = torch.from_numpy(enc.legal_edges)

    logits, fracs, val = model(pf, tf, ef, pm, legal_edges=legal.unsqueeze(0), compute_frac=(not force_all_in))
    logits = logits[0]  # (n, n+1)
    fracs_np = fracs[0].cpu().numpy() if fracs is not None else np.ones((n, n), dtype=np.float32)  # (n, n)
    v_scalar = float(val[0].item())

    mask = torch.cat([torch.ones((n, 1), dtype=torch.bool), legal], dim=-1)
    logits = logits.masked_fill(~mask, -1e9)

    log_probs = F.log_softmax(logits, dim=-1)
    if sample and temperature > 1e-4:
        sample_probs = F.softmax(logits / temperature, dim=-1).cpu().numpy()
    else:
        sample_probs = F.softmax(logits, dim=-1).cpu().numpy()
    log_probs_np = log_probs.cpu().numpy()

    expiring_set = set(enc.expiring_comet_indices)
    for i in range(n):
        if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
            continue
        if sample:
            p_row = sample_probs[i].astype(np.float64)
            if i in expiring_set and p_row[1:].sum() > 1e-9:
                p_row[0] = 0.0
            p_sum = p_row.sum()
            if p_sum <= 0.0:
                cls = 0
            else:
                p_row /= p_sum
                cls = int(rng.choice(len(p_row), p=p_row)) if rng is not None else int(np.random.choice(len(p_row), p=p_row))
        else:
            p_launch = 1.0 - float(sample_probs[i, 0])
            if p_launch < launch_threshold and i not in expiring_set:
                cls = 0
            else:
                cls = 1 + int(np.argmax(sample_probs[i, 1:]))

        action_cls[i] = cls
        old_logp[i] = float(log_probs_np[i, cls])

    moves = _decode_sampled_actions(enc, action_cls, fracs_np, force_all_in=force_all_in)
    return moves, action_cls, old_logp, v_scalar


def collect_episode(
    policy: PlanetTimelineTransformer,
    anchor: PlanetTimelineTransformer,
    *,
    num_players: int,
    opponent_kind: str,
    seed: int,
    temperature: float = 0.80,
    max_steps: int = 220,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    shaping_coef: float = 0.02,
    early_reset: bool = True,
    force_all_in: bool = False,
    opp_model: Optional[PlanetTimelineTransformer] = None,
    hero_slot: Optional[int] = None,
    rollout_d4: bool = False,
    future_horizon: int = 10,
    opp_temperature: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run one on-policy 2P or 4P episode in ``FastOrbitSim`` and compute GAE returns."""
    rng = np.random.default_rng(seed)
    sim = FastOrbitSim.from_seed(seed=seed, num_agents=num_players, max_steps=max_steps)
    hero_slot_val = int(rng.integers(0, num_players)) if hero_slot is None else int(hero_slot)
    active_opp_model = opp_model if opp_model is not None else anchor
    hero_fv = int(getattr(policy, "feature_version", 1))
    opp_fv = int(getattr(active_opp_model, "feature_version", 1)) if active_opp_model is not None else 2
    tl_dim_pol = int(getattr(policy, "timeline_dim", 8))
    k_rot, flip_x = (int(rng.integers(0, 4)), bool(rng.integers(0, 2))) if rollout_d4 else (0, False)

    traj_pf: List[np.ndarray] = []
    traj_tf: List[np.ndarray] = []
    traj_ef: List[np.ndarray] = []
    traj_le: List[np.ndarray] = []
    traj_om: List[np.ndarray] = []
    traj_pm: List[np.ndarray] = []
    traj_act: List[np.ndarray] = []
    traj_logp: List[np.ndarray] = []
    traj_vals: List[float] = []
    traj_rews: List[float] = []
    traj_steps: List[int] = []
    traj_pids: List[List[int]] = []
    planet_snaps: Dict[int, Dict[int, Tuple[int, int]]] = {}

    last_alive = [0] * num_players
    prev_phi = float(evaluate_state(sim, hero_slot_val))
    early_winner: Optional[int] = None

    while not sim.done:
        planet_snaps[int(sim.step_count)] = {int(p[0]): (int(p[1]), int(p[5])) for p in sim.planets}
        actions: List[List[List[float]]] = [[] for _ in range(num_players)]
        recorded_hero = False
        obs_shared = sim.to_observation(player=0)
        shared_phys = build_shared_planet_physics(obs_shared, max_speed=sim.ship_speed)

        for pid in range(num_players):
            if pid == hero_slot_val:
                enc_h = extract_planet_features(
                    obs_shared,
                    player=hero_slot_val,
                    max_speed=sim.ship_speed,
                    feature_version=hero_fv,
                    shared_physics=shared_phys,
                )
                if np.any(enc_h.owned_mask):
                    moves_h, act_cls, old_lp, val_h = _policy_step(
                        policy,
                        enc_h,
                        sample=True,
                        temperature=temperature,
                        force_all_in=force_all_in,
                        rng=rng,
                        k_rot=k_rot,
                        flip_x=flip_x,
                    )
                    actions[pid] = moves_h
                    padded = pad_encoded_state(enc_h, MAX_PLANETS)
                    pf_store = padded["planet_feats"]
                    tf_store = padded["timeline_feats"][..., :tl_dim_pol]
                    if k_rot != 0 or flip_x:
                        pf_store = apply_d4_planet_feats(pf_store, k_rot=k_rot, flip_x=flip_x)
                        tf_store = apply_d4_timeline_feats(tf_store, k_rot=k_rot, flip_x=flip_x)
                    traj_pf.append(pf_store)
                    traj_tf.append(tf_store)
                    traj_ef.append(padded["edge_feats"])
                    traj_le.append(padded["legal_edges"])
                    traj_om.append(padded["owned_mask"])
                    traj_pm.append(padded["pad_mask"])
                    traj_act.append(act_cls)
                    traj_logp.append(old_lp)
                    traj_vals.append(val_h)
                    traj_steps.append(int(sim.step_count))
                    traj_pids.append(list(enc_h.planet_ids))
                    recorded_hero = True
            else:
                if opponent_kind == "heuristic":
                    actions[pid] = heuristic_moves(sim, pid)
                elif opponent_kind == "hober":
                    enc_o = extract_planet_features(
                        obs_shared,
                        player=pid,
                        max_speed=sim.ship_speed,
                        feature_version=2,
                        shared_physics=shared_phys,
                    )
                    actions[pid] = _get_worker_hober().act_batch(
                        [enc_o],
                        [shared_phys],
                        rotation_tta=False,
                        launch_threshold=0.53,
                        temperature=0.0,
                    )[0]
                elif opponent_kind == "subham_v14":
                    try:
                        sub_fn = _get_subham_agent()
                        obs_o = sim.to_observation(player=pid)
                        actions[pid] = sub_fn(obs_o, sim.to_config()) or []
                    except Exception:
                        actions[pid] = heuristic_moves(sim, pid)
                elif opponent_kind == "self":
                    enc_o = extract_planet_features(
                        obs_shared,
                        player=pid,
                        max_speed=sim.ship_speed,
                        feature_version=hero_fv,
                        shared_physics=shared_phys,
                    )
                    mv_o, _, _, _ = _policy_step(
                        policy, enc_o, sample=True, temperature=0.55, force_all_in=force_all_in, rng=rng
                    )
                    actions[pid] = mv_o
                else:  # frozen anchor or historical pool snapshot
                    enc_o = extract_planet_features(
                        obs_shared,
                        player=pid,
                        max_speed=sim.ship_speed,
                        feature_version=opp_fv,
                        shared_physics=shared_phys,
                    )
                    opp_sample = opp_temperature > 1e-4
                    opp_temp = opp_temperature if opp_sample else 1.0
                    mv_o, _, _, _ = _policy_step(
                        active_opp_model,
                        enc_o,
                        sample=opp_sample,
                        temperature=opp_temp,
                        force_all_in=force_all_in,
                        rng=rng,
                    )
                    actions[pid] = mv_o

        sim.step(actions)
        totals = sim.scores()
        for pid in range(num_players):
            if totals[pid] > 0:
                last_alive[pid] = sim.step_count

        if recorded_hero:
            cur_phi = float(evaluate_state(sim, hero_slot_val))
            traj_rews.append(shaping_coef * (gamma * cur_phi - prev_phi))
            prev_phi = cur_phi

        # Zero-planet early reset (#7 Kaggle lesson): cut off dead-fleet garbage frames
        if early_reset and not sim.done and sim.step_count >= 12:
            p_counts = [sim.planet_count(p) for p in range(num_players)]
            alive_planet_players = [p for p, c in enumerate(p_counts) if c > 0]
            if len(alive_planet_players) == 1:
                sole = alive_planet_players[0]
                if totals[sole] >= max(totals) and sum(totals) > 0:
                    early_winner = sole
                    break
            elif p_counts[hero_slot_val] == 0 and totals[hero_slot_val] == 0:
                # Hero completely eliminated in 4P FFA — no further decisions possible
                break

    planet_snaps[int(sim.step_count)] = {int(p[0]): (int(p[1]), int(p[5])) for p in sim.planets}

    # Terminal outcome reward in [-1, +1]
    totals = sim.scores()
    if early_winner is not None:
        win_pid = early_winner
    elif sim.done:
        win_pid = sim.winner()
    else:
        max_s = max(totals)
        winners = [i for i, s in enumerate(totals) if s == max_s]
        win_pid = winners[0] if len(winners) == 1 else -1

    if num_players == 2:
        term_rew = 1.0 if win_pid == hero_slot_val else (0.0 if win_pid < 0 else -1.0)
        hero_rank = 1.0 if win_pid == hero_slot_val else (1.5 if win_pid < 0 else 2.0)
    else:
        order = sorted(range(num_players), key=lambda i: (totals[i], last_alive[i]), reverse=True)
        grad = [1.0, 0.25, -0.35, -1.0]
        rank_idx = order.index(hero_slot_val)
        term_rew = grad[rank_idx]
        hero_rank = float(rank_idx + 1)

    if traj_rews:
        traj_rews[-1] += term_rew

    T = len(traj_rews)
    advs = np.zeros((T,), dtype=np.float32)
    rets = np.zeros((T,), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = 0.0 if (t == T - 1) else traj_vals[t + 1]
        delta = traj_rews[t] + gamma * next_v - traj_vals[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advs[t] = last_gae
        rets[t] = np.clip(last_gae + traj_vals[t], -1.2, 1.2)

    if T == 0:
        empty = {
            "planet_feats": np.zeros((0, MAX_PLANETS, 22), np.float16),
            "timeline_feats": np.zeros((0, MAX_PLANETS, 20, tl_dim_pol), np.float16),
            "edge_feats": np.zeros((0, MAX_PLANETS, MAX_PLANETS, 12), np.float16),
            "legal_edges": np.zeros((0, MAX_PLANETS, MAX_PLANETS), bool),
            "owned_mask": np.zeros((0, MAX_PLANETS), bool),
            "pad_mask": np.zeros((0, MAX_PLANETS), bool),
            "action_cls": np.zeros((0, MAX_PLANETS), np.int64),
            "old_logp": np.zeros((0, MAX_PLANETS), np.float32),
            "advantages": np.zeros((0,), np.float32),
            "returns": np.zeros((0,), np.float32),
            "step_ticks": np.zeros((0,), np.int32),
            "is_hober": np.zeros((0,), bool),
            "aux_owner": np.zeros((0, MAX_PLANETS), np.int64),
            "aux_ships": np.zeros((0, MAX_PLANETS), np.float32),
            "aux_valid": np.zeros((0, MAX_PLANETS), bool),
        }
        return empty, {
            "win": float(win_pid == hero_slot_val),
            "rank": hero_rank,
            "steps": float(sim.step_count),
            "term_rew": term_rew,
            "num_players": float(num_players),
            "hero_slot": float(hero_slot_val),
        }

    traj_ao: List[np.ndarray] = []
    traj_ash: List[np.ndarray] = []
    traj_av: List[np.ndarray] = []
    for idx_t in range(T):
        ao = np.full((MAX_PLANETS,), 2, dtype=np.int64)
        ash = np.zeros((MAX_PLANETS,), dtype=np.float32)
        av = np.zeros((MAX_PLANETS,), dtype=bool)
        fut_map = planet_snaps.get(traj_steps[idx_t] + future_horizon)
        if fut_map is not None:
            for i_p, pid_val in enumerate(traj_pids[idx_t]):
                rec = fut_map.get(int(pid_val))
                if rec is None:
                    continue
                ow, sh = rec
                ao[i_p] = 0 if ow == hero_slot_val else (2 if ow < 0 else 1)
                ash[i_p] = float(np.log1p(max(0, sh)) / 6.0)
                av[i_p] = True
        traj_ao.append(ao)
        traj_ash.append(ash)
        traj_av.append(av)

    batch = {
        "planet_feats": np.stack(traj_pf, axis=0).astype(np.float16, copy=False),
        "timeline_feats": np.stack(traj_tf, axis=0).astype(np.float16, copy=False),
        "edge_feats": np.stack(traj_ef, axis=0).astype(np.float16, copy=False),
        "legal_edges": np.stack(traj_le, axis=0),
        "owned_mask": np.stack(traj_om, axis=0),
        "pad_mask": np.stack(traj_pm, axis=0),
        "action_cls": np.stack(traj_act, axis=0),
        "old_logp": np.stack(traj_logp, axis=0),
        "advantages": advs,
        "returns": rets,
        "step_ticks": np.asarray(traj_steps, dtype=np.int32),
        "is_hober": np.full((T,), opponent_kind == "hober", dtype=bool),
        "aux_owner": np.stack(traj_ao, axis=0),
        "aux_ships": np.stack(traj_ash, axis=0),
        "aux_valid": np.stack(traj_av, axis=0),
    }
    stats = {
        "win": 1.0 if win_pid == hero_slot_val else 0.0,
        "rank": hero_rank,
        "steps": float(sim.step_count),
        "term_rew": float(term_rew),
        "num_players": float(num_players),
        "hero_slot": float(hero_slot_val),
    }
    return batch, stats


def _configure_worker_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def factorized_log_probs(
    logits: torch.Tensor,
    act_mask: torch.Tensor,
    t_l: float,
    t_t: float,
    thr_logit: float,
) -> torch.Tensor:
    """Exact normalized log-probabilities for threshold-centered factorized sampling q_theta(a|s)."""
    neg_inf = -1e4 if logits.dtype == torch.float16 else -1e9
    masked = logits.masked_fill(~act_mask, neg_inf)
    logp_base = F.log_softmax(masked, dim=-1)
    log_p_h = logp_base[..., 0]
    tgt_mask = act_mask[..., 1:]
    has_t = tgt_mask.any(dim=-1, keepdim=True)
    safe_t = tgt_mask | (~has_t)
    log_p_l = torch.logsumexp(logp_base[..., 1:].masked_fill(~safe_t, neg_inf), dim=-1)
    l_z = (log_p_l - log_p_h).clamp(-15.0, 15.0)
    z_s = ((l_z - float(thr_logit)) / max(1e-3, float(t_l))).clamp(-15.0, 15.0)
    lp2 = F.log_softmax(torch.stack([torch.zeros_like(z_s), z_s], dim=-1), dim=-1)
    log_q_h = lp2[..., 0]
    log_q_l = lp2[..., 1]
    raw_t = logp_base[..., 1:].masked_fill(~safe_t, neg_inf) / max(1e-3, float(t_t))
    log_q_t = F.log_softmax(raw_t, dim=-1).masked_fill(~tgt_mask, neg_inf)
    has_t_1 = has_t.squeeze(-1)
    log_q_h = torch.where(has_t_1, log_q_h, torch.zeros_like(log_q_h))
    log_q_l = torch.where(has_t_1, log_q_l, torch.full_like(log_q_l, neg_inf))
    return torch.cat([log_q_h.unsqueeze(-1), log_q_l.unsqueeze(-1) + log_q_t], dim=-1)


@torch.no_grad()
def _batched_policy_steps(
    model: PlanetTimelineTransformer,
    encs: Sequence[EncodedPlanetState],
    *,
    k_rots: Sequence[int],
    flip_xs: Sequence[bool],
    sample: bool = True,
    temperature: float = 0.85,
    launch_threshold: float = 0.50,
    force_all_in: bool = False,
    rngs: Sequence[np.random.Generator],
    factorized_sample: bool = False,
    target_temperature: float = 0.30,
    exact_factorized_ppo: bool = False,
) -> List[Tuple[List[List[float]], np.ndarray, np.ndarray, float]]:
    """Run one batched CPU forward pass over ``len(encs)`` active games simultaneously."""
    b_cnt = len(encs)
    if b_cnt == 0:
        return []
    tl_dim = int(getattr(model, "timeline_dim", 8))
    max_n = max(1, max(len(e.planet_ids) for e in encs))
    pf = np.zeros((b_cnt, max_n, 22), dtype=np.float32)
    tf = np.zeros((b_cnt, max_n, 20, tl_dim), dtype=np.float32)
    ef = np.zeros((b_cnt, max_n, max_n, 12), dtype=np.float32)
    pm = np.zeros((b_cnt, max_n), dtype=bool)
    le = np.zeros((b_cnt, max_n, max_n), dtype=bool)

    for b, e in enumerate(encs):
        n = len(e.planet_ids)
        if n == 0:
            continue
        kr, fx = k_rots[b], flip_xs[b]
        pf_b = apply_d4_planet_feats(e.planet_feats, k_rot=kr, flip_x=fx) if (kr or fx) else e.planet_feats
        tf_raw = e.timeline_feats[..., :tl_dim]
        tf_b = apply_d4_timeline_feats(tf_raw, k_rot=kr, flip_x=fx) if (kr or fx) else tf_raw
        pf[b, :n] = pf_b
        tf[b, :n] = tf_b
        ef[b, :n, :n] = e.edge_feats
        pm[b, :n] = True
        le[b, :n, :n] = e.legal_edges

    pf_t = torch.from_numpy(pf)
    tf_t = torch.from_numpy(tf)
    ef_t = torch.from_numpy(ef)
    pm_t = torch.from_numpy(pm)
    le_t = torch.from_numpy(le)

    logits, fracs, vals = model(pf_t, tf_t, ef_t, pm_t, legal_edges=le_t, compute_frac=(not force_all_in))
    hold_ok = torch.ones((b_cnt, max_n, 1), dtype=torch.bool)
    mask = torch.cat([hold_ok, le_t], dim=-1)
    logits = logits.masked_fill(~mask, -1e9)
    thr_logit = float(np.log(max(1e-5, launch_threshold) / max(1e-5, 1.0 - launch_threshold)))
    t_l = max(1e-3, float(temperature))
    t_t = max(1e-3, float(target_temperature))
    if exact_factorized_ppo and sample and factorized_sample:
        logp = factorized_log_probs(logits, mask, t_l, t_t, thr_logit).numpy()
    else:
        logp = F.log_softmax(logits, dim=-1).numpy()
    if sample and (not factorized_sample) and temperature > 1e-4:
        sprobs = F.softmax(logits / temperature, dim=-1).numpy()
    else:
        sprobs = F.softmax(logits, dim=-1).numpy()
    fracs_np = fracs.numpy() if fracs is not None else None
    vals_np = vals.numpy()

    out: List[Tuple[List[List[float]], np.ndarray, np.ndarray, float]] = []
    for b, e in enumerate(encs):
        n = len(e.planet_ids)
        act_cls = np.zeros((MAX_PLANETS,), dtype=np.int64)
        old_lp = np.zeros((MAX_PLANETS,), dtype=np.float32)
        if n == 0 or not np.any(e.owned_mask):
            out.append(([], act_cls, old_lp, float(vals_np[b])))
            continue
        exp_set = set(e.expiring_comet_indices)
        rng_b = rngs[b]
        for i in range(n):
            if not e.owned_mask[i] or not np.any(e.legal_edges[i]):
                continue
            if sample and factorized_sample:
                # Decoupled Launch vs Target sampling centered on launch_threshold:
                # eliminates multi-target HOLD collapse (sum p_j^(1/T) << 1) in PT-v2!
                p_l_raw = float(np.clip(1.0 - sprobs[b, i, 0], 1e-6, 1.0 - 1e-6))
                l_z = float(np.log(p_l_raw / (1.0 - p_l_raw)))
                p_l_s = 1.0 if (i in exp_set) else float(1.0 / (1.0 + np.exp(-np.clip((l_z - thr_logit) / t_l, -20.0, 20.0))))
                if rng_b.random() < p_l_s:
                    t_raw = sprobs[b, i, 1 : n + 1].astype(np.float64)
                    t_lgt = np.where(e.legal_edges[i], np.log(np.clip(t_raw, 1e-9, 1.0)) / t_t, -1e9)
                    t_lgt -= t_lgt.max()
                    tw = np.where(e.legal_edges[i], np.exp(t_lgt), 0.0)
                    tw_sum = float(tw.sum())
                    if tw_sum <= 0.0:
                        cls = 0
                    else:
                        tw /= tw_sum
                        cls = 1 + int(np.searchsorted(np.cumsum(tw), rng_b.random()))
                        if cls > n:
                            cls = n
                else:
                    cls = 0
            elif sample:
                p_row = sprobs[b, i, : n + 1].astype(np.float64)
                if i in exp_set and p_row[1:].sum() > 1e-9:
                    p_row[0] = 0.0
                ps = p_row.sum()
                if ps <= 0.0:
                    cls = 0
                else:
                    p_row /= ps
                    cls = int(np.searchsorted(np.cumsum(p_row), rng_b.random()))
                    if cls > n:
                        cls = n
            else:
                p_launch = 1.0 - float(sprobs[b, i, 0])
                if p_launch < launch_threshold and i not in exp_set:
                    cls = 0
                else:
                    cls = 1 + int(np.argmax(sprobs[b, i, 1 : n + 1]))
            act_cls[i] = cls
            old_lp[i] = float(logp[b, i, cls])
        fr_b = fracs_np[b, :n, :n] if fracs_np is not None else np.ones((n, n), dtype=np.float32)
        moves = _decode_sampled_actions(e, act_cls, fr_b, force_all_in=force_all_in)
        out.append((moves, act_cls, old_lp, float(vals_np[b])))
    return out


def collect_episodes_batched_2p(
    policy: PlanetTimelineTransformer,
    opp_model: Optional[PlanetTimelineTransformer],
    *,
    specs: Sequence[Tuple[int, ...]],  # (seed, hero_slot) or (seed, hero_slot, group_id, rng_seed)
    temperature: float = 0.80,
    max_steps: int = 220,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    shaping_coef: float = 0.02,
    early_reset: bool = True,
    force_all_in: bool = False,
    rollout_d4: bool = True,
    future_horizon: int = 10,
    opp_temperature: float = 0.0,
    hober_opp: Optional[Any] = None,
    hober_tta: bool = False,
    hober_temperature: float = 0.0,
    hober_launch_threshold: float = 0.53,
    launch_threshold: float = 0.50,
    factorized_sample: bool = False,
    target_temperature: float = 0.30,
    exact_factorized_ppo: bool = False,
) -> List[Tuple[Dict[str, np.ndarray], Dict[str, float]]]:
    """Run ``len(specs)`` 2P neural-pool or live `#2 Hober Malloc` games in lockstep with shared physics."""
    m = len(specs)
    if m == 0:
        return []
    hero_fv = int(getattr(policy, "feature_version", 1))
    opp_fv = int(getattr(opp_model, "feature_version", 1)) if opp_model is not None else (2 if hober_opp is not None else hero_fv)
    tl_dim_pol = int(getattr(policy, "timeline_dim", 8))

    parsed_specs: List[Tuple[int, int, int, int]] = []
    for sp in specs:
        if len(sp) >= 4:
            parsed_specs.append((int(sp[0]), int(sp[1]), int(sp[2]), int(sp[3])))
        else:
            parsed_specs.append((int(sp[0]), int(sp[1]), -1, int(sp[0])))

    rngs = [np.random.default_rng(rseed) for _, _, _, rseed in parsed_specs]
    sims = [FastOrbitSim.from_seed(seed=s, num_agents=2, max_steps=max_steps) for s, _, _, _ in parsed_specs]
    hero_slots = [sl for _, sl, _, _ in parsed_specs]
    group_ids = [gid for _, _, gid, _ in parsed_specs]
    k_rots: List[int] = []
    flip_xs: List[bool] = []
    opp_k_rots: List[int] = []
    for i, (s, _, gid, _) in enumerate(parsed_specs):
        if gid >= 0:
            grp_rng = np.random.default_rng(gid * 31337 + s)
            k_rots.append(int(grp_rng.integers(0, 4)) if rollout_d4 else 0)
            flip_xs.append(bool(grp_rng.integers(0, 2)) if rollout_d4 else False)
            opp_k_rots.append(int(grp_rng.integers(0, 4)))
        else:
            k_rots.append(int(rngs[i].integers(0, 4)) if rollout_d4 else 0)
            flip_xs.append(bool(rngs[i].integers(0, 2)) if rollout_d4 else False)
            opp_k_rots.append(int(rngs[i].integers(0, 4)))

    traj_pf: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_tf: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_ef: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_le: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_om: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_pm: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_act: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_logp: List[List[np.ndarray]] = [[] for _ in range(m)]
    traj_vals: List[List[float]] = [[] for _ in range(m)]
    traj_rews: List[List[float]] = [[] for _ in range(m)]
    traj_steps: List[List[int]] = [[] for _ in range(m)]
    traj_pids: List[List[List[int]]] = [[] for _ in range(m)]
    planet_snaps: List[Dict[int, Dict[int, Tuple[int, int]]]] = [{} for _ in range(m)]
    prev_phis = [float(evaluate_state(sims[i], hero_slots[i])) for i in range(m)]
    early_winners: List[Optional[int]] = [None] * m
    done_flags = [sims[i].done for i in range(m)]

    opp_sample = opp_temperature > 1e-4
    opp_temp = opp_temperature if opp_sample else 1.0
    zero_kr = [0] * m
    zero_fx = [False] * m

    while not all(done_flags):
        active = [i for i in range(m) if not done_flags[i]]
        h_encs: List[EncodedPlanetState] = []
        o_encs: List[EncodedPlanetState] = []
        active_sh_phys: List[Dict[str, Any]] = []
        h_has_owned: List[bool] = []
        act_kr: List[int] = []
        act_fx: List[bool] = []
        act_okr: List[int] = []
        act_rngs: List[np.random.Generator] = []

        for idx in active:
            sim = sims[idx]
            planet_snaps[idx][int(sim.step_count)] = {int(p[0]): (int(p[1]), int(p[5])) for p in sim.planets}
            o_sh = sim.to_observation(player=0)
            sh_phys = build_shared_planet_physics(o_sh, max_speed=sim.ship_speed)
            hp = hero_slots[idx]
            eh = extract_planet_features(
                o_sh, player=hp, max_speed=sim.ship_speed, feature_version=hero_fv, shared_physics=sh_phys
            )
            eo = extract_planet_features(
                o_sh, player=1 - hp, max_speed=sim.ship_speed, feature_version=opp_fv, shared_physics=sh_phys
            )
            h_encs.append(eh)
            o_encs.append(eo)
            active_sh_phys.append(sh_phys)
            h_has_owned.append(bool(np.any(eh.owned_mask)))
            act_kr.append(k_rots[idx])
            act_fx.append(flip_xs[idx])
            act_okr.append(opp_k_rots[idx])
            act_rngs.append(rngs[idx])

        h_out = _batched_policy_steps(
            policy,
            h_encs,
            k_rots=act_kr,
            flip_xs=act_fx,
            sample=True,
            temperature=temperature,
            launch_threshold=launch_threshold,
            force_all_in=force_all_in,
            rngs=act_rngs,
            factorized_sample=factorized_sample,
            target_temperature=target_temperature,
            exact_factorized_ppo=exact_factorized_ppo,
        )
        if hober_opp is not None:
            o_moves_list = hober_opp.act_batch(
                o_encs,
                active_sh_phys,
                rotation_tta=hober_tta,
                launch_threshold=hober_launch_threshold,
                temperature=hober_temperature,
                k_rots=act_okr,
                rngs=act_rngs,
            )
        else:
            o_out = _batched_policy_steps(
                opp_model if opp_model is not None else policy,
                o_encs,
                k_rots=zero_kr[: len(active)],
                flip_xs=zero_fx[: len(active)],
                sample=opp_sample,
                temperature=opp_temp,
                launch_threshold=launch_threshold,
                force_all_in=force_all_in,
                rngs=act_rngs,
                factorized_sample=factorized_sample,
                target_temperature=target_temperature,
            )
            o_moves_list = [o_out[k][0] for k in range(len(active))]

        for k, idx in enumerate(active):
            sim = sims[idx]
            hp = hero_slots[idx]
            eh = h_encs[k]
            moves_h, act_cls, old_lp, val_h = h_out[k]
            moves_o = o_moves_list[k]
            recorded = h_has_owned[k]
            if recorded:
                padded = pad_encoded_state(eh, MAX_PLANETS)
                pf_s = padded["planet_feats"]
                tf_s = padded["timeline_feats"][..., :tl_dim_pol]
                kr, fx = k_rots[idx], flip_xs[idx]
                if kr or fx:
                    pf_s = apply_d4_planet_feats(pf_s, k_rot=kr, flip_x=fx)
                    tf_s = apply_d4_timeline_feats(tf_s, k_rot=kr, flip_x=fx)
                traj_pf[idx].append(pf_s)
                traj_tf[idx].append(tf_s)
                traj_ef[idx].append(padded["edge_feats"])
                traj_le[idx].append(padded["legal_edges"])
                traj_om[idx].append(padded["owned_mask"])
                traj_pm[idx].append(padded["pad_mask"])
                traj_act[idx].append(act_cls)
                traj_logp[idx].append(old_lp)
                traj_vals[idx].append(val_h)
                traj_steps[idx].append(int(sim.step_count))
                traj_pids[idx].append(list(eh.planet_ids))

            acts = [[], []]
            acts[hp] = moves_h if recorded else []
            acts[1 - hp] = moves_o
            sim.step(acts)
            if recorded:
                cur_phi = float(evaluate_state(sim, hp))
                traj_rews[idx].append(shaping_coef * (gamma * cur_phi - prev_phis[idx]))
                prev_phis[idx] = cur_phi

            if sim.done:
                done_flags[idx] = True
            elif early_reset and sim.step_count >= 12:
                totals = sim.scores()
                p0_c = sim.planet_count(0)
                p1_c = sim.planet_count(1)
                if (p0_c > 0) ^ (p1_c > 0):
                    sole = 0 if p0_c > 0 else 1
                    if totals[sole] >= max(totals) and sum(totals) > 0:
                        early_winners[idx] = sole
                        done_flags[idx] = True

    results: List[Tuple[Dict[str, np.ndarray], Dict[str, float]]] = []
    for idx in range(m):
        sim = sims[idx]
        hp = hero_slots[idx]
        planet_snaps[idx][int(sim.step_count)] = {int(p[0]): (int(p[1]), int(p[5])) for p in sim.planets}
        totals = sim.scores()
        if early_winners[idx] is not None:
            win_pid = int(early_winners[idx])
        elif sim.done:
            win_pid = sim.winner()
        else:
            max_s = max(totals)
            winners = [p_i for p_i, sc in enumerate(totals) if sc == max_s]
            win_pid = winners[0] if len(winners) == 1 else -1

        term_rew = 1.0 if win_pid == hp else (0.0 if win_pid < 0 else -1.0)
        hero_rank = 1.0 if win_pid == hp else (1.5 if win_pid < 0 else 2.0)
        rews = traj_rews[idx]
        vals = traj_vals[idx]
        if rews:
            rews[-1] += term_rew

        t_len = len(rews)
        advs = np.zeros((t_len,), dtype=np.float32)
        rets = np.zeros((t_len,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(t_len)):
            next_v = 0.0 if (t == t_len - 1) else vals[t + 1]
            delta = rews[t] + gamma * next_v - vals[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advs[t] = last_gae
            rets[t] = np.clip(last_gae + vals[t], -1.2, 1.2)

        gid = int(group_ids[idx])
        stats = {
            "win": 1.0 if win_pid == hp else 0.0,
            "rank": hero_rank,
            "steps": float(sim.step_count),
            "term_rew": float(term_rew),
            "num_players": 2.0,
            "hero_slot": float(hp),
            "is_hober": 1.0 if hober_opp is not None else 0.0,
            "group_id": float(gid),
        }
        if t_len == 0:
            empty = {
                "planet_feats": np.zeros((0, MAX_PLANETS, 22), np.float16),
                "timeline_feats": np.zeros((0, MAX_PLANETS, 20, tl_dim_pol), np.float16),
                "edge_feats": np.zeros((0, MAX_PLANETS, MAX_PLANETS, 12), np.float16),
                "legal_edges": np.zeros((0, MAX_PLANETS, MAX_PLANETS), bool),
                "owned_mask": np.zeros((0, MAX_PLANETS), bool),
                "pad_mask": np.zeros((0, MAX_PLANETS), bool),
                "action_cls": np.zeros((0, MAX_PLANETS), np.int64),
                "old_logp": np.zeros((0, MAX_PLANETS), np.float32),
                "advantages": np.zeros((0,), np.float32),
                "returns": np.zeros((0,), np.float32),
                "step_ticks": np.zeros((0,), np.int32),
                "is_hober": np.zeros((0,), bool),
                "group_id": np.zeros((0,), np.int64),
                "term_rew": np.zeros((0,), np.float32),
                "aux_owner": np.zeros((0, MAX_PLANETS), np.int64),
                "aux_ships": np.zeros((0, MAX_PLANETS), np.float32),
                "aux_valid": np.zeros((0, MAX_PLANETS), bool),
            }
            results.append((empty, stats))
            continue

        traj_ao: List[np.ndarray] = []
        traj_ash: List[np.ndarray] = []
        traj_av: List[np.ndarray] = []
        snaps = planet_snaps[idx]
        steps_i = traj_steps[idx]
        pids_i = traj_pids[idx]
        for idx_t in range(t_len):
            ao = np.full((MAX_PLANETS,), 2, dtype=np.int64)
            ash = np.zeros((MAX_PLANETS,), dtype=np.float32)
            av = np.zeros((MAX_PLANETS,), dtype=bool)
            fut_map = snaps.get(steps_i[idx_t] + future_horizon)
            if fut_map is not None:
                for i_p, pid_val in enumerate(pids_i[idx_t]):
                    rec = fut_map.get(int(pid_val))
                    if rec is None:
                        continue
                    ow, sh = rec
                    ao[i_p] = 0 if ow == hp else (2 if ow < 0 else 1)
                    ash[i_p] = float(np.log1p(max(0, sh)) / 6.0)
                    av[i_p] = True
            traj_ao.append(ao)
            traj_ash.append(ash)
            traj_av.append(av)

        batch = {
            "planet_feats": np.stack(traj_pf[idx], axis=0).astype(np.float16, copy=False),
            "timeline_feats": np.stack(traj_tf[idx], axis=0).astype(np.float16, copy=False),
            "edge_feats": np.stack(traj_ef[idx], axis=0).astype(np.float16, copy=False),
            "legal_edges": np.stack(traj_le[idx], axis=0),
            "owned_mask": np.stack(traj_om[idx], axis=0),
            "pad_mask": np.stack(traj_pm[idx], axis=0),
            "action_cls": np.stack(traj_act[idx], axis=0),
            "old_logp": np.stack(traj_logp[idx], axis=0),
            "advantages": advs,
            "returns": rets,
            "step_ticks": np.asarray(steps_i, dtype=np.int32),
            "is_hober": np.full((t_len,), hober_opp is not None, dtype=bool),
            "group_id": np.full((t_len,), gid, dtype=np.int64),
            "term_rew": np.full((t_len,), float(term_rew), dtype=np.float32),
            "aux_owner": np.stack(traj_ao, axis=0),
            "aux_ships": np.stack(traj_ash, axis=0),
            "aux_valid": np.stack(traj_av, axis=0),
        }
        results.append((batch, stats))
    return results


def _worker_collect_batch(
    task: Tuple[Any, ...]
) -> List[Tuple[Dict[str, np.ndarray], Dict[str, float]]]:
    """Persistent worker entry point running a batched lockstep chunk of 2P pool or `#2 Hober` games."""
    _configure_worker_threads()
    (
        policy_path,
        policy_version,
        opp_path,
        specs,
        temperature,
        max_steps,
        shaping_coef,
        early_reset,
        force_all_in,
        rollout_d4,
        opp_temperature,
    ) = task[:11]
    gamma = float(task[11]) if len(task) > 11 else 0.99
    gae_lambda = float(task[12]) if len(task) > 12 else 0.95
    hober_tta = bool(task[13]) if len(task) > 13 else False
    hober_temp = float(task[14]) if len(task) > 14 else 0.0
    launch_threshold = float(task[15]) if len(task) > 15 else 0.50
    factorized_sample = bool(task[16]) if len(task) > 16 else False
    target_temperature = float(task[17]) if len(task) > 17 else 0.30
    exact_factorized_ppo = bool(task[18]) if len(task) > 18 else False

    policy = _get_worker_policy(policy_path, policy_version)
    if opp_path == _HOBER_SENTINEL:
        hober_opp = _get_worker_hober()
        opp_model = None
    else:
        hober_opp = None
        opp_model = _get_worker_opp(opp_path)
    return collect_episodes_batched_2p(
        policy,
        opp_model if opp_model is not None else (None if hober_opp is not None else policy),
        specs=specs,
        temperature=temperature,
        max_steps=max_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        shaping_coef=shaping_coef,
        early_reset=early_reset,
        force_all_in=force_all_in,
        rollout_d4=rollout_d4,
        opp_temperature=opp_temperature,
        hober_opp=hober_opp,
        hober_tta=hober_tta,
        hober_temperature=hober_temp,
        launch_threshold=launch_threshold,
        factorized_sample=factorized_sample,
        target_temperature=target_temperature,
        exact_factorized_ppo=exact_factorized_ppo,
    )


def _worker_collect_episode(
    task: Tuple[
        str,
        int,
        Optional[str],
        int,
        str,
        int,
        Optional[int],
        float,
        int,
        float,
        bool,
        bool,
        bool,
        float,
    ]
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Persistent worker entry point using file-cached checkpoints (zero tensor IPC)."""
    _configure_worker_threads()
    (
        policy_path,
        policy_version,
        opp_path,
        num_players,
        opponent_kind,
        seed,
        hero_slot,
        temperature,
        max_steps,
        shaping_coef,
        early_reset,
        force_all_in,
        rollout_d4,
        opp_temperature,
    ) = task
    policy = _get_worker_policy(policy_path, policy_version)
    opp_model = _get_worker_opp(opp_path)
    return collect_episode(
        policy,
        opp_model if opp_model is not None else policy,
        num_players=num_players,
        opponent_kind=opponent_kind,
        seed=seed,
        temperature=temperature,
        max_steps=max_steps,
        shaping_coef=shaping_coef,
        early_reset=early_reset,
        force_all_in=force_all_in,
        opp_model=opp_model,
        hero_slot=hero_slot,
        rollout_d4=rollout_d4,
        opp_temperature=opp_temperature,
    )


def _play_deterministic_eval_game(
    policy: PlanetTimelineTransformer,
    opp_model: Optional[PlanetTimelineTransformer],
    *,
    num_players: int,
    seed: int,
    hero_slot: int,
    max_steps: int,
    force_all_in: bool,
    launch_threshold: float = 0.50,
    hober_opp: Optional[Any] = None,
    hober_tta: bool = True,
    decoupled_logit_tta: bool = False,
) -> Tuple[float, float]:
    sim = FastOrbitSim.from_seed(seed=seed, num_agents=num_players, max_steps=max_steps)
    hero_fv = int(getattr(policy, "feature_version", 1))
    opp_fv = int(getattr(opp_model, "feature_version", 1)) if opp_model is not None else (2 if hober_opp is not None else hero_fv)
    last_alive = [0] * num_players
    while not sim.done:
        acts = []
        obs_shared = sim.to_observation(player=0)
        shared_phys = build_shared_planet_physics(obs_shared, max_speed=sim.ship_speed)
        for pid in range(num_players):
            enc = extract_planet_features(
                obs_shared,
                player=pid,
                max_speed=sim.ship_speed,
                feature_version=(hero_fv if pid == hero_slot else opp_fv),
                shared_physics=shared_phys,
            )
            if pid == hero_slot:
                if hober_opp is not None and hober_tta:
                    # Fair 4x rotational TTA for Hero when evaluated against 4x-TTA #2 Hober
                    n_p = len(enc.planet_ids)
                    if n_p == 0 or not np.any(enc.owned_mask):
                        mv = []
                    else:
                        tl_d = int(getattr(policy, "timeline_dim", 8))
                        tf_raw = enc.timeline_feats[..., :tl_d]
                        pf_4 = np.stack([apply_d4_planet_feats(enc.planet_feats, k_rot=k, flip_x=False) for k in range(4)], 0)
                        tf_4 = np.stack([apply_d4_timeline_feats(tf_raw, k_rot=k, flip_x=False) for k in range(4)], 0)
                        ef_1 = enc.edge_feats[None]
                        pm_4 = np.ones((4, n_p), dtype=bool)
                        le_t = torch.from_numpy(enc.legal_edges)
                        with torch.no_grad():
                            lg, _, _ = policy(
                                torch.from_numpy(pf_4),
                                torch.from_numpy(tf_4),
                                torch.from_numpy(ef_1),
                                torch.from_numpy(pm_4),
                                legal_edges=le_t.unsqueeze(0).expand(4, n_p, n_p),
                                compute_frac=False,
                            )
                            mk = torch.cat([torch.ones((n_p, 1), dtype=torch.bool), le_t], dim=-1)
                            lg_m = lg.masked_fill(~mk.unsqueeze(0), -1e9)
                            if decoupled_logit_tta and getattr(policy, "arch_version", 1) >= 2:
                                lp_all = F.log_softmax(lg_m, dim=-1)
                                lp_h = lp_all[..., 0]
                                has_l = le_t.any(dim=-1)
                                safe_l = le_t | (~has_l.unsqueeze(-1))
                                lp_l = torch.logsumexp(lp_all[..., 1:].masked_fill(~safe_l.unsqueeze(0), -1e9), dim=-1)
                                mean_p_l = torch.where(has_l, torch.sigmoid((lp_l - lp_h).clamp(-15.0, 15.0).mean(dim=0)), torch.zeros_like(has_l, dtype=torch.float32))
                                cond_tgt = (lp_all[..., 1:] - lp_l.unsqueeze(-1)).masked_fill(~safe_l.unsqueeze(0), -1e9)
                                mean_p_t = F.softmax(cond_tgt.mean(dim=0).masked_fill(~safe_l, -1e9), dim=-1).masked_fill(~le_t, 0.0)
                                pr4 = torch.cat([(1.0 - mean_p_l).unsqueeze(-1), mean_p_l.unsqueeze(-1) * mean_p_t], dim=-1).numpy()
                            else:
                                pr4 = F.softmax(lg_m, dim=-1).mean(dim=0).numpy()
                        act_cls = np.zeros((MAX_PLANETS,), dtype=np.int64)
                        exp_set = set(enc.expiring_comet_indices)
                        for i_p in range(n_p):
                            if not enc.owned_mask[i_p] or not np.any(enc.legal_edges[i_p]):
                                continue
                            if (1.0 - float(pr4[i_p, 0])) >= launch_threshold or (i_p in exp_set):
                                act_cls[i_p] = 1 + int(np.argmax(pr4[i_p, 1 : n_p + 1]))
                        mv = _decode_sampled_actions(
                            enc, act_cls, np.ones((n_p, n_p), dtype=np.float32), force_all_in=force_all_in
                        )
                else:
                    mv, _, _, _ = _policy_step(
                        policy,
                        enc,
                        sample=False,
                        launch_threshold=launch_threshold,
                        force_all_in=force_all_in,
                    )
            elif hober_opp is not None:
                mv = hober_opp.act_batch(
                    [enc],
                    [shared_phys],
                    rotation_tta=hober_tta,
                    launch_threshold=0.53,
                    temperature=0.0,
                )[0]
            else:
                assert opp_model is not None
                mv, _, _, _ = _policy_step(
                    opp_model,
                    enc,
                    sample=False,
                    launch_threshold=launch_threshold,
                    force_all_in=force_all_in,
                )
            acts.append(mv)
        sim.step(acts)
        if num_players > 2:
            sc = sim.scores()
            for pid in range(num_players):
                if sc[pid] > 0:
                    last_alive[pid] = sim.step_count

    if num_players == 2:
        w = sim.winner()
        won = 1.0 if w == hero_slot else (0.5 if w < 0 else 0.0)
        rk = 1.0 if w == hero_slot else (1.5 if w < 0 else 2.0)
        return won, rk

    sc = sim.scores()
    order = sorted(range(num_players), key=lambda i: (sc[i], last_alive[i]), reverse=True)
    rk = float(order.index(hero_slot) + 1)
    return (1.0 if rk == 1.0 else 0.0), rk


def _worker_eval_game(
    task: Tuple[Any, ...]
) -> Tuple[str, int, float, float]:
    _configure_worker_threads()
    if len(task) >= 11:
        policy_path, policy_version, opp_path, opp_tag, num_players, seed, hero_slot, max_steps, force_all_in, launch_threshold, decoupled_logit_tta = task[:11]
    elif len(task) >= 10:
        policy_path, policy_version, opp_path, opp_tag, num_players, seed, hero_slot, max_steps, force_all_in, launch_threshold = task[:10]
        decoupled_logit_tta = False
    else:
        policy_path, policy_version, opp_path, opp_tag, num_players, seed, hero_slot, max_steps, force_all_in = task
        launch_threshold = 0.50
        decoupled_logit_tta = False
    policy = _get_worker_policy(policy_path, policy_version)
    if opp_path == _HOBER_SENTINEL:
        hober_opp = _get_worker_hober()
        opp_model = None
    else:
        hober_opp = None
        opp_model = _get_worker_opp(opp_path)
        assert opp_model is not None
    won, rk = _play_deterministic_eval_game(
        policy,
        opp_model,
        num_players=num_players,
        seed=seed,
        hero_slot=hero_slot,
        max_steps=max_steps,
        force_all_in=force_all_in,
        launch_threshold=launch_threshold,
        hober_opp=hober_opp,
        hober_tta=True,
        decoupled_logit_tta=bool(decoupled_logit_tta),
    )
    return opp_tag, num_players, won, rk


def _eval_vs_fixed_anchor(
    policy: PlanetTimelineTransformer,
    fixed_anchor: PlanetTimelineTransformer,
    *,
    seed_base: int = 12000,
    games_2p: int = 6,
    games_4p: int = 4,
    max_steps: int = 240,
    force_all_in: bool = False,
) -> Dict[str, float]:
    """Deterministic greedy evaluation against the frozen starting anchor (2P + optional 4P)."""
    policy.eval()
    fixed_anchor.eval()
    w2 = 0.0
    for g in range(games_2p):
        won, _ = _play_deterministic_eval_game(
            policy,
            fixed_anchor,
            num_players=2,
            seed=seed_base + g,
            hero_slot=g % 2,
            max_steps=max_steps,
            force_all_in=force_all_in,
        )
        w2 += won

    w4 = 0.0
    r4_sum = 0.0
    for g in range(games_4p):
        won, rk = _play_deterministic_eval_game(
            policy,
            fixed_anchor,
            num_players=4,
            seed=seed_base + 500 + g,
            hero_slot=g % 4,
            max_steps=max_steps,
            force_all_in=force_all_in,
        )
        w4 += won
        r4_sum += rk

    wr2 = w2 / max(1, games_2p)
    wr4 = w4 / max(1, games_4p)
    score = wr2 if games_4p == 0 else (0.55 * wr2 + 0.45 * wr4)
    return {
        "eval_2p_wr": wr2,
        "eval_4p_wr": wr4,
        "eval_4p_rank": (r4_sum / games_4p) if games_4p > 0 else 0.0,
        "eval_score": score,
    }


def _eval_vs_anchors_parallel(
    *,
    cpu_policy: PlanetTimelineTransformer,
    policy_path: Optional[str],
    policy_version: int,
    eval_anchors: Sequence[Tuple[str, str, Optional[PlanetTimelineTransformer]]],
    pool_exec: Optional[ProcessPoolExecutor],
    seed_base: int,
    eval_pairs: int,
    two_player_only: bool,
    max_steps: int,
    force_all_in: bool,
    launch_threshold: float = 0.50,
    hober_eval_pairs: int = 0,
    hober_eval_seed_base: int = 500,
    decoupled_logit_tta: bool = False,
) -> Dict[str, Any]:
    """Run paired-seed deterministic 2P (and optional 4P + live `#2 Hober`) validation."""
    results: List[Tuple[str, int, float, float]] = []
    want_hober = hober_eval_pairs > 0
    if pool_exec is not None and policy_path is not None:
        tasks = []
        for a_idx, (tag, opp_path, _) in enumerate(eval_anchors):
            for g in range(eval_pairs):
                s = seed_base + a_idx * 100 + g
                tasks.append((policy_path, policy_version, opp_path, tag, 2, s, 0, max_steps, force_all_in, launch_threshold))
                tasks.append((policy_path, policy_version, opp_path, tag, 2, s, 1, max_steps, force_all_in, launch_threshold))
        if want_hober:
            for g in range(hober_eval_pairs):
                s = hober_eval_seed_base + g
                tasks.append((policy_path, policy_version, _HOBER_SENTINEL, "hober", 2, s, 0, max(360, max_steps), force_all_in, launch_threshold, decoupled_logit_tta))
                tasks.append((policy_path, policy_version, _HOBER_SENTINEL, "hober", 2, s, 1, max(360, max_steps), force_all_in, launch_threshold, decoupled_logit_tta))
        if not two_player_only and eval_anchors:
            tag0, opp_path0, _ = eval_anchors[0]
            for g in range(4):
                tasks.append((policy_path, policy_version, opp_path0, tag0, 4, seed_base + 500 + g, g % 4, max_steps, force_all_in, launch_threshold))
        results = list(pool_exec.map(_worker_eval_game, tasks))
    else:
        cpu_policy.eval()
        for a_idx, (tag, _, opp_m) in enumerate(eval_anchors):
            assert opp_m is not None
            opp_m.eval()
            for g in range(eval_pairs):
                s = seed_base + a_idx * 100 + g
                for seat in (0, 1):
                    w, rk = _play_deterministic_eval_game(
                        cpu_policy, opp_m, num_players=2, seed=s, hero_slot=seat, max_steps=max_steps, force_all_in=force_all_in, launch_threshold=launch_threshold
                    )
                    results.append((tag, 2, w, rk))
        if want_hober:
            hober_local = _get_worker_hober()
            for g in range(hober_eval_pairs):
                s = hober_eval_seed_base + g
                for seat in (0, 1):
                    w, rk = _play_deterministic_eval_game(
                        cpu_policy,
                        None,
                        num_players=2,
                        seed=s,
                        hero_slot=seat,
                        max_steps=max(360, max_steps),
                        force_all_in=force_all_in,
                        launch_threshold=launch_threshold,
                        hober_opp=hober_local,
                        hober_tta=True,
                        decoupled_logit_tta=decoupled_logit_tta,
                    )
                    results.append(("hober", 2, w, rk))
        if not two_player_only and eval_anchors:
            tag0, _, opp_m0 = eval_anchors[0]
            assert opp_m0 is not None
            for g in range(4):
                w, rk = _play_deterministic_eval_game(
                    cpu_policy, opp_m0, num_players=4, seed=seed_base + 500 + g, hero_slot=g % 4, max_steps=max_steps, force_all_in=force_all_in, launch_threshold=launch_threshold
                )
                results.append((tag0, 4, w, rk))

    per_tag_2p: Dict[str, List[float]] = {tag: [] for tag, _, _ in eval_anchors}
    if want_hober:
        per_tag_2p["hober"] = []
    wr4_list: List[float] = []
    rk4_list: List[float] = []
    for tag, n_pl, won, rk in results:
        if n_pl == 2:
            per_tag_2p[tag].append(won)
        else:
            wr4_list.append(won)
            rk4_list.append(rk)

    per_tag_wr = {tag: (float(np.mean(vals)) if vals else 0.0) for tag, vals in per_tag_2p.items()}
    anchor_only = [v for k, v in per_tag_wr.items() if k != "hober"]
    avg_anchors = float(np.mean(anchor_only)) if anchor_only else 0.0
    if want_hober and "hober" in per_tag_wr:
        hober_wr = per_tag_wr["hober"]
        avg_2p = 0.70 * hober_wr + 0.30 * avg_anchors if anchor_only else hober_wr
    else:
        avg_2p = float(np.mean(list(per_tag_wr.values()))) if per_tag_wr else 0.0
    wr4 = float(np.mean(wr4_list)) if wr4_list else 0.0
    rk4 = float(np.mean(rk4_list)) if rk4_list else 0.0
    score = avg_2p if two_player_only else (0.55 * avg_2p + 0.45 * wr4)
    return {
        "per_tag_2p": per_tag_wr,
        "avg_2p": avg_2p,
        "eval_4p_wr": wr4,
        "eval_4p_rank": rk4,
        "eval_score": score,
        "total_games": len(results),
    }


def _trim_batch_n(
    pf: torch.Tensor,
    tf: torch.Tensor,
    ef: torch.Tensor,
    le: torch.Tensor,
    om: torch.Tensor,
    pm: torch.Tensor,
    act: torch.Tensor,
    old_lp: torch.Tensor,
    aux_o: Optional[torch.Tensor] = None,
    aux_s: Optional[torch.Tensor] = None,
    aux_v: Optional[torch.Tensor] = None,
):
    max_n = max(2, int(pm.sum(dim=1).max().item()))
    if max_n < pf.shape[1]:
        pf = pf[:, :max_n]
        tf = tf[:, :max_n]
        ef = ef[:, :max_n, :max_n]
        le = le[:, :max_n, :max_n]
        om = om[:, :max_n]
        pm = pm[:, :max_n]
        act = act[:, :max_n]
        old_lp = old_lp[:, :max_n]
        if aux_o is not None:
            aux_o = aux_o[:, :max_n]
        if aux_s is not None:
            aux_s = aux_s[:, :max_n]
        if aux_v is not None:
            aux_v = aux_v[:, :max_n]
    return pf, tf, ef, le, om, pm, act, old_lp, aux_o, aux_s, aux_v


def _decoupled_factorized_kl(
    logp_all: torch.Tensor,
    ref_logp_all: torch.Tensor,
    act_mask: torch.Tensor,
    target_kl_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DirectML-native binary launch KL + weighted conditional target KL."""
    neg_inf = -1e9
    tgt_mask = act_mask[..., 1:]
    has_t = tgt_mask.any(dim=-1, keepdim=True)
    safe_t = tgt_mask | (~has_t)
    tgt_mask_f = tgt_mask.to(logp_all.dtype)

    log_p_l_raw = torch.logsumexp(logp_all[..., 1:].masked_fill(~safe_t, neg_inf), dim=-1)
    log_q_l_raw = torch.logsumexp(ref_logp_all[..., 1:].masked_fill(~safe_t, neg_inf), dim=-1)
    lp2 = F.log_softmax(torch.stack([logp_all[..., 0], log_p_l_raw], dim=-1), dim=-1)
    lq2 = F.log_softmax(torch.stack([ref_logp_all[..., 0], log_q_l_raw], dim=-1), dim=-1)
    p2 = lp2.exp()
    launch_kl = (p2 * (lp2 - lq2)).sum(dim=-1).clamp_min(0.0)

    log_pt = F.log_softmax(logp_all[..., 1:].masked_fill(~safe_t, neg_inf), dim=-1)
    log_qt = F.log_softmax(ref_logp_all[..., 1:].masked_fill(~safe_t, neg_inf), dim=-1)
    pt = log_pt.exp() * tgt_mask_f
    tgt_kl = (pt * (log_pt - log_qt) * tgt_mask_f).sum(dim=-1).clamp_min(0.0)
    tgt_w = torch.maximum(p2[..., 1].detach(), lq2[..., 1].exp().detach()).clamp_min(0.20)
    total_kl = launch_kl + float(target_kl_weight) * tgt_w * tgt_kl
    return total_kl, launch_kl, tgt_kl


def ppo_update(
    policy: PlanetTimelineTransformer,
    anchor: PlanetTimelineTransformer,
    optimizer: torch.optim.Optimizer,
    rollout: Dict[str, np.ndarray],
    *,
    ppo_epochs: int = 3,
    batch_size: int = 64,
    clip_eps: float = 0.20,
    kl_coef: float = 0.05,
    ent_coef: float = 0.008,
    value_coef: float = 0.30,
    midgame_adv_weight: float = 1.0,
    fixed_anchor: Optional[PlanetTimelineTransformer] = None,
    anchor_kl_coef: float = 0.0,
    aux_coef: float = 0.0,
    device: Optional[torch.device] = None,
    use_dml: bool = False,
    augment_d4: bool = True,
    value_only: bool = False,
    freeze_value_head: bool = False,
    max_kl: float = 0.0,
    detach_value: bool = False,
    exact_factorized_ppo: bool = False,
    target_kl_weight: float = 0.0,
    use_group_grpo: bool = False,
    temperature: float = 0.75,
    target_temperature: float = 0.30,
    launch_threshold: float = 0.50,
) -> Dict[str, float]:
    """Perform clipped per-planet PPO + Dual-KL + Aux world-model updates (or value-head warmup)."""
    n = len(rollout["returns"])
    if n == 0:
        return {
            "loss": 0.0,
            "kl": 0.0,
            "anchor_kl": 0.0,
            "launch_kl": 0.0,
            "target_kl": 0.0,
            "entropy": 0.0,
            "val_loss": 0.0,
            "aux_loss": 0.0,
            "aux_acc": 0.0,
        }

    adv = rollout["advantages"].copy()
    valid_cnts = (rollout["owned_mask"] & rollout["legal_edges"].any(axis=-1)).sum(axis=1).astype(np.float64)

    def _norm_adv_planet_weighted(g_adv: np.ndarray, g_w: np.ndarray) -> np.ndarray:
        w_sum = float(g_w.sum())
        if w_sum > 0.0:
            mean_w = float((g_adv * g_w).sum() / w_sum)
            var_w = float((((g_adv - mean_w) ** 2) * g_w).sum() / w_sum)
            std_w = float(np.sqrt(max(var_w, 1e-12)))
            out = np.clip((g_adv - mean_w) / (std_w + 1e-6), -2.5, 2.5)
            # Re-center after clipping so planet-weighted mean stays identically 0
            out = out - float((out * g_w).sum() / w_sum)
            return out.astype(np.float32)
        return np.clip((g_adv - g_adv.mean()) / (g_adv.std() + 1e-6), -2.5, 2.5).astype(np.float32)

    if use_group_grpo and ("group_id" in rollout) and ("term_rew" in rollout):
        group_ids = rollout["group_id"]
        term_rews = rollout["term_rew"].astype(np.float64)
        adv = np.zeros((n,), dtype=np.float32)
        for gid in np.unique(group_ids):
            if gid < 0:
                continue
            g_mask = group_ids == gid
            g_r = term_rews[g_mask]
            if float(g_r.max() - g_r.min()) < 1e-4:
                continue  # All K won or all K lost -> zero noisy gradient
            g_w = valid_cnts[g_mask]
            w_sum = float(g_w.sum())
            if w_sum <= 0.0:
                continue
            mean_r = float((g_r * g_w).sum() / w_sum)
            var_r = float((((g_r - mean_r) ** 2) * g_w).sum() / w_sum)
            std_r = float(np.sqrt(max(var_r, 1e-6)))
            g_adv = np.clip((g_r - mean_r) / (std_r + 1e-6), -2.5, 2.5)
            g_adv = g_adv - float((g_adv * g_w).sum() / w_sum)
            adv[g_mask] = g_adv.astype(np.float32)
    else:
        is_hober_arr = rollout.get("is_hober")
        if (
            is_hober_arr is not None
            and len(is_hober_arr) == n
            and int(np.sum(is_hober_arr)) >= 16
            and int(np.sum(~is_hober_arr)) >= 16
        ):
            # GRPO-style opponent-group planet-weighted advantage normalization: keeps live Boss wins
            # strongly positive with zero planet-count bias toward majority-class HOLD.
            for grp in (is_hober_arr, ~is_hober_arr):
                adv[grp] = _norm_adv_planet_weighted(adv[grp], valid_cnts[grp])
        else:
            adv = _norm_adv_planet_weighted(adv, valid_cnts)
    step_ticks = rollout.get("step_ticks")
    if step_ticks is not None and midgame_adv_weight != 1.0:
        mid_mask = (step_ticks >= 25) & (step_ticks <= 75)
        w = np.where(mid_mask, float(midgame_adv_weight), 1.0).astype(np.float32)
        w /= max(1e-6, float(w.mean()))
        adv = adv * w

    train_dev = device if device is not None else next(policy.parameters()).device
    is_dml_dev = bool(use_dml or (train_dev.type == "privateuseone"))
    has_aux_data = ("aux_owner" in rollout) and ("aux_ships" in rollout) and ("aux_valid" in rollout)
    want_aux = bool((not value_only) and aux_coef > 0.0 and getattr(policy, "aux_future_head", None) is not None and has_aux_data)

    policy.train()
    anchor.eval()
    if fixed_anchor is not None:
        fixed_anchor.eval()

    saved_req_grad: Optional[List[Tuple[nn.Parameter, bool]]] = None
    if value_only:
        saved_req_grad = [(p, p.requires_grad) for p in policy.parameters()]
        for name, p in policy.named_parameters():
            p.requires_grad_(name.startswith("val_attn.") or name.startswith("val_head."))
    elif freeze_value_head:
        saved_req_grad = [(p, p.requires_grad) for p in policy.parameters()]
        for name, p in policy.named_parameters():
            if name.startswith("val_attn.") or name.startswith("val_head."):
                p.requires_grad_(False)

    loss_acc = torch.zeros((), device=train_dev)
    kl_acc = torch.zeros((), device=train_dev)
    akl_acc = torch.zeros((), device=train_dev)
    lkl_acc = torch.zeros((), device=train_dev)
    tkl_acc = torch.zeros((), device=train_dev)
    ent_acc = torch.zeros((), device=train_dev)
    vl_acc = torch.zeros((), device=train_dev)
    aux_loss_acc = torch.zeros((), device=train_dev)
    aux_hits_acc = torch.zeros((), device=train_dev)
    aux_tot_acc = torch.zeros((), device=train_dev)
    n_steps = 0

    # When D4 augmentation is already applied at rollout time (augment_d4=False),
    # precompute frozen teacher & fixed-anchor log-probs once per iter instead of
    # recomputing both 4.3M reference networks on every PPO epoch (3x -> 1x).
    tl_dim_pol = int(getattr(policy, "timeline_dim", 8))
    max_p_total = int(rollout["pad_mask"].shape[1])
    want_fixed = bool((not value_only) and fixed_anchor is not None and anchor_kl_coef > 0.0)
    use_ref_cache = bool((not value_only) and (not augment_d4) and ppo_epochs > 1)
    ref_logp_cache: Optional[torch.Tensor] = None
    fixed_logp_cache: Optional[torch.Tensor] = None
    p_counts = rollout["pad_mask"].sum(axis=1).astype(np.int32)
    if use_ref_cache:
        ref_logp_cache = torch.zeros((n, max_p_total, max_p_total + 1), dtype=torch.float16)
        if want_fixed:
            fixed_logp_cache = torch.zeros((n, max_p_total, max_p_total + 1), dtype=torch.float16)
        ref_bs = min(max(batch_size, 256), 256)
        ref_batches = snap_bucket_batches(p_counts, ref_bs, buckets=(24, 32, 44), max_cap=max_p_total, shuffle=False)
        with torch.no_grad():
            for idx_c, cn in ref_batches:
                pm_c = rollout["pad_mask"][idx_c]
                pf_c = torch.from_numpy(np.ascontiguousarray(rollout["planet_feats"][idx_c, :cn])).to(train_dev).float()
                tf_c = torch.from_numpy(
                    np.ascontiguousarray(rollout["timeline_feats"][idx_c, :cn, :, :tl_dim_pol])
                ).to(train_dev).float()
                ef_c = torch.from_numpy(np.ascontiguousarray(rollout["edge_feats"][idx_c, :cn, :cn])).to(train_dev).float()
                le_c = torch.from_numpy(np.ascontiguousarray(rollout["legal_edges"][idx_c, :cn, :cn])).to(train_dev)
                pm_ct = torch.from_numpy(np.ascontiguousarray(pm_c[:, :cn])).to(train_dev)
                hold_c = torch.ones((len(idx_c), cn, 1), dtype=torch.bool, device=train_dev)
                mask_c = torch.cat([hold_c, le_c], dim=-1)

                r_log, _, _ = anchor(pf_c, tf_c, ef_c, pm_ct, legal_edges=le_c, compute_frac=False)
                r_lp = F.log_softmax(r_log.masked_fill(~mask_c, -1e9), dim=-1)
                idx_t = torch.from_numpy(idx_c)
                ref_logp_cache[idx_t, :cn, : cn + 1] = r_lp.clamp(-30.0, 0.0).to(torch.float16).cpu()

                if fixed_logp_cache is not None and fixed_anchor is not None:
                    f_log, _, _ = fixed_anchor(pf_c, tf_c, ef_c, pm_ct, legal_edges=le_c, compute_frac=False)
                    f_lp = F.log_softmax(f_log.masked_fill(~mask_c, -1e9), dim=-1)
                    fixed_logp_cache[idx_t, :cn, : cn + 1] = f_lp.clamp(-30.0, 0.0).to(torch.float16).cpu()

    try:
        stop_kl = False
        for _ in range(ppo_epochs):
            if stop_kl:
                break
            batch_idx_list = snap_bucket_batches(
                p_counts, batch_size, buckets=(24, 32, 44), max_cap=max_p_total, shuffle=True
            )
            ep_approx_kl = torch.zeros((), device=train_dev)
            ep_steps = 0
            for idx, max_n in batch_idx_list:
                pm_full = rollout["pad_mask"][idx]

                le_np = np.ascontiguousarray(rollout["legal_edges"][idx, :max_n, :max_n])
                om_np = np.ascontiguousarray(rollout["owned_mask"][idx, :max_n])
                has_valid = bool(np.any(om_np & le_np.any(axis=-1)))

                aux_v_np = np.ascontiguousarray(rollout["aux_valid"][idx, :max_n]) if want_aux else None
                has_aux_valid = bool(aux_v_np is not None and np.any(aux_v_np))

                pf = torch.from_numpy(np.ascontiguousarray(rollout["planet_feats"][idx, :max_n])).to(train_dev).float()
                tf = torch.from_numpy(
                    np.ascontiguousarray(rollout["timeline_feats"][idx, :max_n, :, :tl_dim_pol])
                ).to(train_dev).float()
                if augment_d4:
                    k_rot = random.randrange(4)
                    flip_x = bool(random.randrange(2))
                    pf, tf = apply_d4_batch_gpu(pf, tf, k_rot=k_rot, flip_x=flip_x)
                ef = torch.from_numpy(np.ascontiguousarray(rollout["edge_feats"][idx, :max_n, :max_n])).to(train_dev).float()
                le = torch.from_numpy(le_np).to(train_dev)
                om = torch.from_numpy(om_np).to(train_dev)
                pm = torch.from_numpy(np.ascontiguousarray(pm_full[:, :max_n])).to(train_dev)
                ret_t = torch.from_numpy(rollout["returns"][idx]).to(train_dev)

                if value_only:
                    _, _, val = policy(pf, tf, ef, pm, legal_edges=le, compute_frac=False, value_only=True)
                    v_loss = F.mse_loss(val, ret_t)
                    optimizer.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad],
                        1.0,
                        foreach=not is_dml_dev,
                    )
                    optimizer.step()
                    loss_acc = loss_acc + v_loss.detach()
                    vl_acc = vl_acc + v_loss.detach()
                    n_steps += 1
                    continue

                act = torch.from_numpy(np.ascontiguousarray(rollout["action_cls"][idx, :max_n])).to(train_dev)
                old_lp = torch.from_numpy(np.ascontiguousarray(rollout["old_logp"][idx, :max_n])).to(train_dev)
                adv_t = torch.from_numpy(adv[idx]).to(train_dev)
                aux_o = (
                    torch.from_numpy(np.ascontiguousarray(rollout["aux_owner"][idx, :max_n])).to(train_dev)
                    if want_aux
                    else None
                )
                aux_s = (
                    torch.from_numpy(np.ascontiguousarray(rollout["aux_ships"][idx, :max_n])).to(train_dev)
                    if want_aux
                    else None
                )
                aux_v = torch.from_numpy(aux_v_np).to(train_dev) if aux_v_np is not None else None

                eff_detach_value = bool(detach_value or freeze_value_head)
                if want_aux:
                    logits, _, val, aux_pred = policy(
                        pf, tf, ef, pm, return_aux=True, legal_edges=le, compute_frac=False, detach_value=eff_detach_value
                    )
                else:
                    logits, _, val = policy(
                        pf, tf, ef, pm, legal_edges=le, compute_frac=False, detach_value=eff_detach_value
                    )
                    aux_pred = None

                B, N, _ = logits.shape
                hold_ok = torch.ones((B, N, 1), dtype=torch.bool, device=train_dev)
                act_mask = torch.cat([hold_ok, le], dim=-1)
                logits = logits.masked_fill(~act_mask, -1e9)

                if ref_logp_cache is not None:
                    ref_logp_all = ref_logp_cache[idx, :N, : N + 1].to(device=train_dev, dtype=logits.dtype)
                    fixed_logp_all = (
                        fixed_logp_cache[idx, :N, : N + 1].to(device=train_dev, dtype=logits.dtype)
                        if fixed_logp_cache is not None
                        else None
                    )
                else:
                    with torch.no_grad():
                        ref_logits, _, _ = anchor(pf, tf, ef, pm, legal_edges=le, compute_frac=False)
                        ref_logp_all = F.log_softmax(ref_logits.masked_fill(~act_mask, -1e9), dim=-1)
                        if want_fixed and fixed_anchor is not None:
                            fixed_logits, _, _ = fixed_anchor(pf, tf, ef, pm, legal_edges=le, compute_frac=False)
                            fixed_logp_all = F.log_softmax(fixed_logits.masked_fill(~act_mask, -1e9), dim=-1)
                        else:
                            fixed_logp_all = None

                if has_valid:
                    valid_f = (om & le.any(dim=-1)).float()
                    denom = valid_f.sum().clamp_min(1.0)
                    logp_all = F.log_softmax(logits, dim=-1)
                    probs_all = logp_all.exp()
                    act_mask_f = act_mask.to(logp_all.dtype)

                    act_oh = one_hot_no_scatter(act.clamp(0, N), N + 1, dtype=logp_all.dtype)
                    if exact_factorized_ppo:
                        thr_logit_ppo = float(np.log(max(1e-5, launch_threshold) / max(1e-5, 1.0 - launch_threshold)))
                        logq_all = factorized_log_probs(logits, act_mask, temperature, target_temperature, thr_logit_ppo)
                        new_lp = (logq_all * act_oh).sum(dim=-1)
                    else:
                        new_lp = (logp_all * act_oh).sum(dim=-1)
                    ratio = torch.exp((new_lp - old_lp).clamp(-4.0, 4.0))
                    adv_bn = adv_t.unsqueeze(1).expand(B, N)
                    surr1 = ratio * adv_bn
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_bn
                    pol_loss = -(torch.min(surr1, surr2) * valid_f).sum() / denom

                    if target_kl_weight > 0.0:
                        kl_per, lkl_per, tkl_per = _decoupled_factorized_kl(
                            logp_all, ref_logp_all, act_mask, target_kl_weight
                        )
                        lkl_mean = (lkl_per * valid_f).sum() / denom
                        tkl_mean = (tkl_per * valid_f).sum() / denom
                    else:
                        kl_per = (probs_all * (logp_all - ref_logp_all) * act_mask_f).sum(dim=-1).clamp_min(0.0)
                        lkl_mean = logits.sum() * 0.0
                        tkl_mean = logits.sum() * 0.0
                    kl_mean = (kl_per * valid_f).sum() / denom
                    ent_per = -(probs_all * logp_all * act_mask_f).sum(dim=-1)
                    ent_mean = (ent_per * valid_f).sum() / denom

                    if fixed_logp_all is not None:
                        if target_kl_weight > 0.0:
                            akl_per, _, _ = _decoupled_factorized_kl(
                                logp_all, fixed_logp_all, act_mask, target_kl_weight
                            )
                        else:
                            akl_per = (probs_all * (logp_all - fixed_logp_all) * act_mask_f).sum(dim=-1).clamp_min(0.0)
                        anchor_kl_mean = (akl_per * valid_f).sum() / denom
                    else:
                        anchor_kl_mean = logits.sum() * 0.0
                else:
                    pol_loss = logits.sum() * 0.0
                    kl_mean = logits.sum() * 0.0
                    anchor_kl_mean = logits.sum() * 0.0
                    lkl_mean = logits.sum() * 0.0
                    tkl_mean = logits.sum() * 0.0
                    ent_mean = logits.sum() * 0.0

                v_loss = F.mse_loss(val.detach() if freeze_value_head else val, ret_t)

                if aux_pred is not None and aux_v is not None and has_aux_valid and aux_o is not None and aux_s is not None:
                    av_f = aux_v.float()
                    av_denom = av_f.sum().clamp_min(1.0)
                    ao_logp = F.log_softmax(aux_pred[..., :3], dim=-1)
                    ao_oh = one_hot_no_scatter(aux_o.clamp(0, 2), 3, dtype=ao_logp.dtype)
                    ao_ce = -(ao_logp * ao_oh).sum(dim=-1)
                    ao_loss = (ao_ce * av_f).sum() / av_denom
                    as_loss = (F.smooth_l1_loss(aux_pred[..., 3], aux_s, reduction="none") * av_f).sum() / av_denom
                    aux_loss = ao_loss + as_loss
                    with torch.no_grad():
                        aux_hits_acc = aux_hits_acc + ((aux_pred[..., :3].argmax(dim=-1) == aux_o).float() * av_f).sum()
                        aux_tot_acc = aux_tot_acc + av_f.sum()
                else:
                    aux_loss = logits.sum() * 0.0

                eff_value_coef = 0.0 if freeze_value_head else value_coef
                total_loss = (
                    pol_loss
                    + eff_value_coef * v_loss
                    + kl_coef * kl_mean
                    + anchor_kl_coef * anchor_kl_mean
                    - ent_coef * ent_mean
                    + aux_coef * aux_loss
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    0.7,
                    foreach=not is_dml_dev,
                )
                optimizer.step()

                loss_acc = loss_acc + total_loss.detach()
                kl_acc = kl_acc + kl_mean.detach()
                akl_acc = akl_acc + anchor_kl_mean.detach()
                lkl_acc = lkl_acc + lkl_mean.detach()
                tkl_acc = tkl_acc + tkl_mean.detach()
                ent_acc = ent_acc + ent_mean.detach()
                vl_acc = vl_acc + v_loss.detach()
                aux_loss_acc = aux_loss_acc + aux_loss.detach()
                if max_kl > 0.0 and has_valid:
                    ep_approx_kl = ep_approx_kl + (0.5 * (new_lp - old_lp).detach().pow(2) * valid_f).sum() / denom
                ep_steps += 1
                n_steps += 1
                if max_kl > 0.0 and (not value_only) and (ep_steps % 4 == 0):
                    if float(ep_approx_kl.item()) / ep_steps > float(max_kl):
                        stop_kl = True
                        break
            if max_kl > 0.0 and ep_steps > 0 and (not value_only) and (not stop_kl):
                if float(ep_approx_kl.item()) / ep_steps > float(max_kl):
                    stop_kl = True
                    break
    finally:
        if saved_req_grad is not None:
            for p, req in saved_req_grad:
                p.requires_grad_(req)

    denom_steps = max(1, n_steps)
    aux_tot_val = float(aux_tot_acc.item())
    return {
        "loss": float(loss_acc.item()) / denom_steps,
        "kl": float(kl_acc.item()) / denom_steps,
        "anchor_kl": float(akl_acc.item()) / denom_steps,
        "launch_kl": float(lkl_acc.item()) / denom_steps,
        "target_kl": float(tkl_acc.item()) / denom_steps,
        "entropy": float(ent_acc.item()) / denom_steps,
        "val_loss": float(vl_acc.item()) / denom_steps,
        "aux_loss": float(aux_loss_acc.item()) / denom_steps,
        "aux_acc": (float(aux_hits_acc.item()) / aux_tot_val) if aux_tot_val > 0.0 else 0.0,
    }


def _ema_update_teacher(teacher: PlanetTimelineTransformer, policy: PlanetTimelineTransformer, tau: float) -> None:
    if tau <= 0.0:
        return
    with torch.no_grad():
        for tp, pp in zip(teacher.parameters(), policy.parameters()):
            tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)


def _tag_for_ckpt(path: Path) -> str:
    stem = path.stem.replace("planet_transformer_", "")
    return stem[:14] if stem else path.stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-Play League PPO + Delayed Teacher KL for PlanetTimelineTransformer.")
    parser.add_argument("--init", required=True, help="Starting checkpoint path.")
    parser.add_argument("--pool", nargs="*", default=[], help="Additional historical opponent checkpoints.")
    parser.add_argument("--out", default="checkpoints/planet_transformer_v3_rl.pt")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--episodes-per-iter", type=int, default=12)
    parser.add_argument("--mirror-pairs", type=int, default=0, help="Paired-seed 2P map count per iter (2*M episodes).")
    parser.add_argument("--grpo-k", type=int, default=1, help="Same-seed same-seat counterfactual GRPO group size (1=off, 4=recommended).")
    parser.add_argument("--paired-seeds", action="store_true", help="Force paired-seed rollout mode.")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--train-threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "dml", "cuda", "auto"], default="cpu")
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--lr-min-ratio", type=float, default=0.20)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--anchor-kl-coef", type=float, default=0.025)
    parser.add_argument("--target-kl-weight", type=float, default=0.0, help="Decoupled conditional target-KL weight (0=coupled KL).")
    parser.add_argument("--max-kl", type=float, default=0.0, help="Optional per-epoch approx-KL early-stop ceiling (0=off).")
    parser.add_argument("--detach-value", action="store_true", help="Detach trunk hidden state before value head during PPO updates.")
    parser.add_argument("--aux-coef", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.003, help="Entropy bonus coefficient (keep low during RL sharpening).")
    parser.add_argument("--teacher-ema", type=float, default=0.12, help="EMA rate to update KL teacher on positive iters.")
    parser.add_argument("--shaping-coef", type=float, default=0.02, help="Near-sparse shaping weight (0=pure terminal).")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--target-temperature", type=float, default=0.30, help="Target softmax temperature when --factorized-sample is enabled.")
    parser.add_argument("--factorized-sample", action="store_true", help="Sample Launch vs Hold (centered on --launch-threshold) independently of Target | Launch (fixes PT-v2 HOLD collapse).")
    parser.add_argument("--exact-factorized-ppo", action="store_true", help="Compute exact on-policy log q_theta(a|s) ratios/old_logp matching factorized low-T sampling.")
    parser.add_argument("--decoupled-logit-tta", action="store_true", help="Use decoupled launch/target logit-space TTA during in-loop Hober evaluation.")
    parser.add_argument("--opp-temperature", type=float, default=0.0, help="Pool opponent sampling temperature (0=greedy).")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for GAE returns.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda trace decay.")
    parser.add_argument("--hober-share", type=float, default=0.0, help="Fraction of 2P mirror pairs played live against #2 Hober Malloc.")
    parser.add_argument("--hober-tta", action="store_true", help="Use 4x rotation TTA for live #2 Hober sparring (100% parity; off=fast 96.7%).")
    parser.add_argument("--hober-temperature", type=float, default=0.0, help="Sampling temperature for live #2 Hober sparring (0=deterministic t=0.53).")
    parser.add_argument("--hober-eval-pairs", type=int, default=0, help="Paired-seed games vs exact TTA=1 #2 Hober Malloc in each eval gate.")
    parser.add_argument("--value-warmup-iters", type=int, default=0, help="Warm up only val_attn/val_head for first N iters (eliminates value-head trunk shock).")
    parser.add_argument("--freeze-value-head", action="store_true", help="Freeze val_attn/val_head parameters and skip value loss during PPO/GRPO updates.")
    parser.add_argument("--eval-every", type=int, default=2, help="Run deterministic anchor eval every N iters.")
    parser.add_argument("--eval-pairs", type=int, default=8, help="Paired-seed games per eval anchor.")
    parser.add_argument("--eval-opponents", nargs="*", default=[], help="Extra checkpoints for multi-anchor eval gate.")
    parser.add_argument("--force-all-in", action="store_true", help="Train with 100% all-in sends matching Gen 8/9 2P inference.")
    parser.add_argument("--two-player-only", action="store_true", help="Focus 100% of rollouts and anchor eval on 2P matches.")
    parser.add_argument("--no-weak-opps", action="store_true", help="Replace subham_v14/heuristic slots with 100% neural pool opponents.")
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic snapshot every N iterations (0=off).")
    parser.add_argument("--fixed-anchor", default="", help="Optional separate fixed KL anchor checkpoint path.")
    parser.add_argument("--launch-threshold", type=float, default=0.50, help="Deterministic eval launch threshold.")
    parser.add_argument("--midgame-adv-weight", type=float, default=1.0, help="Policy-gradient multiplier for midgame ticks t in [25, 75].")
    parser.add_argument("--rollout-batch", type=int, default=8, help="Lockstep batched 2P games per worker chunk (1=serial).")
    parser.add_argument("--async-rollout", action="store_true", help="Overlap CPU rollouts for iter k+1 with GPU PPO updates for iter k.")
    parser.add_argument("--seed", type=int, default=9500)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_dev, use_dml = resolve_train_device(args.device)

    init_path = Path(args.init)
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    cpu_policy = PlanetTimelineTransformer(**cfg)
    cpu_policy.load_state_dict(ckpt["model_state_dict"])
    cpu_policy.eval()

    policy = copy.deepcopy(cpu_policy).to(train_dev)
    if args.fixed_anchor and Path(args.fixed_anchor).is_file():
        fa_ckpt = torch.load(args.fixed_anchor, map_location="cpu", weights_only=False)
        fa_cfg = fa_ckpt.get("model_config", cfg)
        fixed_anchor = PlanetTimelineTransformer(**fa_cfg)
        fixed_anchor.load_state_dict(fa_ckpt["model_state_dict"])
        fixed_anchor = fixed_anchor.to(train_dev).eval()
    else:
        fixed_anchor = copy.deepcopy(cpu_policy).to(train_dev).eval()
    teacher = copy.deepcopy(cpu_policy).to(train_dev).eval()
    for m in (fixed_anchor, teacher):
        for p in m.parameters():
            p.requires_grad_(False)

    if args.freeze_value_head:
        opt_params = [
            p for name, p in policy.named_parameters()
            if not (name.startswith("val_attn.") or name.startswith("val_head."))
        ]
    else:
        opt_params = list(policy.parameters())
    opt = build_optimizer(opt_params, lr=args.lr, weight_decay=1e-4, use_dml=use_dml)
    val_head_params = [p for name, p in policy.named_parameters() if name.startswith("val_attn.") or name.startswith("val_head.")]
    val_opt = build_optimizer(val_head_params, lr=max(3e-4, args.lr * 10.0), weight_decay=1e-4, use_dml=use_dml)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iterations - args.value_warmup_iters), eta_min=args.lr * float(args.lr_min_ratio)
    )
    out_path = Path(args.out)
    latest_path = out_path.with_name(out_path.stem + "_latest" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pinned pool checkpoints (persisted permanently) + eval anchor checkpoints
    pinned_pool: List[Tuple[str, str, PlanetTimelineTransformer]] = [
        ("init", str(init_path), copy.deepcopy(cpu_policy).eval())
    ]
    seen_paths = {str(init_path.resolve())}
    for p_path in args.pool:
        pp = Path(p_path)
        if pp.is_file():
            m = _load_worker_model(str(pp))
            pinned_pool.append((_tag_for_ckpt(pp), str(pp), m))
            seen_paths.add(str(pp.resolve()))

    eval_anchors: List[Tuple[str, str, PlanetTimelineTransformer]] = [
        ("init", str(init_path), pinned_pool[0][2])
    ]
    for e_path in args.eval_opponents:
        ep = Path(e_path)
        if ep.is_file() and str(ep.resolve()) not in {str(init_path.resolve())}:
            eval_anchors.append((_tag_for_ckpt(ep), str(ep), _load_worker_model(str(ep))))

    has_subham = (REPO_ROOT / "external" / "opponents" / "subham_v14.py").is_file()
    use_paired = bool(args.paired_seeds or args.mirror_pairs > 0)
    n_pairs = args.mirror_pairs if args.mirror_pairs > 0 else max(1, args.episodes_per_iter // 2)

    if args.two_player_only:
        if args.no_weak_opps:
            league_cycle = [(2, "pool")] * 10
        elif has_subham:
            league_cycle = [(2, "pool")] * 8 + [(2, "subham_v14"), (2, "heuristic")]
        else:
            league_cycle = [(2, "pool")] * 9 + [(2, "heuristic")]
    else:
        league_cycle = [
            (2, "pool"),
            (2, "pool"),
            (4, "pool"),
            (2, "pool"),
            (4, "subham_v14" if has_subham else "pool"),
            (2, "subham_v14" if has_subham else "pool"),
            (4, "pool"),
            (2, "pool"),
            (4, "heuristic"),
            (2, "pool"),
            (4, "subham_v14" if has_subham else "pool"),
            (2, "heuristic"),
        ]

    dev_tag = "dml" if use_dml else str(train_dev)
    hober_tag = (
        f" hober={args.hober_share:.2f}(tta={int(args.hober_tta)},T={args.hober_temperature:.2f},ev={args.hober_eval_pairs}p)"
        if (args.hober_share > 0.0 or args.hober_eval_pairs > 0)
        else ""
    )
    grpo_k = max(1, int(args.grpo_k))
    print(
        f"[RL-League-PPO] Loaded {args.init} | dev={dev_tag} | pool={[n for n, _, _ in pinned_pool]} | "
        f"iters={args.iterations} "
        + (
            (f"grpo={n_pairs}x{grpo_k} (2P={n_pairs*grpo_k}ep)" if grpo_k > 1 else f"pairs={n_pairs} (2P={2*n_pairs}ep)")
            if use_paired
            else f"eps/iter={args.episodes_per_iter}"
        )
        + f" workers={args.workers} rbatch={args.rollout_batch} async={args.async_rollout} | lr={args.lr} clip={args.clip_eps} kl={args.kl_coef} akl={args.anchor_kl_coef} "
        f"ema={args.teacher_ema} aux={args.aux_coef} shaping={args.shaping_coef} gamma={args.gamma} lam={args.gae_lambda} T={args.temperature} "
        f"oppT={args.opp_temperature} max_t={args.max_steps} all_in={args.force_all_in} 2p_only={args.two_player_only} "
        f"freeze_v={int(args.freeze_value_head)} mid_adv={args.midgame_adv_weight}{hober_tag}",
        flush=True,
    )

    t0 = time.time()
    best_eval_score = -1.0
    rolling_snaps: List[Tuple[str, str, PlanetTimelineTransformer]] = []
    tmp_dir_ctx = None
    pool_exec: Optional[ProcessPoolExecutor] = None

    try:
        if args.workers > 1:
            tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="ow_gen10_ppo_")
            pool_exec = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))

        def _save_cur_policy(version: int) -> Optional[str]:
            if tmp_dir_ctx is None:
                return None
            p = Path(tmp_dir_ctx.name) / f"policy_cur_{version % 4}.pt"
            torch.save({"model_config": cpu_policy.config, "model_state_dict": cpu_policy.state_dict()}, p)
            return str(p)

        def _submit_rollout(
            it_idx: int,
            pol_path_arg: Optional[str],
            pol_ver_arg: int,
            active_pool_arg: List[Tuple[str, str, PlanetTimelineTransformer]],
        ):
            """Build and submit rollout jobs (batched 2P lockstep when eligible)."""
            if use_paired:
                n_hober_pairs = max(0, min(n_pairs, int(round(n_pairs * float(args.hober_share))))) if args.two_player_only else 0
                pair_specs: List[Tuple[int, str, int, Optional[str], Optional[PlanetTimelineTransformer]]] = []
                for k in range(n_pairs):
                    n_pl, opp_kind = league_cycle[(it_idx * n_pairs + k) % len(league_cycle)]
                    if args.two_player_only:
                        n_pl = 2
                    map_seed = args.seed + it_idx * 1000 + k
                    opp_path: Optional[str] = None
                    opp_model: Optional[PlanetTimelineTransformer] = None
                    if k < n_hober_pairs:
                        opp_kind = "hober"
                        opp_path = _HOBER_SENTINEL
                    elif opp_kind == "pool":
                        u = random.random()
                        snap_thresh = 0.25 if grpo_k > 1 else 0.30
                        if rolling_snaps and u < snap_thresh:
                            _, opp_path, opp_model = rolling_snaps[-1]
                        elif grpo_k > 1 or u < 0.75:
                            _, opp_path, opp_model = random.choice(pinned_pool)
                        else:
                            _, opp_path, opp_model = random.choice(active_pool_arg)
                    pair_specs.append((n_pl, opp_kind, map_seed, opp_path, opp_model))

                can_batch = (
                    pool_exec is not None
                    and pol_path_arg is not None
                    and args.rollout_batch > 1
                    and all(n_pl == 2 and ok in ("pool", "hober") for n_pl, ok, _, _, _ in pair_specs)
                )
                if can_batch:
                    by_opp: Dict[Optional[str], List[Tuple[int, ...]]] = {}
                    if grpo_k > 1:
                        for g_idx, (_, _, map_seed, opp_path, _) in enumerate(pair_specs):
                            seat = g_idx % 2
                            gid = it_idx * 1000 + g_idx
                            for r_idx in range(grpo_k):
                                rseed = map_seed * 1009 + r_idx * 97 + seat * 13 + it_idx * 7919
                                by_opp.setdefault(opp_path, []).append((map_seed, seat, gid, rseed))
                        chunk_sz = max(grpo_k, (int(args.rollout_batch) // grpo_k) * grpo_k)
                    else:
                        for _, _, map_seed, opp_path, _ in pair_specs:
                            by_opp.setdefault(opp_path, []).extend([(map_seed, 0), (map_seed, 1)])
                        chunk_sz = max(2, (int(args.rollout_batch) // 2) * 2)
                    # Schedule heaviest chunks (live #2 Hober sparring) first to avoid stragglers
                    ordered_items = sorted(
                        by_opp.items(),
                        key=lambda kv: (0 if kv[0] == _HOBER_SENTINEL else 1, -len(kv[1])),
                    )
                    futs = []
                    for opp_path, sp_list in ordered_items:
                        for s_i in range(0, len(sp_list), chunk_sz):
                            chunk = sp_list[s_i : s_i + chunk_sz]
                            futs.append(
                                pool_exec.submit(
                                    _worker_collect_batch,
                                    (
                                        pol_path_arg,
                                        pol_ver_arg,
                                        opp_path,
                                        chunk,
                                        args.temperature,
                                        args.max_steps,
                                        args.shaping_coef,
                                        True,
                                        args.force_all_in,
                                        True,
                                        args.opp_temperature,
                                        args.gamma,
                                        args.gae_lambda,
                                        args.hober_tta,
                                        args.hober_temperature,
                                        args.launch_threshold,
                                        args.factorized_sample,
                                        args.target_temperature,
                                        args.exact_factorized_ppo,
                                    ),
                                )
                            )
                    return ("batched", futs)

                if pool_exec is not None and pol_path_arg is not None:
                    futs = []
                    for n_pl, opp_kind, map_seed, opp_path, _ in pair_specs:
                        for seat in (0, 1):
                            futs.append(
                                pool_exec.submit(
                                    _worker_collect_episode,
                                    (
                                        pol_path_arg,
                                        pol_ver_arg,
                                        opp_path,
                                        n_pl,
                                        opp_kind,
                                        map_seed,
                                        seat,
                                        args.temperature,
                                        args.max_steps,
                                        args.shaping_coef,
                                        True,
                                        args.force_all_in,
                                        True,
                                        args.opp_temperature,
                                    ),
                                )
                            )
                    return ("single", futs)

                direct_res = []
                for n_pl, opp_kind, map_seed, _, opp_m in pair_specs:
                    for seat in (0, 1):
                        direct_res.append(
                            collect_episode(
                                cpu_policy,
                                opp_m if opp_m is not None else cpu_policy,
                                num_players=n_pl,
                                opponent_kind=opp_kind,
                                seed=map_seed,
                                temperature=args.temperature,
                                max_steps=args.max_steps,
                                shaping_coef=args.shaping_coef,
                                early_reset=True,
                                force_all_in=args.force_all_in,
                                opp_model=opp_m,
                                hero_slot=seat,
                                rollout_d4=True,
                                opp_temperature=args.opp_temperature,
                            )
                        )
                return ("direct", direct_res)

            # Non-paired rollout mode
            futs = []
            direct_res = []
            for e in range(args.episodes_per_iter):
                n_pl, opp_kind = league_cycle[(it_idx * args.episodes_per_iter + e) % len(league_cycle)]
                ep_seed = args.seed + it_idx * 1000 + e
                opp_path = None
                opp_model = None
                if opp_kind == "pool":
                    if e % 2 == 0 or len(active_pool_arg) == 1:
                        _, opp_path, opp_model = active_pool_arg[-1]
                    else:
                        _, opp_path, opp_model = random.choice(active_pool_arg)
                if pool_exec is not None and pol_path_arg is not None:
                    futs.append(
                        pool_exec.submit(
                            _worker_collect_episode,
                            (
                                pol_path_arg,
                                pol_ver_arg,
                                opp_path,
                                n_pl,
                                opp_kind,
                                ep_seed,
                                None,
                                args.temperature,
                                args.max_steps,
                                args.shaping_coef,
                                True,
                                args.force_all_in,
                                True,
                                args.opp_temperature,
                            ),
                        )
                    )
                else:
                    direct_res.append(
                        collect_episode(
                            cpu_policy,
                            opp_model if opp_model is not None else cpu_policy,
                            num_players=n_pl,
                            opponent_kind=opp_kind,
                            seed=ep_seed,
                            temperature=args.temperature,
                            max_steps=args.max_steps,
                            shaping_coef=args.shaping_coef,
                            early_reset=True,
                            force_all_in=args.force_all_in,
                            opp_model=opp_model,
                            hero_slot=None,
                            rollout_d4=True,
                            opp_temperature=args.opp_temperature,
                        )
                    )
            return ("single", futs) if futs else ("direct", direct_res)

        def _collect_submitted(handle) -> List[Tuple[Dict[str, np.ndarray], Dict[str, float]]]:
            mode, payload = handle
            if mode == "batched":
                out_list: List[Tuple[Dict[str, np.ndarray], Dict[str, float]]] = []
                for f in payload:
                    out_list.extend(f.result())
                return out_list
            if mode == "single":
                return [f.result() for f in payload]
            return payload

        # Iter 0 baseline validation gate
        pol_p0 = _save_cur_policy(0)
        ev0 = _eval_vs_anchors_parallel(
            cpu_policy=cpu_policy,
            policy_path=pol_p0,
            policy_version=0,
            eval_anchors=eval_anchors,
            pool_exec=pool_exec,
            seed_base=args.seed + 50000,
            eval_pairs=args.eval_pairs,
            two_player_only=args.two_player_only,
            max_steps=max(240, args.max_steps),
            force_all_in=args.force_all_in,
            launch_threshold=args.launch_threshold,
            hober_eval_pairs=args.hober_eval_pairs,
            decoupled_logit_tta=args.decoupled_logit_tta,
        )
        best_eval_score = ev0["eval_score"]
        tag_str0 = " ".join(f"{t}={100*v:.1f}%" for t, v in ev0["per_tag_2p"].items())
        print(
            f"  [Baseline EVAL({ev0['total_games']}g)] {tag_str0} avg={100*ev0['avg_2p']:.1f}%",
            flush=True,
        )

        pending_rollout = None
        for it in range(1, args.iterations + 1):
            torch.set_num_threads(1)
            cpu_policy.eval()
            active_pool = pinned_pool + rolling_snaps

            t_roll = time.time()
            if pending_rollout is not None:
                results = _collect_submitted(pending_rollout)
                pending_rollout = None
                pol_p = _save_cur_policy(it)
            else:
                pol_p = _save_cur_policy(it)
                handle = _submit_rollout(it, pol_p, it, active_pool)
                results = _collect_submitted(handle)
            roll_sec = time.time() - t_roll

            batches: List[Dict[str, np.ndarray]] = []
            ep_wins_2p: List[float] = []
            ep_wins_hober: List[float] = []
            ep_wins_4p: List[float] = []
            ep_ranks_4p: List[float] = []
            ep_rews: List[float] = []

            for b, st in results:
                if len(b["returns"]) > 0:
                    batches.append(b)
                ep_rews.append(st["term_rew"])
                if int(st.get("num_players", 2)) == 2:
                    ep_wins_2p.append(st["win"])
                    if st.get("is_hober", 0.0) > 0.5:
                        ep_wins_hober.append(st["win"])
                else:
                    ep_wins_4p.append(st["win"])
                    ep_ranks_4p.append(st["rank"])

            splits = sweeps = 0
            mixed_groups = total_groups = 0
            if use_paired and grpo_k > 1:
                grp_wins: Dict[int, List[float]] = {}
                for _, st in results:
                    gid = int(st.get("group_id", -1))
                    if gid >= 0:
                        grp_wins.setdefault(gid, []).append(st["win"])
                total_groups = len(grp_wins)
                mixed_groups = sum(1 for ws in grp_wins.values() if max(ws) - min(ws) > 0.2)
            elif use_paired:
                for k in range(0, len(results) - 1, 2):
                    wa = results[k][1]["win"]
                    wb = results[k + 1][1]["win"]
                    if wa > 0.5 and wb > 0.5:
                        sweeps += 1
                    elif (wa > 0.5 and wb < 0.5) or (wb > 0.5 and wa < 0.5):
                        splits += 1

            if not batches:
                continue
            keys = list(batches[0].keys())
            rollout = {k: np.concatenate([b[k] for b in batches], axis=0) for k in keys}

            # Async pipelining: overlap CPU rollouts for iter it+1 with RX 6950 XT GPU update for iter it
            will_eval = (it % args.eval_every == 0) or (it == args.iterations)
            if args.async_rollout and pool_exec is not None and (it < args.iterations) and (not will_eval):
                pol_p_next = _save_cur_policy(it + 1)
                pending_rollout = _submit_rollout(it + 1, pol_p_next, it + 1, active_pool)

            is_val_warmup = it <= args.value_warmup_iters
            t_upd = time.time()
            torch.set_num_threads(2 if (pending_rollout is not None) else max(1, int(args.train_threads)))
            upd = ppo_update(
                policy,
                teacher,
                val_opt if is_val_warmup else opt,
                rollout,
                ppo_epochs=max(3, args.ppo_epochs) if is_val_warmup else args.ppo_epochs,
                batch_size=args.batch_size,
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
                ent_coef=args.ent_coef,
                midgame_adv_weight=args.midgame_adv_weight,
                fixed_anchor=fixed_anchor,
                anchor_kl_coef=args.anchor_kl_coef,
                aux_coef=args.aux_coef,
                device=train_dev,
                use_dml=use_dml,
                augment_d4=False,
                value_only=is_val_warmup,
                freeze_value_head=bool(args.freeze_value_head and not is_val_warmup),
                max_kl=args.max_kl,
                detach_value=args.detach_value,
                exact_factorized_ppo=bool(args.exact_factorized_ppo and args.factorized_sample),
                target_kl_weight=args.target_kl_weight,
                use_group_grpo=(grpo_k > 1),
                temperature=args.temperature,
                target_temperature=args.target_temperature,
                launch_threshold=args.launch_threshold,
            )
            if not is_val_warmup:
                sched.step()
            upd_sec = time.time() - t_upd

            cpu_sd = cpu_state_dict(policy)
            cpu_policy.load_state_dict(cpu_sd)

            wr2 = 100.0 * float(np.mean(ep_wins_2p)) if ep_wins_2p else 0.0
            wr_hober = 100.0 * float(np.mean(ep_wins_hober)) if ep_wins_hober else None
            wr4 = 100.0 * float(np.mean(ep_wins_4p)) if ep_wins_4p else 0.0
            rk4 = float(np.mean(ep_ranks_4p)) if ep_ranks_4p else 0.0
            mean_rew = float(np.mean(ep_rews)) if ep_rews else 0.0

            if (not is_val_warmup) and (mean_rew > 0.05 or (wr_hober is not None and wr_hober >= 40.0)):
                _ema_update_teacher(teacher, policy, args.teacher_ema)

            if (not is_val_warmup) and (it % 3 == 0):
                snap_copy = copy.deepcopy(cpu_policy).eval()
                for p in snap_copy.parameters():
                    p.requires_grad_(False)
                snap_path_str = ""
                if tmp_dir_ctx is not None:
                    sp = Path(tmp_dir_ctx.name) / f"snap_it{it}.pt"
                    torch.save({"model_config": cpu_policy.config, "model_state_dict": cpu_sd}, sp)
                    snap_path_str = str(sp)
                rolling_snaps.append((f"snap_it{it}", snap_path_str, snap_copy))
                if len(rolling_snaps) > 5:
                    rolling_snaps = rolling_snaps[-5:]

            eval_str = ""
            save_best = False
            if it % args.eval_every == 0 or it == args.iterations:
                pol_ev = _save_cur_policy(10000 + it)
                ev = _eval_vs_anchors_parallel(
                    cpu_policy=cpu_policy,
                    policy_path=pol_ev,
                    policy_version=10000 + it,
                    eval_anchors=eval_anchors,
                    pool_exec=pool_exec,
                    seed_base=args.seed + 50000,
                    eval_pairs=args.eval_pairs,
                    two_player_only=args.two_player_only,
                    max_steps=max(240, args.max_steps),
                    force_all_in=args.force_all_in,
                    launch_threshold=args.launch_threshold,
                    hober_eval_pairs=args.hober_eval_pairs,
                    decoupled_logit_tta=args.decoupled_logit_tta,
                )
                tag_str = " ".join(f"{t}={100*v:.1f}%" for t, v in ev["per_tag_2p"].items())
                if args.two_player_only:
                    eval_str = f" | EVAL({ev['total_games']}g): {tag_str} avg={100*ev['avg_2p']:.1f}%"
                else:
                    eval_str = (
                        f" | EVAL({ev['total_games']}g): {tag_str} avg={100*ev['avg_2p']:.1f}% "
                        f"4P={100*ev['eval_4p_wr']:.0f}%(rk={ev['eval_4p_rank']:.2f})"
                    )
                if ev["eval_score"] >= best_eval_score:
                    best_eval_score = ev["eval_score"]
                    save_best = True
                    eval_str += " *BEST*"

            if use_paired and grpo_k > 1:
                pair_str = f" (mix={mixed_groups}/{total_groups})"
            elif use_paired:
                pair_str = f" (sw={sweeps}, sp={splits})"
            else:
                pair_str = ""
            boss_str = f" boss={wr_hober:4.1f}%" if wr_hober is not None else ""
            vdec_mask = rollout["owned_mask"] & rollout["legal_edges"].any(axis=-1)
            lrate_pct = 100.0 * float((rollout["action_cls"][vdec_mask] > 0).mean()) if np.any(vdec_mask) else 0.0
            warm_str = " [VAL-WARMUP]" if is_val_warmup else ""
            stats_4p = "" if args.two_player_only else f" 4P={wr4:5.1f}%(rk={rk4:.2f})"
            aux_str = f" aux={100*upd['aux_acc']:.1f}%" if args.aux_coef > 0.0 and upd["aux_acc"] > 0.0 else ""
            fkl_str = (
                f" lkl={upd['launch_kl']:.4f} tkl={upd['target_kl']:.4f}"
                if args.target_kl_weight > 0.0
                else ""
            )
            print(
                f"  iter {it:2d}/{args.iterations} ({time.time()-t0:5.1f}s [roll={roll_sec:.1f}s upd={upd_sec:.1f}s]) | "
                f"states={len(rollout['returns']):4d} | 2P={wr2:5.1f}%{boss_str}{pair_str}{stats_4p} | "
                f"rew={mean_rew:+.2f} lr%={lrate_pct:4.1f}% kl={upd['kl']:.4f} akl={upd['anchor_kl']:.4f}{fkl_str} ent={upd['entropy']:.3f} "
                f"vl={upd['val_loss']:.3f}{aux_str}{warm_str}{eval_str}",
                flush=True,
            )

            payload = {
                "model_config": cpu_policy.config,
                "model_state_dict": cpu_sd,
                "rl_iteration": it,
                "rl_mean_reward": mean_rew,
                "rl_kl": upd["kl"],
                "rl_anchor_kl": upd["anchor_kl"],
                "best_eval_score": best_eval_score,
            }
            torch.save(payload, latest_path)
            if save_best:
                torch.save(payload, out_path)
            if args.save_every > 0 and it % args.save_every == 0:
                snap_out = out_path.with_name(f"{out_path.stem}_it{it}{out_path.suffix}")
                torch.save(payload, snap_out)
    finally:
        if pool_exec is not None:
            pool_exec.shutdown(wait=True)
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.cleanup()

    if not out_path.is_file() and latest_path.is_file():
        torch.save(torch.load(latest_path, map_location="cpu", weights_only=False), out_path)

    print(f"[RL-League-PPO] Saved best -> {out_path} ({out_path.stat().st_size/1024:.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
