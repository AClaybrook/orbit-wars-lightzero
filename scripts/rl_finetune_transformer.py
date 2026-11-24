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
    build_optimizer,
    cpu_state_dict,
    one_hot_no_scatter,
    resolve_train_device,
)
from orbit_wars_bot.fast_sim import FastOrbitSim  # noqa: E402
from orbit_wars_bot.features import evaluate_state  # noqa: E402
from orbit_wars_bot.heuristic_bot import heuristic_moves  # noqa: E402
from orbit_wars_bot.physics import DEFAULT_SHIP_SPEED, solve_intercept  # noqa: E402
from orbit_wars_bot.planet_transformer import (  # noqa: E402
    MAX_PLANETS,
    MAX_ROUTE_ETA,
    EncodedPlanetState,
    PlanetTimelineTransformer,
    apply_d4_planet_feats,
    extract_planet_features,
    pad_encoded_state,
)

# Cached external opponent callable and model caches per worker process
_SUBHAM_AGENT: Any = None
_WORKER_POLICY_CACHE: Dict[str, Any] = {}
_WORKER_OPP_CACHE: Dict[str, PlanetTimelineTransformer] = {}


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
    pf = torch.from_numpy(pf_np[None])
    tf = torch.from_numpy(enc.timeline_feats[None])
    ef = torch.from_numpy(enc.edge_feats[None])
    pm = torch.ones((1, n), dtype=torch.bool)

    logits, fracs, val = model(pf, tf, ef, pm)
    logits = logits[0]  # (n, n+1)
    fracs_np = fracs[0].cpu().numpy()  # (n, n)
    v_scalar = float(val[0].item())

    legal = torch.from_numpy(enc.legal_edges)
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

        for pid in range(num_players):
            if pid == hero_slot_val:
                obs_h = sim.to_observation(player=hero_slot_val)
                enc_h = extract_planet_features(obs_h, player=hero_slot_val, max_speed=sim.ship_speed)
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
                    if k_rot != 0 or flip_x:
                        pf_store = apply_d4_planet_feats(pf_store, k_rot=k_rot, flip_x=flip_x)
                    traj_pf.append(pf_store)
                    traj_tf.append(padded["timeline_feats"])
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
                elif opponent_kind == "subham_v14":
                    try:
                        sub_fn = _get_subham_agent()
                        obs_o = sim.to_observation(player=pid)
                        actions[pid] = sub_fn(obs_o, sim.to_config()) or []
                    except Exception:
                        actions[pid] = heuristic_moves(sim, pid)
                elif opponent_kind == "self":
                    obs_o = sim.to_observation(player=pid)
                    enc_o = extract_planet_features(obs_o, player=pid, max_speed=sim.ship_speed)
                    mv_o, _, _, _ = _policy_step(
                        policy, enc_o, sample=True, temperature=0.55, force_all_in=force_all_in, rng=rng
                    )
                    actions[pid] = mv_o
                else:  # frozen anchor or historical pool snapshot
                    obs_o = sim.to_observation(player=pid)
                    enc_o = extract_planet_features(obs_o, player=pid, max_speed=sim.ship_speed)
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
            "planet_feats": np.zeros((0, MAX_PLANETS, 22), np.float32),
            "timeline_feats": np.zeros((0, MAX_PLANETS, 20, 8), np.float32),
            "edge_feats": np.zeros((0, MAX_PLANETS, MAX_PLANETS, 12), np.float32),
            "legal_edges": np.zeros((0, MAX_PLANETS, MAX_PLANETS), bool),
            "owned_mask": np.zeros((0, MAX_PLANETS), bool),
            "pad_mask": np.zeros((0, MAX_PLANETS), bool),
            "action_cls": np.zeros((0, MAX_PLANETS), np.int64),
            "old_logp": np.zeros((0, MAX_PLANETS), np.float32),
            "advantages": np.zeros((0,), np.float32),
            "returns": np.zeros((0,), np.float32),
            "step_ticks": np.zeros((0,), np.int32),
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
        "planet_feats": np.stack(traj_pf, axis=0),
        "timeline_feats": np.stack(traj_tf, axis=0),
        "edge_feats": np.stack(traj_ef, axis=0),
        "legal_edges": np.stack(traj_le, axis=0),
        "owned_mask": np.stack(traj_om, axis=0),
        "pad_mask": np.stack(traj_pm, axis=0),
        "action_cls": np.stack(traj_act, axis=0),
        "old_logp": np.stack(traj_logp, axis=0),
        "advantages": advs,
        "returns": rets,
        "step_ticks": np.asarray(traj_steps, dtype=np.int32),
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
    opp_model: PlanetTimelineTransformer,
    *,
    num_players: int,
    seed: int,
    hero_slot: int,
    max_steps: int,
    force_all_in: bool,
) -> Tuple[float, float]:
    sim = FastOrbitSim.from_seed(seed=seed, num_agents=num_players, max_steps=max_steps)
    last_alive = [0] * num_players
    while not sim.done:
        acts = []
        for pid in range(num_players):
            obs = sim.to_observation(player=pid)
            enc = extract_planet_features(obs, player=pid, max_speed=sim.ship_speed)
            m = policy if pid == hero_slot else opp_model
            mv, _, _, _ = _policy_step(m, enc, sample=False, force_all_in=force_all_in)
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
    task: Tuple[str, int, str, str, int, int, int, int, bool]
) -> Tuple[str, int, float, float]:
    _configure_worker_threads()
    policy_path, policy_version, opp_path, opp_tag, num_players, seed, hero_slot, max_steps, force_all_in = task
    policy = _get_worker_policy(policy_path, policy_version)
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
    eval_anchors: Sequence[Tuple[str, str, PlanetTimelineTransformer]],
    pool_exec: Optional[ProcessPoolExecutor],
    seed_base: int,
    eval_pairs: int,
    two_player_only: bool,
    max_steps: int,
    force_all_in: bool,
) -> Dict[str, Any]:
    """Run paired-seed deterministic 2P (and optional 4P) validation across anchor models."""
    results: List[Tuple[str, int, float, float]] = []
    if pool_exec is not None and policy_path is not None:
        tasks = []
        for a_idx, (tag, opp_path, _) in enumerate(eval_anchors):
            for g in range(eval_pairs):
                s = seed_base + a_idx * 100 + g
                tasks.append((policy_path, policy_version, opp_path, tag, 2, s, 0, max_steps, force_all_in))
                tasks.append((policy_path, policy_version, opp_path, tag, 2, s, 1, max_steps, force_all_in))
        if not two_player_only and eval_anchors:
            tag0, opp_path0, _ = eval_anchors[0]
            for g in range(4):
                tasks.append((policy_path, policy_version, opp_path0, tag0, 4, seed_base + 500 + g, g % 4, max_steps, force_all_in))
        results = list(pool_exec.map(_worker_eval_game, tasks))
    else:
        cpu_policy.eval()
        for a_idx, (tag, _, opp_m) in enumerate(eval_anchors):
            opp_m.eval()
            for g in range(eval_pairs):
                s = seed_base + a_idx * 100 + g
                for seat in (0, 1):
                    w, rk = _play_deterministic_eval_game(
                        cpu_policy, opp_m, num_players=2, seed=s, hero_slot=seat, max_steps=max_steps, force_all_in=force_all_in
                    )
                    results.append((tag, 2, w, rk))
        if not two_player_only and eval_anchors:
            tag0, _, opp_m0 = eval_anchors[0]
            for g in range(4):
                w, rk = _play_deterministic_eval_game(
                    cpu_policy, opp_m0, num_players=4, seed=seed_base + 500 + g, hero_slot=g % 4, max_steps=max_steps, force_all_in=force_all_in
                )
                results.append((tag0, 4, w, rk))

    per_tag_2p: Dict[str, List[float]] = {tag: [] for tag, _, _ in eval_anchors}
    wr4_list: List[float] = []
    rk4_list: List[float] = []
    for tag, n_pl, won, rk in results:
        if n_pl == 2:
            per_tag_2p[tag].append(won)
        else:
            wr4_list.append(won)
            rk4_list.append(rk)

    per_tag_wr = {tag: (float(np.mean(vals)) if vals else 0.0) for tag, vals in per_tag_2p.items()}
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
) -> Dict[str, float]:
    """Perform clipped per-planet PPO + Dual-KL + Aux world-model updates."""
    n = len(rollout["returns"])
    if n == 0:
        return {
            "loss": 0.0,
            "kl": 0.0,
            "anchor_kl": 0.0,
            "entropy": 0.0,
            "val_loss": 0.0,
            "aux_loss": 0.0,
            "aux_acc": 0.0,
        }

    adv = rollout["advantages"].copy()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    step_ticks = rollout.get("step_ticks")
    if step_ticks is not None and midgame_adv_weight != 1.0:
        mid_mask = (step_ticks >= 25) & (step_ticks <= 75)
        w = np.where(mid_mask, float(midgame_adv_weight), 1.0).astype(np.float32)
        w /= max(1e-6, float(w.mean()))
        adv = adv * w

    train_dev = device if device is not None else next(policy.parameters()).device
    is_dml_dev = bool(use_dml or (train_dev.type == "privateuseone"))
    has_aux_data = ("aux_owner" in rollout) and ("aux_ships" in rollout) and ("aux_valid" in rollout)
    want_aux = bool(aux_coef > 0.0 and getattr(policy, "aux_future_head", None) is not None and has_aux_data)

    policy.train()
    anchor.eval()
    if fixed_anchor is not None:
        fixed_anchor.eval()

    loss_acc = torch.zeros((), device=train_dev)
    kl_acc = torch.zeros((), device=train_dev)
    akl_acc = torch.zeros((), device=train_dev)
    ent_acc = torch.zeros((), device=train_dev)
    vl_acc = torch.zeros((), device=train_dev)
    aux_loss_acc = torch.zeros((), device=train_dev)
    aux_hits_acc = torch.zeros((), device=train_dev)
    aux_tot_acc = torch.zeros((), device=train_dev)
    n_steps = 0

    for _ in range(ppo_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pm_full = rollout["pad_mask"][idx]
            max_n = max(2, int(pm_full.sum(axis=1).max()))

            pf_np = rollout["planet_feats"][idx, :max_n]
            if augment_d4:
                k_rot = random.randrange(4)
                flip_x = bool(random.randrange(2))
                pf_np = apply_d4_planet_feats(pf_np, k_rot=k_rot, flip_x=flip_x)

            le_np = np.ascontiguousarray(rollout["legal_edges"][idx, :max_n, :max_n])
            om_np = np.ascontiguousarray(rollout["owned_mask"][idx, :max_n])
            has_valid = bool(np.any(om_np & le_np.any(axis=-1)))

            aux_v_np = np.ascontiguousarray(rollout["aux_valid"][idx, :max_n]) if want_aux else None
            has_aux_valid = bool(aux_v_np is not None and np.any(aux_v_np))

            pf = torch.from_numpy(np.ascontiguousarray(pf_np)).to(train_dev)
            tf = torch.from_numpy(np.ascontiguousarray(rollout["timeline_feats"][idx, :max_n])).to(train_dev)
            ef = torch.from_numpy(np.ascontiguousarray(rollout["edge_feats"][idx, :max_n, :max_n])).to(train_dev)
            le = torch.from_numpy(le_np).to(train_dev)
            om = torch.from_numpy(om_np).to(train_dev)
            pm = torch.from_numpy(np.ascontiguousarray(pm_full[:, :max_n])).to(train_dev)
            act = torch.from_numpy(np.ascontiguousarray(rollout["action_cls"][idx, :max_n])).to(train_dev)
            old_lp = torch.from_numpy(np.ascontiguousarray(rollout["old_logp"][idx, :max_n])).to(train_dev)
            adv_t = torch.from_numpy(adv[idx]).to(train_dev)
            ret_t = torch.from_numpy(rollout["returns"][idx]).to(train_dev)
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

            if want_aux:
                logits, _, val, aux_pred = policy(pf, tf, ef, pm, return_aux=True)
            else:
                logits, _, val = policy(pf, tf, ef, pm)
                aux_pred = None

            with torch.no_grad():
                ref_logits, _, _ = anchor(pf, tf, ef, pm)
                if fixed_anchor is not None and anchor_kl_coef > 0.0:
                    fixed_logits, _, _ = fixed_anchor(pf, tf, ef, pm)
                else:
                    fixed_logits = None

            B, N, _ = logits.shape
            hold_ok = torch.ones((B, N, 1), dtype=torch.bool, device=train_dev)
            act_mask = torch.cat([hold_ok, le], dim=-1)
            logits = logits.masked_fill(~act_mask, -1e9)
            ref_logits = ref_logits.masked_fill(~act_mask, -1e9)

            if has_valid:
                valid_f = (om & le.any(dim=-1)).float()
                denom = valid_f.sum().clamp_min(1.0)
                logp_all = F.log_softmax(logits, dim=-1)
                ref_logp_all = F.log_softmax(ref_logits, dim=-1)
                probs_all = logp_all.exp()

                act_oh = one_hot_no_scatter(act.clamp(0, N), N + 1, dtype=logp_all.dtype)
                new_lp = (logp_all * act_oh).sum(dim=-1)
                ratio = torch.exp((new_lp - old_lp).clamp(-4.0, 4.0))
                adv_bn = adv_t.unsqueeze(1).expand(B, N)
                surr1 = ratio * adv_bn
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_bn
                pol_loss = -(torch.min(surr1, surr2) * valid_f).sum() / denom

                kl_per = (probs_all * (logp_all - ref_logp_all)).sum(dim=-1).clamp_min(0.0)
                kl_mean = (kl_per * valid_f).sum() / denom
                ent_per = -(probs_all * logp_all).sum(dim=-1)
                ent_mean = (ent_per * valid_f).sum() / denom

                if fixed_logits is not None:
                    fixed_logits = fixed_logits.masked_fill(~act_mask, -1e9)
                    fixed_logp_all = F.log_softmax(fixed_logits, dim=-1)
                    akl_per = (probs_all * (logp_all - fixed_logp_all)).sum(dim=-1).clamp_min(0.0)
                    anchor_kl_mean = (akl_per * valid_f).sum() / denom
                else:
                    anchor_kl_mean = logits.sum() * 0.0
            else:
                pol_loss = logits.sum() * 0.0
                kl_mean = logits.sum() * 0.0
                anchor_kl_mean = logits.sum() * 0.0
                ent_mean = logits.sum() * 0.0

            v_loss = F.mse_loss(val, ret_t)

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

            total_loss = (
                pol_loss
                + value_coef * v_loss
                + kl_coef * kl_mean
                + anchor_kl_coef * anchor_kl_mean
                - ent_coef * ent_mean
                + aux_coef * aux_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.7, foreach=not is_dml_dev)
            optimizer.step()

            loss_acc = loss_acc + total_loss.detach()
            kl_acc = kl_acc + kl_mean.detach()
            akl_acc = akl_acc + anchor_kl_mean.detach()
            ent_acc = ent_acc + ent_mean.detach()
            vl_acc = vl_acc + v_loss.detach()
            aux_loss_acc = aux_loss_acc + aux_loss.detach()
            n_steps += 1

    denom_steps = max(1, n_steps)
    aux_tot_val = float(aux_tot_acc.item())
    return {
        "loss": float(loss_acc.item()) / denom_steps,
        "kl": float(kl_acc.item()) / denom_steps,
        "anchor_kl": float(akl_acc.item()) / denom_steps,
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
    parser.add_argument("--aux-coef", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.003, help="Entropy bonus coefficient (keep low during RL sharpening).")
    parser.add_argument("--teacher-ema", type=float, default=0.12, help="EMA rate to update KL teacher on positive iters.")
    parser.add_argument("--shaping-coef", type=float, default=0.02, help="Near-sparse shaping weight (0=pure terminal).")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--opp-temperature", type=float, default=0.0, help="Pool opponent sampling temperature (0=greedy).")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--eval-every", type=int, default=2, help="Run deterministic anchor eval every N iters.")
    parser.add_argument("--eval-pairs", type=int, default=8, help="Paired-seed games per eval anchor.")
    parser.add_argument("--eval-opponents", nargs="*", default=[], help="Extra checkpoints for multi-anchor eval gate.")
    parser.add_argument("--force-all-in", action="store_true", help="Train with 100% all-in sends matching Gen 8/9 2P inference.")
    parser.add_argument("--two-player-only", action="store_true", help="Focus 100% of rollouts and anchor eval on 2P matches.")
    parser.add_argument("--midgame-adv-weight", type=float, default=1.0, help="Policy-gradient multiplier for midgame ticks t in [25, 75].")
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
    fixed_anchor = copy.deepcopy(cpu_policy).to(train_dev).eval()
    teacher = copy.deepcopy(cpu_policy).to(train_dev).eval()
    for m in (fixed_anchor, teacher):
        for p in m.parameters():
            p.requires_grad_(False)

    opt = build_optimizer(policy.parameters(), lr=args.lr, weight_decay=1e-4, use_dml=use_dml)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.iterations), eta_min=args.lr * float(args.lr_min_ratio)
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
        if has_subham:
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
    print(
        f"[RL-League-PPO] Loaded {args.init} | dev={dev_tag} | pool={[n for n, _, _ in pinned_pool]} | "
        f"iters={args.iterations} "
        + (f"pairs={n_pairs} (2P={2*n_pairs}ep)" if use_paired else f"eps/iter={args.episodes_per_iter}")
        + f" workers={args.workers} | lr={args.lr} clip={args.clip_eps} kl={args.kl_coef} akl={args.anchor_kl_coef} "
        f"ema={args.teacher_ema} aux={args.aux_coef} shaping={args.shaping_coef} T={args.temperature} "
        f"oppT={args.opp_temperature} all_in={args.force_all_in} 2p_only={args.two_player_only} mid_adv={args.midgame_adv_weight}",
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
            p = Path(tmp_dir_ctx.name) / "policy_cur.pt"
            torch.save({"model_config": cpu_policy.config, "model_state_dict": cpu_policy.state_dict()}, p)
            return str(p)

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
        )
        best_eval_score = ev0["eval_score"]
        tag_str0 = " ".join(f"{t}={100*v:.1f}%" for t, v in ev0["per_tag_2p"].items())
        print(
            f"  [Baseline EVAL({ev0['total_games']}g)] {tag_str0} avg={100*ev0['avg_2p']:.1f}%",
            flush=True,
        )

        for it in range(1, args.iterations + 1):
            torch.set_num_threads(1)
            cpu_policy.eval()
            pol_p = _save_cur_policy(it)
            active_pool = pinned_pool + rolling_snaps

            t_roll = time.time()
            tasks: List[Tuple[Any, ...]] = []
            direct_opps: List[Optional[PlanetTimelineTransformer]] = []

            if use_paired:
                for k in range(n_pairs):
                    n_pl, opp_kind = league_cycle[(it * n_pairs + k) % len(league_cycle)]
                    if args.two_player_only:
                        n_pl = 2
                    map_seed = args.seed + it * 1000 + k
                    opp_path: Optional[str] = None
                    opp_model: Optional[PlanetTimelineTransformer] = None
                    if opp_kind == "pool":
                        u = random.random()
                        if rolling_snaps and u < 0.30:
                            _, opp_path, opp_model = rolling_snaps[-1]
                        elif u < 0.75:
                            _, opp_path, opp_model = random.choice(pinned_pool)
                        else:
                            _, opp_path, opp_model = random.choice(active_pool)

                    for seat in (0, 1):
                        if pool_exec is not None and pol_p is not None:
                            tasks.append((
                                pol_p,
                                it,
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
                            ))
                        else:
                            direct_opps.append(opp_model)
                            tasks.append((
                                n_pl,
                                opp_kind,
                                map_seed,
                                seat,
                            ))
            else:
                for e in range(args.episodes_per_iter):
                    n_pl, opp_kind = league_cycle[(it * args.episodes_per_iter + e) % len(league_cycle)]
                    ep_seed = args.seed + it * 1000 + e
                    opp_path = None
                    opp_model = None
                    if opp_kind == "pool":
                        if e % 2 == 0 or len(active_pool) == 1:
                            _, opp_path, opp_model = active_pool[-1]
                        else:
                            _, opp_path, opp_model = random.choice(active_pool)
                    if pool_exec is not None and pol_p is not None:
                        tasks.append((
                            pol_p,
                            it,
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
                        ))
                    else:
                        direct_opps.append(opp_model)
                        tasks.append((n_pl, opp_kind, ep_seed, None))

            if pool_exec is not None and pol_p is not None:
                results = list(pool_exec.map(_worker_collect_episode, tasks))
            else:
                results = []
                for idx_t, t_info in enumerate(tasks):
                    n_pl, opp_kind, ep_seed, seat = t_info
                    opp_m = direct_opps[idx_t]
                    results.append(
                        collect_episode(
                            cpu_policy,
                            opp_m if opp_m is not None else cpu_policy,
                            num_players=n_pl,
                            opponent_kind=opp_kind,
                            seed=ep_seed,
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
            roll_sec = time.time() - t_roll

            batches: List[Dict[str, np.ndarray]] = []
            ep_wins_2p: List[float] = []
            ep_wins_4p: List[float] = []
            ep_ranks_4p: List[float] = []
            ep_rews: List[float] = []

            for b, st in results:
                if len(b["returns"]) > 0:
                    batches.append(b)
                ep_rews.append(st["term_rew"])
                if int(st.get("num_players", 2)) == 2:
                    ep_wins_2p.append(st["win"])
                else:
                    ep_wins_4p.append(st["win"])
                    ep_ranks_4p.append(st["rank"])

            splits = sweeps = 0
            if use_paired:
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

            t_upd = time.time()
            torch.set_num_threads(max(1, int(args.train_threads)))
            upd = ppo_update(
                policy,
                teacher,
                opt,
                rollout,
                ppo_epochs=args.ppo_epochs,
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
            )
            sched.step()
            upd_sec = time.time() - t_upd

            cpu_sd = cpu_state_dict(policy)
            cpu_policy.load_state_dict(cpu_sd)

            wr2 = 100.0 * float(np.mean(ep_wins_2p)) if ep_wins_2p else 0.0
            wr4 = 100.0 * float(np.mean(ep_wins_4p)) if ep_wins_4p else 0.0
            rk4 = float(np.mean(ep_ranks_4p)) if ep_ranks_4p else 0.0
            mean_rew = float(np.mean(ep_rews)) if ep_rews else 0.0

            if mean_rew > 0.05:
                _ema_update_teacher(teacher, policy, args.teacher_ema)

            if it % 3 == 0:
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

            pair_str = f" (sw={sweeps}, sp={splits})" if use_paired else ""
            stats_4p = "" if args.two_player_only else f" 4P={wr4:5.1f}%(rk={rk4:.2f})"
            aux_str = f" aux={100*upd['aux_acc']:.1f}%" if args.aux_coef > 0.0 and upd["aux_acc"] > 0.0 else ""
            print(
                f"  iter {it:2d}/{args.iterations} ({time.time()-t0:5.1f}s [roll={roll_sec:.1f}s upd={upd_sec:.1f}s]) | "
                f"states={len(rollout['returns']):4d} | 2P={wr2:5.1f}%{pair_str}{stats_4p} | "
                f"rew={mean_rew:+.2f} kl={upd['kl']:.4f} akl={upd['anchor_kl']:.4f} ent={upd['entropy']:.3f} "
                f"vl={upd['val_loss']:.3f}{aux_str}{eval_str}",
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
