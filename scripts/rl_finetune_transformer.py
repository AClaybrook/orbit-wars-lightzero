#!/usr/bin/env python3
"""Self-Play League PPO + Delayed Teacher KL for ``PlanetTimelineTransformer``.

Upgraded Gen 5 RL trainer incorporating key lessons from Kaggle Orbit Wars top-10
solutions (#5 TonyK 1754.6 & #7 Audun 1717.3):

- **Historical Opponent Pool + External Diversity:** Plays against frozen past
  generations (e.g. ``v1``, ``v2``, ``v2_rl``), newly frozen training snapshots,
  ``Heuristic v2``, and ``subham_v14`` — avoiding live self-play overfitting.
- **Delayed Moving Teacher KL (EMA):** Regularizes per-planet policy logits toward
  a slow EMA teacher (updated only on positive-reward iterations) so the policy can
  climb steadily past its starting anchor without catastrophic forgetting.
- **Zero-Planet Early Reset:** Ends training episodes cleanly as soon as a player
  loses their last planet (or only 1 player holds planets), eliminating doomed-fleet
  garbage frames and accelerating rollout collection ~25%.
- **Parallel CPU Rollouts + Frozen-Anchor Gate:** Collects episodes across worker
  processes and tracks best checkpoint against the frozen starting anchor.

Example:
    python scripts/rl_finetune_transformer.py \
        --init checkpoints/planet_transformer_v2_rl.pt \
        --pool checkpoints/planet_transformer_v1_backup.pt checkpoints/planet_transformer_v2.pt \
        --out checkpoints/planet_transformer_v3_rl.pt \
        --iterations 6 --episodes-per-iter 12 --workers 2
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

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

# Cached external opponent callable per process (for subham_v14 league diversity)
_SUBHAM_AGENT: Any = None


def _get_subham_agent():
    global _SUBHAM_AGENT
    if _SUBHAM_AGENT is None:
        from evaluate import resolve_agent

        _SUBHAM_AGENT = resolve_agent("subham_v14")
    return _SUBHAM_AGENT


def _decode_sampled_actions(
    enc: EncodedPlanetState,
    action_cls: np.ndarray,
    fracs: np.ndarray,
    *,
    max_speed: float = DEFAULT_SHIP_SPEED,
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
        send_ships = avail if (forced or f >= 0.88) else max(1, min(avail, int(round(avail * f))))

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
    rng: np.random.Generator | None = None,
) -> Tuple[List[List[float]], np.ndarray, np.ndarray, float]:
    """Run one forward pass and return ``(moves, action_cls[MAX_PLANETS], old_logp[MAX_PLANETS], value)``."""
    n = len(enc.planet_ids)
    action_cls = np.zeros((MAX_PLANETS,), dtype=np.int64)
    old_logp = np.zeros((MAX_PLANETS,), dtype=np.float32)
    if n == 0 or not np.any(enc.owned_mask):
        return [], action_cls, old_logp, 0.0

    pf = torch.from_numpy(enc.planet_feats[None])
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

    moves = _decode_sampled_actions(enc, action_cls, fracs_np)
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
    opp_model: Optional[PlanetTimelineTransformer] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run one on-policy 2P or 4P episode in ``FastOrbitSim`` and compute GAE returns."""
    rng = np.random.default_rng(seed)
    sim = FastOrbitSim.from_seed(seed=seed, num_agents=num_players, max_steps=max_steps)
    hero_slot = int(rng.integers(0, num_players))
    active_opp_model = opp_model if opp_model is not None else anchor

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

    last_alive = [0] * num_players
    prev_phi = float(evaluate_state(sim, hero_slot))
    early_winner: Optional[int] = None

    while not sim.done:
        actions: List[List[List[float]]] = [[] for _ in range(num_players)]
        recorded_hero = False

        for pid in range(num_players):
            if pid == hero_slot:
                obs_h = sim.to_observation(player=hero_slot)
                enc_h = extract_planet_features(obs_h, player=hero_slot, max_speed=sim.ship_speed)
                if np.any(enc_h.owned_mask):
                    moves_h, act_cls, old_lp, val_h = _policy_step(
                        policy, enc_h, sample=True, temperature=temperature, rng=rng
                    )
                    actions[pid] = moves_h
                    padded = pad_encoded_state(enc_h, MAX_PLANETS)
                    traj_pf.append(padded["planet_feats"])
                    traj_tf.append(padded["timeline_feats"])
                    traj_ef.append(padded["edge_feats"])
                    traj_le.append(padded["legal_edges"])
                    traj_om.append(padded["owned_mask"])
                    traj_pm.append(padded["pad_mask"])
                    traj_act.append(act_cls)
                    traj_logp.append(old_lp)
                    traj_vals.append(val_h)
                    recorded_hero = True
            else:
                if opponent_kind == "heuristic":
                    actions[pid] = heuristic_moves(sim, pid)
                elif opponent_kind == "subham_v14":
                    sub_fn = _get_subham_agent()
                    obs_o = sim.to_observation(player=pid)
                    try:
                        actions[pid] = sub_fn(obs_o, sim.to_config()) or []
                    except Exception:
                        actions[pid] = heuristic_moves(sim, pid)
                elif opponent_kind == "self":
                    obs_o = sim.to_observation(player=pid)
                    enc_o = extract_planet_features(obs_o, player=pid, max_speed=sim.ship_speed)
                    mv_o, _, _, _ = _policy_step(policy, enc_o, sample=True, temperature=0.55, rng=rng)
                    actions[pid] = mv_o
                else:  # frozen anchor or historical pool snapshot
                    obs_o = sim.to_observation(player=pid)
                    enc_o = extract_planet_features(obs_o, player=pid, max_speed=sim.ship_speed)
                    mv_o, _, _, _ = _policy_step(active_opp_model, enc_o, sample=False, rng=rng)
                    actions[pid] = mv_o

        sim.step(actions)
        totals = sim.scores()
        for pid in range(num_players):
            if totals[pid] > 0:
                last_alive[pid] = sim.step_count

        if recorded_hero:
            cur_phi = float(evaluate_state(sim, hero_slot))
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
            elif p_counts[hero_slot] == 0 and totals[hero_slot] == 0:
                # Hero completely eliminated in 4P FFA — no further decisions possible
                break

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
        term_rew = 1.0 if win_pid == hero_slot else (0.0 if win_pid < 0 else -1.0)
        hero_rank = 1.0 if win_pid == hero_slot else (1.5 if win_pid < 0 else 2.0)
    else:
        order = sorted(range(num_players), key=lambda i: (totals[i], last_alive[i]), reverse=True)
        grad = [1.0, 0.25, -0.35, -1.0]
        rank_idx = order.index(hero_slot)
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
        }
        return empty, {"win": float(win_pid == hero_slot), "rank": hero_rank, "steps": float(sim.step_count), "term_rew": term_rew}

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
    }
    stats = {
        "win": 1.0 if win_pid == hero_slot else 0.0,
        "rank": hero_rank,
        "steps": float(sim.step_count),
        "term_rew": float(term_rew),
        "num_players": float(num_players),
    }
    return batch, stats


def _worker_collect_episode(task: Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], int, str, int, float, int, float, bool]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Multiprocessing worker entry point for parallel episode collection."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    model_cfg, policy_sd, opp_sd, num_players, opponent_kind, seed, temperature, max_steps, shaping_coef, early_reset = task
    policy = PlanetTimelineTransformer(**model_cfg)
    policy.load_state_dict(policy_sd)
    policy.eval()

    opp_model: Optional[PlanetTimelineTransformer] = None
    if opp_sd is not None:
        opp_model = PlanetTimelineTransformer(**model_cfg)
        opp_model.load_state_dict(opp_sd)
        opp_model.eval()

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
        opp_model=opp_model,
    )


def _eval_vs_fixed_anchor(
    policy: PlanetTimelineTransformer,
    fixed_anchor: PlanetTimelineTransformer,
    *,
    seed_base: int = 12000,
    games_2p: int = 6,
    games_4p: int = 4,
    max_steps: int = 240,
) -> Dict[str, float]:
    """Deterministic greedy evaluation against the frozen starting anchor (2P + 4P)."""
    policy.eval()
    fixed_anchor.eval()
    w2 = 0
    for g in range(games_2p):
        sim = FastOrbitSim.from_seed(seed=seed_base + g, num_agents=2, max_steps=max_steps)
        hero = g % 2
        while not sim.done:
            acts = []
            for pid in range(2):
                obs = sim.to_observation(player=pid)
                enc = extract_planet_features(obs, player=pid, max_speed=sim.ship_speed)
                m = policy if pid == hero else fixed_anchor
                mv, _, _, _ = _policy_step(m, enc, sample=False)
                acts.append(mv)
            sim.step(acts)
        if sim.winner() == hero:
            w2 += 1

    w4 = 0
    r4_sum = 0.0
    for g in range(games_4p):
        sim = FastOrbitSim.from_seed(seed=seed_base + 500 + g, num_agents=4, max_steps=max_steps)
        hero = g % 4
        last_alive = [0] * 4
        while not sim.done:
            acts = []
            for pid in range(4):
                obs = sim.to_observation(player=pid)
                enc = extract_planet_features(obs, player=pid, max_speed=sim.ship_speed)
                m = policy if pid == hero else fixed_anchor
                mv, _, _, _ = _policy_step(m, enc, sample=False)
                acts.append(mv)
            sim.step(acts)
            sc = sim.scores()
            for pid in range(4):
                if sc[pid] > 0:
                    last_alive[pid] = sim.step_count
        sc = sim.scores()
        order = sorted(range(4), key=lambda i: (sc[i], last_alive[i]), reverse=True)
        rk = order.index(hero) + 1
        r4_sum += rk
        if rk == 1:
            w4 += 1

    wr2 = w2 / max(1, games_2p)
    wr4 = w4 / max(1, games_4p)
    return {
        "eval_2p_wr": wr2,
        "eval_4p_wr": wr4,
        "eval_4p_rank": r4_sum / max(1, games_4p),
        "eval_score": 0.55 * wr2 + 0.45 * wr4,
    }


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
) -> Dict[str, float]:
    """Perform clipped per-planet PPO + Delayed-Teacher KL updates on collected rollout transitions."""
    n = len(rollout["returns"])
    if n == 0:
        return {"loss": 0.0, "kl": 0.0, "entropy": 0.0, "val_loss": 0.0}

    adv = rollout["advantages"].copy()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    policy.train()
    anchor.eval()

    losses: List[float] = []
    kls: List[float] = []
    ents: List[float] = []
    vlosses: List[float] = []

    for _ in range(ppo_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pf_np = rollout["planet_feats"][idx]
            k_rot = random.randrange(4)
            flip_x = bool(random.randrange(2))
            pf = torch.from_numpy(apply_d4_planet_feats(pf_np, k_rot=k_rot, flip_x=flip_x))
            tf = torch.from_numpy(rollout["timeline_feats"][idx])
            ef = torch.from_numpy(rollout["edge_feats"][idx])
            le = torch.from_numpy(rollout["legal_edges"][idx])
            om = torch.from_numpy(rollout["owned_mask"][idx])
            pm = torch.from_numpy(rollout["pad_mask"][idx])
            act = torch.from_numpy(rollout["action_cls"][idx])
            old_lp = torch.from_numpy(rollout["old_logp"][idx])
            adv_t = torch.from_numpy(adv[idx])
            ret_t = torch.from_numpy(rollout["returns"][idx])

            logits, _, val = policy(pf, tf, ef, pm)
            with torch.no_grad():
                ref_logits, _, _ = anchor(pf, tf, ef, pm)

            B, N, _ = logits.shape
            hold_ok = torch.ones((B, N, 1), dtype=torch.bool)
            act_mask = torch.cat([hold_ok, le], dim=-1)
            logits = logits.masked_fill(~act_mask, -1e9)
            ref_logits = ref_logits.masked_fill(~act_mask, -1e9)

            valid = om & le.any(dim=-1)  # (B, N)
            if valid.any():
                logp_all = F.log_softmax(logits, dim=-1)
                ref_logp_all = F.log_softmax(ref_logits, dim=-1)
                probs_all = logp_all.exp()

                flat_logp = logp_all[valid]
                flat_ref_logp = ref_logp_all[valid]
                flat_probs = probs_all[valid]
                flat_act = act[valid]
                flat_old_lp = old_lp[valid]
                flat_adv = adv_t.unsqueeze(1).expand(B, N)[valid]

                new_lp = flat_logp.gather(1, flat_act.unsqueeze(1)).squeeze(1)
                ratio = torch.exp((new_lp - flat_old_lp).clamp(-4.0, 4.0))
                surr1 = ratio * flat_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * flat_adv
                pol_loss = -torch.min(surr1, surr2).mean()

                kl_per = (flat_probs * (flat_logp - flat_ref_logp)).sum(dim=-1).clamp_min(0.0)
                kl_mean = kl_per.mean()
                ent_mean = -(flat_probs * flat_logp).sum(dim=-1).mean()
            else:
                pol_loss = logits.sum() * 0.0
                kl_mean = logits.sum() * 0.0
                ent_mean = logits.sum() * 0.0

            v_loss = F.mse_loss(val, ret_t)
            total_loss = pol_loss + value_coef * v_loss + kl_coef * kl_mean - ent_coef * ent_mean

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.7)
            optimizer.step()

            losses.append(float(total_loss.item()))
            kls.append(float(kl_mean.item()))
            ents.append(float(ent_mean.item()))
            vlosses.append(float(v_loss.item()))

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "entropy": float(np.mean(ents)) if ents else 0.0,
        "val_loss": float(np.mean(vlosses)) if vlosses else 0.0,
    }


def _ema_update_teacher(teacher: PlanetTimelineTransformer, policy: PlanetTimelineTransformer, tau: float) -> None:
    if tau <= 0.0:
        return
    with torch.no_grad():
        for tp, pp in zip(teacher.parameters(), policy.parameters()):
            tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-Play League PPO + Delayed Teacher KL for PlanetTimelineTransformer.")
    parser.add_argument("--init", required=True, help="Starting checkpoint path.")
    parser.add_argument("--pool", nargs="*", default=[], help="Additional historical opponent checkpoints.")
    parser.add_argument("--out", default="checkpoints/planet_transformer_v3_rl.pt")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--episodes-per-iter", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=float, default=0.003, help="Entropy bonus coefficient (keep low during RL sharpening).")
    parser.add_argument("--teacher-ema", type=float, default=0.12, help="EMA rate to update KL teacher on positive iters.")
    parser.add_argument("--shaping-coef", type=float, default=0.02, help="Near-sparse shaping weight (0=pure terminal).")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--eval-every", type=int, default=2, help="Run deterministic anchor eval every N iters.")
    parser.add_argument("--seed", type=int, default=9500)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    policy = PlanetTimelineTransformer(**cfg)
    policy.load_state_dict(ckpt["model_state_dict"])

    fixed_anchor = copy.deepcopy(policy)
    fixed_anchor.eval()
    teacher = copy.deepcopy(policy)
    teacher.eval()
    for m in (fixed_anchor, teacher):
        for p in m.parameters():
            p.requires_grad_(False)

    # Build historical opponent state-dict pool (init + any --pool checkpoints)
    pool_sds: List[Tuple[str, Dict[str, Any]]] = [("init", {k: v.cpu().clone() for k, v in policy.state_dict().items()})]
    for p_path in args.pool:
        pp = Path(p_path)
        if pp.is_file():
            p_ckpt = torch.load(pp, map_location="cpu", weights_only=False)
            pool_sds.append((pp.stem, p_ckpt["model_state_dict"]))

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    out_path = Path(args.out)
    latest_path = out_path.with_name(out_path.stem + "_latest" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[RL-League-PPO] Loaded {args.init} | pool={[n for n, _ in pool_sds]} | "
        f"iters={args.iterations} eps/iter={args.episodes_per_iter} workers={args.workers} "
        f"lr={args.lr} kl={args.kl_coef} ema={args.teacher_ema} shaping={args.shaping_coef}",
        flush=True,
    )
    t0 = time.time()
    best_eval_score = -1.0

    # Diverse league schedule: frozen historical pool (2P & 4P), subham_v14 (2P & 4P), Heuristic v2
    league_cycle = [
        (2, "pool"),
        (2, "pool"),
        (4, "pool"),
        (2, "pool"),
        (4, "subham_v14"),
        (2, "subham_v14"),
        (4, "pool"),
        (2, "pool"),
        (4, "heuristic"),
        (2, "pool"),
        (4, "subham_v14"),
        (2, "heuristic"),
    ]

    for it in range(1, args.iterations + 1):
        policy.eval()
        pol_sd = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        tasks = []
        for e in range(args.episodes_per_iter):
            n_pl, opp_kind = league_cycle[(it * args.episodes_per_iter + e) % len(league_cycle)]
            ep_seed = args.seed + it * 1000 + e
            opp_sd = None
            if opp_kind == "pool":
                # 50% latest/init anchor, 50% random historical pool snapshot
                if e % 2 == 0 or len(pool_sds) == 1:
                    opp_sd = pool_sds[-1][1]
                else:
                    opp_sd = pool_sds[random.randrange(len(pool_sds))][1]
            tasks.append((
                cfg,
                pol_sd,
                opp_sd,
                n_pl,
                opp_kind,
                ep_seed,
                args.temperature,
                args.max_steps,
                args.shaping_coef,
                True,
            ))

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_worker_collect_episode, tasks))
        else:
            results = [_worker_collect_episode(t) for t in tasks]

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

        keys = list(batches[0].keys())
        rollout = {k: np.concatenate([b[k] for b in batches], axis=0) for k in keys}
        upd = ppo_update(
            policy,
            teacher,
            opt,
            rollout,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            kl_coef=args.kl_coef,
            ent_coef=args.ent_coef,
        )

        wr2 = 100.0 * float(np.mean(ep_wins_2p)) if ep_wins_2p else 0.0
        wr4 = 100.0 * float(np.mean(ep_wins_4p)) if ep_wins_4p else 0.0
        rk4 = float(np.mean(ep_ranks_4p)) if ep_ranks_4p else 0.0
        mean_rew = float(np.mean(ep_rews)) if ep_rews else 0.0

        # Delayed moving teacher EMA update on positive-reward iterations
        if mean_rew > 0.05:
            _ema_update_teacher(teacher, policy, args.teacher_ema)

        # Periodically add a frozen snapshot of our improving policy into the historical pool
        if it % 2 == 0:
            pool_sds.append((f"snap_it{it}", {k: v.cpu().clone() for k, v in policy.state_dict().items()}))
            if len(pool_sds) > 6:
                # Keep init + most recent 5 snapshots
                pool_sds = [pool_sds[0]] + pool_sds[-5:]

        eval_str = ""
        save_best = False
        if it % args.eval_every == 0 or it == args.iterations:
            ev = _eval_vs_fixed_anchor(policy, fixed_anchor, seed_base=args.seed + 50000, games_2p=4, games_4p=4)
            eval_str = f" | EVAL_vs_init: 2P={100*ev['eval_2p_wr']:.0f}% 4P={100*ev['eval_4p_wr']:.0f}% (rk={ev['eval_4p_rank']:.2f})"
            if ev["eval_score"] >= best_eval_score:
                best_eval_score = ev["eval_score"]
                save_best = True
        elif best_eval_score < 0.0:
            save_best = True

        print(
            f"  iter {it:2d}/{args.iterations} ({time.time()-t0:5.1f}s) | "
            f"states={len(rollout['returns']):4d} | 2P={wr2:5.1f}% 4P={wr4:5.1f}% (rk={rk4:.2f}) | "
            f"rew={mean_rew:+.2f} kl={upd['kl']:.4f} ent={upd['entropy']:.3f} vl={upd['val_loss']:.3f}{eval_str}",
            flush=True,
        )

        payload = {
            "model_config": policy.config,
            "model_state_dict": policy.state_dict(),
            "rl_iteration": it,
            "rl_mean_reward": mean_rew,
            "rl_kl": upd["kl"],
            "best_eval_score": best_eval_score,
        }
        torch.save(payload, latest_path)
        if save_best:
            torch.save(payload, out_path)

    print(f"[RL-League-PPO] Saved best -> {out_path} ({out_path.stat().st_size/1024:.1f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
