#!/usr/bin/env python3
"""Warm-start LightZero AlphaZeroModel via self-play / stance-rollout games on FastOrbitSim.

Generates fast 2P and 4P FFA games using ``FastOrbitSim``, labels spatial
``(12, 16, 16)`` observations with exact match outcomes (plus smooth rank targets
in 4P FFA) and stance-policy targets, and trains an official LightZero
``AlphaZeroModel`` saved in DI-engine checkpoint format (``{"model": state_dict}``).

The resulting checkpoint can be:
1. Loaded directly by ``OnlineMCTSAgent(checkpoint_path=...)`` and ``submission.py``.
2. Passed to ``scripts/train_lightzero.py --algo alphazero --model-path ...`` for
   further on-policy AlphaZero RL fine-tuning.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_bot.fast_sim import FastOrbitSim  # noqa: E402
from orbit_wars_bot.features import OBS_SHAPE, encode_observation, evaluate_state  # noqa: E402
from orbit_wars_bot.heuristic_bot import heuristic_moves  # noqa: E402
from orbit_wars_bot.lightzero_compat import setup_lightzero  # noqa: E402
from orbit_wars_bot.macro_actions import (  # noqa: E402
    NUM_MACRO_ACTIONS,
    MacroAction,
    build_tactical_context,
    execute_macro_action,
    get_legal_action_mask,
)

_EXPLORE_STANCES: Tuple[int, ...] = (
    int(MacroAction.BALANCED_MACRO),
    int(MacroAction.EXPAND_BEST_ROI_MULTI),
    int(MacroAction.EXPAND_HIGH_PROD),
    int(MacroAction.ATTACK_WEAKEST_ENEMY),
    int(MacroAction.ATTACK_HIGH_PROD_ENEMY),
    int(MacroAction.FEED_FRONTLINE),
    int(MacroAction.AGGRESSIVE_SKIRMISH),
)


def _rank_outcome(scores: List[int], player: int) -> float:
    """Map final ship totals to a value target in [-1.0, 1.0] for 2P or 4P FFA."""
    n = len(scores)
    my_s = scores[player]
    better = sum(1 for s in scores if s > my_s)
    tied = sum(1 for s in scores if s == my_s)
    rank = better + 1.0 + 0.5 * (tied - 1)  # 1.0 (best) .. n (worst)
    if n <= 2:
        if rank < 1.5:
            return 1.0
        if rank > 1.5:
            return -1.0
        return 0.0
    # 4P FFA smooth value: 1st -> +1.0, 2nd -> +0.25, 3rd -> -0.45, 4th -> -1.0
    return float(1.0 - 2.0 * (rank - 1.0) / max(1.0, float(n - 1)))


def generate_dataset(
    num_games_2p: int,
    num_games_4p: int,
    *,
    seed: int = 1000,
    sample_stride: int = 6,
    max_steps: int = 360,
    explore_prob: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 2P and 4P games and return ``(obs_arr, value_targets, policy_targets)``."""
    rng = random.Random(seed)
    obs_list: List[np.ndarray] = []
    val_list: List[float] = []
    pol_list: List[np.ndarray] = []

    schedule = [(2, seed + i) for i in range(num_games_2p)] + [
        (4, seed + 10000 + i) for i in range(num_games_4p)
    ]
    rng.shuffle(schedule)

    for g_idx, (n_players, g_seed) in enumerate(schedule):
        sim = FastOrbitSim.from_seed(g_seed, num_agents=n_players, max_steps=max_steps)
        # Assign each player a slight persistent personality mix + per-turn stance exploration
        player_pref = [rng.choice(_EXPLORE_STANCES) for _ in range(n_players)]
        episode_records: List[Tuple[int, np.ndarray, int, float]] = []

        while not sim.done:
            step = sim.step_count
            sample_now = (step % sample_stride == 0)
            actions = []
            for p in range(n_players):
                # Fast skip for eliminated players
                if not any(pl[1] == p for pl in sim.planets):
                    actions.append([])
                    continue
                ctx = build_tactical_context(sim, p, horizon=18)
                mask = get_legal_action_mask(sim, p, ctx=ctx)
                if rng.random() < explore_prob:
                    cand = player_pref[p] if rng.random() < 0.5 else rng.choice(_EXPLORE_STANCES)
                    act_id = cand if mask[cand] else int(MacroAction.BALANCED_MACRO)
                else:
                    act_id = int(MacroAction.BALANCED_MACRO)

                if act_id == int(MacroAction.BALANCED_MACRO):
                    mv = heuristic_moves(sim, p, ctx=ctx)
                else:
                    mv = execute_macro_action(sim, p, act_id, ctx=ctx)
                actions.append(mv)

                if sample_now:
                    obs_t = encode_observation(sim, player=p)
                    h_v = evaluate_state(sim, player=p)
                    episode_records.append((p, obs_t, act_id, h_v))

            sim.step(actions)

        final_scores = sim.scores()
        for p, obs_t, act_id, h_v in episode_records:
            z = _rank_outcome(final_scores, p)
            # Blend terminal outcome with heuristic anchor to reduce small-sample variance
            target_v = 0.80 * z + 0.20 * h_v
            # Policy target: reinforce winning/top-half stances, softly fall back to BALANCED_MACRO otherwise
            pi = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
            if z > 0.0:
                pi[act_id] = 0.85
                pi[int(MacroAction.BALANCED_MACRO)] += 0.15
            else:
                pi[int(MacroAction.BALANCED_MACRO)] = 0.70
                pi[int(MacroAction.EXPAND_BEST_ROI_MULTI)] = 0.15
                pi[int(MacroAction.FEED_FRONTLINE)] = 0.15
            obs_list.append(obs_t)
            val_list.append(float(max(-1.0, min(1.0, target_v))))
            pol_list.append(pi)

        if (g_idx + 1) % 20 == 0 or (g_idx + 1) == len(schedule):
            print(
                f"  Simulated {g_idx + 1}/{len(schedule)} games -> {len(obs_list)} samples",
                flush=True,
            )

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(val_list, dtype=np.float32),
        np.asarray(pol_list, dtype=np.float32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain LightZero AlphaZeroModel for Orbit Wars.")
    parser.add_argument("--games-2p", type=int, default=50, help="Number of 2P games to simulate.")
    parser.add_argument("--games-4p", type=int, default=30, help="Number of 4P FFA games to simulate.")
    parser.add_argument("--epochs", type=int, default=18, help="Supervised training epochs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints/ckpt_best.pth.tar",
        help="Output checkpoint path (LightZero AlphaZeroModel format).",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    print(
        f"Generating dataset: {args.games_2p} 2P games + {args.games_4p} 4P FFA games...",
        flush=True,
    )
    obs_arr, val_arr, pol_arr = generate_dataset(
        args.games_2p,
        args.games_4p,
        seed=args.seed,
    )
    n_total = len(obs_arr)
    print(f"Dataset ready: {n_total} samples in {time.perf_counter() - t0:.1f}s", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    setup_lightzero()
    from lzero.model.alphazero_model import AlphaZeroModel

    model = AlphaZeroModel(
        observation_shape=OBS_SHAPE,
        action_space_size=NUM_MACRO_ACTIONS,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        value_head_channels=8,
        policy_head_channels=8,
        value_head_hidden_channels=[32],
        policy_head_hidden_channels=[32],
        downsample=False,
    )

    perm = np.random.permutation(n_total)
    n_val = max(1, int(n_total * 0.12))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    x_tr = torch.from_numpy(obs_arr[tr_idx])
    v_tr = torch.from_numpy(val_arr[tr_idx])
    p_tr = torch.from_numpy(pol_arr[tr_idx])

    x_va = torch.from_numpy(obs_arr[val_idx])
    v_va = torch.from_numpy(val_arr[val_idx])
    p_va = torch.from_numpy(pol_arr[val_idx])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_tr))
        tr_v_loss = 0.0
        tr_p_loss = 0.0
        batches = 0
        for start in range(0, len(x_tr), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = x_tr[idx]
            vb = v_tr[idx]
            pb = p_tr[idx]

            probs, val_pred = model.compute_policy_value(xb)
            val_pred = val_pred.squeeze(-1)
            loss_v = F.mse_loss(val_pred, vb)
            loss_p = -(pb * torch.log(probs.clamp_min(1e-7))).sum(dim=-1).mean()
            loss = loss_v + 0.5 * loss_p

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_v_loss += float(loss_v.item())
            tr_p_loss += float(loss_p.item())
            batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_probs, va_val = model.compute_policy_value(x_va)
            va_val = va_val.squeeze(-1)
            va_v_loss = float(F.mse_loss(va_val, v_va).item())
            va_p_loss = float((-(p_va * torch.log(va_probs.clamp_min(1e-7))).sum(dim=-1).mean()).item())
            va_total = va_v_loss + 0.5 * va_p_loss
            sign_acc = float(((va_val >= 0) == (v_va >= 0)).float().mean().item())

        if va_total < best_val_loss:
            best_val_loss = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch in (1, 5, 10, args.epochs) or epoch % 5 == 0:
            print(
                f"Epoch {epoch:2d}/{args.epochs} | "
                f"tr_v_mse={tr_v_loss / max(1, batches):.4f} tr_p_ce={tr_p_loss / max(1, batches):.4f} | "
                f"val_v_mse={va_v_loss:.4f} val_p_ce={va_p_loss:.4f} val_sign_acc={sign_acc*100:.1f}%",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state if best_state is not None else model.state_dict()}, out_path)
    print(f"Saved best LightZero AlphaZeroModel checkpoint to {out_path} (val_loss={best_val_loss:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
