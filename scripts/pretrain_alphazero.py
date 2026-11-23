#!/usr/bin/env python3
"""Expert-Iteration & D4-Augmented LightZero AlphaZeroModel trainer for Orbit Wars.

Upgrades over naive behavioral cloning:
1. Multi-opponent 2P & 4P FFA dataset generation (``heuristic`` self-play plus
   external benchmark bots like ``wd_bestv6`` and ``wd_heuristic`` when installed).
2. **Search-distilled policy targets (Expert Iteration)**: on sampled board states,
   runs active-opponent lookahead rollouts across curated full-turn stances to
   compute exact Boltzmann Q-targets ``softmax(Q / tau)``.
3. **Exact 8× D4 dihedral symmetry augmentation** (rotations + reflections with
   consistent 2D fleet momentum vector rotation via ``augment_obs_d4``), giving
   full 4-quadrant invariance in 4P FFA and eliminating overfitting.
4. Saves official LightZero ``AlphaZeroModel`` checkpoints (``{"model": state_dict}``)
   loadable directly by ``OnlineMCTSAgent``, ``submission.py``, and
   ``scripts/train_lightzero.py --algo alphazero --model-path ...``.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_bot.fast_sim import FastOrbitSim  # noqa: E402
from orbit_wars_bot.features import (  # noqa: E402
    OBS_SHAPE,
    augment_obs_d4,
    encode_observation,
    evaluate_state,
)
from orbit_wars_bot.heuristic_bot import heuristic_moves  # noqa: E402
from orbit_wars_bot.lightzero_compat import setup_lightzero  # noqa: E402
from orbit_wars_bot.macro_actions import (  # noqa: E402
    NUM_MACRO_ACTIONS,
    MacroAction,
    build_tactical_context,
    execute_macro_action,
    get_legal_action_mask,
)
from orbit_wars_bot.mcts_agent import _CURATED_STANCES, _moves_signature  # noqa: E402

_EXPLORE_STANCES: Tuple[int, ...] = tuple(
    int(a) for a in _CURATED_STANCES if a != MacroAction.HOLD_ACCUMULATE
)


def _rank_outcome(scores: List[int], player: int) -> float:
    """Map final ship totals to a value target in [-1.0, 1.0] for 2P or 4P FFA."""
    n = len(scores)
    my_s = scores[player]
    better = sum(1 for s in scores if s > my_s)
    tied = sum(1 for s in scores if s == my_s)
    rank = better + 1.0 + 0.5 * (tied - 1)
    if n <= 2:
        if rank < 1.5:
            return 1.0
        if rank > 1.5:
            return -1.0
        return 0.0
    return float(1.0 - 2.0 * (rank - 1.0) / max(1.0, float(n - 1)))


def _load_optional_external_opponents() -> Dict[str, Callable[[FastOrbitSim, int], List[List[float]]]]:
    """Wrap installed open-source opponents (wd_bestv6, wd_heuristic) for FastOrbitSim."""
    opps: Dict[str, Callable[[FastOrbitSim, int], List[List[float]]]] = {}
    try:
        from scripts.evaluate import resolve_agent

        for name in ("wd_bestv6", "wd_heuristic"):
            try:
                fn = resolve_agent(name)

                def _make_wrapper(agent_fn: Callable) -> Callable[[FastOrbitSim, int], List[List[float]]]:
                    def _wrapped(sim: FastOrbitSim, p: int) -> List[List[float]]:
                        obs = sim.to_observation(player=p)
                        return agent_fn(obs, None) or []

                    return _wrapped

                opps[name] = _make_wrapper(fn)
            except Exception:
                pass
    except Exception:
        pass
    return opps


def _distill_search_policy(
    sim: FastOrbitSim,
    player: int,
    ctx: Any,
    mask: np.ndarray,
    opp_root_moves: Dict[int, List[List[float]]],
    tau: float = 0.035,
) -> Tuple[np.ndarray, float]:
    """Run fast 1-ply active-opponent rollouts across distinct stances -> ``(pi, best_q)``."""
    cand_moves: Dict[int, List[List[float]]] = {}
    seen_sigs: Dict[Tuple[Tuple[int, int, int], ...], int] = {}
    distinct: List[int] = []

    for act_enum in _CURATED_STANCES:
        a = int(act_enum)
        if not mask[a]:
            continue
        if act_enum == MacroAction.HOLD_ACCUMULATE and (ctx.threatened_my or ctx.doomed_my):
            continue
        mv = execute_macro_action(sim, player, a, ctx=ctx)
        sig = _moves_signature(mv)
        if sig not in seen_sigs:
            seen_sigs[sig] = a
            cand_moves[a] = mv
            distinct.append(a)

    if len(distinct) <= 1:
        pi = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
        a0 = distinct[0] if distinct else int(MacroAction.BALANCED_MACRO)
        pi[a0] = 1.0
        return pi, evaluate_state(sim, player)

    n_agents = sim.num_agents
    empty = [[] for _ in range(n_agents)]
    q_vals: Dict[int, float] = {}
    mid_tick = 4 if n_agents == 2 else 5

    for a in distinct:
        s_roll = sim.clone()
        acts = [[] for _ in range(n_agents)]
        acts[player] = cand_moves[a]
        for op, omv in opp_root_moves.items():
            if 0 <= op < n_agents and op != player:
                acts[op] = omv
        s_roll.step(acts)
        for k in range(1, 8):
            if s_roll.done:
                break
            if k == mid_tick:
                alive = {int(pl[1]) for pl in s_roll.planets if 0 <= int(pl[1]) < n_agents}
                m_acts = [[] for _ in range(n_agents)]
                for pid in alive:
                    c = build_tactical_context(s_roll, pid, horizon=14)
                    m_acts[pid] = heuristic_moves(s_roll, pid, ctx=c)
                s_roll.step(m_acts)
            else:
                s_roll.step(empty)
        q_vals[a] = evaluate_state(s_roll, player)

    max_q = max(q_vals.values())
    logits = {a: (q - max_q) / max(1e-4, tau) for a, q in q_vals.items()}
    exps = {a: math.exp(max(-20.0, v)) for a, v in logits.items()}
    z_sum = sum(exps.values())
    pi = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
    for a, e in exps.items():
        pi[a] = float(e / z_sum)
    return pi, float(max_q)


def generate_dataset(
    num_games_2p: int,
    num_games_4p: int,
    *,
    seed: int = 2000,
    sample_stride: int = 7,
    max_steps: int = 340,
    explore_prob: float = 0.20,
    distill_prob: float = 0.35,
    use_external_opponents: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 2P & 4P multi-opponent games with search-distilled Expert Iteration labels."""
    rng = random.Random(seed)
    ext_opps = _load_optional_external_opponents() if use_external_opponents else {}
    ext_names = list(ext_opps.keys())
    if ext_names:
        print(f"  External training opponents enabled: {ext_names}", flush=True)

    obs_list: List[np.ndarray] = []
    val_list: List[float] = []
    pol_list: List[np.ndarray] = []

    schedule = [(2, seed + i) for i in range(num_games_2p)] + [
        (4, seed + 20000 + i) for i in range(num_games_4p)
    ]
    rng.shuffle(schedule)

    distilled_count = 0
    for g_idx, (n_players, g_seed) in enumerate(schedule):
        sim = FastOrbitSim.from_seed(g_seed, num_agents=n_players, max_steps=max_steps)

        # Mix self-play with real games against wd_bestv6 / wd_heuristic (~35% of games)
        player_kinds: List[str] = ["self"] * n_players
        if ext_names and rng.random() < 0.35:
            opp_name = rng.choice(ext_names)
            hero_slot = g_idx % n_players
            for p in range(n_players):
                if p != hero_slot:
                    player_kinds[p] = opp_name

        player_pref = [rng.choice(_EXPLORE_STANCES) for _ in range(n_players)]
        # (player, obs_t, act_id, h_v, distilled_pi_or_None, distilled_q_or_None)
        episode_records: List[Tuple[int, np.ndarray, int, float, Optional[np.ndarray], Optional[float]]] = []

        while not sim.done:
            step = sim.step_count
            sample_now = (step >= 4 and step % sample_stride == 0)
            actions: List[List[List[float]]] = [[] for _ in range(n_players)]
            ctxs: Dict[int, Any] = {}
            masks: Dict[int, np.ndarray] = {}
            chosen_acts: Dict[int, int] = {}

            for p in range(n_players):
                if not any(pl[1] == p for pl in sim.planets):
                    continue
                if player_kinds[p] != "self":
                    actions[p] = ext_opps[player_kinds[p]](sim, p)
                    continue
                ctx = build_tactical_context(sim, p, horizon=18)
                mask = get_legal_action_mask(sim, p, ctx=ctx)
                ctxs[p] = ctx
                masks[p] = mask
                if rng.random() < explore_prob:
                    cand = player_pref[p] if rng.random() < 0.5 else rng.choice(_EXPLORE_STANCES)
                    act_id = cand if mask[cand] else int(MacroAction.BALANCED_MACRO)
                else:
                    act_id = int(MacroAction.BALANCED_MACRO)
                chosen_acts[p] = act_id
                if act_id == int(MacroAction.BALANCED_MACRO):
                    actions[p] = heuristic_moves(sim, p, ctx=ctx)
                else:
                    actions[p] = execute_macro_action(sim, p, act_id, ctx=ctx)

            if sample_now:
                for p in ctxs:
                    obs_t = encode_observation(sim, player=p)
                    h_v = evaluate_state(sim, player=p)
                    dist_pi = None
                    dist_q = None
                    if rng.random() < distill_prob:
                        opp_mvs = {op: actions[op] for op in range(n_players) if op != p}
                        dist_pi, dist_q = _distill_search_policy(sim, p, ctxs[p], masks[p], opp_mvs)
                        distilled_count += 1
                    episode_records.append((p, obs_t, chosen_acts[p], h_v, dist_pi, dist_q))

            sim.step(actions)

        final_scores = sim.scores()
        for p, obs_t, act_id, h_v, dist_pi, dist_q in episode_records:
            z = _rank_outcome(final_scores, p)
            # Value target: predominantly true long-horizon game outcome + small rollout/eval anchor
            anchor_v = dist_q if dist_q is not None else h_v
            target_v = 0.82 * z + 0.18 * anchor_v
            if dist_pi is not None:
                pi = dist_pi
            else:
                pi = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
                if z > 0.0:
                    pi[act_id] = 0.75
                    pi[int(MacroAction.BALANCED_MACRO)] += 0.25
                else:
                    pi[int(MacroAction.BALANCED_MACRO)] = 0.72
                    pi[int(MacroAction.EXPAND_BEST_ROI_MULTI)] = 0.14
                    pi[int(MacroAction.FEED_FRONTLINE)] = 0.14
            obs_list.append(obs_t)
            val_list.append(float(max(-1.0, min(1.0, target_v))))
            pol_list.append(pi)

        if (g_idx + 1) % 10 == 0 or (g_idx + 1) == len(schedule):
            print(
                f"  Simulated {g_idx + 1}/{len(schedule)} games -> {len(obs_list)} samples "
                f"({distilled_count} search-distilled)",
                flush=True,
            )

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(val_list, dtype=np.float32),
        np.asarray(pol_list, dtype=np.float32),
    )


def _augment_batch_d4(xb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random exact D4 dihedral symmetries to every sample in a batch."""
    out = np.empty_like(xb)
    syms = rng.integers(0, 8, size=len(xb))
    for i, s in enumerate(syms):
        out[i] = augment_obs_d4(xb[i], int(s))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expert-Iteration LightZero AlphaZeroModel trainer.")
    parser.add_argument("--games-2p", type=int, default=50, help="Number of 2P games to simulate.")
    parser.add_argument("--games-4p", type=int, default=30, help="Number of 4P FFA games to simulate.")
    parser.add_argument("--distill-prob", type=float, default=0.35, help="Fraction of samples to search-distill.")
    parser.add_argument("--no-external", action="store_true", help="Disable external benchmark opponents in training.")
    parser.add_argument("--epochs", type=int, default=20, help="Supervised training epochs with 8x D4 augmentation.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--save-dataset", type=str, default=None, help="Optional .npz path to cache generated dataset.")
    parser.add_argument("--load-dataset", type=str, default=None, help="Optional .npz path to load cached dataset.")
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints/ckpt_best.pth.tar",
        help="Output checkpoint path (LightZero AlphaZeroModel format).",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    if args.load_dataset and Path(args.load_dataset).is_file():
        print(f"Loading cached dataset from {args.load_dataset}...", flush=True)
        npz = np.load(args.load_dataset)
        obs_arr, val_arr, pol_arr = npz["obs"], npz["val"], npz["pol"]
    else:
        print(
            f"Generating Expert-Iteration dataset: {args.games_2p} 2P + {args.games_4p} 4P FFA games "
            f"(distill_prob={args.distill_prob})...",
            flush=True,
        )
        obs_arr, val_arr, pol_arr = generate_dataset(
            args.games_2p,
            args.games_4p,
            seed=args.seed,
            distill_prob=args.distill_prob,
            use_external_opponents=not args.no_external,
        )
        if args.save_dataset:
            ds_path = Path(args.save_dataset)
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ds_path, obs=obs_arr, val=val_arr, pol=pol_arr)
            print(f"Cached dataset to {ds_path}", flush=True)

    n_total = len(obs_arr)
    print(f"Dataset ready: {n_total} samples in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.epochs <= 0:
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    aug_rng = np.random.default_rng(args.seed)

    setup_lightzero()
    from lzero.model.alphazero_model import AlphaZeroModel

    model = AlphaZeroModel(
        observation_shape=OBS_SHAPE,
        action_space_size=NUM_MACRO_ACTIONS,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        value_head_channels=8,
        policy_head_channels=8,
        value_head_hidden_channels=[args.head_hidden],
        policy_head_hidden_channels=[args.head_hidden],
        downsample=False,
    )

    perm = np.random.permutation(n_total)
    n_val = max(1, int(n_total * 0.12))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    x_tr_np = obs_arr[tr_idx]
    v_tr_t = torch.from_numpy(val_arr[tr_idx])
    p_tr_t = torch.from_numpy(pol_arr[tr_idx])

    x_va = torch.from_numpy(obs_arr[val_idx])
    v_va = torch.from_numpy(val_arr[val_idx])
    p_va = torch.from_numpy(pol_arr[val_idx])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = aug_rng.permutation(len(x_tr_np))
        tr_v_loss = 0.0
        tr_p_loss = 0.0
        batches = 0

        for start in range(0, len(x_tr_np), args.batch_size):
            idx = order[start : start + args.batch_size]
            # Exact 8x D4 symmetry augmentation on every minibatch!
            xb_np = _augment_batch_d4(x_tr_np[idx], aug_rng)
            xb = torch.from_numpy(xb_np)
            vb = v_tr_t[idx]
            pb = p_tr_t[idx]

            probs, val_pred = model.compute_policy_value(xb)
            val_pred = val_pred.squeeze(-1)
            loss_v = F.mse_loss(val_pred, vb)
            loss_p = -(pb * torch.log(probs.clamp_min(1e-7))).sum(dim=-1).mean()
            loss = loss_v + 0.65 * loss_p

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
            va_total = va_v_loss + 0.65 * va_p_loss
            sign_acc = float(((va_val >= 0) == (v_va >= 0)).float().mean().item())
            pol_top1 = float((va_probs.argmax(dim=-1) == p_va.argmax(dim=-1)).float().mean().item())

        saved_mark = ""
        if va_total < best_val_loss:
            best_val_loss = va_total
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, out_path)
            saved_mark = " *saved*"

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"tr_v={tr_v_loss / max(1, batches):.4f} tr_p={tr_p_loss / max(1, batches):.4f} | "
            f"val_v={va_v_loss:.4f} val_p={va_p_loss:.4f} sign={sign_acc*100:.1f}% top1={pol_top1*100:.1f}%{saved_mark}",
            flush=True,
        )

    print(f"Finished! Best LightZero AlphaZeroModel checkpoint at {out_path} (val_loss={best_val_loss:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
