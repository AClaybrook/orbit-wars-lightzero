#!/usr/bin/env python3
"""Local 2P Round-Robin & 4P FFA Elo Tournament for Orbit Wars bots.

Runs official ``kaggle_environments`` matches across local and open-source
Orbit Wars agents, then computes:
  * 2P Head-to-Head Win-Rate Matrix (slot-balanced P0/P1)
  * 2P Bradley-Terry MLE Elo (order-independent, anchored at ``starter = 1000``)
  * 4P FFA Leaderboard (Win %, Top-2 %, Avg Placement 1.00-4.00)
  * 4P Plackett-Luce MLE Elo (multi-player generalization of Elo, anchored at ``starter = 1000``)
  * 95% bootstrap confidence intervals for both Elo ladders

Example:
    python scripts/tournament_elo.py --games-2p 4 --games-4p 48 --workers 2 --json-out results/tournament_elo.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


DEFAULT_BOTS = [
    "lightzero_v3",
    "lightzero_v2",
    "heuristic_v2",
    "wd_bestv6",
    "wd_heuristic",
    "kvatsa5",
    "wd_peer",
    "starter",
]

BOT_DISPLAY_NAMES: Dict[str, str] = {
    "lightzero_v3": "LightZero v3 (Ours)",
    "lightzero_v2": "LightZero v2 (Ours)",
    "mcts_no_nn": "MCTS Only (No NN)",
    "heuristic_v2": "Heuristic v2 (Ours)",
    "wd_bestv6": "wd_bestv6 (CMA-ES)",
    "wd_heuristic": "wd_heuristic",
    "kvatsa5": "kvatsa5 (Predator v3)",
    "subham_v14": "subham_v14 (1224-Elo)",
    "ow_proto": "Agent OW-Proto (1000+)",
    "hober_malloc": "Hober Malloc (10B RL)",
    "wd_peer": "wd_peer (mdmahfuzsumon)",
    "wd_sniper": "wd_sniper",
    "wd_swarm": "wd_swarm",
    "wd_turtle": "wd_turtle",
    "starter": "Kaggle Starter",
    "random": "Kaggle Random",
}


# Per-process agent cache so multiprocessing workers only load checkpoints once.
_WORKER_AGENT_CACHE: Dict[Tuple[str, int, float], Any] = {}


def _resolve_tournament_bot(name: str, sims: int, time_budget: float):
    """Map friendly tournament bot IDs to Kaggle agent callables."""
    key = (name, sims, time_budget)
    if key in _WORKER_AGENT_CACHE:
        return _WORKER_AGENT_CACHE[key]

    from evaluate import resolve_agent

    if name == "lightzero_v3":
        ckpt = str(REPO_ROOT / "checkpoints" / "ckpt_best.pth.tar")
        agent = resolve_agent("lightzero", checkpoint=ckpt, sims=sims, time_budget=time_budget)
    elif name == "lightzero_v2":
        ckpt = str(REPO_ROOT / "checkpoints" / "ckpt_v2_backup.pth.tar")
        agent = resolve_agent("lightzero", checkpoint=ckpt, sims=sims, time_budget=time_budget)
    elif name == "mcts_no_nn":
        agent = resolve_agent("mcts", checkpoint=None, sims=sims, time_budget=time_budget)
    elif name == "heuristic_v2":
        agent = resolve_agent("heuristic")
    else:
        agent = resolve_agent(name, checkpoint=None, sims=sims, time_budget=time_budget)

    _WORKER_AGENT_CACHE[key] = agent
    return agent


def _run_single_game(task: Tuple[List[str], int, int, float, int | None]) -> Dict[str, Any]:
    """Worker function: run one official 2P or 4P Orbit Wars episode."""
    bot_names, seed, sims, time_budget, max_steps = task
    num_players = len(bot_names)

    # Keep torch single-threaded inside each worker to prevent thread oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    from evaluate import _final_ship_totals
    from kaggle_environments import make

    agents = [_resolve_tournament_bot(b, sims, time_budget) for b in bot_names]
    random.seed(seed)
    np.random.seed(seed)
    config: Dict[str, Any] = {
        "seed": seed,
        "actTimeout": 30,
        "agentTimeout": 60,
    }
    if max_steps is not None:
        config["episodeSteps"] = max_steps
    env = make("orbit_wars", configuration=config, debug=False)
    steps = env.run(agents)

    final = steps[-1]
    rewards = [
        float(final[i]["reward"]) if final[i]["reward"] is not None else -1.0
        for i in range(num_players)
    ]
    totals = _final_ship_totals(final, num_players)

    # Track exact survival tick & peak ships for each player so eliminated 0-ship
    # players in 4P FFA are ranked by real survival order instead of random coin flips.
    last_alive = [0] * num_players
    peak_ships = [0] * num_players
    for t_idx, st in enumerate(steps):
        obs = st[0]["observation"]
        cur = [0] * num_players
        for p in obs.get("planets", []):
            if 0 <= p[1] < num_players:
                cur[p[1]] += int(p[5])
        for f in obs.get("fleets", []):
            if 0 <= f[1] < num_players:
                cur[f[1]] += int(f[6])
        for i in range(num_players):
            if cur[i] > 0:
                last_alive[i] = t_idx
                if cur[i] > peak_ships[i]:
                    peak_ships[i] = cur[i]

    # Rank players by (reward, final_ships, last_alive_tick, peak_ships) descending.
    keys = [(rewards[i], totals[i], last_alive[i], peak_ships[i]) for i in range(num_players)]
    ranks: List[float] = []
    for i in range(num_players):
        better = sum(1 for k in keys if k > keys[i])
        tied = sum(1 for k in keys if k == keys[i])
        ranks.append(better + 1.0 + 0.5 * (tied - 1))

    # Build strict ordering for Plackett-Luce (tiny random tiebreak only on exact 4-tuple ties).
    order = list(range(num_players))
    rng = random.Random(seed * 7919 + 13)
    order.sort(key=lambda i: (keys[i], rng.random()), reverse=True)

    return {
        "players": bot_names,
        "seed": seed,
        "steps": len(steps),
        "rewards": rewards,
        "totals": totals,
        "last_alive": last_alive,
        "ranks": ranks,
        "order": [bot_names[i] for i in order],
    }


def fit_bradley_terry_elo(
    bots: Sequence[str],
    pair_outcomes: Sequence[Tuple[int, int, float]],
    anchor_bot: str = "starter",
    anchor_elo: float = 1000.0,
    prior_strength: float = 0.25,
    max_iter: int = 500,
) -> Dict[str, float]:
    """Fit regularized Bradley-Terry MLE and convert to Elo scale (400 / ln(10)).

    ``pair_outcomes`` is a list of ``(i, j, score_i)`` where ``score_i`` is
    1.0 for a win by ``i``, 0.5 for a draw, and 0.0 for a loss by ``i``.
    A tiny symmetric prior (``prior_strength`` virtual draws against the pool
    mean) keeps undefeated bots finite and statistically well-calibrated.
    """
    n = len(bots)
    wins = np.zeros((n, n), dtype=np.float64)
    games = np.zeros((n, n), dtype=np.float64)
    for i, j, s in pair_outcomes:
        wins[i, j] += s
        wins[j, i] += 1.0 - s
        games[i, j] += 1.0
        games[j, i] += 1.0

    # Symmetric virtual draws against every other bot prevent infinite ratings
    # for 100% / 0% bots while remaining negligible when real games accumulate.
    if prior_strength > 0.0 and n > 1:
        p_each = prior_strength / (n - 1)
        for i in range(n):
            for j in range(n):
                if i != j:
                    wins[i, j] += 0.5 * p_each
                    games[i, j] += p_each

    # MM algorithm (Hunter 2004) for Bradley-Terry MLE
    gamma = np.ones(n, dtype=np.float64)
    total_wins = wins.sum(axis=1)
    for _ in range(max_iter):
        denom = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j and games[i, j] > 0.0:
                    denom[i] += games[i, j] / (gamma[i] + gamma[j])
        new_gamma = np.where(denom > 0.0, total_wins / np.maximum(denom, 1e-12), gamma)
        new_gamma /= np.exp(np.mean(np.log(np.maximum(new_gamma, 1e-12))))
        if np.max(np.abs(np.log(new_gamma) - np.log(gamma))) < 1e-8:
            gamma = new_gamma
            break
        gamma = new_gamma

    scale = 400.0 / math.log(10.0)
    raw_elo = scale * np.log(np.maximum(gamma, 1e-12))
    anchor_idx = bots.index(anchor_bot) if anchor_bot in bots else None
    offset = (anchor_elo - raw_elo[anchor_idx]) if anchor_idx is not None else (anchor_elo - float(np.mean(raw_elo)))
    return {b: float(raw_elo[idx] + offset) for idx, b in enumerate(bots)}


def fit_plackett_luce_elo(
    bots: Sequence[str],
    rankings: Sequence[Sequence[int]],
    anchor_bot: str = "starter",
    anchor_elo: float = 1000.0,
    prior_strength: float = 0.25,
    max_iter: int = 500,
) -> Dict[str, float]:
    """Fit Plackett-Luce MLE over k-player rankings via Hunter (2004) MM updates.

    Each element of ``rankings`` is a list of player indices from 1st to last.
    For k=2 this reduces identically to Bradley-Terry Elo.
    """
    n = len(bots)
    # Count how often each player beats at least one remaining player in a ranking.
    w = np.zeros(n, dtype=np.float64)
    for r in rankings:
        for pos in range(len(r) - 1):
            w[r[pos]] += 1.0

    # Add symmetric pairwise prior draws to regularize extreme 0%/100% bots.
    prior_pairs: List[Tuple[int, int, float]] = []
    if prior_strength > 0.0 and n > 1:
        p_each = prior_strength / (n - 1)
        for i in range(n):
            for j in range(i + 1, n):
                prior_pairs.append((i, j, p_each))
                w[i] += 0.5 * p_each
                w[j] += 0.5 * p_each

    gamma = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        denom = np.zeros(n, dtype=np.float64)
        for r in rankings:
            m = len(r)
            suffix_sum = 0.0
            suffix_sums = [0.0] * m
            for pos in range(m - 1, -1, -1):
                suffix_sum += gamma[r[pos]]
                suffix_sums[pos] = suffix_sum
            # Player r[k] appears in the denominator of stages 0..min(k, m-2)
            cum = 0.0
            for pos in range(m):
                if pos < m - 1:
                    cum += 1.0 / max(suffix_sums[pos], 1e-12)
                denom[r[pos]] += cum
        for i, j, weight in prior_pairs:
            s = gamma[i] + gamma[j]
            denom[i] += weight / s
            denom[j] += weight / s

        new_gamma = np.where(denom > 0.0, w / np.maximum(denom, 1e-12), gamma)
        new_gamma /= np.exp(np.mean(np.log(np.maximum(new_gamma, 1e-12))))
        if np.max(np.abs(np.log(new_gamma) - np.log(gamma))) < 1e-8:
            gamma = new_gamma
            break
        gamma = new_gamma

    scale = 400.0 / math.log(10.0)
    raw_elo = scale * np.log(np.maximum(gamma, 1e-12))
    anchor_idx = bots.index(anchor_bot) if anchor_bot in bots else None
    offset = (anchor_elo - raw_elo[anchor_idx]) if anchor_idx is not None else (anchor_elo - float(np.mean(raw_elo)))
    return {b: float(raw_elo[idx] + offset) for idx, b in enumerate(bots)}


def bootstrap_elo_ci(
    bots: Sequence[str],
    items: Sequence[Any],
    fit_fn,
    *,
    anchor_bot: str = "starter",
    n_boot: int = 200,
    seed: int = 123,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Elo ratings."""
    if not items:
        return {b: (1000.0, 1000.0) for b in bots}
    rng = random.Random(seed)
    m = len(items)
    samples: Dict[str, List[float]] = {b: [] for b in bots}
    for _ in range(n_boot):
        resampled = [items[rng.randrange(m)] for _ in range(m)]
        elos = fit_fn(bots, resampled, anchor_bot=anchor_bot)
        for b, val in elos.items():
            samples[b].append(val)
    cis: Dict[str, Tuple[float, float]] = {}
    for b in bots:
        arr = np.asarray(samples[b], dtype=np.float64)
        cis[b] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return cis


def build_4p_schedule(bots: Sequence[str], num_games: int, seed: int) -> List[List[str]]:
    """Build a balanced 4P FFA schedule where every bot plays equally often and across all 4 seats."""
    if len(bots) < 4:
        raise ValueError("Need at least 4 bots for a 4P FFA tournament.")
    rng = random.Random(seed)
    all_combos = list(itertools.combinations(bots, 4))
    rng.shuffle(all_combos)

    counts = {b: 0 for b in bots}
    seat_counts = {b: [0, 0, 0, 0] for b in bots}
    schedule: List[List[str]] = []

    for g in range(num_games):
        # Pick 4 distinct bots prioritizing those with the fewest games played so far.
        order = list(bots)
        rng.shuffle(order)
        order.sort(key=lambda b: counts[b])
        chosen = order[:4]
        for b in chosen:
            counts[b] += 1

        # Assign seats to balance P0..P3 exposure per bot.
        best_perm = list(chosen)
        best_cost = 1e9
        for perm in itertools.permutations(chosen):
            cost = sum((seat_counts[b][s] + 1) ** 2 for s, b in enumerate(perm)) + rng.random() * 0.01
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        for s, b in enumerate(best_perm):
            seat_counts[b][s] += 1
        schedule.append(best_perm)

    return schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 2P & 4P Orbit Wars Elo tournament.")
    parser.add_argument("--bots", nargs="+", default=DEFAULT_BOTS, help="Bot IDs to include.")
    parser.add_argument("--games-2p", type=int, default=4, help="Slot-balanced 2P games per pair (0 to skip).")
    parser.add_argument("--games-4p", type=int, default=48, help="Total 4P FFA tables to play (0 to skip).")
    parser.add_argument("--sims", type=int, default=12, help="MCTS simulations/turn for LightZero bots.")
    parser.add_argument("--time-budget", type=float, default=0.30, help="Per-turn time budget (seconds).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max episode steps.")
    parser.add_argument("--anchor", default="starter", help="Bot anchored at 1000 Elo.")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1), help="Parallel worker processes.")
    parser.add_argument("--seed", type=int, default=2000, help="Base RNG seed.")
    parser.add_argument("--json-out", default=None, help="Optional JSON path to save full results.")
    args = parser.parse_args(argv)

    bots = list(args.bots)
    bot_to_idx = {b: i for i, b in enumerate(bots)}
    t_start = time.time()

    results_2p: List[Dict[str, Any]] = []
    results_4p: List[Dict[str, Any]] = []

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        # ---- 1. 2P Round-Robin ----
        if args.games_2p > 0:
            tasks_2p: List[Tuple[List[str], int, int, float, int | None]] = []
            seed_cursor = args.seed
            for i, j in itertools.combinations(range(len(bots)), 2):
                for g in range(args.games_2p):
                    pair = [bots[i], bots[j]] if (g % 2 == 0) else [bots[j], bots[i]]
                    tasks_2p.append((pair, seed_cursor, args.sims, args.time_budget, args.max_steps))
                    seed_cursor += 1

            print(
                f"[2P Round-Robin] Running {len(tasks_2p)} games across {len(bots)} bots "
                f"({args.games_2p} games/pair, workers={args.workers})...",
                flush=True,
            )
            t0 = time.time()
            if pool is None:
                for idx, task in enumerate(tasks_2p, 1):
                    results_2p.append(_run_single_game(task))
                    if idx % 20 == 0 or idx == len(tasks_2p):
                        print(f"  2P progress: {idx}/{len(tasks_2p)} ({time.time() - t0:.1f}s)", flush=True)
            else:
                futs = [pool.submit(_run_single_game, t) for t in tasks_2p]
                for idx, fut in enumerate(as_completed(futs), 1):
                    results_2p.append(fut.result())
                    if idx % 20 == 0 or idx == len(tasks_2p):
                        print(f"  2P progress: {idx}/{len(tasks_2p)} ({time.time() - t0:.1f}s)", flush=True)

        # ---- 2. 4P FFA Tournament ----
        if args.games_4p > 0 and len(bots) >= 4:
            schedule_4p = build_4p_schedule(bots, args.games_4p, seed=args.seed + 10_000)
            tasks_4p = [
                (table, args.seed + 20_000 + g, args.sims, args.time_budget, args.max_steps)
                for g, table in enumerate(schedule_4p)
            ]
            print(
                f"[4P FFA] Running {len(tasks_4p)} tables across {len(bots)} bots "
                f"(~{4 * len(tasks_4p) // len(bots)} games/bot, workers={args.workers})...",
                flush=True,
            )
            t0 = time.time()
            if pool is None:
                for idx, task in enumerate(tasks_4p, 1):
                    results_4p.append(_run_single_game(task))
                    if idx % 10 == 0 or idx == len(tasks_4p):
                        print(f"  4P progress: {idx}/{len(tasks_4p)} ({time.time() - t0:.1f}s)", flush=True)
            else:
                futs = [pool.submit(_run_single_game, t) for t in tasks_4p]
                for idx, fut in enumerate(as_completed(futs), 1):
                    results_4p.append(fut.result())
                    if idx % 10 == 0 or idx == len(tasks_4p):
                        print(f"  4P progress: {idx}/{len(tasks_4p)} ({time.time() - t0:.1f}s)", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # ---- 3. Aggregate 2P Stats & Bradley-Terry Elo ----
    matrix_2p: Dict[str, Dict[str, Dict[str, int]]] = {
        a: {b: {"w": 0, "d": 0, "l": 0} for b in bots} for a in bots
    }
    pair_outcomes: List[Tuple[int, int, float]] = []
    stats_2p: Dict[str, Dict[str, float]] = {
        b: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for b in bots
    }

    for g in results_2p:
        p0, p1 = g["players"]
        r0, r1 = g["rewards"]
        i0, i1 = bot_to_idx[p0], bot_to_idx[p1]
        stats_2p[p0]["games"] += 1
        stats_2p[p1]["games"] += 1
        if r0 > r1:
            matrix_2p[p0][p1]["w"] += 1
            matrix_2p[p1][p0]["l"] += 1
            stats_2p[p0]["wins"] += 1
            stats_2p[p1]["losses"] += 1
            pair_outcomes.append((i0, i1, 1.0))
        elif r1 > r0:
            matrix_2p[p1][p0]["w"] += 1
            matrix_2p[p0][p1]["l"] += 1
            stats_2p[p1]["wins"] += 1
            stats_2p[p0]["losses"] += 1
            pair_outcomes.append((i0, i1, 0.0))
        else:
            matrix_2p[p0][p1]["d"] += 1
            matrix_2p[p1][p0]["d"] += 1
            stats_2p[p0]["draws"] += 1
            stats_2p[p1]["draws"] += 1
            pair_outcomes.append((i0, i1, 0.5))

    elo_2p = fit_bradley_terry_elo(bots, pair_outcomes, anchor_bot=args.anchor) if pair_outcomes else {}
    ci_2p = (
        bootstrap_elo_ci(bots, pair_outcomes, fit_bradley_terry_elo, anchor_bot=args.anchor, n_boot=300)
        if pair_outcomes
        else {}
    )

    # ---- 4. Aggregate 4P FFA Stats & Plackett-Luce Elo ----
    rankings_4p: List[List[int]] = []
    stats_4p: Dict[str, Dict[str, Any]] = {
        b: {"games": 0, "wins": 0, "top2": 0, "rank_sum": 0.0} for b in bots
    }
    for g in results_4p:
        rankings_4p.append([bot_to_idx[b] for b in g["order"]])
        max_r = max(g["rewards"])
        winners = [i for i, r in enumerate(g["rewards"]) if r == max_r]
        for idx, b in enumerate(g["players"]):
            stats_4p[b]["games"] += 1
            stats_4p[b]["rank_sum"] += g["ranks"][idx]
            if idx in winners and len(winners) == 1:
                stats_4p[b]["wins"] += 1
            if g["ranks"][idx] <= 2.0:
                stats_4p[b]["top2"] += 1

    elo_4p = fit_plackett_luce_elo(bots, rankings_4p, anchor_bot=args.anchor) if rankings_4p else {}
    ci_4p = (
        bootstrap_elo_ci(bots, rankings_4p, fit_plackett_luce_elo, anchor_bot=args.anchor, n_boot=300)
        if rankings_4p
        else {}
    )

    # ---- 5. Print Clean Summary Tables ----
    if pair_outcomes:
        print("\n=== 2P ROUND-ROBIN ELO LADDER (Bradley-Terry MLE, anchor: " f"{args.anchor}=1000) ===")
        print(f"{'Rank':<5} {'Bot':<24} {'Elo':>7} {'95% CI':>15} {'W-D-L':>11} {'Win%':>7}")
        print("-" * 74)
        ranked_2p = sorted(bots, key=lambda b: elo_2p[b], reverse=True)
        for rank, b in enumerate(ranked_2p, 1):
            s = stats_2p[b]
            wr = 100.0 * (s["wins"] + 0.5 * s["draws"]) / max(1, s["games"])
            lo, hi = ci_2p[b]
            wdl = f"{int(s['wins'])}-{int(s['draws'])}-{int(s['losses'])}"
            print(
                f"{rank:<5} {BOT_DISPLAY_NAMES.get(b, b):<24} {elo_2p[b]:7.1f} "
                f"[{lo:6.0f}, {hi:6.0f}] {wdl:>11} {wr:6.1f}%"
            )

        print("\n=== 2P HEAD-TO-HEAD WIN-RATE MATRIX (Row vs Col) ===")
        header = f"{'Bot':<14} " + " ".join(f"{b[:12]:>12}" for b in ranked_2p)
        print(header)
        print("-" * len(header))
        for a in ranked_2p:
            cells = []
            for b in ranked_2p:
                if a == b:
                    cells.append(f"{'—':>12}")
                else:
                    rec = matrix_2p[a][b]
                    tot = rec["w"] + rec["d"] + rec["l"]
                    pct = 100.0 * (rec["w"] + 0.5 * rec["d"]) / max(1, tot)
                    cells.append(f"{pct:3.0f}% ({rec['w']}-{rec['l']})".rjust(12))
            print(f"{a:<14} " + " ".join(cells))

    if rankings_4p:
        print("\n=== 4P FFA ELO LADDER (Plackett-Luce MLE, anchor: " f"{args.anchor}=1000) ===")
        print(f"{'Rank':<5} {'Bot':<24} {'4P Elo':>7} {'95% CI':>15} {'Games':>6} {'Win%':>7} {'Top2%':>7} {'AvgPlc':>7}")
        print("-" * 86)
        ranked_4p = sorted(bots, key=lambda b: elo_4p[b], reverse=True)
        for rank, b in enumerate(ranked_4p, 1):
            s = stats_4p[b]
            g_cnt = max(1, s["games"])
            wr = 100.0 * s["wins"] / g_cnt
            t2 = 100.0 * s["top2"] / g_cnt
            avg_plc = s["rank_sum"] / g_cnt
            lo, hi = ci_4p[b]
            print(
                f"{rank:<5} {BOT_DISPLAY_NAMES.get(b, b):<24} {elo_4p[b]:7.1f} "
                f"[{lo:6.0f}, {hi:6.0f}] {s['games']:6d} {wr:6.1f}% {t2:6.1f}% {avg_plc:7.2f}"
            )

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s.", flush=True)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bots": bots,
            "display_names": {b: BOT_DISPLAY_NAMES.get(b, b) for b in bots},
            "anchor": args.anchor,
            "games_2p_per_pair": args.games_2p,
            "total_2p_games": len(results_2p),
            "total_4p_games": len(results_4p),
            "sims": args.sims,
            "elapsed_sec": round(elapsed, 2),
            "elo_2p": elo_2p,
            "ci_2p": {b: list(v) for b, v in ci_2p.items()},
            "stats_2p": stats_2p,
            "matrix_2p": matrix_2p,
            "elo_4p": elo_4p,
            "ci_4p": {b: list(v) for b, v in ci_4p.items()},
            "stats_4p": {
                b: {
                    "games": s["games"],
                    "wins": s["wins"],
                    "top2": s["top2"],
                    "win_rate": s["wins"] / max(1, s["games"]),
                    "top2_rate": s["top2"] / max(1, s["games"]),
                    "avg_rank": s["rank_sum"] / max(1, s["games"]),
                }
                for b, s in stats_4p.items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved tournament JSON -> {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
