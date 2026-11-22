#!/usr/bin/env python3
"""Benchmark Orbit Wars agents in official ``kaggle_environments``.

Evaluates side-balanced matches (half as Player 0, half as Player 1) and reports
win/draw/loss rate, score margins, and per-turn wall-clock latency against the
500ms Kaggle turn budget.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _wrap_timed(agent: Any, latencies_ms: List[float]):
    """Wrap a callable agent to record per-turn latency in milliseconds."""
    if not callable(agent):
        return agent

    def _timed(obs, config):
        t0 = time.perf_counter()
        action = agent(obs, config)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return action

    return _timed


def _resolve(spec: str, checkpoint: str | None, sims: int, time_budget: float):
    if spec in ("random", "starter", "do_nothing"):
        return spec
    if spec == "heuristic":
        from orbit_wars_bot.heuristic_bot import heuristic_agent

        return heuristic_agent
    if spec in ("mcts", "submission", "lightzero"):
        from orbit_wars_bot.mcts_agent import create_kaggle_agent

        return create_kaggle_agent(
            checkpoint_path=checkpoint,
            num_simulations=sims,
            time_budget_sec=time_budget,
        )
    raise ValueError(f"Unknown agent spec: {spec!r}")


def evaluate_matchup(
    candidate_spec: str,
    opponent_spec: str,
    *,
    num_games: int = 10,
    seed: int = 42,
    checkpoint: str | None = None,
    sims: int = 20,
    time_budget: float = 0.35,
    max_steps: int | None = None,
) -> Dict[str, Any]:
    """Run ``num_games`` side-balanced Orbit Wars matches and return summary metrics."""
    from kaggle_environments import make

    wins = 0
    losses = 0
    draws = 0
    score_diffs: List[float] = []
    latencies_ms: List[float] = []
    game_lengths: List[int] = []

    cand_raw = _resolve(candidate_spec, checkpoint, sims, time_budget)
    opp_raw = _resolve(opponent_spec, checkpoint=None, sims=sims, time_budget=time_budget)

    for g in range(num_games):
        random.seed(seed + g)
        np.random.seed(seed + g)
        config = {"episodeSteps": max_steps} if max_steps is not None else {}
        env = make("orbit_wars", configuration=config, debug=False)

        cand = _wrap_timed(cand_raw, latencies_ms)
        cand_is_p0 = (g % 2 == 0)
        agents = [cand, opp_raw] if cand_is_p0 else [opp_raw, cand]
        steps = env.run(agents)
        game_lengths.append(len(steps))

        final = steps[-1]
        r0 = final[0]["reward"] if final[0]["reward"] is not None else -1
        r1 = final[1]["reward"] if final[1]["reward"] is not None else -1
        c_r, o_r = (r0, r1) if cand_is_p0 else (r1, r0)

        if c_r > o_r:
            wins += 1
        elif c_r < o_r:
            losses += 1
        else:
            draws += 1
        score_diffs.append(float(c_r - o_r))

    lat_arr = np.asarray(latencies_ms, dtype=np.float64) if latencies_ms else np.zeros(1)
    return {
        "candidate": candidate_spec,
        "opponent": opponent_spec,
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / max(1, num_games),
        "avg_score_diff": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "avg_game_steps": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "mean_latency_ms": float(np.mean(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "max_latency_ms": float(np.max(lat_arr)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Orbit Wars agents.")
    parser.add_argument("--candidate", default="heuristic", help="Candidate agent (heuristic|mcts|starter|random).")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "starter"],
        help="Opponent agents to benchmark against.",
    )
    parser.add_argument("--games", type=int, default=6, help="Games per opponent (side-balanced).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for mcts candidate.")
    parser.add_argument("--sims", type=int, default=20, help="MCTS simulations per turn.")
    parser.add_argument("--time-budget", type=float, default=0.35, help="Per-turn time budget (seconds).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional episode step cap.")
    args = parser.parse_args(argv)

    print(
        f"Evaluating candidate='{args.candidate}' ({args.games} games/opponent, seed={args.seed})",
        flush=True,
    )
    print("-" * 92, flush=True)
    print(
        f"{'Opponent':<12} {'W-D-L':<10} {'WinRate':>8} {'AvgDiff':>9} "
        f"{'AvgLen':>8} {'Mean(ms)':>10} {'P95(ms)':>10} {'Max(ms)':>10}",
        flush=True,
    )
    print("-" * 92, flush=True)
    for opp in args.opponents:
        res = evaluate_matchup(
            args.candidate,
            opp,
            num_games=args.games,
            seed=args.seed,
            checkpoint=args.checkpoint,
            sims=args.sims,
            time_budget=args.time_budget,
            max_steps=args.max_steps,
        )
        wdl = f"{res['wins']}-{res['draws']}-{res['losses']}"
        print(
            f"{opp:<12} {wdl:<10} {res['win_rate']*100:7.1f}% {res['avg_score_diff']:9.1f} "
            f"{res['avg_game_steps']:8.1f} {res['mean_latency_ms']:10.2f} "
            f"{res['p95_latency_ms']:10.2f} {res['max_latency_ms']:10.2f}",
            flush=True,
        )
    print("-" * 92, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
