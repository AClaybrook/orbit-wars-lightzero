#!/usr/bin/env python3
"""Run a Kaggle Orbit Wars match and export an interactive HTML or JSON replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_agent(spec: str, checkpoint: str | None = None, sims: int = 24, time_budget: float = 0.35):
    """Resolve an agent specification into a Kaggle callable or built-in name."""
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
    path = Path(spec)
    if path.is_file():
        return str(path)
    raise ValueError(
        f"Unknown agent spec {spec!r}. Use 'heuristic', 'mcts', 'starter', 'random', or a file path."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Kaggle Orbit Wars replay.")
    parser.add_argument("--p0", default="heuristic", help="Player 0 agent (heuristic|mcts|starter|random|path).")
    parser.add_argument("--p1", default="starter", help="Player 1 agent (heuristic|mcts|starter|random|path).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for mcts agent.")
    parser.add_argument("--sims", type=int, default=24, help="MCTS simulations per turn for mcts agent.")
    parser.add_argument("--time-budget", type=float, default=0.35, help="Per-turn time budget (seconds).")
    parser.add_argument("--out", default="replays/match.html", help="Output replay file (.html or .json).")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for environment creation.")
    args = parser.parse_args(argv)

    import random
    import numpy as np
    from kaggle_environments import make

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    env = make("orbit_wars", debug=False)
    agent0 = resolve_agent(args.p0, checkpoint=args.checkpoint, sims=args.sims, time_budget=args.time_budget)
    agent1 = resolve_agent(args.p1, checkpoint=args.checkpoint, sims=args.sims, time_budget=args.time_budget)

    steps = env.run([agent0, agent1])
    final_state = steps[-1]
    r0 = final_state[0]["reward"]
    r1 = final_state[1]["reward"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(steps), encoding="utf-8")
    else:
        html = env.render(mode="html")
        out_path.write_text(html, encoding="utf-8")

    print(f"Saved replay to {out_path} ({len(steps)} ticks) | scores: p0={r0}, p1={r1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
