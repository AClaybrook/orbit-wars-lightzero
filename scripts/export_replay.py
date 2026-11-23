#!/usr/bin/env python3
"""Run a Kaggle Orbit Wars match (2P or 4P) and export an interactive HTML or JSON replay."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate import resolve_agent  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Kaggle Orbit Wars replay (2P or 4P).")
    parser.add_argument("--p0", default="heuristic", help="Player 0 agent spec or .py path.")
    parser.add_argument("--p1", default="starter", help="Player 1 agent spec or .py path.")
    parser.add_argument("--p2", default=None, help="Optional Player 2 agent (enables 4P FFA).")
    parser.add_argument("--p3", default=None, help="Optional Player 3 agent (enables 4P FFA).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for mcts agent.")
    parser.add_argument("--sims", type=int, default=24, help="MCTS simulations per turn for mcts agent.")
    parser.add_argument("--time-budget", type=float, default=0.35, help="Per-turn time budget (seconds).")
    parser.add_argument("--out", default="replays/match.html", help="Output replay file (.html or .json).")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for environment creation.")
    args = parser.parse_args(argv)

    from kaggle_environments import make

    config = {}
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        config["seed"] = args.seed

    env = make("orbit_wars", configuration=config, debug=False)
    specs = [args.p0, args.p1]
    if args.p2 is not None or args.p3 is not None:
        specs.append(args.p2 or args.p1)
        specs.append(args.p3 or args.p1)

    agents = [
        resolve_agent(s, checkpoint=args.checkpoint, sims=args.sims, time_budget=args.time_budget)
        for s in specs
    ]

    steps = env.run(agents)
    rewards = [st["reward"] for st in steps[-1]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(steps), encoding="utf-8")
    else:
        html = env.render(mode="html")
        out_path.write_text(html, encoding="utf-8")

    print(f"Saved replay to {out_path} ({len(steps)} ticks) | players={specs} | rewards={rewards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
