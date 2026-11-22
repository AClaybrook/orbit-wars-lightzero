#!/usr/bin/env python3
"""LightZero training entry point for Orbit Wars.

Examples:
    # Fast CPU smoke test (AlphaZero with pure-Python MCTS)
    python scripts/train_lightzero.py --algo alphazero --smoke-test

    # AlphaZero self-play training
    python scripts/train_lightzero.py --algo alphazero --mode self_play --max-env-step 50000

    # MuZero training vs heuristic bot
    python scripts/train_lightzero.py --algo muzero --mode bot --max-env-step 50000

    # Gumbel MuZero (Linux/WSL2 with compiled C++ ctree)
    python scripts/train_lightzero.py --algo gumbel_muzero --ctree --max-env-step 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_bot.lightzero_compat import (  # noqa: E402
    is_ctree_alphazero_available,
    is_ctree_muzero_available,
    setup_lightzero,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Orbit Wars agents with LightZero.")
    parser.add_argument(
        "--algo",
        choices=["alphazero", "muzero", "gumbel_muzero"],
        default="alphazero",
        help="LightZero algorithm to train.",
    )
    parser.add_argument(
        "--mode",
        choices=["bot", "self_play"],
        default="bot",
        help="Training battle mode: play_with_bot_mode ('bot') or self_play_mode ('self_play').",
    )
    parser.add_argument(
        "--bot",
        choices=["heuristic", "starter", "random"],
        default="heuristic",
        help="Opponent bot type when --mode bot is selected.",
    )
    parser.add_argument(
        "--ctree",
        action="store_true",
        default=None,
        help="Force C++ ctree MCTS (requires compiled LightZero extensions).",
    )
    parser.add_argument(
        "--ptree",
        action="store_true",
        default=False,
        help="Force pure-Python ptree MCTS (works on Windows and Linux without C++ build).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--collector-env-num", type=int, default=4)
    parser.add_argument("--evaluator-env-num", type=int, default=2)
    parser.add_argument("--n-episode", type=int, default=4)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--update-per-collect", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-env-step", type=int, default=50_000)
    parser.add_argument("--max-train-iter", type=int, default=10_000)
    parser.add_argument(
        "--env-manager",
        choices=["base", "subprocess"],
        default="base",
        help="DI-engine env manager ('base' is simplest on Windows/CPU, 'subprocess' for parallel).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny fast end-to-end training loop to verify wiring and checkpointing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_lightzero()

    battle_mode = "self_play_mode" if args.mode == "self_play" else "play_with_bot_mode"

    if args.ptree:
        use_ctree = False
    elif args.ctree:
        use_ctree = True
    else:
        use_ctree = args.algo == "gumbel_muzero"

    if use_ctree:
        if args.algo == "alphazero" and not is_ctree_alphazero_available():
            print(
                "[warn] C++ ctree_alphazero is not compiled; falling back to pure-Python ptree.",
                file=sys.stderr,
            )
            use_ctree = False
        elif args.algo in ("muzero", "gumbel_muzero") and not is_ctree_muzero_available():
            if args.algo == "gumbel_muzero":
                raise RuntimeError(
                    "GumbelMuZero requires LightZero's compiled C++ ctree extensions. "
                    "Build them on Linux/WSL2 via `cd external/LightZero && pip install -e . --no-deps`."
                )
            print(
                "[warn] C++ ctree_muzero is not compiled; falling back to pure-Python ptree.",
                file=sys.stderr,
            )
            use_ctree = False

    if args.smoke_test:
        args.collector_env_num = 1
        args.evaluator_env_num = 1
        args.n_episode = 1
        args.num_simulations = 4
        args.update_per_collect = 2
        args.batch_size = 8
        args.decision_interval = 10
        args.max_steps = 80
        args.max_env_step = 20
        args.max_train_iter = 4
        args.env_manager = "base"
        if args.exp_name is None:
            args.exp_name = f"outputs/smoke_{args.algo}_{args.mode}_seed{args.seed}"

    # Ensure environment registration is loaded before LightZero compiles config.
    import orbit_wars_bot.lightzero_env  # noqa: F401
    from orbit_wars_bot.configs.alphazero_config import make_alphazero_config
    from orbit_wars_bot.configs.gumbel_muzero_config import make_gumbel_muzero_config
    from orbit_wars_bot.configs.muzero_config import make_muzero_config

    common_kwargs = dict(
        battle_mode=battle_mode,
        bot_action_type=args.bot,
        mcts_ctree=use_ctree,
        collector_env_num=args.collector_env_num,
        evaluator_env_num=args.evaluator_env_num,
        n_episode=args.n_episode,
        num_simulations=args.num_simulations,
        update_per_collect=args.update_per_collect,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        decision_interval=args.decision_interval,
        cuda=args.cuda,
        env_manager_type=args.env_manager,
        exp_name=args.exp_name,
        seed=args.seed,
    )

    if args.algo == "alphazero":
        from lzero.entry import train_alphazero

        main_cfg, create_cfg = make_alphazero_config(**common_kwargs)
        if args.smoke_test:
            main_cfg.policy.eval_freq = 2
            main_cfg.policy.model.num_res_blocks = 1
            main_cfg.policy.model.num_channels = 16
        train_alphazero(
            [main_cfg, create_cfg],
            seed=args.seed,
            model_path=args.model_path,
            max_train_iter=args.max_train_iter,
            max_env_step=args.max_env_step,
        )
    elif args.algo == "muzero":
        from lzero.entry import train_muzero

        main_cfg, create_cfg = make_muzero_config(**common_kwargs)
        if args.smoke_test:
            main_cfg.policy.eval_freq = 2
            main_cfg.policy.game_segment_length = 8
            main_cfg.policy.td_steps = 4
            main_cfg.policy.num_unroll_steps = 2
            main_cfg.policy.model.num_res_blocks = 1
            main_cfg.policy.model.num_channels = 16
        train_muzero(
            [main_cfg, create_cfg],
            seed=args.seed,
            model_path=args.model_path,
            max_train_iter=args.max_train_iter,
            max_env_step=args.max_env_step,
        )
    else:
        from lzero.entry import train_muzero

        main_cfg, create_cfg = make_gumbel_muzero_config(**common_kwargs)
        if args.smoke_test:
            main_cfg.policy.eval_freq = 2
            main_cfg.policy.game_segment_length = 8
            main_cfg.policy.td_steps = 4
            main_cfg.policy.num_unroll_steps = 2
            main_cfg.policy.max_num_considered_actions = 4
            main_cfg.policy.model.num_res_blocks = 1
            main_cfg.policy.model.num_channels = 16
        train_muzero(
            [main_cfg, create_cfg],
            seed=args.seed,
            model_path=args.model_path,
            max_train_iter=args.max_train_iter,
            max_env_step=args.max_env_step,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
