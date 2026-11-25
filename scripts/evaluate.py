#!/usr/bin/env python3
"""Benchmark Orbit Wars agents in official ``kaggle_environments`` (2P and 4P FFA).

Supports:
- Built-in Kaggle bots: ``random``, ``starter``
- Local agents: ``heuristic``, ``mcts`` (optionally with ``--checkpoint``)
- Open-source benchmark bots (via ``python scripts/fetch_opponents.py``):
  ``kvatsa5``, ``wd_bestv6``, ``wd_heuristic``, ``wd_peer``, ``wd_sniper``,
  ``wd_swarm``, ``wd_turtle``
- Arbitrary external Python files defining ``agent(obs, config)`` or ``agent(obs)``
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import inspect
import os
import random
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPP_DIR = REPO_ROOT / "external" / "opponents"
_HOBER_AGENT_CACHE: Callable | None = None


def _adapt_kaggle_callable(fn: Callable) -> Callable:
    """Wrap an agent callable so it works whether it takes 1 or 2 arguments."""
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_varargs or len(params) >= 2:
        return fn

    def _wrapped(obs, config=None):
        del config
        return fn(obs)

    return _wrapped


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


def _load_py_agent(file_path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location(f"ext_agent_{file_path.stem}", str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"{file_path} does not define an `agent` function")
    return _adapt_kaggle_callable(mod.agent)


def _load_wanderduck_agent(kind: str) -> Callable:
    wd_src = OPP_DIR / "wanderduck_orbit_wars" / "src"
    if not wd_src.is_dir():
        raise FileNotFoundError(
            f"Missing {wd_src}. Run `python scripts/fetch_opponents.py` first."
        )
    wd_str = str(wd_src)
    if wd_str not in sys.path:
        sys.path.insert(0, wd_str)

    if kind == "wd_bestv6":
        from orbit_wars.opponents import bestv6_overhaul

        return _adapt_kaggle_callable(bestv6_overhaul.agent)
    if kind == "wd_heuristic":
        from orbit_wars.heuristic import strategy

        return _adapt_kaggle_callable(strategy.agent)
    if kind == "wd_peer":
        from orbit_wars.opponents import peer_mdmahfuzsumon

        return _adapt_kaggle_callable(peer_mdmahfuzsumon.agent)
    if kind == "wd_sniper":
        from orbit_wars.opponents import competent_sniper

        return _adapt_kaggle_callable(competent_sniper.agent)
    if kind == "wd_swarm":
        from orbit_wars.opponents import aggressive_swarm

        return _adapt_kaggle_callable(aggressive_swarm.agent)
    if kind == "wd_turtle":
        from orbit_wars.opponents import defensive_turtle

        return _adapt_kaggle_callable(defensive_turtle.agent)
    raise ValueError(f"Unknown wanderduck agent {kind!r}")


def _resolve_hober_main_path() -> Path:
    """Return Hober Malloc's ``main.py``, extracting the prebuilt package when ``ow-agent`` is absent."""
    hober_root = OPP_DIR / "simjeg_hober_malloc"
    direct_main = hober_root / "main.py"
    if not direct_main.is_file():
        raise FileNotFoundError(
            f"Missing {direct_main}. Run `python scripts/fetch_opponents.py` first."
        )
    compiled_bins = (
        hober_root / "ow-agent",
        hober_root / "target" / "release" / "ow-agent",
        hober_root / "target" / "debug" / "ow-agent",
    )
    if any(b.is_file() for b in compiled_bins):
        return direct_main

    archive = hober_root / "submitted_packages" / "submission1.tar.gz"
    if archive.is_file():
        pkg_dir = OPP_DIR / "hober_malloc_pkg"
        cli_bin = pkg_dir / "agent_cli"
        pkg_main = pkg_dir / "main.py"
        if not (cli_bin.is_file() and pkg_main.is_file()):
            pkg_dir.mkdir(parents=True, exist_ok=True)
            dest_root = pkg_dir.resolve()
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    target = (dest_root / member.name).resolve()
                    if target != dest_root and dest_root not in target.parents:
                        raise RuntimeError(f"Unsafe path in {archive.name}: {member.name!r}")
                if sys.version_info >= (3, 12):
                    tar.extractall(dest_root, filter="data")
                else:
                    tar.extractall(dest_root)
            os.chmod(cli_bin, 0o755)
        return pkg_main
    return direct_main


def _parse_threshold_spec(prefix_base: str, spec: str) -> tuple[bool, float, str | None]:
    """Match ``<prefix_base>[_tNN][:<arg>]`` and return ``(matched, threshold, arg)``."""
    m = re.fullmatch(rf"{re.escape(prefix_base)}(?:_t(\d{{2}}))?(?::(.*))?", spec)
    if not m:
        return False, 0.50, None
    thr = int(m.group(1)) / 100.0 if m.group(1) is not None else 0.50
    arg = m.group(2) if m.group(2) else None
    return True, thr, arg


def resolve_agent(spec: str, checkpoint: str | None = None, sims: int = 20, time_budget: float = 0.35):
    """Resolve an agent identifier or file path to a Kaggle Environments agent."""
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
    if spec in ("planet_transformer", "transformer", "pt") or spec.startswith("planet_transformer:") or spec.startswith("pt:"):
        from orbit_wars_bot.planet_transformer import create_planet_transformer_agent

        override_ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt_path = override_ckpt or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_best.pt")
        return create_planet_transformer_agent(checkpoint_path=ckpt_path)
    if spec in ("planet_transformer_d4", "pt_d4") or spec.startswith("planet_transformer_d4:") or spec.startswith("pt_d4:"):
        from orbit_wars_bot.planet_transformer import create_planet_transformer_agent

        override_ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt_path = override_ckpt or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_best.pt")
        return create_planet_transformer_agent(checkpoint_path=ckpt_path, use_d4_tta=True)
    matched_allin, thr_allin, arg_allin = _parse_threshold_spec("pt_d4_allin", spec)
    if matched_allin:
        from orbit_wars_bot.planet_transformer import create_planet_transformer_agent

        ckpt_path = arg_allin or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_best.pt")
        return create_planet_transformer_agent(
            checkpoint_path=ckpt_path,
            launch_threshold=thr_allin,
            use_d4_tta=True,
            force_all_in=True,
        )
    if spec.startswith("pt_d4_gen9:") or spec == "pt_d4_gen9":
        from orbit_wars_bot.planet_transformer import create_planet_transformer_agent

        override_ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt_path = override_ckpt or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_gen9_balanced.pt")
        return create_planet_transformer_agent(
            checkpoint_path=ckpt_path,
            use_d4_tta=True,
            force_all_in=True,
            reinforce_threshold=0.60,
        )
    matched_search_allin, thr_s_allin, arg_s_allin = _parse_threshold_spec("pt_search_d4_allin", spec)
    if matched_search_allin:
        from orbit_wars_bot.planet_transformer import create_search_planet_transformer_agent

        ckpt_path = arg_s_allin or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_best.pt")
        return create_search_planet_transformer_agent(
            checkpoint_path=ckpt_path,
            launch_threshold=thr_s_allin,
            use_d4_tta=True,
            force_all_in=True,
            neural_mid_replan=True,
            rollout_horizon=10,
            time_budget_sec=min(0.35, time_budget),
        )
    if spec in ("planet_transformer_search", "pt_search") or spec.startswith("planet_transformer_search:") or spec.startswith("pt_search:"):
        from orbit_wars_bot.planet_transformer import create_search_planet_transformer_agent

        override_ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt_path = override_ckpt or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_best.pt")
        return create_search_planet_transformer_agent(
            checkpoint_path=ckpt_path,
            time_budget_sec=min(0.24, time_budget),
        )
    if spec in ("planet_transformer_search_d4", "pt_search_d4") or spec.startswith("planet_transformer_search_d4:") or spec.startswith("pt_search_d4:"):
        from orbit_wars_bot.planet_transformer import create_search_planet_transformer_agent

        override_ckpt = spec.split(":", 1)[1] if ":" in spec else None
        ckpt_path = override_ckpt or checkpoint or str(REPO_ROOT / "checkpoints" / "planet_transformer_best.pt")
        return create_search_planet_transformer_agent(
            checkpoint_path=ckpt_path,
            use_d4_tta=True,
            time_budget_sec=min(0.26, time_budget),
        )
    matched_ens_allin, thr_ens_allin, arg_ens_allin = _parse_threshold_spec("pt_ens_allin", spec)
    if matched_ens_allin and arg_ens_allin:
        from orbit_wars_bot.planet_transformer import create_ensemble_planet_transformer_agent

        ckpt_paths = [p.strip() for p in arg_ens_allin.split(",") if p.strip()]
        return create_ensemble_planet_transformer_agent(
            checkpoint_paths=ckpt_paths,
            launch_threshold=thr_ens_allin,
            use_d4_tta=True,
            force_all_in=True,
        )
    if spec.startswith("planet_transformer_ensemble:") or spec.startswith("pt_ens:") or spec.startswith("pt_ens_d4:"):
        from orbit_wars_bot.planet_transformer import create_ensemble_planet_transformer_agent

        use_d4 = spec.startswith("pt_ens_d4:")
        raw_paths = spec.split(":", 1)[1]
        ckpt_paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
        return create_ensemble_planet_transformer_agent(
            checkpoint_paths=ckpt_paths,
            use_d4_tta=use_d4,
            force_all_in=False,
        )
    if spec.startswith("pt_specialist:"):
        from orbit_wars_bot.planet_transformer import EnsemblePlanetTransformerAgent, PlanetTransformerAgent

        p2_raw, p4_path = [p.strip() for p in spec.split(":", 1)[1].split("|", 1)]
        p2_paths = [p.strip() for p in p2_raw.split(",") if p.strip()]
        ag_2p: Any = (
            EnsemblePlanetTransformerAgent(p2_paths, weights=[0.55, 0.45] if len(p2_paths) == 2 else None, use_d4_tta=True, force_all_in=True)
            if len(p2_paths) > 1
            else PlanetTransformerAgent(p2_paths[0], use_d4_tta=True, force_all_in=True)
        )
        ag_4p = PlanetTransformerAgent(p4_path, use_d4_tta=True, force_all_in=False)

        def _specialist_agent(obs, config=None):
            owners = {int(row[1]) for row in obs.get("planets", []) if int(row[1]) >= 0}
            owners.update(int(row[1]) for row in obs.get("fleets", []) if int(row[1]) >= 0)
            return ag_4p.act(obs, config) if len(owners) > 2 else ag_2p.act(obs, config)

        return _specialist_agent
    if spec == "kvatsa5":
        kv_path = OPP_DIR / "kvatsa5_agent.py"
        if not kv_path.is_file():
            raise FileNotFoundError(
                f"Missing {kv_path}. Run `python scripts/fetch_opponents.py` first."
            )
        return _load_py_agent(kv_path)
    if spec == "subham_v14":
        p = OPP_DIR / "subham_v14.py"
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run `python scripts/fetch_opponents.py` first."
            )
        return _load_py_agent(p)
    if spec == "ow_proto":
        p = OPP_DIR / "simjeg_hober_malloc" / "ow_tools" / "agent_ow_proto.py"
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run `python scripts/fetch_opponents.py` first."
            )
        return _load_py_agent(p)
    if spec == "hober_malloc":
        global _HOBER_AGENT_CACHE
        if _HOBER_AGENT_CACHE is not None:
            return _HOBER_AGENT_CACHE

        main_py = _resolve_hober_main_path()
        os.environ.setdefault("OW_PYTHON", sys.executable)
        os.environ.setdefault("ORBIT_WARS_BC_ALL_IN_PYTHON", sys.executable)
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("AGENT_LOG_EVERY", "0")
        os.environ.setdefault("AGENT_MAX_CHILD_LOG_LINES", "0")
        venv_bin = str(Path(sys.executable).parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
        if not path_parts or path_parts[0] != venv_bin:
            os.environ["PATH"] = os.pathsep.join([venv_bin, *[p for p in path_parts if p != venv_bin]])

        mod_spec = importlib.util.spec_from_file_location("ext_agent_hober_malloc", str(main_py))
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"Could not load agent from {main_py}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        if not hasattr(mod, "agent"):
            raise AttributeError(f"{main_py} does not define an `agent` function")
        raw_fn = _adapt_kaggle_callable(mod.agent)

        def _cleanup_hober_proc() -> None:
            proc = getattr(mod, "_PROC", None)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

        atexit.register(_cleanup_hober_proc)

        def _hober_wrapped(obs, config=None):
            # kaggle_environments(debug=False) replaces sys.stderr with StringIO,
            # which breaks subprocess.Popen(..., stderr=sys.stderr) inside main.py.
            prev_err = sys.stderr
            try:
                if sys.__stderr__ is not None:
                    sys.stderr = sys.__stderr__
                return raw_fn(obs, config)
            finally:
                sys.stderr = prev_err

        _HOBER_AGENT_CACHE = _hober_wrapped
        return _HOBER_AGENT_CACHE
    if spec.startswith("wd_"):
        return _load_wanderduck_agent(spec)

    path = Path(spec)
    if path.is_file():
        return _load_py_agent(path)

    raise ValueError(f"Unknown agent spec: {spec!r}")


def _final_ship_totals(final_step: List[Dict[str, Any]], num_players: int) -> List[int]:
    obs = final_step[0]["observation"]
    totals = [0] * num_players
    for p in obs.get("planets", []):
        if 0 <= p[1] < num_players:
            totals[p[1]] += int(p[5])
    for f in obs.get("fleets", []):
        if 0 <= f[1] < num_players:
            totals[f[1]] += int(f[6])
    return totals


def evaluate_matchup(
    candidate_spec: str,
    opponent_spec: str,
    *,
    num_games: int = 6,
    num_players: int = 2,
    seed: int = 42,
    checkpoint: str | None = None,
    sims: int = 20,
    time_budget: float = 0.35,
    max_steps: int | None = None,
) -> Dict[str, Any]:
    """Run ``num_games`` slot-balanced Orbit Wars matches (2P or 4P FFA)."""
    from kaggle_environments import make

    wins = 0
    losses = 0
    draws = 0
    ranks: List[float] = []
    latencies_ms: List[float] = []
    game_lengths: List[int] = []

    cand_raw = resolve_agent(candidate_spec, checkpoint, sims, time_budget)
    opp_raw = resolve_agent(opponent_spec, checkpoint=None, sims=sims, time_budget=time_budget)

    for g in range(num_games):
        random.seed(seed + g)
        np.random.seed(seed + g)
        config: Dict[str, Any] = {
            "seed": seed + g,
            "actTimeout": 30,
            "agentTimeout": 60,
        }
        if max_steps is not None:
            config["episodeSteps"] = max_steps
        env = make("orbit_wars", configuration=config, debug=False)

        cand = _wrap_timed(cand_raw, latencies_ms)
        cand_slot = g % num_players
        agents = [cand if slot == cand_slot else opp_raw for slot in range(num_players)]
        steps = env.run(agents)
        game_lengths.append(len(steps))

        final = steps[-1]
        rewards = [
            final[i]["reward"] if final[i]["reward"] is not None else -1
            for i in range(num_players)
        ]
        totals = _final_ship_totals(final, num_players)
        my_r = rewards[cand_slot]
        max_r = max(rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_r]

        if cand_slot in winners and len(winners) == 1:
            wins += 1
        elif cand_slot in winners:
            draws += 1
        else:
            losses += 1

        # Compute exact placement (1 = best .. num_players = worst) using ship totals
        my_ships = totals[cand_slot]
        better = sum(1 for s in totals if s > my_ships)
        tied = sum(1 for s in totals if s == my_ships)
        ranks.append(better + 1.0 + 0.5 * (tied - 1))

    lat_arr = np.asarray(latencies_ms, dtype=np.float64) if latencies_ms else np.zeros(1)
    return {
        "candidate": candidate_spec,
        "opponent": opponent_spec,
        "players": num_players,
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / max(1, num_games),
        "avg_rank": float(np.mean(ranks)) if ranks else 0.0,
        "avg_game_steps": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "mean_latency_ms": float(np.mean(lat_arr)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "max_latency_ms": float(np.max(lat_arr)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Orbit Wars agents (2P or 4P FFA).")
    parser.add_argument("--candidate", default="heuristic", help="Candidate agent spec or .py path.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "starter"],
        help="Opponent specs (random, starter, heuristic, mcts, kvatsa5, wd_bestv6, wd_peer, wd_sniper, wd_swarm, wd_turtle, or .py path).",
    )
    parser.add_argument("--players", type=int, choices=[2, 4], default=2, help="2 for 1v1, 4 for 4P FFA.")
    parser.add_argument("--games", type=int, default=6, help="Games per opponent (slot-balanced).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for mcts candidate.")
    parser.add_argument("--sims", type=int, default=20, help="MCTS simulations per turn.")
    parser.add_argument("--time-budget", type=float, default=0.35, help="Per-turn time budget (seconds).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional episode step cap.")
    args = parser.parse_args(argv)

    print(
        f"Evaluating candidate='{args.candidate}' ({args.players}P, {args.games} games/opponent, seed={args.seed})",
        flush=True,
    )
    print("-" * 96, flush=True)
    print(
        f"{'Opponent':<14} {'W-D-L':<10} {'WinRate':>8} {'AvgRank':>8} "
        f"{'AvgLen':>8} {'Mean(ms)':>10} {'P95(ms)':>10} {'Max(ms)':>10}",
        flush=True,
    )
    print("-" * 96, flush=True)
    for opp in args.opponents:
        res = evaluate_matchup(
            args.candidate,
            opp,
            num_games=args.games,
            num_players=args.players,
            seed=args.seed,
            checkpoint=args.checkpoint,
            sims=args.sims,
            time_budget=args.time_budget,
            max_steps=args.max_steps,
        )
        wdl = f"{res['wins']}-{res['draws']}-{res['losses']}"
        print(
            f"{opp:<14} {wdl:<10} {res['win_rate']*100:7.1f}% {res['avg_rank']:8.2f} "
            f"{res['avg_game_steps']:8.1f} {res['mean_latency_ms']:10.2f} "
            f"{res['p95_latency_ms']:10.2f} {res['max_latency_ms']:10.2f}",
            flush=True,
        )
    print("-" * 96, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
