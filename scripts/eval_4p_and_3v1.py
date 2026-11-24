"""Evaluate Gen 8 vs. Gen 7 in 4P FFA and run the 3v1 Coalition Gauntlet.

Usage:
  python scripts/eval_4p_and_3v1.py --suite 4p_gen8_vs_gen7
  python scripts/eval_4p_and_3v1.py --suite 3v1_subham
  python scripts/eval_4p_and_3v1.py --suite 3v1_boss_hober
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_environments import make

from orbit_wars_bot.coalition import CoalitionCoordinator, create_coalition_team
from scripts.evaluate import resolve_agent


def _run_4p_match(
    agents: List[Any],
    seed: int,
    max_steps: int = 500,
) -> Dict[str, Any]:
    """Run one official 4-player Kaggle Orbit Wars match and return rich stats."""
    t0 = time.perf_counter()
    env = make(
        "orbit_wars",
        configuration={"episodeSteps": max_steps, "seed": seed, "actTimeout": 5.0},
        debug=False,
    )
    steps = env.run(agents)
    elapsed = time.perf_counter() - t0
    final_state = steps[-1]
    rewards = [float(s.reward if s.reward is not None else -1e9) for s in final_state]
    statuses = [str(s.status) for s in final_state]

    # Determine elimination step per player (first tick with 0 planets and 0 fleets)
    elim_step: Dict[int, Optional[int]] = {i: None for i in range(4)}
    for st in steps:
        obs = st[0].observation
        t = int(obs.get("step", 0))
        if t == 0:
            continue
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        alive = {int(p[1]) for p in planets if int(p[1]) >= 0} | {
            int(f[1]) for f in fleets if int(f[1]) >= 0
        }
        for p in range(4):
            if elim_step[p] is None and p not in alive:
                elim_step[p] = t

    # Rank players 1..4 (1 = winner, 4 = first out / lowest reward)
    order = sorted(
        range(4),
        key=lambda i: (rewards[i], elim_step[i] if elim_step[i] is not None else 9999),
        reverse=True,
    )
    ranks = {p: idx + 1 for idx, p in enumerate(order)}
    winner = order[0]

    return {
        "seed": seed,
        "ticks": len(steps) - 1,
        "elapsed": round(elapsed, 2),
        "rewards": rewards,
        "statuses": statuses,
        "winner": winner,
        "ranks": ranks,
        "elim_step": elim_step,
        "steps": steps,
    }


def _pack_replay_from_steps(
    title: str,
    subtitle: str,
    labels: List[str],
    seed: int,
    steps: Any,
    stride: int = 1,
) -> Dict[str, Any]:
    """Pack a Kaggle Environments episode into the exact gameplay_viewer.html schema."""
    obs0 = steps[0][0].observation
    ang_vel = round(float(obs0.get("angular_velocity", 0.04)), 6)
    init_planets = [
        [
            int(p[0]),
            int(p[1]),
            round(float(p[2]), 3),
            round(float(p[3]), 3),
            round(float(p[4]), 3),
            int(p[5]),
            int(p[6]),
        ]
        for p in obs0.get("planets", [])
    ]

    # Collect comet spawns by first appearance step
    comets_by_step: Dict[str, Dict[str, Any]] = {}
    seen_comet_pids: set[int] = set()
    for st in steps:
        obs = st[0].observation
        t = int(obs.get("step", 0))
        by_id = {int(p[0]): p for p in obs.get("planets", [])}
        for grp in obs.get("comets", []):
            pids = [int(pid) for pid in grp.get("planet_ids", [])]
            if not pids or pids[0] in seen_comet_pids:
                continue
            for pid in pids:
                seen_comet_pids.add(pid)
            ships = int(by_id[pids[0]][5]) if pids[0] in by_id else 5
            paths = [
                [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in path]
                for path in grp.get("paths", [])
            ]
            comets_by_step[str(t)] = {"pids": pids, "ships": ships, "paths": paths}

    # Collect non-empty action steps
    n_players = len(labels)
    actions_map: Dict[str, List[List[List[Any]]]] = {}
    for idx in range(len(steps) - 1):
        t = int(steps[idx][0].observation.get("step", idx))
        per_p = []
        any_act = False
        for p in range(n_players):
            raw = steps[idx + 1][p].action or []
            clean = [[int(a[0]), round(float(a[1]), 4), int(a[2])] for a in raw]
            if clean:
                any_act = True
            per_p.append(clean)
        if any_act:
            actions_map[str(t)] = per_p

    rewards = [float(s.reward if s.reward is not None else 0.0) for s in steps[-1]]
    return {
        "title": title,
        "players": labels,
        "rewards": rewards,
        "ticks": len(steps) - 1,
        "angVel": ang_vel,
        "initPlanets": init_planets,
        "comets": comets_by_step,
        "actions": actions_map,
    }


def run_4p_gen8_vs_gen7(save_replay: Optional[Path] = None) -> Dict[str, Any]:
    print("=== SUITE 1: Gen 8 (PT-Gen8) vs. Gen 7 (PT-v5-Boss) in 4P FFA ===")
    gen8_spec = "pt_d4_allin:checkpoints/planet_transformer_gen8_2p_softkd.pt"
    gen7_spec = "pt_d4:checkpoints/planet_transformer_v5_boss.pt"

    # A) 2x Gen 8 vs 2x Gen 7 (balanced seats across 6 games)
    seats_schedule = [
        ("G8", "G7", "G8", "G7"),
        ("G7", "G8", "G7", "G8"),
        ("G8", "G8", "G7", "G7"),
        ("G7", "G7", "G8", "G8"),
        ("G8", "G7", "G7", "G8"),
        ("G7", "G8", "G8", "G7"),
    ]
    g8_wins = 0
    g7_wins = 0
    g8_ranks: List[int] = []
    g7_ranks: List[int] = []
    g8_top2 = 0
    g7_top2 = 0
    games_2v2 = []
    best_replay = None

    for idx, seats in enumerate(seats_schedule):
        seed = 600 + idx
        agents = [resolve_agent(gen8_spec if s == "G8" else gen7_spec) for s in seats]
        res = _run_4p_match(agents, seed=seed)
        w_tag = seats[res["winner"]]
        if w_tag == "G8":
            g8_wins += 1
        else:
            g7_wins += 1
        for p, s in enumerate(seats):
            r = res["ranks"][p]
            if s == "G8":
                g8_ranks.append(r)
                if r <= 2:
                    g8_top2 += 1
            else:
                g7_ranks.append(r)
                if r <= 2:
                    g7_top2 += 1
        print(
            f"  [2xG8 vs 2xG7] seed={seed} seats={seats} winner=P{res['winner']}({w_tag}) "
            f"ranks={res['ranks']} ticks={res['ticks']} ({res['elapsed']}s)"
        )
        games_2v2.append(
            {
                "seed": seed,
                "seats": list(seats),
                "winner": w_tag,
                "ranks": res["ranks"],
                "ticks": res["ticks"],
            }
        )
        if best_replay is None and w_tag == "G7" or (idx == 0 and best_replay is None):
            best_replay = _pack_replay_from_steps(
                f"4P FFA: 2× PT-Gen8 vs. 2× Gen 7 PT-v5-Boss (seed {seed})",
                f"Winner: P{res['winner']} ({'PT-Gen8' if w_tag == 'G8' else 'Gen 7 PT-v5-Boss'}) in {res['ticks']} ticks",
                ["PT-Gen8" if s == "G8" else "Gen7-v5-Boss" for s in seats],
                seed,
                res["steps"],
                stride=2,
            )

    # B) 1x PT-Gen8 vs 3x subham_v14 (4P FFA, seeds 400-403, direct comparison to Gen 7's 4-0 133.8t!)
    sub_wins = 0
    sub_ticks = []
    games_sub = []
    for i in range(4):
        seed = 400 + i
        agents = [resolve_agent(gen8_spec)] + [resolve_agent("subham_v14") for _ in range(3)]
        res = _run_4p_match(agents, seed=seed)
        won = res["winner"] == 0
        if won:
            sub_wins += 1
        sub_ticks.append(res["ticks"])
        print(
            f"  [G8 vs 3x subham_v14] seed={seed} winner=P{res['winner']} "
            f"G8_rank={res['ranks'][0]} ticks={res['ticks']} ({res['elapsed']}s)"
        )
        games_sub.append(
            {"seed": seed, "won": won, "rank": res["ranks"][0], "ticks": res["ticks"]}
        )

    out = {
        "2x_gen8_vs_2x_gen7": {
            "gen8_wins": g8_wins,
            "gen7_wins": g7_wins,
            "gen8_avg_rank": round(sum(g8_ranks) / len(g8_ranks), 2),
            "gen7_avg_rank": round(sum(g7_ranks) / len(g7_ranks), 2),
            "gen8_top2_rate": round(100.0 * g8_top2 / len(g8_ranks), 1),
            "gen7_top2_rate": round(100.0 * g7_top2 / len(g7_ranks), 1),
            "games": games_2v2,
        },
        "gen8_vs_3x_subham_v14": {
            "wins": sub_wins,
            "games": 4,
            "avg_ticks": round(sum(sub_ticks) / len(sub_ticks), 1),
            "details": games_sub,
        },
    }
    print("SUMMARY 4P:", json.dumps(out, indent=2))
    if save_replay and best_replay:
        save_replay.write_text(json.dumps(best_replay))
    return out


def run_3v1_gauntlet(
    boss_name: str,
    boss_spec: str,
    ally_name: str,
    ally_spec: str,
    modes: Sequence[Tuple[str, int]],
    seeds: Sequence[int] = (500, 501, 502, 503),
    save_replay: Optional[Path] = None,
) -> Dict[str, Any]:
    print(f"=== 3v1 GAUNTLET: 3× {ally_name} vs. Boss {boss_name} ===")
    results: Dict[str, Any] = {}
    saved_replay = None

    for mode, latency in modes:
        key = f"{mode}_lat{latency}"
        boss_wins = 0
        boss_elims = 0
        boss_ranks: List[int] = []
        elim_ticks: List[int] = []
        ff_blocked_tot = 0
        nc_avoid_tot = 0
        volleys_tot = 0
        per_game = []

        for seed in seeds:
            boss_ag = resolve_agent(boss_spec)
            coord, team = create_coalition_team(
                lambda: resolve_agent(ally_spec),
                boss_id=0,
                ally_ids=(1, 2, 3),
                mode=mode,
                comm_latency=latency,
            )
            agents = [boss_ag, team[1], team[2], team[3]]
            res = _run_4p_match(agents, seed=seed)

            b_won = res["winner"] == 0
            b_rank = res["ranks"][0]
            b_elim = res["elim_step"][0]
            if b_won:
                boss_wins += 1
            if not b_won:
                boss_elims += 1
                if b_elim is not None:
                    elim_ticks.append(b_elim)
            boss_ranks.append(b_rank)
            ff_blocked_tot += coord.stats.friendly_fire_blocked
            nc_avoid_tot += coord.stats.neutral_conflicts_avoided
            volleys_tot += coord.stats.sync_volleys_launched

            print(
                f"  [{key}] seed={seed} Boss_won={b_won} Boss_rank={b_rank} "
                f"Boss_elim_t={b_elim} ticks={res['ticks']} "
                f"ff_blk={coord.stats.friendly_fire_blocked} "
                f"nc_avd={coord.stats.neutral_conflicts_avoided} "
                f"volleys={coord.stats.sync_volleys_launched} ({res['elapsed']}s)"
            )
            per_game.append(
                {
                    "seed": seed,
                    "boss_won": b_won,
                    "boss_rank": b_rank,
                    "boss_elim_step": b_elim,
                    "ticks": res["ticks"],
                    "ff_blocked": coord.stats.friendly_fire_blocked,
                    "nc_avoided": coord.stats.neutral_conflicts_avoided,
                    "sync_volleys": coord.stats.sync_volleys_launched,
                }
            )
            # Save the most dramatic coalition takedown replay (or first focus_fire game)
            if (
                save_replay is not None
                and mode == "focus_fire"
                and latency == 0
                and (saved_replay is None or (not b_won and saved_replay.get("_boss_won", True)))
            ):
                rep = _pack_replay_from_steps(
                    f"3v1 Coalition: 3× {ally_name} (Focus-Fire) vs. {boss_name} (seed {seed})",
                    (
                        f"Coalition TAKEDOWN! Boss eliminated at t={b_elim} (rank #{b_rank})"
                        if not b_won
                        else f"Boss SURVIVED 3v1 focus-fire and won in {res['ticks']}t!"
                    ),
                    [boss_name, f"{ally_name}#1", f"{ally_name}#2", f"{ally_name}#3"],
                    seed,
                    res["steps"],
                    stride=2,
                )
                rep["_boss_won"] = b_won
                saved_replay = rep

        n = len(seeds)
        results[key] = {
            "mode": mode,
            "comm_latency": latency,
            "games": n,
            "boss_wins": boss_wins,
            "coalition_takedowns": boss_elims,
            "coalition_win_rate": round(100.0 * boss_elims / n, 1),
            "boss_avg_rank": round(sum(boss_ranks) / n, 2),
            "avg_boss_elim_tick": round(sum(elim_ticks) / len(elim_ticks), 1) if elim_ticks else None,
            "avg_ff_blocked": round(ff_blocked_tot / n, 1),
            "avg_nc_avoided": round(nc_avoid_tot / n, 1),
            "avg_sync_volleys": round(volleys_tot / n, 1),
            "per_game": per_game,
        }

    if save_replay is not None and saved_replay is not None:
        saved_replay.pop("_boss_won", None)
        save_replay.write_text(json.dumps(saved_replay))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "4p_gen8_vs_gen7",
            "3v1_subham_vs_gen8",
            "3v1_subham_vs_gen7",
            "3v1_subham_vs_hober",
            "3v1_gen7_vs_hober",
            "3v1_gen8_vs_hober",
        ],
        required=True,
    )
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated mode:lat pairs, e.g. focus_fire:0,focus_fire:5")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--save-replay", type=str, default=None)
    args = parser.parse_args()

    save_rep = Path(args.save_replay) if args.save_replay else None
    gen8_spec = "pt_d4_allin:checkpoints/planet_transformer_gen8_2p_softkd.pt"
    gen7_spec = "pt_d4:checkpoints/planet_transformer_v5_boss.pt"
    full_modes = [("none", 0), ("truce", 0), ("focus_fire", 0), ("focus_fire", 5)]
    if args.modes:
        full_modes = [(m.split(":")[0], int(m.split(":")[1])) for m in args.modes.split(",")]

    if args.suite == "4p_gen8_vs_gen7":
        res = run_4p_gen8_vs_gen7(save_replay=save_rep)
    elif args.suite == "3v1_subham_vs_gen8":
        res = run_3v1_gauntlet(
            "PT-Gen8", gen8_spec, "subham_v14", "subham_v14", full_modes, seeds=(500, 501, 502, 503), save_replay=save_rep
        )
    elif args.suite == "3v1_subham_vs_gen7":
        res = run_3v1_gauntlet(
            "Gen7-v5-Boss", gen7_spec, "subham_v14", "subham_v14", full_modes, seeds=(500, 501, 502, 503), save_replay=save_rep
        )
    elif args.suite == "3v1_subham_vs_hober":
        m_hober = full_modes if args.modes else [("none", 0), ("truce", 0), ("focus_fire", 0)]
        res = run_3v1_gauntlet(
            "#2 Hober Malloc", "hober_malloc", "subham_v14", "subham_v14", m_hober, seeds=(500, 501, 502), save_replay=save_rep
        )
    elif args.suite == "3v1_gen7_vs_hober":
        m_hober = full_modes if args.modes else [("none", 0), ("focus_fire", 0)]
        res = run_3v1_gauntlet(
            "#2 Hober Malloc", "hober_malloc", "Gen7-v5-Boss", gen7_spec, m_hober, seeds=(500, 501, 502), save_replay=save_rep
        )
    else:  # 3v1_gen8_vs_hober
        m_hober = full_modes if args.modes else [("none", 0), ("focus_fire", 0)]
        res = run_3v1_gauntlet(
            "#2 Hober Malloc", "hober_malloc", "PT-Gen8", gen8_spec, m_hober, seeds=(500, 501, 502), save_replay=save_rep
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
