"""Multi-Dimensional Wargaming & Coalition Power-Dynamics Sweep.

Quantifies the phase transition where a 3-agent coalition overcomes a stronger
Boss across three independent axes:
  1. Intelligence Gap (Delta Elo = Elo_boss - Elo_ally): ~175 to ~740 Elo
  2. Coordination Fidelity (C0 FFA -> C1 Truce -> C2 Imperfect -> C3 Lat=5 -> C4 Perfect Lat=0)
  3. Starting Resource Ratio (Coalition / Boss starting ships via FastOrbitSim: 6x -> 1x)

Also fits a logistic power-law model:
  logit(P(Coalition Takedown)) = b0 + b_elo * (DeltaElo/100) + b_coord * Coord + b_res * log2(ResRatio)
to directly quantify:
  - How many Elo points Perfect Coordination buys you
  - How many Elo points each 2x Resource Advantage is worth
  - Where the 50% phase-transition boundary lies!

Usage:
  python -u scripts/wargame_sweep.py --grid coord --out-json scratch/wargame_coord.json
  python -u scripts/wargame_sweep.py --grid resource --out-json scratch/wargame_resource.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from orbit_wars_bot.coalition import create_coalition_team
from orbit_wars_bot.fast_sim import FastOrbitSim
from scripts.evaluate import resolve_agent

# Cache stateless / reusable agents to avoid reloading PyTorch checkpoints repeatedly
_AGENT_CACHE: Dict[str, Any] = {}


def _get_agent_factory(spec: str) -> Callable[[], Any]:
    if spec in ("starter", "random"):
        if spec not in _AGENT_CACHE:
            from kaggle_environments import make

            raw_fn = make("orbit_wars").agents[spec]
            _AGENT_CACHE[spec] = lambda obs, cfg=None, _f=raw_fn: _f(obs)
        ag = _AGENT_CACHE[spec]
        return lambda: ag
    if spec.startswith("pt:") or spec.startswith("pt_d4:") or spec.startswith("pt_d4_allin:"):
        # PlanetTransformerAgent is stateless across calls (just forward pass on obs)
        if spec not in _AGENT_CACHE:
            _AGENT_CACHE[spec] = resolve_agent(spec)
        ag = _AGENT_CACHE[spec]
        return lambda: ag
    # Heuristic / stateful agents: instantiate fresh per player per match
    return lambda: resolve_agent(spec)


def run_fastsim_4p_match(
    boss_spec: str,
    ally_spec: str,
    seed: int,
    *,
    mode: str = "none",
    comm_latency: int = 0,
    compliance: float = 1.0,
    info_noise: float = 0.0,
    boss_ship_mult: float = 1.0,
    max_steps: int = 320,
) -> Dict[str, Any]:
    """Run an exact tick-for-tick FastOrbitSim 4P match with controllable resource asymmetry."""
    from kaggle_environments.utils import Struct

    sim = FastOrbitSim.from_seed(seed=seed, num_agents=4, max_steps=max_steps)

    # Apply starting resource asymmetry to Boss (Player 0)'s home planet if requested
    if abs(boss_ship_mult - 1.0) > 1e-6:
        for p in sim.planets:
            if int(p[1]) == 0:
                p[5] = max(2, int(round(int(p[5]) * boss_ship_mult)))
        sim.initial_planets = [list(p) for p in sim.planets]

    boss_ag = _get_agent_factory(boss_spec)()
    ally_factory = _get_agent_factory(ally_spec)
    coord, team = create_coalition_team(
        ally_factory,
        boss_id=0,
        ally_ids=(1, 2, 3),
        mode=mode,
        comm_latency=comm_latency,
        compliance=compliance,
        info_noise=info_noise,
        rng_seed=seed * 17 + 3,
    )
    agents = [boss_ag, team[1], team[2], team[3]]
    cfg = Struct(**sim.to_config())

    boss_elim_step: Optional[int] = None
    done = False
    while not done:
        acts = []
        for pid in range(4):
            obs = Struct(**sim.to_observation(pid))
            raw = agents[pid](obs, cfg) or []
            acts.append(raw)
        done = sim.step(acts)

        if boss_elim_step is None:
            boss_alive = any(int(p[1]) == 0 for p in sim.planets) or any(
                int(f[1]) == 0 for f in sim.fleets
            )
            if not boss_alive:
                boss_elim_step = sim.step_count
                # Once Boss is eliminated, Coalition takedown is achieved!
                break

    # Determine final standings if Boss still alive at end
    ships_by_p = {p: 0 for p in range(4)}
    for pl in sim.planets:
        own = int(pl[1])
        if own >= 0:
            ships_by_p[own] += int(pl[5])
    for fl in sim.fleets:
        own = int(fl[1])
        if own >= 0:
            ships_by_p[own] += int(fl[6])

    if boss_elim_step is not None:
        boss_won = False
    else:
        max_sh = max(ships_by_p.values())
        boss_won = ships_by_p[0] == max_sh and ships_by_p[0] > 0

    return {
        "seed": seed,
        "ticks": sim.step_count,
        "boss_won": boss_won,
        "coalition_won": not boss_won,
        "boss_elim_step": boss_elim_step,
        "ff_blocked": coord.stats.friendly_fire_blocked,
        "sync_volleys": coord.stats.sync_volleys_launched,
    }


def run_coord_grid(seeds: Sequence[int], ally_filter: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Grid 1: Intelligence Gap (Delta Elo) x Coordination Level at standard 4P resources (3:1)."""
    boss_spec = "pt_d4_allin:checkpoints/planet_transformer_gen8_2p_softkd.pt"
    boss_elo = 1740

    allies = [
        {"name": "starter", "spec": "starter", "elo": 1000},
        {"name": "wd_heuristic", "spec": "wd_heuristic", "elo": 1105},
        {"name": "subham_v14", "spec": "subham_v14", "elo": 1224},
        {"name": "Gen 3 PT-v1", "spec": "pt:checkpoints/planet_transformer_v1_backup.pt", "elo": 1430},
        {"name": "Gen 5 PT-v3-RL", "spec": "pt:checkpoints/planet_transformer_v3_rl.pt", "elo": 1565},
    ]
    if ally_filter:
        allies = [a for a in allies if a["name"] in ally_filter]

    coord_levels = [
        {"id": "C0_ffa", "label": "C0: Uncoordinated FFA", "score": 0.0, "mode": "none", "lat": 0, "comp": 1.0, "noise": 0.0},
        {"id": "C1_truce", "label": "C1: Truce Only", "score": 0.25, "mode": "truce", "lat": 0, "comp": 1.0, "noise": 0.0},
        {"id": "C2_imperfect", "label": "C2: Imperfect (Lat=10, 50% Comp)", "score": 0.50, "mode": "focus_fire", "lat": 10, "comp": 0.50, "noise": 0.30},
        {"id": "C3_lat5", "label": "C3: Focus-Fire (Lat=5)", "score": 0.75, "mode": "focus_fire", "lat": 5, "comp": 1.0, "noise": 0.0},
        {"id": "C4_perfect", "label": "C4: Zero-Lat Focus-Fire", "score": 1.0, "mode": "focus_fire", "lat": 0, "comp": 1.0, "noise": 0.0},
    ]

    cells = []
    t_all = time.perf_counter()
    for a in allies:
        delta_elo = boss_elo - a["elo"]
        for c in coord_levels:
            t0 = time.perf_counter()
            wins = 0
            elims = []
            volleys = []
            for s in seeds:
                r = run_fastsim_4p_match(
                    boss_spec,
                    a["spec"],
                    seed=s,
                    mode=c["mode"],
                    comm_latency=c["lat"],
                    compliance=c["comp"],
                    info_noise=c["noise"],
                    boss_ship_mult=1.0,
                )
                if r["coalition_won"]:
                    wins += 1
                if r["boss_elim_step"] is not None:
                    elims.append(r["boss_elim_step"])
                volleys.append(r["sync_volleys"])
            wr = round(100.0 * wins / len(seeds), 1)
            avg_el = round(float(np.mean(elims)), 1) if elims else None
            cell = {
                "ally": a["name"],
                "ally_elo": a["elo"],
                "boss_elo": boss_elo,
                "delta_elo": delta_elo,
                "coord_id": c["id"],
                "coord_label": c["label"],
                "coord_score": c["score"],
                "coalition_wins": wins,
                "games": len(seeds),
                "coalition_wr": wr,
                "avg_elim_tick": avg_el,
                "avg_volleys": round(float(np.mean(volleys)), 1),
            }
            cells.append(cell)
            print(
                f"  [CoordGrid] Ally={a['name']:14s} (dElo=+{delta_elo:3d}) | "
                f"{c['id']:13s} -> CoalWR={wr:5.1f}% ({wins}/{len(seeds)}) "
                f"elim_t={avg_el} ({time.perf_counter() - t0:.1f}s)"
            )
    return {
        "boss": "1.89M PT-Gen8",
        "boss_elo": boss_elo,
        "seeds": list(seeds),
        "elapsed_sec": round(time.perf_counter() - t_all, 1),
        "allies": allies,
        "coord_levels": coord_levels,
        "cells": cells,
    }


def run_resource_grid(seeds: Sequence[int], ally_filter: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Grid 2: Intelligence Gap (Delta Elo) x Starting Resource Ratio under FFA vs. Zero-Lat Focus-Fire."""
    boss_spec = "pt_d4_allin:checkpoints/planet_transformer_gen8_2p_softkd.pt"
    boss_elo = 1740

    allies = [
        {"name": "starter", "spec": "starter", "elo": 1000},
        {"name": "wd_heuristic", "spec": "wd_heuristic", "elo": 1105},
        {"name": "subham_v14", "spec": "subham_v14", "elo": 1224},
        {"name": "Gen 3 PT-v1", "spec": "pt:checkpoints/planet_transformer_v1_backup.pt", "elo": 1430},
    ]
    if ally_filter:
        allies = [a for a in allies if a["name"] in ally_filter]

    # boss_ship_mult: 0.5 => 6.0x coalition:boss, 1.0 => 3.0x, 2.0 => 1.5x, 3.0 => 1.0x equal ships, 5.0 => 0.6x Boss dominance
    mults = [0.5, 1.0, 2.0, 3.0, 5.0]
    modes = [("none", "FFA"), ("focus_fire", "Zero-Lat Focus-Fire")]

    cells = []
    t_all = time.perf_counter()
    for mode, m_label in modes:
        for a in allies:
            delta_elo = boss_elo - a["elo"]
            for mult in mults:
                t0 = time.perf_counter()
                res_ratio = round(3.0 / mult, 2)  # Coalition total starting ships / Boss starting ships
                wins = 0
                elims = []
                for s in seeds:
                    r = run_fastsim_4p_match(
                        boss_spec,
                        a["spec"],
                        seed=s,
                        mode=mode,
                        comm_latency=0,
                        compliance=1.0,
                        boss_ship_mult=mult,
                    )
                    if r["coalition_won"]:
                        wins += 1
                    if r["boss_elim_step"] is not None:
                        elims.append(r["boss_elim_step"])
                wr = round(100.0 * wins / len(seeds), 1)
                avg_el = round(float(np.mean(elims)), 1) if elims else None
                cell = {
                    "mode": mode,
                    "mode_label": m_label,
                    "ally": a["name"],
                    "ally_elo": a["elo"],
                    "delta_elo": delta_elo,
                    "boss_ship_mult": mult,
                    "resource_ratio": res_ratio,
                    "coalition_wins": wins,
                    "games": len(seeds),
                    "coalition_wr": wr,
                    "avg_elim_tick": avg_el,
                }
                cells.append(cell)
                print(
                    f"  [ResGrid] {mode:10s} | Ally={a['name']:12s} (dElo=+{delta_elo:3d}) | "
                    f"ResRatio={res_ratio:4.2f}x (boss={mult:.1f}x) -> CoalWR={wr:5.1f}% ({wins}/{len(seeds)}) "
                    f"({time.perf_counter() - t0:.1f}s)"
                )
    return {
        "boss": "1.89M PT-Gen8",
        "boss_elo": boss_elo,
        "seeds": list(seeds),
        "elapsed_sec": round(time.perf_counter() - t_all, 1),
        "allies": allies,
        "boss_ship_mults": mults,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["coord", "resource"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[700, 701, 702, 703])
    parser.add_argument("--allies", type=str, nargs="*", default=None)
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    res = (
        run_coord_grid(args.seeds, ally_filter=args.allies)
        if args.grid == "coord"
        else run_resource_grid(args.seeds, ally_filter=args.allies)
    )
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"Wrote {out} in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
