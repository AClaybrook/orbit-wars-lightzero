"""Deep Midgame Gameplay Forensics: PT-Gen8 vs. #2 Hober Malloc (1800.1).

Plays official 2P matches on seeds 200-205 and measures:
  1. Phase-by-phase economy curves (planets, production, garrison, in-flight ships)
  2. Action tempo / APM, multi-source strikes, synchronized multi-fleet arrivals (±2t),
     send-fraction distribution, and Expand / Attack / Reinforce split
  3. Colonization & Target Profile (neutral prod, neutral cost, ROI, ETA, comets, snipes)
  4. Combat efficiency & waste (bounced/failed attacks, overkill ships, sun losses)
  5. Counterfactual Policy Divergence vs. #2 Hober Malloc's exact JAX policy on our
     own board states during Early (t=0..25) and Midgame Tipping Window (t=26..65).

Usage:
  python scripts/analyze_boss_gap.py --seeds 200 201 202 203 204 205 --out-json scratch/boss_forensics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from kaggle_environments import make

from orbit_wars_bot.coalition import _predict_order_target
from orbit_wars_bot.physics import (
    DEFAULT_SHIP_SPEED,
    build_comet_lookup,
    fleet_speed,
    predict_all_planet_tracks,
    predict_fleet_arrivals,
)
from orbit_wars_bot.planet_transformer import (
    PlanetTransformerAgent,
    extract_planet_features,
    match_action_to_target,
)
from scripts.distill_soft_kd import HoberSoftTeacher
from scripts.evaluate import resolve_agent


def _phase(t: int) -> str:
    if t <= 25:
        return "early_0_25"
    if t <= 65:
        return "mid_26_65"
    return "late_66_plus"


def _analyze_episode(
    steps: Any,
    config: Dict[str, Any],
    seed: int,
    teacher: Optional[HoberSoftTeacher],
    our_agent: PlanetTransformerAgent,
    our_pid: int = 0,
) -> Dict[str, Any]:
    """Extract exhaustive tick-by-tick telemetry and counterfactual policy divergence."""
    hober_pid = 1 - our_pid
    ang_vel = float(config.get("angular_velocity", 0.04))
    max_speed = float(config.get("ship_speed", DEFAULT_SHIP_SPEED))

    n_steps = len(steps)
    obs0 = steps[0][0].observation
    initial_by_id = {int(p[0]): list(p) for p in obs0.get("planets", [])}
    comet_ids: set[int] = set()

    # Checkpoints for economy curve
    sample_ticks = (0, 10, 20, 30, 40, 50, 60, 75, 90)
    snapshots: Dict[int, Dict[str, Any]] = {}

    # Per-player counters
    stats = {
        p: {
            "active_turns": 0,
            "total_turns": 0,
            "fleets_launched": 0,
            "ships_launched": 0,
            "multi_source_turns": 0,
            "send_fracs": [],
            "order_types": Counter(),  # neutral / enemy / reinforce / miss
            "order_types_by_phase": {
                "early_0_25": Counter(),
                "mid_26_65": Counter(),
                "late_66_plus": Counter(),
            },
            "neutral_captures": 0,
            "neutral_prod_sum": 0,
            "neutral_cost_sum": 0,
            "neutral_roi_sum": 0.0,
            "neutral_low_prod": 0,   # prod <= 2
            "neutral_mid_prod": 0,   # prod == 3
            "neutral_high_prod": 0,  # prod >= 4
            "comet_captures": 0,
            "enemy_captures": 0,
            "bounced_attacks": 0,
            "bounced_ships": 0,
            "overkill_ships": 0,
            "snipes": 0,  # captured a neutral weakened by opponent within 4 ticks
            "sync_arrivals": 0,  # arrival events where >=2 fleets from player hit same non-friendly target within ±2t
            "inflight_ratios_early": [],
            "inflight_ratios_mid": [],
            "inflight_ratios_late": [],
        }
        for p in (0, 1)
    }

    # Track neutral weakening events & arrival history per planet
    recent_neutral_dmg: Dict[int, Tuple[int, int]] = {}  # pid -> (tick, attacker)
    combat_arrivals_by_tgt: Dict[Tuple[int, int], List[int]] = defaultdict(list)  # (player, pid) -> [ticks]

    # Tipping point tracker: first tick where Hober leads in prod by >=4 or ships by >=25%
    tipping_prod_tick: Optional[int] = None
    tipping_ship_tick: Optional[int] = None

    # Counterfactual Hober policy divergence on our boards
    cf_by_phase: Dict[str, Counter] = {
        "early_0_25": Counter(),
        "mid_26_65": Counter(),
        "late_66_plus": Counter(),
    }
    cf_kl_by_phase: Dict[str, List[float]] = {
        "early_0_25": [],
        "mid_26_65": [],
        "late_66_plus": [],
    }
    notable_divergences: List[Dict[str, Any]] = []

    prev_planets: Dict[int, List[Any]] = {}
    prev_fleets: Dict[int, List[Any]] = {}

    for idx in range(n_steps):
        obs = steps[idx][0].observation
        t = int(obs.get("step", idx))
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        comets = obs.get("comets", [])
        for c in comets:
            for pid in c.get("planet_ids", ()):
                comet_ids.add(int(pid))
        by_id = {int(p[0]): p for p in planets}

        # 1. Economy snapshot
        econ = {}
        for p in (0, 1):
            p_pl = [row for row in planets if int(row[1]) == p]
            p_fl = [row for row in fleets if int(row[1]) == p]
            garrison = sum(int(row[5]) for row in p_pl)
            inflight = sum(int(row[6]) for row in p_fl)
            prod = sum(int(row[6]) for row in p_pl)
            tot = garrison + inflight
            econ[p] = {
                "planets": len(p_pl),
                "prod": prod,
                "garrison": garrison,
                "inflight": inflight,
                "total_ships": tot,
                "inflight_pct": round(100.0 * inflight / max(1, tot), 1),
            }
            if t > 0:
                ph = _phase(t)
                key = f"inflight_ratios_{ph.split('_')[0]}"
                stats[p][key].append(inflight / max(1.0, float(tot)))

        if t in sample_ticks:
            snapshots[t] = {
                "ours": econ[our_pid],
                "hober": econ[hober_pid],
                "prod_diff": econ[our_pid]["prod"] - econ[hober_pid]["prod"],
                "ship_diff": econ[our_pid]["total_ships"] - econ[hober_pid]["total_ships"],
            }

        if t >= 10:
            if tipping_prod_tick is None and (econ[hober_pid]["prod"] - econ[our_pid]["prod"]) >= 4:
                tipping_prod_tick = t
            if (
                tipping_ship_tick is None
                and econ[hober_pid]["total_ships"] > 0
                and (econ[hober_pid]["total_ships"] - econ[our_pid]["total_ships"])
                / max(1.0, float(econ[hober_pid]["total_ships"]))
                >= 0.20
            ):
                tipping_ship_tick = t

        # 2. Detect arrivals / captures / bounced attacks / overkill between idx-1 and idx
        if idx > 0:
            cur_fids = {int(f[0]) for f in fleets}
            # Fleets that disappeared since last tick either landed on a planet, hit sun, or went OOB
            vanished = [f for fid, f in prev_fleets.items() if fid not in cur_fids]
            # Match each vanished fleet to nearest planet at current tick (within r + speed + 1.5)
            landed_by_pid: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for vf in vanished:
                owner = int(vf[1])
                ang_f = float(vf[4])
                sh = int(vf[6])
                spd_f = fleet_speed(sh, max_speed)
                fx = float(vf[2]) + math.cos(ang_f) * spd_f
                fy = float(vf[3]) + math.sin(ang_f) * spd_f
                best_pid = None
                best_d = 999.0
                for pid, cp in by_id.items():
                    d = math.hypot(fx - float(cp[2]), fy - float(cp[3]))
                    if d <= float(cp[4]) + spd_f + 2.0 and d < best_d:
                        best_d = d
                        best_pid = pid
                if best_pid is not None:
                    landed_by_pid[best_pid].append((owner, sh))

            for pid, cp in by_id.items():
                if pid not in prev_planets:
                    continue
                pp = prev_planets[pid]
                old_own, new_own = int(pp[1]), int(cp[1])
                old_sh, new_sh = int(pp[5]), int(cp[5])
                prod = int(cp[6])
                landings = landed_by_pid.get(pid, [])
                if not landings:
                    continue

                # Record combat arrivals for synchronization tracking
                for owner, _ in landings:
                    if owner != old_own:
                        combat_arrivals_by_tgt[(owner, pid)].append(t)

                if old_own != new_own and new_own in (0, 1):
                    # Ownership flipped to new_own!
                    if old_own == -1:
                        stats[new_own]["neutral_captures"] += 1
                        stats[new_own]["neutral_prod_sum"] += prod
                        stats[new_own]["neutral_cost_sum"] += old_sh
                        stats[new_own]["neutral_roi_sum"] += prod / max(1.0, float(old_sh + 1))
                        if prod <= 2:
                            stats[new_own]["neutral_low_prod"] += 1
                        elif prod == 3:
                            stats[new_own]["neutral_mid_prod"] += 1
                        else:
                            stats[new_own]["neutral_high_prod"] += 1
                        if pid in comet_ids:
                            stats[new_own]["comet_captures"] += 1
                        # Check if opponent weakened this neutral recently (snipe!)
                        if pid in recent_neutral_dmg:
                            dmg_t, dmg_p = recent_neutral_dmg[pid]
                            if dmg_p != new_own and (t - dmg_t) <= 5:
                                stats[new_own]["snipes"] += 1
                        # Overkill on neutral: excess ships left above capture cost
                        inb_winner = sum(sh for o, sh in landings if o == new_own)
                        excess = max(0, inb_winner - (old_sh + 1))
                        stats[new_own]["overkill_ships"] += excess
                    else:
                        stats[new_own]["enemy_captures"] += 1
                else:
                    # Ownership did NOT flip — any non-owner fleet that landed BOUNCED!
                    for owner, sh in landings:
                        if owner != new_own and owner in (0, 1):
                            stats[owner]["bounced_attacks"] += 1
                            stats[owner]["bounced_ships"] += sh
                            if new_own == -1:
                                recent_neutral_dmg[pid] = (t, owner)

        # 3. Analyze outgoing actions submitted at tick t (found on steps[idx+1][p].action)
        if idx + 1 < n_steps:
            comet_lookup = build_comet_lookup(comets)
            tracks = predict_all_planet_tracks(
                planets, 26, t, ang_vel, initial_by_id, comet_lookup
            )
            ph = _phase(t)
            for p in (0, 1):
                acts = steps[idx + 1][p].action or []
                stats[p]["total_turns"] += 1
                if acts:
                    stats[p]["active_turns"] += 1
                    if len(acts) >= 2:
                        stats[p]["multi_source_turns"] += 1
                for o in acts:
                    if isinstance(o, dict):
                        src_id, ang, sh = int(o["from_planet_id"]), float(o["angle"]), int(o["ships"])
                    else:
                        src_id, ang, sh = int(o[0]), float(o[1]), int(o[2])
                    src_p = by_id.get(src_id)
                    if src_p is None:
                        continue
                    stats[p]["fleets_launched"] += 1
                    stats[p]["ships_launched"] += sh
                    stats[p]["send_fracs"].append(sh / max(1.0, float(src_p[5])))
                    tid = _predict_order_target(
                        src_p, ang, sh, planets, tracks, max_dt=26, max_speed=max_speed
                    )
                    if tid is None or tid not in by_id:
                        otype = "miss"
                    else:
                        town = int(by_id[tid][1])
                        if town == p:
                            otype = "reinforce"
                        elif town == -1:
                            otype = "neutral"
                        else:
                            otype = "enemy"
                    stats[p]["order_types"][otype] += 1
                    stats[p]["order_types_by_phase"][ph][otype] += 1

            # 4. Counterfactual Hober policy divergence on our exact board states (every 2nd tick up to t=72)
            if teacher is not None and 2 <= t <= 72 and (t % 2 == 0):
                our_obs = steps[idx][our_pid].observation
                enc = extract_planet_features(
                    our_obs,
                    player=our_pid,
                    max_speed=max_speed,
                )
                n = len(enc.planet_ids)
                if n > 0 and np.any(enc.owned_mask):
                    # Exact Hober soft probs on our board state
                    q_soft = teacher.query_soft_probs(our_obs, enc, config)[:n, : n + 1]
                    # Our actual executed actions mapped to target indices
                    our_acts = steps[idx + 1][our_pid].action or []
                    our_choice: Dict[int, int] = {}
                    for a in our_acts:
                        m = match_action_to_target(a, enc, max_speed=max_speed)
                        if m is not None:
                            src_i, dst_i, _ = m
                            our_choice[src_i] = dst_i + 1  # 0 = HOLD, 1..n = target idx + 1

                    # Exact 8x D4 TTA predicted soft probs from PT-Gen8
                    p_probs, _, _ = our_agent.predict_probs_and_value(enc)

                    for i in range(n):
                        if not enc.owned_mask[i]:
                            continue
                        src_pid = int(enc.planet_ids[i])
                        src_sh = int(by_id[src_pid][5])
                        if src_sh < 5:
                            continue
                        q_i = q_soft[i]
                        p_i = np.clip(p_probs[i], 1e-6, 1.0)
                        q_safe = np.clip(q_i, 1e-6, 1.0)
                        kl = float(np.sum(q_i * (np.log(q_safe) - np.log(p_i))))
                        cf_kl_by_phase[ph].append(kl)

                        h_best = int(np.argmax(q_i))
                        o_act = our_choice.get(i, 0)

                        def _cls_label(c_idx: int) -> str:
                            if c_idx == 0:
                                return "HOLD"
                            tpid = int(enc.planet_ids[c_idx - 1])
                            to = int(by_id[tpid][1])
                            if to == our_pid:
                                return "REINFORCE"
                            if to == -1:
                                return "NEUTRAL"
                            return "ENEMY"

                        h_lbl = _cls_label(h_best)
                        o_lbl = _cls_label(o_act)
                        cf_by_phase[ph]["total_decisions"] += 1
                        if o_act == h_best:
                            cf_by_phase[ph]["exact_match"] += 1
                        elif o_act == 0 and h_best > 0:
                            cf_by_phase[ph][f"ours_HOLD_hober_{h_lbl}"] += 1
                        elif o_act > 0 and h_best == 0:
                            cf_by_phase[ph][f"ours_{o_lbl}_hober_HOLD"] += 1
                        else:
                            cf_by_phase[ph][f"diff_tgt_ours_{o_lbl}_hober_{h_lbl}"] += 1

                        if kl >= 1.2 and src_sh >= 15 and 18 <= t <= 62 and len(notable_divergences) < 12:
                            h_pid = int(enc.planet_ids[h_best - 1]) if h_best > 0 else -1
                            o_pid = int(enc.planet_ids[o_act - 1]) if o_act > 0 else -1
                            notable_divergences.append(
                                {
                                    "seed": seed,
                                    "t": t,
                                    "src_pid": src_pid,
                                    "src_ships": src_sh,
                                    "kl": round(kl, 2),
                                    "ours": f"{o_lbl}({o_pid})",
                                    "hober": f"{h_lbl}({h_pid}, p={q_i[h_best]:.2f})",
                                    "our_prod": econ[our_pid]["prod"],
                                    "hober_prod": econ[hober_pid]["prod"],
                                }
                            )

        prev_planets = {int(p[0]): list(p) for p in planets}
        prev_fleets = {int(f[0]): list(f) for f in fleets}

    # Count synchronized multi-fleet combat arrivals (2+ fleets from same player hitting same target within ±2 ticks)
    for (p, _pid), ticks in combat_arrivals_by_tgt.items():
        if len(ticks) < 2:
            continue
        ticks.sort()
        i = 0
        while i < len(ticks):
            j = i + 1
            while j < len(ticks) and ticks[j] - ticks[i] <= 2:
                j += 1
            if j - i >= 2:
                stats[p]["sync_arrivals"] += 1
                i = j
            else:
                i += 1

    # Summarize per-player metrics
    def _summarize_player(p: int) -> Dict[str, Any]:
        s = stats[p]
        nc = max(1, s["neutral_captures"])
        return {
            "active_turn_pct": round(100.0 * s["active_turns"] / max(1, s["total_turns"]), 1),
            "fleets_launched": s["fleets_launched"],
            "launches_per_active_turn": round(s["fleets_launched"] / max(1, s["active_turns"]), 2),
            "multi_source_turns": s["multi_source_turns"],
            "avg_send_frac": round(float(np.mean(s["send_fracs"])) if s["send_fracs"] else 0.0, 3),
            "order_types": dict(s["order_types"]),
            "order_types_by_phase": {k: dict(v) for k, v in s["order_types_by_phase"].items()},
            "neutral_captures": s["neutral_captures"],
            "avg_neutral_prod": round(s["neutral_prod_sum"] / nc, 2),
            "avg_neutral_cost": round(s["neutral_cost_sum"] / nc, 2),
            "avg_neutral_roi": round(s["neutral_roi_sum"] / nc, 3),
            "neutral_prod_buckets": {
                "low_le2": s["neutral_low_prod"],
                "mid_3": s["neutral_mid_prod"],
                "high_ge4": s["neutral_high_prod"],
            },
            "comet_captures": s["comet_captures"],
            "enemy_captures": s["enemy_captures"],
            "bounced_attacks": s["bounced_attacks"],
            "bounced_ships": s["bounced_ships"],
            "overkill_ships": s["overkill_ships"],
            "snipes": s["snipes"],
            "sync_arrivals": s["sync_arrivals"],
            "avg_inflight_pct": {
                "early": round(100.0 * float(np.mean(s["inflight_ratios_early"])) if s["inflight_ratios_early"] else 0.0, 1),
                "mid": round(100.0 * float(np.mean(s["inflight_ratios_mid"])) if s["inflight_ratios_mid"] else 0.0, 1),
                "late": round(100.0 * float(np.mean(s["inflight_ratios_late"])) if s["inflight_ratios_late"] else 0.0, 1),
            },
        }

    return {
        "seed": seed,
        "ticks": n_steps - 1,
        "tipping_prod_tick": tipping_prod_tick,
        "tipping_ship_tick": tipping_ship_tick,
        "snapshots": snapshots,
        "ours": _summarize_player(our_pid),
        "hober": _summarize_player(hober_pid),
        "counterfactual_by_phase": {k: dict(v) for k, v in cf_by_phase.items()},
        "counterfactual_mean_kl": {
            k: round(float(np.mean(v)), 3) if v else None for k, v in cf_kl_by_phase.items()
        },
        "notable_divergences": notable_divergences[:6],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202, 203, 204, 205])
    parser.add_argument("--ckpt", type=str, default="checkpoints/planet_transformer_gen8_2p_softkd.pt")
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--no-counterfactual", action="store_true")
    parser.add_argument("--out-json", type=str, default="scratch/boss_forensics.json")
    args = parser.parse_args()

    from orbit_wars_bot.planet_transformer import EnsemblePlanetTransformerAgent

    t_start = time.perf_counter()
    teacher = None if args.no_counterfactual else HoberSoftTeacher(rotation_tta=True, opponent_tta=False)
    ckpt_list = [p.strip() for p in args.ckpt.split(",") if p.strip()]
    if len(ckpt_list) > 1:
        our_agent: Any = EnsemblePlanetTransformerAgent(
            ckpt_list, weights=args.weights, use_d4_tta=True, force_all_in=True
        )
    else:
        our_agent = PlanetTransformerAgent(ckpt_list[0], use_d4_tta=True, force_all_in=True)

    episodes = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        hober_ag = resolve_agent("hober_malloc")
        if len(ckpt_list) > 1:
            ep_ag = EnsemblePlanetTransformerAgent(
                ckpt_list, weights=args.weights, use_d4_tta=True, force_all_in=True
            )
            pt_ag = lambda obs, cfg, _ag=ep_ag: _ag.act(obs, cfg)
        else:
            pt_ag = resolve_agent(f"pt_d4_allin:{ckpt_list[0]}")
        env = make(
            "orbit_wars",
            configuration={"episodeSteps": 500, "seed": seed, "actTimeout": 5.0},
            debug=False,
        )
        steps = env.run([pt_ag, hober_ag])
        ep = _analyze_episode(steps, env.configuration, seed, teacher, our_agent, our_pid=0)
        print(
            f"seed={seed} ticks={ep['ticks']} tip_prod=t{ep['tipping_prod_tick']} "
            f"tip_ship=t{ep['tipping_ship_tick']} "
            f"bounced={ep['ours']['bounced_ships']}v{ep['hober']['bounced_ships']} "
            f"sync={ep['ours']['sync_arrivals']}v{ep['hober']['sync_arrivals']} "
            f"comets={ep['ours']['comet_captures']}v{ep['hober']['comet_captures']} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        episodes.append(ep)

    # Aggregate across all seeds
    def _agg_player(side: str) -> Dict[str, Any]:
        eps = [e[side] for e in episodes]
        ot = Counter()
        ot_ph = {ph: Counter() for ph in ("early_0_25", "mid_26_65", "late_66_plus")}
        for e in eps:
            ot.update(e["order_types"])
            for ph in ot_ph:
                ot_ph[ph].update(e["order_types_by_phase"][ph])
        return {
            "active_turn_pct": round(float(np.mean([e["active_turn_pct"] for e in eps])), 1),
            "avg_fleets_launched": round(float(np.mean([e["fleets_launched"] for e in eps])), 1),
            "launches_per_active_turn": round(float(np.mean([e["launches_per_active_turn"] for e in eps])), 2),
            "avg_multi_source_turns": round(float(np.mean([e["multi_source_turns"] for e in eps])), 1),
            "avg_send_frac": round(float(np.mean([e["avg_send_frac"] for e in eps])), 3),
            "order_types_total": dict(ot),
            "order_types_by_phase": {ph: dict(c) for ph, c in ot_ph.items()},
            "avg_neutral_captures": round(float(np.mean([e["neutral_captures"] for e in eps])), 2),
            "avg_neutral_prod": round(float(np.mean([e["avg_neutral_prod"] for e in eps])), 2),
            "avg_neutral_cost": round(float(np.mean([e["avg_neutral_cost"] for e in eps])), 2),
            "avg_neutral_roi": round(float(np.mean([e["avg_neutral_roi"] for e in eps])), 3),
            "neutral_prod_buckets_total": {
                k: sum(e["neutral_prod_buckets"][k] for e in eps)
                for k in ("low_le2", "mid_3", "high_ge4")
            },
            "total_comet_captures": sum(e["comet_captures"] for e in eps),
            "total_enemy_captures": sum(e["enemy_captures"] for e in eps),
            "total_bounced_attacks": sum(e["bounced_attacks"] for e in eps),
            "total_bounced_ships": sum(e["bounced_ships"] for e in eps),
            "total_overkill_ships": sum(e["overkill_ships"] for e in eps),
            "total_snipes": sum(e["snipes"] for e in eps),
            "total_sync_arrivals": sum(e["sync_arrivals"] for e in eps),
            "avg_inflight_pct": {
                k: round(float(np.mean([e["avg_inflight_pct"][k] for e in eps])), 1)
                for k in ("early", "mid", "late")
            },
        }

    # Economy curve averages across ticks
    curve = {}
    for tk in (0, 10, 20, 30, 40, 50, 60, 75, 90):
        snaps = [e["snapshots"][tk] for e in episodes if tk in e["snapshots"]]
        if snaps:
            curve[str(tk)] = {
                "n": len(snaps),
                "our_planets": round(float(np.mean([s["ours"]["planets"] for s in snaps])), 1),
                "hober_planets": round(float(np.mean([s["hober"]["planets"] for s in snaps])), 1),
                "our_prod": round(float(np.mean([s["ours"]["prod"] for s in snaps])), 1),
                "hober_prod": round(float(np.mean([s["hober"]["prod"] for s in snaps])), 1),
                "prod_diff": round(float(np.mean([s["prod_diff"] for s in snaps])), 1),
                "our_ships": round(float(np.mean([s["ours"]["total_ships"] for s in snaps])), 1),
                "hober_ships": round(float(np.mean([s["hober"]["total_ships"] for s in snaps])), 1),
                "ship_diff": round(float(np.mean([s["ship_diff"] for s in snaps])), 1),
                "our_inflight_pct": round(float(np.mean([s["ours"]["inflight_pct"] for s in snaps])), 1),
                "hober_inflight_pct": round(float(np.mean([s["hober"]["inflight_pct"] for s in snaps])), 1),
            }

    cf_agg = {ph: Counter() for ph in ("early_0_25", "mid_26_65", "late_66_plus")}
    for e in episodes:
        for ph in cf_agg:
            cf_agg[ph].update(e["counterfactual_by_phase"][ph])

    summary = {
        "seeds": args.seeds,
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
        "avg_ticks": round(float(np.mean([e["ticks"] for e in episodes])), 1),
        "avg_tipping_prod_tick": round(
            float(np.mean([e["tipping_prod_tick"] for e in episodes if e["tipping_prod_tick"] is not None])), 1
        ),
        "avg_tipping_ship_tick": round(
            float(np.mean([e["tipping_ship_tick"] for e in episodes if e["tipping_ship_tick"] is not None])), 1
        ),
        "economy_curve": curve,
        "ours_agg": _agg_player("ours"),
        "hober_agg": _agg_player("hober"),
        "counterfactual_agg": {ph: dict(c) for ph, c in cf_agg.items()},
        "episodes": episodes,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} in {summary['elapsed_sec']}s")
    print("=== ECONOMY CURVE ===")
    for tk, row in curve.items():
        print(
            f"  t={int(tk):2d} (n={row['n']}): "
            f"Pl {row['our_planets']:4.1f} vs {row['hober_planets']:4.1f} | "
            f"Prod {row['our_prod']:4.1f} vs {row['hober_prod']:4.1f} ({row['prod_diff']:+5.1f}) | "
            f"Ships {row['our_ships']:5.1f} vs {row['hober_ships']:5.1f} ({row['ship_diff']:+6.1f}) | "
            f"InFlight% {row['our_inflight_pct']:4.1f}% vs {row['hober_inflight_pct']:4.1f}%"
        )
    print("=== OURS vs HOBER AGG ===")
    print(json.dumps({"ours": summary["ours_agg"], "hober": summary["hober_agg"]}, indent=2))
    print("=== COUNTERFACTUAL POLICY DIVERGENCE ===")
    print(json.dumps(summary["counterfactual_agg"], indent=2))


if __name__ == "__main__":
    main()
