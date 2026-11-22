"""16 discrete tactical macro-actions for Orbit Wars LightZero agents.

Maps high-level Galcon / Orbit Wars strategic directives to exact, sun-safe,
swept-pair-verified ``[from_planet_id, angle, ships]`` launch orders using
``orbit_wars_bot.physics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orbit_wars_bot.physics import (
    DEFAULT_EPISODE_STEPS,
    DEFAULT_SHIP_SPEED,
    FleetArrival,
    InterceptSolution,
    PlanetFutureState,
    build_comet_lookup,
    distance,
    fleet_speed,
    is_orbiting_planet,
    predict_all_planet_tracks,
    predict_fleet_arrivals,
    simulate_planet_future,
    solve_intercept,
)

NUM_MACRO_ACTIONS: int = 16


class MacroAction(IntEnum):
    HOLD_ACCUMULATE = 0
    EXPAND_BEST_ROI_SINGLE = 1
    EXPAND_BEST_ROI_MULTI = 2
    EXPAND_HIGH_PROD = 3
    EXPAND_ORBITING = 4
    EXPAND_COMET_SNIPE = 5
    DEFEND_URGENT = 6
    COUNTER_EVACUATE = 7
    VULTURE_SNIPE = 8
    ATTACK_WEAKEST_ENEMY = 9
    ATTACK_HIGH_PROD_ENEMY = 10
    ATTACK_CLOSEST_ENEMY = 11
    COORDINATED_ALL_IN = 12
    FEED_FRONTLINE = 13
    BALANCED_MACRO = 14
    AGGRESSIVE_SKIRMISH = 15


@dataclass
class TacticalContext:
    """Precomputed tactical view of a state for fast macro-action execution."""

    player: int
    step: int
    max_steps: int
    remaining_steps: int
    angular_velocity: float
    ship_speed: float
    planets: List[List[Any]]
    planet_by_id: Dict[int, List[Any]]
    initial_by_id: Dict[int, Sequence[Any]]
    comet_ids: set
    comet_lookup: Dict[int, Tuple[List[List[float]], int]]
    tracks: Dict[int, List[Optional[Tuple[float, float]]]]
    arrivals: Dict[int, List[FleetArrival]]
    futures: Dict[int, PlanetFutureState]
    my_planets: List[List[Any]]
    enemy_planets: List[List[Any]]
    neutral_planets: List[List[Any]]
    surplus: Dict[int, int]
    threatened_my: List[Tuple[List[Any], PlanetFutureState]]
    doomed_my: List[Tuple[List[Any], PlanetFutureState]]
    vulture_targets: List[Tuple[List[Any], int, int]]  # (planet, steal_eta, enemy_ships_after)
    enemy_centroid: Optional[Tuple[float, float]]
    my_total_ships: int
    enemy_total_ships: int
    _intercept_cache: Dict[Tuple[int, int, int], Optional[InterceptSolution]]

    def intercept(self, src: Sequence[Any], dst: Sequence[Any], ships: int, max_eta: int = 24) -> Optional[InterceptSolution]:
        # Quantize ships slightly for cache reuse while preserving exact verification for chosen send
        if ships <= 0:
            return None
        key = (int(src[0]), int(dst[0]), int(ships))
        if key in self._intercept_cache:
            return self._intercept_cache[key]
        sol = solve_intercept(
            src,
            dst,
            ships,
            self.planets,
            self.step,
            self.angular_velocity,
            self.initial_by_id,
            comet_lookup=self.comet_lookup,
            tracks=self.tracks,
            max_eta=min(max_eta, max(1, self.remaining_steps)),
            max_speed=self.ship_speed,
        )
        self._intercept_cache[key] = sol
        return sol


def _extract_state(state: Any, player: int) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]], List[Dict[str, Any]], List[int], float, int, int, float]:
    if hasattr(state, "step_count"):
        return (
            state.planets,
            state.initial_planets,
            state.fleets,
            state.comets,
            state.comet_planet_ids,
            float(state.angular_velocity),
            int(state.step_count),
            int(state.max_steps),
            float(state.ship_speed),
        )
    if isinstance(state, dict):
        planets = state.get("planets", [])
        init_p = state.get("initial_planets") or planets
        return (
            planets,
            init_p,
            state.get("fleets", []),
            state.get("comets", []) or [],
            state.get("comet_planet_ids", []) or [],
            float(state.get("angular_velocity", 0.0)),
            int(state.get("step", 0)),
            DEFAULT_EPISODE_STEPS,
            DEFAULT_SHIP_SPEED,
        )
    planets = getattr(state, "planets", [])
    init_p = getattr(state, "initial_planets", None) or planets
    return (
        planets,
        init_p,
        getattr(state, "fleets", []),
        getattr(state, "comets", []) or [],
        getattr(state, "comet_planet_ids", []) or [],
        float(getattr(state, "angular_velocity", 0.0)),
        int(getattr(state, "step", 0)),
        DEFAULT_EPISODE_STEPS,
        DEFAULT_SHIP_SPEED,
    )


def build_tactical_context(state: Any, player: int, horizon: int = 24) -> TacticalContext:
    """Build a ``TacticalContext`` from a ``FastOrbitSim`` or Kaggle ``obs``."""
    planets, init_planets, fleets, comets, comet_pids, ang_vel, step, max_steps, ship_speed = _extract_state(state, player)
    remaining = max(1, max_steps - step)
    eff_horizon = min(horizon, remaining)

    planet_by_id = {int(p[0]): p for p in planets}
    initial_by_id = {int(p[0]): p for p in init_planets}
    comet_ids = set(int(pid) for pid in comet_pids)
    comet_lookup = build_comet_lookup(comets)

    tracks = predict_all_planet_tracks(
        planets,
        eff_horizon,
        step,
        ang_vel,
        initial_by_id,
        comet_lookup,
    )
    arrivals = predict_fleet_arrivals(
        fleets,
        planets,
        step,
        ang_vel,
        initial_by_id,
        comet_lookup=comet_lookup,
        tracks=tracks,
        max_horizon=eff_horizon,
        max_speed=ship_speed,
    )

    futures: Dict[int, PlanetFutureState] = {}
    my_planets: List[List[Any]] = []
    enemy_planets: List[List[Any]] = []
    neutral_planets: List[List[Any]] = []
    surplus: Dict[int, int] = {}
    threatened_my: List[Tuple[List[Any], PlanetFutureState]] = []
    doomed_my: List[Tuple[List[Any], PlanetFutureState]] = []
    vulture_targets: List[Tuple[List[Any], int, int]] = []

    my_total_ships = 0
    enemy_total_ships = 0

    for p in planets:
        pid = int(p[0])
        owner = int(p[1])
        ships = int(p[5])
        fut = simulate_planet_future(p, arrivals.get(pid, ()), horizon=eff_horizon)
        futures[pid] = fut

        if owner == player:
            my_planets.append(p)
            my_total_ships += ships
            # Safe surplus = ships we can send now without losing the planet to incoming fleets
            if fut.will_be_captured_by is not None and fut.will_be_captured_by != player:
                surplus[pid] = 0
                threatened_my.append((p, fut))
                if fut.capture_tick is not None and fut.capture_tick <= 6:
                    doomed_my.append((p, fut))
            else:
                # Keep a small reserve proportional to incoming threat
                safe = max(0, min(ships, fut.min_owner_ships - 1))
                surplus[pid] = safe
        elif owner == -1:
            neutral_planets.append(p)
            # Check if an enemy fleet is about to capture or weaken this neutral
            if fut.will_be_captured_by is not None and fut.will_be_captured_by != player and fut.capture_tick is not None:
                # Ships the enemy will have right after capture
                _, post_ships = fut.timeline[fut.capture_tick]
                vulture_targets.append((p, fut.capture_tick, post_ships))
        else:
            enemy_planets.append(p)
            enemy_total_ships += ships

    for f in fleets:
        if int(f[1]) == player:
            my_total_ships += int(f[6])
        else:
            enemy_total_ships += int(f[6])

    if enemy_planets:
        ex = sum(float(p[2]) for p in enemy_planets) / len(enemy_planets)
        ey = sum(float(p[3]) for p in enemy_planets) / len(enemy_planets)
        enemy_centroid: Optional[Tuple[float, float]] = (ex, ey)
    else:
        enemy_centroid = None

    return TacticalContext(
        player=player,
        step=step,
        max_steps=max_steps,
        remaining_steps=remaining,
        angular_velocity=ang_vel,
        ship_speed=ship_speed,
        planets=planets,
        planet_by_id=planet_by_id,
        initial_by_id=initial_by_id,
        comet_ids=comet_ids,
        comet_lookup=comet_lookup,
        tracks=tracks,
        arrivals=arrivals,
        futures=futures,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        surplus=surplus,
        threatened_my=threatened_my,
        doomed_my=doomed_my,
        vulture_targets=vulture_targets,
        enemy_centroid=enemy_centroid,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        _intercept_cache={},
    )


def _approx_eta(src: Sequence[Any], dst: Sequence[Any], ships: int, max_speed: float) -> int:
    d = max(0.0, distance((src[2], src[3]), (dst[2], dst[3])) - float(src[4]) - float(dst[4]))
    spd = fleet_speed(max(1, ships), max_speed)
    return max(1, int(math.ceil(d / spd)))


def _needed_to_take(ctx: TacticalContext, target: Sequence[Any], eta: int, extra_buffer: int = 1) -> int:
    pid = int(target[0])
    fut = ctx.futures[pid]
    t_idx = min(eta, fut.horizon)
    owner_at_t, ships_at_t = fut.timeline[t_idx]
    if owner_at_t == ctx.player:
        return 0
    return int(ships_at_t) + extra_buffer


def _comet_remaining_life(ctx: TacticalContext, pid: int) -> int:
    if pid not in ctx.comet_lookup:
        return ctx.remaining_steps
    path, cur_idx = ctx.comet_lookup[pid]
    return max(0, len(path) - 1 - cur_idx)


def _score_neutral_roi(ctx: TacticalContext, src: Sequence[Any], target: Sequence[Any], avail_ships: int) -> float:
    pid = int(target[0])
    if pid in ctx.comet_ids:
        life = _comet_remaining_life(ctx, pid)
        if life <= 6:
            return -1e9
    est_send = max(2, min(avail_ships, int(target[5]) + 2))
    eta = _approx_eta(src, target, est_send, ctx.ship_speed)
    if pid in ctx.comet_ids:
        life = _comet_remaining_life(ctx, pid)
        if eta >= life - 3:
            return -1e9
    needed = _needed_to_take(ctx, target, eta, extra_buffer=1)
    if needed <= 0:
        return -1e9
    prod = float(target[6])
    # Remaining payoff ticks after arrival
    payoff_ticks = min(ctx.remaining_steps - eta, _comet_remaining_life(ctx, pid) - eta)
    if payoff_ticks <= 2:
        return -1e9
    gross_value = prod * min(45.0, float(payoff_ticks))
    cost = float(needed) + 1.35 * float(eta) + 3.0
    # Slight preference for planets closer to the center/frontline for map control
    center_dist = distance((target[2], target[3]), (50.0, 50.0))
    geo_bonus = 1.0 + 0.15 * max(0.0, 1.0 - center_dist / 60.0)
    return (gross_value / cost) * geo_bonus


def _try_single_send(
    ctx: TacticalContext,
    src: Sequence[Any],
    target: Sequence[Any],
    desired_ships: int,
    available: Dict[int, int],
    moves: List[List[float]],
    min_ships: int = 1,
) -> int:
    """Attempt to emit a verified single-planet launch; returns ships sent (0 if failed)."""
    sid = int(src[0])
    avail = available.get(sid, 0)
    send = min(avail, int(desired_ships))
    if send < min_ships:
        return 0
    sol = ctx.intercept(src, target, send)
    if sol is None:
        return 0
    moves.append([sid, float(sol.angle), int(send)])
    available[sid] = avail - send
    return send


def _try_coordinated_capture(
    ctx: TacticalContext,
    target: Sequence[Any],
    available: Dict[int, int],
    moves: List[List[float]],
    max_sources: int = 3,
    extra_buffer: int = 1,
) -> bool:
    """Coordinate 1..max_sources friendly planets to capture ``target``."""
    tid = int(target[0])
    sources = [p for p in ctx.my_planets if available.get(int(p[0]), 0) >= 2 and int(p[0]) != tid]
    if not sources:
        return False

    # Sort sources by estimated ETA to target
    sources.sort(key=lambda s: _approx_eta(s, target, available[int(s[0])], ctx.ship_speed))
    # First try single closest capable source
    for s in sources[:3]:
        sid = int(s[0])
        avail = available[sid]
        eta_est = _approx_eta(s, target, min(avail, int(target[5]) + 5), ctx.ship_speed)
        need = _needed_to_take(ctx, target, eta_est, extra_buffer=extra_buffer)
        if need <= 0:
            return False
        if avail >= need:
            sol = ctx.intercept(s, target, need)
            if sol is not None:
                # Re-check exact needed ships at exact sol.eta
                exact_need = _needed_to_take(ctx, target, sol.eta, extra_buffer=extra_buffer)
                if exact_need <= 0:
                    return False
                if avail >= exact_need:
                    sol2 = sol if exact_need == need else ctx.intercept(s, target, exact_need)
                    if sol2 is not None:
                        moves.append([sid, float(sol2.angle), int(exact_need)])
                        available[sid] = avail - exact_need
                        return True

    # Multi-source coordination
    chosen: List[Tuple[int, float, int, int]] = []
    worst_eta = 1
    total_avail = 0
    for s in sources[:max_sources]:
        sid = int(s[0])
        avail = available[sid]
        eta_est = _approx_eta(s, target, avail, ctx.ship_speed)
        worst_eta = max(worst_eta, eta_est)
        total_avail += avail
        chosen.append((sid, 0.0, avail, eta_est))

    need = _needed_to_take(ctx, target, worst_eta, extra_buffer=extra_buffer)
    if need <= 0 or total_avail < need:
        return False

    # Allocate ships from closest sources first
    planned: List[Tuple[int, float, int]] = []
    rem = need
    for sid, _, avail, _ in chosen:
        if rem <= 0:
            break
        send = min(avail, rem)
        sol = ctx.intercept(ctx.planet_by_id[sid], target, send)
        if sol is None:
            continue
        planned.append((sid, sol.angle, send))
        rem -= send

    if rem > 0 or not planned:
        return False

    for sid, ang, send in planned:
        moves.append([sid, float(ang), int(send)])
        available[sid] -= send
    return True


def _do_defend_urgent(ctx: TacticalContext, available: Dict[int, int], moves: List[List[float]]) -> bool:
    if not ctx.threatened_my:
        return False
    # Prioritize high-production threatened planets with soonest capture tick
    ordered = sorted(
        ctx.threatened_my,
        key=lambda pf: (-int(pf[0][6]), pf[1].capture_tick if pf[1].capture_tick is not None else 999),
    )
    did_any = False
    for target, fut in ordered[:2]:
        tid = int(target[0])
        cap_tick = fut.capture_tick if fut.capture_tick is not None else 10
        deficit = abs(fut.min_owner_ships) + 2
        helpers = [
            p for p in ctx.my_planets
            if int(p[0]) != tid and available.get(int(p[0]), 0) >= 2
        ]
        helpers.sort(key=lambda h: distance((h[2], h[3]), (target[2], target[3])))
        rem = deficit
        for h in helpers[:3]:
            if rem <= 0:
                break
            hid = int(h[0])
            send = min(available[hid], rem)
            sol = ctx.intercept(h, target, send)
            if sol is None:
                continue
            # Either arrive before capture to defend, or arrive right after to retake
            if sol.eta <= cap_tick + 6:
                moves.append([hid, float(sol.angle), int(send)])
                available[hid] -= send
                rem -= send
                did_any = True
    return did_any


def _do_evacuate_doomed(ctx: TacticalContext, available: Dict[int, int], moves: List[List[float]]) -> bool:
    if not ctx.doomed_my:
        return False
    did_any = False
    safe_friends = [p for p in ctx.my_planets if int(p[0]) not in {int(d[0][0]) for d in ctx.doomed_my}]
    for p, _ in ctx.doomed_my:
        pid = int(p[0])
        ships = int(p[5])
        if ships <= 1:
            continue
        # Evacuate entire garrison minus 0 (or all ships) to safest closest friendly or weak enemy
        targets = safe_friends if safe_friends else ctx.enemy_planets
        if not targets:
            continue
        targets = sorted(targets, key=lambda t: distance((p[2], p[3]), (t[2], t[3])))
        for t in targets[:3]:
            sol = ctx.intercept(p, t, ships)
            if sol is not None:
                moves.append([pid, float(sol.angle), int(ships)])
                available[pid] = 0
                did_any = True
                break
    return did_any


def _do_vulture_snipe(ctx: TacticalContext, available: Dict[int, int], moves: List[List[float]]) -> bool:
    if not ctx.vulture_targets:
        return False
    did_any = False
    # Sort by highest production, lowest post-capture ships
    ordered = sorted(ctx.vulture_targets, key=lambda x: (-int(x[0][6]), x[2]))
    for target, cap_tick, post_ships in ordered[:3]:
        tid = int(target[0])
        sources = [p for p in ctx.my_planets if available.get(int(p[0]), 0) >= post_ships + 2]
        sources.sort(key=lambda s: abs(_approx_eta(s, target, available[int(s[0])], ctx.ship_speed) - (cap_tick + 1)))
        for s in sources[:3]:
            sid = int(s[0])
            need_est = post_ships + int(target[6]) * 2 + 2
            if available[sid] < need_est:
                continue
            sol = ctx.intercept(s, target, need_est)
            if sol is None:
                continue
            # Launch if our arrival lands just after the enemy captures the neutral!
            if cap_tick <= sol.eta <= cap_tick + 5:
                exact_need = _needed_to_take(ctx, target, sol.eta, extra_buffer=2)
                if 0 < exact_need <= available[sid]:
                    sol2 = sol if exact_need == need_est else ctx.intercept(s, target, exact_need)
                    if sol2 is not None and sol2.eta >= cap_tick:
                        moves.append([sid, float(sol2.angle), int(exact_need)])
                        available[sid] -= exact_need
                        did_any = True
                        break
    return did_any


def _do_expand_neutrals(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    *,
    max_targets: int = 1,
    filter_mode: str = "roi",
) -> bool:
    if not ctx.neutral_planets:
        return False
    candidates: List[List[Any]] = []
    for p in ctx.neutral_planets:
        pid = int(p[0])
        # Skip neutrals already slated to be captured by us
        if ctx.futures[pid].final_owner == ctx.player:
            continue
        if filter_mode == "orbiting":
            if pid in ctx.comet_ids or not is_orbiting_planet(float(p[2]), float(p[3]), float(p[4])):
                continue
        elif filter_mode == "comet":
            if pid not in ctx.comet_ids:
                continue
        candidates.append(p)

    if not candidates:
        return False

    total_surplus = sum(available.values())
    if total_surplus < 2:
        return False

    scored: List[Tuple[float, List[Any]]] = []
    for target in candidates:
        best_s = -1e9
        for src in ctx.my_planets:
            av = available.get(int(src[0]), 0)
            if av < 2:
                continue
            sc = _score_neutral_roi(ctx, src, target, total_surplus)
            if filter_mode == "high_prod":
                sc += float(target[6]) * 2.5
            elif filter_mode == "comet":
                sc += 5.0
            if sc > best_s:
                best_s = sc
        if best_s > -1e8:
            scored.append((best_s, target))

    scored.sort(key=lambda x: x[0], reverse=True)
    launched = 0
    for _, target in scored:
        if launched >= max_targets:
            break
        if _try_coordinated_capture(ctx, target, available, moves, max_sources=2, extra_buffer=1):
            launched += 1
    return launched > 0


def _do_attack_enemy(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    *,
    target_mode: str = "weakest",
    max_targets: int = 1,
    all_in: bool = False,
) -> bool:
    if not ctx.enemy_planets:
        return False
    candidates = [p for p in ctx.enemy_planets if ctx.futures[int(p[0])].final_owner != ctx.player]
    if not candidates:
        candidates = list(ctx.enemy_planets)

    if target_mode == "weakest":
        candidates.sort(key=lambda p: (int(p[5]) - int(p[6]) * 2, min((distance((s[2], s[3]), (p[2], p[3])) for s in ctx.my_planets), default=100.0)))
    elif target_mode == "high_prod":
        candidates.sort(key=lambda p: (-int(p[6]), int(p[5])))
    elif target_mode == "closest":
        candidates.sort(key=lambda p: min((distance((s[2], s[3]), (p[2], p[3])) for s in ctx.my_planets), default=100.0))

    if all_in:
        target = candidates[0]
        did_any = False
        for src in ctx.my_planets:
            sid = int(src[0])
            avail = available.get(sid, 0)
            send = int(avail * 0.75)
            if send >= 3:
                if _try_single_send(ctx, src, target, send, available, moves, min_ships=3) > 0:
                    did_any = True
        return did_any

    launched = 0
    for target in candidates:
        if launched >= max_targets:
            break
        if _try_coordinated_capture(ctx, target, available, moves, max_sources=3, extra_buffer=2):
            launched += 1
    return launched > 0


def _do_feed_frontline(ctx: TacticalContext, available: Dict[int, int], moves: List[List[float]]) -> bool:
    if len(ctx.my_planets) < 2 or ctx.enemy_centroid is None:
        return False
    ex, ey = ctx.enemy_centroid
    # Rank friendly planets by proximity to enemy centroid
    ranked = sorted(ctx.my_planets, key=lambda p: distance((p[2], p[3]), (ex, ey)))
    frontline = ranked[0]
    front_id = int(frontline[0])
    front_dist = distance((frontline[2], frontline[3]), (ex, ey))

    did_any = False
    for p in ranked[1:]:
        pid = int(p[0])
        avail = available.get(pid, 0)
        if avail < 6:
            continue
        my_dist = distance((p[2], p[3]), (ex, ey))
        if my_dist <= front_dist + 8.0:
            continue
        send = int(avail * 0.65)
        if send >= 5:
            if _try_single_send(ctx, p, frontline, send, available, moves, min_ships=5) > 0:
                did_any = True
    return did_any


def get_legal_action_mask(
    state: Any,
    player: int,
    ctx: Optional[TacticalContext] = None,
) -> np.ndarray:
    """Compute an ``int8`` action mask of shape ``(16,)`` for legal macro-actions."""
    if ctx is None:
        ctx = build_tactical_context(state, player)

    mask = np.zeros(NUM_MACRO_ACTIONS, dtype=np.int8)
    mask[MacroAction.HOLD_ACCUMULATE] = 1

    if not ctx.my_planets:
        return mask

    total_surplus = sum(ctx.surplus.values())
    has_surplus = total_surplus >= 2
    has_neutral = len(ctx.neutral_planets) > 0
    has_enemy = len(ctx.enemy_planets) > 0

    if has_surplus and has_neutral:
        mask[MacroAction.EXPAND_BEST_ROI_SINGLE] = 1
        mask[MacroAction.EXPAND_BEST_ROI_MULTI] = 1
        mask[MacroAction.EXPAND_HIGH_PROD] = 1
        if any(
            int(p[0]) not in ctx.comet_ids and is_orbiting_planet(float(p[2]), float(p[3]), float(p[4]))
            for p in ctx.neutral_planets
        ):
            mask[MacroAction.EXPAND_ORBITING] = 1
        if any(
            int(p[0]) in ctx.comet_ids and _comet_remaining_life(ctx, int(p[0])) > 6
            for p in ctx.neutral_planets
        ):
            mask[MacroAction.EXPAND_COMET_SNIPE] = 1

    if has_surplus and ctx.threatened_my:
        mask[MacroAction.DEFEND_URGENT] = 1

    if ctx.doomed_my and any(int(p[5]) > 1 for p, _ in ctx.doomed_my):
        mask[MacroAction.COUNTER_EVACUATE] = 1

    if has_surplus and ctx.vulture_targets:
        mask[MacroAction.VULTURE_SNIPE] = 1

    if has_surplus and has_enemy:
        mask[MacroAction.ATTACK_WEAKEST_ENEMY] = 1
        mask[MacroAction.ATTACK_HIGH_PROD_ENEMY] = 1
        mask[MacroAction.ATTACK_CLOSEST_ENEMY] = 1
        if total_surplus >= 10:
            mask[MacroAction.COORDINATED_ALL_IN] = 1
        mask[MacroAction.AGGRESSIVE_SKIRMISH] = 1

    if len(ctx.my_planets) >= 2 and ctx.enemy_centroid is not None and any(v >= 6 for v in ctx.surplus.values()):
        mask[MacroAction.FEED_FRONTLINE] = 1

    if has_surplus or ctx.doomed_my:
        mask[MacroAction.BALANCED_MACRO] = 1

    return mask


def execute_macro_action(
    state: Any,
    player: int,
    action_id: int,
    ctx: Optional[TacticalContext] = None,
) -> List[List[float]]:
    """Translate a discrete ``MacroAction`` index into exact Orbit Wars launch orders."""
    act = MacroAction(int(action_id))
    if act == MacroAction.HOLD_ACCUMULATE:
        return []

    if ctx is None:
        ctx = build_tactical_context(state, player)
    if not ctx.my_planets:
        return []

    available = dict(ctx.surplus)
    moves: List[List[float]] = []

    if act == MacroAction.EXPAND_BEST_ROI_SINGLE:
        _do_expand_neutrals(ctx, available, moves, max_targets=1, filter_mode="roi")
    elif act == MacroAction.EXPAND_BEST_ROI_MULTI:
        _do_expand_neutrals(ctx, available, moves, max_targets=3, filter_mode="roi")
    elif act == MacroAction.EXPAND_HIGH_PROD:
        _do_expand_neutrals(ctx, available, moves, max_targets=2, filter_mode="high_prod")
    elif act == MacroAction.EXPAND_ORBITING:
        _do_expand_neutrals(ctx, available, moves, max_targets=2, filter_mode="orbiting")
    elif act == MacroAction.EXPAND_COMET_SNIPE:
        _do_expand_neutrals(ctx, available, moves, max_targets=2, filter_mode="comet")
    elif act == MacroAction.DEFEND_URGENT:
        _do_defend_urgent(ctx, available, moves)
    elif act == MacroAction.COUNTER_EVACUATE:
        _do_evacuate_doomed(ctx, available, moves)
    elif act == MacroAction.VULTURE_SNIPE:
        _do_vulture_snipe(ctx, available, moves)
    elif act == MacroAction.ATTACK_WEAKEST_ENEMY:
        _do_attack_enemy(ctx, available, moves, target_mode="weakest", max_targets=1)
    elif act == MacroAction.ATTACK_HIGH_PROD_ENEMY:
        _do_attack_enemy(ctx, available, moves, target_mode="high_prod", max_targets=1)
    elif act == MacroAction.ATTACK_CLOSEST_ENEMY:
        _do_attack_enemy(ctx, available, moves, target_mode="closest", max_targets=1)
    elif act == MacroAction.COORDINATED_ALL_IN:
        _do_attack_enemy(ctx, available, moves, target_mode="weakest", all_in=True)
    elif act == MacroAction.FEED_FRONTLINE:
        _do_feed_frontline(ctx, available, moves)
    elif act == MacroAction.BALANCED_MACRO:
        _do_defend_urgent(ctx, available, moves)
        _do_evacuate_doomed(ctx, available, moves)
        _do_vulture_snipe(ctx, available, moves)
        # Early/mid game prioritize neutrals; late game prioritize enemy pressure
        if ctx.remaining_steps > 60:
            _do_expand_neutrals(ctx, available, moves, max_targets=2, filter_mode="roi")
        _do_attack_enemy(ctx, available, moves, target_mode="weakest", max_targets=1)
        _do_feed_frontline(ctx, available, moves)
    elif act == MacroAction.AGGRESSIVE_SKIRMISH:
        _do_defend_urgent(ctx, available, moves)
        _do_vulture_snipe(ctx, available, moves)
        _do_attack_enemy(ctx, available, moves, target_mode="weakest", max_targets=2)
        _do_attack_enemy(ctx, available, moves, target_mode="high_prod", max_targets=1)
        _do_expand_neutrals(ctx, available, moves, max_targets=1, filter_mode="roi")

    return moves
