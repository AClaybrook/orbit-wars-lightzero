"""16 discrete tactical macro-actions for Orbit Wars LightZero agents.

Maps high-level Galcon / Orbit Wars strategic directives to exact, sun-safe,
swept-pair-verified ``[from_planet_id, angle, ships]`` launch orders using
``orbit_wars_bot.physics``.

Supports both 2-player 1v1 and 4-player FFA with:
- Per-opponent ship/production/planet tracking (leader & weakest identification).
- Light border garrison reserves (preventing naked borders without freezing strikes).
- Atomic defense verification (never double-retaking or trickling into lost fights) + last-tick evacuation.
- Neutral-race trap avoidance + enemy-vs-enemy crash-exploit vulture sniping.
- Static vs orbiting vs comet durability scoring + unified neutral/enemy ROI allocation + elimination bonus.
- Tightly synchronized multi-source captures (no slow snail tails).
- Short-hop anti-ping-pong backline-to-frontline stack consolidation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
DEFAULT_TACTICAL_HORIZON: int = 36


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
    horizon: int
    angular_velocity: float
    ship_speed: float
    planets: List[List[Any]]
    planet_by_id: Dict[int, List[Any]]
    initial_by_id: Dict[int, Sequence[Any]]
    comet_ids: Set[int]
    comet_lookup: Dict[int, Tuple[List[List[float]], int]]
    tracks: Dict[int, List[Optional[Tuple[float, float]]]]
    arrivals: Dict[int, List[FleetArrival]]
    futures: Dict[int, PlanetFutureState]
    my_planets: List[List[Any]]
    enemy_planets: List[List[Any]]
    neutral_planets: List[List[Any]]
    raw_surplus: Dict[int, int]
    border_reserve: Dict[int, int]
    surplus: Dict[int, int]
    threatened_my: List[Tuple[List[Any], PlanetFutureState]]
    doomed_my: List[Tuple[List[Any], PlanetFutureState]]
    vulture_targets: List[Tuple[List[Any], int, int]]  # (planet, impact_tick, post_ships)
    enemy_centroid: Optional[Tuple[float, float]]
    nearest_enemy_dist: Dict[int, float]
    my_total_ships: int
    enemy_total_ships: int
    active_opponents: List[int]
    is_ffa: bool
    ships_by_player: Dict[int, int]
    prod_by_player: Dict[int, int]
    planets_by_player: Dict[int, int]
    leader_enemy_id: Optional[int]
    weakest_enemy_id: Optional[int]
    leader_ships: int
    my_prod: int
    leader_prod: int
    _intercept_cache: Dict[Tuple[int, int, int], Optional[InterceptSolution]]

    def intercept(
        self,
        src: Sequence[Any],
        dst: Sequence[Any],
        ships: int,
        max_eta: Optional[int] = None,
    ) -> Optional[InterceptSolution]:
        if ships <= 0:
            return None
        eff_max = self.horizon if max_eta is None else min(int(max_eta), self.horizon)
        eff_max = min(eff_max, max(1, self.remaining_steps))
        key = (int(src[0]), int(dst[0]), int(ships))
        if key in self._intercept_cache:
            sol = self._intercept_cache[key]
            if sol is None or sol.eta <= eff_max:
                return sol
            return None
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
            max_eta=eff_max,
            max_speed=self.ship_speed,
        )
        self._intercept_cache[key] = sol
        return sol


def _extract_state(
    state: Any, player: int
) -> Tuple[
    List[List[Any]],
    List[List[Any]],
    List[List[Any]],
    List[Dict[str, Any]],
    List[int],
    float,
    int,
    int,
    float,
]:
    del player
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


def build_tactical_context(
    state: Any,
    player: int,
    horizon: int = DEFAULT_TACTICAL_HORIZON,
) -> TacticalContext:
    """Build a ``TacticalContext`` from a ``FastOrbitSim`` or Kaggle ``obs``."""
    planets, init_planets, fleets, comets, comet_pids, ang_vel, step, max_steps, ship_speed = (
        _extract_state(state, player)
    )
    remaining = max(1, max_steps - step)
    eff_horizon = min(int(horizon), remaining)

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
    raw_surplus: Dict[int, int] = {}
    threatened_my: List[Tuple[List[Any], PlanetFutureState]] = []
    doomed_my: List[Tuple[List[Any], PlanetFutureState]] = []
    vulture_targets: List[Tuple[List[Any], int, int]] = []

    ships_by_player: Dict[int, int] = {}
    prod_by_player: Dict[int, int] = {}
    planets_by_player: Dict[int, int] = {}

    for p in planets:
        pid = int(p[0])
        owner = int(p[1])
        ships = int(p[5])
        prod = int(p[6])
        p_arrivals = arrivals.get(pid, ())
        fut = simulate_planet_future(p, p_arrivals, horizon=eff_horizon)
        futures[pid] = fut

        if owner >= 0:
            ships_by_player[owner] = ships_by_player.get(owner, 0) + ships
            prod_by_player[owner] = prod_by_player.get(owner, 0) + prod
            planets_by_player[owner] = planets_by_player.get(owner, 0) + 1

        if owner == player:
            my_planets.append(p)
            if fut.will_be_captured_by is not None and fut.will_be_captured_by != player:
                raw_surplus[pid] = 0
                threatened_my.append((p, fut))
                if fut.capture_tick is not None and fut.capture_tick <= 2:
                    doomed_my.append((p, fut))
            else:
                has_enemy_incoming = any(a.owner != player for a in p_arrivals)
                if has_enemy_incoming:
                    safe = max(0, min(ships, fut.min_owner_ships - 1))
                else:
                    # Safe against all in-flight fleets; Phase 1 production immediately adds +prod
                    safe = max(0, ships)
                raw_surplus[pid] = safe
        elif owner == -1:
            neutral_planets.append(p)
            if fut.will_be_captured_by is not None and fut.will_be_captured_by != player and fut.capture_tick is not None:
                _, post_ships = fut.timeline[fut.capture_tick]
                vulture_targets.append((p, fut.capture_tick, int(post_ships)))
            elif p_arrivals:
                enemy_arr = [a for a in p_arrivals if a.owner != player]
                if enemy_arr:
                    first_t = min(a.eta for a in enemy_arr)
                    if 1 <= first_t <= fut.horizon:
                        o_after, s_after = fut.timeline[first_t]
                        if o_after != player and s_after <= max(8, ships // 3):
                            vulture_targets.append((p, first_t, int(s_after)))
        else:
            enemy_planets.append(p)
            # Detect enemy-vs-enemy battles (crash-exploit sniping in 4P FFA!)
            if p_arrivals:
                hostile_enemy_arr = [a for a in p_arrivals if a.owner != player and a.owner != owner]
                if hostile_enemy_arr:
                    best_hit_t = None
                    best_post = 10**9
                    for a in hostile_enemy_arr:
                        t_hit = a.eta
                        if 1 <= t_hit <= fut.horizon:
                            o_t, s_t = fut.timeline[t_hit]
                            if o_t != player and s_t < best_post:
                                best_post = int(s_t)
                                best_hit_t = t_hit
                    if best_hit_t is not None and best_post <= max(18, prod * 5):
                        vulture_targets.append((p, best_hit_t, best_post))

    for f in fleets:
        f_owner = int(f[1])
        if f_owner >= 0:
            ships_by_player[f_owner] = ships_by_player.get(f_owner, 0) + int(f[6])

    my_total_ships = ships_by_player.get(player, 0)
    my_prod = prod_by_player.get(player, 0)
    active_opponents = [pid for pid, s in ships_by_player.items() if pid != player and s > 0]
    enemy_total_ships = sum(ships_by_player.get(pid, 0) for pid in active_opponents)
    is_ffa = len(active_opponents) >= 2

    leader_enemy_id: Optional[int] = None
    weakest_enemy_id: Optional[int] = None
    if active_opponents:
        prod_w = min(28, int(remaining * 0.5))
        leader_enemy_id = max(
            active_opponents,
            key=lambda pid: ships_by_player.get(pid, 0) + prod_by_player.get(pid, 0) * prod_w,
        )
        weakest_enemy_id = min(
            active_opponents,
            key=lambda pid: ships_by_player.get(pid, 0) + prod_by_player.get(pid, 0) * prod_w,
        )
    leader_ships = ships_by_player.get(leader_enemy_id, 0) if leader_enemy_id is not None else 0
    leader_prod = prod_by_player.get(leader_enemy_id, 0) if leader_enemy_id is not None else 0

    # Nearest enemy distance & light border garrison reserve per friendly planet.
    # Light enough not to freeze decisive strikes, strong enough to deter free snipes.
    nearest_enemy_dist: Dict[int, float] = {}
    border_reserve: Dict[int, int] = {}
    surplus: Dict[int, int] = {}

    use_border_reserve = (step >= 26 and len(my_planets) >= 3 and len(enemy_planets) > 0)

    for mp in my_planets:
        mpid = int(mp[0])
        mx, my = float(mp[2]), float(mp[3])
        mships = int(mp[5])
        raw_s = raw_surplus[mpid]

        min_d = 140.0
        local_threat = 0.0
        for ep in enemy_planets:
            d = distance((mx, my), (float(ep[2]), float(ep[3])))
            if d < min_d:
                min_d = d
            if use_border_reserve and d < 32.0:
                eships = int(ep[5])
                if eships > 5:
                    prox = max(0.0, 1.0 - (d - 8.0) / 26.0)
                    local_threat += eships * prox * (0.28 if is_ffa else 0.18)

        nearest_enemy_dist[mpid] = min_d
        if use_border_reserve and local_threat > 0.0 and raw_s > 0:
            cap_frac = 0.25 if is_ffa else 0.16
            reserve = min(int(local_threat), int(mships * cap_frac), max(8, int(mp[6]) * 8))
            reserve = max(0, min(reserve, raw_s))
        else:
            reserve = 0
        border_reserve[mpid] = reserve
        surplus[mpid] = max(0, raw_s - reserve)

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
        horizon=eff_horizon,
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
        raw_surplus=raw_surplus,
        border_reserve=border_reserve,
        surplus=surplus,
        threatened_my=threatened_my,
        doomed_my=doomed_my,
        vulture_targets=vulture_targets,
        enemy_centroid=enemy_centroid,
        nearest_enemy_dist=nearest_enemy_dist,
        my_total_ships=my_total_ships,
        enemy_total_ships=enemy_total_ships,
        active_opponents=active_opponents,
        is_ffa=is_ffa,
        ships_by_player=ships_by_player,
        prod_by_player=prod_by_player,
        planets_by_player=planets_by_player,
        leader_enemy_id=leader_enemy_id,
        weakest_enemy_id=weakest_enemy_id,
        leader_ships=leader_ships,
        my_prod=my_prod,
        leader_prod=leader_prod,
        _intercept_cache={},
    )


def _deduct_ships(
    pid: int,
    amount: int,
    available: Dict[int, int],
    raw_available: Optional[Dict[int, int]] = None,
) -> None:
    if raw_available is not None:
        raw_available[pid] = max(0, raw_available.get(pid, 0) - amount)
    available[pid] = max(0, available.get(pid, 0) - amount)
    if raw_available is not None and available[pid] > raw_available[pid]:
        available[pid] = raw_available[pid]


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


def _neutral_race_is_trap(ctx: TacticalContext, target: Sequence[Any], eta: int) -> bool:
    """Return True if an enemy fleet arrives right after our ``eta`` and would steal the neutral."""
    pid = int(target[0])
    p_arrivals = ctx.arrivals.get(pid, ())
    if not p_arrivals:
        return False
    prod = int(target[6])
    for a in p_arrivals:
        if a.owner != ctx.player and eta <= a.eta <= eta + 9:
            if a.ships >= 1 + prod * (a.eta - eta):
                return True
    return False


def _score_neutral_roi(
    ctx: TacticalContext,
    src: Sequence[Any],
    target: Sequence[Any],
    avail_ships: int,
) -> Tuple[float, int, int]:
    """Return ``(score, needed, eta)`` for capturing neutral ``target`` from ``src``.

    Heavily favors durable static planets in our own quadrant, caps orbiting hold
    horizons (since orbiting planets revolve into rival sectors in ~40 ticks), and
    discounts short-lived 1-prod comets.
    """
    pid = int(target[0])
    is_comet = pid in ctx.comet_ids
    if is_comet:
        life = _comet_remaining_life(ctx, pid)
        if life <= 9:
            return (-1e9, 0, 999)

    est_send = max(2, min(max(avail_ships, int(target[5]) + 2), int(target[5]) + 2))
    eta = _approx_eta(src, target, est_send, ctx.ship_speed)
    if eta >= ctx.horizon:
        return (-1e9, 0, eta)

    if is_comet:
        life = _comet_remaining_life(ctx, pid)
        if eta >= life - 5:
            return (-1e9, 0, eta)

    needed = _needed_to_take(ctx, target, eta, extra_buffer=1)
    if needed <= 0:
        return (-1e9, 0, eta)

    if _neutral_race_is_trap(ctx, target, eta):
        return (-1e9, needed, eta)

    prod = float(target[6])
    payoff_ticks = min(ctx.remaining_steps - eta, _comet_remaining_life(ctx, pid) - eta)
    min_profit = 5.0 if is_comet else 2.0
    if payoff_ticks <= 3 or prod * float(payoff_ticks) <= float(needed) + min_profit:
        return (-1e9, needed, eta)

    is_orbiting = (not is_comet) and is_orbiting_planet(float(target[2]), float(target[3]), float(target[4]))
    if is_comet:
        eff_horizon = float(payoff_ticks)
        durability_mult = 0.28
    elif is_orbiting:
        # Orbiting planets revolve ~90 deg in ~35-55 ticks, so they are harder to hold long-term in 4P FFA
        eff_horizon = min(30.0 if ctx.is_ffa else 38.0, float(payoff_ticks))
        durability_mult = 0.85 if ctx.is_ffa else 0.96
    else:
        # Static planets stay in your quadrant permanently -> high strategic durability!
        eff_horizon = min(52.0, float(payoff_ticks))
        durability_mult = 1.42 if ctx.is_ffa else 1.22

    gross_value = prod * eff_horizon
    cost = float(needed) + 1.40 * float(eta) + 2.5

    # Sector safety: strongly prefer neutrals in our quadrant rather than contested/enemy sectors
    my_d = distance((src[2], src[3]), (target[2], target[3]))
    if ctx.enemy_planets:
        closest_e_d = min(distance((ep[2], ep[3]), (target[2], target[3])) for ep in ctx.enemy_planets)
        if my_d < closest_e_d * 0.90:
            sector_mult = 1.32 if ctx.is_ffa else 1.18
        elif closest_e_d < my_d * 0.78:
            sector_mult = 0.42 if ctx.is_ffa else 0.72
        else:
            sector_mult = 0.82 if ctx.is_ffa else 0.95
    else:
        sector_mult = 1.0

    return ((gross_value / cost) * sector_mult * durability_mult, needed, eta)


def _score_enemy_attack(
    ctx: TacticalContext,
    target: Sequence[Any],
    target_mode: str = "roi",
) -> Tuple[float, int, int]:
    """Return ``(roi_score, est_needed, best_eta)`` for attacking enemy planet ``target``."""
    if not ctx.my_planets:
        return (-1e9, 0, 999)
    pid = int(target[0])
    owner = int(target[1])
    prod = float(target[6])
    best_src = min(
        ctx.my_planets,
        key=lambda s: distance((s[2], s[3]), (target[2], target[3])),
    )
    est_ships = max(8, int(target[5]) + int(prod) * 4 + 2)
    eta = _approx_eta(best_src, target, est_ships, ctx.ship_speed)
    if eta >= ctx.horizon:
        return (-1e9, 0, eta)

    needed = _needed_to_take(ctx, target, eta, extra_buffer=2)
    if needed <= 0:
        return (-1e9, 0, eta)

    payoff_ticks = min(44.0, float(max(0, ctx.remaining_steps - eta)))
    if payoff_ticks <= 2.0:
        return (-1e9, needed, eta)

    # Capturing an enemy planet is a 2x production swing in 2P, slightly discounted in 4P FFA
    swing = 1.65 if ctx.is_ffa else 2.05
    gross = swing * prod * payoff_ticks + 8.0
    cost = float(needed) + 1.50 * float(eta) + 3.5
    roi = gross / cost

    # Static enemy fortresses are permanently valuable captures
    is_comet = pid in ctx.comet_ids
    is_orbiting = (not is_comet) and is_orbiting_planet(float(target[2]), float(target[3]), float(target[4]))
    if is_comet:
        roi *= 0.25
    elif not is_orbiting:
        roi *= 1.24 if ctx.is_ffa else 1.15

    # Strongly prefer short/medium range strikes over long cross-map flights
    if eta > 16:
        roi *= max(0.40, 1.0 - 0.035 * (eta - 16))
    elif eta <= 8:
        roi *= 1.20

    # Massive elimination bonus: permanently removing a rival in 4P FFA is huge!
    opp_planets = ctx.planets_by_player.get(owner, 99)
    opp_ships = ctx.ships_by_player.get(owner, 9999)
    if opp_planets == 1:
        roi *= 2.15 if ctx.is_ffa else 1.65
    elif opp_planets <= 2 and opp_ships < max(70, ctx.my_total_ships * 0.60):
        roi *= 1.48

    # Mode-specific weighting
    if target_mode == "high_prod":
        roi *= 1.0 + 0.16 * prod
    elif target_mode == "closest":
        roi *= 1.85 / (1.0 + 0.12 * eta)
    elif target_mode == "leader":
        if ctx.leader_enemy_id is not None and owner == ctx.leader_enemy_id:
            roi *= 1.70
        else:
            roi *= 0.70
    else:
        if ctx.is_ffa and ctx.leader_enemy_id is not None and owner == ctx.leader_enemy_id:
            if ctx.leader_ships > ctx.my_total_ships * 1.15:
                roi *= 1.20

    return (roi, needed, eta)


def _try_single_send(
    ctx: TacticalContext,
    src: Sequence[Any],
    target: Sequence[Any],
    desired_ships: int,
    available: Dict[int, int],
    moves: List[List[float]],
    min_ships: int = 1,
    raw_available: Optional[Dict[int, int]] = None,
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
    _deduct_ships(sid, send, available, raw_available)
    return send


def _try_coordinated_capture(
    ctx: TacticalContext,
    target: Sequence[Any],
    available: Dict[int, int],
    moves: List[List[float]],
    max_sources: int = 3,
    extra_buffer: int = 1,
    raw_available: Optional[Dict[int, int]] = None,
    claimed: Optional[Set[int]] = None,
) -> bool:
    """Coordinate 1..max_sources friendly planets to capture ``target`` cleanly.

    Strongly prefers a single fast stack; when coordinating multiple sources,
    enforces tight arrival synchronization (no slow 5-ship snail tails arriving
    7 ticks late and gifting vultures!).
    """
    tid = int(target[0])
    if claimed is not None and tid in claimed:
        return False

    sources = [p for p in ctx.my_planets if available.get(int(p[0]), 0) >= 2 and int(p[0]) != tid]
    if not sources:
        return False

    sources.sort(key=lambda s: _approx_eta(s, target, available[int(s[0])], ctx.ship_speed))

    # 1. Single closest capable source (fastest & cleanest!)
    for s in sources[:3]:
        sid = int(s[0])
        avail = available[sid]
        eta_est = _approx_eta(s, target, min(avail, int(target[5]) + 6), ctx.ship_speed)
        if eta_est >= ctx.horizon:
            continue
        need = _needed_to_take(ctx, target, eta_est, extra_buffer=extra_buffer)
        if need <= 0:
            return False
        if avail >= need:
            sol = ctx.intercept(s, target, need)
            if sol is not None:
                exact_need = _needed_to_take(ctx, target, sol.eta, extra_buffer=extra_buffer)
                if exact_need <= 0:
                    return False
                if avail >= exact_need:
                    sol2 = sol if exact_need == need else ctx.intercept(s, target, exact_need)
                    if sol2 is not None:
                        moves.append([sid, float(sol2.angle), int(exact_need)])
                        _deduct_ships(sid, exact_need, available, raw_available)
                        if claimed is not None:
                            claimed.add(tid)
                        return True

    # 2. Tightly synchronized multi-source coordination
    is_enemy = int(target[1]) >= 0 and int(target[1]) != ctx.player
    max_spread = 3 if is_enemy else 4
    chosen: List[Tuple[int, int, int]] = []
    worst_eta = 1
    total_avail = 0
    for s in sources[:max_sources]:
        sid = int(s[0])
        avail = available[sid]
        if avail < 6:
            continue
        eta_est = _approx_eta(s, target, avail, ctx.ship_speed)
        if eta_est >= ctx.horizon:
            continue
        worst_eta = max(worst_eta, eta_est)
        total_avail += avail
        chosen.append((sid, avail, eta_est))

    if len(chosen) < 2:
        return False

    need = _needed_to_take(ctx, target, worst_eta, extra_buffer=extra_buffer)
    if need <= 0 or total_avail < need:
        return False

    # Allocate ships from closest sources first, avoiding tiny slow tail fleets
    planned: List[Tuple[int, float, int, int]] = []
    rem = need
    min_chunk = max(5, need // 5)
    for sid, avail, _ in chosen:
        if rem <= 0:
            break
        send = min(avail, rem)
        # Don't create a tiny slow snail tail if we can absorb it or skip
        if send < min_chunk and rem == send and planned:
            # Check if we can top up the previous source instead
            prev_sid, _, prev_send, _ = planned[-1]
            if available[prev_sid] >= prev_send + send:
                sol_p = ctx.intercept(ctx.planet_by_id[prev_sid], target, prev_send + send)
                if sol_p is not None:
                    planned[-1] = (prev_sid, sol_p.angle, prev_send + send, sol_p.eta)
                    rem = 0
                    break
        if send < 5 and len(planned) > 0:
            return False
        sol = ctx.intercept(ctx.planet_by_id[sid], target, send)
        if sol is None:
            continue
        planned.append((sid, sol.angle, send, sol.eta))
        rem -= send

    if rem > 0 or not planned:
        return False

    etas = [x[3] for x in planned]
    min_verified_eta = min(etas)
    max_verified_eta = max(etas)
    if max_verified_eta - min_verified_eta > max_spread:
        return False

    exact_total = _needed_to_take(ctx, target, max_verified_eta, extra_buffer=extra_buffer)
    if exact_total > sum(x[2] for x in planned):
        extra = exact_total - sum(x[2] for x in planned)
        topped = False
        for idx, (sid, _, send, _) in enumerate(planned):
            if available[sid] >= send + extra:
                sol_top = ctx.intercept(ctx.planet_by_id[sid], target, send + extra)
                if sol_top is not None:
                    planned[idx] = (sid, sol_top.angle, send + extra, sol_top.eta)
                    topped = True
                    break
        if not topped:
            return False

    for sid, ang, send, _ in planned:
        moves.append([sid, float(ang), int(send)])
        _deduct_ships(sid, send, available, raw_available)
    if claimed is not None:
        claimed.add(tid)
    return True


def _do_defend_urgent(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    raw_available: Optional[Dict[int, int]] = None,
    claimed: Optional[Set[int]] = None,
) -> Set[int]:
    """Defend threatened planets ONLY when helpers can actually cover the deficit.

    Never double-sends counter-retake fleets when ``fut.final_owner == ctx.player``
    (i.e. a friendly retake fleet is already in flight!).
    """
    saved_pids: Set[int] = set()
    if not ctx.threatened_my:
        return saved_pids

    pool = raw_available if raw_available is not None else available
    ordered = sorted(
        ctx.threatened_my,
        key=lambda pf: (-int(pf[0][6]), pf[1].capture_tick if pf[1].capture_tick is not None else 999),
    )

    for target, fut in ordered[:3]:
        tid = int(target[0])
        cap_tick = fut.capture_tick if fut.capture_tick is not None else 10
        deficit = abs(fut.min_owner_ships) + 2
        prod = int(target[6])

        # Never overspend huge stacks to save a low-prod planet that isn't worth it
        if prod == 1 and deficit > 45 and len(ctx.my_planets) >= 4:
            continue

        helpers = [
            p for p in ctx.my_planets
            if int(p[0]) != tid and pool.get(int(p[0]), 0) >= 2
        ]
        helpers.sort(key=lambda h: _approx_eta(h, target, pool[int(h[0])], ctx.ship_speed))

        # 1. Try atomic IN-TIME defense (all helper fleets arrive <= cap_tick)
        in_time_plan: List[Tuple[int, float, int]] = []
        rem = deficit
        for h in helpers[:4]:
            if rem <= 0:
                break
            hid = int(h[0])
            send = min(pool[hid], rem)
            sol = ctx.intercept(h, target, send, max_eta=cap_tick)
            if sol is not None and sol.eta <= cap_tick:
                in_time_plan.append((hid, sol.angle, send))
                rem -= send

        if rem <= 0 and in_time_plan:
            for hid, ang, send in in_time_plan:
                moves.append([hid, float(ang), int(send)])
                _deduct_ships(hid, send, available, raw_available)
            saved_pids.add(tid)
            if claimed is not None:
                claimed.add(tid)
            continue

        # 2. Cannot save in time -> consider immediate counter-retake ONLY if we don't
        # ALREADY have a retake fleet in flight (fut.final_owner != ctx.player)!
        if fut.final_owner == ctx.player:
            continue

        if prod >= 2 and (claimed is None or tid not in claimed):
            retake_plan: List[Tuple[int, float, int, int]] = []
            max_retake_eta = min(ctx.horizon, cap_tick + 4)
            base_post = int(fut.timeline[min(cap_tick, fut.horizon)][1])
            rem_r = base_post + prod * 2 + 2
            if rem_r <= prod * 14 + 8:
                for h in helpers[:3]:
                    if rem_r <= 0:
                        break
                    hid = int(h[0])
                    send = min(pool[hid], rem_r)
                    sol = ctx.intercept(h, target, send, max_eta=max_retake_eta)
                    if sol is not None and cap_tick < sol.eta <= max_retake_eta:
                        retake_plan.append((hid, sol.angle, send, sol.eta))
                        rem_r -= send
                if rem_r <= 0 and retake_plan:
                    worst_eta = max(x[3] for x in retake_plan)
                    exact_need = base_post + prod * max(1, worst_eta - cap_tick) + 2
                    if sum(x[2] for x in retake_plan) >= exact_need:
                        for hid, ang, send, _ in retake_plan:
                            moves.append([hid, float(ang), int(send)])
                            _deduct_ships(hid, send, available, raw_available)
                        if claimed is not None:
                            claimed.add(tid)
    return saved_pids


def _do_evacuate_doomed(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    saved_pids: Optional[Set[int]] = None,
    raw_available: Optional[Dict[int, int]] = None,
) -> bool:
    """Evacuate unsaveable planets ONLY on the final tick before impact (``capture_tick <= 1``)."""
    if not ctx.doomed_my:
        return False
    saved = saved_pids or set()
    did_any = False
    threatened_ids = {int(t[0][0]) for t in ctx.threatened_my}
    safe_friends = [p for p in ctx.my_planets if int(p[0]) not in threatened_ids]
    if not safe_friends:
        safe_friends = [p for p in ctx.my_planets if int(p[0]) not in {int(d[0][0]) for d in ctx.doomed_my}]

    for p, fut in ctx.doomed_my:
        pid = int(p[0])
        if pid in saved:
            continue
        if fut.capture_tick is None or fut.capture_tick > 1:
            continue
        ships = int(p[5])
        if ships <= 1:
            continue
        targets = safe_friends if safe_friends else ctx.enemy_planets
        if not targets:
            continue
        targets = sorted(targets, key=lambda t: distance((p[2], p[3]), (t[2], t[3])))
        for t in targets[:3]:
            sol = ctx.intercept(p, t, ships)
            if sol is not None:
                moves.append([pid, float(sol.angle), int(ships)])
                available[pid] = 0
                if raw_available is not None:
                    raw_available[pid] = 0
                did_any = True
                break
    return did_any


def _do_vulture_snipe(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    raw_available: Optional[Dict[int, int]] = None,
    claimed: Optional[Set[int]] = None,
) -> bool:
    """Snipe neutrals or enemy-vs-enemy battle planets 1-4 ticks AFTER impact."""
    if not ctx.vulture_targets:
        return False
    pool = raw_available if raw_available is not None else available
    did_any = False
    ordered = sorted(ctx.vulture_targets, key=lambda x: (-int(x[0][6]), x[2], x[1]))
    seen_tids: Set[int] = set()

    for target, cap_tick, post_ships in ordered:
        tid = int(target[0])
        if tid in seen_tids or (claimed is not None and tid in claimed):
            continue
        seen_tids.add(tid)
        if ctx.futures[tid].final_owner == ctx.player:
            continue
        prod = int(target[6])
        if prod <= 0:
            continue
        if post_ships > max(18, prod * 6):
            continue

        sources = [p for p in ctx.my_planets if pool.get(int(p[0]), 0) >= post_ships + 2]
        if not sources:
            continue
        sources.sort(
            key=lambda s: abs(_approx_eta(s, target, pool[int(s[0])], ctx.ship_speed) - (cap_tick + 1))
        )
        for s in sources[:3]:
            sid = int(s[0])
            need_est = post_ships + prod * 2 + 2
            if pool[sid] < need_est:
                continue
            sol = ctx.intercept(s, target, need_est, max_eta=min(ctx.horizon, cap_tick + 5))
            if sol is None:
                continue
            if cap_tick + 1 <= sol.eta <= cap_tick + 4:
                exact_need = _needed_to_take(ctx, target, sol.eta, extra_buffer=2)
                if 0 < exact_need <= pool[sid] and exact_need <= max(14, prod * 12 + 4):
                    sol2 = sol if exact_need == need_est else ctx.intercept(s, target, exact_need, max_eta=min(ctx.horizon, cap_tick + 5))
                    if sol2 is not None and cap_tick + 1 <= sol2.eta <= cap_tick + 5:
                        moves.append([sid, float(sol2.angle), int(exact_need)])
                        _deduct_ships(sid, exact_need, available, raw_available)
                        if claimed is not None:
                            claimed.add(tid)
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
    raw_available: Optional[Dict[int, int]] = None,
    claimed: Optional[Set[int]] = None,
    allow_reserve_wait: bool = True,
) -> bool:
    if not ctx.neutral_planets:
        return False
    candidates: List[List[Any]] = []
    for p in ctx.neutral_planets:
        pid = int(p[0])
        if claimed is not None and pid in claimed:
            continue
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

    scored: List[Tuple[float, List[Any], int, int, int]] = []
    for target in candidates:
        best_s = -1e9
        best_need = 0
        best_eta = 999
        best_sid = -1
        for src in ctx.my_planets:
            sid = int(src[0])
            av = available.get(sid, 0)
            if av < 2 and total_surplus < 4:
                continue
            sc, need, eta = _score_neutral_roi(ctx, src, target, max(av, total_surplus))
            if sc <= -1e8:
                continue
            if filter_mode == "high_prod":
                sc *= 1.0 + 0.18 * float(target[6])
            elif filter_mode == "comet":
                sc += 5.0
            if sc > best_s:
                best_s = sc
                best_need = need
                best_eta = eta
                best_sid = sid
        if best_s > -1e8 and best_sid >= 0:
            scored.append((best_s, target, best_need, best_eta, best_sid))

    if not scored:
        return False

    scored.sort(key=lambda x: x[0], reverse=True)

    reserved_sources: Set[int] = set()
    best_unaffordable_score = -1e9
    if allow_reserve_wait and ctx.step < 80 and filter_mode in ("roi", "high_prod"):
        top_score, top_target, top_need, _, top_sid = scored[0]
        if total_surplus < top_need:
            my_prod = max(1, ctx.my_prod)
            wait_ticks = math.ceil((top_need - total_surplus) / my_prod)
            if 1 <= wait_ticks <= 4 and int(top_target[6]) >= 2:
                best_unaffordable_score = top_score
                reserved_sources.add(top_sid)

    launched = 0
    for sc, target, need, _, sid in scored:
        if launched >= max_targets:
            break
        if best_unaffordable_score > -1e8 and sc < best_unaffordable_score * 0.78:
            if sid in reserved_sources and sum(available.values()) < need + 10:
                continue
        if _try_coordinated_capture(
            ctx,
            target,
            available,
            moves,
            max_sources=2,
            extra_buffer=1,
            raw_available=raw_available,
            claimed=claimed,
        ):
            launched += 1
    return launched > 0


def _do_attack_enemy(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    *,
    target_mode: str = "roi",
    max_targets: int = 1,
    all_in: bool = False,
    min_roi: float = 1.0,
    raw_available: Optional[Dict[int, int]] = None,
    claimed: Optional[Set[int]] = None,
) -> bool:
    if not ctx.enemy_planets:
        return False
    candidates = [
        p for p in ctx.enemy_planets
        if ctx.futures[int(p[0])].final_owner != ctx.player
        and (claimed is None or int(p[0]) not in claimed)
    ]
    if not candidates:
        return False

    mode_key = "roi" if target_mode == "weakest" else target_mode
    scored: List[Tuple[float, List[Any], int, int]] = []
    for target in candidates:
        roi, need, eta = _score_enemy_attack(ctx, target, target_mode=mode_key)
        if roi > -1e8:
            scored.append((roi, target, need, eta))

    if not scored:
        return False
    scored.sort(key=lambda x: x[0], reverse=True)

    # Allow decisive high-ROI strikes to draw from raw_available (no artificial border freeze)
    pool = raw_available if (raw_available is not None and not ctx.is_ffa) else available

    if all_in:
        best_roi, target, _, _ = scored[0]
        if best_roi < min_roi * 0.6:
            return False
        did_any = False
        srcs = sorted(ctx.my_planets, key=lambda s: distance((s[2], s[3]), (target[2], target[3])))
        for src in srcs:
            sid = int(src[0])
            avail = pool.get(sid, 0)
            send = int(avail * 0.72)
            if send >= 6:
                if _try_single_send(
                    ctx, src, target, send, pool, moves, min_ships=6, raw_available=available
                ) > 0:
                    did_any = True
        if did_any and claimed is not None:
            claimed.add(int(target[0]))
        return did_any

    launched = 0
    for roi, target, _, _ in scored:
        if launched >= max_targets:
            break
        if roi < min_roi:
            break
        # Eliminations and super-high-ROI enemy targets may also draw from raw_available in 4P
        use_pool = raw_available if (raw_available is not None and (roi >= 2.2 or ctx.planets_by_player.get(int(target[1]), 99) == 1)) else pool
        other_pool = available if use_pool is raw_available else raw_available
        if _try_coordinated_capture(
            ctx,
            target,
            use_pool,
            moves,
            max_sources=3,
            extra_buffer=2,
            raw_available=other_pool,
            claimed=claimed,
        ):
            launched += 1
    return launched > 0


def _do_unified_expand_and_attack(
    ctx: TacticalContext,
    available: Dict[int, int],
    raw_available: Dict[int, int],
    moves: List[List[float]],
    claimed: Set[int],
    *,
    max_total_targets: int = 3,
    enemy_mult: float = 1.0,
    min_enemy_roi: float = 1.35,
    enemy_target_mode: str = "roi",
    allow_reserve_wait: bool = True,
) -> bool:
    """Score safe neutrals AND enemy planets on a unified ROI scale so we always pick the highest-EV play."""
    total_avail = sum(available.values())
    total_raw = sum(raw_available.values())
    if total_raw < 3:
        return False

    # (score, is_enemy, target, needed, best_src_id)
    unified: List[Tuple[float, bool, List[Any], int, int]] = []

    # 1. Candidate neutrals
    for target in ctx.neutral_planets:
        tid = int(target[0])
        if tid in claimed or ctx.futures[tid].final_owner == ctx.player:
            continue
        best_s = -1e9
        best_need = 0
        best_sid = -1
        for src in ctx.my_planets:
            sid = int(src[0])
            av = available.get(sid, 0)
            if av < 2 and total_avail < 4:
                continue
            sc, need, _ = _score_neutral_roi(ctx, src, target, max(av, total_avail))
            if sc > best_s:
                best_s = sc
                best_need = need
                best_sid = sid
        if best_s > -1e8 and best_sid >= 0:
            unified.append((best_s, False, target, best_need, best_sid))

    # 2. Candidate enemy planets
    if enemy_mult > 0.0 and ctx.enemy_planets:
        for target in ctx.enemy_planets:
            tid = int(target[0])
            if tid in claimed or ctx.futures[tid].final_owner == ctx.player:
                continue
            roi, need, _ = _score_enemy_attack(ctx, target, target_mode=enemy_target_mode)
            if roi <= -1e8:
                continue
            # Early 4P FFA discipline: discount non-elimination enemy fights before step 25
            owner = int(target[1])
            is_elim = ctx.planets_by_player.get(owner, 99) == 1
            eff_mult = enemy_mult
            if ctx.is_ffa and ctx.step < 25 and not is_elim:
                if need > int(target[6]) * 3 + 4:
                    eff_mult *= 0.50
            adj_roi = roi * eff_mult
            if adj_roi < min_enemy_roi and not is_elim:
                continue
            best_src = min(ctx.my_planets, key=lambda s: distance((s[2], s[3]), (target[2], target[3])))
            unified.append((adj_roi, True, target, need, int(best_src[0])))

    if not unified:
        return False

    unified.sort(key=lambda x: x[0], reverse=True)

    # Opening / early-mid patience: if the #1 target is an unaffordable high-ROI neutral
    # we can afford in <= 4 ticks, hold off on inferior neutrals!
    reserved_sources: Set[int] = set()
    best_unaffordable_score = -1e9
    if allow_reserve_wait and ctx.step < 75 and not unified[0][1]:
        top_sc, _, top_t, top_need, top_sid = unified[0]
        if total_avail < top_need and int(top_t[6]) >= 2:
            wait_ticks = math.ceil((top_need - total_avail) / max(1, ctx.my_prod))
            if 1 <= wait_ticks <= 4:
                best_unaffordable_score = top_sc
                reserved_sources.add(top_sid)

    launched = 0
    for sc, is_enemy, target, need, sid in unified:
        if launched >= max_total_targets:
            break
        if not is_enemy and best_unaffordable_score > -1e8 and sc < best_unaffordable_score * 0.78:
            if sid in reserved_sources and sum(available.values()) < need + 10:
                continue
        if is_enemy:
            is_elim = ctx.planets_by_player.get(int(target[1]), 99) == 1
            use_pool = raw_available if (not ctx.is_ffa or is_elim or sc >= 2.2) else available
            other_pool = available if use_pool is raw_available else raw_available
            ok = _try_coordinated_capture(
                ctx,
                target,
                use_pool,
                moves,
                max_sources=3,
                extra_buffer=2,
                raw_available=other_pool,
                claimed=claimed,
            )
        else:
            ok = _try_coordinated_capture(
                ctx,
                target,
                available,
                moves,
                max_sources=2,
                extra_buffer=1,
                raw_available=raw_available,
                claimed=claimed,
            )
        if ok:
            launched += 1

    return launched > 0


def _do_feed_frontline(
    ctx: TacticalContext,
    available: Dict[int, int],
    moves: List[List[float]],
    raw_available: Optional[Dict[int, int]] = None,
    min_stack: int = 24,
) -> bool:
    """Short-hop anti-ping-pong consolidation from deeply safe backline to frontline planets."""
    if len(ctx.my_planets) < 2 or not ctx.enemy_planets or ctx.step < 25:
        return False

    threatened_ids = {int(t[0][0]) for t in ctx.threatened_my}
    d_map = ctx.nearest_enemy_dist
    min_border_d = min(d_map.get(int(p[0]), 140.0) for p in ctx.my_planets)

    frontline = [
        p for p in ctx.my_planets
        if d_map.get(int(p[0]), 140.0) <= min_border_d + 12.0
        and int(p[0]) not in {int(d[0][0]) for d in ctx.doomed_my}
    ]
    if not frontline:
        return False

    did_any = False
    for p in ctx.my_planets:
        pid = int(p[0])
        if pid in threatened_ids:
            continue
        my_d = d_map.get(pid, 140.0)
        # Must be genuinely safe backline (prevents frontline ping-pong!)
        if my_d < min_border_d + 16.0:
            continue
        # Don't immediately re-export ships if friendly reinforcement is already inbound here
        if any(a.owner == ctx.player for a in ctx.arrivals.get(pid, ())):
            continue
        avail = available.get(pid, 0)
        threshold = max(min_stack, int(p[6]) * 6)
        if avail < threshold:
            continue

        # Short-hop only: prefer a frontline planet strictly closer to enemy and within <= 6 ticks
        valid_fronts = [
            fp for fp in frontline
            if int(fp[0]) != pid and d_map.get(int(fp[0]), 140.0) + 12.0 <= my_d
        ]
        if not valid_fronts:
            continue
        best_front = min(valid_fronts, key=lambda fp: distance((p[2], p[3]), (fp[2], fp[3])))
        send = int(avail * 0.55)
        if send < 16:
            continue
        sol = ctx.intercept(p, best_front, send, max_eta=6)
        if sol is not None and sol.eta <= 6:
            moves.append([pid, float(sol.angle), int(send)])
            _deduct_ships(pid, send, available, raw_available)
            did_any = True
    return did_any


def run_strategic_stance(
    ctx: TacticalContext,
    *,
    max_targets: int = 3,
    expand_only: bool = False,
    expand_mode: str = "roi",
    attack_mode: str = "roi",
    enemy_mult: float = 1.0,
    min_attack_roi: float = 1.40,
    all_in: bool = False,
    feed_frontline: bool = True,
    feed_min_stack: int = 24,
    reserve_scale: float = 1.0,
    allow_reserve_wait: bool = True,
) -> List[List[float]]:
    """Execute a complete, defense-guarded full-turn strategic stance."""
    if not ctx.my_planets:
        return []

    raw_available = dict(ctx.raw_surplus)
    if reserve_scale == 1.0:
        available = dict(ctx.surplus)
    else:
        available = {
            pid: max(0, raw_available[pid] - int(ctx.border_reserve.get(pid, 0) * reserve_scale))
            for pid in raw_available
        }

    moves: List[List[float]] = []
    claimed: Set[int] = set()

    # 1. Mandatory emergency defense + last-tick evacuation + free vulture snipes
    saved_pids = _do_defend_urgent(ctx, available, moves, raw_available=raw_available, claimed=claimed)
    _do_evacuate_doomed(ctx, available, moves, saved_pids=saved_pids, raw_available=raw_available)
    _do_vulture_snipe(ctx, available, moves, raw_available=raw_available, claimed=claimed)

    # 2. Endgame discipline
    if ctx.remaining_steps <= 40:
        if not expand_only and ctx.enemy_planets:
            end_mode = "closest" if ctx.my_total_ships <= ctx.leader_ships * 1.1 else attack_mode
            _do_attack_enemy(
                ctx,
                available,
                moves,
                target_mode=end_mode,
                max_targets=max(2, max_targets),
                all_in=all_in,
                min_roi=0.75,
                raw_available=raw_available,
                claimed=claimed,
            )
        return moves

    # 3. Expansion / Offense
    if all_in and ctx.enemy_planets:
        _do_attack_enemy(
            ctx,
            available,
            moves,
            target_mode=attack_mode,
            all_in=True,
            min_roi=0.75,
            raw_available=raw_available,
            claimed=claimed,
        )
    elif expand_only or expand_mode != "roi":
        _do_expand_neutrals(
            ctx,
            available,
            moves,
            max_targets=max_targets,
            filter_mode=expand_mode,
            raw_available=raw_available,
            claimed=claimed,
            allow_reserve_wait=allow_reserve_wait,
        )
    else:
        _do_unified_expand_and_attack(
            ctx,
            available,
            raw_available,
            moves,
            claimed,
            max_total_targets=max_targets,
            enemy_mult=enemy_mult,
            min_enemy_roi=min_attack_roi,
            enemy_target_mode=attack_mode,
            allow_reserve_wait=allow_reserve_wait,
        )

    # 4. Short-hop frontline consolidation
    if feed_frontline:
        _do_feed_frontline(
            ctx,
            available,
            moves,
            raw_available=raw_available,
            min_stack=feed_min_stack,
        )

    return moves


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

    total_surplus = sum(ctx.raw_surplus.values())
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

    if ctx.doomed_my and any(int(p[5]) > 1 and (f.capture_tick is not None and f.capture_tick <= 1) for p, f in ctx.doomed_my):
        mask[MacroAction.COUNTER_EVACUATE] = 1

    if has_surplus and ctx.vulture_targets:
        mask[MacroAction.VULTURE_SNIPE] = 1

    if has_surplus and has_enemy:
        mask[MacroAction.ATTACK_WEAKEST_ENEMY] = 1
        mask[MacroAction.ATTACK_HIGH_PROD_ENEMY] = 1
        mask[MacroAction.ATTACK_CLOSEST_ENEMY] = 1
        if sum(ctx.surplus.values()) >= 14:
            mask[MacroAction.COORDINATED_ALL_IN] = 1
        mask[MacroAction.AGGRESSIVE_SKIRMISH] = 1

    if len(ctx.my_planets) >= 2 and has_enemy and any(v >= 16 for v in ctx.surplus.values()):
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
    raw_available = dict(ctx.raw_surplus)
    moves: List[List[float]] = []
    claimed: Set[int] = set()

    if act == MacroAction.EXPAND_BEST_ROI_SINGLE:
        _do_expand_neutrals(
            ctx, available, moves, max_targets=1, filter_mode="roi", raw_available=raw_available, claimed=claimed
        )
    elif act == MacroAction.EXPAND_BEST_ROI_MULTI:
        return run_strategic_stance(
            ctx,
            max_targets=3,
            expand_only=True,
            expand_mode="roi",
            feed_frontline=True,
            reserve_scale=0.5,
            allow_reserve_wait=True,
        )
    elif act == MacroAction.EXPAND_HIGH_PROD:
        return run_strategic_stance(
            ctx,
            max_targets=2,
            expand_only=True,
            expand_mode="high_prod",
            feed_frontline=False,
            reserve_scale=0.7,
            allow_reserve_wait=True,
        )
    elif act == MacroAction.EXPAND_ORBITING:
        _do_expand_neutrals(
            ctx, available, moves, max_targets=2, filter_mode="orbiting", raw_available=raw_available, claimed=claimed
        )
    elif act == MacroAction.EXPAND_COMET_SNIPE:
        _do_expand_neutrals(
            ctx, available, moves, max_targets=2, filter_mode="comet", raw_available=raw_available, claimed=claimed
        )
    elif act == MacroAction.DEFEND_URGENT:
        saved = _do_defend_urgent(ctx, available, moves, raw_available=raw_available, claimed=claimed)
        _do_evacuate_doomed(ctx, available, moves, saved_pids=saved, raw_available=raw_available)
    elif act == MacroAction.COUNTER_EVACUATE:
        _do_evacuate_doomed(ctx, available, moves, raw_available=raw_available)
    elif act == MacroAction.VULTURE_SNIPE:
        _do_vulture_snipe(ctx, available, moves, raw_available=raw_available, claimed=claimed)
    elif act == MacroAction.ATTACK_WEAKEST_ENEMY:
        return run_strategic_stance(
            ctx,
            max_targets=2,
            attack_mode="roi",
            enemy_mult=1.25,
            min_attack_roi=1.10,
            feed_frontline=True,
            reserve_scale=0.5,
        )
    elif act == MacroAction.ATTACK_HIGH_PROD_ENEMY:
        return run_strategic_stance(
            ctx,
            max_targets=2,
            attack_mode="leader" if ctx.is_ffa else "high_prod",
            enemy_mult=1.25,
            min_attack_roi=1.10,
            feed_frontline=True,
            reserve_scale=0.6,
        )
    elif act == MacroAction.ATTACK_CLOSEST_ENEMY:
        return run_strategic_stance(
            ctx,
            max_targets=2,
            attack_mode="closest",
            enemy_mult=1.20,
            min_attack_roi=1.05,
            feed_frontline=True,
        )
    elif act == MacroAction.COORDINATED_ALL_IN:
        return run_strategic_stance(
            ctx,
            max_targets=2,
            attack_mode="roi",
            all_in=True,
            feed_frontline=False,
            reserve_scale=0.2,
        )
    elif act == MacroAction.FEED_FRONTLINE:
        # Defensive turtle & consolidation stance
        return run_strategic_stance(
            ctx,
            max_targets=1,
            expand_only=True,
            feed_frontline=True,
            feed_min_stack=18,
            reserve_scale=1.5,
        )
    elif act == MacroAction.BALANCED_MACRO:
        from orbit_wars_bot.heuristic_bot import heuristic_moves

        return heuristic_moves(state, player, ctx=ctx)
    elif act == MacroAction.AGGRESSIVE_SKIRMISH:
        return run_strategic_stance(
            ctx,
            max_targets=3,
            attack_mode="roi",
            enemy_mult=1.35,
            min_attack_roi=0.95,
            feed_frontline=True,
            reserve_scale=0.4,
            allow_reserve_wait=False,
        )

    return moves
