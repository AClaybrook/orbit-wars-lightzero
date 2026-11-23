"""3v1 Coalition coordination wrappers for Orbit Wars 4-player matches.

Models three independent agents teaming up against a single "boss" player
without merging resources or violating any Orbit Wars rules:
  - Each ally remains a separate player with its own planets and fleets.
  - Mode ``truce``: Non-aggression pact (allies never attack each other's
    planets) plus neutral-capture deconfliction so allies stop fighting over
    the same neutral targets.
  - Mode ``focus_fire``: Truce + deconfliction + a shared coalition tactical
    plan that selects a primary Boss target and synchronizes multi-angle
    launches from all three allies to arrive together.
  - ``comm_latency`` (ticks): Models imperfect/delayed communication by
    delivering coalition plans and neutral claims ``comm_latency`` ticks late.
  - Once the boss has 0 planets and 0 fleets, the coalition automatically
    dissolves and the remaining players finish the FFA normally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from orbit_wars_bot.physics import (
    BOARD_SIZE,
    CENTER,
    DEFAULT_SHIP_SPEED,
    SUN_RADIUS,
    build_comet_lookup,
    fleet_speed,
    point_to_segment_distance,
    predict_all_planet_tracks,
    predict_fleet_arrivals,
    simulate_planet_future,
    solve_intercept,
    swept_pair_hit,
)

DEFAULT_ANGULAR_VELOCITY = 0.04

AgentCallable = Callable[[Dict[str, Any], Optional[Dict[str, Any]]], List[Dict[str, Any]]]


@dataclass(frozen=True)
class CoalitionMessage:
    """Shared tactical broadcast produced by the coalition at tick ``step``."""

    step: int
    boss_alive: bool
    primary_target_id: Optional[int]
    secondary_target_id: Optional[int]
    sync_arrival_tick: int
    ally_claimed_neutrals: Dict[int, int]
    ally_boss_inbound: Dict[int, int]


@dataclass
class CoalitionStats:
    """Per-match coordination telemetry."""

    friendly_fire_blocked: int = 0
    neutral_conflicts_avoided: int = 0
    sync_volleys_launched: int = 0
    sync_ships_launched: int = 0
    boss_eliminated_step: Optional[int] = None
    last_step: int = 0
    sync_holds: int = 0


def _predict_order_target(
    from_planet: Sequence[float],
    angle: float,
    ships: int,
    planets: Sequence[Sequence[float]],
    tracks: Dict[int, List[Optional[Tuple[float, float]]]],
    max_dt: int = 28,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Optional[int]:
    """Return the planet_id first struck by a launch order, or None if miss/sun/OOB."""
    if ships <= 0:
        return None
    spd = fleet_speed(ships, max_speed)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    fx = float(from_planet[2]) + cos_a * (float(from_planet[4]) + 0.1)
    fy = float(from_planet[3]) + sin_a * (float(from_planet[4]) + 0.1)
    vx = cos_a * spd
    vy = sin_a * spd

    specs = [
        (int(p[0]), float(p[4]), tracks[int(p[0])], spd + 4.5 + float(p[4]))
        for p in planets
        if int(p[0]) in tracks
    ]
    horizon = min(max_dt, min((len(tr) - 1 for _, _, tr, _ in specs), default=0))
    for dt in range(1, horizon + 1):
        ox, oy = fx, fy
        fx += vx
        fy += vy
        old_pos = (ox, oy)
        new_pos = (fx, fy)
        for pid, pr, tr, cull in specs:
            p_old = tr[dt - 1]
            if p_old is None or p_old[0] < 0.0:
                continue
            dx = ox - p_old[0]
            if dx > cull or dx < -cull:
                continue
            dy = oy - p_old[1]
            if dy > cull or dy < -cull:
                continue
            p_new = tr[dt]
            if p_new is None:
                if swept_pair_hit(old_pos, new_pos, p_old, p_old, pr):
                    return pid
                continue
            if swept_pair_hit(old_pos, new_pos, p_old, p_new, pr):
                return pid
        if fx < 0.0 or fx > BOARD_SIZE or fy < 0.0 or fy > BOARD_SIZE:
            return None
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            return None
    return None


def _compute_safe_keep(
    planet: Sequence[float],
    arrivals: Sequence[Any],
    player_id: int,
    ally_ids: Sequence[int],
    horizon: int = 18,
) -> int:
    """Minimum garrison ships needed to survive non-ally incoming fleets."""
    hostile_arrivals = [a for a in arrivals if a.owner != player_id and a.owner not in ally_ids]
    if not hostile_arrivals:
        return 1
    ships = int(planet[5])
    # Binary search smallest garrison that keeps ownership against hostile arrivals
    lo, hi = 1, ships
    while lo < hi:
        mid = (lo + hi) // 2
        p_test = list(planet)
        p_test[5] = mid
        fut = simulate_planet_future(p_test, arrivals, horizon=horizon)
        if fut.will_be_captured_by is None and fut.final_owner == player_id:
            hi = mid
        else:
            lo = mid + 1
    return min(ships, lo + 2)


class CoalitionCoordinator:
    """Coordinates a coalition of independent ally agents against a single boss."""

    def __init__(
        self,
        boss_id: int = 0,
        ally_ids: Sequence[int] = (1, 2, 3),
        mode: str = "focus_fire",
        comm_latency: int = 0,
        start_focus_step: int = 12,
        min_volley_ships: int = 8,
        compliance: float = 1.0,
        info_noise: float = 0.0,
        rng_seed: int = 42,
    ) -> None:
        if mode not in ("none", "truce", "focus_fire"):
            raise ValueError(f"Unsupported coalition mode: {mode!r}")
        if comm_latency < 0:
            raise ValueError("comm_latency must be >= 0")
        if not (0.0 <= compliance <= 1.0):
            raise ValueError("compliance must be in [0, 1]")
        if info_noise < 0.0:
            raise ValueError("info_noise must be >= 0")
        self.boss_id = int(boss_id)
        self.ally_ids = tuple(int(a) for a in ally_ids)
        self.ally_set = set(self.ally_ids)
        self.mode = mode
        self.comm_latency = int(comm_latency)
        self.start_focus_step = int(start_focus_step)
        self.min_volley_ships = int(min_volley_ships)
        self.compliance = float(compliance)
        self.info_noise = float(info_noise)
        self.rng_seed = int(rng_seed)

        self.stats = CoalitionStats()
        self._initial_by_id: Dict[int, List[float]] = {}
        self._messages: Dict[int, CoalitionMessage] = {}
        self._last_computed_step: int = -1
        self._step_cache: Dict[str, Any] = {}

    def _det_rand(self, step: int, tag: int) -> float:
        """Deterministic pseudo-random uniform [0, 1) keyed by (rng_seed, step, tag)."""
        x = (self.rng_seed * 1103515245 + step * 2654435761 + tag * 2246822519) & 0xFFFFFFFF
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        return (x & 0xFFFFFF) / float(0x1000000)

    def reset(self) -> None:
        self.stats = CoalitionStats()
        self._initial_by_id = {}
        self._messages = {}
        self._last_computed_step = -1
        self._step_cache = {}

    def _ensure_step_plan(self, obs: Dict[str, Any], config: Optional[Dict[str, Any]]) -> None:
        step = int(obs.get("step", 0))
        if step == 0 and self._last_computed_step != 0:
            self.reset()
        if step == self._last_computed_step:
            return

        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        comets = obs.get("comets", [])
        cfg = config or {}
        ang_vel = float(cfg.get("angular_velocity", DEFAULT_ANGULAR_VELOCITY))
        max_speed = float(cfg.get("ship_speed", DEFAULT_SHIP_SPEED))

        if step == 0 or not self._initial_by_id:
            self._initial_by_id = {int(p[0]): list(p) for p in planets}

        comet_lookup = build_comet_lookup(comets)
        horizon = 26
        tracks = predict_all_planet_tracks(
            planets, horizon, step, ang_vel, self._initial_by_id, comet_lookup
        )
        arrivals = predict_fleet_arrivals(
            fleets,
            planets,
            step,
            ang_vel,
            self._initial_by_id,
            comet_lookup=comet_lookup,
            tracks=tracks,
            max_horizon=horizon,
            max_speed=max_speed,
        )

        by_id = {int(p[0]): p for p in planets}
        futures = {
            int(p[0]): simulate_planet_future(p, arrivals.get(int(p[0]), ()), horizon=horizon)
            for p in planets
        }

        boss_planets = [p for p in planets if int(p[1]) == self.boss_id]
        boss_fleets = [f for f in fleets if int(f[1]) == self.boss_id]
        boss_alive = bool(boss_planets or boss_fleets)
        if not boss_alive and self.stats.boss_eliminated_step is None and step > 0:
            self.stats.boss_eliminated_step = step
        self.stats.last_step = step

        # Identify neutral planets already going to be captured by an ally fleet
        ally_claimed_neutrals: Dict[int, int] = {}
        for p in planets:
            pid = int(p[0])
            if int(p[1]) == -1:
                fut = futures[pid]
                if fut.final_owner in self.ally_set:
                    ally_claimed_neutrals[pid] = fut.final_owner

        # Track total coalition ships already inbound to each boss planet
        ally_boss_inbound: Dict[int, int] = {}
        for p in boss_planets:
            pid = int(p[0])
            inb = sum(a.ships for a in arrivals.get(pid, ()) if a.owner in self.ally_set)
            if inb > 0:
                ally_boss_inbound[pid] = inb

        # Pick synchronized focus-fire target(s) among Boss planets
        primary_id: Optional[int] = None
        secondary_id: Optional[int] = None
        sync_tick = step + 10

        prev_msg = self._messages.get(step - 1) if step > 0 else None
        if boss_alive and boss_planets and self.mode == "focus_fire":
            ally_planets_by_owner: Dict[int, List[Sequence[float]]] = {a: [] for a in self.ally_ids}
            for p in planets:
                own = int(p[1])
                if own in self.ally_set and int(p[5]) >= self.min_volley_ships:
                    ally_planets_by_owner[own].append(p)

            # Score each non-doomed Boss planet by (prod, softness, multi-ally reachability)
            scored: List[Tuple[float, int, int]] = []
            for bp in boss_planets:
                bpid = int(bp[0])
                fut = futures[bpid]
                if fut.final_owner in self.ally_set:
                    # Already guaranteed to fall to coalition fleets
                    continue
                prod = float(bp[6])
                eff_def = float(max(1, fut.final_ships))
                if self.info_noise > 0.0:
                    jitter = 1.0 + (2.0 * self._det_rand(step, bpid + 97) - 1.0) * self.info_noise
                    eff_def = max(1.0, eff_def * jitter)
                bx, by = float(bp[2]), float(bp[3])

                # Estimate per-ally closest strike ETA & available force
                ally_best_etas: List[int] = []
                total_nearby_force = 0
                for a_id in self.ally_ids:
                    best_eta = 999
                    for ap in ally_planets_by_owner[a_id]:
                        d = math.hypot(float(ap[2]) - bx, float(ap[3]) - by)
                        est_eta = max(2, int(round(d / max(3.0, max_speed * 0.72))))
                        if est_eta <= 18:
                            total_nearby_force += int(ap[5])
                            if est_eta < best_eta:
                                best_eta = est_eta
                    if best_eta < 999:
                        ally_best_etas.append(best_eta)

                n_allies_in_range = len(ally_best_etas)
                if n_allies_in_range == 0:
                    continue

                # Prefer targets reachable by 2-3 allies where combined nearby force can break defense
                coord_mult = 1.0 + 0.45 * (n_allies_in_range - 1)
                break_mult = 1.35 if (total_nearby_force + ally_boss_inbound.get(bpid, 0)) > eff_def * 1.15 else 0.75
                inb_bonus = 1.0 + min(0.6, ally_boss_inbound.get(bpid, 0) / max(10.0, eff_def))
                # Slight stickiness to previous focus target to finish coordinated volleys
                sticky = 1.28 if (prev_msg is not None and prev_msg.primary_target_id == bpid) else 1.0
                score = ((prod + 1.8) / (eff_def + 8.0)) * coord_mult * break_mult * inb_bonus * sticky

                # Synchronize arrival on the 2nd-closest ally's ETA (so 2-3 allies hit together)
                ally_best_etas.sort()
                ref_eta = ally_best_etas[min(1, len(ally_best_etas) - 1)]
                ref_eta = max(5, min(15, ref_eta))
                scored.append((score, bpid, ref_eta))

            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                primary_id = scored[0][1]
                # Preserve existing sync window while primary target stays the same & window is live
                if (
                    prev_msg is not None
                    and prev_msg.primary_target_id == primary_id
                    and prev_msg.sync_arrival_tick > step + 1
                ):
                    sync_tick = prev_msg.sync_arrival_tick
                else:
                    sync_tick = step + scored[0][2]
                if len(scored) > 1:
                    secondary_id = scored[1][1]

        msg = CoalitionMessage(
            step=step,
            boss_alive=boss_alive,
            primary_target_id=primary_id,
            secondary_target_id=secondary_id,
            sync_arrival_tick=sync_tick,
            ally_claimed_neutrals=ally_claimed_neutrals,
            ally_boss_inbound=ally_boss_inbound,
        )
        self._messages[step] = msg
        self._last_computed_step = step
        self._step_cache = {
            "planets": planets,
            "by_id": by_id,
            "tracks": tracks,
            "arrivals": arrivals,
            "futures": futures,
            "comet_lookup": comet_lookup,
            "ang_vel": ang_vel,
            "max_speed": max_speed,
            "boss_alive": boss_alive,
        }

    def get_delayed_message(self, step: int) -> Optional[CoalitionMessage]:
        """Return the coalition broadcast visible at ``step`` given ``comm_latency``."""
        if step < self.comm_latency:
            return None
        target_step = step - self.comm_latency
        if target_step in self._messages:
            return self._messages[target_step]
        # Fallback to latest message at or before target_step
        earlier = [s for s in self._messages if s <= target_step]
        return self._messages[max(earlier)] if earlier else None

    def act(
        self,
        player_id: int,
        obs: Dict[str, Any],
        config: Optional[Dict[str, Any]],
        base_agent: AgentCallable,
    ) -> List[Dict[str, Any]]:
        """Produce coordinated actions for ``player_id`` wrapping ``base_agent``."""
        if self.mode == "none" or player_id not in self.ally_set:
            return base_agent(obs, config)

        self._ensure_step_plan(obs, config)
        step = int(obs.get("step", 0))
        cache = self._step_cache
        if not cache["boss_alive"]:
            return base_agent(obs, config)

        msg = self.get_delayed_message(step)
        raw_orders = base_agent(obs, config) or []

        planets = cache["planets"]
        by_id: Dict[int, Sequence[float]] = cache["by_id"]
        tracks = cache["tracks"]
        arrivals = cache["arrivals"]
        futures = cache["futures"]
        comet_lookup = cache["comet_lookup"]
        ang_vel = cache["ang_vel"]
        max_speed = cache["max_speed"]

        # Step 1: Truce + Neutral Deconfliction filter on base_agent's orders
        kept_orders: List[Dict[str, Any]] = []
        used_sources: set[int] = set()
        claimed_neutrals = msg.ally_claimed_neutrals if msg is not None else {}

        for order in raw_orders:
            if isinstance(order, dict):
                src_id = int(order["from_planet_id"])
                angle = float(order["angle"])
                ships = int(order["ships"])
            else:
                src_id = int(order[0])
                angle = float(order[1])
                ships = int(order[2])
            src_p = by_id.get(src_id)
            if src_p is None or int(src_p[1]) != player_id or ships <= 0:
                continue

            tid = _predict_order_target(
                src_p, angle, ships, planets, tracks, max_dt=26, max_speed=max_speed
            )
            if tid is not None and tid in by_id:
                t_owner = int(by_id[tid][1])
                t_fut_owner = futures[tid].final_owner
                # Rule 1: Non-aggression pact — never attack another ally's planet
                if (t_owner in self.ally_set and t_owner != player_id) or (
                    t_fut_owner in self.ally_set and t_fut_owner != player_id
                ):
                    self.stats.friendly_fire_blocked += 1
                    continue
                # Rule 2: Neutral deconfliction — don't race an ally's already-winning claim
                if t_owner == -1:
                    claimer = claimed_neutrals.get(tid)
                    if claimer is not None and claimer != player_id:
                        self.stats.neutral_conflicts_avoided += 1
                        continue

            kept_orders.append([src_id, angle, ships])
            used_sources.add(src_id)

        if self.mode != "focus_fire" or msg is None or step < self.start_focus_step:
            return kept_orders

        # Imperfect compliance: ally may ignore the focus-fire directive this tick
        if self.compliance < 1.0 and self._det_rand(step, player_id + 313) > self.compliance:
            return kept_orders

        # Step 2: Synchronized multi-ally focus-fire against primary/secondary Boss target
        target_ids = [
            tid
            for tid in (msg.primary_target_id, msg.secondary_target_id)
            if tid is not None and tid in by_id and int(by_id[tid][1]) == self.boss_id
        ]
        if not target_ids:
            return kept_orders

        desired_dt = max(1, msg.sync_arrival_tick - step)
        my_planets = [p for p in planets if int(p[1]) == player_id]

        # Re-evaluate kept orders: if a planet was about to spend ships on a non-Boss
        # target right as the synchronized volley window opens (or 1-2 ticks before),
        # redirect/hold it for the coordinated hammer blow!
        if step >= self.start_focus_step + 4:
            filtered_kept: List[List[Any]] = []
            used_sources.clear()
            primary_p = by_id[target_ids[0]]
            for order in kept_orders:
                src_id = int(order[0])
                src_p = by_id[src_id]
                tid = _predict_order_target(
                    src_p, float(order[1]), int(order[2]), planets, tracks, max_dt=26, max_speed=max_speed
                )
                is_boss_or_def = tid is not None and tid in by_id and int(by_id[tid][1]) in (self.boss_id, player_id)
                if not is_boss_or_def:
                    # Check if this planet is in prime range to join the synchronized boss strike
                    d_boss = math.hypot(float(src_p[2]) - float(primary_p[2]), float(src_p[3]) - float(primary_p[3]))
                    est_eta = max(2, int(round(d_boss / max(3.0, max_speed * 0.72))))
                    if est_eta <= 16 and abs(est_eta - desired_dt) <= 3 and int(src_p[5]) >= self.min_volley_ships + 4:
                        # Divert this planet to the synchronized coalition volley
                        continue
                filtered_kept.append(order)
                used_sources.add(src_id)
            kept_orders = filtered_kept

        # Launch synchronized volleys from spare capacity on unused owned planets
        for p in my_planets:
            src_id = int(p[0])
            if src_id in used_sources:
                continue
            ships = int(p[5])
            keep = _compute_safe_keep(p, arrivals.get(src_id, ()), player_id, self.ally_ids)
            spare = ships - keep
            if spare < self.min_volley_ships:
                continue

            # Send 88% of spare (keep a small production/buffer seed)
            send_ships = max(self.min_volley_ships, int(round(spare * 0.88)))
            send_ships = min(spare, send_ships)

            launched = False
            for tid in target_ids:
                tgt_p = by_id[tid]
                sol = solve_intercept(
                    p,
                    tgt_p,
                    send_ships,
                    planets,
                    step,
                    ang_vel,
                    self._initial_by_id,
                    comet_lookup=comet_lookup,
                    tracks=tracks,
                    max_eta=20,
                    max_speed=max_speed,
                )
                if sol is None:
                    continue

                # Synchronized window: launch if ETA matches desired_dt (±2), or if we are
                # farther out and must launch now to catch the wave, or if target is close & ripe
                in_sync_window = abs(sol.eta - desired_dt) <= 2
                must_launch_early = (sol.eta > desired_dt + 2) and (sol.eta <= 17) and (send_ships >= 14)
                opportunistic_kill = (sol.eta <= 10) and (send_ships > max(6, futures[tid].final_ships + 2))
                if in_sync_window or must_launch_early or opportunistic_kill:
                    kept_orders.append([src_id, float(sol.angle), int(send_ships)])
                    used_sources.add(src_id)
                    self.stats.sync_volleys_launched += 1
                    self.stats.sync_ships_launched += int(send_ships)
                    launched = True
                    break
                if sol.eta < desired_dt - 2 and desired_dt - sol.eta <= 5:
                    # Wait for the synchronized launch tick so all 3 allies hit together
                    self.stats.sync_holds += 1
                    launched = True
                    break
            if launched:
                continue

        return kept_orders


def create_coalition_team(
    base_agent_factory: Callable[[], AgentCallable],
    boss_id: int = 0,
    ally_ids: Sequence[int] = (1, 2, 3),
    mode: str = "focus_fire",
    comm_latency: int = 0,
    compliance: float = 1.0,
    info_noise: float = 0.0,
    rng_seed: int = 42,
) -> Tuple[CoalitionCoordinator, Dict[int, AgentCallable]]:
    """Build a coordinated team of 3 independent ally agents sharing a coordinator."""
    coord = CoalitionCoordinator(
        boss_id=boss_id,
        ally_ids=ally_ids,
        mode=mode,
        comm_latency=comm_latency,
        compliance=compliance,
        info_noise=info_noise,
        rng_seed=rng_seed,
    )
    agents: Dict[int, AgentCallable] = {}
    for ally_id in ally_ids:
        base_ag = base_agent_factory()

        def _make_wrapper(pid: int, ag: AgentCallable) -> AgentCallable:
            def _wrapped(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> List[Any]:
                return coord.act(pid, obs, config, ag)

            return _wrapped

        agents[int(ally_id)] = _make_wrapper(int(ally_id), base_ag)
    return coord, agents
