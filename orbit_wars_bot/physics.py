"""Exact Orbit Wars physics, orbital/comet prediction, and intercept solvers.

Mirrors the mechanics of ``kaggle_environments.envs.orbit_wars.orbit_wars``:
- Board: 100x100, central sun at (50, 50) with radius 10.0.
- Orbiting planets: planets with ``orbital_radius + radius < 50.0``.
- Empirical orbital phase offset: when observing ``obs.step == t`` and
  submitting moves, ``orbit_wars.py`` computes end-of-tick planet positions at
  ``init_angle + angular_velocity * t`` and increments ``step`` afterward.
  Therefore, after ``dt >= 1`` ticks from ``current_step``, an orbiting planet's
  angle is ``init_angle + angular_velocity * (current_step + dt - 1)``.
- Fleet speed: ``min(max_speed, 1.0 + (max_speed - 1.0) * (ln(ships) / ln(1000)) ** 1.5)``.
- Continuous swept-pair collision detection between moving fleets and planets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Official Orbit Wars constants
BOARD_SIZE: float = 100.0
CENTER: float = BOARD_SIZE / 2.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: int = 1
PLANET_CLEARANCE: float = 7.0
MIN_PLANET_GROUPS: int = 5
MAX_PLANET_GROUPS: int = 10
MIN_STATIC_GROUPS: int = 3
COMET_SPAWN_STEPS: Tuple[int, ...] = (50, 150, 250, 350, 450)
DEFAULT_SHIP_SPEED: float = 6.0
DEFAULT_COMET_SPEED: float = 4.0
DEFAULT_EPISODE_STEPS: int = 500

_LN_1000: float = math.log(1000.0)
_TWO_PI: float = 2.0 * math.pi


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def fleet_speed(ships: int, max_speed: float = DEFAULT_SHIP_SPEED) -> float:
    """Exact Orbit Wars fleet speed as a function of ship count."""
    if ships <= 1:
        return 1.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / _LN_1000) ** 1.5
    return speed if speed < max_speed else max_speed


def point_to_segment_distance(
    p: Tuple[float, float],
    v: Tuple[float, float],
    w: Tuple[float, float],
) -> float:
    """Minimum Euclidean distance from point ``p`` to line segment ``v-w``."""
    dx = w[0] - v[0]
    dy = w[1] - v[1]
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(p[0] - v[0], p[1] - v[1])
    t = ((p[0] - v[0]) * dx + (p[1] - v[1]) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    proj_x = v[0] + t * dx
    proj_y = v[1] + t * dy
    return math.hypot(p[0] - proj_x, p[1] - proj_y)


def swept_pair_hit(
    a: Tuple[float, float],
    b: Tuple[float, float],
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    r: float,
) -> bool:
    """Exact continuous swept-pair collision check mirroring ``orbit_wars.py``.

    Returns True iff a fleet moving linearly ``a -> b`` and a planet moving
    linearly ``p0 -> p1`` come within distance ``r`` for some ``t in [0, 1]``.
    """
    d0x = a[0] - p0[0]
    d0y = a[1] - p0[1]
    dvx = (b[0] - a[0]) - (p1[0] - p0[0])
    dvy = (b[1] - a[1]) - (p1[1] - p0[1])
    qa = dvx * dvx + dvy * dvy
    qb = 2.0 * (d0x * dvx + d0y * dvy)
    qc = d0x * d0x + d0y * d0y - r * r
    if qa < 1e-12:
        return qc <= 0.0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-qb - sq) / (2.0 * qa)
    t2 = (-qb + sq) / (2.0 * qa)
    return t2 >= 0.0 and t1 <= 1.0


def segment_hits_sun(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    margin: float = 0.0,
) -> bool:
    """Return True if segment ``p0 -> p1`` comes within ``SUN_RADIUS + margin`` of center."""
    return point_to_segment_distance((CENTER, CENTER), p0, p1) < (SUN_RADIUS + margin)


def is_orbiting_planet(x: float, y: float, radius: float) -> bool:
    """Return True if a non-comet planet at ``(x, y)`` orbits the central sun."""
    return math.hypot(x - CENTER, y - CENTER) + radius < ROTATION_RADIUS_LIMIT


def _build_comet_lookup(comets: Sequence[Dict[str, Any]]) -> Dict[int, Tuple[List[List[float]], int]]:
    """Map comet planet_id -> (path, current_path_index)."""
    lookup: Dict[int, Tuple[List[List[float]], int]] = {}
    for group in comets:
        idx = int(group["path_index"])
        pids = group["planet_ids"]
        paths = group["paths"]
        for i, pid in enumerate(pids):
            lookup[int(pid)] = (paths[i], idx)
    return lookup


def predict_planet_position(
    planet: Sequence[float],
    dt: int,
    current_step: int,
    angular_velocity: float,
    initial_by_id: Dict[int, Sequence[float]],
    comet_lookup: Optional[Dict[int, Tuple[List[List[float]], int]]] = None,
) -> Optional[Tuple[float, float]]:
    """Predict a planet's or comet's position after ``dt`` ticks from ``current_step``.

    Args:
        planet: Current planet tuple/list ``[id, owner, x, y, radius, ships, prod]``.
        dt: Number of ticks ahead (``0`` = current position, ``dt >= 1`` = end of tick ``dt``).
        current_step: Current ``obs.step`` value.
        angular_velocity: Orbital angular velocity in rad/tick.
        initial_by_id: Map from planet id to its ``initial_planets`` entry.
        comet_lookup: Optional map from comet planet id to ``(path, path_index)``.

    Returns:
        ``(x, y)`` position, or ``None`` if the planet is a comet that has expired
        before tick ``dt``.
    """
    if dt <= 0:
        return (float(planet[2]), float(planet[3]))

    pid = int(planet[0])
    if comet_lookup is not None and pid in comet_lookup:
        path, cur_idx = comet_lookup[pid]
        future_idx = cur_idx + dt
        if future_idx < 0:
            return (-99.0, -99.0)
        if future_idx >= len(path):
            return None
        pt = path[future_idx]
        return (float(pt[0]), float(pt[1]))

    init_p = initial_by_id.get(pid)
    if init_p is None:
        return (float(planet[2]), float(planet[3]))

    dx = float(init_p[2]) - CENTER
    dy = float(init_p[3]) - CENTER
    orb_r = math.hypot(dx, dy)
    if orb_r + float(planet[4]) < ROTATION_RADIUS_LIMIT:
        init_angle = math.atan2(dy, dx)
        # Empirical phase offset: tick 1 uses step = current_step, tick dt uses current_step + dt - 1.
        target_angle = init_angle + angular_velocity * (current_step + dt - 1)
        return (
            CENTER + orb_r * math.cos(target_angle),
            CENTER + orb_r * math.sin(target_angle),
        )
    return (float(planet[2]), float(planet[3]))


def predict_all_planet_tracks(
    planets: Sequence[Sequence[float]],
    max_dt: int,
    current_step: int,
    angular_velocity: float,
    initial_by_id: Dict[int, Sequence[float]],
    comet_lookup: Optional[Dict[int, Tuple[List[List[float]], int]]] = None,
) -> Dict[int, List[Optional[Tuple[float, float]]]]:
    """Precompute ``tracks[pid][dt]`` for ``dt in 0..max_dt`` for every planet."""
    tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    for p in planets:
        pid = int(p[0])
        if comet_lookup is not None and pid in comet_lookup:
            path, cur_idx = comet_lookup[pid]
            seq: List[Optional[Tuple[float, float]]] = [(float(p[2]), float(p[3]))]
            for dt in range(1, max_dt + 1):
                fidx = cur_idx + dt
                if fidx < 0:
                    seq.append((-99.0, -99.0))
                elif fidx >= len(path):
                    seq.append(None)
                else:
                    pt = path[fidx]
                    seq.append((float(pt[0]), float(pt[1])))
            tracks[pid] = seq
            continue

        init_p = initial_by_id.get(pid)
        if init_p is not None:
            dx = float(init_p[2]) - CENTER
            dy = float(init_p[3]) - CENTER
            orb_r = math.hypot(dx, dy)
            if orb_r + float(p[4]) < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(dy, dx)
                seq = [(float(p[2]), float(p[3]))]
                for dt in range(1, max_dt + 1):
                    ang = init_angle + angular_velocity * (current_step + dt - 1)
                    seq.append((CENTER + orb_r * math.cos(ang), CENTER + orb_r * math.sin(ang)))
                tracks[pid] = seq
                continue

        pos = (float(p[2]), float(p[3]))
        tracks[pid] = [pos] * (max_dt + 1)
    return tracks


def predict_all_planet_tracks_arrays(
    planets: Sequence[Sequence[float]],
    max_dt: int,
    current_step: int,
    angular_velocity: float,
    initial_by_id: Dict[int, Sequence[float]],
    comet_lookup: Optional[Dict[int, Tuple[List[List[float]], int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, List[Optional[Tuple[float, float]]]]]:
    """Vectorized version of ``predict_all_planet_tracks`` returning dense arrays + dict.

    Returns ``(tr_x, tr_y, tr_valid, tracks)`` where ``tr_x``/``tr_y`` have shape
    ``(N, max_dt + 1)`` (``NaN`` for expired comets) and ``tr_valid`` is True iff
    the planet is present on-board (not ``None`` and ``x >= 0``).
    """
    n = len(planets)
    h1 = max_dt + 1
    tr_x = np.full((n, h1), np.nan, dtype=np.float64)
    tr_y = np.full((n, h1), np.nan, dtype=np.float64)
    tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    if n == 0:
        return tr_x, tr_y, np.zeros((0, h1), dtype=bool), tracks

    dts = np.arange(1, h1, dtype=np.float64)
    ang_steps = angular_velocity * (current_step + dts - 1.0)

    orb_indices: List[int] = []
    orb_r_list: List[float] = []
    orb_ang0_list: List[float] = []
    orb_pids: List[int] = []
    orb_x0: List[float] = []
    orb_y0: List[float] = []

    for i, p in enumerate(planets):
        pid = int(p[0])
        px, py, pr = float(p[2]), float(p[3]), float(p[4])
        if comet_lookup is not None and pid in comet_lookup:
            path, cur_idx = comet_lookup[pid]
            plen = len(path)
            tr_x[i, 0] = px
            tr_y[i, 0] = py
            seq: List[Optional[Tuple[float, float]]] = [(px, py)]
            for dt in range(1, h1):
                fidx = cur_idx + dt
                if fidx < 0:
                    tr_x[i, dt] = -99.0
                    tr_y[i, dt] = -99.0
                    seq.append((-99.0, -99.0))
                elif fidx >= plen:
                    seq.append(None)
                else:
                    pt = path[fidx]
                    fx, fy = float(pt[0]), float(pt[1])
                    tr_x[i, dt] = fx
                    tr_y[i, dt] = fy
                    seq.append((fx, fy))
            tracks[pid] = seq
            continue

        init_p = initial_by_id.get(pid)
        if init_p is not None:
            dx = float(init_p[2]) - CENTER
            dy = float(init_p[3]) - CENTER
            orb_r = math.hypot(dx, dy)
            if orb_r + pr < ROTATION_RADIUS_LIMIT:
                orb_indices.append(i)
                orb_r_list.append(orb_r)
                orb_ang0_list.append(math.atan2(dy, dx))
                orb_pids.append(pid)
                orb_x0.append(px)
                orb_y0.append(py)
                continue

        tr_x[i, :] = px
        tr_y[i, :] = py
        tracks[pid] = [(px, py)] * h1

    if orb_indices:
        idx_arr = np.asarray(orb_indices, dtype=np.int64)
        r_arr = np.asarray(orb_r_list, dtype=np.float64)[:, None]
        a0_arr = np.asarray(orb_ang0_list, dtype=np.float64)[:, None]
        ang = a0_arr + ang_steps[None, :]
        ox = CENTER + r_arr * np.cos(ang)
        oy = CENTER + r_arr * np.sin(ang)
        tr_x[idx_arr, 0] = orb_x0
        tr_y[idx_arr, 0] = orb_y0
        tr_x[idx_arr, 1:] = ox
        tr_y[idx_arr, 1:] = oy
        ox_list = ox.tolist()
        oy_list = oy.tolist()
        for k, pid in enumerate(orb_pids):
            tracks[pid] = [(orb_x0[k], orb_y0[k])] + list(zip(ox_list[k], oy_list[k]))

    tr_valid = (~np.isnan(tr_x)) & (tr_x >= 0.0)
    return tr_x, tr_y, tr_valid, tracks


def predict_fleet_arrivals_vectorized(
    fleets: Sequence[Sequence[float]],
    planets: Sequence[Sequence[float]],
    tr_x: np.ndarray,
    tr_y: np.ndarray,
    max_horizon: int = 24,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Dict[int, List[FleetArrival]]:
    """Bit-exact 1-shot ``(F, H, N)`` vectorized swept-pair fleet arrival predictor."""
    planet_ids = [int(p[0]) for p in planets]
    arrivals: Dict[int, List[FleetArrival]] = {pid: [] for pid in planet_ids}
    nf = len(fleets)
    n = len(planets)
    if nf == 0 or n == 0:
        return arrivals

    eff_h = min(max_horizon, int(tr_x.shape[1]) - 1)
    if eff_h <= 0:
        return arrivals

    fl_arr = np.asarray(fleets, dtype=np.float64)
    fids = fl_arr[:, 0].astype(np.int64)
    fown = fl_arr[:, 1].astype(np.int64)
    fx0 = fl_arr[:, 2]
    fy0 = fl_arr[:, 3]
    fang = fl_arr[:, 4]
    fships = fl_arr[:, 6].astype(np.int64)
    fspd = np.where(
        fships <= 1,
        1.0,
        np.minimum(max_speed, 1.0 + (max_speed - 1.0) * (np.log(np.maximum(1, fships)) / _LN_1000) ** 1.5),
    )

    vx = np.cos(fang) * fspd
    vy = np.sin(fang) * fspd
    pr = np.asarray([float(p[4]) for p in planets], dtype=np.float64)
    pr2 = pr * pr

    t_idx = np.arange(eff_h, dtype=np.float64)
    ox = fx0[:, None] + vx[:, None] * t_idx[None, :]
    oy = fy0[:, None] + vy[:, None] * t_idx[None, :]
    nx = ox + vx[:, None]
    ny = oy + vy[:, None]

    oob = (nx < 0.0) | (nx > BOARD_SIZE) | (ny < 0.0) | (ny > BOARD_SIZE)
    l2 = vx * vx + vy * vy
    safe_l2 = np.where(l2 > 0.0, l2, 1.0)[:, None]
    tt = np.clip(((CENTER - ox) * vx[:, None] + (CENTER - oy) * vy[:, None]) / safe_l2, 0.0, 1.0)
    tt = np.where((l2 > 0.0)[:, None], tt, 0.0)
    sun_hit = ((CENTER - (ox + tt * vx[:, None])) ** 2 + (CENTER - (oy + tt * vy[:, None])) ** 2) < (
        SUN_RADIUS * SUN_RADIUS
    )
    dead_after = oob | sun_hit

    p_old_x = tr_x[:, :eff_h].T
    p_old_y = tr_y[:, :eff_h].T
    p_old_ok = (~np.isnan(p_old_x)) & (p_old_x >= 0.0)
    p_new_x_raw = tr_x[:, 1 : eff_h + 1].T
    p_new_y_raw = tr_y[:, 1 : eff_h + 1].T
    p_new_none = np.isnan(p_new_x_raw)
    p_new_x = np.where(p_new_none, p_old_x, p_new_x_raw)
    p_new_y = np.where(p_new_none, p_old_y, p_new_y_raw)
    p_dvx = p_new_x - p_old_x
    p_dvy = p_new_y - p_old_y

    d0x = ox[:, :, None] - p_old_x[None, :, :]
    d0y = oy[:, :, None] - p_old_y[None, :, :]
    cull = (fspd[:, None, None] + 4.5) + pr[None, None, :]
    cand = p_old_ok[None, :, :] & (np.abs(d0x) <= cull) & (np.abs(d0y) <= cull)

    hits = np.zeros((nf, eff_h, n), dtype=bool)
    cf, ct, cp = np.nonzero(cand)
    if cf.size > 0:
        cd0x = d0x[cf, ct, cp]
        cd0y = d0y[cf, ct, cp]
        cdvx = vx[cf] - p_dvx[ct, cp]
        cdvy = vy[cf] - p_dvy[ct, cp]
        qa = cdvx * cdvx + cdvy * cdvy
        qb = 2.0 * (cd0x * cdvx + cd0y * cdvy)
        qc = cd0x * cd0x + cd0y * cd0y - pr2[cp]
        disc = qb * qb - 4.0 * qa * qc
        small_qa = qa < 1e-12
        safe_qa = np.where(small_qa, 1.0, qa)
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-qb - sq) / (2.0 * safe_qa)
        t2 = (-qb + sq) / (2.0 * safe_qa)
        chits = np.where(small_qa, qc <= 0.0, (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))
        if np.any(chits):
            hits[cf[chits], ct[chits], cp[chits]] = True

    hit_tick_any = hits.any(axis=2)
    any_event = hit_tick_any | dead_after
    has_event = any_event.any(axis=1)
    first_t = any_event.argmax(axis=1)
    for fi in np.flatnonzero(has_event):
        t = int(first_t[fi])
        if hit_tick_any[fi, t]:
            pj = int(np.argmax(hits[fi, t]))
            if not p_new_none[t, pj]:
                arrivals[planet_ids[pj]].append(
                    FleetArrival(
                        fleet_id=int(fids[fi]),
                        owner=int(fown[fi]),
                        ships=int(fships[fi]),
                        planet_id=planet_ids[pj],
                        eta=t + 1,
                    )
                )

    for pid in arrivals:
        if len(arrivals[pid]) > 1:
            arrivals[pid].sort(key=lambda a: a.eta)
    return arrivals


@dataclass(frozen=True)
class InterceptSolution:
    """Verified launch solution from a source planet to a target planet."""

    angle: float
    eta: int
    impact_pos: Tuple[float, float]
    speed: float


def verify_launch_trajectory(
    from_planet: Sequence[float],
    target_id: int,
    angle: float,
    ships: int,
    planets: Sequence[Sequence[float]],
    tracks: Dict[int, List[Optional[Tuple[float, float]]]],
    max_dt: int,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Optional[Tuple[int, Tuple[float, float]]]:
    """Simulate a candidate launch tick-by-tick against all planets, sun, and bounds.

    Returns ``(eta, impact_pos)`` if the fleet hits ``target_id`` cleanly without
    colliding with any other planet, the sun, or leaving the board first; else ``None``.
    """
    spd = fleet_speed(ships, max_speed)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    fx = float(from_planet[2]) + cos_a * (float(from_planet[4]) + 0.1)
    fy = float(from_planet[3]) + sin_a * (float(from_planet[4]) + 0.1)
    vx = cos_a * spd
    vy = sin_a * spd

    # Pre-extract (pid, radius, track, cull_bound) once to avoid dict lookups per tick
    planet_specs = [
        (int(p[0]), float(p[4]), tracks[int(p[0])], spd + 4.5 + float(p[4]))
        for p in planets
    ]

    for dt in range(1, max_dt + 1):
        ox, oy = fx, fy
        fx += vx
        fy += vy
        old_pos = (ox, oy)
        new_pos = (fx, fy)

        for pid, pr, tr, cull in planet_specs:
            p_old = tr[dt - 1]
            if p_old is None or p_old[0] < 0.0:
                continue
            # Fast AABB rejection: mathematically impossible to hit within tick dt
            dx = ox - p_old[0]
            if dx > cull or dx < -cull:
                continue
            dy = oy - p_old[1]
            if dy > cull or dy < -cull:
                continue
            p_new = tr[dt]
            if p_new is None:
                if swept_pair_hit(old_pos, new_pos, p_old, p_old, pr):
                    return None
                continue
            if swept_pair_hit(old_pos, new_pos, p_old, p_new, pr):
                if pid == target_id:
                    return (dt, p_new)
                return None

        if fx < 0.0 or fx > BOARD_SIZE or fy < 0.0 or fy > BOARD_SIZE:
            return None
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            return None
    return None


def solve_intercept(
    from_planet: Sequence[float],
    target_planet: Sequence[float],
    ships: int,
    planets: Sequence[Sequence[float]],
    current_step: int,
    angular_velocity: float,
    initial_by_id: Dict[int, Sequence[float]],
    comet_lookup: Optional[Dict[int, Tuple[List[List[float]], int]]] = None,
    tracks: Optional[Dict[int, List[Optional[Tuple[float, float]]]]] = None,
    max_eta: int = 32,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Optional[InterceptSolution]:
    """Compute an exact, sun-safe, obstruction-free launch angle to hit ``target_planet``.

    Searches arrival ticks ``dt = 1..max_eta`` using the target's exact future
    position (for static planets, orbiting planets, and moving comets), checks
    the direct lead angle plus small tangent offsets if obstructed, and verifies
    the resulting swept-pair trajectory against all planets and the sun.
    """
    if ships <= 0:
        return None
    from_id = int(from_planet[0])
    target_id = int(target_planet[0])
    if from_id == target_id:
        return None

    spd = fleet_speed(ships, max_speed)
    if tracks is None:
        tracks = predict_all_planet_tracks(
            planets,
            max_eta,
            current_step,
            angular_velocity,
            initial_by_id,
            comet_lookup,
        )

    fx0 = float(from_planet[2])
    fy0 = float(from_planet[3])
    fr = float(from_planet[4])
    tr_r = float(target_planet[4])
    target_track = tracks[target_id]
    eff_max_eta = min(max_eta, len(target_track) - 1)
    if eff_max_eta < 1:
        return None

    # Collect top candidate (dt, angle, err) sorted by timing match
    candidates: List[Tuple[float, int, float]] = []
    for dt in range(1, eff_max_eta + 1):
        p_new = target_track[dt]
        if p_new is None:
            break
        p_old = target_track[dt - 1]
        if p_old is None or p_old[0] < 0.0:
            continue

        dist_end = math.hypot(p_new[0] - fx0, p_new[1] - fy0)
        reach_lo = (fr + 0.1) + spd * (dt - 1) - tr_r - 1.5
        reach_hi = (fr + 0.1) + spd * dt + tr_r + 1.5
        if reach_lo <= dist_end <= reach_hi:
            # Skip rays blocked squarely by the central sun
            if point_to_segment_distance((CENTER, CENTER), (fx0, fy0), p_new) < SUN_RADIUS - 0.5:
                continue
            expected_reach = (fr + 0.1) + spd * (dt - 0.5)
            err = abs(dist_end - expected_reach)
            ang_end = math.atan2(p_new[1] - fy0, p_new[0] - fx0)
            candidates.append((err, dt, ang_end))

    if not candidates:
        p_next = target_track[1]
        if p_next is not None and point_to_segment_distance((CENTER, CENTER), (fx0, fy0), p_next) >= SUN_RADIUS - 0.5:
            candidates.append((0.0, 1, math.atan2(p_next[1] - fy0, p_next[0] - fx0)))

    candidates.sort(key=lambda x: (x[1], x[0]))
    top_candidates = candidates[:4]

    # First pass: exact lead angles (delta = 0.0)
    for _, dt_hint, base_angle in top_candidates:
        res = verify_launch_trajectory(
            from_planet,
            target_id,
            base_angle,
            ships,
            planets,
            tracks,
            min(eff_max_eta, dt_hint + 3),
            max_speed=max_speed,
        )
        if res is not None:
            eta, impact_pos = res
            return InterceptSolution(angle=base_angle, eta=eta, impact_pos=impact_pos, speed=spd)

    # Second pass: tiny angular nudges on the best 2 candidates to skirt grazing obstacles
    for _, dt_hint, base_angle in top_candidates[:2]:
        for delta in (0.04, -0.04):
            ang = base_angle + delta
            res = verify_launch_trajectory(
                from_planet,
                target_id,
                ang,
                ships,
                planets,
                tracks,
                min(eff_max_eta, dt_hint + 3),
                max_speed=max_speed,
            )
            if res is not None:
                eta, impact_pos = res
                return InterceptSolution(angle=ang, eta=eta, impact_pos=impact_pos, speed=spd)

    return None


@dataclass(frozen=True)
class FleetArrival:
    """Predicted arrival of an in-flight fleet at a planet."""

    fleet_id: int
    owner: int
    ships: int
    planet_id: int
    eta: int


def predict_fleet_arrivals(
    fleets: Sequence[Sequence[float]],
    planets: Sequence[Sequence[float]],
    current_step: int,
    angular_velocity: float,
    initial_by_id: Dict[int, Sequence[float]],
    comet_lookup: Optional[Dict[int, Tuple[List[List[float]], int]]] = None,
    tracks: Optional[Dict[int, List[Optional[Tuple[float, float]]]]] = None,
    max_horizon: int = 24,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Dict[int, List[FleetArrival]]:
    """Simulate every active fleet forward to predict exact arrival planet and tick.

    Returns:
        Dictionary mapping ``planet_id -> sorted list of FleetArrival``.
    """
    arrivals: Dict[int, List[FleetArrival]] = {int(p[0]): [] for p in planets}
    if not fleets:
        return arrivals

    if tracks is None:
        tracks = predict_all_planet_tracks(
            planets,
            max_horizon,
            current_step,
            angular_velocity,
            initial_by_id,
            comet_lookup,
        )

    eff_horizon = min(max_horizon, min((len(tr) - 1 for tr in tracks.values()), default=0))
    planet_specs = [
        (int(p[0]), float(p[4]), tracks[int(p[0])])
        for p in planets
    ]

    for fl in fleets:
        fid = int(fl[0])
        owner = int(fl[1])
        fx = float(fl[2])
        fy = float(fl[3])
        ang = float(fl[4])
        ships = int(fl[6])
        spd = fleet_speed(ships, max_speed)
        vx = math.cos(ang) * spd
        vy = math.sin(ang) * spd
        base_cull = spd + 4.5

        for dt in range(1, eff_horizon + 1):
            ox, oy = fx, fy
            fx += vx
            fy += vy
            old_pos = (ox, oy)
            new_pos = (fx, fy)

            hit = False
            for pid, pr, tr in planet_specs:
                p_old = tr[dt - 1]
                if p_old is None or p_old[0] < 0.0:
                    continue
                cull = base_cull + pr
                dx = ox - p_old[0]
                if dx > cull or dx < -cull:
                    continue
                dy = oy - p_old[1]
                if dy > cull or dy < -cull:
                    continue
                p_new = tr[dt]
                if p_new is None:
                    if swept_pair_hit(old_pos, new_pos, p_old, p_old, pr):
                        hit = True
                        break
                    continue
                if swept_pair_hit(old_pos, new_pos, p_old, p_new, pr):
                    arrivals[pid].append(
                        FleetArrival(
                            fleet_id=fid,
                            owner=owner,
                            ships=ships,
                            planet_id=pid,
                            eta=dt,
                        )
                    )
                    hit = True
                    break
            if hit:
                break
            if fx < 0.0 or fx > BOARD_SIZE or fy < 0.0 or fy > BOARD_SIZE:
                break
            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
                break

    for pid in arrivals:
        if len(arrivals[pid]) > 1:
            arrivals[pid].sort(key=lambda a: a.eta)
    return arrivals


@dataclass(frozen=True)
class PlanetFutureState:
    """Simulated future ownership and ship trajectory for a single planet."""

    planet_id: int
    horizon: int
    final_owner: int
    final_ships: int
    min_owner_ships: int
    will_be_captured_by: Optional[int]
    capture_tick: Optional[int]
    # Per-tick (owner, ships) for dt = 0..horizon
    timeline: Tuple[Tuple[int, int], ...]


def simulate_planet_future(
    planet: Sequence[float],
    arrivals: Sequence[FleetArrival],
    horizon: int = 25,
    extra_arrivals: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> PlanetFutureState:
    """Simulate a planet's exact ownership & ship count over ``1..horizon`` ticks.

    Follows exact ``orbit_wars.py`` tick order:
    1. Production (if ``owner != -1``, ``ships += production``).
    2. Combat resolution for all fleets arriving on tick ``dt``.

    Args:
        planet: Current planet ``[id, owner, x, y, radius, ships, production]``.
        arrivals: Predicted ``FleetArrival`` objects targeting this planet.
        horizon: Number of future ticks to simulate.
        extra_arrivals: Optional list of ``(eta, owner, ships)`` hypothetical fleets.
    """
    pid = int(planet[0])
    owner = int(planet[1])
    ships = int(planet[5])
    prod = int(planet[6])
    initial_owner = owner

    by_tick: Dict[int, Dict[int, int]] = {}
    for arr in arrivals:
        if 1 <= arr.eta <= horizon:
            bucket = by_tick.setdefault(arr.eta, {})
            bucket[arr.owner] = bucket.get(arr.owner, 0) + arr.ships
    if extra_arrivals:
        for eta, fl_owner, fl_ships in extra_arrivals:
            if 1 <= eta <= horizon and fl_ships > 0:
                bucket = by_tick.setdefault(int(eta), {})
                bucket[int(fl_owner)] = bucket.get(int(fl_owner), 0) + int(fl_ships)

    timeline: List[Tuple[int, int]] = [(owner, ships)]
    min_owner_ships = ships
    unflipped_ships = ships
    captured_by: Optional[int] = None
    capture_tick: Optional[int] = None

    for dt in range(1, horizon + 1):
        # 1. Production
        if owner != -1:
            ships += prod
        if initial_owner != -1:
            unflipped_ships += prod

        # 2. Combat resolution
        combat = by_tick.get(dt)
        if combat:
            sorted_p = sorted(combat.items(), key=lambda kv: kv[1], reverse=True)
            top_owner, top_ships = sorted_p[0]
            if len(sorted_p) > 1:
                second_ships = sorted_p[1][1]
                survivor_ships = top_ships - second_ships
                if top_ships == second_ships:
                    survivor_ships = 0
                survivor_owner = top_owner if survivor_ships > 0 else -1
            else:
                survivor_owner = top_owner
                survivor_ships = top_ships

            if survivor_ships > 0:
                if initial_owner != -1:
                    if survivor_owner == initial_owner:
                        unflipped_ships += survivor_ships
                    else:
                        unflipped_ships -= survivor_ships
                if owner == survivor_owner:
                    ships += survivor_ships
                else:
                    ships -= survivor_ships
                    if ships < 0:
                        owner = survivor_owner
                        ships = abs(ships)
                        if captured_by is None and owner != initial_owner:
                            captured_by = owner
                            capture_tick = dt

        if initial_owner != -1:
            if unflipped_ships < min_owner_ships:
                min_owner_ships = unflipped_ships
        elif owner == initial_owner:
            if ships < min_owner_ships:
                min_owner_ships = ships
        else:
            min_owner_ships = min(min_owner_ships, -ships)

        timeline.append((owner, ships))

    return PlanetFutureState(
        planet_id=pid,
        horizon=horizon,
        final_owner=owner,
        final_ships=ships,
        min_owner_ships=min_owner_ships,
        will_be_captured_by=captured_by,
        capture_tick=capture_tick,
        timeline=tuple(timeline),
    )


def ships_needed_to_capture(
    target_planet: Sequence[float],
    arrivals: Sequence[FleetArrival],
    attacker_id: int,
    eta: int,
) -> int:
    """Compute minimum ships an attacker needs at tick ``eta`` to capture/hold ``target_planet``."""
    future = simulate_planet_future(target_planet, arrivals, horizon=eta)
    owner_at_eta, ships_at_eta = future.timeline[eta]
    if owner_at_eta == attacker_id:
        return 0
    return ships_at_eta + 1


def build_comet_lookup(comets: Sequence[Dict[str, Any]]) -> Dict[int, Tuple[List[List[float]], int]]:
    """Public helper to map comet planet IDs to ``(path, path_index)``."""
    return _build_comet_lookup(comets)
