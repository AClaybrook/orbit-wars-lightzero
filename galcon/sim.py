"""Fast continuous-coordinate Python Galcon simulator matching Phil Hassey's ``gbots`` physics.

Physics constants match real Galcon / ``gbots`` and ``esparano/LightZero``:
- Production rate: ``production / 50.0`` ships per second (``prod=100`` -> 2.0 ships/sec).
- Constant fleet speed: ``40.0`` units/sec.
- Default tick step: ``0.25`` seconds (4 ticks/sec, 240 ticks = 60s classic round).
- Actions: ``SendOrder(src_id, dst_id, percent)`` (1..100%) and ``RedirOrder(fleet_id, dst_id)``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PRODUCTION_DIVISOR = 50.0  # production=100 => 2.0 ships/sec (120 ships/min)
DEFAULT_FLEET_SPEED = 40.0
DEFAULT_TICK_DT = 0.25
DEFAULT_MAX_TICKS = 240  # 60 seconds at 4 ticks/sec
SHIP_RADIUS = 6.0
PLANET_SETTLE_DELTA = 0.5
MIN_PLANET_GAP = 2.0 * SHIP_RADIUS + PLANET_SETTLE_DELTA  # 12.5
DEFAULT_MAP_WIDTH = 480.0
DEFAULT_MAP_HEIGHT = 320.0
DEFAULT_NUM_PLANETS = 23
NEUTRAL_SHIPS_MIN = 0
NEUTRAL_SHIPS_MAX = 45
NEUTRAL_PROD_MIN = 15
NEUTRAL_PROD_MAX = 100
TIE_EPS = 1e-9
ARRIVAL_EPS = 1e-9


@dataclass
class Planet:
    id: int
    owner: int  # 0 = neutral, 1..K = player IDs
    ships: float
    x: float
    y: float
    production: float
    radius: float


@dataclass
class Fleet:
    id: int
    owner: int
    ships: float
    x: float
    y: float
    source: int
    target: int
    radius: float = SHIP_RADIUS


@dataclass(frozen=True)
class SendOrder:
    src_id: int
    dst_id: int
    percent: int  # integer in [1, 100], matching gbots `/SEND PERCENT SOURCE TARGET`


@dataclass(frozen=True)
class RedirOrder:
    fleet_id: int
    dst_id: int  # matching gbots `/REDIR FLEET_ID TARGET`


Order = Union[SendOrder, RedirOrder]


def planet_radius_from_production(production: float) -> float:
    """Exact Galcon 2 planet radius from production (15..100 -> 12.0..24.0, clamped below 15).

    Note: Galcon 3 uses ``(production + 50.0) * 16.0 / 100.0``; we stick with the
    Galcon 2 formula ``max(12.0, (production * 12.0 / 5.0 + 168.0) / 17.0)`` for now.
    """
    return max(12.0, (float(production) * 12.0 / 5.0 + 168.0) / 17.0)


def _axis_bounds(radius: float, span: float) -> Tuple[float, float]:
    """Legal symmetric ``[lo, hi]`` planet-center bounds along one map axis."""
    margin = float(radius) + SHIP_RADIUS
    if span - 2.0 * margin >= 0.0:
        return margin, span - margin
    fallback = float(radius) + 2.0
    if span - 2.0 * fallback >= 0.0:
        return fallback, span - fallback
    mid = span * 0.5
    return mid, mid


def _clone_planets(planets: Sequence[Planet]) -> List[Planet]:
    return [
        Planet(p.id, p.owner, float(p.ships), float(p.x), float(p.y), float(p.production), float(p.radius))
        for p in planets
    ]


def _planets_clearance_ok(
    planets: Sequence[Planet],
    min_gap: float = MIN_PLANET_GAP,
    tol: float = 1e-6,
) -> bool:
    n = len(planets)
    for i in range(n):
        pi = planets[i]
        for j in range(i + 1, n):
            pj = planets[j]
            if math.hypot(pi.x - pj.x, pi.y - pj.y) < pi.radius + pj.radius + min_gap - tol:
                return False
    return True


def settle_planets(
    planets: Sequence[Planet],
    width: float,
    height: float,
    min_gap: float = MIN_PLANET_GAP,
    symmetric: bool = True,
    max_iters: int = 30,
) -> bool:
    """Iteratively push overlapping / too-close planets apart to enforce ``min_gap`` clearance.

    Matches Galcon 2's ``planets_settle`` and ``origin/galcon-test2``'s relaxation loop:
    - Keeps every planet within ``[r + SHIP_RADIUS, span - r - SHIP_RADIUS]`` (falling back to
      ``[r + 2.0, span - r - 2.0]`` only if the map is too small for the full margin).
    - When ``symmetric=True``, preserves exact 180-degree rotational symmetry
      (``x_a + x_b == width``, ``y_a + y_b == height``) for symmetric pairs and pins any odd
      center planet at ``(width / 2, height / 2)``.

    Mutates ``planets`` in place and returns ``True`` if all pairs satisfy clearance.
    """
    n = len(planets)
    if n == 0:
        return True
    width = float(width)
    height = float(height)
    cx, cy = width * 0.5, height * 0.5
    min_gap = float(min_gap)

    pairs: List[Tuple[int, int]] = []
    center_indices: List[int] = []
    free_indices: List[int] = []

    if symmetric:
        used = [False] * n
        for i in range(0, n - 1, 2):
            pa, pb = planets[i], planets[i + 1]
            if (
                abs((pa.x + pb.x) - width) <= 1e-3
                and abs((pa.y + pb.y) - height) <= 1e-3
                and abs(pa.radius - pb.radius) <= 1e-3
            ):
                pairs.append((i, i + 1))
                used[i] = used[i + 1] = True
        for i in range(n):
            if used[i]:
                continue
            pa = planets[i]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                pb = planets[j]
                if (
                    abs((pa.x + pb.x) - width) <= 1e-3
                    and abs((pa.y + pb.y) - height) <= 1e-3
                    and abs(pa.radius - pb.radius) <= 1e-3
                ):
                    pairs.append((i, j))
                    used[i] = used[j] = True
                    break
        for i in range(n):
            if used[i]:
                continue
            p = planets[i]
            if abs(p.x - cx) <= 1e-3 and abs(p.y - cy) <= 1e-3:
                center_indices.append(i)
            else:
                free_indices.append(i)
    else:
        free_indices = list(range(n))

    center_set = set(center_indices)

    # Snap bounds and exact 180-degree symmetry before iterating
    for i in center_indices:
        planets[i].x = cx
        planets[i].y = cy
    for a, b in pairs:
        pa, pb = planets[a], planets[b]
        lo_x, hi_x = _axis_bounds(pa.radius, width)
        lo_y, hi_y = _axis_bounds(pa.radius, height)
        pa.x = max(lo_x, min(hi_x, float(pa.x)))
        pa.y = max(lo_y, min(hi_y, float(pa.y)))
        pb.x = width - pa.x
        pb.y = height - pa.y
    for i in free_indices:
        pi = planets[i]
        lo_x, hi_x = _axis_bounds(pi.radius, width)
        lo_y, hi_y = _axis_bounds(pi.radius, height)
        pi.x = max(lo_x, min(hi_x, float(pi.x)))
        pi.y = max(lo_y, min(hi_y, float(pi.y)))

    eps_tol = 1e-7
    nudge_eps = 1e-3
    max_step = max(min_gap, 0.25 * min(width, height))

    def _repulsion_for(idx: int, it: int) -> Tuple[float, float, bool]:
        pi = planets[idx]
        dx_total = 0.0
        dy_total = 0.0
        hit = False
        for j in range(n):
            if j == idx:
                continue
            pj = planets[j]
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dist = math.hypot(dx, dy)
            min_dist = pi.radius + pj.radius + min_gap
            if dist < min_dist - eps_tol:
                overlap = min_dist - dist
                if dist < 1e-6:
                    angle = idx * 2.399963229728653 + j * 0.7319 + it * 1.13
                    ux, uy = math.cos(angle), math.sin(angle)
                else:
                    ux = dx / dist
                    uy = dy / dist
                frac = 1.0 if j in center_set else 0.5
                step = frac * overlap + nudge_eps
                dx_total += ux * step
                dy_total += uy * step
                hit = True
        mag = math.hypot(dx_total, dy_total)
        if mag > max_step:
            scale = max_step / mag
            dx_total *= scale
            dy_total *= scale
        return dx_total, dy_total, hit

    for it in range(max(1, int(max_iters))):
        moved = False
        pair_disp: List[Tuple[float, float]] = []
        free_disp: List[Tuple[float, float]] = []

        for a, b in pairs:
            dxa, dya, ha = _repulsion_for(a, it)
            if free_indices:
                dxb, dyb, hb = _repulsion_for(b, it)
                dx = 0.5 * (dxa - dxb)
                dy = 0.5 * (dya - dyb)
                if ha or hb:
                    moved = True
            else:
                dx, dy = dxa, dya
                if ha:
                    moved = True
            pair_disp.append((dx, dy))

        for idx in free_indices:
            dx, dy, hit = _repulsion_for(idx, it)
            if hit:
                moved = True
            free_disp.append((dx, dy))

        if not moved:
            return _planets_clearance_ok(planets, min_gap=min_gap)

        for (a, b), (dx, dy) in zip(pairs, pair_disp):
            if dx == 0.0 and dy == 0.0:
                continue
            pa, pb = planets[a], planets[b]
            lo_x, hi_x = _axis_bounds(pa.radius, width)
            lo_y, hi_y = _axis_bounds(pa.radius, height)
            pa.x = max(lo_x, min(hi_x, pa.x + dx))
            pa.y = max(lo_y, min(hi_y, pa.y + dy))
            pb.x = width - pa.x
            pb.y = height - pa.y

        for idx, (dx, dy) in zip(free_indices, free_disp):
            if dx == 0.0 and dy == 0.0:
                continue
            pi = planets[idx]
            lo_x, hi_x = _axis_bounds(pi.radius, width)
            lo_y, hi_y = _axis_bounds(pi.radius, height)
            pi.x = max(lo_x, min(hi_x, pi.x + dx))
            pi.y = max(lo_y, min(hi_y, pi.y + dy))

    return _planets_clearance_ok(planets, min_gap=min_gap)


def generate_galcon_map(
    *,
    seed: int = 42,
    num_planets: int = DEFAULT_NUM_PLANETS,
    width: float = DEFAULT_MAP_WIDTH,
    height: float = DEFAULT_MAP_HEIGHT,
    home_ships: float = 100.0,
    home_prod: float = 100.0,
    symmetric: bool = True,
) -> List[Planet]:
    """Generate a balanced 2P Galcon map (180-degree rotationally symmetric by default)."""
    rng = random.Random(seed)
    width = float(width)
    height = float(height)
    cx, cy = width * 0.5, height * 0.5
    min_gap = MIN_PLANET_GAP
    home_r = planet_radius_from_production(home_prod)

    # Place P1 and P2 homes symmetrically on an outer ellipse within legal map bounds
    h_min_x, _ = _axis_bounds(home_r, width)
    h_min_y, _ = _axis_bounds(home_r, height)
    rx_max = max(0.0, cx - h_min_x)
    ry_max = max(0.0, cy - h_min_y)
    min_home_center_dist = home_r + 0.5 * min_gap

    hx, hy = cx, cy
    placed_home = False
    for _ in range(64):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        rx = rx_max * rng.uniform(0.70, 0.90)
        ry = ry_max * rng.uniform(0.70, 0.90)
        dx = rx * math.cos(angle)
        dy = ry * math.sin(angle)
        d_center = math.hypot(dx, dy)
        if 1e-6 < d_center < min_home_center_dist:
            scale = min_home_center_dist / d_center
            cand_dx = dx * scale
            cand_dy = dy * scale
            if abs(cand_dx) <= rx_max + 1e-9 and abs(cand_dy) <= ry_max + 1e-9:
                dx, dy = cand_dx, cand_dy
                d_center = math.hypot(dx, dy)
        if d_center >= min_home_center_dist - 1e-9:
            hx = cx + max(-rx_max, min(rx_max, dx))
            hy = cy + max(-ry_max, min(ry_max, dy))
            placed_home = True
            break
    if not placed_home:
        if rx_max >= ry_max:
            hx, hy = cx + rx_max, cy
        else:
            hx, hy = cx, cy + ry_max

    home1 = Planet(id=1, owner=1, ships=float(home_ships), x=hx, y=hy, production=float(home_prod), radius=home_r)
    home2 = Planet(
        id=2,
        owner=2,
        ships=float(home_ships),
        x=width - hx,
        y=height - hy,
        production=float(home_prod),
        radius=home_r,
    )

    def fits(x: float, y: float, r: float, existing: Sequence[Planet]) -> bool:
        lo_x, hi_x = _axis_bounds(r, width)
        lo_y, hi_y = _axis_bounds(r, height)
        if x < lo_x - 1e-9 or x > hi_x + 1e-9 or y < lo_y - 1e-9 or y > hi_y + 1e-9:
            return False
        for p in existing:
            if math.hypot(x - p.x, y - p.y) < r + p.radius + min_gap:
                return False
        return True

    target_neutrals = max(2, int(num_planets) - 2)
    if symmetric:
        pairs = target_neutrals // 2
        pair_planets: List[Planet] = [home1, home2]
        center_planet: Optional[Planet] = None

        if target_neutrals % 2 == 1:
            c_prod = float(rng.randint(NEUTRAL_PROD_MIN, NEUTRAL_PROD_MAX))
            c_r = planet_radius_from_production(c_prod)
            c_ships = float(rng.randint(NEUTRAL_SHIPS_MIN, NEUTRAL_SHIPS_MAX))
            cand_center = Planet(id=0, owner=0, ships=c_ships, x=cx, y=cy, production=c_prod, radius=c_r)
            if fits(cx, cy, c_r, pair_planets):
                center_planet = cand_center
            else:
                trial = _clone_planets(pair_planets) + [cand_center]
                if settle_planets(trial, width, height, min_gap=min_gap, symmetric=True, max_iters=30):
                    pair_planets = trial[:-1]
                    center_planet = trial[-1]

        for _ in range(pairs):
            prod = float(rng.randint(NEUTRAL_PROD_MIN, NEUTRAL_PROD_MAX))
            r = planet_radius_from_production(prod)
            ships = float(rng.randint(NEUTRAL_SHIPS_MIN, NEUTRAL_SHIPS_MAX))
            lo_x, hi_x = _axis_bounds(r, width)
            lo_y, hi_y = _axis_bounds(r, height)
            existing = pair_planets + ([center_planet] if center_planet is not None else [])
            placed = False
            best_margin = -1e18
            best_xy: Optional[Tuple[float, float]] = None

            for _attempt in range(160):
                x = rng.uniform(lo_x, hi_x)
                y = rng.uniform(lo_y, hi_y)
                mx, my = width - x, height - y
                min_clear = math.hypot(x - mx, y - my) - (2.0 * r + min_gap)
                if min_clear > best_margin or min_clear >= 0.0:
                    for p in existing:
                        c1 = math.hypot(x - p.x, y - p.y) - (r + p.radius + min_gap)
                        if c1 < min_clear:
                            min_clear = c1
                            if min_clear < best_margin and min_clear < 0.0:
                                break
                        c2 = math.hypot(mx - p.x, my - p.y) - (r + p.radius + min_gap)
                        if c2 < min_clear:
                            min_clear = c2
                            if min_clear < best_margin and min_clear < 0.0:
                                break
                    if min_clear >= 0.0:
                        pair_planets.append(Planet(id=0, owner=0, ships=ships, x=x, y=y, production=prod, radius=r))
                        pair_planets.append(Planet(id=0, owner=0, ships=ships, x=mx, y=my, production=prod, radius=r))
                        placed = True
                        break
                    if min_clear > best_margin:
                        best_margin = min_clear
                        best_xy = (x, y)

            if not placed and best_xy is not None:
                bx, by = best_xy
                trial_pairs = _clone_planets(pair_planets)
                trial_pairs.append(Planet(id=0, owner=0, ships=ships, x=bx, y=by, production=prod, radius=r))
                trial_pairs.append(
                    Planet(id=0, owner=0, ships=ships, x=width - bx, y=height - by, production=prod, radius=r)
                )
                trial = trial_pairs + (_clone_planets([center_planet]) if center_planet is not None else [])
                if settle_planets(trial, width, height, min_gap=min_gap, symmetric=True, max_iters=30):
                    if center_planet is not None:
                        pair_planets = trial[:-1]
                        center_planet = trial[-1]
                    else:
                        pair_planets = trial
                    placed = True
            if not placed:
                continue

        planets: List[Planet] = list(pair_planets)
        if center_planet is not None:
            planets.append(center_planet)
    else:
        planets = [home1, home2]
        for _ in range(target_neutrals):
            prod = float(rng.randint(NEUTRAL_PROD_MIN, NEUTRAL_PROD_MAX))
            r = planet_radius_from_production(prod)
            ships = float(rng.randint(NEUTRAL_SHIPS_MIN, NEUTRAL_SHIPS_MAX))
            lo_x, hi_x = _axis_bounds(r, width)
            lo_y, hi_y = _axis_bounds(r, height)
            placed = False
            best_margin = -1e18
            best_xy = None

            for _attempt in range(160):
                x = rng.uniform(lo_x, hi_x)
                y = rng.uniform(lo_y, hi_y)
                min_clear = 1e18
                for p in planets:
                    c1 = math.hypot(x - p.x, y - p.y) - (r + p.radius + min_gap)
                    if c1 < min_clear:
                        min_clear = c1
                        if min_clear < best_margin and min_clear < 0.0:
                            break
                if min_clear >= 0.0:
                    planets.append(Planet(id=0, owner=0, ships=ships, x=x, y=y, production=prod, radius=r))
                    placed = True
                    break
                if min_clear > best_margin:
                    best_margin = min_clear
                    best_xy = (x, y)

            if not placed and best_xy is not None:
                bx, by = best_xy
                trial = _clone_planets(planets)
                trial.append(Planet(id=0, owner=0, ships=ships, x=bx, y=by, production=prod, radius=r))
                if settle_planets(trial, width, height, min_gap=min_gap, symmetric=False, max_iters=30):
                    planets = trial
                    placed = True
            if not placed:
                continue

    settle_planets(planets, width, height, min_gap=min_gap, symmetric=symmetric, max_iters=30)
    while len(planets) > 2 and not _planets_clearance_ok(planets, min_gap=min_gap):
        if symmetric:
            if len(planets) % 2 == 1:
                planets.pop()
            elif len(planets) >= 4:
                planets.pop()
                planets.pop()
            else:
                break
        else:
            planets.pop()
        settle_planets(planets, width, height, min_gap=min_gap, symmetric=symmetric, max_iters=30)

    for idx, p in enumerate(planets, start=1):
        p.id = idx

    return planets


class GalconSim:
    """Fast, cloneable Galcon simulation engine with send & mid-flight redirect support."""

    def __init__(
        self,
        planets: Sequence[Planet],
        fleets: Sequence[Fleet] = (),
        *,
        width: float = DEFAULT_MAP_WIDTH,
        height: float = DEFAULT_MAP_HEIGHT,
        fleet_speed: float = DEFAULT_FLEET_SPEED,
        tick_dt: float = DEFAULT_TICK_DT,
        max_ticks: int = DEFAULT_MAX_TICKS,
        tick: int = 0,
        next_fleet_id: int = 1000,
        allow_redirect: bool = True,
    ) -> None:
        self.planets: Dict[int, Planet] = {
            p.id: Planet(p.id, p.owner, float(p.ships), float(p.x), float(p.y), float(p.production), float(p.radius))
            for p in planets
        }
        self.fleets: Dict[int, Fleet] = {
            f.id: Fleet(f.id, f.owner, float(f.ships), float(f.x), float(f.y), f.source, f.target, float(f.radius))
            for f in fleets
        }
        self.width = float(width)
        self.height = float(height)
        self.fleet_speed = float(fleet_speed)
        self.tick_dt = float(tick_dt)
        self.max_ticks = int(max_ticks)
        self.tick = int(tick)
        self.next_fleet_id = int(next_fleet_id)
        self.allow_redirect = bool(allow_redirect)
        self.done = False
        self._winner = 0

    @classmethod
    def create(
        cls,
        *,
        seed: int = 42,
        num_planets: int = DEFAULT_NUM_PLANETS,
        width: float = DEFAULT_MAP_WIDTH,
        height: float = DEFAULT_MAP_HEIGHT,
        fleet_speed: float = DEFAULT_FLEET_SPEED,
        tick_dt: float = DEFAULT_TICK_DT,
        max_ticks: int = DEFAULT_MAX_TICKS,
        symmetric: bool = True,
        allow_redirect: bool = True,
    ) -> "GalconSim":
        planets = generate_galcon_map(
            seed=seed, num_planets=num_planets, width=width, height=height, symmetric=symmetric
        )
        return cls(
            planets,
            width=width,
            height=height,
            fleet_speed=fleet_speed,
            tick_dt=tick_dt,
            max_ticks=max_ticks,
            allow_redirect=allow_redirect,
        )

    def clone(self) -> "GalconSim":
        sim = GalconSim(
            list(self.planets.values()),
            list(self.fleets.values()),
            width=self.width,
            height=self.height,
            fleet_speed=self.fleet_speed,
            tick_dt=self.tick_dt,
            max_ticks=self.max_ticks,
            tick=self.tick,
            next_fleet_id=self.next_fleet_id,
            allow_redirect=self.allow_redirect,
        )
        sim.done = self.done
        sim._winner = self._winner
        return sim

    def eta_ticks(self, x1: float, y1: float, dst: Planet) -> int:
        """Exact arrival ticks from ``(x1, y1)`` to ``dst`` boundary."""
        dist = max(0.0, math.hypot(dst.x - x1, dst.y - y1) - dst.radius)
        step_dist = self.fleet_speed * self.tick_dt
        return max(1, int(math.ceil((dist - ARRIVAL_EPS) / max(1e-6, step_dist))))

    def planet_eta_ticks(self, src: Planet, dst: Planet) -> int:
        """Exact arrival ticks for a fleet launched from ``src`` edge to ``dst`` edge."""
        dist = max(0.0, math.hypot(dst.x - src.x, dst.y - src.y) - src.radius - dst.radius)
        step_dist = self.fleet_speed * self.tick_dt
        return max(1, int(math.ceil((dist - ARRIVAL_EPS) / max(1e-6, step_dist))))

    def step(self, orders_by_player: Dict[int, Sequence[Order]]) -> None:
        """Advance the Galcon world by one tick (orders -> production -> movement -> combat)."""
        if self.done:
            return

        # 1. Execute player orders simultaneously (deterministic by player ID)
        for pid in sorted(orders_by_player.keys()):
            for order in orders_by_player[pid]:
                if isinstance(order, SendOrder):
                    src = self.planets.get(order.src_id)
                    dst = self.planets.get(order.dst_id)
                    if src is None or dst is None or src.id == dst.id or src.owner != pid:
                        continue
                    pct = max(1, min(100, int(order.percent)))
                    send_ships = math.floor(src.ships * (pct / 100.0) + 1e-9)
                    if send_ships < 1:
                        continue
                    send_ships = min(float(send_ships), src.ships)
                    src.ships -= send_ships
                    dx = dst.x - src.x
                    dy = dst.y - src.y
                    d = max(1e-6, math.hypot(dx, dy))
                    spawn_x = src.x + (dx / d) * min(src.radius, d)
                    spawn_y = src.y + (dy / d) * min(src.radius, d)
                    fid = self.next_fleet_id
                    self.next_fleet_id += 1
                    self.fleets[fid] = Fleet(
                        id=fid,
                        owner=pid,
                        ships=send_ships,
                        x=spawn_x,
                        y=spawn_y,
                        source=src.id,
                        target=dst.id,
                    )
                elif isinstance(order, RedirOrder) and self.allow_redirect:
                    fl = self.fleets.get(order.fleet_id)
                    dst = self.planets.get(order.dst_id)
                    if fl is not None and dst is not None and fl.owner == pid and fl.target != dst.id:
                        fl.target = dst.id

        # 2. Ship production on owned planets
        for p in self.planets.values():
            if p.owner > 0:
                p.ships += (p.production / PRODUCTION_DIVISOR) * self.tick_dt

        # 3. Fleet movement & arrival collection
        step_dist = self.fleet_speed * self.tick_dt
        arrivals: Dict[int, List[Fleet]] = {}
        for fid in list(self.fleets.keys()):
            fl = self.fleets[fid]
            dst = self.planets.get(fl.target)
            if dst is None:
                del self.fleets[fid]
                continue
            dx = dst.x - fl.x
            dy = dst.y - fl.y
            dist = math.hypot(dx, dy)
            if dist <= dst.radius + step_dist + ARRIVAL_EPS:
                arrivals.setdefault(dst.id, []).append(fl)
                del self.fleets[fid]
            else:
                fl.x += (dx / dist) * step_dist
                fl.y += (dy / dist) * step_dist

        # 4. Resolve simultaneous arrivals at each planet
        for pid_planet, incoming in arrivals.items():
            p = self.planets[pid_planet]
            forces: Dict[int, float] = {}
            if p.ships > 0.0:
                forces[p.owner] = forces.get(p.owner, 0.0) + p.ships
            for fl in incoming:
                forces[fl.owner] = forces.get(fl.owner, 0.0) + fl.ships

            ranked = sorted(forces.items(), key=lambda kv: (-kv[1], kv[0] != p.owner))
            if len(ranked) == 1:
                p.owner, p.ships = ranked[0][0], ranked[0][1]
            elif abs(ranked[0][1] - ranked[1][1]) <= TIE_EPS:
                # Exact tie (within float epsilon): planet keeps owner with 0 ships (neutral stays neutral)
                p.ships = 0.0
            else:
                p.owner = ranked[0][0]
                p.ships = ranked[0][1] - ranked[1][1]

        self.tick += 1
        self._check_terminal()

    def total_ships(self, player: int) -> float:
        s = sum(p.ships for p in self.planets.values() if p.owner == player)
        s += sum(f.ships for f in self.fleets.values() if f.owner == player)
        return s

    def total_production(self, player: int) -> float:
        return sum(p.production for p in self.planets.values() if p.owner == player)

    def _check_terminal(self) -> None:
        alive = []
        for pid in (1, 2):
            has_p = any(p.owner == pid for p in self.planets.values())
            has_f = any(f.owner == pid and f.ships > 0 for f in self.fleets.values())
            if has_p or has_f:
                alive.append(pid)
        if len(alive) == 1:
            self.done = True
            self._winner = alive[0]
            return
        if len(alive) == 0:
            self.done = True
            self._winner = 0
            return
        if self.tick >= self.max_ticks:
            self.done = True
            s1, s2 = self.total_ships(1), self.total_ships(2)
            p1, p2 = self.total_production(1), self.total_production(2)
            score1 = s1 + 12.0 * (p1 / PRODUCTION_DIVISOR)
            score2 = s2 + 12.0 * (p2 / PRODUCTION_DIVISOR)
            if abs(score1 - score2) < 1e-3:
                self._winner = 0
            else:
                self._winner = 1 if score1 > score2 else 2

    def winner(self) -> int:
        return self._winner


def project_planet_future(
    sim: GalconSim,
    planet_id: int,
    player: int,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single planet's exact future ownership/ship trajectory over ``horizon`` ticks.

    Returns ``(my_arr, en_arr, proj_owner, proj_ships)`` each of shape ``(horizon,)``.
    ``proj_owner`` is ``+1`` (mine), ``-1`` (enemy), ``0`` (neutral).
    """
    p = sim.planets[planet_id]
    my_arr = np.zeros(horizon, dtype=np.float32)
    en_arr = np.zeros(horizon, dtype=np.float32)
    for fl in sim.fleets.values():
        if fl.target != planet_id:
            continue
        eta = sim.eta_ticks(fl.x, fl.y, p)
        idx = eta - 1
        if 0 <= idx < horizon:
            if fl.owner == player:
                my_arr[idx] += float(fl.ships)
            else:
                en_arr[idx] += float(fl.ships)

    proj_owner = np.zeros(horizon, dtype=np.float32)
    proj_ships = np.zeros(horizon, dtype=np.float32)

    cur_owner = 1 if p.owner == player else (0 if p.owner == 0 else -1)
    cur_ships = float(p.ships)
    growth_per_tick = (p.production / PRODUCTION_DIVISOR) * sim.tick_dt

    for t in range(horizon):
        if cur_owner != 0:
            cur_ships += growth_per_tick
        ma = float(my_arr[t])
        ea = float(en_arr[t])
        if ma > 0.0 or ea > 0.0:
            # Resolve 3-way (neutral / mine / enemy)
            f_my = ma + (cur_ships if cur_owner == 1 else 0.0)
            f_en = ea + (cur_ships if cur_owner == -1 else 0.0)
            f_nu = cur_ships if cur_owner == 0 else 0.0
            cands = [(f_my, 1), (f_en, -1), (f_nu, 0)]
            cands.sort(key=lambda kv: -kv[0])
            if abs(cands[0][0] - cands[1][0]) <= TIE_EPS:
                cur_ships = 0.0
            else:
                cur_owner = cands[0][1]
                cur_ships = cands[0][0] - cands[1][0]
        proj_owner[t] = float(cur_owner)
        proj_ships[t] = float(cur_ships)

    return my_arr, en_arr, proj_owner, proj_ships


def project_all_planet_futures(
    sim: GalconSim,
    planets: Sequence[Planet],
    player: int,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact future ownership/ship trajectories for all planets in ``(N, horizon)`` arrays.

    Fast-paths planets with zero incoming fleets in pure NumPy and runs exact 3-way combat only
    on planets that actually have incoming fleets. Matches ``project_planet_future`` bit-for-bit.
    """
    n = len(planets)
    my_arrs = np.zeros((n, horizon), dtype=np.float32)
    en_arrs = np.zeros((n, horizon), dtype=np.float32)
    if n == 0:
        z = np.zeros((0, horizon), dtype=np.float32)
        return my_arrs, en_arrs, z, z.copy()

    id_to_idx = {p.id: i for i, p in enumerate(planets)}
    has_inc = np.zeros(n, dtype=bool)
    step_dist = max(1e-6, sim.fleet_speed * sim.tick_dt)
    for fl in sim.fleets.values():
        idx_p = id_to_idx.get(fl.target)
        if idx_p is None:
            continue
        dst = planets[idx_p]
        dist = max(0.0, math.hypot(dst.x - fl.x, dst.y - fl.y) - dst.radius)
        eta = max(1, int(math.ceil((dist - ARRIVAL_EPS) / step_dist)))
        t_i = eta - 1
        if 0 <= t_i < horizon:
            has_inc[idx_p] = True
            if fl.owner == player:
                my_arrs[idx_p, t_i] += float(fl.ships)
            else:
                en_arrs[idx_p, t_i] += float(fl.ships)

    owners_i = np.array(
        [1.0 if p.owner == player else (0.0 if p.owner == 0 else -1.0) for p in planets],
        dtype=np.float64,
    )
    ships_0 = np.array([float(p.ships) for p in planets], dtype=np.float64)
    growth = np.array([(p.production / PRODUCTION_DIVISOR) * sim.tick_dt for p in planets], dtype=np.float64)
    t_steps = np.arange(1, horizon + 1, dtype=np.float64)[None, :]  # (1, H)

    # Exact fast-path for all planets (exact wherever has_inc is False)
    proj_owner = np.broadcast_to(owners_i[:, None], (n, horizon)).astype(np.float32).copy()
    proj_ships = (ships_0[:, None] + np.where(owners_i[:, None] != 0.0, growth[:, None] * t_steps, 0.0)).astype(
        np.float32
    )

    # Exact tick-by-tick 3-way combat only for planets with incoming fleets
    for i in np.flatnonzero(has_inc):
        cur_owner = int(owners_i[i])
        cur_ships = float(ships_0[i])
        g_i = float(growth[i])
        ma_row = my_arrs[i]
        ea_row = en_arrs[i]
        po_row = proj_owner[i]
        ps_row = proj_ships[i]
        for t in range(horizon):
            if cur_owner != 0:
                cur_ships += g_i
            ma = float(ma_row[t])
            ea = float(ea_row[t])
            if ma > 0.0 or ea > 0.0:
                f_my = ma + (cur_ships if cur_owner == 1 else 0.0)
                f_en = ea + (cur_ships if cur_owner == -1 else 0.0)
                f_nu = cur_ships if cur_owner == 0 else 0.0
                if f_my >= f_en and f_my >= f_nu:
                    top_v, top_o = f_my, 1
                    sec_v = f_en if f_en >= f_nu else f_nu
                elif f_en >= f_my and f_en >= f_nu:
                    top_v, top_o = f_en, -1
                    sec_v = f_my if f_my >= f_nu else f_nu
                else:
                    top_v, top_o = f_nu, 0
                    sec_v = f_my if f_my >= f_en else f_en
                if abs(top_v - sec_v) <= TIE_EPS:
                    cur_ships = 0.0
                else:
                    cur_owner = top_o
                    cur_ships = top_v - sec_v
            po_row[t] = float(cur_owner)
            ps_row[t] = float(cur_ships)

    return my_arrs, en_arrs, proj_owner, proj_ships


def evaluate_galcon_state(sim: GalconSim, player: int) -> float:
    """Fast tactical value estimate in ``[-1, +1]`` for lookahead & RL shaping."""
    if sim.done:
        w = sim.winner()
        return 1.0 if w == player else (0.0 if w == 0 else -1.0)
    opp = 2 if player == 1 else 1
    my_ships = sim.total_ships(player)
    en_ships = sim.total_ships(opp)
    my_prod = sim.total_production(player)
    en_prod = sim.total_production(opp)
    ship_adv = (my_ships - en_ships) / max(30.0, my_ships + en_ships)
    prod_adv = (my_prod - en_prod) / max(40.0, my_prod + en_prod)
    # Also value future projected ownership after current fleets land
    planets = list(sim.planets.values())
    _, _, po_all, _ = project_all_planet_futures(sim, planets, player, horizon=24)
    prods = np.array([p.production for p in planets], dtype=np.float64)
    po_end = po_all[:, -1]
    fut_my_prod = float(prods[po_end > 0].sum())
    fut_en_prod = float(prods[po_end < 0].sum())
    fut_prod_adv = (fut_my_prod - fut_en_prod) / max(40.0, fut_my_prod + fut_en_prod)
    raw = 1.15 * ship_adv + 0.70 * prod_adv + 1.10 * fut_prod_adv
    return float(math.tanh(raw))
