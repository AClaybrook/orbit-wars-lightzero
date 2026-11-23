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
    radius: float = 6.0


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
    """Classic Galcon planet radius scaled smoothly with production (15..100 -> ~14..28)."""
    return 11.5 + 0.165 * float(production)


def generate_galcon_map(
    *,
    seed: int = 42,
    num_planets: int = 18,
    width: float = 400.0,
    height: float = 240.0,
    home_ships: float = 100.0,
    home_prod: float = 100.0,
    symmetric: bool = True,
) -> List[Planet]:
    """Generate a balanced 2P Galcon map (180-degree rotationally symmetric by default)."""
    rng = random.Random(seed)
    cx, cy = width * 0.5, height * 0.5
    margin = 34.0
    home_r = planet_radius_from_production(home_prod)

    # Place P1 and P2 homes symmetrically on an outer ellipse
    angle = rng.uniform(0.0, 2.0 * math.pi)
    rx = max(20.0, (width * 0.5 - margin) * rng.uniform(0.70, 0.90))
    ry = max(20.0, (height * 0.5 - margin) * rng.uniform(0.70, 0.90))
    hx = cx + rx * math.cos(angle)
    hy = cy + ry * math.sin(angle)

    planets: List[Planet] = [
        Planet(id=1, owner=1, ships=float(home_ships), x=hx, y=hy, production=float(home_prod), radius=home_r),
        Planet(
            id=2,
            owner=2,
            ships=float(home_ships),
            x=width - hx,
            y=height - hy,
            production=float(home_prod),
            radius=home_r,
        ),
    ]

    min_gap = 12.0  # clearance so fleets pass cleanly between planets
    next_id = 3

    def fits(x: float, y: float, r: float) -> bool:
        if x - r < 8.0 or x + r > width - 8.0 or y - r < 8.0 or y + r > height - 8.0:
            return False
        for p in planets:
            if math.hypot(x - p.x, y - p.y) < r + p.radius + min_gap:
                return False
        return True

    target_neutrals = max(2, num_planets - 2)
    if symmetric:
        pairs = target_neutrals // 2
        for _ in range(pairs):
            prod = float(rng.randint(15, 100))
            r = planet_radius_from_production(prod)
            ships = float(rng.randint(0, 50))
            placed = False
            for _attempt in range(120):
                x = rng.uniform(r + 10.0, width - r - 10.0)
                y = rng.uniform(r + 10.0, height - r - 10.0)
                mx, my = width - x, height - y
                if math.hypot(x - mx, y - my) < 2.0 * r + min_gap:
                    continue
                if fits(x, y, r) and fits(mx, my, r):
                    planets.append(Planet(id=next_id, owner=0, ships=ships, x=x, y=y, production=prod, radius=r))
                    next_id += 1
                    planets.append(Planet(id=next_id, owner=0, ships=ships, x=mx, y=my, production=prod, radius=r))
                    next_id += 1
                    placed = True
                    break
            if not placed:
                break
        if target_neutrals % 2 == 1:
            prod = float(rng.randint(35, 95))
            r = planet_radius_from_production(prod)
            if fits(cx, cy, r):
                planets.append(
                    Planet(id=next_id, owner=0, ships=float(rng.randint(10, 45)), x=cx, y=cy, production=prod, radius=r)
                )
    else:
        for _ in range(target_neutrals):
            prod = float(rng.randint(15, 100))
            r = planet_radius_from_production(prod)
            ships = float(rng.randint(0, 50))
            for _attempt in range(120):
                x = rng.uniform(r + 10.0, width - r - 10.0)
                y = rng.uniform(r + 10.0, height - r - 10.0)
                if fits(x, y, r):
                    planets.append(Planet(id=next_id, owner=0, ships=ships, x=x, y=y, production=prod, radius=r))
                    next_id += 1
                    break

    return planets


class GalconSim:
    """Fast, cloneable Galcon simulation engine with send & mid-flight redirect support."""

    def __init__(
        self,
        planets: Sequence[Planet],
        fleets: Sequence[Fleet] = (),
        *,
        width: float = 400.0,
        height: float = 240.0,
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
        num_planets: int = 18,
        width: float = 400.0,
        height: float = 240.0,
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
        return max(1, int(math.ceil(dist / max(1e-6, step_dist))))

    def planet_eta_ticks(self, src: Planet, dst: Planet) -> int:
        """Exact arrival ticks for a fleet launched from ``src`` edge to ``dst`` edge."""
        dist = max(0.0, math.hypot(dst.x - src.x, dst.y - src.y) - src.radius - dst.radius)
        step_dist = self.fleet_speed * self.tick_dt
        return max(1, int(math.ceil(dist / max(1e-6, step_dist))))

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
            if dist <= dst.radius + step_dist:
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
            elif ranked[0][1] == ranked[1][1]:
                # Exact tie: planet keeps owner with 0 ships (neutral stays neutral)
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
            if cands[0][0] == cands[1][0]:
                cur_ships = 0.0
            else:
                cur_owner = cands[0][1]
                cur_ships = cands[0][0] - cands[1][0]
        proj_owner[t] = float(cur_owner)
        proj_ships[t] = float(cur_ships)

    return my_arr, en_arr, proj_owner, proj_ships


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
    fut_my_prod = 0.0
    fut_en_prod = 0.0
    for p in sim.planets.values():
        _, _, po, _ = project_planet_future(sim, p.id, player, horizon=24)
        if po[-1] > 0:
            fut_my_prod += p.production
        elif po[-1] < 0:
            fut_en_prod += p.production
    fut_prod_adv = (fut_my_prod - fut_en_prod) / max(40.0, fut_my_prod + fut_en_prod)
    raw = 1.15 * ship_adv + 0.70 * prod_adv + 1.10 * fut_prod_adv
    return float(math.tanh(raw))
