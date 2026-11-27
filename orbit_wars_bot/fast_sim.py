"""Fast, cheaply cloneable Orbit Wars simulator mirroring ``kaggle_environments``.

Designed for:
1. Tick-for-tick parity with ``kaggle_environments.envs.orbit_wars.orbit_wars``.
2. Microsecond-scale ``clone()`` for AlphaZero / online MCTS rollouts by sharing
   immutable initial planet geometry and comet path coordinate tables.
3. Seamless ``from_observation(obs, config)`` reconstruction during Kaggle games.
"""

from __future__ import annotations

import math
import pickle
import random
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from orbit_wars_bot.fast_sim_cpp import (
    HAS_ORBIT_SIM_CPP,
    CppFastOrbitSim,
    step_fast_sim_cpp,
    step_n_fast_sim_cpp,
)
from orbit_wars_bot.physics import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    DEFAULT_COMET_SPEED,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_SHIP_SPEED,
    MAX_PLANET_GROUPS,
    MIN_PLANET_GROUPS,
    MIN_STATIC_GROUPS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    distance,
    fleet_speed,
    point_to_segment_distance,
    swept_pair_hit,
)


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def generate_planets(rng: Optional[random.Random] = None) -> List[List[Any]]:
    """Mirror ``orbit_wars.generate_planets`` exactly."""
    if rng is None:
        rng = random
    planets: List[List[Any]] = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1.0 + math.log(prod)
        angle = rng.uniform(0.0, math.pi / 2.0)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        if x + r > BOARD_SIZE or x - r < 0.0 or y + r > BOARD_SIZE or y - r < 0.0:
            continue
        if (BOARD_SIZE - x) - r < 0.0 or (BOARD_SIZE - y) - r < 0.0:
            continue
        if (x - CENTER) < r + 5.0 or (y - CENTER) < r + 5.0:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        valid = True
        for tp in temp_planets:
            for p in planets:
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1.0 + math.log(prod)
        x = rng.uniform(CENTER + 15.0, BOARD_SIZE - r - 5.0)
        y = rng.uniform(CENTER + 15.0, BOARD_SIZE - r - 5.0)

        orbital_radius = distance((x, y), (CENTER, CENTER))
        if orbital_radius < SUN_RADIUS + r + 10.0:
            continue

        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0.0 or y + r > BOARD_SIZE or y - r < 0.0:
                continue

        valid = True
        ships = rng.randint(5, 30)
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT

                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break
            if not valid:
                break

        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4

    return planets


def generate_comet_paths(
    initial_planets: Sequence[Sequence[Any]],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: Optional[Sequence[int]] = None,
    comet_speed: float = DEFAULT_COMET_SPEED,
    rng: Optional[random.Random] = None,
) -> Optional[List[List[List[float]]]]:
    """Mirror ``orbit_wars.generate_comet_paths`` exactly."""
    if rng is None:
        rng = random
    comet_id_set = set(comet_planet_ids) if comet_planet_ids is not None else set()

    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60.0, 150.0)
        perihelion = a * (1.0 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1.0 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6.0, math.pi / 3.0)

        dense = []
        num = 5000
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * cos_phi - ey * sin_phi
            y = CENTER + ex * sin_phi + ey * cos_phi
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_id_set:
                continue
            pr = distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


class FastOrbitSim:
    """Fast, exact Orbit Wars game engine for MCTS rollouts and RL training."""

    __slots__ = (
        "planets",
        "initial_planets",
        "fleets",
        "comets",
        "comet_planet_ids",
        "angular_velocity",
        "next_fleet_id",
        "step_count",
        "num_agents",
        "max_steps",
        "ship_speed",
        "comet_speed",
        "episode_seed",
        "done",
        "rewards",
        "_orbit_meta",
        "use_cpp",
    )

    def __init__(
        self,
        planets: List[List[Any]],
        initial_planets: List[List[Any]],
        fleets: List[List[Any]],
        comets: List[Dict[str, Any]],
        comet_planet_ids: List[int],
        angular_velocity: float,
        next_fleet_id: int = 0,
        step_count: int = 0,
        num_agents: int = 2,
        max_steps: int = DEFAULT_EPISODE_STEPS,
        ship_speed: float = DEFAULT_SHIP_SPEED,
        comet_speed: float = DEFAULT_COMET_SPEED,
        episode_seed: Optional[int] = None,
        done: bool = False,
        rewards: Optional[List[int]] = None,
        _orbit_meta: Optional[Dict[int, Optional[Tuple[float, float]]]] = None,
        use_cpp: Optional[bool] = None,
    ) -> None:
        self.planets = planets
        self.initial_planets = initial_planets
        self.fleets = fleets
        self.comets = comets
        self.comet_planet_ids = comet_planet_ids
        self.angular_velocity = float(angular_velocity)
        self.next_fleet_id = int(next_fleet_id)
        self.step_count = int(step_count)
        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.ship_speed = float(ship_speed)
        self.comet_speed = float(comet_speed)
        self.episode_seed = episode_seed
        self.done = bool(done)
        self.rewards = list(rewards) if rewards is not None else [0] * self.num_agents
        if _orbit_meta is not None:
            self._orbit_meta = _orbit_meta
        else:
            self._orbit_meta = self._compute_orbit_meta(initial_planets, comet_planet_ids)
        self.use_cpp = bool(HAS_ORBIT_SIM_CPP) if use_cpp is None else bool(use_cpp and HAS_ORBIT_SIM_CPP)

    @staticmethod
    def _compute_orbit_meta(
        initial_planets: Sequence[Sequence[Any]],
        comet_planet_ids: Sequence[int],
    ) -> Dict[int, Optional[Tuple[float, float]]]:
        comet_set = set(comet_planet_ids)
        meta: Dict[int, Optional[Tuple[float, float]]] = {}
        for ip in initial_planets:
            pid = int(ip[0])
            if pid in comet_set:
                meta[pid] = None
                continue
            dx = float(ip[2]) - CENTER
            dy = float(ip[3]) - CENTER
            r = math.sqrt(dx ** 2 + dy ** 2)
            if r + float(ip[4]) < ROTATION_RADIUS_LIMIT:
                meta[pid] = (r, math.atan2(dy, dx))
            else:
                meta[pid] = None
        return meta

    @classmethod
    def from_seed(
        cls,
        seed: Optional[int] = None,
        *,
        num_agents: int = 2,
        max_steps: int = DEFAULT_EPISODE_STEPS,
        ship_speed: float = DEFAULT_SHIP_SPEED,
        comet_speed: float = DEFAULT_COMET_SPEED,
        use_cpp: Optional[bool] = None,
    ) -> "FastOrbitSim":
        """Create a fresh Orbit Wars game mirroring ``orbit_wars.interpreter`` init."""
        if seed is None:
            seed = random.randint(0, 2 ** 31 - 1)
        init_rng = random.Random(seed)
        angular_velocity = init_rng.uniform(0.025, 0.05)
        planets = generate_planets(init_rng)
        initial_planets = [p.copy() for p in planets]

        num_groups = len(planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4
            if num_agents == 2:
                planets[base][1] = 0
                planets[base][5] = 10
                planets[base + 3][1] = 1
                planets[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    planets[base + j][1] = j
                    planets[base + j][5] = 10

        return cls(
            planets=planets,
            initial_planets=initial_planets,
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=angular_velocity,
            next_fleet_id=0,
            step_count=0,
            num_agents=num_agents,
            max_steps=max_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            episode_seed=seed,
            done=False,
            use_cpp=use_cpp,
        )

    @classmethod
    def from_observation(
        cls,
        obs: Any,
        config: Any = None,
        *,
        num_agents: Optional[int] = None,
        episode_seed: Optional[int] = None,
        use_cpp: Optional[bool] = None,
    ) -> "FastOrbitSim":
        """Reconstruct a ``FastOrbitSim`` from a Kaggle ``obs`` and optional ``config``."""
        planets = [list(p) for p in _get_field(obs, "planets", [])]
        raw_init = _get_field(obs, "initial_planets", None)
        initial_planets = [list(p) for p in raw_init] if raw_init else [p[:] for p in planets]
        fleets = [list(f) for f in _get_field(obs, "fleets", [])]
        raw_comets = _get_field(obs, "comets", []) or []
        comets = [
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": g["paths"],  # immutable coordinate arrays; safe to share
                "path_index": int(g["path_index"]),
            }
            for g in raw_comets
        ]
        comet_planet_ids = list(_get_field(obs, "comet_planet_ids", []) or [])
        angular_velocity = float(_get_field(obs, "angular_velocity", 0.0))
        next_fleet_id = int(_get_field(obs, "next_fleet_id", len(fleets)))
        step_count = int(_get_field(obs, "step", 0))

        if num_agents is None:
            max_pid = int(_get_field(obs, "player", 0))
            for p in planets:
                if int(p[1]) > max_pid:
                    max_pid = int(p[1])
            for f in fleets:
                if int(f[1]) > max_pid:
                    max_pid = int(f[1])
            num_agents = 4 if max_pid >= 2 else 2

        max_steps = int(_get_field(config, "episodeSteps", DEFAULT_EPISODE_STEPS)) if config is not None else DEFAULT_EPISODE_STEPS
        ship_speed = float(_get_field(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config is not None else DEFAULT_SHIP_SPEED
        comet_speed = float(_get_field(config, "cometSpeed", DEFAULT_COMET_SPEED)) if config is not None else DEFAULT_COMET_SPEED

        return cls(
            planets=planets,
            initial_planets=initial_planets,
            fleets=fleets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            angular_velocity=angular_velocity,
            next_fleet_id=next_fleet_id,
            step_count=step_count,
            num_agents=num_agents,
            max_steps=max_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            episode_seed=episode_seed,
            done=False,
            use_cpp=use_cpp,
        )

    def clone(self, use_cpp: Optional[bool] = None) -> "FastOrbitSim":
        """Create a fast independent copy suitable for MCTS rollouts."""
        return FastOrbitSim(
            planets=[p[:] for p in self.planets],
            initial_planets=self.initial_planets,  # only replaced on comet spawn/expiry
            fleets=[f[:] for f in self.fleets],
            comets=[
                {
                    "planet_ids": g["planet_ids"][:],
                    "paths": g["paths"],
                    "path_index": g["path_index"],
                }
                for g in self.comets
            ],
            comet_planet_ids=self.comet_planet_ids[:],
            angular_velocity=self.angular_velocity,
            next_fleet_id=self.next_fleet_id,
            step_count=self.step_count,
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            ship_speed=self.ship_speed,
            comet_speed=self.comet_speed,
            episode_seed=self.episode_seed,
            done=self.done,
            rewards=self.rewards[:],
            _orbit_meta=self._orbit_meta,
            use_cpp=self.use_cpp if use_cpp is None else use_cpp,
        )

    def to_cpp(self) -> CppFastOrbitSim:
        """Export into a resident zero-marshal ``CppFastOrbitSim`` instance."""
        return CppFastOrbitSim.from_fast_sim(self)

    def step_n(self, n_ticks: int) -> bool:
        """Advance up to ``n_ticks`` empty-action ticks (accelerated via C++17 when available)."""
        n_ticks = int(n_ticks)
        if n_ticks <= 0 or self.done:
            return self.done
        if self.use_cpp and HAS_ORBIT_SIM_CPP:
            return step_n_fast_sim_cpp(self, n_ticks)
        empty: List[List[Any]] = [[] for _ in range(self.num_agents)]
        for _ in range(n_ticks):
            if self.done:
                break
            self._step_python(empty)
        return self.done

    def step(self, actions: Sequence[Optional[Sequence[Sequence[Any]]]]) -> bool:
        """Advance the simulation by one exact Orbit Wars tick.

        Args:
            actions: Sequence of length ``num_agents``, where ``actions[i]`` is
                a list of ``[from_planet_id, angle, ships]`` moves for player ``i``.

        Returns:
            ``self.done`` (True if the game terminated on this tick).
        """
        if self.done:
            return True
        if self.use_cpp and HAS_ORBIT_SIM_CPP:
            return step_fast_sim_cpp(self, actions)
        return self._step_python(actions)

    def _step_python(self, actions: Sequence[Optional[Sequence[Sequence[Any]]]]) -> bool:
        """Pure-Python fallback & reference implementation for one Orbit Wars tick."""
        if self.done:
            return True

        # Remove expired comets before fleet launch
        if self.comets:
            expired_comet_pids = []
            for group in self.comets:
                idx = group["path_index"]
                for i, pid in enumerate(group["planet_ids"]):
                    if idx >= len(group["paths"][i]):
                        expired_comet_pids.append(pid)
            if expired_comet_pids:
                self._purge_comets(set(expired_comet_pids))

        # Spawn comets at designated steps (when episode_seed is known)
        step = self.step_count
        if (step + 1) in COMET_SPAWN_STEPS and self.episode_seed is not None:
            comet_rng = random.Random(f"orbit_wars-comet-{self.episode_seed}-{step + 1}")
            comet_paths = generate_comet_paths(
                self.initial_planets,
                self.angular_velocity,
                step + 1,
                self.comet_planet_ids,
                self.comet_speed,
                rng=comet_rng,
            )
            if comet_paths:
                next_id = max(p[0] for p in self.planets) + 1
                comet_ships = min(
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                )
                group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
                # Copy initial_planets before mutating so shared clones stay isolated
                self.initial_planets = [p[:] for p in self.initial_planets]
                self._orbit_meta = dict(self._orbit_meta)
                for i in range(len(comet_paths)):
                    pid = next_id + i
                    group["planet_ids"].append(pid)
                    self.comet_planet_ids.append(pid)
                    planet = [
                        pid,
                        -1,
                        -99,
                        -99,
                        COMET_RADIUS,
                        comet_ships,
                        COMET_PRODUCTION,
                    ]
                    self.planets.append(planet)
                    self.initial_planets.append(planet[:])
                    self._orbit_meta[pid] = None
                self.comets.append(group)

        # Build fast planet lookup by ID
        planet_by_id = {p[0]: p for p in self.planets}

        # 0. Fleet launch
        for player_id in range(min(self.num_agents, len(actions))):
            action = actions[player_id]
            if not action:
                continue
            for move in action:
                if len(move) != 3:
                    continue
                from_id, angle, ships = move
                ships = int(ships)
                from_planet = planet_by_id.get(from_id)
                if from_planet is not None and from_planet[1] == player_id:
                    if from_planet[5] >= ships and ships > 0:
                        from_planet[5] -= ships
                        ang_f = float(angle)
                        start_x = from_planet[2] + math.cos(ang_f) * (from_planet[4] + 0.1)
                        start_y = from_planet[3] + math.sin(ang_f) * (from_planet[4] + 0.1)
                        self.fleets.append(
                            [
                                self.next_fleet_id,
                                player_id,
                                start_x,
                                start_y,
                                ang_f,
                                from_id,
                                ships,
                            ]
                        )
                        self.next_fleet_id += 1

        # 1. Production
        for planet in self.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

        # 2. Compute planet end-of-tick positions up front
        ang_vel = self.angular_velocity
        comet_pid_set = set(self.comet_planet_ids) if self.comet_planet_ids else ()
        planet_paths: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], bool]] = {}
        expired_comet_pids = []

        for planet in self.planets:
            pid = planet[0]
            if pid in comet_pid_set:
                continue
            old_pos = (planet[2], planet[3])
            meta = self._orbit_meta.get(pid)
            if meta is not None:
                orb_r, init_angle = meta
                cur_angle = init_angle + ang_vel * step
                new_pos = (
                    CENTER + orb_r * math.cos(cur_angle),
                    CENTER + orb_r * math.sin(cur_angle),
                )
            else:
                new_pos = old_pos
            planet_paths[pid] = (old_pos, new_pos, True)

        for group in self.comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                planet = planet_by_id.get(pid)
                if planet is None:
                    continue
                p_path = group["paths"][i]
                old_pos = (planet[2], planet[3])
                if idx >= len(p_path):
                    expired_comet_pids.append(pid)
                    planet_paths[pid] = (old_pos, old_pos, True)
                else:
                    new_pos = (p_path[idx][0], p_path[idx][1])
                    planet_paths[pid] = (old_pos, new_pos, old_pos[0] >= 0)

        # 3. Fleet movement & swept-pair collision detection
        max_speed = self.ship_speed
        surviving_fleets: List[List[Any]] = []
        combat_lists: Dict[int, List[List[Any]]] = {}

        for fleet in self.fleets:
            angle = fleet[4]
            ships = fleet[6]
            spd = fleet_speed(ships, max_speed)
            old_pos = (fleet[2], fleet[3])
            nx = fleet[2] + math.cos(angle) * spd
            ny = fleet[3] + math.sin(angle) * spd
            fleet[2] = nx
            fleet[3] = ny
            new_pos = (nx, ny)

            hit_planet = False
            for planet in self.planets:
                pid = planet[0]
                ppath = planet_paths.get(pid)
                if ppath is None or not ppath[2]:
                    continue
                if swept_pair_hit(old_pos, new_pos, ppath[0], ppath[1], planet[4]):
                    bucket = combat_lists.get(pid)
                    if bucket is None:
                        combat_lists[pid] = [fleet]
                    else:
                        bucket.append(fleet)
                    hit_planet = True
                    break
            if hit_planet:
                continue

            if not (0.0 <= nx <= BOARD_SIZE and 0.0 <= ny <= BOARD_SIZE):
                continue

            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
                continue

            surviving_fleets.append(fleet)

        self.fleets = surviving_fleets

        # 4. Apply planet movement
        for planet in self.planets:
            ppath = planet_paths.get(planet[0])
            if ppath is not None:
                planet[2], planet[3] = ppath[1]

        # Remove expired comets before combat resolution (mirrors orbit_wars.py)
        if expired_comet_pids:
            expired_set = set(expired_comet_pids)
            self._purge_comets(expired_set)
            for pid in expired_set:
                planet_by_id.pop(pid, None)

        # 5. Combat resolution
        for pid, planet_fleets in combat_lists.items():
            planet = planet_by_id.get(pid)
            if planet is None or not planet_fleets:
                continue

            player_ships: Dict[int, int] = {}
            for fleet in planet_fleets:
                owner = fleet[1]
                player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

            if not player_ships:
                continue

            sorted_players = sorted(
                player_ships.items(), key=lambda item: item[1], reverse=True
            )
            top_player, top_ships = sorted_players[0]
            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = top_ships - second_ships
                if top_ships == second_ships:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships

            if survivor_ships > 0:
                if planet[1] == survivor_owner:
                    planet[5] += survivor_ships
                else:
                    planet[5] -= survivor_ships
                    if planet[5] < 0:
                        planet[1] = survivor_owner
                        planet[5] = abs(planet[5])

        # 6. Termination check
        terminated = step >= self.max_steps - 2
        if not terminated:
            alive_players = set()
            for p in self.planets:
                if p[1] != -1:
                    alive_players.add(p[1])
            for f in self.fleets:
                alive_players.add(f[1])
            if len(alive_players) <= 1:
                terminated = True

        self.step_count = step + 1

        if terminated:
            self.done = True
            sc = self.scores()
            max_score = max(sc) if sc else 0
            self.rewards = [
                1 if (s == max_score and max_score > 0) else -1 for s in sc
            ]
        return self.done

    def _purge_comets(self, expired_set: set[int]) -> None:
        self.planets = [p for p in self.planets if p[0] not in expired_set]
        self.initial_planets = [p for p in self.initial_planets if p[0] not in expired_set]
        self.comet_planet_ids = [pid for pid in self.comet_planet_ids if pid not in expired_set]
        new_comets = []
        for group in self.comets:
            kept_ids = [pid for pid in group["planet_ids"] if pid not in expired_set]
            if kept_ids:
                new_comets.append(
                    {
                        "planet_ids": kept_ids,
                        "paths": group["paths"],
                        "path_index": group["path_index"],
                    }
                )
        self.comets = new_comets

    def scores(self) -> List[int]:
        """Return total ships (planets + active fleets) per player."""
        sc = [0] * self.num_agents
        for p in self.planets:
            if 0 <= p[1] < self.num_agents:
                sc[p[1]] += int(p[5])
        for f in self.fleets:
            if 0 <= f[1] < self.num_agents:
                sc[f[1]] += int(f[6])
        return sc

    def total_production(self, player: int) -> int:
        return sum(int(p[6]) for p in self.planets if p[1] == player)

    def planet_count(self, player: int) -> int:
        return sum(1 for p in self.planets if p[1] == player)

    def winner(self) -> int:
        """Return winning player id (0..num_agents-1), or -1 on draw / not done."""
        if not self.done:
            return -1
        sc = self.scores()
        max_s = max(sc)
        winners = [i for i, s in enumerate(sc) if s == max_s]
        if len(winners) == 1:
            return winners[0]
        return -1

    def to_observation(self, player: int = 0) -> Dict[str, Any]:
        """Export a Kaggle-compatible observation dictionary for ``player``."""
        return {
            "step": self.step_count,
            "player": player,
            "planets": [p[:] for p in self.planets],
            "initial_planets": [p[:] for p in self.initial_planets],
            "fleets": [f[:] for f in self.fleets],
            "next_fleet_id": self.next_fleet_id,
            "angular_velocity": self.angular_velocity,
            "comets": [
                {
                    "planet_ids": g["planet_ids"][:],
                    "paths": g["paths"],
                    "path_index": g["path_index"],
                }
                for g in self.comets
            ],
            "comet_planet_ids": self.comet_planet_ids[:],
            "remainingOverageTime": 60.0,
        }

    def to_config(self) -> Dict[str, Any]:
        return {
            "episodeSteps": self.max_steps,
            "shipSpeed": self.ship_speed,
            "cometSpeed": self.comet_speed,
        }

    def serialize_bytes(self) -> bytes:
        """Compact compressed snapshot for LightZero state serialization."""
        payload = (
            self.planets,
            self.initial_planets,
            self.fleets,
            self.comets,
            self.comet_planet_ids,
            self.angular_velocity,
            self.next_fleet_id,
            self.step_count,
            self.num_agents,
            self.max_steps,
            self.ship_speed,
            self.comet_speed,
            self.episode_seed,
            self.done,
            self.rewards,
        )
        return zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), level=1)

    @classmethod
    def deserialize_bytes(cls, data: bytes) -> "FastOrbitSim":
        (
            planets,
            initial_planets,
            fleets,
            comets,
            comet_planet_ids,
            angular_velocity,
            next_fleet_id,
            step_count,
            num_agents,
            max_steps,
            ship_speed,
            comet_speed,
            episode_seed,
            done,
            rewards,
        ) = pickle.loads(zlib.decompress(data))
        return cls(
            planets=planets,
            initial_planets=initial_planets,
            fleets=fleets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            angular_velocity=angular_velocity,
            next_fleet_id=next_fleet_id,
            step_count=step_count,
            num_agents=num_agents,
            max_steps=max_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            episode_seed=episode_seed,
            done=done,
            rewards=rewards,
        )
