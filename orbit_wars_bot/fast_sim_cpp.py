"""Auto-compiled C++17 Orbit Wars simulator and physics hot-path accelerator.

Compiles ``orbit_wars_bot/orbit_sim_core.cpp`` via::

    g++ -O3 -march=native -ffp-contract=off -shared -fPIC -std=c++17

into a SHA-256 hash-keyed shared library in the OS temp directory (shared safely
across worker processes via atomic ``os.replace``).

Exposes:
- ``HAS_ORBIT_SIM_CPP``: ``True`` when the compiled native library loaded cleanly.
- ``CppFastOrbitSim``: Resident C++17 Orbit Wars engine with ~1 us clone/step,
  ``step_n()``, and lockstep ``step_batch()`` for online search & RL rollouts.
- ``build_shared_planet_physics_cpp()`` and ``solve_intercept_cpp()``: Bit-exact
  C++ implementations of ``planet_transformer.build_shared_planet_physics`` and
  ``physics.solve_intercept``.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orbit_wars_bot.physics import (
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    DEFAULT_COMET_SPEED,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_SHIP_SPEED,
    ROTATION_RADIUS_LIMIT,
    FleetArrival,
    InterceptSolution,
    build_comet_lookup,
)

MAX_PLANETS_CPP = 44
TIMELINE_HORIZON_CPP = 20
MAX_ROUTE_ETA_CPP = 20

_COMPILE_FLAGS = ("-O3", "-march=native", "-ffp-contract=off", "-shared", "-fPIC", "-std=c++17")
_C_VP = ctypes.c_void_p
_C_I32 = ctypes.c_int32
_C_F64 = ctypes.c_double


class CMove(ctypes.Structure):
    """C ABI launch move matching ``struct CMove`` in ``orbit_sim_core.cpp``."""

    _fields_ = [
        ("player_id", ctypes.c_int32),
        ("from_id", ctypes.c_int32),
        ("angle", ctypes.c_double),
        ("ships", ctypes.c_int32),
    ]


def _find_gpp() -> Optional[str]:
    if sys.platform == "win32":
        return None
    for cand in ("g++", "clang++"):
        path = shutil.which(cand)
        if path:
            return path
    return None


def _load_or_build_lib() -> Optional[ctypes.CDLL]:
    gpp = _find_gpp()
    if gpp is None:
        return None

    cpp_path = Path(__file__).resolve().parent / "orbit_sim_core.cpp"
    if not cpp_path.is_file():
        return None

    try:
        src_bytes = cpp_path.read_bytes()
    except OSError:
        return None

    flags_key = " ".join(_COMPILE_FLAGS).encode("utf-8")
    h = hashlib.sha256(src_bytes + flags_key).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "orbit_wars_fast_sim_cpp_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    so_path = cache_dir / f"orbit_sim_core_{h}.so"

    if not so_path.is_file():
        pid = os.getpid()
        so_tmp = cache_dir / f"orbit_sim_core_{h}_{pid}.so"
        try:
            subprocess.run(
                [gpp, *_COMPILE_FLAGS, str(cpp_path), "-lm", "-o", str(so_tmp)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(so_tmp, so_path)
        except Exception:
            try:
                so_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    try:
        lib = ctypes.CDLL(str(so_path))

        lib.orbit_sim_create.argtypes = [
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_F64, _C_I32, _C_I32, _C_I32, _C_I32, _C_F64, _C_F64, _C_I32, _C_VP,
        ]
        lib.orbit_sim_create.restype = _C_VP

        lib.orbit_sim_destroy.argtypes = [_C_VP]
        lib.orbit_sim_destroy.restype = None

        lib.orbit_sim_clone.argtypes = [_C_VP]
        lib.orbit_sim_clone.restype = _C_VP

        lib.orbit_sim_copy_into.argtypes = [_C_VP, _C_VP]
        lib.orbit_sim_copy_into.restype = None

        lib.orbit_sim_add_comet_group.argtypes = [
            _C_VP, _C_I32, _C_I32, _C_I32, _C_VP, _C_VP, _C_VP, _C_I32, _C_I32, _C_F64, _C_I32,
        ]
        lib.orbit_sim_add_comet_group.restype = None

        lib.orbit_sim_step.argtypes = [_C_VP, _C_VP, _C_I32]
        lib.orbit_sim_step.restype = _C_I32

        lib.orbit_sim_step_n.argtypes = [_C_VP, _C_I32]
        lib.orbit_sim_step_n.restype = _C_I32

        lib.orbit_sim_step_batch.argtypes = [_C_VP, _C_I32, _C_VP, _C_VP, _C_I32]
        lib.orbit_sim_step_batch.restype = None

        lib.orbit_sim_get_meta.argtypes = [_C_VP, _C_VP]
        lib.orbit_sim_get_meta.restype = None

        lib.orbit_sim_get_scores.argtypes = [_C_VP, _C_VP, _C_VP]
        lib.orbit_sim_get_scores.restype = None

        lib.orbit_sim_winner.argtypes = [_C_VP]
        lib.orbit_sim_winner.restype = _C_I32

        lib.orbit_sim_export_planets.argtypes = [
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.orbit_sim_export_planets.restype = None

        lib.orbit_sim_export_fleets.argtypes = [
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.orbit_sim_export_fleets.restype = None

        lib.orbit_sim_export_comet_indices.argtypes = [_C_VP, _C_VP]
        lib.orbit_sim_export_comet_indices.restype = None

        lib.orbit_sim_build_shared_physics.argtypes = [
            _C_VP, _C_I32, _C_I32, _C_I32,
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.orbit_sim_build_shared_physics.restype = _C_I32

        lib.orbit_compute_shared_physics_arrays.argtypes = [
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_F64, _C_F64, _C_I32, _C_I32,
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
            _C_I32, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.orbit_compute_shared_physics_arrays.restype = _C_I32

        lib.orbit_solve_intercept.argtypes = [
            _C_I32, _C_I32, _C_I32, _C_I32, _C_I32, _C_VP, _C_VP, _C_VP, _C_I32, _C_F64,
            _C_VP, _C_VP, _C_VP, _C_VP, _C_VP,
        ]
        lib.orbit_solve_intercept.restype = _C_I32

        return lib
    except Exception:
        return None


_LIB: Optional[ctypes.CDLL] = _load_or_build_lib()
HAS_ORBIT_SIM_CPP: bool = _LIB is not None


class _TlsBuffers(threading.local):
    def __init__(self) -> None:
        self.meta = np.zeros((8,), dtype=np.int32)
        self.scores = np.zeros((8,), dtype=np.int32)
        self.rewards = np.zeros((8,), dtype=np.int32)
        self.move_cap = 64
        self.moves = (CMove * self.move_cap)()
        self.out_ang = ctypes.c_double()
        self.out_eta = ctypes.c_int32()
        self.out_ix = ctypes.c_double()
        self.out_iy = ctypes.c_double()
        self.out_spd = ctypes.c_double()
        self.arr_cnt = np.zeros((1,), dtype=np.int32)

    def ensure_moves(self, n: int) -> Any:
        if n > self.move_cap:
            self.move_cap = max(n, self.move_cap * 2)
            self.moves = (CMove * self.move_cap)()
        return self.moves


_TLS = _TlsBuffers()


def _obs_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _compute_orbit_params(
    planets: Sequence[Sequence[Any]],
    initial_by_id: Dict[int, Sequence[Any]],
    comet_ids: set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(planets)
    is_orb = np.zeros((n,), dtype=np.int32)
    orb_r = np.zeros((n,), dtype=np.float64)
    init_ang = np.zeros((n,), dtype=np.float64)
    is_com = np.zeros((n,), dtype=np.int32)
    for i, p in enumerate(planets):
        pid = int(p[0])
        if pid in comet_ids:
            is_com[i] = 1
            continue
        ip = initial_by_id.get(pid)
        if ip is None:
            continue
        dx = float(ip[2]) - CENTER
        dy = float(ip[3]) - CENTER
        r = math.sqrt(dx * dx + dy * dy)
        if r + float(ip[4]) < ROTATION_RADIUS_LIMIT:
            is_orb[i] = 1
            orb_r[i] = r
            init_ang[i] = math.atan2(dy, dx)
    return is_orb, orb_r, init_ang, is_com


class CppFastOrbitSim:
    """Resident C++17 Orbit Wars engine mirroring ``FastOrbitSim`` bit-for-bit."""

    __slots__ = (
        "_handle",
        "initial_planets",
        "comet_planet_ids",
        "_comet_specs",
        "angular_velocity",
        "ship_speed",
        "comet_speed",
        "episode_seed",
        "num_agents",
        "max_steps",
        "step_count",
        "next_fleet_id",
        "done",
        "rewards",
        "_n_planets",
        "_n_fleets",
        "_initial_by_id",
        "_orbit_meta",
    )

    def __init__(
        self,
        handle: int,
        *,
        initial_planets: List[List[Any]],
        comet_planet_ids: List[int],
        comet_specs: List[Dict[str, Any]],
        angular_velocity: float,
        ship_speed: float,
        comet_speed: float,
        episode_seed: Optional[int],
        num_agents: int,
        max_steps: int,
        step_count: int,
        next_fleet_id: int,
        done: bool,
        rewards: Sequence[int],
        n_planets: int,
        n_fleets: int,
        initial_by_id: Optional[Dict[int, Sequence[Any]]] = None,
        orbit_meta: Optional[Dict[int, Optional[Tuple[float, float]]]] = None,
    ) -> None:
        self._handle = int(handle)
        self.initial_planets = initial_planets
        self.comet_planet_ids = comet_planet_ids
        self._comet_specs = comet_specs
        self.angular_velocity = float(angular_velocity)
        self.ship_speed = float(ship_speed)
        self.comet_speed = float(comet_speed)
        self.episode_seed = episode_seed
        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.step_count = int(step_count)
        self.next_fleet_id = int(next_fleet_id)
        self.done = bool(done)
        self.rewards = list(rewards)
        self._n_planets = int(n_planets)
        self._n_fleets = int(n_fleets)
        self._initial_by_id = (
            initial_by_id
            if initial_by_id is not None
            else {int(ip[0]): ip for ip in initial_planets}
        )
        if orbit_meta is not None:
            self._orbit_meta = orbit_meta
        else:
            comet_set = set(comet_planet_ids)
            meta: Dict[int, Optional[Tuple[float, float]]] = {}
            for ip in initial_planets:
                pid = int(ip[0])
                if pid in comet_set:
                    meta[pid] = None
                    continue
                dx = float(ip[2]) - CENTER
                dy = float(ip[3]) - CENTER
                r = math.sqrt(dx * dx + dy * dy)
                meta[pid] = (r, math.atan2(dy, dx)) if (r + float(ip[4]) < ROTATION_RADIUS_LIMIT) else None
            self._orbit_meta = meta

    def __del__(self) -> None:
        h = getattr(self, "_handle", 0)
        if h and _LIB is not None:
            _LIB.orbit_sim_destroy(h)
            self._handle = 0

    def _sync_meta(self) -> None:
        assert _LIB is not None
        meta = _TLS.meta
        _LIB.orbit_sim_get_meta(self._handle, meta.ctypes.data)
        self.step_count = int(meta[0])
        self.next_fleet_id = int(meta[1])
        self.num_agents = int(meta[2])
        self.max_steps = int(meta[3])
        self.done = bool(meta[4])
        self._n_planets = int(meta[5])
        self._n_fleets = int(meta[6])
        if self.done:
            rew = _TLS.rewards
            _LIB.orbit_sim_get_scores(self._handle, _TLS.scores.ctypes.data, rew.ctypes.data)
            self.rewards = [int(rew[i]) for i in range(self.num_agents)]

    @classmethod
    def from_parts(
        cls,
        *,
        planets: Sequence[Sequence[Any]],
        initial_planets: Sequence[Sequence[Any]],
        fleets: Sequence[Sequence[Any]],
        comets: Sequence[Dict[str, Any]],
        comet_planet_ids: Sequence[int],
        angular_velocity: float,
        next_fleet_id: int = 0,
        step_count: int = 0,
        num_agents: int = 2,
        max_steps: int = DEFAULT_EPISODE_STEPS,
        ship_speed: float = DEFAULT_SHIP_SPEED,
        comet_speed: float = DEFAULT_COMET_SPEED,
        episode_seed: Optional[int] = None,
        done: bool = False,
        rewards: Optional[Sequence[int]] = None,
        orbit_meta: Optional[Dict[int, Optional[Tuple[float, float]]]] = None,
    ) -> "CppFastOrbitSim":
        if _LIB is None:
            raise RuntimeError("Orbit Wars C++ engine not compiled (HAS_ORBIT_SIM_CPP=False).")

        np_cnt = len(planets)
        nf_cnt = len(fleets)
        init_list = [list(p) for p in initial_planets]
        init_by_id = {int(p[0]): p for p in init_list}
        comet_ids_list = [int(pid) for pid in comet_planet_ids]
        comet_id_set = set(comet_ids_list)
        for g in comets:
            for pid in g["planet_ids"]:
                comet_id_set.add(int(pid))

        p_id = np.asarray([int(p[0]) for p in planets], dtype=np.int32)
        p_own = np.asarray([int(p[1]) for p in planets], dtype=np.int32)
        p_x = np.asarray([float(p[2]) for p in planets], dtype=np.float64)
        p_y = np.asarray([float(p[3]) for p in planets], dtype=np.float64)
        p_r = np.asarray([float(p[4]) for p in planets], dtype=np.float64)
        p_sh = np.asarray([int(p[5]) for p in planets], dtype=np.int32)
        p_pr = np.asarray([int(p[6]) for p in planets], dtype=np.int32)
        p_orb, p_or, p_oa, p_ic = _compute_orbit_params(planets, init_by_id, comet_id_set)

        f_id = np.asarray([int(f[0]) for f in fleets], dtype=np.int32) if nf_cnt else np.zeros((0,), dtype=np.int32)
        f_own = np.asarray([int(f[1]) for f in fleets], dtype=np.int32) if nf_cnt else np.zeros((0,), dtype=np.int32)
        f_x = np.asarray([float(f[2]) for f in fleets], dtype=np.float64) if nf_cnt else np.zeros((0,), dtype=np.float64)
        f_y = np.asarray([float(f[3]) for f in fleets], dtype=np.float64) if nf_cnt else np.zeros((0,), dtype=np.float64)
        f_ang = np.asarray([float(f[4]) for f in fleets], dtype=np.float64) if nf_cnt else np.zeros((0,), dtype=np.float64)
        f_fr = np.asarray([int(f[5]) for f in fleets], dtype=np.int32) if nf_cnt else np.zeros((0,), dtype=np.int32)
        f_sh = np.asarray([int(f[6]) for f in fleets], dtype=np.int32) if nf_cnt else np.zeros((0,), dtype=np.int32)

        rew_list = [int(r) for r in rewards] if rewards is not None else [0] * int(num_agents)
        rew_arr = np.asarray(rew_list + [0] * max(0, 8 - len(rew_list)), dtype=np.int32)

        handle = _LIB.orbit_sim_create(
            np_cnt,
            p_id.ctypes.data, p_own.ctypes.data, p_x.ctypes.data, p_y.ctypes.data,
            p_r.ctypes.data, p_sh.ctypes.data, p_pr.ctypes.data, p_orb.ctypes.data,
            p_or.ctypes.data, p_oa.ctypes.data, p_ic.ctypes.data,
            nf_cnt,
            f_id.ctypes.data, f_own.ctypes.data, f_x.ctypes.data, f_y.ctypes.data,
            f_ang.ctypes.data, f_fr.ctypes.data, f_sh.ctypes.data,
            float(angular_velocity),
            int(next_fleet_id),
            int(step_count),
            int(num_agents),
            int(max_steps),
            float(ship_speed),
            float(comet_speed),
            1 if done else 0,
            rew_arr.ctypes.data,
        )

        comet_specs: List[Dict[str, Any]] = []
        for g in comets:
            pids = [int(pid) for pid in g["planet_ids"]]
            paths = g["paths"]
            p_idx = int(g["path_index"])
            comet_specs.append({"planet_ids": pids, "paths": paths, "path_index": p_idx})
            m = len(pids)
            if m > 0:
                plen = len(paths[0])
                px_arr = np.asarray([[float(pt[0]) for pt in p_row] for p_row in paths], dtype=np.float64)
                py_arr = np.asarray([[float(pt[1]) for pt in p_row] for p_row in paths], dtype=np.float64)
                pids_arr = np.asarray(pids, dtype=np.int32)
                _LIB.orbit_sim_add_comet_group(
                    handle, p_idx, m, plen,
                    pids_arr.ctypes.data, px_arr.ctypes.data, py_arr.ctypes.data,
                    0, 0, COMET_RADIUS, COMET_PRODUCTION,
                )

        return cls(
            handle,
            initial_planets=init_list,
            comet_planet_ids=comet_ids_list,
            comet_specs=comet_specs,
            angular_velocity=angular_velocity,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            episode_seed=episode_seed,
            num_agents=num_agents,
            max_steps=max_steps,
            step_count=step_count,
            next_fleet_id=next_fleet_id,
            done=done,
            rewards=rew_list,
            n_planets=np_cnt,
            n_fleets=nf_cnt,
            initial_by_id=init_by_id,
            orbit_meta=orbit_meta,
        )

    @classmethod
    def from_fast_sim(cls, sim: Any) -> "CppFastOrbitSim":
        return cls.from_parts(
            planets=sim.planets,
            initial_planets=sim.initial_planets,
            fleets=sim.fleets,
            comets=sim.comets,
            comet_planet_ids=sim.comet_planet_ids,
            angular_velocity=sim.angular_velocity,
            next_fleet_id=sim.next_fleet_id,
            step_count=sim.step_count,
            num_agents=sim.num_agents,
            max_steps=sim.max_steps,
            ship_speed=sim.ship_speed,
            comet_speed=sim.comet_speed,
            episode_seed=sim.episode_seed,
            done=sim.done,
            rewards=sim.rewards,
            orbit_meta=getattr(sim, "_orbit_meta", None),
        )

    @classmethod
    def from_seed(
        cls,
        seed: Optional[int] = None,
        *,
        num_agents: int = 2,
        max_steps: int = DEFAULT_EPISODE_STEPS,
        ship_speed: float = DEFAULT_SHIP_SPEED,
        comet_speed: float = DEFAULT_COMET_SPEED,
    ) -> "CppFastOrbitSim":
        from orbit_wars_bot.fast_sim import FastOrbitSim

        py_sim = FastOrbitSim.from_seed(
            seed=seed,
            num_agents=num_agents,
            max_steps=max_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            use_cpp=False,
        )
        return cls.from_fast_sim(py_sim)

    @classmethod
    def from_observation(
        cls,
        obs: Any,
        config: Any = None,
        *,
        num_agents: Optional[int] = None,
        episode_seed: Optional[int] = None,
    ) -> "CppFastOrbitSim":
        planets = [list(p) for p in _obs_get(obs, "planets", [])]
        raw_init = _obs_get(obs, "initial_planets", None)
        initial_planets = [list(p) for p in raw_init] if raw_init else [p[:] for p in planets]
        fleets = [list(f) for f in _obs_get(obs, "fleets", [])]
        raw_comets = _obs_get(obs, "comets", []) or []
        comets = [
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": g["paths"],
                "path_index": int(g["path_index"]),
            }
            for g in raw_comets
        ]
        comet_planet_ids = [int(pid) for pid in (_obs_get(obs, "comet_planet_ids", []) or [])]
        angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0))
        next_fleet_id = int(_obs_get(obs, "next_fleet_id", len(fleets)))
        step_count = int(_obs_get(obs, "step", 0))

        if num_agents is None:
            max_pid = int(_obs_get(obs, "player", 0))
            for p in planets:
                if int(p[1]) > max_pid:
                    max_pid = int(p[1])
            for f in fleets:
                if int(f[1]) > max_pid:
                    max_pid = int(f[1])
            num_agents = 4 if max_pid >= 2 else 2

        max_steps = int(_obs_get(config, "episodeSteps", DEFAULT_EPISODE_STEPS)) if config is not None else DEFAULT_EPISODE_STEPS
        ship_speed = float(_obs_get(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config is not None else DEFAULT_SHIP_SPEED
        comet_speed = float(_obs_get(config, "cometSpeed", DEFAULT_COMET_SPEED)) if config is not None else DEFAULT_COMET_SPEED

        return cls.from_parts(
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
        )

    def clone(self) -> "CppFastOrbitSim":
        assert _LIB is not None
        h = _LIB.orbit_sim_clone(self._handle)
        # Share immutable initial_planets & comet_specs until a comet spawns/expires cleanly
        return CppFastOrbitSim(
            h,
            initial_planets=self.initial_planets,
            comet_planet_ids=self.comet_planet_ids[:] if self.comet_planet_ids else [],
            comet_specs=[
                {"planet_ids": g["planet_ids"][:], "paths": g["paths"], "path_index": g["path_index"]}
                for g in self._comet_specs
            ]
            if self._comet_specs
            else [],
            angular_velocity=self.angular_velocity,
            ship_speed=self.ship_speed,
            comet_speed=self.comet_speed,
            episode_seed=self.episode_seed,
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            step_count=self.step_count,
            next_fleet_id=self.next_fleet_id,
            done=self.done,
            rewards=self.rewards[:],
            n_planets=self._n_planets,
            n_fleets=self._n_fleets,
            initial_by_id=self._initial_by_id,
            orbit_meta=self._orbit_meta,
        )

    def copy_into(self, dst: "CppFastOrbitSim") -> None:
        assert _LIB is not None
        _LIB.orbit_sim_copy_into(self._handle, dst._handle)
        dst.initial_planets = self.initial_planets
        dst.comet_planet_ids = self.comet_planet_ids[:] if self.comet_planet_ids else []
        dst._comet_specs = [
            {"planet_ids": g["planet_ids"][:], "paths": g["paths"], "path_index": g["path_index"]}
            for g in self._comet_specs
        ] if self._comet_specs else []
        dst.angular_velocity = self.angular_velocity
        dst.ship_speed = self.ship_speed
        dst.comet_speed = self.comet_speed
        dst.episode_seed = self.episode_seed
        dst.num_agents = self.num_agents
        dst.max_steps = self.max_steps
        dst.step_count = self.step_count
        dst.next_fleet_id = self.next_fleet_id
        dst.done = self.done
        dst.rewards = self.rewards[:]
        dst._n_planets = self._n_planets
        dst._n_fleets = self._n_fleets
        dst._initial_by_id = self._initial_by_id
        dst._orbit_meta = self._orbit_meta

    def _sync_comet_indices_if_any(self) -> None:
        if not self._comet_specs:
            return
        assert _LIB is not None
        self._sync_meta()
        meta_c_cnt = int(_TLS.meta[7])
        if meta_c_cnt == 0:
            if self.comet_planet_ids:
                expired = {pid for g in self._comet_specs for pid in g["planet_ids"]}
                self.comet_planet_ids = []
                self._comet_specs = []
                self.initial_planets = [p for p in self.initial_planets if int(p[0]) not in expired]
                self._initial_by_id = {int(p[0]): p for p in self.initial_planets}
            return
        idxs = np.zeros((meta_c_cnt,), dtype=np.int32)
        _LIB.orbit_sim_export_comet_indices(self._handle, idxs.ctypes.data)
        if meta_c_cnt == len(self._comet_specs):
            for i, g in enumerate(self._comet_specs):
                g["path_index"] = int(idxs[i])
        else:
            # Prune expired planet_ids using length vs path_index
            new_specs = []
            expired_pids = set()
            kept_cids = []
            cur_idx = 0
            for g in self._comet_specs:
                pi_val = int(g["path_index"])
                plen = len(g["paths"][0]) if g["paths"] else 0
                if cur_idx < meta_c_cnt and pi_val + 1 <= plen:
                    g2 = {
                        "planet_ids": g["planet_ids"][:],
                        "paths": g["paths"],
                        "path_index": int(idxs[cur_idx]),
                    }
                    new_specs.append(g2)
                    kept_cids.extend(g2["planet_ids"])
                    cur_idx += 1
                else:
                    expired_pids.update(g["planet_ids"])
            self._comet_specs = new_specs
            self.comet_planet_ids = kept_cids
            if expired_pids:
                self.initial_planets = [p for p in self.initial_planets if int(p[0]) not in expired_pids]
                self._initial_by_id = {int(p[0]): p for p in self.initial_planets}

    def _maybe_handle_comets_before_step(self) -> None:
        if self.done:
            return
        assert _LIB is not None
        # Purge pre-expired comets in Python metadata if any group reached path end
        if self._comet_specs:
            self._sync_comet_indices_if_any()
        spawn_step = self.step_count + 1
        if spawn_step not in COMET_SPAWN_STEPS or self.episode_seed is None:
            return
        from orbit_wars_bot.fast_sim import generate_comet_paths

        comet_rng = random.Random(f"orbit_wars-comet-{self.episode_seed}-{spawn_step}")
        comet_paths = generate_comet_paths(
            self.initial_planets,
            self.angular_velocity,
            spawn_step,
            self.comet_planet_ids,
            self.comet_speed,
            rng=comet_rng,
        )
        if not comet_paths:
            return
        self._sync_meta()
        p_ids = np.zeros((self._n_planets,), dtype=np.int32)
        p_own = np.zeros((self._n_planets,), dtype=np.int32)
        p_x = np.zeros((self._n_planets,), dtype=np.float64)
        p_y = np.zeros((self._n_planets,), dtype=np.float64)
        p_r = np.zeros((self._n_planets,), dtype=np.float64)
        p_sh = np.zeros((self._n_planets,), dtype=np.int32)
        p_pr = np.zeros((self._n_planets,), dtype=np.int32)
        _LIB.orbit_sim_export_planets(
            self._handle,
            p_ids.ctypes.data, p_own.ctypes.data, p_x.ctypes.data, p_y.ctypes.data,
            p_r.ctypes.data, p_sh.ctypes.data, p_pr.ctypes.data, 0,
        )
        next_id = int(p_ids.max()) + 1 if self._n_planets > 0 else 0
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        m = len(comet_paths)
        plen = len(comet_paths[0])
        new_pids = [next_id + i for i in range(m)]
        self.initial_planets = [p[:] for p in self.initial_planets]
        self._initial_by_id = dict(self._initial_by_id)
        self._orbit_meta = dict(self._orbit_meta)
        for pid in new_pids:
            self.comet_planet_ids.append(pid)
            planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
            self.initial_planets.append(planet[:])
            self._initial_by_id[pid] = planet[:]
            self._orbit_meta[pid] = None
        self._comet_specs.append({"planet_ids": new_pids, "paths": comet_paths, "path_index": -1})

        pids_arr = np.asarray(new_pids, dtype=np.int32)
        px_arr = np.asarray([[float(pt[0]) for pt in p_row] for p_row in comet_paths], dtype=np.float64)
        py_arr = np.asarray([[float(pt[1]) for pt in p_row] for p_row in comet_paths], dtype=np.float64)
        _LIB.orbit_sim_add_comet_group(
            self._handle, -1, m, plen,
            pids_arr.ctypes.data, px_arr.ctypes.data, py_arr.ctypes.data,
            1, int(comet_ships), COMET_RADIUS, COMET_PRODUCTION,
        )
        self._n_planets += m

    def step(self, actions: Sequence[Optional[Sequence[Sequence[Any]]]]) -> bool:
        assert _LIB is not None
        if self.done:
            return True
        if self.episode_seed is not None and ((self.step_count + 1) in COMET_SPAWN_STEPS or self._comet_specs):
            self._maybe_handle_comets_before_step()

        total_m = 0
        for p_id in range(min(self.num_agents, len(actions))):
            act = actions[p_id]
            if act:
                total_m += len(act)
        if total_m > 0:
            buf = _TLS.ensure_moves(total_m)
            idx = 0
            for p_id in range(min(self.num_agents, len(actions))):
                act = actions[p_id]
                if not act:
                    continue
                for mv in act:
                    if len(mv) != 3:
                        continue
                    buf[idx].player_id = p_id
                    buf[idx].from_id = int(mv[0])
                    buf[idx].angle = float(mv[1])
                    buf[idx].ships = int(mv[2])
                    idx += 1
            _LIB.orbit_sim_step(self._handle, ctypes.addressof(buf), idx)
        else:
            _LIB.orbit_sim_step(self._handle, 0, 0)

        if self._comet_specs:
            self._sync_comet_indices_if_any()
        else:
            self._sync_meta()
        return self.done

    def step_n(self, n_ticks: int) -> bool:
        assert _LIB is not None
        rem = int(n_ticks)
        if rem <= 0 or self.done:
            return self.done

        if self.episode_seed is None and not self._comet_specs:
            _LIB.orbit_sim_step_n(self._handle, rem)
            self._sync_meta()
            return self.done

        while rem > 0 and not self.done:
            if (self.step_count + 1) in COMET_SPAWN_STEPS or self._comet_specs:
                self._maybe_handle_comets_before_step()
            # Run C++ chunk up to just before the next comet spawn step (or end)
            chunk = rem
            if self.episode_seed is not None:
                cur = self.step_count
                for sp in COMET_SPAWN_STEPS:
                    if sp - 1 > cur:
                        chunk = min(chunk, (sp - 1) - cur)
                        break
                    if sp - 1 == cur:
                        chunk = 1
                        break
            if self._comet_specs:
                # Step 1 tick when active comets could expire, or run C++ step_n which handles expiry internally!
                _LIB.orbit_sim_step_n(self._handle, chunk)
                self._sync_comet_indices_if_any()
            else:
                _LIB.orbit_sim_step_n(self._handle, chunk)
                self._sync_meta()
            rem -= chunk
        return self.done

    @staticmethod
    def step_batch(
        sims: Sequence["CppFastOrbitSim"],
        actions_per_sim: Sequence[Sequence[Optional[Sequence[Sequence[Any]]]]],
        *,
        n_empty_after: int = 0,
    ) -> None:
        assert _LIB is not None
        k = len(sims)
        if k == 0:
            return

        # Handle exact comet spawn ticks cleanly in Python when episode_seed is present
        has_seed_spawn_soon = False
        for s in sims:
            if s.done:
                continue
            if s.episode_seed is not None:
                end_step = s.step_count + 1 + n_empty_after
                for sp in COMET_SPAWN_STEPS:
                    if s.step_count + 1 <= sp <= end_step:
                        has_seed_spawn_soon = True
                        break
            if has_seed_spawn_soon:
                break

        if has_seed_spawn_soon:
            for s, acts in zip(sims, actions_per_sim):
                if not s.done:
                    s.step(acts)
                    if n_empty_after > 0 and not s.done:
                        s.step_n(n_empty_after)
            return

        for s in sims:
            if s._comet_specs and not s.done:
                s._sync_comet_indices_if_any()

        total_m = 0
        for s, acts in zip(sims, actions_per_sim):
            if s.done:
                continue
            for p_id in range(min(s.num_agents, len(acts))):
                if acts[p_id]:
                    total_m += len(acts[p_id])  # type: ignore[arg-type]

        buf = _TLS.ensure_moves(max(1, total_m))
        offsets = np.zeros((k + 1,), dtype=np.int32)
        ptrs = np.zeros((k,), dtype=np.uintp)
        m_idx = 0
        for bi, (s, acts) in enumerate(zip(sims, actions_per_sim)):
            ptrs[bi] = s._handle
            offsets[bi] = m_idx
            if not s.done:
                for p_id in range(min(s.num_agents, len(acts))):
                    act = acts[p_id]
                    if not act:
                        continue
                    for mv in act:
                        if len(mv) != 3:
                            continue
                        buf[m_idx].player_id = p_id
                        buf[m_idx].from_id = int(mv[0])
                        buf[m_idx].angle = float(mv[1])
                        buf[m_idx].ships = int(mv[2])
                        m_idx += 1
        offsets[k] = m_idx

        _LIB.orbit_sim_step_batch(
            ptrs.ctypes.data,
            k,
            ctypes.addressof(buf) if m_idx > 0 else 0,
            offsets.ctypes.data,
            int(n_empty_after),
        )
        for s in sims:
            if s._comet_specs:
                s._sync_comet_indices_if_any()
            else:
                s._sync_meta()

    def scores(self) -> List[int]:
        assert _LIB is not None
        sc = _TLS.scores
        _LIB.orbit_sim_get_scores(self._handle, sc.ctypes.data, 0)
        return [int(sc[i]) for i in range(self.num_agents)]

    def winner(self) -> int:
        assert _LIB is not None
        return int(_LIB.orbit_sim_winner(self._handle))

    def export_planets(self) -> List[List[Any]]:
        assert _LIB is not None
        self._sync_meta()
        n = self._n_planets
        if n <= 0:
            return []
        p_id = np.empty((n,), dtype=np.int32)
        p_own = np.empty((n,), dtype=np.int32)
        p_x = np.empty((n,), dtype=np.float64)
        p_y = np.empty((n,), dtype=np.float64)
        p_r = np.empty((n,), dtype=np.float64)
        p_sh = np.empty((n,), dtype=np.int32)
        p_pr = np.empty((n,), dtype=np.int32)
        _LIB.orbit_sim_export_planets(
            self._handle,
            p_id.ctypes.data, p_own.ctypes.data, p_x.ctypes.data, p_y.ctypes.data,
            p_r.ctypes.data, p_sh.ctypes.data, p_pr.ctypes.data, 0,
        )
        return [
            [int(p_id[i]), int(p_own[i]), float(p_x[i]), float(p_y[i]), float(p_r[i]), int(p_sh[i]), int(p_pr[i])]
            for i in range(n)
        ]

    def export_fleets(self) -> List[List[Any]]:
        assert _LIB is not None
        self._sync_meta()
        nf = self._n_fleets
        if nf <= 0:
            return []
        f_id = np.empty((nf,), dtype=np.int32)
        f_own = np.empty((nf,), dtype=np.int32)
        f_x = np.empty((nf,), dtype=np.float64)
        f_y = np.empty((nf,), dtype=np.float64)
        f_ang = np.empty((nf,), dtype=np.float64)
        f_fr = np.empty((nf,), dtype=np.int32)
        f_sh = np.empty((nf,), dtype=np.int32)
        _LIB.orbit_sim_export_fleets(
            self._handle,
            f_id.ctypes.data, f_own.ctypes.data, f_x.ctypes.data, f_y.ctypes.data,
            f_ang.ctypes.data, f_fr.ctypes.data, f_sh.ctypes.data,
        )
        return [
            [int(f_id[i]), int(f_own[i]), float(f_x[i]), float(f_y[i]), float(f_ang[i]), int(f_fr[i]), int(f_sh[i])]
            for i in range(nf)
        ]

    def to_fast_sim(self):
        from orbit_wars_bot.fast_sim import FastOrbitSim

        if self._comet_specs:
            self._sync_comet_indices_if_any()
        planets = self.export_planets()
        fleets = self.export_fleets()
        comets = [
            {"planet_ids": g["planet_ids"][:], "paths": g["paths"], "path_index": int(g["path_index"])}
            for g in self._comet_specs
        ]
        return FastOrbitSim(
            planets=planets,
            initial_planets=[p[:] for p in self.initial_planets],
            fleets=fleets,
            comets=comets,
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
        )

    def to_observation(self, player: int = 0) -> Dict[str, Any]:
        if self._comet_specs:
            self._sync_comet_indices_if_any()
        return {
            "step": self.step_count,
            "player": player,
            "planets": self.export_planets(),
            "initial_planets": [p[:] for p in self.initial_planets],
            "fleets": self.export_fleets(),
            "next_fleet_id": self.next_fleet_id,
            "angular_velocity": self.angular_velocity,
            "comets": [
                {"planet_ids": g["planet_ids"][:], "paths": g["paths"], "path_index": int(g["path_index"])}
                for g in self._comet_specs
            ],
            "comet_planet_ids": self.comet_planet_ids[:],
            "remainingOverageTime": 60.0,
        }

    def build_shared_physics(
        self,
        *,
        max_horizon: int = TIMELINE_HORIZON_CPP,
        max_route_eta: int = MAX_ROUTE_ETA_CPP,
        include_py_tracks: bool = False,
    ) -> Dict[str, Any]:
        """Build exact ``build_shared_planet_physics`` dict directly from C++ state."""
        assert _LIB is not None
        if self._comet_specs:
            self._sync_comet_indices_if_any()
        else:
            self._sync_meta()

        n = min(self._n_planets, MAX_PLANETS_CPP)
        h1 = max_horizon + 1
        nf = self._n_fleets

        planets = self.export_planets()[:n]
        fleets = self.export_fleets()

        tr_x = np.empty((n, h1), dtype=np.float64)
        tr_y = np.empty((n, h1), dtype=np.float64)
        tr_valid_u8 = np.empty((n, h1), dtype=np.uint8)
        fut_owner = np.empty((n, h1), dtype=np.int32)
        fut_ships = np.empty((n, h1), dtype=np.int32)
        valid_route_u8 = np.empty((n, n), dtype=np.uint8)
        best_eta = np.empty((n, n), dtype=np.int32)
        best_dist = np.empty((n, n), dtype=np.float64)

        max_arr = max(64, nf)
        arr_cnt = _TLS.arr_cnt
        arr_fid = np.empty((max_arr,), dtype=np.int32)
        arr_own = np.empty((max_arr,), dtype=np.int32)
        arr_sh = np.empty((max_arr,), dtype=np.int32)
        arr_pidx = np.empty((max_arr,), dtype=np.int32)
        arr_eta = np.empty((max_arr,), dtype=np.int32)

        _LIB.orbit_sim_build_shared_physics(
            self._handle,
            MAX_PLANETS_CPP,
            max_horizon,
            max_route_eta,
            tr_x.ctypes.data,
            tr_y.ctypes.data,
            tr_valid_u8.ctypes.data,
            fut_owner.ctypes.data,
            fut_ships.ctypes.data,
            valid_route_u8.ctypes.data,
            best_eta.ctypes.data,
            best_dist.ctypes.data,
            max_arr,
            arr_cnt.ctypes.data,
            arr_fid.ctypes.data,
            arr_own.ctypes.data,
            arr_sh.ctypes.data,
            arr_pidx.ctypes.data,
            arr_eta.ctypes.data,
        )

        return _assemble_shared_physics_dict(
            step=self.step_count,
            ang_vel=self.angular_velocity,
            planets=planets,
            initial_planets=self.initial_planets,
            fleets=fleets,
            comets=self._comet_specs,
            comet_ids=set(self.comet_planet_ids),
            tr_x=tr_x,
            tr_y=tr_y,
            tr_valid=tr_valid_u8.view(bool),
            fut_owner=fut_owner,
            fut_ships=fut_ships,
            valid_route=valid_route_u8.view(bool),
            best_eta=best_eta,
            best_dist=best_dist,
            n_arr=int(arr_cnt[0]),
            arr_fid=arr_fid,
            arr_own=arr_own,
            arr_sh=arr_sh,
            arr_pidx=arr_pidx,
            arr_eta=arr_eta,
            max_horizon=max_horizon,
            include_py_tracks=include_py_tracks,
        )


def _assemble_shared_physics_dict(
    *,
    step: int,
    ang_vel: float,
    planets: List[List[Any]],
    initial_planets: Sequence[Sequence[Any]],
    fleets: List[List[Any]],
    comets: Sequence[Dict[str, Any]],
    comet_ids: set[int],
    tr_x: np.ndarray,
    tr_y: np.ndarray,
    tr_valid: np.ndarray,
    fut_owner: np.ndarray,
    fut_ships: np.ndarray,
    valid_route: np.ndarray,
    best_eta: np.ndarray,
    best_dist: np.ndarray,
    n_arr: int,
    arr_fid: np.ndarray,
    arr_own: np.ndarray,
    arr_sh: np.ndarray,
    arr_pidx: np.ndarray,
    arr_eta: np.ndarray,
    max_horizon: int,
    include_py_tracks: bool,
) -> Dict[str, Any]:
    n = len(planets)
    planet_ids = [int(p[0]) for p in planets]
    id_to_idx = {pid: idx for idx, pid in enumerate(planet_ids)}
    comet_lookup = build_comet_lookup(comets)
    initial_by_id = {int(p[0]): p for p in initial_planets}

    arrivals: Dict[int, List[FleetArrival]] = {pid: [] for pid in planet_ids}
    for k in range(n_arr):
        pj = int(arr_pidx[k])
        pid = planet_ids[pj]
        arrivals[pid].append(
            FleetArrival(
                fleet_id=int(arr_fid[k]),
                owner=int(arr_own[k]),
                ships=int(arr_sh[k]),
                planet_id=pid,
                eta=int(arr_eta[k]),
            )
        )
    for pid in arrivals:
        if len(arrivals[pid]) > 1:
            arrivals[pid].sort(key=lambda a: a.eta)

    tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    futures: List[Tuple[Tuple[int, int], ...]] = []
    if include_py_tracks and n > 0:
        tr_x_list = tr_x.tolist()
        tr_y_list = tr_y.tolist()
        for i, pid in enumerate(planet_ids):
            xr = tr_x_list[i]
            yr = tr_y_list[i]
            tracks[pid] = [None if math.isnan(xv) else (xv, yv) for xv, yv in zip(xr, yr)]
        fo_list = fut_owner.tolist()
        fs_list = fut_ships.tolist()
        for i in range(n):
            futures.append(tuple(zip(fo_list[i], fs_list[i])))
    elif n > 0:
        fo_list = fut_owner.tolist()
        fs_list = fut_ships.tolist()
        for i in range(n):
            futures.append(tuple(zip(fo_list[i], fs_list[i])))

    px = np.asarray([float(p[2]) for p in planets], dtype=np.float64)
    py = np.asarray([float(p[3]) for p in planets], dtype=np.float64)
    pr = np.asarray([float(p[4]) for p in planets], dtype=np.float64)
    pships = np.asarray([int(p[5]) for p in planets], dtype=np.int32)
    pprod = np.asarray([int(p[6]) for p in planets], dtype=np.int32)
    pown = np.asarray([int(p[1]) for p in planets], dtype=np.int32)
    is_comet_arr = np.asarray([(pid in comet_ids) or (pid in comet_lookup) for pid in planet_ids], dtype=bool)

    init_owners = {int(p[1]) for p in initial_planets if int(p[1]) >= 0}
    cur_owners = {int(p[1]) for p in planets if int(p[1]) >= 0} | {int(f[1]) for f in fleets if int(f[1]) >= 0}
    is_4p = 1.0 if (len(init_owners | cur_owners) > 2 or max((init_owners | cur_owners), default=0) >= 2) else 0.0

    del max_horizon
    return {
        "step": step,
        "ang_vel": ang_vel,
        "planets": planets,
        "fleets": fleets,
        "initial_by_id": initial_by_id,
        "comet_ids": comet_ids,
        "comet_lookup": comet_lookup,
        "n": n,
        "planet_ids": planet_ids,
        "id_to_idx": id_to_idx,
        "tr_x": tr_x,
        "tr_y": tr_y,
        "tr_valid": tr_valid,
        "tracks": tracks,
        "arrivals": arrivals,
        "futures": futures,
        "fut_owner": fut_owner,
        "fut_ships": fut_ships,
        "px": px,
        "py": py,
        "pr": pr,
        "pships": pships,
        "pprod": pprod,
        "pown": pown,
        "is_comet_arr": is_comet_arr,
        "valid_route": valid_route,
        "best_eta": best_eta,
        "best_dist": best_dist,
        "is_4p": is_4p,
    }


def build_shared_planet_physics_cpp(
    obs: Any,
    max_speed: float = DEFAULT_SHIP_SPEED,
    *,
    max_horizon: int = TIMELINE_HORIZON_CPP,
    max_route_eta: int = MAX_ROUTE_ETA_CPP,
    include_py_tracks: bool = True,
) -> Dict[str, Any]:
    """C++-accelerated drop-in replacement for ``planet_transformer.build_shared_planet_physics``."""
    if _LIB is None:
        from orbit_wars_bot.planet_transformer import _build_shared_planet_physics_python

        return _build_shared_planet_physics_python(obs, max_speed=max_speed)

    step = int(_obs_get(obs, "step", 0))
    ang_vel = float(_obs_get(obs, "angular_velocity", 0.0))
    planets = [list(p) for p in (_obs_get(obs, "planets", []) or [])][:MAX_PLANETS_CPP]
    fleets = [list(f) for f in (_obs_get(obs, "fleets", []) or [])]
    init_planets = _obs_get(obs, "initial_planets", []) or planets
    comets = _obs_get(obs, "comets", []) or []
    comet_ids = {int(c) for c in (_obs_get(obs, "comet_planet_ids", []) or [])}
    comet_lookup = build_comet_lookup(comets)
    initial_by_id = {int(p[0]): p for p in init_planets}

    n = len(planets)
    nf = len(fleets)
    h1 = max_horizon + 1

    if n == 0:
        return _assemble_shared_physics_dict(
            step=step, ang_vel=ang_vel, planets=planets, initial_planets=init_planets,
            fleets=fleets, comets=comets, comet_ids=comet_ids,
            tr_x=np.zeros((0, h1), dtype=np.float64),
            tr_y=np.zeros((0, h1), dtype=np.float64),
            tr_valid=np.zeros((0, h1), dtype=bool),
            fut_owner=np.zeros((0, h1), dtype=np.int32),
            fut_ships=np.zeros((0, h1), dtype=np.int32),
            valid_route=np.zeros((0, 0), dtype=bool),
            best_eta=np.zeros((0, 0), dtype=np.int32),
            best_dist=np.zeros((0, 0), dtype=np.float64),
            n_arr=0,
            arr_fid=np.zeros((0,), dtype=np.int32),
            arr_own=np.zeros((0,), dtype=np.int32),
            arr_sh=np.zeros((0,), dtype=np.int32),
            arr_pidx=np.zeros((0,), dtype=np.int32),
            arr_eta=np.zeros((0,), dtype=np.int32),
            max_horizon=max_horizon, include_py_tracks=include_py_tracks,
        )

    all_comet_ids = comet_ids | set(comet_lookup.keys())
    p_id = np.asarray([int(p[0]) for p in planets], dtype=np.int32)
    p_own = np.asarray([int(p[1]) for p in planets], dtype=np.int32)
    p_x = np.asarray([float(p[2]) for p in planets], dtype=np.float64)
    p_y = np.asarray([float(p[3]) for p in planets], dtype=np.float64)
    p_r = np.asarray([float(p[4]) for p in planets], dtype=np.float64)
    p_sh = np.asarray([int(p[5]) for p in planets], dtype=np.int32)
    p_pr = np.asarray([int(p[6]) for p in planets], dtype=np.int32)
    p_orb, p_or, p_oa, p_ic = _compute_orbit_params(planets, initial_by_id, all_comet_ids)

    comet_tx = np.zeros((n, h1), dtype=np.float64)
    comet_ty = np.zeros((n, h1), dtype=np.float64)
    comet_tval = np.zeros((n, h1), dtype=np.uint8)
    if comet_lookup:
        for i in range(n):
            if p_ic[i]:
                info = comet_lookup.get(int(p_id[i]))
                if info is not None:
                    path, cur_idx = info
                    plen = len(path)
                    for dt in range(1, h1):
                        fidx = cur_idx + dt
                        if fidx < 0:
                            comet_tval[i, dt] = 2
                        elif fidx >= plen:
                            comet_tval[i, dt] = 0
                        else:
                            comet_tx[i, dt] = float(path[fidx][0])
                            comet_ty[i, dt] = float(path[fidx][1])
                            comet_tval[i, dt] = 1
                else:
                    comet_tx[i, 1:] = p_x[i]
                    comet_ty[i, 1:] = p_y[i]
                    comet_tval[i, 1:] = 1 if p_x[i] >= 0.0 else 2

    f_id = np.asarray([int(f[0]) for f in fleets], dtype=np.int32) if nf else np.zeros((0,), dtype=np.int32)
    f_own = np.asarray([int(f[1]) for f in fleets], dtype=np.int32) if nf else np.zeros((0,), dtype=np.int32)
    f_x = np.asarray([float(f[2]) for f in fleets], dtype=np.float64) if nf else np.zeros((0,), dtype=np.float64)
    f_y = np.asarray([float(f[3]) for f in fleets], dtype=np.float64) if nf else np.zeros((0,), dtype=np.float64)
    f_ang = np.asarray([float(f[4]) for f in fleets], dtype=np.float64) if nf else np.zeros((0,), dtype=np.float64)
    f_sh = np.asarray([int(f[6]) for f in fleets], dtype=np.int32) if nf else np.zeros((0,), dtype=np.int32)

    tr_x = np.empty((n, h1), dtype=np.float64)
    tr_y = np.empty((n, h1), dtype=np.float64)
    tr_valid_u8 = np.empty((n, h1), dtype=np.uint8)
    fut_owner = np.empty((n, h1), dtype=np.int32)
    fut_ships = np.empty((n, h1), dtype=np.int32)
    valid_route_u8 = np.empty((n, n), dtype=np.uint8)
    best_eta = np.empty((n, n), dtype=np.int32)
    best_dist = np.empty((n, n), dtype=np.float64)

    max_arr = max(64, nf)
    arr_cnt = _TLS.arr_cnt
    arr_fid = np.empty((max_arr,), dtype=np.int32)
    arr_own = np.empty((max_arr,), dtype=np.int32)
    arr_sh = np.empty((max_arr,), dtype=np.int32)
    arr_pidx = np.empty((max_arr,), dtype=np.int32)
    arr_eta = np.empty((max_arr,), dtype=np.int32)

    _LIB.orbit_compute_shared_physics_arrays(
        n,
        p_id.ctypes.data, p_own.ctypes.data, p_x.ctypes.data, p_y.ctypes.data, p_r.ctypes.data,
        p_sh.ctypes.data, p_pr.ctypes.data, p_orb.ctypes.data, p_or.ctypes.data, p_oa.ctypes.data, p_ic.ctypes.data,
        comet_tx.ctypes.data, comet_ty.ctypes.data, comet_tval.ctypes.data,
        nf,
        f_id.ctypes.data, f_own.ctypes.data, f_x.ctypes.data, f_y.ctypes.data, f_ang.ctypes.data, f_sh.ctypes.data,
        step, float(ang_vel), float(max_speed), max_horizon, max_route_eta,
        tr_x.ctypes.data, tr_y.ctypes.data, tr_valid_u8.ctypes.data,
        fut_owner.ctypes.data, fut_ships.ctypes.data,
        valid_route_u8.ctypes.data, best_eta.ctypes.data, best_dist.ctypes.data,
        max_arr, arr_cnt.ctypes.data,
        arr_fid.ctypes.data, arr_own.ctypes.data, arr_sh.ctypes.data, arr_pidx.ctypes.data, arr_eta.ctypes.data,
    )

    return _assemble_shared_physics_dict(
        step=step,
        ang_vel=ang_vel,
        planets=planets,
        initial_planets=init_planets,
        fleets=fleets,
        comets=comets,
        comet_ids=comet_ids,
        tr_x=tr_x,
        tr_y=tr_y,
        tr_valid=tr_valid_u8.view(bool),
        fut_owner=fut_owner,
        fut_ships=fut_ships,
        valid_route=valid_route_u8.view(bool),
        best_eta=best_eta,
        best_dist=best_dist,
        n_arr=int(arr_cnt[0]),
        arr_fid=arr_fid,
        arr_own=arr_own,
        arr_sh=arr_sh,
        arr_pidx=arr_pidx,
        arr_eta=arr_eta,
        max_horizon=max_horizon,
        include_py_tracks=include_py_tracks,
    )


def solve_intercept_cpp(
    from_idx: int,
    target_idx: int,
    ships: int,
    tr_x: np.ndarray,
    tr_y: np.ndarray,
    p_r: np.ndarray,
    *,
    max_eta: int = 32,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Optional[InterceptSolution]:
    """C++-accelerated exact ``solve_intercept`` over dense `(N, H+1)` track arrays."""
    if _LIB is None:
        return None
    n_planets, track_stride = int(tr_x.shape[0]), int(tr_x.shape[1])
    if n_planets == 0 or track_stride < 2:
        return None
    tx_c = np.ascontiguousarray(tr_x, dtype=np.float64)
    ty_c = np.ascontiguousarray(tr_y, dtype=np.float64)
    pr_c = np.ascontiguousarray(p_r, dtype=np.float64)
    tls = _TLS
    ok = _LIB.orbit_solve_intercept(
        int(from_idx),
        int(target_idx),
        int(ships),
        n_planets,
        track_stride,
        tx_c.ctypes.data,
        ty_c.ctypes.data,
        pr_c.ctypes.data,
        int(max_eta),
        float(max_speed),
        ctypes.byref(tls.out_ang),
        ctypes.byref(tls.out_eta),
        ctypes.byref(tls.out_ix),
        ctypes.byref(tls.out_iy),
        ctypes.byref(tls.out_spd),
    )
    if not ok:
        return None
    return InterceptSolution(
        angle=float(tls.out_ang.value),
        eta=int(tls.out_eta.value),
        impact_pos=(float(tls.out_ix.value), float(tls.out_iy.value)),
        speed=float(tls.out_spd.value),
    )


def step_fast_sim_cpp(sim: Any, actions: Sequence[Optional[Sequence[Sequence[Any]]]]) -> bool:
    """Execute one tick of ``FastOrbitSim`` through C++17 (exact parity check helper)."""
    cpp_sim = CppFastOrbitSim.from_fast_sim(sim)
    cpp_sim.step(actions)
    py_out = cpp_sim.to_fast_sim()
    sim.planets = py_out.planets
    sim.initial_planets = py_out.initial_planets
    sim.fleets = py_out.fleets
    sim.comets = py_out.comets
    sim.comet_planet_ids = py_out.comet_planet_ids
    sim.next_fleet_id = py_out.next_fleet_id
    sim.step_count = py_out.step_count
    sim.done = py_out.done
    sim.rewards = py_out.rewards
    sim._orbit_meta = py_out._orbit_meta
    return sim.done


def step_n_fast_sim_cpp(sim: Any, n_ticks: int) -> bool:
    """Execute ``n_ticks`` empty ticks of ``FastOrbitSim`` inside C++17."""
    cpp_sim = CppFastOrbitSim.from_fast_sim(sim)
    cpp_sim.step_n(n_ticks)
    py_out = cpp_sim.to_fast_sim()
    sim.planets = py_out.planets
    sim.initial_planets = py_out.initial_planets
    sim.fleets = py_out.fleets
    sim.comets = py_out.comets
    sim.comet_planet_ids = py_out.comet_planet_ids
    sim.next_fleet_id = py_out.next_fleet_id
    sim.step_count = py_out.step_count
    sim.done = py_out.done
    sim.rewards = py_out.rewards
    sim._orbit_meta = py_out._orbit_meta
    return sim.done
