"""High-fidelity Planet-Timeline Transformer for Orbit Wars (2P & 4P FFA).

Replaces the lossy 16x16 spatial CNN grid and 16 global macro stances with:
  1. Lossless per-planet / per-comet entity tokens (up to ``MAX_PLANETS = 44``).
  2. Exact 20-step projected fleet-arrival timelines per planet encoded via a
     1D-CNN (using ``orbit_wars_bot.physics.predict_fleet_arrivals`` and
     ``simulate_planet_future``).
  3. 100% D4-invariant directed ``(source -> target)`` edge features for all
     short-ETA (``ETA <= 20``) legal orbital routes.
  4. A 4-layer edge-biased Transformer encoder producing:
     - Independent per-planet action logits over ``[HOLD, target_0..N-1]``
     - Per-edge launch-fraction predictions in ``(0, 1]``
     - Global value estimate in ``[-1, +1]`` (2P & 4P compatible)

Runs in ~5-10 ms per forward pass on 1 CPU thread (~25 ms with 4x rotational
test-time augmentation), leaving wide headroom under Kaggle's 500 ms turn cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit_wars_bot.physics import (
    BOARD_SIZE,
    CENTER,
    DEFAULT_SHIP_SPEED,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    build_comet_lookup,
    distance,
    fleet_speed,
    is_orbiting_planet,
    predict_all_planet_tracks,
    predict_all_planet_tracks_arrays,
    predict_fleet_arrivals,
    predict_fleet_arrivals_vectorized,
    segment_hits_sun,
    simulate_planet_future,
    solve_intercept,
)

MAX_PLANETS = 44
TIMELINE_HORIZON = 20
MAX_ROUTE_ETA = 20
PLANET_FEAT_DIM = 22
TIMELINE_FEAT_DIM = 8
TIMELINE_FEAT_DIM_V2 = 14
EDGE_FEAT_DIM = 12


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


@dataclass
class EncodedPlanetState:
    """Lossless structured observation tensors for one player perspective."""

    planet_feats: np.ndarray  # (N, PLANET_FEAT_DIM) float32
    timeline_feats: np.ndarray  # (N, TIMELINE_HORIZON, TIMELINE_FEAT_DIM) float32
    edge_feats: np.ndarray  # (N, N, EDGE_FEAT_DIM) float32
    legal_edges: np.ndarray  # (N, N) bool
    owned_mask: np.ndarray  # (N,) bool
    planet_ids: List[int]
    id_to_idx: Dict[int, int]
    expiring_comet_indices: List[int]
    planet_rows: List[List[float]]
    tracks: Dict[int, List[Optional[Tuple[float, float]]]]
    initial_by_id: Dict[int, Sequence[float]]
    comet_lookup: Dict[int, Tuple[List[List[float]], int]]
    current_step: int
    angular_velocity: float
    player: int
    arrivals: Dict[int, List[Any]] = field(default_factory=dict)
    futures: List[Tuple[Tuple[int, int], ...]] = field(default_factory=list)


def _enemy_slot_map(
    planets: Sequence[Sequence[float]],
    fleets: Sequence[Sequence[float]],
    player: int,
) -> Tuple[Dict[int, int], int, int, int, int]:
    """Rank active enemy player IDs by total ships descending -> slots 0, 1, 2."""
    ships_by_owner: Dict[int, int] = {}
    prod_by_owner: Dict[int, int] = {}
    for p in planets:
        o = int(p[1])
        if o >= 0:
            ships_by_owner[o] = ships_by_owner.get(o, 0) + int(p[5])
            prod_by_owner[o] = prod_by_owner.get(o, 0) + int(p[6])
    for fl in fleets:
        o = int(fl[1])
        if o >= 0:
            ships_by_owner[o] = ships_by_owner.get(o, 0) + int(fl[6])

    enemies = sorted(
        (o for o in ships_by_owner if o != player and ships_by_owner[o] > 0),
        key=lambda o: (-ships_by_owner[o], o),
    )
    emap = {o: idx for idx, o in enumerate(enemies[:3])}
    my_ships = ships_by_owner.get(player, 0)
    tot_ships = sum(ships_by_owner.values())
    my_prod = prod_by_owner.get(player, 0)
    tot_prod = sum(prod_by_owner.values())
    return emap, my_ships, tot_ships, my_prod, tot_prod


def build_shared_planet_physics(
    obs: Any,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Dict[str, Any]:
    """Compute player-agnostic orbital tracks, fleet arrivals, futures, and route geometry once per tick.

    Can be shared across all players on the same simulation tick in multi-player self-play
    rollouts for a ~6.5x speedup over scalar per-player extraction with bit-exact parity.
    """
    step = int(_obs_get(obs, "step", 0))
    ang_vel = float(_obs_get(obs, "angular_velocity", 0.0))
    planets = [list(p) for p in (_obs_get(obs, "planets", []) or [])][:MAX_PLANETS]
    fleets = _obs_get(obs, "fleets", []) or []
    init_planets = _obs_get(obs, "initial_planets", []) or planets
    comets = _obs_get(obs, "comets", []) or []
    comet_ids = set(int(c) for c in (_obs_get(obs, "comet_planet_ids", []) or []))
    comet_lookup = build_comet_lookup(comets)
    initial_by_id = {int(p[0]): p for p in init_planets}

    n = len(planets)
    planet_ids = [int(p[0]) for p in planets]
    id_to_idx = {pid: idx for idx, pid in enumerate(planet_ids)}

    tr_x, tr_y, tr_valid, tracks = predict_all_planet_tracks_arrays(
        planets, TIMELINE_HORIZON, step, ang_vel, initial_by_id, comet_lookup
    )
    arrivals = predict_fleet_arrivals_vectorized(
        fleets, planets, tr_x, tr_y, max_horizon=TIMELINE_HORIZON, max_speed=max_speed
    )

    futures: List[Tuple[Tuple[int, int], ...]] = []
    fut_owner = np.empty((n, TIMELINE_HORIZON + 1), dtype=np.int32)
    fut_ships = np.empty((n, TIMELINE_HORIZON + 1), dtype=np.int32)
    for i, p in enumerate(planets):
        fut = simulate_planet_future(p, arrivals.get(planet_ids[i], ()), horizon=TIMELINE_HORIZON)
        tl = fut.timeline
        futures.append(tl)
        for dt in range(TIMELINE_HORIZON + 1):
            fut_owner[i, dt] = tl[dt][0]
            fut_ships[i, dt] = tl[dt][1]

    px = np.asarray([float(p[2]) for p in planets], dtype=np.float64)
    py = np.asarray([float(p[3]) for p in planets], dtype=np.float64)
    pr = np.asarray([float(p[4]) for p in planets], dtype=np.float64)
    pships = np.asarray([int(p[5]) for p in planets], dtype=np.int32)
    pprod = np.asarray([int(p[6]) for p in planets], dtype=np.int32)
    pown = np.asarray([int(p[1]) for p in planets], dtype=np.int32)
    is_comet_arr = np.asarray([(pid in comet_ids) or (pid in comet_lookup) for pid in planet_ids], dtype=bool)

    if n > 0:
        spds = np.asarray([fleet_speed(max(1, int(s)), max_speed) for s in pships], dtype=np.float64)
        dts = np.arange(1, MAX_ROUTE_ETA + 1, dtype=np.float64)
        tx = tr_x[:, 1 : MAX_ROUTE_ETA + 1]
        ty = tr_y[:, 1 : MAX_ROUTE_ETA + 1]
        tval_prefix = np.cumprod(tr_valid[:, 1 : MAX_ROUTE_ETA + 1], axis=1).astype(bool)

        dx = tx[None, :, :] - px[:, None, None]
        dy = ty[None, :, :] - py[:, None, None]
        d2 = dx * dx + dy * dy
        reach = spds[:, None, None] * dts[None, None, :] + (pr[:, None] + 0.1 + pr[None, :])[:, :, None]
        hit = tval_prefix[None, :, :] & (d2 <= reach * reach)
        idx_diag = np.arange(n)
        hit[idx_diag, idx_diag, :] = False

        any_hit = hit.any(axis=2)
        first_k = hit.argmax(axis=2)
        j_idx = np.arange(n)[None, :]
        imp_x = tx[j_idx, first_k]
        imp_y = ty[j_idx, first_k]
        best_dist = np.sqrt(np.where(any_hit, (imp_x - px[:, None]) ** 2 + (imp_y - py[:, None]) ** 2, 0.0))

        sdx = imp_x - px[:, None]
        sdy = imp_y - py[:, None]
        sl2 = sdx * sdx + sdy * sdy
        safe_sl2 = np.where(sl2 > 0.0, sl2, 1.0)
        st = np.clip(((CENTER - px[:, None]) * sdx + (CENTER - py[:, None]) * sdy) / safe_sl2, 0.0, 1.0)
        st = np.where(sl2 > 0.0, st, 0.0)
        spx = px[:, None] + st * sdx
        spy = py[:, None] + st * sdy
        clear_sun = np.hypot(CENTER - spx, CENTER - spy) >= (SUN_RADIUS + 0.25)

        valid_route = any_hit & clear_sun
        best_eta = np.where(valid_route, first_k + 1, 0).astype(np.int32)
    else:
        valid_route = np.zeros((0, 0), dtype=bool)
        best_eta = np.zeros((0, 0), dtype=np.int32)
        best_dist = np.zeros((0, 0), dtype=np.float64)

    init_owners = {int(p[1]) for p in init_planets if int(p[1]) >= 0}
    cur_owners = {int(p[1]) for p in planets if int(p[1]) >= 0} | {int(f[1]) for f in fleets if int(f[1]) >= 0}
    is_4p = 1.0 if (len(init_owners | cur_owners) > 2 or max((init_owners | cur_owners), default=0) >= 2) else 0.0

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


def compute_ships_for_capture(
    shared_physics: Dict[str, Any],
    player: int,
    max_timesteps: int = TIMELINE_HORIZON + 1,
) -> np.ndarray:
    """Compute exact `ships_for_capture(planet, dt, player)` matching Hober's Rust ledger."""
    n = int(shared_physics["n"])
    if n == 0:
        return np.zeros((0, max_timesteps), dtype=np.int32)

    fo_all = shared_physics["fut_owner"]
    fs_all = shared_physics["fut_ships"]
    tv_all = shared_physics["tr_valid"]
    t_lim = min(int(max_timesteps), int(fo_all.shape[1]), int(tv_all.shape[1]))
    fo = fo_all[:, :t_lim]  # (n, T)
    fs = fs_all[:, :t_lim]  # (n, T)
    tv = tv_all[:, :t_lim]  # (n, T)
    pprod = shared_physics["pprod"]  # (n,)
    planet_ids = shared_physics["planet_ids"]
    arrivals = shared_physics["arrivals"]

    # Default formula (exact whenever no simultaneous arrivals occur on that tick)
    cap = np.where((fo == player) | (~tv), 0, np.maximum(fs, 0) + 1).astype(np.int32)

    # Refine ticks that have incoming fleet arrivals (mirrors Hober movegen ledger)
    for i, pid in enumerate(planet_ids):
        arr_list = arrivals.get(pid)
        if not arr_list:
            continue
        by_dt: Dict[int, List[Any]] = {}
        for a in arr_list:
            if 1 <= a.eta < t_lim:
                by_dt.setdefault(int(a.eta), []).append(a)
        for dt, turn_arrs in by_dt.items():
            if not tv[i, dt] or fo[i, dt] == player:
                cap[i, dt] = 0
                continue
            prev_own = int(fo[i, dt - 1]) if tv[i, dt - 1] else int(fo[i, dt])
            prev_sh = int(fs[i, dt - 1]) if tv[i, dt - 1] else int(fs[i, dt])
            pre_sh = prev_sh + (int(pprod[i]) if prev_own != -1 else 0)
            p_arr = 0
            other_arr: Dict[int, int] = {}
            for a in turn_arrs:
                ao = int(a.owner)
                if ao == player:
                    p_arr += int(a.ships)
                else:
                    other_arr[ao] = other_arr.get(ao, 0) + int(a.ships)
            strongest_other = max(other_arr.values(), default=0)
            if prev_own == player:
                needed = strongest_other - max(0, pre_sh) - p_arr
            else:
                needed = strongest_other + max(0, pre_sh) + 1 - p_arr
            cap[i, dt] = max(0, int(needed))
    if t_lim < max_timesteps:
        out = np.zeros((n, max_timesteps), dtype=np.int32)
        out[:, :t_lim] = cap
        return out
    return cap


def extract_planet_features(
    obs: Any,
    player: Optional[int] = None,
    max_speed: float = DEFAULT_SHIP_SPEED,
    feature_version: int = 1,
    shared_physics: Optional[Dict[str, Any]] = None,
) -> EncodedPlanetState:
    """Extract lossless per-planet, 20-step timeline, and directed edge features (vectorized NumPy)."""
    if player is None:
        player = int(_obs_get(obs, "player", 0))
    shared = shared_physics if shared_physics is not None else build_shared_planet_physics(obs, max_speed=max_speed)

    step = shared["step"]
    ang_vel = shared["ang_vel"]
    planets = shared["planets"]
    fleets = shared["fleets"]
    comet_lookup = shared["comet_lookup"]
    n = shared["n"]
    planet_ids = shared["planet_ids"]
    tr_x = shared["tr_x"]
    tr_y = shared["tr_y"]
    tr_valid = shared["tr_valid"]
    arrivals = shared["arrivals"]
    fut_owner = shared["fut_owner"]
    fut_ships = shared["fut_ships"]
    px = shared["px"]
    py = shared["py"]
    pr = shared["pr"]
    pships = shared["pships"]
    pprod = shared["pprod"]
    pown = shared["pown"]
    is_comet_arr = shared["is_comet_arr"]

    emap, my_ships, tot_ships, my_prod, tot_prod = _enemy_slot_map(planets, fleets, player)
    alive_enemies = len(emap)
    my_ship_share = my_ships / max(1.0, float(tot_ships))
    my_prod_share = my_prod / max(1.0, float(tot_prod))
    step_norm = min(1.0, step / 500.0)
    ang_vel_norm = ang_vel / 0.05
    alive_en_norm = alive_enemies / 3.0
    is_4p = shared["is_4p"]

    tl_dim = TIMELINE_FEAT_DIM_V2 if feature_version >= 2 else TIMELINE_FEAT_DIM
    planet_feats = np.zeros((n, PLANET_FEAT_DIM), dtype=np.float32)
    timeline_feats = np.zeros((n, TIMELINE_HORIZON, tl_dim), dtype=np.float32)
    owned_mask = (pown == player) & (pships > 0)
    expiring_comet_indices: List[int] = []
    exact_cap: Optional[np.ndarray] = None

    if n > 0:
        orb_r_raw = np.hypot(px - CENTER, py - CENTER)
        orbiting = (~is_comet_arr) & ((orb_r_raw + pr) < ROTATION_RADIUS_LIMIT)
        has_v = (~np.isnan(tr_x[:, 0])) & (~np.isnan(tr_x[:, 1]))
        vx = np.where(has_v, (tr_x[:, 1] - tr_x[:, 0]) / 6.0, 0.0)
        vy = np.where(has_v, (tr_y[:, 1] - tr_y[:, 0]) / 6.0, 0.0)

        comet_rem = np.zeros(n, dtype=np.float32)
        if np.any(is_comet_arr):
            for i in np.flatnonzero(is_comet_arr):
                pid = planet_ids[i]
                if pid in comet_lookup:
                    cpath, cidx = comet_lookup[pid]
                    rem = max(0, len(cpath) - 1 - cidx)
                    comet_rem[i] = min(1.0, rem / 40.0)
                    if pown[i] == player and pships[i] > 0 and rem <= 1:
                        expiring_comet_indices.append(int(i))

        planet_feats[:, 0] = (px - CENTER) / 50.0
        planet_feats[:, 1] = (py - CENTER) / 50.0
        planet_feats[:, 2] = vx
        planet_feats[:, 3] = vy
        planet_feats[:, 4] = orb_r_raw / 50.0
        planet_feats[:, 5] = orbiting.astype(np.float32)
        planet_feats[:, 6] = is_comet_arr.astype(np.float32)
        planet_feats[:, 7] = comet_rem
        planet_feats[:, 8] = pr / 3.0
        planet_feats[:, 9] = pprod / 5.0
        planet_feats[:, 10] = np.log1p(np.maximum(0, pships)) / 6.0
        planet_feats[:, 11] = (pown == -1).astype(np.float32)
        planet_feats[:, 12] = (pown == player).astype(np.float32)
        for i in range(n):
            ow = int(pown[i])
            if ow >= 0 and ow != player:
                eslot = emap.get(ow, 2)
                planet_feats[i, 13 + min(eslot, 2)] = 1.0
        planet_feats[:, 16] = step_norm
        planet_feats[:, 17] = is_4p
        planet_feats[:, 18] = my_ship_share
        planet_feats[:, 19] = my_prod_share
        planet_feats[:, 20] = ang_vel_norm
        planet_feats[:, 21] = alive_en_norm

        my_in = np.zeros((n, TIMELINE_HORIZON), dtype=np.float32)
        en_in = np.zeros((n, TIMELINE_HORIZON), dtype=np.float32)
        for i, pid in enumerate(planet_ids):
            arr_list = arrivals.get(pid)
            if arr_list:
                for a in arr_list:
                    if 1 <= a.eta <= TIMELINE_HORIZON:
                        if a.owner == player:
                            my_in[i, a.eta - 1] += a.ships
                        elif a.owner >= 0:
                            en_in[i, a.eta - 1] += a.ships

        fo_cur = fut_owner[:, 1:]
        fo_prev = fut_owner[:, :-1]
        fs_cur = fut_ships[:, 1:]
        timeline_feats[:, :, 0] = (fo_cur == -1).astype(np.float32)
        timeline_feats[:, :, 1] = (fo_cur == player).astype(np.float32)
        timeline_feats[:, :, 2] = ((fo_cur >= 0) & (fo_cur != player)).astype(np.float32)
        timeline_feats[:, :, 3] = np.log1p(np.maximum(0, fs_cur)) / 6.0
        timeline_feats[:, :, 4] = np.log1p(my_in) / 6.0
        timeline_feats[:, :, 5] = np.log1p(en_in) / 6.0
        timeline_feats[:, :, 6] = (en_in > 0.0).astype(np.float32)
        timeline_feats[:, :, 7] = (fo_cur != fo_prev).astype(np.float32)
        if feature_version >= 2:
            dts_i = np.arange(1, TIMELINE_HORIZON + 1, dtype=np.int32)
            step_plus = step + dts_i
            tx_h = tr_x[:, 1 : TIMELINE_HORIZON + 1]
            ty_h = tr_y[:, 1 : TIMELINE_HORIZON + 1]
            pos_valid = tr_valid[:, 1 : TIMELINE_HORIZON + 1] & (step_plus[None, :] <= 500)
            safe_x = np.where(pos_valid, tx_h, CENTER)
            safe_y = np.where(pos_valid, ty_h, CENTER)
            timeline_feats[:, :, 8] = np.where(pos_valid, (safe_x - CENTER) / 50.0, 0.0)
            timeline_feats[:, :, 9] = np.where(pos_valid, (safe_y - CENTER) / 50.0, 0.0)
            timeline_feats[:, :, 10] = np.where(pos_valid, np.hypot(safe_x - CENTER, safe_y - CENTER) / 70.0, 0.0)
            timeline_feats[:, :, 11] = pos_valid.astype(np.float32)
            if feature_version >= 3:
                exact_cap = compute_ships_for_capture(shared, player, max_timesteps=TIMELINE_HORIZON + 1)
                cap_need = np.maximum(0, exact_cap[:, 1 : TIMELINE_HORIZON + 1])
            else:
                cap_need = np.where(fo_cur == player, 0, np.maximum(0, fs_cur) + 1)
            timeline_feats[:, :, 12] = np.where(pos_valid, np.log1p(cap_need) / 6.0, 0.0)
            timeline_feats[:, :, 13] = np.minimum(1.0, step_plus / 500.0)[None, :]

    valid_route = shared["valid_route"]
    best_eta = shared["best_eta"]
    best_dist = shared["best_dist"]
    legal_edges = valid_route & owned_mask[:, None]

    edge_feats = np.zeros((n, n, EDGE_FEAT_DIM), dtype=np.float32)
    if n > 0 and np.any(valid_route):
        ii, jj = np.nonzero(valid_route)
        etas = best_eta[ii, jj]
        dists = best_dist[ii, jj]
        po = fut_owner[jj, etas]
        ps = fut_ships[jj, etas]
        t_prod = pprod[jj].astype(np.float32)
        src_s = pships[ii].astype(np.float32)

        is_t_neutral = (po == -1).astype(np.float32)
        is_t_me = (po == player).astype(np.float32)
        is_t_enemy = ((po >= 0) & (po != player)).astype(np.float32)
        ps_f = ps.astype(np.float32)
        ps_pos = np.maximum(0.0, ps_f)

        if feature_version >= 3:
            assert exact_cap is not None
            ec = exact_cap[jj, etas].astype(np.float32)
            need_pos = np.maximum(0.0, ec)
            is_already_mine = (ec <= 0.0).astype(np.float32)
            margin = np.where(is_already_mine > 0.5, src_s, src_s - need_pos + 1.0)
            cap_feasible = np.where((is_already_mine > 0.5) | (src_s >= need_pos), 1.0, 0.0).astype(np.float32)
            cost = 1.0 + np.where(is_already_mine > 0.5, 0.0, np.maximum(0.0, need_pos - 1.0)) + 0.5 * etas.astype(np.float32)
            roi = np.minimum(2.0, t_prod / cost)
            edge_ships_norm = np.log1p(np.where(is_already_mine > 0.5, ps_pos, np.maximum(0.0, need_pos - 1.0))) / 6.0
        else:
            margin = np.where(is_t_me > 0.5, src_s, src_s - ps_f)
            cap_feasible = np.where((is_t_me > 0.5) | (src_s > ps_f), 1.0, 0.0).astype(np.float32)
            cost = 1.0 + np.where(is_t_me > 0.5, 0.0, ps_pos) + 0.5 * etas.astype(np.float32)
            roi = np.minimum(2.0, t_prod / cost)
            edge_ships_norm = np.log1p(ps_pos) / 6.0

        edge_feats[ii, jj, 0] = 1.0
        edge_feats[ii, jj, 1] = etas / float(MAX_ROUTE_ETA)
        edge_feats[ii, jj, 2] = np.minimum(1.5, dists / 100.0)
        edge_feats[ii, jj, 3] = t_prod / 5.0
        edge_feats[ii, jj, 4] = is_t_neutral
        edge_feats[ii, jj, 5] = is_t_me
        edge_feats[ii, jj, 6] = is_t_enemy
        edge_feats[ii, jj, 7] = is_comet_arr[jj].astype(np.float32)
        edge_feats[ii, jj, 8] = edge_ships_norm
        edge_feats[ii, jj, 9] = np.tanh(margin / 25.0)
        edge_feats[ii, jj, 10] = cap_feasible
        edge_feats[ii, jj, 11] = roi

    return EncodedPlanetState(
        planet_feats=planet_feats,
        timeline_feats=timeline_feats,
        edge_feats=edge_feats,
        legal_edges=legal_edges,
        owned_mask=owned_mask,
        planet_ids=planet_ids,
        id_to_idx=shared["id_to_idx"],
        expiring_comet_indices=expiring_comet_indices,
        planet_rows=planets,
        tracks=shared["tracks"],
        initial_by_id=shared["initial_by_id"],
        comet_lookup=comet_lookup,
        current_step=step,
        angular_velocity=ang_vel,
        player=player,
        arrivals=arrivals,
        futures=shared["futures"],
    )


def apply_d4_planet_feats(planet_feats: np.ndarray, k_rot: int = 0, flip_x: bool = False) -> np.ndarray:
    """Apply exact D4 dihedral symmetry to ``(x, y)`` (cols 0-1) and ``(vx, vy)`` (cols 2-3).

    Because all edge features and ``(source, target)`` indices are 100% D4-invariant,
    transforming only these 4 vector columns gives exact 8x data augmentation and 4x TTA.
    """
    out = planet_feats.copy()
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    vx = out[..., 2].copy()
    vy = out[..., 3].copy()

    if flip_x:
        x = -x
        vx = -vx

    k = k_rot % 4
    if k == 1:  # 90 deg CCW: (x, y) -> (-y, x)
        x, y = -y, x
        vx, vy = -vy, vx
    elif k == 2:  # 180 deg
        x, y = -x, -y
        vx, vy = -vx, -vy
    elif k == 3:  # 270 deg CCW: (x, y) -> (y, -x)
        x, y = y, -x
        vx, vy = vy, -vx

    out[..., 0] = x
    out[..., 1] = y
    out[..., 2] = vx
    out[..., 3] = vy
    return out


def apply_d4_timeline_feats(timeline_feats: np.ndarray, k_rot: int = 0, flip_x: bool = False) -> np.ndarray:
    """Apply exact D4 dihedral symmetry to per-step orbital coordinates ``(dx_t, dy_t)`` (cols 8-9)."""
    k = k_rot % 4
    if timeline_feats.shape[-1] < 10 or (k == 0 and not flip_x):
        return timeline_feats.copy()
    out = timeline_feats.copy()
    x = out[..., 8].copy()
    y = out[..., 9].copy()
    if flip_x:
        x = -x
    if k == 1:
        x, y = -y, x
    elif k == 2:
        x, y = -x, -y
    elif k == 3:
        x, y = y, -x
    out[..., 8] = x
    out[..., 9] = y
    return out


def match_action_to_target(
    move: Sequence[float],
    enc: EncodedPlanetState,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> Optional[Tuple[int, int, float]]:
    """Recover exact ``(src_idx, dst_idx, send_fraction)`` for a recorded Kaggle move."""
    from_id = int(move[0])
    ang = float(move[1])
    ships = int(move[2])
    if from_id not in enc.id_to_idx or ships <= 0:
        return None
    src_idx = enc.id_to_idx[from_id]
    p_src = enc.planet_rows[src_idx]
    garr = max(1, int(p_src[5]))
    frac = min(1.0, max(0.05, ships / float(garr)))

    # Spawn a hypothetical fleet and trace its exact hit planet via physics.py
    sx = float(p_src[2]) + math.cos(ang) * (float(p_src[4]) + 0.1)
    sy = float(p_src[3]) + math.sin(ang) * (float(p_src[4]) + 0.1)
    fake_fleet = [[999999, enc.player, sx, sy, ang, 0.0, ships]]
    arrs = predict_fleet_arrivals(
        fake_fleet,
        enc.planet_rows,
        enc.current_step,
        enc.angular_velocity,
        enc.initial_by_id,
        comet_lookup=enc.comet_lookup,
        tracks=enc.tracks,
        max_horizon=TIMELINE_HORIZON,
        max_speed=max_speed,
    )
    best_pid = None
    best_eta = 999
    for pid, lst in arrs.items():
        if pid == from_id:
            continue
        for a in lst:
            if a.eta < best_eta:
                best_eta = a.eta
                best_pid = pid
    if best_pid is None or best_pid not in enc.id_to_idx:
        return None
    return src_idx, enc.id_to_idx[best_pid], frac


class _ResConv1dBlock(nn.Module):
    """Residual 1D convolution + GELU + LayerNorm over planet timeline channels."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = int(kernel_size) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel_size), padding=pad)
        self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.norm = nn.LayerNorm(out_ch)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_ntc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = self.res_proj(x) if self.res_proj is not None else x
        y = F.gelu(self.conv(x)) + res
        is_cpu = y.device.type == "cpu"
        y_t = y.transpose(1, 2)
        if is_cpu:
            y_t = y_t.contiguous()
        y_t = self.norm(y_t)
        if mask_ntc is not None:
            y_t = y_t * mask_ntc
            out = y_t.transpose(1, 2)
            return out.contiguous() if is_cpu else out
        out = y_t.transpose(1, 2)
        if is_cpu:
            out = out.contiguous()
        if mask is not None:
            out = out * mask
        return out


class _EdgeBiasedTransformerLayer(nn.Module):
    """Pre-LN Multi-Head Self-Attention block with directed pairwise edge bias."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dim_ff: int = 256,
        edge_dim: int = EDGE_FEAT_DIM,
        use_geglu: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_geglu = bool(use_geglu)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_bias = nn.Linear(edge_dim, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        if self.use_geglu:
            self.ffn_wi = nn.Linear(d_model, 2 * dim_ff)
            self.ffn_wo = nn.Linear(dim_ff, d_model)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Linear(dim_ff, d_model),
            )

    def forward(self, x: torch.Tensor, edge_feats: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), edge_feats: (B, N, N, E), pad_mask: (B, N) True where valid
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, Dh)
        q = q.transpose(1, 2)  # (B, H, N, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        # Add learned pairwise edge attention bias
        bias = self.edge_bias(edge_feats).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + bias

        # Mask out padded planet slots
        key_mask = (~pad_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        neg_inf = -1e4 if attn.dtype == torch.float16 else -1e9
        attn = attn.masked_fill(key_mask, neg_inf)
        probs = F.softmax(attn, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.out_proj(out)
        if self.use_geglu:
            a, b = self.ffn_wi(self.norm2(x)).chunk(2, dim=-1)
            x = x + self.ffn_wo(F.gelu(a) * b)
        else:
            x = x + self.ffn(self.norm2(x))
        return x


class PlanetTimelineTransformer(nn.Module):
    """Entity-level Planet-Timeline Transformer with per-planet action & global value heads."""

    def __init__(
        self,
        planet_dim: int = PLANET_FEAT_DIM,
        timeline_dim: int = TIMELINE_FEAT_DIM,
        edge_dim: int = EDGE_FEAT_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 256,
        use_aux_future: bool = False,
        feature_version: int = 1,
        arch_version: int = 1,
        target_cond_launch: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.use_aux_future = bool(use_aux_future)
        self.timeline_dim = int(timeline_dim)
        self.feature_version = int(feature_version)
        self.arch_version = int(arch_version)
        self.target_cond_launch = bool(target_cond_launch)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.config = {
            "planet_dim": planet_dim,
            "timeline_dim": timeline_dim,
            "edge_dim": edge_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dim_ff": dim_ff,
        }
        if self.use_aux_future:
            self.config["use_aux_future"] = True
        if self.feature_version > 1:
            self.config["feature_version"] = self.feature_version
        if self.arch_version > 1:
            self.config["arch_version"] = self.arch_version
        if self.target_cond_launch:
            self.config["target_cond_launch"] = True

        # 1D-CNN encoder over the 20-step arrival & ownership timeline per planet
        if self.arch_version >= 3:
            tl_in_ch = timeline_dim + 5  # + [lin_ships, lin_my_in, lin_en_in, lin_cap, prod]
            self.timeline_blocks = nn.ModuleList(
                [
                    _ResConv1dBlock(tl_in_ch, 128, kernel_size=5),
                    _ResConv1dBlock(128, 128, kernel_size=5),
                    _ResConv1dBlock(128, 128, kernel_size=5),
                    _ResConv1dBlock(128, 128, kernel_size=5),
                ]
            )
            tl_pool_dim = 256
        elif self.arch_version >= 2:
            self.timeline_blocks = nn.ModuleList(
                [
                    _ResConv1dBlock(timeline_dim, 64, kernel_size=3),
                    _ResConv1dBlock(64, 64, kernel_size=3),
                    _ResConv1dBlock(64, 64, kernel_size=3),
                ]
            )
            tl_pool_dim = 128
        else:
            self.timeline_conv = nn.Sequential(
                nn.Conv1d(timeline_dim, 48, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(48, 64, kernel_size=3, padding=1),
                nn.GELU(),
            )
            tl_pool_dim = 128
        self.node_in = nn.Sequential(
            nn.Linear(planet_dim + tl_pool_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        use_geglu = self.arch_version >= 2
        self.layers = nn.ModuleList(
            [_EdgeBiasedTransformerLayer(d_model, n_heads, dim_ff, edge_dim, use_geglu=use_geglu) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.src_proj = nn.Linear(d_model, d_model // 2)
        self.dst_proj = nn.Linear(d_model, d_model // 2)
        self.edge_proj = nn.Linear(edge_dim, d_model // 2)
        if self.arch_version >= 2:
            self.launch_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            self.tgt_q = nn.Linear(d_model, d_model)
            self.tgt_k = nn.Linear(d_model, d_model)
            self.tgt_head_weights = nn.Parameter(torch.zeros(n_heads))
            self.edge_target_head = nn.Sequential(
                nn.Linear(edge_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            self.frac_head = nn.Sequential(
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            self.launch_ctx_head: Optional[nn.Module] = None
            if self.target_cond_launch:
                # Zero-initialized target-aware residual on launch logit for bit-exact PT-v3 warm-start
                self.launch_ctx_head = nn.Sequential(
                    nn.Linear(2 * d_model + edge_dim + 3, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                nn.init.zeros_(self.launch_ctx_head[-1].weight)
                nn.init.zeros_(self.launch_ctx_head[-1].bias)
        else:
            # Per-planet HOLD logit
            self.hold_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            # Pairwise target logit & send-fraction head conditioned on (h_src, h_dst, edge_feat)
            self.pair_head = nn.Sequential(
                nn.GELU(),
                nn.Linear(d_model // 2, 2),  # [target_logit, frac_logit]
            )

        # Global value head (attention + mean pooled in PT-v3; attention pooled in PT-v1/v2)
        self.val_attn = nn.Linear(d_model, 1)
        if self.arch_version >= 3:
            self.val_head = nn.Sequential(
                nn.Linear(2 * d_model, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
        else:
            self.val_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        # Optional training-only auxiliary future-state world-model head (inspired by #7 Audun):
        # predicts [owner_mine, owner_enemy, owner_neutral, log1p_ships_norm] at t + 10.
        # Never invoked during inference -> zero latency cost.
        self.aux_future_head: Optional[nn.Module] = None
        if self.use_aux_future:
            self.aux_future_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 4),
            )

    def _compute_value(self, h: torch.Tensor, pad_mask: torch.Tensor, neg_inf: float, detach_value: bool) -> torch.Tensor:
        h_v = h.detach() if detach_value else h
        va = self.val_attn(h_v).squeeze(-1).masked_fill(~pad_mask, neg_inf)
        vw = F.softmax(va, dim=-1).unsqueeze(-1)
        pooled_attn = (h_v * vw).sum(dim=1)
        if self.arch_version >= 3:
            pm_f = pad_mask.unsqueeze(-1).to(h_v.dtype)
            pooled_mean = (h_v * pm_f).sum(dim=1) / pm_f.sum(dim=1).clamp_min(1.0)
            pooled = torch.cat([pooled_attn, pooled_mean], dim=-1)
        else:
            pooled = pooled_attn
        return torch.tanh(self.val_head(pooled).squeeze(-1))

    def forward(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        return_aux: bool = False,
        legal_edges: Optional[torch.Tensor] = None,
        compute_frac: bool = True,
        value_only: bool = False,
        detach_value: bool = False,
    ):
        """Forward pass.

        Args:
            planet_feats: ``(B, N, PLANET_FEAT_DIM)``
            timeline_feats: ``(B, N, T, TIMELINE_FEAT_DIM)``
            edge_feats: ``(B, N, N, EDGE_FEAT_DIM)``
            pad_mask: ``(B, N)`` bool, True for valid planets.
            return_aux: When ``True`` (and ``use_aux_future=True``), returns a 4-tuple
                including ``aux_future`` of shape ``(B, N, 4)``.
            legal_edges: Optional ``(B, N, N)`` bool mask for legal targets in ``arch_version >= 2``.
            detach_value: When ``True``, stops gradients from ``val_head`` into the shared trunk.

        Returns:
            action_logits: ``(B, N, N + 1)`` where index ``0`` is HOLD and ``1 + j`` is target ``j``.
            frac_preds: ``(B, N, N)`` in ``(0, 1)`` predicted send fraction for ``(i -> j)``.
            value: ``(B,)`` predicted outcome in ``[-1, +1]``.
        """
        if timeline_feats.shape[-1] > self.timeline_dim:
            timeline_feats = timeline_feats[..., : self.timeline_dim]
        B, N, T, Ct = timeline_feats.shape
        neg_inf = -1e4 if planet_feats.dtype == torch.float16 else -1e9
        if self.arch_version >= 3:
            zeros_bn = torch.zeros((B, N), dtype=planet_feats.dtype, device=planet_feats.device)
            if Ct >= 14:
                t0_en = (planet_feats[..., 13] + planet_feats[..., 14] + planet_feats[..., 15]).clamp(0.0, 1.0)
                t0_ships_lin = (torch.exp((planet_feats[..., 10] * 6.0).clamp(0.0, 9.0)) - 1.0).clamp_min(0.0)
                t0_cap_log = torch.where(
                    planet_feats[..., 12] > 0.5,
                    zeros_bn,
                    torch.log1p(t0_ships_lin + 1.0) / 6.0,
                )
                t0_slice = torch.stack(
                    [
                        planet_feats[..., 11],
                        planet_feats[..., 12],
                        t0_en,
                        planet_feats[..., 10],
                        zeros_bn,
                        zeros_bn,
                        zeros_bn,
                        zeros_bn,
                        planet_feats[..., 0],
                        planet_feats[..., 1],
                        planet_feats[..., 4] * (50.0 / 70.0),
                        pad_mask.to(planet_feats.dtype),
                        t0_cap_log,
                        planet_feats[..., 16],
                    ],
                    dim=-1,
                ).unsqueeze(2)  # (B, N, 1, 14)
                tl_seq = torch.cat([t0_slice, timeline_feats], dim=2)  # (B, N, T+1, 14)
                t_valid = (tl_seq[..., 11] > 0.5) & pad_mask.unsqueeze(-1)
                lin_cap = ((torch.exp((tl_seq[..., 12] * 6.0).clamp(0.0, 9.0)) - 1.0) / 100.0).clamp(0.0, 10.0)
            else:
                tl_seq = timeline_feats
                t_valid = pad_mask.unsqueeze(-1).expand(B, N, T)
                lin_cap = torch.zeros((B, N, T), dtype=planet_feats.dtype, device=planet_feats.device)

            T_eff = tl_seq.shape[2]
            if Ct >= 14:
                lin_src = torch.cat([tl_seq[..., 3:6], tl_seq[..., 12:13]], dim=-1)
                lin_4 = ((torch.exp((lin_src * 6.0).clamp(0.0, 9.0)) - 1.0) / 100.0).clamp(0.0, 10.0)
            else:
                lin_3 = ((torch.exp((tl_seq[..., 3:6] * 6.0).clamp(0.0, 9.0)) - 1.0) / 100.0).clamp(0.0, 10.0)
                lin_4 = torch.cat([lin_3, lin_cap.unsqueeze(-1)], dim=-1)
            prod_seq = planet_feats[..., 9:10].unsqueeze(2).expand(B, N, T_eff, 1)
            tl_aug = torch.cat([tl_seq, lin_4, prod_seq], dim=-1)  # (B, N, T_eff, Ct+5)
            t_valid_bn = t_valid.reshape(B * N, 1, T_eff)
            t_valid_f = t_valid_bn.to(tl_aug.dtype)
            t_valid_ntc = t_valid_f.transpose(1, 2)
            tl_in = tl_aug.reshape(B * N, T_eff, -1).transpose(1, 2) * t_valid_f
            if tl_in.device.type == "cpu":
                tl_in = tl_in.contiguous()
                t_valid_ntc = t_valid_ntc.contiguous()

            # On CPU inference/rollouts with padded boards, pack only valid planets before 4x Conv1D
            # (cuts Conv1D compute proportionally to padding with zero shader-cache overhead).
            if (not tl_in.requires_grad) and tl_in.device.type == "cpu":
                flat_pm = pad_mask.reshape(-1)
                if not bool(flat_pm.all()):
                    v_idx = flat_pm.nonzero(as_tuple=False).squeeze(-1)
                    if v_idx.numel() == 0:
                        tl_pool = torch.zeros((B, N, 256), dtype=tl_in.dtype, device=tl_in.device)
                    else:
                        tl_sub = tl_in.index_select(0, v_idx)
                        tv_sub_f = t_valid_f.index_select(0, v_idx)
                        tv_sub_ntc = t_valid_ntc.index_select(0, v_idx)
                        tv_sub_bn = t_valid_bn.index_select(0, v_idx)
                        tl_h = tl_sub
                        for blk in self.timeline_blocks:
                            tl_h = blk(tl_h, mask_ntc=tv_sub_ntc)
                        tl_mean = tl_h.sum(dim=-1) / tv_sub_f.sum(dim=-1).clamp_min(1.0)
                        any_v = tv_sub_bn.any(dim=-1)
                        tl_max_raw = tl_h.masked_fill(~tv_sub_bn, neg_inf).amax(dim=-1)
                        tl_max = torch.where(any_v, tl_max_raw, torch.zeros_like(tl_mean))
                        pool_sub = torch.cat([tl_mean, tl_max], dim=-1)
                        pool_full = torch.zeros((B * N, 256), dtype=pool_sub.dtype, device=pool_sub.device)
                        pool_full.index_copy_(0, v_idx, pool_sub)
                        tl_pool = pool_full.view(B, N, 256)
                else:
                    tl_h = tl_in
                    for blk in self.timeline_blocks:
                        tl_h = blk(tl_h, mask_ntc=t_valid_ntc)
                    tl_mean = tl_h.sum(dim=-1) / t_valid_f.sum(dim=-1).clamp_min(1.0)
                    any_v = t_valid_bn.any(dim=-1)
                    tl_max_raw = tl_h.masked_fill(~t_valid_bn, neg_inf).amax(dim=-1)
                    tl_max = torch.where(any_v, tl_max_raw, torch.zeros_like(tl_mean))
                    tl_pool = torch.cat([tl_mean, tl_max], dim=-1).view(B, N, 256)
            else:
                tl_h = tl_in
                for blk in self.timeline_blocks:
                    tl_h = blk(tl_h, mask_ntc=t_valid_ntc)
                tl_mean = tl_h.sum(dim=-1) / t_valid_f.sum(dim=-1).clamp_min(1.0)
                any_v = t_valid_bn.any(dim=-1)  # (B*N, 1)
                tl_max_raw = tl_h.masked_fill(~t_valid_bn, neg_inf).amax(dim=-1)
                tl_max = torch.where(any_v, tl_max_raw, torch.zeros_like(tl_mean))
                tl_pool = torch.cat([tl_mean, tl_max], dim=-1).view(B, N, 256)
        elif self.arch_version >= 2:
            tl = timeline_feats.reshape(B * N, T, Ct).transpose(1, 2)  # (B*N, Ct, T)
            tl_h = tl
            for blk in self.timeline_blocks:
                tl_h = blk(tl_h)
            tl_pool = torch.cat([tl_h.mean(dim=-1), tl_h.amax(dim=-1)], dim=-1).view(B, N, 128)
        else:
            tl = timeline_feats.reshape(B * N, T, Ct).transpose(1, 2)  # (B*N, Ct, T)
            tl_h = self.timeline_conv(tl)  # (B*N, 64, T)
            tl_pool = torch.cat([tl_h.mean(dim=-1), tl_h.amax(dim=-1)], dim=-1).view(B, N, 128)

        x = self.node_in(torch.cat([planet_feats, tl_pool], dim=-1))
        if self.arch_version >= 3:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            x = layer(x, edge_feats, pad_mask)
        h = self.final_norm(x)  # (B, N, D)
        if self.arch_version >= 3:
            h = h * pad_mask.unsqueeze(-1).to(h.dtype)

        if value_only:
            value = self._compute_value(h, pad_mask, neg_inf, detach_value=detach_value)
            if return_aux and self.aux_future_head is not None:
                return None, None, value, self.aux_future_head(h)
            return None, None, value

        if self.arch_version >= 2:
            q = self.tgt_q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.tgt_k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            hw = F.softmax(self.tgt_head_weights, dim=0).view(1, self.n_heads, 1, 1)
            raw_tgt = (scores * hw).sum(dim=1) + self.edge_target_head(edge_feats).squeeze(-1)

            if legal_edges is not None:
                tgt_mask = legal_edges.bool()
            else:
                tgt_mask = (
                    pad_mask.unsqueeze(2)
                    & pad_mask.unsqueeze(1)
                    & (~torch.eye(N, dtype=torch.bool, device=h.device).unsqueeze(0))
                )
            has_legal = tgt_mask.any(dim=-1, keepdim=True)
            safe_mask = tgt_mask | (~has_legal)
            log_p_tgt = F.log_softmax(raw_tgt.masked_fill(~safe_mask, neg_inf), dim=-1).masked_fill(~tgt_mask, neg_inf)

            launch_logit = self.launch_head(h)  # (B, N, 1)
            if self.launch_ctx_head is not None:
                p_tgt = log_p_tgt.exp().masked_fill(~tgt_mask, 0.0)
                h_tgt = torch.bmm(p_tgt, h)
                e_tgt = (p_tgt.unsqueeze(-1) * edge_feats).sum(dim=2)
                max_p_tgt = p_tgt.amax(dim=-1, keepdim=True)
                safe_lp_tgt = log_p_tgt.masked_fill(~tgt_mask, 0.0)
                tgt_ent = -(p_tgt * safe_lp_tgt).sum(dim=-1, keepdim=True)
                best_raw = raw_tgt.masked_fill(~safe_mask, neg_inf).amax(dim=-1, keepdim=True)
                best_raw = torch.where(has_legal, best_raw.clamp(-10.0, 10.0) * 0.1, torch.zeros_like(best_raw))
                launch_ctx = torch.cat([h, h_tgt, e_tgt, max_p_tgt, tgt_ent, best_raw], dim=-1)
                launch_logit = launch_logit + self.launch_ctx_head(launch_ctx) * pad_mask.unsqueeze(-1).to(h.dtype)

            launch_lp2 = F.log_softmax(torch.cat([torch.zeros_like(launch_logit), launch_logit], dim=-1), dim=-1)
            log_p_hold = launch_lp2[..., :1]
            log_p_launch = launch_lp2[..., 1:2]
            action_logits = torch.cat([log_p_hold, log_p_launch + log_p_tgt], dim=-1)
            if compute_frac:
                pair_in = (
                    self.src_proj(h).unsqueeze(2)
                    + self.dst_proj(h).unsqueeze(1)
                    + self.edge_proj(edge_feats)
                )  # (B, N, N, D/2)
                frac_preds = torch.sigmoid(self.frac_head(pair_in).squeeze(-1))
            else:
                frac_preds = None
        else:
            pair_in = (
                self.src_proj(h).unsqueeze(2)
                + self.dst_proj(h).unsqueeze(1)
                + self.edge_proj(edge_feats)
            )  # (B, N, N, D/2)
            hold_logit = self.hold_head(h)  # (B, N, 1)
            pair_out = self.pair_head(pair_in)  # (B, N, N, 2)
            target_logits = pair_out[..., 0]  # (B, N, N)
            frac_preds = torch.sigmoid(pair_out[..., 1])  # (B, N, N)
            action_logits = torch.cat([hold_logit, target_logits], dim=-1)  # (B, N, N + 1)

        value = self._compute_value(h, pad_mask, neg_inf, detach_value=detach_value)
        if return_aux and self.aux_future_head is not None:
            aux_future = self.aux_future_head(h)  # (B, N, 4)
            return action_logits, frac_preds, value, aux_future
        return action_logits, frac_preds, value


def pad_encoded_state(enc: EncodedPlanetState, max_n: int = MAX_PLANETS) -> Dict[str, np.ndarray]:
    """Pad an ``EncodedPlanetState`` to fixed ``max_n`` for batched training."""
    n = len(enc.planet_ids)
    tl_dim = int(enc.timeline_feats.shape[-1]) if enc.timeline_feats.ndim == 3 else TIMELINE_FEAT_DIM
    pf = np.zeros((max_n, PLANET_FEAT_DIM), dtype=np.float32)
    tf = np.zeros((max_n, TIMELINE_HORIZON, tl_dim), dtype=np.float32)
    ef = np.zeros((max_n, max_n, EDGE_FEAT_DIM), dtype=np.float32)
    le = np.zeros((max_n, max_n), dtype=bool)
    om = np.zeros((max_n,), dtype=bool)
    pm = np.zeros((max_n,), dtype=bool)

    pf[:n] = enc.planet_feats
    tf[:n] = enc.timeline_feats
    ef[:n, :n] = enc.edge_feats
    le[:n, :n] = enc.legal_edges
    om[:n] = enc.owned_mask
    pm[:n] = True
    return {
        "planet_feats": pf,
        "timeline_feats": tf,
        "edge_feats": ef,
        "legal_edges": le,
        "owned_mask": om,
        "pad_mask": pm,
    }


class PlanetTransformerAgent:
    """Fast Kaggle-compatible inference agent powered by ``PlanetTimelineTransformer``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        use_rot_tta: bool = True,
        use_d4_tta: bool = False,
        force_all_in: bool = False,
        reinforce_threshold: Optional[float] = None,
        surgical_neutral_cap: bool = False,
        anti_bounce_guard: bool = False,
        decoupled_logit_tta: Optional[bool] = None,
        device: str = "cpu",
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.launch_threshold = float(launch_threshold)
        self.use_rot_tta = bool(use_rot_tta)
        self.use_d4_tta = bool(use_d4_tta)
        self.force_all_in = bool(force_all_in)
        self.reinforce_threshold = None if reinforce_threshold is None else float(reinforce_threshold)
        self.surgical_neutral_cap = bool(surgical_neutral_cap)
        self.anti_bounce_guard = bool(anti_bounce_guard)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        self.decoupled_logit_tta = (
            bool(cfg.get("target_cond_launch", False))
            if decoupled_logit_tta is None
            else bool(decoupled_logit_tta)
        )
        self.feature_version = int(cfg.get("feature_version", 1))
        self.timeline_dim = int(cfg.get("timeline_dim", TIMELINE_FEAT_DIM))
        self.model = PlanetTimelineTransformer(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _tta_symmetries(
        self,
        *,
        use_rot_tta: Optional[bool] = None,
        use_d4_tta: Optional[bool] = None,
    ) -> List[Tuple[int, bool]]:
        d4 = self.use_d4_tta if use_d4_tta is None else bool(use_d4_tta)
        if d4:
            return [(k, f) for f in (False, True) for k in range(4)]
        rot = self.use_rot_tta if use_rot_tta is None else bool(use_rot_tta)
        if rot:
            return [(k, False) for k in range(4)]
        return [(0, False)]

    @torch.no_grad()
    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        max_speed = float(_obs_get(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config else DEFAULT_SHIP_SPEED
        enc = extract_planet_features(obs, max_speed=max_speed, feature_version=self.feature_version)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return []

        probs, mean_fracs, _ = self.predict_probs_and_value(enc)
        return self._decode_plan(
            enc,
            probs,
            mean_fracs,
            launch_threshold=self.launch_threshold,
            force_all_in=self.force_all_in,
            max_speed=max_speed,
            reinforce_threshold=self.reinforce_threshold,
            surgical_neutral_cap=self.surgical_neutral_cap,
            anti_bounce_guard=self.anti_bounce_guard,
        )

    @torch.no_grad()
    def predict_probs_and_value(
        self,
        enc: EncodedPlanetState,
        *,
        use_rot_tta: Optional[bool] = None,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the Transformer on ``enc`` and return ``(probs, mean_fracs, value)``."""
        n = len(enc.planet_ids)
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0.0

        syms = self._tta_symmetries(use_rot_tta=use_rot_tta, use_d4_tta=use_d4_tta)
        k_batch = len(syms)
        tf_base = enc.timeline_feats[..., : self.timeline_dim]
        if k_batch > 1:
            pf_batch = np.stack(
                [apply_d4_planet_feats(enc.planet_feats, k_rot=k, flip_x=f) for k, f in syms],
                axis=0,
            )
            tf_batch = np.stack(
                [apply_d4_timeline_feats(tf_base, k_rot=k, flip_x=f) for k, f in syms],
                axis=0,
            )
            # edge_feats is 100% D4-invariant: pass (1, N, N, E) to broadcast across TTA symmetries (8x less edge FLOPs)
            ef_batch = enc.edge_feats[None]
            pm_batch = np.ones((k_batch, n), dtype=bool)
        else:
            pf_batch = enc.planet_feats[None]
            tf_batch = tf_base[None]
            ef_batch = enc.edge_feats[None]
            pm_batch = np.ones((1, n), dtype=bool)

        legal = torch.from_numpy(enc.legal_edges).to(self.device)
        want_frac = not self.force_all_in
        logits, fracs, vals = self.model(
            torch.from_numpy(pf_batch).to(self.device),
            torch.from_numpy(tf_batch).to(self.device),
            torch.from_numpy(ef_batch).to(self.device),
            torch.from_numpy(pm_batch).to(self.device),
            legal_edges=legal.unsqueeze(0).expand(k_batch, n, n),
            compute_frac=want_frac,
        )
        mask = torch.cat([torch.ones((n, 1), dtype=torch.bool, device=self.device), legal], dim=-1)
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        if self.decoupled_logit_tta and k_batch > 1 and getattr(self.model, "arch_version", 1) >= 2:
            lp_all = F.log_softmax(logits, dim=-1)
            lp_h = lp_all[..., 0]
            has_l = legal.any(dim=-1)
            safe_l = legal | (~has_l.unsqueeze(-1))
            lp_l = torch.logsumexp(lp_all[..., 1:].masked_fill(~safe_l.unsqueeze(0), -1e9), dim=-1)
            launch_z = (lp_l - lp_h).clamp(-15.0, 15.0)
            mean_p_l = torch.sigmoid(launch_z.mean(dim=0))
            mean_p_l = torch.where(has_l, mean_p_l, torch.zeros_like(mean_p_l))
            cond_tgt = (lp_all[..., 1:] - lp_l.unsqueeze(-1)).masked_fill(~safe_l.unsqueeze(0), -1e9)
            mean_p_t = F.softmax(cond_tgt.mean(dim=0).masked_fill(~safe_l, -1e9), dim=-1)
            mean_p_t = mean_p_t.masked_fill(~legal, 0.0)
            probs = torch.cat([(1.0 - mean_p_l).unsqueeze(-1), mean_p_l.unsqueeze(-1) * mean_p_t], dim=-1).cpu().numpy()
        else:
            probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        mean_fracs = (
            fracs.mean(dim=0).cpu().numpy()
            if fracs is not None
            else np.ones((n, n), dtype=np.float32)
        )
        val = float(vals.mean().item())
        return probs, mean_fracs, val

    @torch.no_grad()
    def predict_batch(
        self,
        encs: Sequence[EncodedPlanetState],
        *,
        compute_frac: Optional[bool] = None,
        value_only: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Run a single batched forward pass over multiple distinct encoded board states.

        Handles variable planet counts across branches (e.g. comet spawn/despawn) via
        minimal right-padding when needed, with a zero-copy fast path when all states
        share the same planet count.
        """
        b_size = len(encs)
        if b_size == 0:
            return [], [], np.zeros((0,), dtype=np.float32)

        ns = [len(e.planet_ids) for e in encs]
        max_n = max(ns)
        if max_n == 0:
            empty_p = [np.zeros((0, 1), dtype=np.float32) for _ in range(b_size)]
            empty_f = [np.zeros((0, 0), dtype=np.float32) for _ in range(b_size)]
            return empty_p, empty_f, np.zeros((b_size,), dtype=np.float32)

        if all(n_i == max_n for n_i in ns):
            pf_np = np.stack([e.planet_feats for e in encs], axis=0)
            tf_np = np.stack([e.timeline_feats[..., : self.timeline_dim] for e in encs], axis=0)
            ef_np = np.stack([e.edge_feats for e in encs], axis=0)
            le_np = np.stack([e.legal_edges for e in encs], axis=0)
            pm_np = np.ones((b_size, max_n), dtype=bool)
        else:
            pf_np = np.zeros((b_size, max_n, PLANET_FEAT_DIM), dtype=np.float32)
            tf_np = np.zeros((b_size, max_n, TIMELINE_HORIZON, self.timeline_dim), dtype=np.float32)
            ef_np = np.zeros((b_size, max_n, max_n, EDGE_FEAT_DIM), dtype=np.float32)
            le_np = np.zeros((b_size, max_n, max_n), dtype=bool)
            pm_np = np.zeros((b_size, max_n), dtype=bool)
            for idx_b, (e, n_i) in enumerate(zip(encs, ns)):
                if n_i > 0:
                    pf_np[idx_b, :n_i] = e.planet_feats
                    tf_np[idx_b, :n_i] = e.timeline_feats[..., : self.timeline_dim]
                    ef_np[idx_b, :n_i, :n_i] = e.edge_feats
                    le_np[idx_b, :n_i, :n_i] = e.legal_edges
                    pm_np[idx_b, :n_i] = True
                else:
                    pm_np[idx_b, 0] = True

        want_frac = (not self.force_all_in) if compute_frac is None else bool(compute_frac)
        le_t = torch.from_numpy(le_np).to(self.device)
        logits, fracs, vals = self.model(
            torch.from_numpy(pf_np).to(self.device),
            torch.from_numpy(tf_np).to(self.device),
            torch.from_numpy(ef_np).to(self.device),
            torch.from_numpy(pm_np).to(self.device),
            legal_edges=le_t,
            compute_frac=want_frac and (not value_only),
            value_only=value_only,
        )
        vals_np = vals.cpu().numpy().astype(np.float32, copy=False)
        for idx_b, n_i in enumerate(ns):
            if n_i == 0:
                vals_np[idx_b] = 0.0
        if value_only:
            return [], [], vals_np

        hold_ones = torch.ones((b_size, max_n, 1), dtype=torch.bool, device=self.device)
        mask = torch.cat([hold_ones, le_t], dim=-1)
        probs_all = F.softmax(logits.masked_fill(~mask, -1e9), dim=-1).cpu().numpy()
        fracs_all = fracs.cpu().numpy() if fracs is not None else None

        probs_list: List[np.ndarray] = []
        fracs_list: List[np.ndarray] = []
        for idx_b, n_i in enumerate(ns):
            if n_i == 0:
                probs_list.append(np.zeros((0, 1), dtype=np.float32))
                fracs_list.append(np.zeros((0, 0), dtype=np.float32))
            else:
                probs_list.append(probs_all[idx_b, :n_i, : n_i + 1])
                if fracs_all is not None:
                    fracs_list.append(fracs_all[idx_b, :n_i, :n_i])
                else:
                    fracs_list.append(np.ones((n_i, n_i), dtype=np.float32))
        return probs_list, fracs_list, vals_np

    def _decode_plan(
        self,
        enc: EncodedPlanetState,
        probs: np.ndarray,
        mean_fracs: np.ndarray,
        *,
        launch_threshold: float,
        force_all_in: bool,
        max_speed: float,
        overrides: Optional[Dict[int, int]] = None,
        top_k_fallback: int = 3,
        reinforce_threshold: Optional[float] = None,
        surgical_neutral_cap: bool = False,
        anti_bounce_guard: bool = False,
    ) -> List[List[float]]:
        """Convert per-planet probabilities (+ optional per-planet overrides) into Kaggle moves.

        ``overrides`` maps ``planet_idx -> action_class`` where ``0`` = HOLD and ``1 + j`` = target ``j``.
        Optional Gen 9 midgame guards (all off by default for 100% backward compatibility):
          - ``reinforce_threshold``: raises the launch bar for peaceful friendly-to-friendly
            shuffling while preserving normal thresholds for threatened rescues.
          - ``surgical_neutral_cap``: caps uncontested neutral captures to ``needed + margin``
            when doing so preserves ETA within <= +2 ticks, curbing neutral ship overkill.
          - ``anti_bounce_guard``: holds low-confidence solo enemy attacks that cannot flip
            the target even after pooling all same-tick friendly launches + in-flight fleets.
        """
        n = len(enc.planet_ids)
        expiring_set = set(enc.expiring_comet_indices)
        has_futures = len(enc.futures) == n

        # Pass 1: collect candidate launches per owned planet
        proposals: List[Tuple[int, int, int, float, int, bool, float]] = []
        # Tuple: (src_idx, dst_idx, send_ships, angle, eta, forced, p_target)

        for i in range(n):
            if not enc.owned_mask[i]:
                continue
            if not np.any(enc.legal_edges[i]):
                continue
            forced = i in expiring_set
            p_launch = 1.0 - float(probs[i, 0])

            if overrides is not None and i in overrides and not forced:
                cls = int(overrides[i])
                if cls == 0:
                    continue
                cand_targets = [cls - 1]
            else:
                if (p_launch < launch_threshold) and (not forced):
                    continue
                order = np.argsort(-probs[i, 1:])
                cand_targets = []
                for idx_j in order:
                    j_int = int(idx_j)
                    if not enc.legal_edges[i, j_int]:
                        continue
                    if cand_targets and (not forced) and float(probs[i, 1 + j_int]) < 0.05:
                        break
                    cand_targets.append(j_int)
                    if len(cand_targets) >= max(1, int(top_k_fallback)):
                        break

            src_row = enc.planet_rows[i]
            avail = int(src_row[5])
            if avail <= 0:
                continue

            for best_j in cand_targets:
                if best_j < 0 or best_j >= n or not enc.legal_edges[i, best_j]:
                    continue
                dst_row = enc.planet_rows[best_j]
                dst_pid = int(dst_row[0])
                dst_owner = int(dst_row[1])
                p_tgt = float(probs[i, 1 + best_j])

                # Guard 1 (Hold Calibration): curb premature peaceful friendly shuffles
                if (
                    reinforce_threshold is not None
                    and (not forced)
                    and overrides is None
                    and dst_owner == enc.player
                    and p_launch < reinforce_threshold
                ):
                    # Always allow urgent defensive rescue if friendly target is projected to fall
                    threatened = False
                    if has_futures:
                        threatened = any(ow != enc.player for ow, _ in enc.futures[best_j])
                    if not threatened:
                        break

                if force_all_in or forced:
                    send_ships = avail
                else:
                    f = float(mean_fracs[i, best_j])
                    send_ships = avail if f >= 0.88 else max(1, min(avail, int(round(avail * f))))

                sol = solve_intercept(
                    src_row,
                    dst_row,
                    send_ships,
                    enc.planet_rows,
                    enc.current_step,
                    enc.angular_velocity,
                    enc.initial_by_id,
                    comet_lookup=enc.comet_lookup,
                    tracks=enc.tracks,
                    max_eta=MAX_ROUTE_ETA + 4,
                    max_speed=max_speed,
                )
                if sol is None:
                    continue

                # Guard 2 (Surgical Neutral Cap): reclaim excess overkill on uncontested neutrals
                if (
                    surgical_neutral_cap
                    and (not forced)
                    and dst_owner == -1
                    and send_ships >= 12
                ):
                    dst_arrs = enc.arrivals.get(dst_pid, ())
                    enemy_inbound = any(a.owner != enc.player for a in dst_arrs)
                    if not enemy_inbound:
                        if has_futures:
                            t_idx = min(sol.eta, len(enc.futures[best_j]) - 1)
                            fut_own, fut_sh = enc.futures[best_j][t_idx]
                        else:
                            fut_own, fut_sh = -1, int(dst_row[5])
                        if fut_own == -1:
                            margin = 5 if int(dst_row[6]) >= 4 else 3
                            capped = max(1, int(fut_sh) + 1 + margin)
                            if capped + 8 <= send_ships:
                                sol_cap = solve_intercept(
                                    src_row,
                                    dst_row,
                                    capped,
                                    enc.planet_rows,
                                    enc.current_step,
                                    enc.angular_velocity,
                                    enc.initial_by_id,
                                    comet_lookup=enc.comet_lookup,
                                    tracks=enc.tracks,
                                    max_eta=MAX_ROUTE_ETA + 4,
                                    max_speed=max_speed,
                                )
                                if sol_cap is not None and sol_cap.eta <= sol.eta + 2:
                                    send_ships = capped
                                    sol = sol_cap

                proposals.append((i, best_j, int(send_ships), float(sol.angle), int(sol.eta), forced, p_tgt))
                break

        # Pass 2 (Anti-Bounce Guard): filter solo enemy attacks guaranteed to bounce
        if not anti_bounce_guard or not proposals:
            return [[int(enc.planet_rows[i][0]), ang, sh] for i, _, sh, ang, _, _, _ in proposals]

        planned_to_dst: Dict[int, int] = {}
        for _, dst_j, sh, _, _, _, _ in proposals:
            planned_to_dst[dst_j] = planned_to_dst.get(dst_j, 0) + sh

        moves: List[List[float]] = []
        for i, dst_j, sh, ang, eta, forced, p_tgt in proposals:
            dst_row = enc.planet_rows[dst_j]
            dst_owner = int(dst_row[1])
            if (not forced) and dst_owner >= 0 and dst_owner != enc.player and p_tgt < 0.78 and has_futures:
                t_idx = min(eta, len(enc.futures[dst_j]) - 1)
                fut_own, fut_sh = enc.futures[dst_j][t_idx]
                if fut_own != enc.player and planned_to_dst.get(dst_j, sh) <= fut_sh:
                    # Guaranteed to bounce even pooling all same-tick friendly waves -> hold & accumulate
                    continue
            moves.append([int(enc.planet_rows[i][0]), ang, sh])
        return moves


class SearchPlanetTransformerAgent(PlanetTransformerAgent):
    """Batched multi-ply value-guided lookahead search over ``PlanetTimelineTransformer``.

    Uses the Transformer's per-planet policy posterior to propose diverse, ship-weighted
    borderline full-turn plans (surgical launch-threshold shifts, wait/hold-all, top
    ship-weighted Launch<->Hold flips, alternative target switches, and synchronized
    2-planet pair flips), rolls ALL candidate branches forward in lockstep inside
    ``FastOrbitSim`` with batched neural counter-replanning for both players, and scores
    leaves via batched symmetric zero-sum neural value (+ optional tactical blend) under
    a strict wall-clock budget.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        use_rot_tta: bool = True,
        use_d4_tta: bool = False,
        force_all_in: bool = False,
        neural_mid_replan: bool = False,
        rollout_horizon: int = 10,
        time_budget_sec: float = 0.28,
        neural_value_weight: float = 0.85,
        replan_steps: Optional[Sequence[int]] = None,
        max_candidates: int = 8,
        prior_bonus: float = 0.006,
        symmetric_value: bool = True,
        opp_rot_tta: bool = False,
        opp_launch_threshold: Optional[float] = None,
        trajectory_val_blend: float = 0.0,
        leaf_rot_tta_top_k: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            launch_threshold=launch_threshold,
            use_rot_tta=use_rot_tta,
            use_d4_tta=use_d4_tta,
            force_all_in=force_all_in,
            device=device,
        )
        self.neural_mid_replan = bool(neural_mid_replan)
        self.rollout_horizon = max(2, int(rollout_horizon))
        self.time_budget_sec = float(time_budget_sec)
        self.neural_value_weight = float(neural_value_weight)
        self.max_candidates = max(2, int(max_candidates))
        self.prior_bonus = float(prior_bonus)
        self.symmetric_value = bool(symmetric_value)
        self.opp_rot_tta = bool(opp_rot_tta)
        self.opp_launch_threshold = self.launch_threshold if opp_launch_threshold is None else float(opp_launch_threshold)
        self.trajectory_val_blend = float(np.clip(trajectory_val_blend, 0.0, 0.9))
        self.leaf_rot_tta_top_k = max(0, int(leaf_rot_tta_top_k))
        if replan_steps is not None:
            self.replan_steps: Tuple[int, ...] = tuple(
                sorted({int(s) for s in replan_steps if 0 < int(s) < self.rollout_horizon})
            )
        elif self.neural_mid_replan:
            if self.rollout_horizon >= 10:
                raw_rs = (2, 5, 8)
            elif self.rollout_horizon >= 7:
                raw_rs = (2, 5)
            else:
                raw_rs = (max(1, self.rollout_horizon // 2),)
            self.replan_steps = tuple(s for s in raw_rs if 0 < s < self.rollout_horizon)
        else:
            self.replan_steps = (max(1, self.rollout_horizon // 2),)

    def _plan_key(self, moves: Sequence[Sequence[float]]) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(sorted((int(m[0]), int(round(float(m[1]) * 100.0)), int(m[2])) for m in moves))

    def _plan_log_prior(
        self,
        enc: EncodedPlanetState,
        probs: np.ndarray,
        moves: Sequence[Sequence[float]],
        *,
        overrides: Optional[Dict[int, int]] = None,
        launch_threshold: Optional[float] = None,
    ) -> float:
        """Factorized threshold-centered log-prior (zero multi-target HOLD bias).

        At ``p_launch == thr``, HOLD and launching at the top target have equal prior (0.0);
        switching to a secondary target ``j2`` incurs ``log(p(j2) / p(j1)) <= 0``.
        """
        thr = float(np.clip(self.launch_threshold if launch_threshold is None else launch_threshold, 1e-3, 1.0 - 1e-3))
        expiring_set = set(enc.expiring_comet_indices)
        by_src = {int(m[0]): m for m in moves}
        lp = 0.0
        for i in range(len(enc.planet_ids)):
            if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]) or i in expiring_set:
                continue
            pid = enc.planet_ids[i]
            p_hold = max(1e-4, float(probs[i, 0]))
            p_launch = max(1e-4, 1.0 - float(probs[i, 0]))
            if pid not in by_src:
                lp += math.log(p_hold / (1.0 - thr))
            else:
                lp += math.log(p_launch / thr)
                legal_js = np.flatnonzero(enc.legal_edges[i])
                p_tgt1 = max(1e-4, float(np.max(probs[i, 1 + legal_js])))
                if overrides is not None and i in overrides and overrides[i] > 0:
                    p_chosen = max(1e-4, float(probs[i, int(overrides[i])]))
                    lp += math.log(min(1.0, p_chosen / p_tgt1))
        return lp

    def _build_candidates(
        self,
        enc: EncodedPlanetState,
        probs: np.ndarray,
        mean_fracs: np.ndarray,
        max_speed: float,
        shared_physics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[List[List[float]]], List[float]]:
        """Generate deduplicated ship-weighted & tactical candidate plans with exact log-priors."""
        n = len(enc.planet_ids)
        expiring_set = set(enc.expiring_comet_indices)
        thr = self.launch_threshold

        # Candidate tuples: (moves, exact_log_prior)
        raw: List[Tuple[List[List[float]], float]] = []

        def _add_candidate(
            *,
            c_thr: float = thr,
            c_all_in: bool = self.force_all_in,
            overrides: Optional[Dict[int, int]] = None,
        ) -> None:
            mv = self._decode_plan(
                enc,
                probs,
                mean_fracs,
                launch_threshold=c_thr,
                force_all_in=c_all_in,
                max_speed=max_speed,
                overrides=overrides,
            )
            lp = self._plan_log_prior(enc, probs, mv, overrides=overrides, launch_threshold=thr)
            raw.append((mv, lp))

        # 1. Greedy policy baseline (always index 0)
        _add_candidate(c_thr=thr, c_all_in=self.force_all_in)

        # 2. Surgical tempo shifts + wait-and-accumulate hold candidate
        if self.force_all_in:
            _add_candidate(c_thr=max(0.36, thr - 0.015), c_all_in=True)
            _add_candidate(c_thr=min(0.70, thr + 0.015), c_all_in=True)
            # Pure wait (holds non-expiring planets while still evacuating expiring comets)
            _add_candidate(c_thr=1.01, c_all_in=True)
        else:
            _add_candidate(c_thr=0.35, c_all_in=False)
            _add_candidate(c_thr=0.65, c_all_in=False)
            _add_candidate(c_thr=0.45, c_all_in=True)

        # 3. Tactical proposals from shared physics: Defensive Rescue & Synchronized 2-Planet Strike
        best_eta = shared_physics.get("best_eta") if shared_physics is not None else None
        if self.force_all_in and best_eta is not None and len(enc.futures) == n:
            # (a) Urgent defensive rescue for highest-value owned planet projected to flip within 14t
            best_rescue: Optional[Tuple[float, int, int]] = None
            for d_idx in range(n):
                if not enc.owned_mask[d_idx]:
                    continue
                fut_d = enc.futures[d_idx]
                flip_t = None
                for t_step in range(1, min(15, len(fut_d))):
                    if fut_d[t_step][0] != enc.player:
                        flip_t = t_step
                        break
                if flip_t is None:
                    continue
                val_d = float(enc.planet_rows[d_idx][6]) * 20.0 + float(enc.planet_rows[d_idx][5])
                for s_idx in range(n):
                    if s_idx == d_idx or (not enc.owned_mask[s_idx]) or (s_idx in expiring_set):
                        continue
                    if not enc.legal_edges[s_idx, d_idx]:
                        continue
                    eta_sd = int(best_eta[s_idx, d_idx])
                    if 1 <= eta_sd <= flip_t:
                        sc = val_d + float(enc.planet_rows[s_idx][5]) * 0.5
                        if best_rescue is None or sc > best_rescue[0]:
                            best_rescue = (sc, s_idx, d_idx)
            if best_rescue is not None:
                _add_candidate(overrides={best_rescue[1]: 1 + best_rescue[2]})

            # (b) Synchronized 2-planet focus-fire strike onto a contested non-friendly target
            best_sync: Optional[Tuple[float, int, int, int]] = None
            greedy_targets: Dict[int, int] = {}
            for i in range(n):
                if enc.owned_mask[i] and np.any(enc.legal_edges[i]) and (1.0 - float(probs[i, 0]) >= thr):
                    ljs = np.flatnonzero(enc.legal_edges[i])
                    greedy_targets[i] = int(ljs[int(np.argmax(probs[i, 1 + ljs]))])
            for i1, dst in greedy_targets.items():
                if int(enc.planet_rows[dst][1]) == enc.player:
                    continue
                eta1 = int(best_eta[i1, dst])
                if eta1 <= 0:
                    continue
                sh1 = float(enc.planet_rows[i1][5])
                dst_prod = float(enc.planet_rows[dst][6])
                for i2 in range(n):
                    if i2 == i1 or (not enc.owned_mask[i2]) or (i2 in expiring_set):
                        continue
                    if not enc.legal_edges[i2, dst] or greedy_targets.get(i2) == dst:
                        continue
                    eta2 = int(best_eta[i2, dst])
                    if eta2 <= 0 or abs(eta2 - eta1) > 3:
                        continue
                    p_l2 = 1.0 - float(probs[i2, 0])
                    p_d2 = float(probs[i2, 1 + dst])
                    if p_l2 >= thr - 0.10 or p_d2 >= 0.08:
                        sc = (dst_prod + 1.0) * (sh1 + float(enc.planet_rows[i2][5]))
                        if best_sync is None or sc > best_sync[0]:
                            best_sync = (sc, i1, i2, dst)
            if best_sync is not None:
                _add_candidate(overrides={best_sync[1]: 1 + best_sync[3], best_sync[2]: 1 + best_sync[3]})

        # 4. Ship-weighted borderline flips (both Launch<->Hold and 2nd-target switches)
        flip_proposals: List[Tuple[float, int, int]] = []
        for i in range(n):
            if not enc.owned_mask[i] or i in expiring_set:
                continue
            legal_js = np.flatnonzero(enc.legal_edges[i])
            if len(legal_js) == 0:
                continue
            avail = float(enc.planet_rows[i][5])
            prod = float(enc.planet_rows[i][6])
            if avail <= 0.0:
                continue
            weight = avail + 3.0 * prod
            p_launch = 1.0 - float(probs[i, 0])
            order = legal_js[np.argsort(-probs[i, 1 + legal_js])]
            j1 = int(order[0])

            # (a) Launch <-> Hold flip when within borderline band around threshold
            margin = abs(p_launch - thr)
            if margin <= 0.16:
                alt_cls = 0 if p_launch >= thr else (1 + j1)
                score = weight * (0.18 - margin)
                flip_proposals.append((score, i, alt_cls))

            # (b) Alternative target switch when planet launches and has a credible 2nd target
            if p_launch >= thr and len(order) >= 2:
                j2 = int(order[1])
                p1 = float(probs[i, 1 + j1])
                p2 = float(probs[i, 1 + j2])
                if p2 >= max(0.06, 0.16 * p1):
                    score = weight * p2 * 1.15
                    flip_proposals.append((score, i, 1 + j2))

        flip_proposals.sort(key=lambda x: x[0], reverse=True)

        for _, p_idx, alt_cls in flip_proposals[:5]:
            _add_candidate(overrides={p_idx: alt_cls})

        # 5. Synchronized pair flips across top distinct borderline planets
        top_by_planet: List[Tuple[int, int]] = []
        seen_p: set[int] = set()
        for _, p_idx, alt_cls in flip_proposals:
            if p_idx not in seen_p:
                seen_p.add(p_idx)
                top_by_planet.append((p_idx, alt_cls))
            if len(top_by_planet) >= 3:
                break
        if len(top_by_planet) >= 2:
            p0, c0 = top_by_planet[0]
            p1, c1 = top_by_planet[1]
            _add_candidate(overrides={p0: c0, p1: c1})
        if len(top_by_planet) >= 3:
            p0, c0 = top_by_planet[0]
            p2, c2 = top_by_planet[2]
            _add_candidate(overrides={p0: c0, p2: c2})

        # Deduplicate while preserving priority order (greedy baseline first)
        candidates: List[List[List[float]]] = []
        log_priors: List[float] = []
        seen_keys: set[Tuple[Tuple[int, int, int], ...]] = set()
        for plan, lp in raw:
            k = self._plan_key(plan)
            if k not in seen_keys:
                seen_keys.add(k)
                candidates.append(plan)
                log_priors.append(lp)
                if len(candidates) >= self.max_candidates:
                    break
        return candidates, log_priors

    @torch.no_grad()
    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        import time

        from orbit_wars_bot.fast_sim import FastOrbitSim
        from orbit_wars_bot.features import evaluate_state
        from orbit_wars_bot.heuristic_bot import heuristic_moves

        t_start = time.perf_counter()
        deadline = t_start + self.time_budget_sec
        max_speed = float(_obs_get(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config else DEFAULT_SHIP_SPEED

        player = int(_obs_get(obs, "player", 0))
        shared0 = build_shared_planet_physics(obs, max_speed=max_speed)
        enc = extract_planet_features(
            obs,
            player=player,
            max_speed=max_speed,
            feature_version=self.feature_version,
            shared_physics=shared0,
        )
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return []

        probs, mean_fracs, _ = self.predict_probs_and_value(enc, use_rot_tta=self.use_rot_tta)
        dt_root = time.perf_counter() - t_start
        candidates, log_priors = self._build_candidates(
            enc, probs, mean_fracs, max_speed=max_speed, shared_physics=shared0
        )
        # Adapt candidate count on huge boards (O(N^2) attention) to respect strict turn budget
        if n > 34:
            cap_n = min(self.max_candidates, 5)
        elif n > 28:
            cap_n = min(self.max_candidates, 6)
        else:
            cap_n = self.max_candidates
        if len(candidates) > cap_n:
            candidates = candidates[:cap_n]
            log_priors = log_priors[:cap_n]

        if len(candidates) <= 1 or time.perf_counter() >= deadline - max(0.045, dt_root * 1.5):
            return candidates[0] if candidates else []

        base_sim = FastOrbitSim.from_observation(obs, config)
        n_players = base_sim.num_agents
        k_cand = len(candidates)

        # Precompute simultaneous turn-0 opponent responses ONCE reusing root shared physics
        opp_moves_t0: Dict[int, List[List[float]]] = {}
        opp_ids_active: List[int] = []
        opp_encs_t0: List[EncodedPlanetState] = []
        for opp_id in range(n_players):
            if opp_id == player:
                continue
            opp_enc = extract_planet_features(
                obs,
                player=opp_id,
                max_speed=max_speed,
                feature_version=self.feature_version,
                shared_physics=shared0,
            )
            if np.any(opp_enc.owned_mask):
                opp_ids_active.append(opp_id)
                opp_encs_t0.append(opp_enc)
            else:
                opp_moves_t0[opp_id] = []

        if opp_encs_t0:
            if self.opp_rot_tta:
                for opp_id, opp_enc in zip(opp_ids_active, opp_encs_t0):
                    o_probs, o_fracs, _ = self.predict_probs_and_value(
                        opp_enc, use_rot_tta=True, use_d4_tta=False
                    )
                    opp_moves_t0[opp_id] = self._decode_plan(
                        opp_enc,
                        o_probs,
                        o_fracs,
                        launch_threshold=self.opp_launch_threshold,
                        force_all_in=self.force_all_in,
                        max_speed=max_speed,
                    )
            else:
                o_probs_l, o_fracs_l, _ = self.predict_batch(
                    opp_encs_t0, compute_frac=not self.force_all_in, value_only=False
                )
                for opp_id, opp_enc, o_probs, o_fracs in zip(opp_ids_active, opp_encs_t0, o_probs_l, o_fracs_l):
                    opp_moves_t0[opp_id] = self._decode_plan(
                        opp_enc,
                        o_probs,
                        o_fracs,
                        launch_threshold=self.opp_launch_threshold,
                        force_all_in=self.force_all_in,
                        max_speed=max_speed,
                    )

        # Initialize K candidate rollouts in lockstep and apply simultaneous t=0 moves
        sims: List[FastOrbitSim] = []
        for plan in candidates:
            sim = base_sim.clone()
            actions_t0: List[List[List[float]]] = [[] for _ in range(n_players)]
            actions_t0[player] = plan
            for opp_id, omv in opp_moves_t0.items():
                actions_t0[opp_id] = omv
            sim.step(actions_t0)
            sims.append(sim)

        replan_set = set(self.replan_steps)
        empty_acts = [[] for _ in range(n_players)]
        use_sym = self.symmetric_value and (n_players == 2)
        # Dynamic pass-time estimate so we never blow past deadline on large N boards
        est_replan_sec = max(0.035, dt_root * (0.28 * k_cand * (2 if n_players == 2 else 1) / 8.0))
        est_leaf_sec = max(0.030, est_replan_sec * (0.85 if use_sym else 0.50))

        # Optional zero-cost trajectory value accumulation from intermediate replan passes
        traj_val_wsum = np.zeros((k_cand,), dtype=np.float64)
        traj_weight = np.zeros((k_cand,), dtype=np.float64)

        for step_i in range(1, self.rollout_horizon):
            if step_i in replan_set and (time.perf_counter() + est_replan_sec + est_leaf_sec < deadline):
                if self.neural_mid_replan:
                    t_rep = time.perf_counter()
                    # Extract both players' features per alive branch sharing physics once per sim
                    req_meta: List[Tuple[int, int, EncodedPlanetState]] = []
                    step_acts_all: List[List[List[List[float]]]] = [
                        [[] for _ in range(n_players)] for _ in range(k_cand)
                    ]
                    for idx_c, sim in enumerate(sims):
                        if sim.done:
                            continue
                        obs_shared = sim.to_observation(player=player)
                        sh = build_shared_planet_physics(obs_shared, max_speed=max_speed)
                        for pid_cur in range(n_players):
                            m_enc = extract_planet_features(
                                obs_shared,
                                player=pid_cur,
                                max_speed=max_speed,
                                feature_version=self.feature_version,
                                shared_physics=sh,
                            )
                            if np.any(m_enc.owned_mask):
                                req_meta.append((idx_c, pid_cur, m_enc))
                    if req_meta:
                        m_probs_l, m_fracs_l, m_vals = self.predict_batch(
                            [m[2] for m in req_meta],
                            compute_frac=not self.force_all_in,
                            value_only=False,
                        )
                        if self.trajectory_val_blend > 0.0:
                            step_vsum = np.zeros((k_cand,), dtype=np.float64)
                            step_vcnt = np.zeros((k_cand,), dtype=np.int32)
                            for (idx_c, pid_cur, _), v_raw in zip(req_meta, m_vals):
                                if pid_cur == player:
                                    step_vsum[idx_c] += float(v_raw)
                                    step_vcnt[idx_c] += 1
                                elif use_sym:
                                    step_vsum[idx_c] -= float(v_raw)
                                    step_vcnt[idx_c] += 1
                            w_step = float(step_i) / float(self.rollout_horizon)
                            for idx_c in range(k_cand):
                                if step_vcnt[idx_c] > 0:
                                    traj_val_wsum[idx_c] += w_step * (step_vsum[idx_c] / step_vcnt[idx_c])
                                    traj_weight[idx_c] += w_step
                        for (idx_c, pid_cur, m_enc), m_p, m_f in zip(req_meta, m_probs_l, m_fracs_l):
                            step_acts_all[idx_c][pid_cur] = self._decode_plan(
                                m_enc,
                                m_p,
                                m_f,
                                launch_threshold=(
                                    self.launch_threshold if pid_cur == player else self.opp_launch_threshold
                                ),
                                force_all_in=self.force_all_in,
                                max_speed=max_speed,
                            )
                    for idx_c, sim in enumerate(sims):
                        if not sim.done:
                            sim.step(step_acts_all[idx_c])
                    dt_rep = time.perf_counter() - t_rep
                    est_replan_sec = max(est_replan_sec, dt_rep)
                    est_leaf_sec = max(0.025, est_replan_sec * (0.85 if use_sym else 0.50))
                else:
                    for sim in sims:
                        if not sim.done:
                            sim.step([heuristic_moves(sim, pid_cur) for pid_cur in range(n_players)])
            else:
                for sim in sims:
                    if not sim.done:
                        sim.step(empty_acts)

        # Batched leaf value evaluation across all non-terminal branches
        leaf_encs: List[EncodedPlanetState] = []
        leaf_slots: List[Tuple[int, int]] = []  # (cand_idx, +1 for hero or -1 for opp)
        leaf_enc_by_cand: Dict[int, List[Tuple[EncodedPlanetState, int]]] = {}
        nn_val_sum = np.zeros((k_cand,), dtype=np.float64)
        nn_val_cnt = np.zeros((k_cand,), dtype=np.int32)

        for idx_c, sim in enumerate(sims):
            if sim.done:
                continue
            obs_leaf = sim.to_observation(player=player)
            sh_leaf = build_shared_planet_physics(obs_leaf, max_speed=max_speed)
            hero_enc = extract_planet_features(
                obs_leaf,
                player=player,
                max_speed=max_speed,
                feature_version=self.feature_version,
                shared_physics=sh_leaf,
            )
            leaf_encs.append(hero_enc)
            leaf_slots.append((idx_c, 1))
            leaf_enc_by_cand[idx_c] = [(hero_enc, 1)]
            if use_sym:
                opp_enc = extract_planet_features(
                    obs_leaf,
                    player=1 - player,
                    max_speed=max_speed,
                    feature_version=self.feature_version,
                    shared_physics=sh_leaf,
                )
                leaf_encs.append(opp_enc)
                leaf_slots.append((idx_c, -1))
                leaf_enc_by_cand[idx_c].append((opp_enc, -1))

        if leaf_encs:
            _, _, leaf_vals = self.predict_batch(leaf_encs, compute_frac=False, value_only=True)
            for (idx_c, sign), v_raw in zip(leaf_slots, leaf_vals):
                nn_val_sum[idx_c] += float(sign) * float(v_raw)
                nn_val_cnt[idx_c] += 1

        base_lp = log_priors[0]

        def _score_candidate(idx_c: int, nn_leaf_val: float) -> float:
            sim = sims[idx_c]
            if sim.done:
                win_pid = sim.winner()
                if win_pid == player:
                    v_total = 1.0
                elif win_pid < 0:
                    v_total = 0.0
                else:
                    v_total = -1.0
            else:
                if self.trajectory_val_blend > 0.0 and traj_weight[idx_c] > 0.0:
                    t_avg = float(traj_val_wsum[idx_c] / traj_weight[idx_c])
                    nn_eff = (1.0 - self.trajectory_val_blend) * nn_leaf_val + self.trajectory_val_blend * t_avg
                else:
                    nn_eff = nn_leaf_val
                if self.neural_value_weight >= 0.999:
                    v_total = nn_eff
                else:
                    h_val = float(evaluate_state(sim, player))
                    v_total = self.neural_value_weight * nn_eff + (1.0 - self.neural_value_weight) * h_val
            return v_total + self.prior_bonus * (log_priors[idx_c] - base_lp)

        stage1_scores = [
            _score_candidate(idx_c, float(nn_val_sum[idx_c] / max(1, int(nn_val_cnt[idx_c]))))
            for idx_c in range(k_cand)
        ]

        # Optional Stage-2: 4x-rotation symmetric TTA value refinement on top-K non-terminal finalists
        if self.leaf_rot_tta_top_k >= 2 and leaf_enc_by_cand:
            alive_order = sorted(leaf_enc_by_cand.keys(), key=lambda i: stage1_scores[i], reverse=True)
            finalists = alive_order[: self.leaf_rot_tta_top_k]
            if 0 in leaf_enc_by_cand and 0 not in finalists:
                finalists[-1] = 0
            # Evaluate 3 additional rotations (k=1,2,3) reusing k=0 already computed in Stage 1
            tta_pf: List[np.ndarray] = []
            tta_tf: List[np.ndarray] = []
            tta_ef: List[np.ndarray] = []
            tta_le: List[np.ndarray] = []
            tta_ns: List[int] = []
            tta_slots: List[Tuple[int, int]] = []
            for idx_c in finalists:
                for l_enc, sign in leaf_enc_by_cand[idx_c]:
                    n_l = len(l_enc.planet_ids)
                    if n_l == 0:
                        continue
                    tf_base = l_enc.timeline_feats[..., : self.timeline_dim]
                    for k_rot in (1, 2, 3):
                        tta_pf.append(apply_d4_planet_feats(l_enc.planet_feats, k_rot=k_rot, flip_x=False))
                        tta_tf.append(apply_d4_timeline_feats(tf_base, k_rot=k_rot, flip_x=False))
                        tta_ef.append(l_enc.edge_feats)
                        tta_le.append(l_enc.legal_edges)
                        tta_ns.append(n_l)
                        tta_slots.append((idx_c, sign))
            # Check remaining budget before running Stage-2 TTA batch
            est_tta_sec = est_leaf_sec * (len(tta_slots) / max(1, len(leaf_slots)))
            if tta_slots and (time.perf_counter() + est_tta_sec < deadline + 0.02):
                max_nl = max(tta_ns)
                b_tta = len(tta_slots)
                if all(x == max_nl for x in tta_ns):
                    pf_t = torch.from_numpy(np.stack(tta_pf, axis=0)).to(self.device)
                    tf_t = torch.from_numpy(np.stack(tta_tf, axis=0)).to(self.device)
                    ef_t = torch.from_numpy(np.stack(tta_ef, axis=0)).to(self.device)
                    le_t = torch.from_numpy(np.stack(tta_le, axis=0)).to(self.device)
                    pm_t = torch.ones((b_tta, max_nl), dtype=torch.bool, device=self.device)
                else:
                    pf_np = np.zeros((b_tta, max_nl, PLANET_FEAT_DIM), dtype=np.float32)
                    tf_np = np.zeros((b_tta, max_nl, TIMELINE_HORIZON, self.timeline_dim), dtype=np.float32)
                    ef_np = np.zeros((b_tta, max_nl, max_nl, EDGE_FEAT_DIM), dtype=np.float32)
                    le_np = np.zeros((b_tta, max_nl, max_nl), dtype=bool)
                    pm_np = np.zeros((b_tta, max_nl), dtype=bool)
                    for ib, n_l in enumerate(tta_ns):
                        pf_np[ib, :n_l] = tta_pf[ib]
                        tf_np[ib, :n_l] = tta_tf[ib]
                        ef_np[ib, :n_l, :n_l] = tta_ef[ib]
                        le_np[ib, :n_l, :n_l] = tta_le[ib]
                        pm_np[ib, :n_l] = True
                    pf_t = torch.from_numpy(pf_np).to(self.device)
                    tf_t = torch.from_numpy(tf_np).to(self.device)
                    ef_t = torch.from_numpy(ef_np).to(self.device)
                    le_t = torch.from_numpy(le_np).to(self.device)
                    pm_t = torch.from_numpy(pm_np).to(self.device)
                _, _, tta_vals = self.model(
                    pf_t, tf_t, ef_t, pm_t, legal_edges=le_t, compute_frac=False, value_only=True
                )
                for (idx_c, sign), v_raw in zip(tta_slots, tta_vals.cpu().numpy()):
                    nn_val_sum[idx_c] += float(sign) * float(v_raw)
                    nn_val_cnt[idx_c] += 1
                for idx_c in finalists:
                    stage1_scores[idx_c] = _score_candidate(
                        idx_c, float(nn_val_sum[idx_c] / max(1, int(nn_val_cnt[idx_c])))
                    )

        best_idx = int(np.argmax(stage1_scores))
        return candidates[best_idx]


class EnsemblePlanetTransformerAgent:
    """Multi-checkpoint probability & value ensemble of ``PlanetTransformerAgent`` models.

    Supports heterogeneous model sizes (e.g., 1.89M + 1.89M + 0.59M) and averages
    masked per-planet probabilities, launch fractions, and global values before decoding
    with ``_decode_plan`` (including top-k blocked-shot fallback).
    """

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        *,
        weights: Optional[Sequence[float]] = None,
        launch_threshold: float = 0.50,
        use_rot_tta: bool = True,
        use_d4_tta: bool = False,
        force_all_in: bool = False,
        decoupled_logit: bool = False,
        device: str = "cpu",
    ) -> None:
        paths = [Path(p) for p in checkpoint_paths]
        if not paths:
            raise ValueError("EnsemblePlanetTransformerAgent requires at least one checkpoint.")
        self.agents = [
            PlanetTransformerAgent(
                checkpoint_path=p,
                launch_threshold=launch_threshold,
                use_rot_tta=use_rot_tta,
                use_d4_tta=use_d4_tta,
                force_all_in=force_all_in,
                device=device,
            )
            for p in paths
        ]
        if weights is None:
            raw_w = [1.0] * len(self.agents)
        else:
            if len(weights) != len(self.agents):
                raise ValueError("weights length must match checkpoint_paths length.")
            raw_w = [max(0.0, float(w)) for w in weights]
        total_w = sum(raw_w)
        if total_w <= 0.0:
            raise ValueError("weights must sum to a positive number.")
        self.weights = [w / total_w for w in raw_w]
        self.launch_threshold = float(launch_threshold)
        self.force_all_in = bool(force_all_in)
        self.decoupled_logit = bool(decoupled_logit)
        self.feature_version = max(a.feature_version for a in self.agents)

    @torch.no_grad()
    def predict_probs_and_value(
        self,
        enc: EncodedPlanetState,
        *,
        use_rot_tta: Optional[bool] = None,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = len(enc.planet_ids)
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0.0
        fracs_acc = np.zeros((n, n), dtype=np.float64)
        val_acc = 0.0
        if self.decoupled_logit:
            legal = enc.legal_edges
            has_legal = np.any(legal, axis=-1, keepdims=True)
            launch_logit_acc = np.zeros((n, 1), dtype=np.float64)
            log_tgt_acc = np.zeros((n, n), dtype=np.float64)
            for agent, w in zip(self.agents, self.weights):
                p, f, v = agent.predict_probs_and_value(enc, use_rot_tta=use_rot_tta, use_d4_tta=use_d4_tta)
                p_hold_k = np.clip(p[:, :1].astype(np.float64), 1e-6, 1.0 - 1e-6)
                p_launch_k = 1.0 - p_hold_k
                launch_logit_acc += w * (np.log(p_launch_k) - np.log(p_hold_k))
                cond_k = np.where(
                    legal,
                    np.clip(p[:, 1:].astype(np.float64) / np.maximum(p_launch_k, 1e-7), 1e-8, 1.0),
                    1.0,
                )
                log_tgt_acc += w * np.log(cond_k)
                fracs_acc += w * f
                val_acc += w * v
            p_launch = 1.0 / (1.0 + np.exp(-np.clip(launch_logit_acc, -20.0, 20.0)))
            p_hold = 1.0 - p_launch
            masked_tgt = np.where(legal, log_tgt_acc, -1e9)
            max_t = np.max(masked_tgt, axis=-1, keepdims=True)
            exp_t = np.where(legal, np.exp(masked_tgt - max_t), 0.0)
            sum_t = np.sum(exp_t, axis=-1, keepdims=True)
            p_tgt = np.where(has_legal & (sum_t > 0.0), exp_t / np.maximum(sum_t, 1e-12), 0.0)
            probs = np.concatenate([p_hold, p_launch * p_tgt], axis=-1).astype(np.float32)
            return probs, fracs_acc.astype(np.float32), float(val_acc)
        probs_acc = np.zeros((n, n + 1), dtype=np.float64)
        for agent, w in zip(self.agents, self.weights):
            p, f, v = agent.predict_probs_and_value(enc, use_rot_tta=use_rot_tta, use_d4_tta=use_d4_tta)
            probs_acc += w * p
            fracs_acc += w * f
            val_acc += w * v
        return probs_acc.astype(np.float32), fracs_acc.astype(np.float32), float(val_acc)

    @torch.no_grad()
    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        max_speed = float(_obs_get(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config else DEFAULT_SHIP_SPEED
        enc = extract_planet_features(obs, max_speed=max_speed, feature_version=self.feature_version)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return []
        probs, mean_fracs, _ = self.predict_probs_and_value(enc)
        return self.agents[0]._decode_plan(
            enc,
            probs,
            mean_fracs,
            launch_threshold=self.launch_threshold,
            force_all_in=self.force_all_in,
            max_speed=max_speed,
        )


def create_planet_transformer_agent(
    checkpoint_path: str | Path,
    *,
    launch_threshold: float = 0.50,
    use_rot_tta: bool = True,
    use_d4_tta: bool = False,
    force_all_in: bool = False,
    reinforce_threshold: Optional[float] = None,
    surgical_neutral_cap: bool = False,
    anti_bounce_guard: bool = False,
    decoupled_logit_tta: Optional[bool] = None,
):
    """Factory returning a Kaggle ``agent(obs, config)`` callable."""
    agent_obj = PlanetTransformerAgent(
        checkpoint_path=checkpoint_path,
        launch_threshold=launch_threshold,
        use_rot_tta=use_rot_tta,
        use_d4_tta=use_d4_tta,
        force_all_in=force_all_in,
        reinforce_threshold=reinforce_threshold,
        surgical_neutral_cap=surgical_neutral_cap,
        anti_bounce_guard=anti_bounce_guard,
        decoupled_logit_tta=decoupled_logit_tta,
    )

    def _kaggle_agent(obs: Any, config: Any = None) -> List[List[float]]:
        return agent_obj.act(obs, config)

    return _kaggle_agent


def create_search_planet_transformer_agent(
    checkpoint_path: str | Path,
    *,
    launch_threshold: float = 0.50,
    use_rot_tta: bool = True,
    use_d4_tta: bool = False,
    force_all_in: bool = False,
    neural_mid_replan: bool = False,
    rollout_horizon: int = 10,
    time_budget_sec: float = 0.28,
    neural_value_weight: float = 0.85,
    replan_steps: Optional[Sequence[int]] = None,
    max_candidates: int = 8,
    prior_bonus: float = 0.006,
    symmetric_value: bool = True,
    opp_rot_tta: bool = False,
    opp_launch_threshold: Optional[float] = None,
    trajectory_val_blend: float = 0.0,
    leaf_rot_tta_top_k: int = 0,
):
    """Factory returning a batched value-guided lookahead ``PlanetTimelineTransformer`` Kaggle agent."""
    agent_obj = SearchPlanetTransformerAgent(
        checkpoint_path=checkpoint_path,
        launch_threshold=launch_threshold,
        use_rot_tta=use_rot_tta,
        use_d4_tta=use_d4_tta,
        force_all_in=force_all_in,
        neural_mid_replan=neural_mid_replan,
        rollout_horizon=rollout_horizon,
        time_budget_sec=time_budget_sec,
        neural_value_weight=neural_value_weight,
        replan_steps=replan_steps,
        max_candidates=max_candidates,
        prior_bonus=prior_bonus,
        symmetric_value=symmetric_value,
        opp_rot_tta=opp_rot_tta,
        opp_launch_threshold=opp_launch_threshold,
        trajectory_val_blend=trajectory_val_blend,
        leaf_rot_tta_top_k=leaf_rot_tta_top_k,
    )

    def _kaggle_agent(obs: Any, config: Any = None) -> List[List[float]]:
        return agent_obj.act(obs, config)

    return _kaggle_agent


def create_ensemble_planet_transformer_agent(
    checkpoint_paths: Sequence[str | Path],
    *,
    weights: Optional[Sequence[float]] = None,
    launch_threshold: float = 0.50,
    use_rot_tta: bool = True,
    use_d4_tta: bool = False,
    force_all_in: bool = False,
):
    """Factory returning a multi-checkpoint ensemble ``PlanetTimelineTransformer`` Kaggle agent."""
    agent_obj = EnsemblePlanetTransformerAgent(
        checkpoint_paths=checkpoint_paths,
        weights=weights,
        launch_threshold=launch_threshold,
        use_rot_tta=use_rot_tta,
        use_d4_tta=use_d4_tta,
        force_all_in=force_all_in,
    )

    def _kaggle_agent(obs: Any, config: Any = None) -> List[List[float]]:
        return agent_obj.act(obs, config)

    return _kaggle_agent
