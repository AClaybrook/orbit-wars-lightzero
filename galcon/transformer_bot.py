"""Cross-pollinated PlanetTimelineTransformer agent for Galcon & ``gbots``.

Directly reuses ``PlanetTimelineTransformer`` from ``orbit_wars_bot.planet_transformer``
with scale-invariant continuous coordinates, exact 20-step arrival timelines, 100% D4-invariant
pairwise ETA/ROI edges, 8x D4 dihedral TTA, and:

- **Gen 2 ``GalconPlanetTransformer``** (~1.28M params): wraps ``PlanetTimelineTransformer``
  with an auxiliary ``t+10`` future-state world-model head (ownership + ships) and a native
  D4-invariant mid-flight ``/REDIR`` head over flying friendly fleets.
- **Batched AlphaZero Lookahead Search** (``GalconSearchTransformerAgent``): evaluates diverse
  threshold, fraction-scale, single-planet flip, rush-guard, and redirect candidates in a single
  batched neural leaf forward pass (~30-45 ms/tick on 1 CPU thread).
- **Full v1 backward compatibility**: seamlessly loads v1 (``planet_dim=18, edge_dim=10``)
  and Gen 2 (``planet_dim=22, edge_dim=12``) checkpoints side by side for head-to-head evals.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from galcon.fast_c import HAS_FAST_C, extract_features_fast_c
from galcon.gen5_model import GalconGen5Transformer, build_galcon_gen5_transformer
from galcon.sim import (
    PRODUCTION_DIVISOR,
    GalconSim,
    Order,
    Planet,
    RedirOrder,
    SendOrder,
    evaluate_galcon_state,
    project_all_planet_futures,
    project_planet_future,
)
from orbit_wars_bot.planet_transformer import PlanetTimelineTransformer

GALCON_MAX_PLANETS = 32
GALCON_TIMELINE_HORIZON = 20
GALCON_PLANET_DIM = 18
GALCON_TIMELINE_DIM = 8
GALCON_EDGE_DIM = 10

# Gen 2 enriched D4-invariant feature dimensions
GALCON_PLANET_DIM_V2 = 22
GALCON_EDGE_DIM_V2 = 12
GALCON_REDIR_DIM = 10


@dataclass
class EncodedGalconState:
    player: int
    planet_ids: List[int]
    planets: List[Planet]
    planet_feats: np.ndarray  # (N, 18 or 22)
    timeline_feats: np.ndarray  # (N, 20, 8)
    edge_feats: np.ndarray  # (N, N, 10 or 12)
    legal_edges: np.ndarray  # (N, N) bool
    owned_mask: np.ndarray  # (N,) bool
    safe_surplus: np.ndarray  # (N,) float32
    feature_version: int = 1
    futures: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None


@dataclass
class EncodedRedirState:
    fleet_ids: List[int]
    cur_dst_idx: np.ndarray  # (F,) int64
    redir_feats: np.ndarray  # (F, N, GALCON_REDIR_DIM) float32


def apply_galcon_d4(planet_feats: np.ndarray, k_rot: int = 0, flip_x: bool = False) -> np.ndarray:
    """Exact 8x D4 dihedral symmetry on normalized ``(x, y)`` (cols 0-1).

    Because all timeline, pairwise edge, and fleet redirect features are 100% D4-invariant,
    transforming only columns 0 and 1 gives exact 8x data augmentation and 8x test-time
    augmentation on any map aspect ratio and any number of planets.
    """
    out = planet_feats.copy()
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    if flip_x:
        x = -x
    k = k_rot % 4
    if k == 1:
        x, y = -y, x
    elif k == 2:
        x, y = -x, -y
    elif k == 3:
        x, y = y, -x
    out[..., 0] = x
    out[..., 1] = y
    return out


def canonicalize_galcon_d4(planet_feats: np.ndarray) -> np.ndarray:
    """Map ``planet_feats`` (shape ``(N, F)``) to its unique canonical D4 octant in O(N) time.

    Because the 8 D4 transforms form a closed finite group, selecting the unique orbit member
    minimizing a deterministic coordinate/ownership key guarantees exact 8x D4 invariance
    in a single ``1x`` forward pass (~5x faster than 8x D4 TTA on CPU).
    """
    if planet_feats.ndim != 2 or planet_feats.shape[0] == 0:
        return planet_feats.copy()
    is_me = planet_feats[:, 2]
    is_en = planet_feats[:, 3]
    ships = planet_feats[:, 5]
    prod = planet_feats[:, 6]
    base_w = 0.25 + prod + ships
    w_me = is_me * base_w
    w_en = is_en * base_w
    sum_me = float(w_me.sum())
    sum_en = float(w_en.sum())

    x0 = planet_feats[:, 0]
    y0 = planet_feats[:, 1]
    mx0 = float((w_me * x0).sum() / sum_me) if sum_me > 1e-6 else 0.0
    my0 = float((w_me * y0).sum() / sum_me) if sum_me > 1e-6 else 0.0
    ex0 = float((w_en * x0).sum() / sum_en) if sum_en > 1e-6 else 0.0
    ey0 = float((w_en * y0).sum() / sum_en) if sum_en > 1e-6 else 0.0
    vx0, vy0 = mx0 - ex0, my0 - ey0
    if math.hypot(vx0, vy0) < 1e-4:
        vx0, vy0 = mx0, my0

    # Evaluate all 8 D4 transforms on the 2D centroid vector first (O(1)), avoiding 8x array copies
    # and computing the coordinate tie-break lazily only on exact oct_key ties (<0.1% of states).
    best_oct = None
    tied: List[Tuple[bool, int]] = []
    for flip_x in (False, True):
        fx = -vx0 if flip_x else vx0
        fy = vy0
        for k_rot in range(4):
            if k_rot == 0:
                vx, vy = fx, fy
            elif k_rot == 1:
                vx, vy = -fy, fx
            elif k_rot == 2:
                vx, vy = -fx, -fy
            else:
                vx, vy = fy, -fx
            oct_key = (
                0 if vx <= 1e-5 else 1,
                0 if vy <= 1e-5 else 1,
                0 if abs(vx) >= abs(vy) - 1e-5 else 1,
                round(vx, 4),
                round(vy, 4),
            )
            if best_oct is None or oct_key < best_oct:
                best_oct = oct_key
                tied = [(flip_x, k_rot)]
            elif oct_key == best_oct:
                tied.append((flip_x, k_rot))

    if len(tied) == 1:
        flip_x, k_rot = tied[0]
        return apply_galcon_d4(planet_feats, k_rot=k_rot, flip_x=flip_x)

    best_tie = None
    best_arr = None
    n_p = planet_feats.shape[0]
    for flip_x, k_rot in tied:
        cand = apply_galcon_d4(planet_feats, k_rot=k_rot, flip_x=flip_x)
        cx = cand[:, 0]
        cy = cand[:, 1]
        tie_key = tuple(
            sorted(
                (round(float(cx[i]), 3), round(float(cy[i]), 3), round(float(is_me[i] - is_en[i]), 1), round(float(prod[i]), 2))
                for i in range(n_p)
            )
        )
        if best_tie is None or tie_key < best_tie:
            best_tie = tie_key
            best_arr = cand
    assert best_arr is not None
    return best_arr


def extract_galcon_features(
    sim: GalconSim,
    player: int,
    *,
    feature_version: int = 1,
    _use_fast_c: bool = True,
) -> EncodedGalconState:
    """Extract lossless, scale-invariant entity + arrival-timeline + edge features for Galcon."""
    planets = sorted(sim.planets.values(), key=lambda p: p.id)[:GALCON_MAX_PLANETS]
    n = len(planets)
    planet_ids = [p.id for p in planets]
    p_dim = GALCON_PLANET_DIM_V2 if feature_version >= 2 else GALCON_PLANET_DIM
    e_dim = GALCON_EDGE_DIM_V2 if feature_version >= 2 else GALCON_EDGE_DIM

    if n == 0:
        return EncodedGalconState(
            player=player,
            planet_ids=[],
            planets=[],
            planet_feats=np.zeros((0, p_dim), dtype=np.float32),
            timeline_feats=np.zeros((0, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32),
            edge_feats=np.zeros((0, 0, e_dim), dtype=np.float32),
            legal_edges=np.zeros((0, 0), dtype=bool),
            owned_mask=np.zeros((0,), dtype=bool),
            safe_surplus=np.zeros((0,), dtype=np.float32),
            feature_version=feature_version,
            futures=[],
        )

    if _use_fast_c and HAS_FAST_C and len(sim.planets) <= GALCON_MAX_PLANETS:
        res_c = extract_features_fast_c(
            sim, planets, player, GALCON_TIMELINE_HORIZON, feature_version, PRODUCTION_DIVISOR
        )
        if res_c is not None:
            my_arrs, en_arrs, po_all, ps_all, pf, tf, ef, legal, owned_mask, safe_surplus = res_c
            futures = [(my_arrs[i], en_arrs[i], po_all[i], ps_all[i]) for i in range(n)]
            return EncodedGalconState(
                player=player,
                planet_ids=planet_ids,
                planets=planets,
                planet_feats=pf,
                timeline_feats=tf,
                edge_feats=ef,
                legal_edges=legal,
                owned_mask=owned_mask,
                safe_surplus=safe_surplus,
                feature_version=feature_version,
                futures=futures,
            )

    # Scale-invariant centering works identically on 200x125, 400x240, or 600x400 gbots maps
    xs64 = np.array([p.x for p in planets], dtype=np.float64)
    ys64 = np.array([p.y for p in planets], dtype=np.float64)
    rs64 = np.array([p.radius for p in planets], dtype=np.float64)
    ships64 = np.array([p.ships for p in planets], dtype=np.float64)
    prods64 = np.array([p.production for p in planets], dtype=np.float64)
    owners_raw = np.array([p.owner for p in planets], dtype=np.int32)

    cx = 0.5 * (float(xs64.min()) + float(xs64.max()))
    cy = 0.5 * (float(ys64.min()) + float(ys64.max()))
    half_span = max(50.0, 0.5 * max(float(xs64.max() - xs64.min()), float(ys64.max() - ys64.min())))

    opp = 2 if player == 1 else 1
    my_tot_ships = sim.total_ships(player)
    en_tot_ships = sim.total_ships(opp)
    my_tot_prod = sim.total_production(player)
    en_tot_prod = sim.total_production(opp)
    ship_ratio = (my_tot_ships - en_tot_ships) / max(30.0, my_tot_ships + en_tot_ships)
    prod_ratio = (my_tot_prod - en_tot_prod) / max(40.0, my_tot_prod + en_tot_prod)
    step_prog = min(1.0, float(sim.tick) / max(1.0, float(sim.max_ticks)))

    my_arrs, en_arrs, po_all, ps_all = project_all_planet_futures(
        sim, planets, player, horizon=GALCON_TIMELINE_HORIZON
    )
    futures = [(my_arrs[i], en_arrs[i], po_all[i], ps_all[i]) for i in range(n)]

    # Vectorized exact pairwise ETA matrix (N, N) matching sim.planet_eta_ticks bit-for-bit
    step_dist = max(1e-6, sim.fleet_speed * sim.tick_dt)
    dx = xs64[None, :] - xs64[:, None]
    dy = ys64[None, :] - ys64[:, None]
    dist = np.maximum(0.0, np.hypot(dx, dy) - rs64[:, None] - rs64[None, :])
    etas = np.maximum(1.0, np.ceil(dist / step_dist)).astype(np.float32)
    np.fill_diagonal(etas, 0.0)

    is_mine_b = owners_raw == player
    is_en_b = (owners_raw != 0) & (owners_raw != player)
    is_neu_b = owners_raw == 0
    owned_mask = is_mine_b & (ships64 >= 2.0)

    # Vectorized safe_surplus and threat signals
    lost_mask = po_all <= 0.0  # (N, H)
    any_lost = lost_mask.any(axis=1)
    first_lost = np.argmax(lost_mask, axis=1)  # valid wherever any_lost is True
    min_hold = ps_all.min(axis=1)
    safe_surplus = np.where(
        is_mine_b & (~any_lost),
        np.maximum(0.0, np.minimum(ships64.astype(np.float32), min_hold)),
        0.0,
    ).astype(np.float32)
    is_threatened = (is_mine_b & any_lost).astype(np.float32)
    threat_urgency = np.where(
        is_mine_b & any_lost,
        1.0 / (1.0 + 0.25 * (first_lost.astype(np.float32) + 1.0)),
        0.0,
    ).astype(np.float32)
    lost_ps = ps_all[np.arange(n), first_lost]
    deficit_log = np.where(
        is_mine_b & any_lost,
        np.log1p(np.maximum(0.0, lost_ps + 2.0)) / 5.5,
        0.0,
    ).astype(np.float32)

    # Nearest enemy / friendly ETAs
    if np.any(is_en_b):
        en_idx = np.flatnonzero(is_en_b)
        min_en_eta = etas[:, en_idx].min(axis=1).astype(np.float32)
    else:
        en_idx = np.empty((0,), dtype=np.int64)
        min_en_eta = np.full((n,), 40.0, dtype=np.float32)

    if np.any(is_mine_b):
        my_idx = np.flatnonzero(is_mine_b)
        if len(my_idx) == 1:
            min_my_eta = etas[:, my_idx[0]].astype(np.float32).copy()
            min_my_eta[my_idx[0]] = 40.0
        else:
            etas_my = etas[:, my_idx].copy()
            for col_k, p_i in enumerate(my_idx):
                etas_my[p_i, col_k] = 1e9
            min_my_eta = etas_my.min(axis=1).astype(np.float32)
    else:
        min_my_eta = np.full((n,), 40.0, dtype=np.float32)

    frontline = (min_my_eta - min_en_eta) / np.maximum(10.0, min_my_eta + min_en_eta)

    pf = np.empty((n, p_dim), dtype=np.float32)
    pf[:, 0] = ((xs64 - cx) / half_span).astype(np.float32)
    pf[:, 1] = ((ys64 - cy) / half_span).astype(np.float32)
    pf[:, 2] = is_mine_b.astype(np.float32)
    pf[:, 3] = is_en_b.astype(np.float32)
    pf[:, 4] = is_neu_b.astype(np.float32)
    pf[:, 5] = (np.log1p(np.maximum(0.0, ships64)) / 5.5).astype(np.float32)
    pf[:, 6] = (prods64 / 100.0).astype(np.float32)
    pf[:, 7] = (rs64 / 35.0).astype(np.float32)
    pf[:, 8] = (np.log1p(safe_surplus) / 5.5).astype(np.float32)
    pf[:, 9] = (np.log1p(my_arrs.sum(axis=1)) / 5.5).astype(np.float32)
    pf[:, 10] = (np.log1p(en_arrs.sum(axis=1)) / 5.5).astype(np.float32)
    pf[:, 11] = (po_all[:, -1] * (np.log1p(np.maximum(0.0, ps_all[:, -1])) / 5.5)).astype(np.float32)
    pf[:, 12] = np.minimum(1.0, min_en_eta / 35.0)
    pf[:, 13] = np.minimum(1.0, min_my_eta / 35.0)
    pf[:, 14] = frontline.astype(np.float32)
    pf[:, 15] = float(ship_ratio)
    pf[:, 16] = float(prod_ratio)
    pf[:, 17] = float(step_prog)

    if feature_version >= 2:
        rush_threat = np.zeros((n,), dtype=np.float32)
        if len(en_idx) > 0 and np.any(is_mine_b):
            # Nearest enemy planet j minimizing etas[j, i]
            sub_etas = etas[en_idx, :]  # (E, N)
            best_e_loc = np.argmin(sub_etas, axis=0)  # (N,)
            nearest_en = en_idx[best_e_loc]
            eta_e = sub_etas[best_e_loc, np.arange(n)]
            growth_dur = (prods64 / PRODUCTION_DIVISOR) * sim.tick_dt * eta_e
            rt = (ships64[nearest_en] * 0.85 - ships64 - growth_dur) / 60.0
            rush_threat = np.where(is_mine_b, np.clip(rt, 0.0, 2.0), 0.0).astype(np.float32)
        pf[:, 18] = is_threatened
        pf[:, 19] = threat_urgency
        pf[:, 20] = deficit_log
        pf[:, 21] = rush_threat

    # Vectorized timeline features (N, H, 8)
    tf = np.empty((n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32)
    tf[:, :, 0] = np.log1p(my_arrs) / 5.0
    tf[:, :, 1] = np.log1p(en_arrs) / 5.0
    tf[:, :, 2] = (po_all > 0.0).astype(np.float32)
    tf[:, :, 3] = (po_all < 0.0).astype(np.float32)
    tf[:, :, 4] = (po_all == 0.0).astype(np.float32)
    tf[:, :, 5] = np.log1p(np.maximum(0.0, ps_all)) / 5.5
    tf[:, 0, 6] = 0.0
    tf[:, 1:, 6] = (po_all[:, 1:] != po_all[:, :-1]).astype(np.float32)
    tf[:, :, 7] = (np.arange(GALCON_TIMELINE_HORIZON, dtype=np.float32) / float(GALCON_TIMELINE_HORIZON))[None, :]

    # Vectorized D4-invariant edge features (N, N, e_dim)
    legal = ~np.eye(n, dtype=bool)
    t_idx = np.clip(etas.astype(np.int64) - 1, 0, GALCON_TIMELINE_HORIZON - 1)  # (N, N)
    j_grid = np.broadcast_to(np.arange(n)[None, :], (n, n))
    owner_eta = po_all[j_grid, t_idx]  # (N, N)
    ships_eta = ps_all[j_grid, t_idx]  # (N, N)
    prod_rate_j = (prods64 / PRODUCTION_DIVISOR).astype(np.float32)[None, :]  # (1, N)
    extra = np.where(
        owner_eta < 0.0,
        prod_rate_j * float(sim.tick_dt) * np.maximum(0.0, etas - float(GALCON_TIMELINE_HORIZON)),
        0.0,
    ).astype(np.float32)
    need = ships_eta + extra + 1.5
    src_ships = ships64.astype(np.float32)[:, None]  # (N, 1)
    src_safe = safe_surplus[:, None]  # (N, 1)
    can_capture = ((owner_eta <= 0.0) & (src_ships > need)).astype(np.float32)
    safe_capture = ((owner_eta <= 0.0) & (src_safe > need)).astype(np.float32)
    steal_flag = ((is_neu_b[None, :]) & (owner_eta < 0.0)).astype(np.float32)
    defend_flag = ((is_mine_b[None, :]) & (owner_eta <= 0.0)).astype(np.float32)
    roi = (prod_rate_j * 20.0) / np.maximum(3.0, need + 0.45 * etas * np.maximum(0.3, prod_rate_j))

    ef = np.zeros((n, n, e_dim), dtype=np.float32)
    ef[:, :, 0] = np.minimum(1.0, etas / 35.0)
    ef[:, :, 1] = 1.0 / (1.0 + 0.12 * etas)
    ef[:, :, 2] = np.minimum(2.0, src_ships / np.maximum(1.0, need))
    ef[:, :, 3] = can_capture
    ef[:, :, 4] = safe_capture
    ef[:, :, 5] = steal_flag
    ef[:, :, 6] = defend_flag
    ef[:, :, 7] = np.minimum(2.0, roi)
    ef[:, :, 8] = owner_eta
    ef[:, :, 9] = pf[:, 12:13] - pf[None, :, 12]
    if feature_version >= 2:
        ef[:, :, 10] = np.minimum(1.0, need / np.maximum(1.0, src_ships))
        ef[:, :, 11] = ((is_en_b[None, :]) & (prods64[None, :] >= 70.0) & (src_ships > need)).astype(np.float32)
    ef[~legal] = 0.0

    return EncodedGalconState(
        player=player,
        planet_ids=planet_ids,
        planets=planets,
        planet_feats=pf,
        timeline_feats=tf,
        edge_feats=ef,
        legal_edges=legal,
        owned_mask=owned_mask,
        safe_surplus=safe_surplus,
        feature_version=feature_version,
        futures=futures,
    )


def extract_galcon_redir_features(sim: GalconSim, enc: EncodedGalconState) -> EncodedRedirState:
    """Extract 100% D4-invariant fleet-to-planet features for all flying friendly fleets."""
    n = len(enc.planet_ids)
    if n == 0 or not sim.fleets or not sim.allow_redirect:
        return EncodedRedirState(
            fleet_ids=[],
            cur_dst_idx=np.zeros((0,), dtype=np.int64),
            redir_feats=np.zeros((0, n, GALCON_REDIR_DIM), dtype=np.float32),
        )

    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    my_fleets = sorted(
        (fl for fl in sim.fleets.values() if fl.owner == enc.player and fl.target in id_to_idx),
        key=lambda f: f.id,
    )
    if not my_fleets:
        return EncodedRedirState(
            fleet_ids=[],
            cur_dst_idx=np.zeros((0,), dtype=np.int64),
            redir_feats=np.zeros((0, n, GALCON_REDIR_DIM), dtype=np.float32),
        )

    if enc.futures:
        po_all = np.stack([f[2] for f in enc.futures], axis=0)
        ps_all = np.stack([f[3] for f in enc.futures], axis=0)
    else:
        _, _, po_all, ps_all = project_all_planet_futures(
            sim, enc.planets, enc.player, horizon=GALCON_TIMELINE_HORIZON
        )

    px = np.fromiter((p.x for p in enc.planets), dtype=np.float64, count=n)
    py = np.fromiter((p.y for p in enc.planets), dtype=np.float64, count=n)
    pr = np.fromiter((p.radius for p in enc.planets), dtype=np.float64, count=n)
    pprod = np.fromiter((p.production for p in enc.planets), dtype=np.float64, count=n)
    is_mine = np.fromiter((p.owner == enc.player for p in enc.planets), dtype=np.bool_, count=n)

    lost_mask = po_all <= 0.0
    any_lost = lost_mask.any(axis=1)
    first_lost = np.argmax(lost_mask, axis=1)
    threat_eta = np.where(is_mine & any_lost, first_lost + 1, -1).astype(np.int32)

    f_cnt = len(my_fleets)
    fleet_ids = [fl.id for fl in my_fleets]
    cur_dst_idx = np.fromiter((id_to_idx[fl.target] for fl in my_fleets), dtype=np.int64, count=f_cnt)
    fx = np.fromiter((fl.x for fl in my_fleets), dtype=np.float64, count=f_cnt)
    fy = np.fromiter((fl.y for fl in my_fleets), dtype=np.float64, count=f_cnt)
    fsh = np.fromiter((fl.ships for fl in my_fleets), dtype=np.float32, count=f_cnt)

    step_dist = max(1e-6, sim.fleet_speed * sim.tick_dt)
    dist = np.maximum(0.0, np.hypot(px[None, :] - fx[:, None], py[None, :] - fy[:, None]) - pr[None, :])
    eta_mat = np.maximum(1.0, np.ceil(dist / step_dist)).astype(np.float32)  # (F, N)
    t_mat = np.clip(eta_mat.astype(np.int64) - 1, 0, GALCON_TIMELINE_HORIZON - 1)  # (F, N)
    j_grid = np.broadcast_to(np.arange(n)[None, :], (f_cnt, n))
    owner_eta = po_all[j_grid, t_mat]  # (F, N)
    ships_eta = ps_all[j_grid, t_mat]  # (F, N)

    f_arange = np.arange(f_cnt)
    eta_cur = eta_mat[f_arange, cur_dst_idx]  # (F,)
    t_cur = np.clip(eta_cur.astype(np.int64) - 1, 0, GALCON_TIMELINE_HORIZON - 1)
    c_po_end = po_all[cur_dst_idx, -1]
    c_ps_end = ps_all[cur_dst_idx, -1]
    c_po_cur = po_all[cur_dst_idx, t_cur]
    c_ps_cur = ps_all[cur_dst_idx, t_cur]
    c_is_mine = is_mine[cur_dst_idx]
    wasted_friendly = c_is_mine & (c_po_end > 0.0) & (c_ps_end > fsh + 15.0)
    doomed_attack = (~c_is_mine) & (c_po_cur <= 0.0) & (fsh < c_ps_cur * 0.80)
    cur_wasted = (wasted_friendly | doomed_attack).astype(np.float32)[:, None]

    fsh_col = fsh[:, None]
    need_mat = ships_eta + 1.5
    prod_rate = (pprod / PRODUCTION_DIVISOR).astype(np.float32)[None, :]
    roi_mat = (prod_rate * 20.0) / np.maximum(3.0, need_mat + 0.45 * eta_mat * np.maximum(0.3, prod_rate))

    rf = np.empty((f_cnt, n, GALCON_REDIR_DIM), dtype=np.float32)
    rf[:, :, 0] = (j_grid == cur_dst_idx[:, None]).astype(np.float32)
    rf[:, :, 1] = np.minimum(1.0, eta_mat / 35.0)
    rf[:, :, 2] = np.tanh((eta_mat - eta_cur[:, None]) / 10.0)
    rf[:, :, 3] = (np.log1p(np.maximum(0.0, fsh)) / 5.5)[:, None]
    rf[:, :, 4] = np.minimum(2.0, fsh_col / np.maximum(1.0, need_mat))
    rf[:, :, 5] = ((owner_eta <= 0.0) & (fsh_col > need_mat)).astype(np.float32)
    rf[:, :, 6] = ((threat_eta[None, :] > 0) & (eta_mat <= threat_eta[None, :])).astype(np.float32)
    rf[:, :, 7] = cur_wasted
    rf[:, :, 8] = np.minimum(2.0, roi_mat)
    rf[:, :, 9] = owner_eta

    return EncodedRedirState(fleet_ids=fleet_ids, cur_dst_idx=cur_dst_idx, redir_feats=rf)


def compute_smart_redirects(
    sim: GalconSim,
    enc: EncodedGalconState,
    *,
    verify_rollout: bool = False,
    min_gain: float = 0.025,
) -> Tuple[List[RedirOrder], Dict[int, int]]:
    """Upgraded tactical/rollout-verified mid-flight ``/REDIR`` oracle.

    Handles three high-impact Galcon redirect scenarios:
    1. **Over-reinforcement rescue**: fleet heading to a friendly planet that is already
       comfortably safe -> pivot to a vulnerable enemy/neutral planet or threatened ally.
    2. **Doomed-attack rescue**: fleet heading to an enemy/neutral planet where it will bounce
       off without capturing -> divert to a winnable target or threatened ally instead of dying.
    3. **Emergency home/planet defense**: pull nearby non-essential fleets to save a threatened
       high-production friendly planet.

    Returns ``(redir_orders, target_idx_by_fleet_id)`` where ``target_idx_by_fleet_id`` gives
    the verified target planet index for EVERY friendly fleet (for supervising ``redir_head``).
    """
    n = len(enc.planet_ids)
    if n == 0 or not sim.fleets or not sim.allow_redirect:
        return [], {}

    player = enc.player
    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    futures = enc.futures or [
        project_planet_future(sim, p.id, player, horizon=GALCON_TIMELINE_HORIZON) for p in enc.planets
    ]

    threatened: Dict[int, Tuple[float, int]] = {}
    for j, p in enumerate(enc.planets):
        if p.owner == player:
            _, _, po, ps = futures[j]
            lost = next((t for t in range(GALCON_TIMELINE_HORIZON) if po[t] <= 0), -1)
            if lost >= 0:
                threatened[p.id] = (float(ps[lost]) + 2.0, lost + 1)

    base_val = None
    if verify_rollout:
        roll0 = sim.clone()
        for _ in range(12):
            if roll0.done:
                break
            roll0.step({})
        base_val = evaluate_galcon_state(roll0, player)

    redirs: List[RedirOrder] = []
    labels: Dict[int, int] = {}

    for fl in sorted(sim.fleets.values(), key=lambda f: f.id):
        if fl.owner != player or fl.target not in id_to_idx:
            continue
        cur_idx = id_to_idx[fl.target]
        labels[fl.id] = cur_idx
        dst = enc.planets[cur_idx]
        _, _, po, ps = futures[cur_idx]
        eta_cur = sim.eta_ticks(fl.x, fl.y, dst)
        t_cur = min(GALCON_TIMELINE_HORIZON - 1, max(0, eta_cur - 1))

        wasted_friendly = dst.owner == player and po[-1] > 0 and ps[-1] > fl.ships + 15.0
        doomed_attack = dst.owner != player and po[t_cur] <= 0 and fl.ships < ps[t_cur] * 0.78
        can_abandon_for_def = (dst.owner == 0 and po[t_cur] >= 0) or wasted_friendly or doomed_attack

        if not (wasted_friendly or doomed_attack or (threatened and can_abandon_for_def)):
            continue

        best_alt_idx: Optional[int] = None
        best_sc = 0.0
        for j, cand in enumerate(enc.planets):
            if j == cur_idx:
                continue
            eta = sim.eta_ticks(fl.x, fl.y, cand)
            if cand.id in threatened:
                deficit, urg_eta = threatened[cand.id]
                if eta <= urg_eta and fl.ships >= deficit * 0.45:
                    sc = (cand.production * 3.4) / max(1.0, float(eta))
                    if sc > best_sc:
                        best_sc = sc
                        best_alt_idx = j
            elif (wasted_friendly or doomed_attack) and cand.owner != player:
                _, _, cpo, cps = futures[j]
                t_idx = min(GALCON_TIMELINE_HORIZON - 1, max(0, eta - 1))
                if cpo[t_idx] <= 0 and fl.ships > cps[t_idx] + 1.5:
                    mult = 2.4 if cand.owner != 0 else 1.15
                    sc = (cand.production * mult) / max(2.0, float(cps[t_idx]) + eta * 1.4)
                    if sc > best_sc:
                        best_sc = sc
                        best_alt_idx = j

        if best_alt_idx is not None:
            alt_pid = enc.planet_ids[best_alt_idx]
            if verify_rollout and base_val is not None:
                roll1 = sim.clone()
                roll1.step({player: [RedirOrder(fl.id, alt_pid)]})
                for _ in range(11):
                    if roll1.done:
                        break
                    roll1.step({})
                alt_val = evaluate_galcon_state(roll1, player)
                if alt_val >= base_val + min_gain:
                    redirs.append(RedirOrder(fl.id, alt_pid))
                    labels[fl.id] = best_alt_idx
            else:
                redirs.append(RedirOrder(fl.id, alt_pid))
                labels[fl.id] = best_alt_idx

    return redirs, labels


def pad_galcon_state(enc: EncodedGalconState) -> Dict[str, np.ndarray]:
    """Pad variable-N Galcon features to ``GALCON_MAX_PLANETS`` for batched training."""
    n = len(enc.planet_ids)
    p_dim = enc.planet_feats.shape[-1] if enc.planet_feats.ndim == 2 else (
        GALCON_PLANET_DIM_V2 if enc.feature_version >= 2 else GALCON_PLANET_DIM
    )
    e_dim = enc.edge_feats.shape[-1] if enc.edge_feats.ndim == 3 else (
        GALCON_EDGE_DIM_V2 if enc.feature_version >= 2 else GALCON_EDGE_DIM
    )
    pf = np.zeros((GALCON_MAX_PLANETS, p_dim), dtype=np.float32)
    tf = np.zeros((GALCON_MAX_PLANETS, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32)
    ef = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS, e_dim), dtype=np.float32)
    le = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=bool)
    om = np.zeros((GALCON_MAX_PLANETS,), dtype=bool)
    pm = np.zeros((GALCON_MAX_PLANETS,), dtype=bool)
    if n > 0:
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


def build_galcon_transformer(
    *,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    dim_ff: int = 256,
) -> PlanetTimelineTransformer:
    """Instantiate our shared v1 ``PlanetTimelineTransformer`` configured for Galcon."""
    return PlanetTimelineTransformer(
        planet_dim=GALCON_PLANET_DIM,
        timeline_dim=GALCON_TIMELINE_DIM,
        edge_dim=GALCON_EDGE_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
    )


class GalconPlanetTransformer(nn.Module):
    """Gen 2/3/4 Galcon Transformer wrapping ``PlanetTimelineTransformer`` with world-model, ``/REDIR`` & native multi-send heads.

    Directly reuses ``orbit_wars_bot.planet_transformer.PlanetTimelineTransformer`` as its
    core trunk + policy/fraction/value heads (keeping zero modifications to ``orbit_wars_bot/``)
    and adds:
    1. ``aux_future_head``: predicts each planet's ``t+10`` ownership (3-way) + ``log1p(ships)/5.5``.
    2. ``redir_*`` heads: native D4-invariant mid-flight ``/REDIR`` scorer over flying fleets.
    3. ``ms_*`` heads (Gen 4, optional): independent pairwise Bernoulli multi-target send head
       ``(ms_logits, ms_fracs)`` enabling native simultaneous multi-neutral openings & pincers.
    """

    def __init__(
        self,
        *,
        planet_dim: int = GALCON_PLANET_DIM_V2,
        timeline_dim: int = GALCON_TIMELINE_DIM,
        edge_dim: int = GALCON_EDGE_DIM_V2,
        redir_dim: int = GALCON_REDIR_DIM,
        d_model: int = 160,
        n_heads: int = 4,
        n_layers: int = 5,
        dim_ff: int = 384,
        feature_version: int = 2,
        enable_multi_send: bool = False,
    ) -> None:
        super().__init__()
        self.enable_multi_send = bool(enable_multi_send)
        self.backbone = PlanetTimelineTransformer(
            planet_dim=planet_dim,
            timeline_dim=timeline_dim,
            edge_dim=edge_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
        )
        self.config = {
            **self.backbone.config,
            "redir_dim": redir_dim,
            "feature_version": feature_version,
            "enable_multi_send": self.enable_multi_send,
            "model_class": "GalconPlanetTransformer",
        }
        # Auxiliary t+10 future-state world-model head: [owner_neu, owner_me, owner_en, ships_log]
        self.aux_future_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )
        # Native D4-invariant mid-flight /REDIR head conditioned on (h_cur_dst, h_cand_dst, redir_feat)
        self.redir_cur_proj = nn.Linear(d_model, d_model // 2)
        self.redir_dst_proj = nn.Linear(d_model, d_model // 2)
        self.redir_feat_proj = nn.Linear(redir_dim, d_model // 2)
        self.redir_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Gen 4 native pairwise multi-target send head: independent (send_logit, send_frac) per edge (i, j)
        if self.enable_multi_send:
            self.ms_src_proj = nn.Linear(d_model, d_model // 2)
            self.ms_dst_proj = nn.Linear(d_model, d_model // 2)
            self.ms_edge_proj = nn.Linear(edge_dim, d_model // 2)
            self.ms_head = nn.Sequential(
                nn.GELU(),
                nn.Linear(d_model // 2, 2),
            )
            # Initialize send logit bias conservatively (-2.2) so warm-started Gen 4 starts calm
            nn.init.constant_(self.ms_head[1].bias[0], -2.2)
            nn.init.constant_(self.ms_head[1].bias[1], -0.4)

    def encode_planets(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contextual per-planet embeddings ``h`` of shape ``(B, N, D)`` via ``self.backbone``."""
        B, N, T, Ct = timeline_feats.shape
        tl = timeline_feats.reshape(B * N, T, Ct).transpose(1, 2)
        tl_h = self.backbone.timeline_conv(tl)
        tl_pool = torch.cat([tl_h.mean(dim=-1), tl_h.amax(dim=-1)], dim=-1).view(B, N, 128)
        x = self.backbone.node_in(torch.cat([planet_feats, tl_pool], dim=-1))
        for layer in self.backbone.layers:
            x = layer(x, edge_feats, pad_mask)
        return self.backbone.final_norm(x)

    def decode_value_from_hidden(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Compute scalar value prediction ``(B,)`` directly from contextual planet embeddings."""
        va = self.backbone.val_attn(h).squeeze(-1).masked_fill(~pad_mask, -1e9)
        vw = F.softmax(va, dim=-1).unsqueeze(-1)
        pooled = (h * vw).sum(dim=1)
        return torch.tanh(self.backbone.val_head(pooled).squeeze(-1))

    def decode_from_hidden(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the shared ``PlanetTimelineTransformer`` policy, fraction, and value heads."""
        hold_logit = self.backbone.hold_head(h)
        pair_in = (
            self.backbone.src_proj(h).unsqueeze(2)
            + self.backbone.dst_proj(h).unsqueeze(1)
            + self.backbone.edge_proj(edge_feats)
        )
        pair_out = self.backbone.pair_head(pair_in)
        target_logits = pair_out[..., 0]
        frac_preds = torch.sigmoid(pair_out[..., 1])
        action_logits = torch.cat([hold_logit, target_logits], dim=-1)
        value = self.decode_value_from_hidden(h, pad_mask)
        return action_logits, frac_preds, value

    def predict_multi_send(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict independent pairwise ``(ms_logits, ms_fracs)`` of shape ``(B, N, N)`` each."""
        if not self.enable_multi_send:
            raise RuntimeError("predict_multi_send called on a model without enable_multi_send=True")
        pair = (
            self.ms_src_proj(h).unsqueeze(2)
            + self.ms_dst_proj(h).unsqueeze(1)
            + self.ms_edge_proj(edge_feats)
        )
        out = self.ms_head(pair)
        return out[..., 0], torch.sigmoid(out[..., 1])

    def predict_aux_future(self, h: torch.Tensor) -> torch.Tensor:
        """Predict per-planet ``t+10`` state ``(B, N, 4)``: ``[own_neu, own_me, own_en, log_ships]``."""
        return self.aux_future_head(h)

    def predict_redir_logits(
        self,
        h: torch.Tensor,
        cur_dst_idx: torch.Tensor,
        redir_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate redirect targets ``(B, F, N)`` for flying fleets."""
        # h: (B, N, D), cur_dst_idx: (B, F), redir_feats: (B, F, N, R), pad_mask: (B, N)
        B, N, D = h.shape
        gather_idx = cur_dst_idx.clamp(0, N - 1).unsqueeze(-1).expand(-1, -1, D)
        h_cur = torch.gather(h, 1, gather_idx)  # (B, F, D)
        pair = (
            self.redir_cur_proj(h_cur).unsqueeze(2)
            + self.redir_dst_proj(h).unsqueeze(1)
            + self.redir_feat_proj(redir_feats)
        )  # (B, F, N, D/2)
        logits = self.redir_head(pair).squeeze(-1)  # (B, F, N)
        return logits.masked_fill(~pad_mask.unsqueeze(1), -1e9)

    def forward_multi_send(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        return_aux: bool = False,
        return_hidden: bool = False,
    ):
        """Fast Gen 4 multi-send forward pass skipping the unused Gen 2/3 legacy pairwise head."""
        h = self.encode_planets(planet_feats, timeline_feats, edge_feats, pad_mask)
        ms_logits, ms_fracs = self.predict_multi_send(h, edge_feats)
        value = self.decode_value_from_hidden(h, pad_mask)
        if return_aux and return_hidden:
            return ms_logits, ms_fracs, value, self.predict_aux_future(h), h
        if return_aux:
            return ms_logits, ms_fracs, value, self.predict_aux_future(h)
        if return_hidden:
            return ms_logits, ms_fracs, value, h
        return ms_logits, ms_fracs, value

    def forward(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        return_aux: bool = False,
        return_hidden: bool = False,
    ):
        h = self.encode_planets(planet_feats, timeline_feats, edge_feats, pad_mask)
        action_logits, frac_preds, value = self.decode_from_hidden(h, edge_feats, pad_mask)
        if return_aux and return_hidden:
            return action_logits, frac_preds, value, self.predict_aux_future(h), h
        if return_aux:
            return action_logits, frac_preds, value, self.predict_aux_future(h)
        if return_hidden:
            return action_logits, frac_preds, value, h
        return action_logits, frac_preds, value


def build_galcon_gen2_transformer(
    *,
    d_model: int = 160,
    n_heads: int = 4,
    n_layers: int = 5,
    dim_ff: int = 384,
    feature_version: int = 2,
    enable_multi_send: bool = False,
) -> GalconPlanetTransformer:
    """Instantiate the Gen 2/3/4 ``GalconPlanetTransformer`` with world-model, ``/REDIR`` & optional multi-send heads."""
    return GalconPlanetTransformer(
        planet_dim=GALCON_PLANET_DIM_V2 if feature_version >= 2 else GALCON_PLANET_DIM,
        timeline_dim=GALCON_TIMELINE_DIM,
        edge_dim=GALCON_EDGE_DIM_V2 if feature_version >= 2 else GALCON_EDGE_DIM,
        redir_dim=GALCON_REDIR_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        feature_version=feature_version,
        enable_multi_send=enable_multi_send,
    )


def build_galcon_gen4_transformer(
    *,
    d_model: int = 160,
    n_heads: int = 4,
    n_layers: int = 5,
    dim_ff: int = 384,
    feature_version: int = 2,
) -> GalconPlanetTransformer:
    """Instantiate the Gen 4 ``1.29M`` ``GalconPlanetTransformer`` with native pairwise multi-target send head."""
    return build_galcon_gen2_transformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        feature_version=feature_version,
        enable_multi_send=True,
    )


def load_galcon_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, int, bool]:
    """Load a v1, Gen 2/3, or Gen 4 multi-send Galcon Transformer checkpoint.

    Returns ``(model, feature_version, has_native_redir)``.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    model_cls = cfg.pop("model_class", None) or ckpt.get("model_class")
    has_redir = bool(ckpt.get("has_trained_redir", False))
    sd = ckpt["model_state_dict"]
    has_ms = bool(cfg.get("enable_multi_send", False) or any(k.startswith("ms_head.") for k in sd.keys()))

    if model_cls == "GalconGen5Transformer" or any(k.startswith("val_wdl_head.") for k in sd.keys()):
        fv = int(cfg.get("feature_version", 2))
        model = GalconGen5Transformer(
            planet_dim=int(cfg.get("planet_dim", GALCON_PLANET_DIM_V2)),
            timeline_dim=int(cfg.get("timeline_dim", GALCON_TIMELINE_DIM)),
            edge_dim=int(cfg.get("edge_dim", GALCON_EDGE_DIM_V2)),
            redir_dim=int(cfg.get("redir_dim", GALCON_REDIR_DIM)),
            d_model=int(cfg.get("d_model", 192)),
            n_heads=int(cfg.get("n_heads", 6)),
            n_layers=int(cfg.get("n_layers", 6)),
            dim_ff=int(cfg.get("dim_ff", 384)),
            edge_hidden=int(cfg.get("edge_hidden", 24)),
            num_val_queries=int(cfg.get("num_val_queries", 4)),
            feature_version=fv,
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
        model.load_state_dict(sd)
        model.has_trained_multi_send = bool(ckpt.get("has_trained_multi_send", True))
        model.use_canonical_d4_trained = bool(ckpt.get("use_canonical_d4", True))
        model.eval()
        return model, fv, has_redir

    if model_cls == "GalconPlanetTransformer" or "redir_dim" in cfg or any(
        k.startswith("backbone.") for k in sd.keys()
    ):
        fv = int(cfg.get("feature_version", 2 if cfg.get("planet_dim", 18) >= 22 else 1))
        model = GalconPlanetTransformer(
            planet_dim=int(cfg.get("planet_dim", GALCON_PLANET_DIM_V2)),
            timeline_dim=int(cfg.get("timeline_dim", GALCON_TIMELINE_DIM)),
            edge_dim=int(cfg.get("edge_dim", GALCON_EDGE_DIM_V2)),
            redir_dim=int(cfg.get("redir_dim", GALCON_REDIR_DIM)),
            d_model=int(cfg.get("d_model", 160)),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_layers", 5)),
            dim_ff=int(cfg.get("dim_ff", 384)),
            feature_version=fv,
            enable_multi_send=has_ms,
        ).to(device)
        model.load_state_dict(sd)
        model.has_trained_multi_send = bool(ckpt.get("has_trained_multi_send", has_ms))
        model.use_canonical_d4_trained = bool(ckpt.get("use_canonical_d4", False))
        model.eval()
        return model, fv, has_redir

    fv = 2 if int(cfg.get("planet_dim", 18)) >= 22 else 1
    clean_cfg = {k: v for k, v in cfg.items() if k in ("planet_dim", "timeline_dim", "edge_dim", "d_model", "n_heads", "n_layers", "dim_ff")}
    model = PlanetTimelineTransformer(**clean_cfg).to(device)
    model.load_state_dict(sd)
    model.has_trained_multi_send = False
    model.use_canonical_d4_trained = False
    model.eval()
    return model, fv, False


_MULTI_SEND_MODELS = (GalconPlanetTransformer, GalconGen5Transformer)


class GalconTransformerAgent:
    """Fast Galcon neural agent powered by ``PlanetTimelineTransformer`` / ``GalconPlanetTransformer`` / ``GalconGen5Transformer``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        redir_threshold: float = 0.55,
        multi_send_threshold: float = 0.45,
        max_sends_per_planet: int = 6,
        use_d4_tta: bool = True,
        use_d4_canonical: Optional[bool] = None,
        enable_redirect_overlay: bool = True,
        use_native_redir: bool = True,
        use_multi_send: bool = True,
        contempt: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.launch_threshold = float(launch_threshold)
        self.redir_threshold = float(redir_threshold)
        self.multi_send_threshold = float(multi_send_threshold)
        self.max_sends_per_planet = max(1, int(max_sends_per_planet))
        self.use_d4_tta = bool(use_d4_tta)
        self.enable_redirect_overlay = bool(enable_redirect_overlay)
        self.use_native_redir = bool(use_native_redir)
        self.use_multi_send = bool(use_multi_send)

        self.model, self.feature_version, self.has_trained_redir = load_galcon_model_from_checkpoint(
            checkpoint_path, device=self.device
        )
        if contempt is not None and hasattr(self.model, "contempt"):
            self.model.contempt = float(contempt)
        self.has_trained_multi_send = bool(getattr(self.model, "has_trained_multi_send", False))
        if use_d4_canonical is None:
            self.use_d4_canonical = bool(getattr(self.model, "use_canonical_d4_trained", False))
        else:
            self.use_d4_canonical = bool(use_d4_canonical)

    @torch.no_grad()
    def predict_probs_and_value(
        self,
        enc: EncodedGalconState,
        *,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        probs, mean_fracs, val, _, _, _ = self._predict_all(enc, redir_enc=None, use_d4_tta=use_d4_tta)
        return probs, mean_fracs, val

    @torch.no_grad()
    def predict_probs_value_and_redir(
        self,
        enc: EncodedGalconState,
        redir_enc: Optional[EncodedRedirState],
        *,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        probs, mean_fracs, val, redir_probs, _, _ = self._predict_all(
            enc, redir_enc=redir_enc, use_d4_tta=use_d4_tta
        )
        return probs, mean_fracs, val, redir_probs

    @torch.no_grad()
    def _predict_all(
        self,
        enc: EncodedGalconState,
        *,
        redir_enc: Optional[EncodedRedirState] = None,
        use_d4_tta: Optional[bool] = None,
        use_d4_canonical: Optional[bool] = None,
        need_value: bool = True,
        ms_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        n = len(enc.planet_ids)
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0.0, None, None, None
        canon = self.use_d4_canonical if use_d4_canonical is None else bool(use_d4_canonical)
        d4 = False if canon else (self.use_d4_tta if use_d4_tta is None else bool(use_d4_tta))
        if canon:
            pf_b = canonicalize_galcon_d4(enc.planet_feats)[None]
            k_b = 1
        else:
            syms = [(k, f) for f in (False, True) for k in range(4)] if d4 else [(0, False)]
            k_b = len(syms)
            pf_b = np.stack([apply_galcon_d4(enc.planet_feats, k_rot=k, flip_x=f) for k, f in syms], axis=0)
        tf_b = np.broadcast_to(enc.timeline_feats[None], (k_b, n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM)).copy()
        ef_b = np.broadcast_to(enc.edge_feats[None], (k_b, n, n, enc.edge_feats.shape[-1])).copy()
        pm_b = np.ones((k_b, n), dtype=bool)

        pf_t = torch.from_numpy(pf_b).to(self.device)
        tf_t = torch.from_numpy(tf_b).to(self.device)
        ef_t = torch.from_numpy(ef_b).to(self.device)
        pm_t = torch.from_numpy(pm_b).to(self.device)

        want_redir = (
            isinstance(self.model, _MULTI_SEND_MODELS)
            and redir_enc is not None
            and len(redir_enc.fleet_ids) > 0
        )
        want_ms = (
            isinstance(self.model, _MULTI_SEND_MODELS)
            and self.model.enable_multi_send
        )
        if ms_only and want_ms:
            if isinstance(self.model, GalconGen5Transformer):
                h, e_hid = self.model.encode_planets_and_edges(pf_t, tf_t, ef_t, pm_t)
                ms_logits, ms_fr = self.model.predict_multi_send(h, ef_t, edge_hidden=e_hid)
            else:
                h = self.model.encode_planets(pf_t, tf_t, ef_t, pm_t)
                ms_logits, ms_fr = self.model.predict_multi_send(h, ef_t)
            legal_t = torch.from_numpy(enc.legal_edges).to(self.device)
            ms_logits = ms_logits.masked_fill(~legal_t.unsqueeze(0), -1e9)
            ms_probs = torch.sigmoid(ms_logits).mean(dim=0).cpu().numpy()
            ms_fracs = ms_fr.mean(dim=0).cpu().numpy()
            redir_probs = None
            if want_redir and redir_enc is not None:
                f_cnt = len(redir_enc.fleet_ids)
                cd_t = torch.from_numpy(
                    np.broadcast_to(redir_enc.cur_dst_idx[None], (k_b, f_cnt)).copy()
                ).to(self.device)
                rf_t = torch.from_numpy(
                    np.broadcast_to(redir_enc.redir_feats[None], (k_b, f_cnt, n, GALCON_REDIR_DIM)).copy()
                ).to(self.device)
                r_logits = self.model.predict_redir_logits(h, cd_t, rf_t, pm_t)
                redir_probs = F.softmax(r_logits, dim=-1).mean(dim=0).cpu().numpy()
            val_f = float(self.model.decode_value_from_hidden(h, pm_t).mean().item()) if need_value else 0.0
            return (
                np.zeros((n, n + 1), dtype=np.float32),
                np.zeros((n, n), dtype=np.float32),
                val_f,
                redir_probs,
                ms_probs,
                ms_fracs,
            )

        if want_redir or want_ms:
            if isinstance(self.model, GalconGen5Transformer):
                h, e_hid = self.model.encode_planets_and_edges(pf_t, tf_t, ef_t, pm_t)
                ms_logits, ms_fr = self.model.predict_multi_send(h, ef_t, edge_hidden=e_hid)
                vals = self.model.decode_value_from_hidden(h, pm_t)
                logits = torch.cat([torch.zeros_like(ms_logits[:, :, :1]), ms_logits], dim=-1)
                fracs = ms_fr
                legal_t = torch.from_numpy(enc.legal_edges).to(self.device)
                ms_logits_m = ms_logits.masked_fill(~legal_t.unsqueeze(0), -1e9)
                ms_probs = torch.sigmoid(ms_logits_m).mean(dim=0).cpu().numpy()
                ms_fracs = ms_fr.mean(dim=0).cpu().numpy()
                redir_probs = None
                if want_redir and redir_enc is not None:
                    f_cnt = len(redir_enc.fleet_ids)
                    cd_t = torch.from_numpy(
                        np.broadcast_to(redir_enc.cur_dst_idx[None], (k_b, f_cnt)).copy()
                    ).to(self.device)
                    rf_t = torch.from_numpy(
                        np.broadcast_to(redir_enc.redir_feats[None], (k_b, f_cnt, n, GALCON_REDIR_DIM)).copy()
                    ).to(self.device)
                    r_logits = self.model.predict_redir_logits(h, cd_t, rf_t, pm_t)
                    redir_probs = F.softmax(r_logits, dim=-1).mean(dim=0).cpu().numpy()
            else:
                logits, fracs, vals, h = self.model(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
                redir_probs = None
                if want_redir and redir_enc is not None:
                    f_cnt = len(redir_enc.fleet_ids)
                    cd_t = torch.from_numpy(
                        np.broadcast_to(redir_enc.cur_dst_idx[None], (k_b, f_cnt)).copy()
                    ).to(self.device)
                    rf_t = torch.from_numpy(
                        np.broadcast_to(redir_enc.redir_feats[None], (k_b, f_cnt, n, GALCON_REDIR_DIM)).copy()
                    ).to(self.device)
                    r_logits = self.model.predict_redir_logits(h, cd_t, rf_t, pm_t)
                    redir_probs = F.softmax(r_logits, dim=-1).mean(dim=0).cpu().numpy()
                ms_probs = None
                ms_fracs = None
                if want_ms:
                    ms_logits, ms_fr = self.model.predict_multi_send(h, ef_t)
                    legal_t = torch.from_numpy(enc.legal_edges).to(self.device)
                    ms_logits = ms_logits.masked_fill(~legal_t.unsqueeze(0), -1e9)
                    ms_probs = torch.sigmoid(ms_logits).mean(dim=0).cpu().numpy()
                    ms_fracs = ms_fr.mean(dim=0).cpu().numpy()
        else:
            logits, fracs, vals = self.model(pf_t, tf_t, ef_t, pm_t)
            redir_probs = None
            ms_probs = None
            ms_fracs = None

        legal = torch.from_numpy(enc.legal_edges).to(self.device)
        mask = torch.cat([torch.ones((n, 1), dtype=torch.bool, device=self.device), legal], dim=-1)
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        mean_fracs = fracs.mean(dim=0).cpu().numpy()
        return probs, mean_fracs, float(vals.mean().item()), redir_probs, ms_probs, ms_fracs

    def _append_redirects(
        self,
        orders: List[Order],
        sim: GalconSim,
        enc: EncodedGalconState,
        *,
        redir_enc: Optional[EncodedRedirState],
        redir_probs: Optional[np.ndarray],
        include_redirects: bool,
    ) -> None:
        if not (include_redirects and sim.allow_redirect and sim.fleets):
            return
        emitted_fids = set()
        if (
            self.use_native_redir
            and self.has_trained_redir
            and redir_enc is not None
            and redir_probs is not None
            and len(redir_enc.fleet_ids) > 0
        ):
            for fi, fid in enumerate(redir_enc.fleet_ids):
                cur_idx = int(redir_enc.cur_dst_idx[fi])
                p_row = redir_probs[fi]
                best_j = int(np.argmax(p_row))
                if best_j != cur_idx and float(p_row[best_j]) >= self.redir_threshold:
                    orders.append(RedirOrder(fid, enc.planet_ids[best_j]))
                    emitted_fids.add(fid)
        elif self.enable_redirect_overlay:
            smart_redirs, _ = compute_smart_redirects(sim, enc, verify_rollout=False)
            for ro in smart_redirs:
                if ro.fleet_id not in emitted_fids:
                    orders.append(ro)

    def _decode_orders(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        probs: np.ndarray,
        mean_fracs: np.ndarray,
        *,
        launch_threshold: float,
        frac_scale: float = 1.0,
        overrides: Optional[Dict[int, int]] = None,
        redir_enc: Optional[EncodedRedirState] = None,
        redir_probs: Optional[np.ndarray] = None,
        include_redirects: bool = True,
    ) -> List[Order]:
        n = len(enc.planet_ids)
        orders: List[Order] = []
        for i in range(n):
            if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                continue
            if overrides is not None and i in overrides:
                cls = int(overrides[i])
                if cls == 0:
                    continue
                best_j = cls - 1
            else:
                if (1.0 - float(probs[i, 0])) < launch_threshold:
                    continue
                best_j = int(np.argmax(probs[i, 1:]))
            if best_j < 0 or best_j >= n or not enc.legal_edges[i, best_j]:
                continue
            f = float(mean_fracs[i, best_j]) * float(frac_scale)
            f = max(0.05, min(1.0, f))
            pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
            orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[best_j], pct))

        self._append_redirects(
            orders, sim, enc, redir_enc=redir_enc, redir_probs=redir_probs, include_redirects=include_redirects
        )
        return orders

    def _decode_multi_send_orders(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        ms_probs: np.ndarray,
        ms_fracs: np.ndarray,
        *,
        threshold: float,
        frac_scale: float = 1.0,
        edge_overrides: Optional[Dict[Tuple[int, int], bool]] = None,
        redir_enc: Optional[EncodedRedirState] = None,
        redir_probs: Optional[np.ndarray] = None,
        include_redirects: bool = True,
    ) -> List[Order]:
        """Decode Gen 4 independent pairwise multi-target send probabilities into exact sequential ``SendOrder``s."""
        n = len(enc.planet_ids)
        orders: List[Order] = []
        for i in range(n):
            if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                continue
            src = enc.planets[i]
            orig_ships = float(src.ships)
            if orig_ships < 2.0:
                continue

            selected: List[Tuple[float, int]] = []
            for j in range(n):
                if j == i or not enc.legal_edges[i, j]:
                    continue
                p_ij = float(ms_probs[i, j])
                if edge_overrides is not None and (i, j) in edge_overrides:
                    fire = bool(edge_overrides[(i, j)])
                else:
                    fire = p_ij >= threshold
                if fire:
                    selected.append((p_ij, j))
            if not selected:
                continue
            selected.sort(key=lambda x: -x[0])
            selected = selected[: self.max_sends_per_planet]

            rem = orig_ships
            for _, j in selected:
                if rem < 2.0:
                    break
                want = float(ms_fracs[i, j]) * float(frac_scale) * orig_ships
                want = min(rem, max(1.0, want))
                ratio = want / max(1.0, rem)
                pct = 100 if ratio >= 0.92 else max(5, min(100, int(round(ratio * 100.0))))
                sent = math.floor(rem * (pct / 100.0) + 1e-9)
                if sent < 1:
                    continue
                orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                rem = max(0.0, rem - sent)

        self._append_redirects(
            orders, sim, enc, redir_enc=redir_enc, redir_probs=redir_probs, include_redirects=include_redirects
        )
        return orders

    @torch.no_grad()
    def act(self, sim: GalconSim, player: int) -> List[Order]:
        enc = extract_galcon_features(sim, player, feature_version=self.feature_version)
        if not enc.planet_ids or not np.any(enc.owned_mask):
            return []
        redir_enc = (
            extract_galcon_redir_features(sim, enc)
            if (self.use_native_redir and self.has_trained_redir and sim.allow_redirect and sim.fleets)
            else None
        )
        want_ms_only = bool(self.use_multi_send and self.has_trained_multi_send)
        probs, mean_fracs, _, redir_probs, ms_probs, ms_fracs = self._predict_all(
            enc, redir_enc=redir_enc, need_value=False, ms_only=want_ms_only
        )
        if self.use_multi_send and self.has_trained_multi_send and ms_probs is not None and ms_fracs is not None:
            return self._decode_multi_send_orders(
                sim,
                enc,
                ms_probs,
                ms_fracs,
                threshold=self.multi_send_threshold,
                redir_enc=redir_enc,
                redir_probs=redir_probs,
            )
        return self._decode_orders(
            sim,
            enc,
            probs,
            mean_fracs,
            launch_threshold=self.launch_threshold,
            redir_enc=redir_enc,
            redir_probs=redir_probs,
        )


class GalconSearchTransformerAgent(GalconTransformerAgent):
    """Gen 3/4 Batched Opponent-Adaptive Multi-Ply Reactive Neural Search over ``GalconTransformerAgent``.

    Evaluates ~10–12 policy-proposed candidate plans (including Gen 4 native multi-send sweeps and
    ROI fan-out proposals) using a batched 2-ply or 3-ply reactive neural rollout:
    - Tracks the opponent's live style online (multi-send macro expander vs. early home-rusher) so
      simulated opponent replies adapt dynamically to bots like ``esparano Sparky v1.6`` or rushers.
    - Uses exact ``1x`` Canonical D4 when enabled (~5x faster than 8x D4 TTA on CPU).
    - Evaluates all rollout leaves in ONE batched forward pass with calibrated ``neural_value_weight=0.85``
      and a ``+0.025`` prior-trust margin on the base pure policy.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        redir_threshold: float = 0.55,
        multi_send_threshold: float = 0.45,
        use_d4_tta: bool = True,
        use_d4_canonical: Optional[bool] = True,
        rollout_horizon: int = 16,
        num_plies: int = 2,
        time_budget_sec: float = 0.28,
        neural_value_weight: float = 0.70,
        reactive_mid_ply: bool = True,
        opponent_adaptive: bool = True,
        adversarial_search: bool = True,
        enable_rescue_snipe: bool = False,
        enable_opp_direct_rush: bool = False,
        enable_def_stream: bool = True,
        enable_opening_def_guard: bool = True,
        prior_bonus: float = 0.015,
        contempt: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            launch_threshold=launch_threshold,
            redir_threshold=redir_threshold,
            multi_send_threshold=multi_send_threshold,
            use_d4_tta=use_d4_tta,
            use_d4_canonical=use_d4_canonical,
            contempt=contempt,
            device=device,
        )
        if contempt is None and isinstance(self.model, GalconGen5Transformer):
            self.model.contempt = 0.30
        self.rollout_horizon = max(4, int(rollout_horizon))
        self.num_plies = max(1, min(3, int(num_plies))) if reactive_mid_ply else 1
        self.time_budget_sec = float(time_budget_sec)
        self.neural_value_weight = float(neural_value_weight)
        self.reactive_mid_ply = bool(reactive_mid_ply)
        self.opponent_adaptive = bool(opponent_adaptive)
        self.adversarial_search = bool(adversarial_search)
        self.enable_rescue_snipe = bool(enable_rescue_snipe)
        self.enable_opp_direct_rush = bool(enable_opp_direct_rush)
        self.enable_def_stream = bool(enable_def_stream)
        self.enable_opening_def_guard = bool(enable_opening_def_guard)
        self.prior_bonus = float(prior_bonus)
        # Online opponent style tracker (reset automatically whenever sim.tick decreases)
        self._last_obs_tick = 10**9
        self._last_seen_fid = -1
        self._opp_multi_send_ema = 0.0
        self._opp_rush_ema = 0.0

    def _update_opp_profile(self, sim: GalconSim, player: int) -> None:
        """Observe newly launched opponent fleets to classify multi-send expanders vs. early rushers."""
        if sim.tick <= self._last_obs_tick:
            self._last_seen_fid = -1
            self._opp_multi_send_ema = 0.0
            self._opp_rush_ema = 0.0
        self._last_obs_tick = sim.tick
        opp = 2 if player == 1 else 1
        new_by_src: Dict[int, List[Fleet]] = {}
        max_fid = self._last_seen_fid
        for fid, fl in sim.fleets.items():
            if fid > max_fid:
                max_fid = fid
            if fid > self._last_seen_fid and fl.owner == opp:
                new_by_src.setdefault(fl.source, []).append(fl)
        self._last_seen_fid = max_fid
        if not new_by_src:
            return
        max_fan = max(len(v) for v in new_by_src.values())
        multi_sig = 1.0 if max_fan >= 2 else 0.0
        self._opp_multi_send_ema = 0.65 * self._opp_multi_send_ema + 0.35 * multi_sig
        if sim.tick <= 20:
            my_homes = {p.id for p in sim.planets.values() if p.owner == player}
            rush_ships = sum(fl.ships for fls in new_by_src.values() for fl in fls if fl.target in my_homes)
            rush_sig = 1.0 if rush_ships >= 45.0 else 0.0
            self._opp_rush_ema = 0.60 * self._opp_rush_ema + 0.40 * rush_sig

    def _prep_pf(self, enc: EncodedGalconState) -> np.ndarray:
        return canonicalize_galcon_d4(enc.planet_feats) if self.use_d4_canonical else enc.planet_feats

    @torch.no_grad()
    def _batch_forward_policy_and_val(
        self, encs: Sequence[EncodedGalconState], *, need_value: bool = True, ms_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run ONE batched 1x forward pass over a list of same-N encoded states."""
        k = len(encs)
        if k == 0:
            return (
                np.zeros((0, 0, 1), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                None,
                None,
            )
        n = len(encs[0].planet_ids)
        pf_b = np.stack([self._prep_pf(e) for e in encs], axis=0)
        tf_b = np.stack([e.timeline_feats for e in encs], axis=0)
        ef_b = np.stack([e.edge_feats for e in encs], axis=0)
        le_b = np.stack([e.legal_edges for e in encs], axis=0)
        pm_b = np.ones((k, n), dtype=bool)
        pf_t = torch.from_numpy(pf_b).to(self.device)
        tf_t = torch.from_numpy(tf_b).to(self.device)
        ef_t = torch.from_numpy(ef_b).to(self.device)
        pm_t = torch.from_numpy(pm_b).to(self.device)
        le_t = torch.from_numpy(le_b).to(self.device)

        want_ms = isinstance(self.model, _MULTI_SEND_MODELS) and self.model.enable_multi_send
        if ms_only and want_ms:
            if isinstance(self.model, GalconGen5Transformer):
                h, e_hid = self.model.encode_planets_and_edges(pf_t, tf_t, ef_t, pm_t)
                ms_l, ms_f = self.model.predict_multi_send(h, ef_t, edge_hidden=e_hid)
            else:
                h = self.model.encode_planets(pf_t, tf_t, ef_t, pm_t)
                ms_l, ms_f = self.model.predict_multi_send(h, ef_t)
            ms_l = ms_l.masked_fill(~le_t, -1e9)
            ms_p = torch.sigmoid(ms_l).cpu().numpy()
            ms_f_np = ms_f.cpu().numpy()
            vals_np = (
                self.model.decode_value_from_hidden(h, pm_t).cpu().numpy()
                if need_value
                else np.zeros((k,), dtype=np.float32)
            )
            return (
                np.zeros((k, n, n + 1), dtype=np.float32),
                np.zeros((k, n, n), dtype=np.float32),
                vals_np,
                ms_p,
                ms_f_np,
            )

        if want_ms:
            if isinstance(self.model, GalconGen5Transformer):
                h, e_hid = self.model.encode_planets_and_edges(pf_t, tf_t, ef_t, pm_t)
                ms_l, ms_f = self.model.predict_multi_send(h, ef_t, edge_hidden=e_hid)
                vals = self.model.decode_value_from_hidden(h, pm_t)
                logits = torch.cat([torch.zeros_like(ms_l[:, :, :1]), ms_l], dim=-1)
                fracs = ms_f
            else:
                logits, fracs, vals, h = self.model(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
                ms_l, ms_f = self.model.predict_multi_send(h, ef_t)
            ms_l = ms_l.masked_fill(~le_t, -1e9)
            ms_p = torch.sigmoid(ms_l).cpu().numpy()
            ms_f_np = ms_f.cpu().numpy()
        else:
            logits, fracs, vals = self.model(pf_t, tf_t, ef_t, pm_t)
            ms_p = None
            ms_f_np = None

        mask = torch.cat([torch.ones((k, n, 1), dtype=torch.bool, device=self.device), le_t], dim=-1)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs, fracs.cpu().numpy(), vals.cpu().numpy(), ms_p, ms_f_np

    @torch.no_grad()
    def _batch_evaluate_leaves(self, leaf_encs: Sequence[EncodedGalconState]) -> np.ndarray:
        """Evaluate all non-terminal candidate rollout leaves in ONE batched forward pass."""
        k = len(leaf_encs)
        if k == 0:
            return np.zeros((0,), dtype=np.float32)
        n = len(leaf_encs[0].planet_ids)
        pf_t = torch.from_numpy(np.stack([self._prep_pf(e) for e in leaf_encs], axis=0)).to(self.device)
        tf_t = torch.from_numpy(np.stack([e.timeline_feats for e in leaf_encs], axis=0)).to(self.device)
        ef_t = torch.from_numpy(np.stack([e.edge_feats for e in leaf_encs], axis=0)).to(self.device)
        pm_t = torch.ones((k, n), dtype=torch.bool, device=self.device)
        if isinstance(self.model, _MULTI_SEND_MODELS):
            h = self.model.encode_planets(pf_t, tf_t, ef_t, pm_t)
            vals = self.model.decode_value_from_hidden(h, pm_t)
        else:
            _, _, vals = self.model(pf_t, tf_t, ef_t, pm_t)
        return vals.cpu().numpy()

    def _build_fanout_plan(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        probs: np.ndarray,
        *,
        ms_probs: Optional[np.ndarray] = None,
        max_targets_per_planet: int = 4,
        reserve_ratio: float = 0.15,
        base_redirs: Sequence[Order] = (),
    ) -> List[Order]:
        """Build a neural-prior + ROI guided multi-target fan-out plan for surplus-rich planets."""
        n = len(enc.planet_ids)
        futures = enc.futures or [
            project_planet_future(sim, p.id, enc.player, horizon=GALCON_TIMELINE_HORIZON) for p in enc.planets
        ]
        claimed_targets = set()
        orders: List[Order] = []
        for i in range(n):
            if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                continue
            src = enc.planets[i]
            safe = float(enc.safe_surplus[i])
            reserve = max(5.0, float(src.ships) * reserve_ratio)
            budget = min(safe, float(src.ships) - reserve)
            if budget < 14.0:
                continue

            scored: List[Tuple[float, int, float]] = []
            for j in range(n):
                if j == i or not enc.legal_edges[i, j] or j in claimed_targets:
                    continue
                dst = enc.planets[j]
                if dst.owner == enc.player:
                    continue
                eta = sim.planet_eta_ticks(src, dst)
                t_idx = min(GALCON_TIMELINE_HORIZON - 1, max(0, eta - 1))
                _, _, po, ps = futures[j]
                if po[t_idx] > 0:
                    continue
                extra = (
                    0.0
                    if dst.owner == 0
                    else (dst.production / PRODUCTION_DIVISOR) * 0.25 * max(0, eta - GALCON_TIMELINE_HORIZON)
                )
                need = math.ceil(float(ps[t_idx]) + extra) + 2.0
                if need <= 0.0 or need > budget:
                    continue
                roi = (dst.production / max(1.0, PRODUCTION_DIVISOR)) * 20.0 / max(4.0, need + 0.55 * eta)
                if dst.owner != 0:
                    roi *= 1.35
                p_prior = float(ms_probs[i, j]) if ms_probs is not None else float(probs[i, 1 + j])
                sc = (0.35 + p_prior * 3.0) * roi
                scored.append((sc, j, need))

            if len(scored) < 2:
                continue
            scored.sort(key=lambda x: -x[0])
            rem = float(src.ships)
            rem_budget = budget
            sent_count = 0
            for _, j, need in scored:
                if sent_count >= max_targets_per_planet or rem_budget < need or rem < 2.0:
                    continue
                pct = max(1, min(100, int(math.ceil(100.0 * need / max(1.0, rem)))))
                actual = math.floor(rem * (pct / 100.0) + 1e-9)
                if actual < 1 or actual > rem_budget + 2.0:
                    continue
                orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                claimed_targets.add(j)
                rem = max(0.0, rem - actual)
                rem_budget = max(0.0, rem_budget - actual)
                sent_count += 1
        orders.extend(base_redirs)
        return orders

    def _build_focus_fire_plan(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        probs: np.ndarray,
        *,
        ms_probs: Optional[np.ndarray] = None,
        max_sources: int = 4,
        base_redirs: Sequence[Order] = (),
    ) -> List[Order]:
        """Coordinate a multi-source focus-fire strike from 2-4 friendly planets onto the best contested target."""
        n = len(enc.planet_ids)
        owned = [i for i in range(n) if enc.owned_mask[i] and float(enc.safe_surplus[i]) >= 8.0]
        if len(owned) < 2:
            return []
        futures = enc.futures or [
            project_planet_future(sim, p.id, enc.player, horizon=GALCON_TIMELINE_HORIZON) for p in enc.planets
        ]
        best_target = -1
        best_plan: List[Order] = []
        best_score = 0.0

        for j in range(n):
            dst = enc.planets[j]
            if dst.owner == enc.player:
                continue
            cand_srcs: List[Tuple[int, int, float]] = []
            for i in owned:
                if i == j or not enc.legal_edges[i, j]:
                    continue
                eta = sim.planet_eta_ticks(enc.planets[i], dst)
                if eta > 18:
                    continue
                budget = min(float(enc.safe_surplus[i]), float(enc.planets[i].ships) * 0.80)
                if budget >= 8.0:
                    cand_srcs.append((eta, i, budget))
            if len(cand_srcs) < 2:
                continue
            cand_srcs.sort(key=lambda x: (x[0], -x[2]))
            chosen = cand_srcs[:max_sources]
            max_eta = max(x[0] for x in chosen)
            t_idx = min(GALCON_TIMELINE_HORIZON - 1, max(0, max_eta - 1))
            _, _, po, ps = futures[j]
            if po[t_idx] > 0:
                continue
            extra = (
                0.0
                if dst.owner == 0
                else (dst.production / PRODUCTION_DIVISOR) * 0.25 * max(0, max_eta - GALCON_TIMELINE_HORIZON)
            )
            need = math.ceil(float(ps[t_idx]) + extra) + 4.0
            total_avail = sum(x[2] for x in chosen)
            if total_avail < need or need <= 0.0:
                continue
            # Skip neutrals that a single friendly planet can comfortably solo
            max_single = max(x[2] for x in chosen)
            if dst.owner == 0 and max_single >= need * 1.15:
                continue
            prior_boost = float(
                max((ms_probs[i, j] if ms_probs is not None else probs[i, 1 + j]) for _, i, _ in chosen)
            )
            val = (dst.production / max(1.0, PRODUCTION_DIVISOR)) * (1.85 if dst.owner != 0 else 1.0)
            sc = val * (0.45 + 2.5 * prior_boost) / max(4.0, need * 0.35 + max_eta)
            if sc > best_score:
                rem_need = need
                orders: List[Order] = []
                for _, i, budget in chosen:
                    if rem_need <= 0.0:
                        break
                    send_s = min(budget, rem_need)
                    pct = max(5, min(100, int(math.ceil(100.0 * send_s / max(1.0, float(enc.planets[i].ships))))))
                    actual = math.floor(float(enc.planets[i].ships) * (pct / 100.0) + 1e-9)
                    if actual >= 1:
                        orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                        rem_need -= actual
                if len(orders) >= 2:
                    best_score = sc
                    best_target = j
                    best_plan = orders

        if best_target < 0 or not best_plan:
            return []
        best_plan.extend(base_redirs)
        return best_plan

    def _build_tunnel_feed_plan(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        *,
        tunnel_const: float = 1.18,
        base_redirs: Sequence[Order] = (),
    ) -> List[Order]:
        """Stream rear safe surplus toward frontline relay planets along near-collinear tunnels."""
        n = len(enc.planet_ids)
        my_idx = [i for i in range(n) if enc.owned_mask[i]]
        opp_idx = [j for j in range(n) if enc.planets[j].owner not in (0, enc.player)]
        if len(my_idx) < 2 or not opp_idx:
            return []

        dist_to_enemy: Dict[int, float] = {}
        nearest_enemy: Dict[int, int] = {}
        for i in my_idx:
            pi = enc.planets[i]
            best_d = 1e9
            best_e = opp_idx[0]
            for ej in opp_idx:
                pe = enc.planets[ej]
                d = math.hypot(pi.x - pe.x, pi.y - pe.y)
                if d < best_d:
                    best_d = d
                    best_e = ej
            dist_to_enemy[i] = best_d
            nearest_enemy[i] = best_e

        orders: List[Order] = []
        for i in my_idx:
            safe = float(enc.safe_surplus[i])
            if safe < 12.0:
                continue
            pi = enc.planets[i]
            pe = enc.planets[nearest_enemy[i]]
            d_direct = dist_to_enemy[i]
            best_relay = -1
            best_relay_d = 1e9
            for b in my_idx:
                if b == i or not enc.legal_edges[i, b]:
                    continue
                if dist_to_enemy[b] >= d_direct * 0.90:
                    continue
                pb = enc.planets[b]
                d1 = math.hypot(pi.x - pb.x, pi.y - pb.y)
                d2 = math.hypot(pb.x - pe.x, pb.y - pe.y)
                if d1 < d_direct and d2 < d_direct and (d1 + d2) < d_direct * tunnel_const:
                    if d1 < best_relay_d:
                        best_relay_d = d1
                        best_relay = b
            if best_relay >= 0:
                send_s = min(safe * 0.72, float(pi.ships) * 0.65)
                pct = max(10, min(90, int(round(100.0 * send_s / max(1.0, float(pi.ships))))))
                if math.floor(float(pi.ships) * (pct / 100.0) + 1e-9) >= 4:
                    orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[best_relay], pct))
        if not orders:
            return []
        orders.extend(base_redirs)
        return orders

    def _build_rescue_and_snipe_plan(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        probs: np.ndarray,
        *,
        ms_probs: Optional[np.ndarray] = None,
        base_redirs: Sequence[Order] = (),
    ) -> List[Order]:
        """Wave 12 surgical exact-defense rescue + single-target high-ROI snipe/capture plan."""
        n = len(enc.planet_ids)
        futures = enc.futures or [
            project_planet_future(sim, p.id, enc.player, horizon=GALCON_TIMELINE_HORIZON) for p in enc.planets
        ]
        orders: List[Order] = []
        avail = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            if enc.owned_mask[i]:
                avail[i] = max(0.0, min(float(enc.safe_surplus[i]), float(enc.planets[i].ships) * 0.85))
        rem_ships = np.array([float(p.ships) for p in enc.planets], dtype=np.float64)

        # 1. Exact defensive rescue for any friendly planet projected to fall within horizon
        threatened: List[Tuple[float, int, int, float]] = []
        for j in range(n):
            if not enc.owned_mask[j]:
                continue
            _, _, po, ps = futures[j]
            lost = [t for t in range(len(po)) if po[t] <= 0.0]
            if not lost:
                continue
            t_fall = lost[0] + 1
            deficit = float(ps[lost[0]]) + 3.0
            if deficit > 0.0:
                threatened.append((float(enc.planets[j].production) / max(1.0, float(t_fall)), j, t_fall, deficit))
        threatened.sort(key=lambda x: -x[0])

        for _, tgt_j, t_fall, need in threatened[:2]:
            dst = enc.planets[tgt_j]
            helpers = []
            for i in range(n):
                if i == tgt_j or not enc.owned_mask[i] or avail[i] < 4.0 or not enc.legal_edges[i, tgt_j]:
                    continue
                eta = sim.planet_eta_ticks(enc.planets[i], dst)
                if eta <= t_fall:
                    helpers.append((eta, i))
            helpers.sort(key=lambda x: x[0])
            if sum(avail[i] for _, i in helpers) < need * 0.85:
                continue
            rem_need = need
            for _, i in helpers:
                if rem_need <= 0.5 or rem_ships[i] < 2.0:
                    break
                send_s = min(avail[i], rem_need)
                if send_s < 2.0:
                    continue
                pct = max(5, min(100, int(math.ceil(100.0 * send_s / max(1.0, rem_ships[i])))))
                actual = math.floor(rem_ships[i] * (pct / 100.0) + 1e-9)
                if actual >= 1:
                    orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[tgt_j], pct))
                    avail[i] = max(0.0, avail[i] - actual)
                    rem_ships[i] = max(0.0, rem_ships[i] - actual)
                    rem_need -= actual

        # 2. Opportunistic exact-need single-target snipe / stalemate-breaking high-ROI capture
        claimed = set()
        for i in range(n):
            if not enc.owned_mask[i] or avail[i] < 6.0 or rem_ships[i] < 6.0:
                continue
            src = enc.planets[i]
            best_j = -1
            best_sc = 0.0
            best_need = 0.0
            for j in range(n):
                if j == i or j in claimed or not enc.legal_edges[i, j]:
                    continue
                dst = enc.planets[j]
                if dst.owner == enc.player:
                    continue
                eta = sim.planet_eta_ticks(src, dst)
                if eta > 18:
                    continue
                t_idx = min(GALCON_TIMELINE_HORIZON - 1, max(0, eta - 1))
                _, _, po, ps = futures[j]
                if po[t_idx] > 0.0:
                    continue
                extra = (
                    0.0
                    if dst.owner == 0
                    else (dst.production / PRODUCTION_DIVISOR) * 0.25 * max(0, eta - GALCON_TIMELINE_HORIZON)
                )
                need = math.ceil(float(ps[t_idx]) + extra) + 2.0
                if need <= 0.0 or need > avail[i]:
                    continue
                roi = (dst.production / max(1.0, PRODUCTION_DIVISOR)) * 22.0 / max(4.0, need + 0.60 * eta)
                if dst.owner != 0:
                    roi *= 1.65
                p_prior = float(ms_probs[i, j]) if ms_probs is not None else float(probs[i, 1 + j])
                sc = (0.40 + p_prior * 2.8) * roi
                if sc > best_sc:
                    best_sc = sc
                    best_j = j
                    best_need = need
            if best_j >= 0 and best_sc >= 1.2:
                pct = max(5, min(100, int(math.ceil(100.0 * best_need / max(1.0, rem_ships[i])))))
                actual = math.floor(rem_ships[i] * (pct / 100.0) + 1e-9)
                if actual >= 1:
                    orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[best_j], pct))
                    claimed.add(best_j)
                    avail[i] = max(0.0, avail[i] - actual)
                    rem_ships[i] = max(0.0, rem_ships[i] - actual)

        if not orders:
            return []
        orders.extend(base_redirs)
        return orders

    def _build_opp_direct_rush_plan(self, sim: GalconSim, opp_enc: EncodedGalconState) -> List[Order]:
        """Wave 12 soft-minimax fallback: 85% strike from opponent's strongest planet at our closest/weakest planet."""
        n = len(opp_enc.planet_ids)
        opp_owned = [i for i in range(n) if opp_enc.owned_mask[i] and opp_enc.planets[i].ships >= 25.0]
        my_planets = [j for j in range(n) if opp_enc.planets[j].owner not in (0, opp_enc.player)]
        if not opp_owned or not my_planets:
            return []
        src_i = max(opp_owned, key=lambda i: opp_enc.planets[i].ships)
        src = opp_enc.planets[src_i]
        tgt_j = min(my_planets, key=lambda j: sim.planet_eta_ticks(src, opp_enc.planets[j]) + 0.15 * opp_enc.planets[j].ships)
        if not opp_enc.legal_edges[src_i, tgt_j]:
            return []
        return [SendOrder(opp_enc.planet_ids[src_i], opp_enc.planet_ids[tgt_j], 85)]

    def _build_defense_and_pressure_plan(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        *,
        base_redirs: Sequence[Order] = (),
    ) -> List[Order]:
        """Wave 16 active midgame logistics: exact multi-helper defense + rear-to-front streaming + safe frontline pressure."""
        n = len(enc.planet_ids)
        my_idx = [i for i in range(n) if enc.owned_mask[i]]
        opp_idx = [j for j in range(n) if enc.planets[j].owner not in (0, enc.player)]
        if not my_idx or not opp_idx:
            return []
        futures = enc.futures or [
            project_planet_future(sim, p.id, enc.player, horizon=GALCON_TIMELINE_HORIZON) for p in enc.planets
        ]
        avail = np.zeros((n,), dtype=np.float64)
        rem_ships = np.array([float(p.ships) for p in enc.planets], dtype=np.float64)
        for i in my_idx:
            avail[i] = max(0.0, min(float(enc.safe_surplus[i]), float(enc.planets[i].ships) * 0.85))

        orders: List[Order] = []

        # 1. Multi-helper exact defense for EVERY threatened owned planet (closing the 5x undefended-fall gap)
        threatened: List[Tuple[float, int, int, float]] = []
        for j in my_idx:
            _, _, po, ps = futures[j]
            lost = next((t for t in range(len(po)) if po[t] <= 0.0), -1)
            if lost >= 0:
                deficit = float(ps[lost]) + 2.5
                if deficit > 0.0:
                    threatened.append((float(enc.planets[j].production) / max(1.0, float(lost + 1)), j, lost + 1, deficit))
        threatened.sort(key=lambda x: -x[0])

        for _, tgt_j, t_fall, need in threatened:
            dst = enc.planets[tgt_j]
            helpers = [
                (sim.planet_eta_ticks(enc.planets[i], dst), i)
                for i in my_idx
                if i != tgt_j and avail[i] >= 3.0 and enc.legal_edges[i, tgt_j] and sim.planet_eta_ticks(enc.planets[i], dst) <= t_fall
            ]
            helpers.sort(key=lambda x: x[0])
            if sum(avail[i] for _, i in helpers) < need * 0.75:
                continue
            rem_need = need
            for _, i in helpers:
                if rem_need <= 0.5 or rem_ships[i] < 2.0:
                    break
                # Pull up to 55% of helper's safe surplus per pass (or full surplus if single helper)
                cap = avail[i] if len(helpers) == 1 else max(3.0, avail[i] * 0.65)
                send_s = min(cap, rem_need)
                pct = max(5, min(100, int(math.ceil(100.0 * send_s / max(1.0, rem_ships[i])))))
                actual = math.floor(rem_ships[i] * (pct / 100.0) + 1e-9)
                if actual >= 1:
                    orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[tgt_j], pct))
                    avail[i] = max(0.0, avail[i] - actual)
                    rem_ships[i] = max(0.0, rem_ships[i] - actual)
                    rem_need -= actual

        # Skip passive midgame logistics spam on tick 0..8 when openings dominate
        if sim.tick <= 8 and not orders:
            return []

        # 2. Classify front vs rear planets & stream rear surplus forward + pressure closest enemy
        dist_en: Dict[int, float] = {}
        nearest_en: Dict[int, int] = {}
        for i in my_idx:
            pi = enc.planets[i]
            best_d = 1e9
            best_e = opp_idx[0]
            for ej in opp_idx:
                pe = enc.planets[ej]
                d = math.hypot(pi.x - pe.x, pi.y - pe.y)
                if d < best_d:
                    best_d = d
                    best_e = ej
            dist_en[i] = best_d
            nearest_en[i] = best_e

        for i in my_idx:
            if avail[i] < 6.0 or rem_ships[i] < 6.0:
                continue
            pi = enc.planets[i]
            pe = enc.planets[nearest_en[i]]
            d_direct = dist_en[i]
            # Check if a closer friendly relay exists along a collinear tunnel (const=1.22)
            best_relay = -1
            best_rd = 1e9
            for b in my_idx:
                if b == i or not enc.legal_edges[i, b] or dist_en[b] >= d_direct * 0.90:
                    continue
                pb = enc.planets[b]
                d1 = math.hypot(pi.x - pb.x, pi.y - pb.y)
                d2 = math.hypot(pb.x - pe.x, pb.y - pe.y)
                if d1 < d_direct and d2 < d_direct and (d1 + d2) < d_direct * 1.22 and d1 < best_rd:
                    best_rd = d1
                    best_relay = b
            if best_relay >= 0:
                send_s = min(avail[i] * 0.78, rem_ships[i] * 0.75)
                pct = max(10, min(90, int(round(100.0 * send_s / max(1.0, rem_ships[i])))))
                actual = math.floor(rem_ships[i] * (pct / 100.0) + 1e-9)
                if actual >= 3:
                    orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[best_relay], pct))
                    avail[i] = max(0.0, avail[i] - actual)
                    rem_ships[i] = max(0.0, rem_ships[i] - actual)
            elif sim.tick > 10:
                # Frontline planet: pressure nearest enemy (or weakest enemy within 1.25x distance)
                tgt_e = nearest_en[i]
                best_sc = -1e9
                for ej in opp_idx:
                    if not enc.legal_edges[i, ej]:
                        continue
                    eta = sim.planet_eta_ticks(pi, enc.planets[ej])
                    if eta > 18:
                        continue
                    t_idx = min(GALCON_TIMELINE_HORIZON - 1, max(0, eta - 1))
                    _, _, po, ps = futures[ej]
                    if po[t_idx] > 0.0:
                        continue
                    sc = (enc.planets[ej].production * 1.5) / max(4.0, float(ps[t_idx]) + 1.2 * eta)
                    if sc > best_sc:
                        best_sc = sc
                        tgt_e = ej
                if enc.legal_edges[i, tgt_e] and sim.planet_eta_ticks(pi, enc.planets[tgt_e]) <= 18:
                    send_s = min(avail[i] * 0.80, rem_ships[i] * 0.78)
                    pct = max(10, min(90, int(round(100.0 * send_s / max(1.0, rem_ships[i])))))
                    actual = math.floor(rem_ships[i] * (pct / 100.0) + 1e-9)
                    if actual >= 4:
                        orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[tgt_e], pct))
                        avail[i] = max(0.0, avail[i] - actual)
                        rem_ships[i] = max(0.0, rem_ships[i] - actual)

        if not orders:
            return []
        orders.extend(base_redirs)
        return orders

    def _predict_opponent_reply(
        self,
        sim: GalconSim,
        opp_enc: EncodedGalconState,
        op: np.ndarray,
        of: np.ndarray,
        oms_p: Optional[np.ndarray],
        oms_f: Optional[np.ndarray],
    ) -> List[Order]:
        """Model the opponent's response, adapting to observed multi-send expanders or early rushers."""
        if not opp_enc.planet_ids or not np.any(opp_enc.owned_mask):
            return []
        if self.use_multi_send and self.has_trained_multi_send and oms_p is not None and oms_f is not None:
            thr = max(0.32, self.multi_send_threshold - (0.08 if self._opp_multi_send_ema >= 0.30 else 0.0))
            return self._decode_multi_send_orders(
                sim, opp_enc, oms_p, oms_f, threshold=thr, include_redirects=False
            )
        if self.opponent_adaptive and self._opp_multi_send_ema >= 0.30:
            fan = self._build_fanout_plan(sim, opp_enc, op, max_targets_per_planet=4, reserve_ratio=0.12)
            if fan:
                return fan
        if self.opponent_adaptive and self._opp_rush_ema >= 0.35 and sim.tick <= 24:
            return self._decode_orders(
                sim, opp_enc, op, of, launch_threshold=max(0.28, self.launch_threshold - 0.14), frac_scale=1.20, include_redirects=False
            )
        return self._decode_orders(sim, opp_enc, op, of, launch_threshold=self.launch_threshold, include_redirects=False)

    @torch.no_grad()
    def act_with_search_info(
        self,
        sim: GalconSim,
        player: int,
    ) -> Tuple[List[Order], List[List[Order]], List[float], float]:
        """Run batched opponent-adaptive reactive lookahead search and return ``(best_plan, candidates, scores, best_score)``."""
        t0 = time.perf_counter()
        if self.opponent_adaptive:
            self._update_opp_profile(sim, player)
        enc = extract_galcon_features(sim, player, feature_version=self.feature_version)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return [], [], [], 0.0
        eff_budget = max(self.time_budget_sec, 0.30) if n >= 22 else self.time_budget_sec
        deadline = t0 + eff_budget

        redir_enc = (
            extract_galcon_redir_features(sim, enc)
            if (self.use_native_redir and self.has_trained_redir and sim.allow_redirect and sim.fleets)
            else None
        )
        probs, mean_fracs, root_val, redir_probs, ms_probs, ms_fracs = self._predict_all(enc, redir_enc=redir_enc)
        self._last_root_ms_probs = ms_probs
        self._last_root_val = float(root_val)
        using_ms = self.use_multi_send and self.has_trained_multi_send and ms_probs is not None and ms_fracs is not None

        raw_cands: List[List[Order]] = []
        if using_ms and ms_probs is not None and ms_fracs is not None:
            for thr in (
                self.multi_send_threshold,
                max(0.24, self.multi_send_threshold - 0.12),
                min(0.75, self.multi_send_threshold + 0.12),
            ):
                raw_cands.append(
                    self._decode_multi_send_orders(
                        sim, enc, ms_probs, ms_fracs, threshold=thr, redir_enc=redir_enc, redir_probs=redir_probs
                    )
                )
            for fs in (0.82, 1.18):
                raw_cands.append(
                    self._decode_multi_send_orders(
                        sim,
                        enc,
                        ms_probs,
                        ms_fracs,
                        threshold=self.multi_send_threshold,
                        frac_scale=fs,
                        redir_enc=redir_enc,
                        redir_probs=redir_probs,
                    )
                )
            # Flip top-3 most uncertain legal edges (closest to threshold)
            uncert: List[Tuple[float, int, int, bool]] = []
            for i in range(n):
                if not enc.owned_mask[i]:
                    continue
                for j in range(n):
                    if enc.legal_edges[i, j]:
                        p_ij = float(ms_probs[i, j])
                        if 0.18 <= p_ij <= 0.82:
                            uncert.append((abs(p_ij - self.multi_send_threshold), i, j, p_ij < self.multi_send_threshold))
            uncert.sort(key=lambda x: x[0])
            for _, i_e, j_e, new_val in uncert[:3]:
                raw_cands.append(
                    self._decode_multi_send_orders(
                        sim,
                        enc,
                        ms_probs,
                        ms_fracs,
                        threshold=self.multi_send_threshold,
                        edge_overrides={(i_e, j_e): new_val},
                        redir_enc=redir_enc,
                        redir_probs=redir_probs,
                    )
                )
            # Also include the single-target head plan as a conservative baseline candidate
            raw_cands.append(
                self._decode_orders(
                    sim, enc, probs, mean_fracs, launch_threshold=self.launch_threshold, redir_enc=redir_enc, redir_probs=redir_probs
                )
            )
        else:
            for thr in (
                self.launch_threshold,
                max(0.24, self.launch_threshold - 0.16),
                min(0.78, self.launch_threshold + 0.16),
            ):
                raw_cands.append(
                    self._decode_orders(
                        sim, enc, probs, mean_fracs, launch_threshold=thr, redir_enc=redir_enc, redir_probs=redir_probs
                    )
                )
            for fs in (0.80, 1.22):
                raw_cands.append(
                    self._decode_orders(
                        sim,
                        enc,
                        probs,
                        mean_fracs,
                        launch_threshold=self.launch_threshold,
                        frac_scale=fs,
                        redir_enc=redir_enc,
                        redir_probs=redir_probs,
                    )
                )
            entropies: List[Tuple[float, int, int]] = []
            for i in range(n):
                if not enc.owned_mask[i]:
                    continue
                p_row = np.clip(probs[i], 1e-6, 1.0)
                ent = float(-np.sum(p_row * np.log(p_row)))
                top2 = np.argsort(-p_row)[:2]
                alt_c = int(top2[1])
                if p_row[alt_c] >= 0.12:
                    entropies.append((ent, i, alt_c))
            entropies.sort(key=lambda x: -x[0])
            for _, p_idx, alt_c in entropies[:3]:
                raw_cands.append(
                    self._decode_orders(
                        sim,
                        enc,
                        probs,
                        mean_fracs,
                        launch_threshold=self.launch_threshold,
                        overrides={p_idx: alt_c},
                        redir_enc=redir_enc,
                        redir_probs=redir_probs,
                    )
                )

        # Neural-guided ROI multi-target fan-out proposals (evaluated alongside neural candidates)
        base_redirs = [o for o in raw_cands[0] if isinstance(o, RedirOrder)]
        prior_ms = ms_probs if using_ms else None
        fan_cons = self._build_fanout_plan(
            sim, enc, probs, ms_probs=prior_ms, max_targets_per_planet=3, reserve_ratio=0.25, base_redirs=base_redirs
        )
        if not fan_cons and self.enable_rescue_snipe:
            fan_cons = self._build_rescue_and_snipe_plan(
                sim, enc, probs, ms_probs=prior_ms, base_redirs=base_redirs
            )
        if fan_cons:
            raw_cands.append(fan_cons)
        fan_agg = self._build_fanout_plan(
            sim, enc, probs, ms_probs=prior_ms, max_targets_per_planet=5, reserve_ratio=0.10, base_redirs=base_redirs
        )
        if fan_agg:
            raw_cands.append(fan_agg)
        if sim.tick <= 20 and n >= 16:
            fan_open = self._build_fanout_plan(
                sim, enc, probs, ms_probs=prior_ms, max_targets_per_planet=6, reserve_ratio=0.06, base_redirs=base_redirs
            )
            if fan_open:
                raw_cands.append(fan_open)

        use_adv = self.adversarial_search and (n < 22)
        if use_adv:
            ff_plan = self._build_focus_fire_plan(
                sim, enc, probs, ms_probs=prior_ms, max_sources=4, base_redirs=base_redirs
            )
            if ff_plan:
                raw_cands.append(ff_plan)
            tun_plan = self._build_tunnel_feed_plan(sim, enc, base_redirs=base_redirs)
            if tun_plan:
                raw_cands.append(tun_plan)
                base_srcs = {o.src_id for o in raw_cands[0] if isinstance(o, SendOrder)}
                extra_tun = [o for o in tun_plan if isinstance(o, SendOrder) and o.src_id not in base_srcs]
                if extra_tun:
                    raw_cands.append(list(raw_cands[0]) + extra_tun)

        # Defensive Hold-All / rush-guard candidate early or when threatened
        any_threat = any(
            enc.planets[i].owner == player and enc.safe_surplus[i] < enc.planets[i].ships * 0.5
            for i in range(n)
        )
        if sim.tick <= 24 or any_threat:
            raw_cands.append([])

        # Smart redirect-augmented vs no-redirect comparison when friendly fleets are flying
        smart_redirs: List[RedirOrder] = []
        if sim.allow_redirect and any(fl.owner == player for fl in sim.fleets.values()):
            smart_redirs, _ = compute_smart_redirects(sim, enc, verify_rollout=False)
            if smart_redirs:
                base_sends = [o for o in raw_cands[0] if isinstance(o, SendOrder)]
                raw_cands.append(base_sends + list(smart_redirs))
                raw_cands.append(base_sends)

        if self.enable_def_stream:
            def_plan = self._build_defense_and_pressure_plan(
                sim, enc, base_redirs=(smart_redirs if smart_redirs else base_redirs)
            )
            if def_plan:
                raw_cands.append(def_plan)

        # Deduplicate preserving order (candidates[0] is always the base pure policy)
        seen = set()
        candidates: List[List[Order]] = []
        for plan in raw_cands:
            key = tuple(repr(o) for o in plan)
            if key not in seen:
                seen.add(key)
                candidates.append(plan)

        if len(candidates) <= 1 or time.perf_counter() >= deadline - 0.015:
            return (candidates[0] if candidates else []), candidates, [root_val], root_val

        # Predict opponent's immediate tick-0 response (1x fast forward pass, opponent-adaptive)
        opp = 2 if player == 1 else 1
        opp_enc = extract_galcon_features(sim, opp, feature_version=self.feature_version)
        opp_orders: List[Order] = []
        opp_adv_orders: List[Order] = []
        if opp_enc.planet_ids and np.any(opp_enc.owned_mask):
            op, of, _, _, oms_p, oms_f = self._predict_all(
                opp_enc, use_d4_tta=False, need_value=False, ms_only=(using_ms and not use_adv)
            )
            opp_orders = self._predict_opponent_reply(sim, opp_enc, op, of, oms_p, oms_f)
            if use_adv:
                opp_adv_orders = self._build_fanout_plan(
                    sim, opp_enc, op, max_targets_per_planet=4, reserve_ratio=0.10
                )
                if not opp_adv_orders:
                    opp_adv_orders = self._build_focus_fire_plan(sim, opp_enc, op, max_sources=3)
                if not opp_adv_orders and self.enable_opp_direct_rush:
                    opp_adv_orders = self._build_opp_direct_rush_plan(sim, opp_enc)

        # Wave 8/9 map-adaptive lookahead: on Large (>=22pl) maps, extend rollout to 22 ticks and 3 reactive plies
        # (66.7% Large 24pl Asym / 50.0% Paired vs Sparky); on <22pl Std/Compact, pair 2-ply h=16 with
        # adversarial focus-fire/soft-minimax (+10% Mixed, +2.5% 18pl Std -> 65.0% @ ~50ms/tick).
        eff_horizon = max(self.rollout_horizon, 22) if n >= 22 else self.rollout_horizon
        eff_plies = max(self.num_plies, 3) if (n >= 22 and self.reactive_mid_ply) else self.num_plies
        ply_steps = [
            max(1, int(round(eff_horizon * (k + 1) / eff_plies)))
            for k in range(eff_plies)
        ]
        ply_steps[-1] = eff_horizon

        eval_cands: List[List[Order]] = []
        rolls: List[GalconSim] = []

        # Ply 1: step each candidate to ply_steps[0]
        for idx, plan in enumerate(candidates):
            if idx > 0 and time.perf_counter() >= deadline - 0.025:
                break
            roll = sim.clone()
            roll.step({player: plan, opp: opp_orders})
            for _ in range(1, ply_steps[0]):
                if roll.done:
                    break
                roll.step({})
            eval_cands.append(plan)
            rolls.append(roll)

        # Subsequent reactive plies (Ply 2 .. num_plies): ONE batched forward pass per ply
        for p_idx in range(1, len(ply_steps)):
            prev_t = ply_steps[p_idx - 1]
            next_t = ply_steps[p_idx]
            if next_t <= prev_t or time.perf_counter() >= deadline - 0.015:
                break
            live_indices = [i for i, r in enumerate(rolls) if not r.done]
            if not live_indices:
                break
            mid_encs: List[EncodedGalconState] = []
            for i in live_indices:
                mid_encs.append(extract_galcon_features(rolls[i], player, feature_version=self.feature_version))
                mid_encs.append(extract_galcon_features(rolls[i], opp, feature_version=self.feature_version))
            m_probs, m_fracs, _, m_msp, m_msf = self._batch_forward_policy_and_val(
                mid_encs, need_value=False, ms_only=using_ms
            )
            for k_idx, r_idx in enumerate(live_indices):
                r = rolls[r_idx]
                p_enc = mid_encs[2 * k_idx]
                o_enc = mid_encs[2 * k_idx + 1]
                if using_ms and m_msp is not None and m_msf is not None:
                    p_ord = self._decode_multi_send_orders(
                        r, p_enc, m_msp[2 * k_idx], m_msf[2 * k_idx],
                        threshold=self.multi_send_threshold, include_redirects=False,
                    )
                else:
                    p_ord = self._decode_orders(
                        r, p_enc, m_probs[2 * k_idx], m_fracs[2 * k_idx],
                        launch_threshold=self.launch_threshold, include_redirects=False,
                    )
                o_ord = self._predict_opponent_reply(
                    r,
                    o_enc,
                    m_probs[2 * k_idx + 1],
                    m_fracs[2 * k_idx + 1],
                    m_msp[2 * k_idx + 1] if m_msp is not None else None,
                    m_msf[2 * k_idx + 1] if m_msf is not None else None,
                )
                r.step({player: p_ord, opp: o_ord})
                for _ in range(prev_t + 1, next_t):
                    if r.done:
                        break
                    r.step({})

        # Leaf evaluation: ONE batched forward pass across all non-terminal leaves
        h_vals: List[float] = []
        done_flags: List[bool] = []
        leaf_encs: List[EncodedGalconState] = []
        leaf_pos: List[int] = []
        for idx, roll in enumerate(rolls):
            h_vals.append(evaluate_galcon_state(roll, player))
            done_flags.append(roll.done)
            if not roll.done:
                leaf_pos.append(idx)
                leaf_encs.append(extract_galcon_features(roll, player, feature_version=self.feature_version))

        nn_vals = self._batch_evaluate_leaves(leaf_encs)
        nn_map = {pos: float(nn_vals[i]) for i, pos in enumerate(leaf_pos)}

        scores: List[float] = []
        for i in range(len(eval_cands)):
            if done_flags[i]:
                sc = h_vals[i]
            else:
                sc = self.neural_value_weight * nn_map[i] + (1.0 - self.neural_value_weight) * h_vals[i]
            scores.append(sc)

        # Wave 5/9 adversarial soft-minimax check (on <22pl Std/Compact): stress-test top-4 candidates against aggressive enemy fan-out/focus-fire
        if (
            use_adv
            and opp_adv_orders
            and len(eval_cands) > 1
            and time.perf_counter() < deadline - 0.015
        ):
            top_k = sorted(range(len(eval_cands)), key=lambda i: -(scores[i] + (self.prior_bonus if i == 0 else 0.0)))[:4]
            adv_rolls: List[GalconSim] = []
            adv_h: List[float] = []
            adv_done: List[bool] = []
            adv_encs: List[EncodedGalconState] = []
            adv_pos: List[int] = []
            adv_horizon = max(4, ply_steps[0])
            for loc_i, cand_i in enumerate(top_k):
                r_adv = sim.clone()
                r_adv.step({player: eval_cands[cand_i], opp: opp_adv_orders})
                for _ in range(1, adv_horizon):
                    if r_adv.done:
                        break
                    r_adv.step({})
                adv_rolls.append(r_adv)
                adv_h.append(evaluate_galcon_state(r_adv, player))
                adv_done.append(r_adv.done)
                if not r_adv.done:
                    adv_pos.append(loc_i)
                    adv_encs.append(extract_galcon_features(r_adv, player, feature_version=self.feature_version))
            adv_nn = self._batch_evaluate_leaves(adv_encs)
            adv_nn_map = {pos: float(adv_nn[idx_l]) for idx_l, pos in enumerate(adv_pos)}
            for loc_i, cand_i in enumerate(top_k):
                s_adv = (
                    adv_h[loc_i]
                    if adv_done[loc_i]
                    else (self.neural_value_weight * adv_nn_map[loc_i] + (1.0 - self.neural_value_weight) * adv_h[loc_i])
                )
                scores[cand_i] = 0.65 * scores[cand_i] + 0.35 * min(scores[cand_i], s_adv)

        # Wave 16 opening deep-strike defensive soft-minimax guard:
        # Exposes suicidal long-distance opening rushes (eta >= 14 > rollout_horizon) by testing top candidates
        # against a calm defending opponent (opp=[]) out to min(28, max_eta + 2) ticks.
        if (
            self.enable_opening_def_guard
            and sim.tick <= 8
            and len(eval_cands) > 1
            and time.perf_counter() < deadline - 0.015
        ):
            max_strike_eta = 0
            for plan in eval_cands:
                for o in plan:
                    if isinstance(o, SendOrder) and o.percent >= 45:
                        sp = sim.planets.get(o.src_id)
                        dp = sim.planets.get(o.dst_id)
                        if sp is not None and dp is not None and dp.owner not in (0, player):
                            eta_sd = sim.planet_eta_ticks(sp, dp)
                            if eta_sd > max_strike_eta:
                                max_strike_eta = eta_sd
            if max_strike_eta >= 14:
                def_horizon = min(28, max_strike_eta + 2)
                top_k_def = sorted(
                    range(len(eval_cands)),
                    key=lambda i: -(scores[i] + (self.prior_bonus if i == 0 else 0.0)),
                )[:6]
                def_h: List[float] = []
                def_done: List[bool] = []
                def_encs: List[EncodedGalconState] = []
                def_pos: List[int] = []
                for loc_i, cand_i in enumerate(top_k_def):
                    r_def = sim.clone()
                    r_def.step({player: eval_cands[cand_i], opp: []})
                    for _ in range(1, def_horizon):
                        if r_def.done:
                            break
                        r_def.step({})
                    def_h.append(evaluate_galcon_state(r_def, player))
                    def_done.append(r_def.done)
                    if not r_def.done:
                        def_pos.append(loc_i)
                        def_encs.append(extract_galcon_features(r_def, player, feature_version=self.feature_version))
                def_nn = self._batch_evaluate_leaves(def_encs)
                def_nn_map = {pos: float(def_nn[idx_l]) for idx_l, pos in enumerate(def_pos)}
                for loc_i, cand_i in enumerate(top_k_def):
                    s_def = (
                        def_h[loc_i]
                        if def_done[loc_i]
                        else (self.neural_value_weight * def_nn_map[loc_i] + (1.0 - self.neural_value_weight) * def_h[loc_i])
                    )
                    scores[cand_i] = 0.55 * scores[cand_i] + 0.45 * min(scores[cand_i], s_def)

        best_idx = 0
        best_score = -1e9
        for i in range(len(eval_cands)):
            eff_sc = scores[i] + (self.prior_bonus if i == 0 else 0.0)
            if eff_sc > best_score:
                best_score = eff_sc
                best_idx = i

        return eval_cands[best_idx], eval_cands, scores, scores[best_idx]

    @torch.no_grad()
    def act(self, sim: GalconSim, player: int) -> List[Order]:
        best_plan, _, _, _ = self.act_with_search_info(sim, player)
        return best_plan


class EnsembleGalconSearchAgent(GalconSearchTransformerAgent):
    """Blends 2+ ``GalconGen4Transformer`` checkpoints in exact 1x Canonical D4 for Pure or Multi-Ply Search."""

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        *,
        weights: Optional[Sequence[float]] = None,
        value_weights: Optional[Sequence[float]] = None,
        enable_rescue_snipe: bool = True,
        enable_opp_direct_rush: bool = True,
        **kwargs,
    ) -> None:
        paths = [Path(p) for p in checkpoint_paths]
        if not paths:
            raise ValueError("EnsembleGalconSearchAgent requires at least one checkpoint path.")
        super().__init__(
            paths[0],
            enable_rescue_snipe=enable_rescue_snipe,
            enable_opp_direct_rush=enable_opp_direct_rush,
            **kwargs,
        )
        self.ensemble_agents: List[GalconTransformerAgent] = [self]
        for p in paths[1:]:
            self.ensemble_agents.append(
                GalconTransformerAgent(
                    p,
                    launch_threshold=self.launch_threshold,
                    redir_threshold=self.redir_threshold,
                    multi_send_threshold=self.multi_send_threshold,
                    max_sends_per_planet=self.max_sends_per_planet,
                    use_d4_tta=self.use_d4_tta,
                    use_d4_canonical=self.use_d4_canonical,
                    enable_redirect_overlay=self.enable_redirect_overlay,
                    use_native_redir=self.use_native_redir,
                    use_multi_send=self.use_multi_send,
                    contempt=getattr(self.model, "contempt", None) if isinstance(self.model, GalconGen5Transformer) else None,
                    device=str(self.device),
                )
            )
        if weights is None:
            w = np.ones((len(self.ensemble_agents),), dtype=np.float32) / float(len(self.ensemble_agents))
        else:
            w = np.array(weights, dtype=np.float32)
            w = w / float(np.sum(w))
        self.ensemble_weights = w
        if value_weights is None:
            vw = w.copy()
        else:
            vw = np.array(value_weights, dtype=np.float32)
            vw = vw / float(np.sum(vw))
        self.value_weights = vw

    @torch.no_grad()
    def _predict_all(
        self,
        enc: EncodedGalconState,
        *,
        redir_enc: Optional[EncodedRedirState] = None,
        use_d4_tta: Optional[bool] = None,
        use_d4_canonical: Optional[bool] = None,
        need_value: bool = True,
        ms_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        p_sum = f_sum = r_sum = msp_sum = msf_sum = None
        r_w = ms_w = 0.0
        v_sum = 0.0
        for idx, ag in enumerate(self.ensemble_agents):
            w = float(self.ensemble_weights[idx])
            vw = float(self.value_weights[idx]) if need_value else 0.0
            if w <= 0.0 and vw <= 0.0:
                continue
            ag_need_v = vw > 0.0
            if idx == 0:
                p, f, v, r, msp, msf = super()._predict_all(
                    enc,
                    redir_enc=redir_enc,
                    use_d4_tta=use_d4_tta,
                    use_d4_canonical=use_d4_canonical,
                    need_value=ag_need_v,
                    ms_only=ms_only,
                )
            else:
                p, f, v, r, msp, msf = ag._predict_all(
                    enc,
                    redir_enc=redir_enc,
                    use_d4_tta=use_d4_tta,
                    use_d4_canonical=use_d4_canonical,
                    need_value=ag_need_v,
                    ms_only=ms_only,
                )
            if w > 0.0:
                p_sum = w * p if p_sum is None else p_sum + w * p
                f_sum = w * f if f_sum is None else f_sum + w * f
                if r is not None:
                    r_sum = w * r if r_sum is None else r_sum + w * r
                    r_w += w
                if msp is not None and msf is not None:
                    msp_sum = w * msp if msp_sum is None else msp_sum + w * msp
                    msf_sum = w * msf if msf_sum is None else msf_sum + w * msf
                    ms_w += w
            if vw > 0.0:
                v_sum += vw * float(v)
        if r_sum is not None and r_w > 0.0:
            r_sum = r_sum / r_w
        if msp_sum is not None and msf_sum is not None and ms_w > 0.0:
            msp_sum = msp_sum / ms_w
            msf_sum = msf_sum / ms_w
        assert p_sum is not None and f_sum is not None
        return p_sum, f_sum, v_sum, r_sum, msp_sum, msf_sum

    @torch.no_grad()
    def _batch_forward_policy_and_val(
        self, encs: Sequence[EncodedGalconState], *, need_value: bool = True, ms_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run one batched forward pass per ensemble member and blend predictions."""
        k = len(encs)
        if k == 0:
            return super()._batch_forward_policy_and_val(encs, need_value=need_value, ms_only=ms_only)
        n = len(encs[0].planet_ids)
        pf_t = torch.from_numpy(np.stack([self._prep_pf(e) for e in encs], axis=0)).to(self.device)
        tf_t = torch.from_numpy(np.stack([e.timeline_feats for e in encs], axis=0)).to(self.device)
        ef_t = torch.from_numpy(np.stack([e.edge_feats for e in encs], axis=0)).to(self.device)
        le_t = torch.from_numpy(np.stack([e.legal_edges for e in encs], axis=0)).to(self.device)
        pm_t = torch.ones((k, n), dtype=torch.bool, device=self.device)
        mask = torch.cat([torch.ones((k, n, 1), dtype=torch.bool, device=self.device), le_t], dim=-1)

        p_sum = f_sum = msp_sum = msf_sum = None
        v_sum = np.zeros((k,), dtype=np.float32)
        ms_w = 0.0
        for idx, ag in enumerate(self.ensemble_agents):
            w = float(self.ensemble_weights[idx])
            vw = float(self.value_weights[idx]) if need_value else 0.0
            if w <= 0.0 and vw <= 0.0:
                continue
            want_ms = (w > 0.0) and isinstance(ag.model, _MULTI_SEND_MODELS) and ag.model.enable_multi_send
            if ms_only and want_ms:
                if isinstance(ag.model, GalconGen5Transformer):
                    h, e_hid = ag.model.encode_planets_and_edges(pf_t, tf_t, ef_t, pm_t)
                    ms_l, ms_f = ag.model.predict_multi_send(h, ef_t, edge_hidden=e_hid)
                else:
                    h = ag.model.encode_planets(pf_t, tf_t, ef_t, pm_t)
                    ms_l, ms_f = ag.model.predict_multi_send(h, ef_t)
                ms_l = ms_l.masked_fill(~le_t, -1e9)
                ms_p = torch.sigmoid(ms_l).cpu().numpy()
                ms_f_np = ms_f.cpu().numpy()
                msp_sum = w * ms_p if msp_sum is None else msp_sum + w * ms_p
                msf_sum = w * ms_f_np if msf_sum is None else msf_sum + w * ms_f_np
                ms_w += w
                if vw > 0.0:
                    v_sum += vw * ag.model.decode_value_from_hidden(h, pm_t).cpu().numpy()
                continue
            if want_ms:
                if isinstance(ag.model, GalconGen5Transformer):
                    h, e_hid = ag.model.encode_planets_and_edges(pf_t, tf_t, ef_t, pm_t)
                    ms_l, ms_f = ag.model.predict_multi_send(h, ef_t, edge_hidden=e_hid)
                    vals = ag.model.decode_value_from_hidden(h, pm_t)
                    logits = torch.cat([torch.zeros_like(ms_l[:, :, :1]), ms_l], dim=-1)
                    fracs = ms_f
                else:
                    logits, fracs, vals, h = ag.model(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
                    ms_l, ms_f = ag.model.predict_multi_send(h, ef_t)
                ms_l = ms_l.masked_fill(~le_t, -1e9)
                ms_p = torch.sigmoid(ms_l).cpu().numpy()
                ms_f_np = ms_f.cpu().numpy()
                msp_sum = w * ms_p if msp_sum is None else msp_sum + w * ms_p
                msf_sum = w * ms_f_np if msf_sum is None else msf_sum + w * ms_f_np
                ms_w += w
            else:
                logits, fracs, vals = ag.model(pf_t, tf_t, ef_t, pm_t)
            if w > 0.0:
                logits = logits.masked_fill(~mask, -1e9)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                fr_np = fracs.cpu().numpy()
                p_sum = w * probs if p_sum is None else p_sum + w * probs
                f_sum = w * fr_np if f_sum is None else f_sum + w * fr_np
            if vw > 0.0:
                v_sum += vw * vals.cpu().numpy()
        if msp_sum is not None and msf_sum is not None and ms_w > 0.0:
            msp_sum = msp_sum / ms_w
            msf_sum = msf_sum / ms_w
        if p_sum is None:
            p_sum = np.zeros((k, n, n + 1), dtype=np.float32)
        if f_sum is None:
            f_sum = np.zeros((k, n, n), dtype=np.float32)
        return p_sum, f_sum, v_sum, msp_sum, msf_sum

    @torch.no_grad()
    def _batch_evaluate_leaves(self, leaf_encs: Sequence[EncodedGalconState]) -> np.ndarray:
        k = len(leaf_encs)
        if k == 0:
            return np.zeros((0,), dtype=np.float32)
        n = len(leaf_encs[0].planet_ids)
        pf_t = torch.from_numpy(np.stack([self._prep_pf(e) for e in leaf_encs], axis=0)).to(self.device)
        tf_t = torch.from_numpy(np.stack([e.timeline_feats for e in leaf_encs], axis=0)).to(self.device)
        ef_t = torch.from_numpy(np.stack([e.edge_feats for e in leaf_encs], axis=0)).to(self.device)
        pm_t = torch.ones((k, n), dtype=torch.bool, device=self.device)
        v_tot = np.zeros((k,), dtype=np.float32)
        for idx, ag in enumerate(self.ensemble_agents):
            vw = float(self.value_weights[idx])
            if vw <= 0.0:
                continue
            if isinstance(ag.model, _MULTI_SEND_MODELS):
                h = ag.model.encode_planets(pf_t, tf_t, ef_t, pm_t)
                vals = ag.model.decode_value_from_hidden(h, pm_t)
            else:
                _, _, vals = ag.model(pf_t, tf_t, ef_t, pm_t)
            v_tot += vw * vals.cpu().numpy()
        return v_tot

    @torch.no_grad()
    def act_pure(self, sim: GalconSim, player: int) -> List[Order]:
        """Pure ensemble forward pass without multi-ply search."""
        enc = extract_galcon_features(sim, player, feature_version=self.feature_version)
        if not enc.planet_ids or not np.any(enc.owned_mask):
            return []
        redir_enc = (
            extract_galcon_redir_features(sim, enc)
            if (self.use_native_redir and self.has_trained_redir and sim.allow_redirect and sim.fleets)
            else None
        )
        want_ms_only = bool(self.use_multi_send and self.has_trained_multi_send)
        probs, mean_fracs, _, redir_probs, ms_probs, ms_fracs = self._predict_all(
            enc, redir_enc=redir_enc, need_value=False, ms_only=want_ms_only
        )
        if self.use_multi_send and self.has_trained_multi_send and ms_probs is not None and ms_fracs is not None:
            return self._decode_multi_send_orders(
                sim, enc, ms_probs, ms_fracs, threshold=self.multi_send_threshold, redir_enc=redir_enc, redir_probs=redir_probs
            )
        return self._decode_orders(
            sim, enc, probs, mean_fracs, launch_threshold=self.launch_threshold, redir_enc=redir_enc, redir_probs=redir_probs
        )
