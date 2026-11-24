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

from galcon.sim import (
    PRODUCTION_DIVISOR,
    GalconSim,
    Order,
    Planet,
    RedirOrder,
    SendOrder,
    evaluate_galcon_state,
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
    w_me = is_me * (0.25 + prod + ships)
    w_en = is_en * (0.25 + prod + ships)
    sum_me = float(w_me.sum())
    sum_en = float(w_en.sum())

    best_key = None
    best_arr = None
    for flip_x in (False, True):
        for k_rot in range(4):
            cand = apply_galcon_d4(planet_feats, k_rot=k_rot, flip_x=flip_x)
            cx = cand[:, 0]
            cy = cand[:, 1]
            mx = float((w_me * cx).sum() / sum_me) if sum_me > 1e-6 else 0.0
            my = float((w_me * cy).sum() / sum_me) if sum_me > 1e-6 else 0.0
            ex = float((w_en * cx).sum() / sum_en) if sum_en > 1e-6 else 0.0
            ey = float((w_en * cy).sum() / sum_en) if sum_en > 1e-6 else 0.0
            vx, vy = mx - ex, my - ey
            if math.hypot(vx, vy) < 1e-4:
                vx, vy = mx, my
            # Canonical octant: vx <= 0, vy <= 0, |vx| >= |vy|; exact tie-break via sorted planet coords
            oct_key = (
                0 if vx <= 1e-5 else 1,
                0 if vy <= 1e-5 else 1,
                0 if abs(vx) >= abs(vy) - 1e-5 else 1,
                round(vx, 4),
                round(vy, 4),
            )
            tie_key = tuple(
                sorted(
                    (round(float(cx[i]), 3), round(float(cy[i]), 3), round(float(is_me[i] - is_en[i]), 1), round(float(prod[i]), 2))
                    for i in range(cand.shape[0])
                )
            )
            full_key = (oct_key, tie_key)
            if best_key is None or full_key < best_key:
                best_key = full_key
                best_arr = cand
    assert best_arr is not None
    return best_arr


def extract_galcon_features(
    sim: GalconSim,
    player: int,
    *,
    feature_version: int = 1,
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

    # Scale-invariant centering works identically on 200x125, 400x240, or 600x400 gbots maps
    xs = np.array([p.x for p in planets], dtype=np.float32)
    ys = np.array([p.y for p in planets], dtype=np.float32)
    cx = 0.5 * (float(xs.min()) + float(xs.max()))
    cy = 0.5 * (float(ys.min()) + float(ys.max()))
    half_span = max(50.0, 0.5 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min())))

    opp = 2 if player == 1 else 1
    my_tot_ships = sim.total_ships(player)
    en_tot_ships = sim.total_ships(opp)
    my_tot_prod = sim.total_production(player)
    en_tot_prod = sim.total_production(opp)
    ship_ratio = (my_tot_ships - en_tot_ships) / max(30.0, my_tot_ships + en_tot_ships)
    prod_ratio = (my_tot_prod - en_tot_prod) / max(40.0, my_tot_prod + en_tot_prod)
    step_prog = min(1.0, float(sim.tick) / max(1.0, float(sim.max_ticks)))

    pf = np.zeros((n, p_dim), dtype=np.float32)
    tf = np.zeros((n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32)
    ef = np.zeros((n, n, e_dim), dtype=np.float32)
    legal = np.zeros((n, n), dtype=bool)
    owned_mask = np.zeros((n,), dtype=bool)
    safe_surplus = np.zeros((n,), dtype=np.float32)

    futures = [project_planet_future(sim, p.id, player, horizon=GALCON_TIMELINE_HORIZON) for p in planets]

    my_indices = [i for i, p in enumerate(planets) if p.owner == player]
    en_indices = [i for i, p in enumerate(planets) if p.owner != 0 and p.owner != player]

    etas = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                etas[i, j] = float(sim.planet_eta_ticks(planets[i], planets[j]))

    for i, p in enumerate(planets):
        my_arr, en_arr, po, ps = futures[i]
        is_mine = 1.0 if p.owner == player else 0.0
        is_en = 1.0 if (p.owner != 0 and p.owner != player) else 0.0
        is_neu = 1.0 if p.owner == 0 else 0.0
        owned_mask[i] = p.owner == player and p.ships >= 2.0

        is_threatened = 0.0
        threat_urgency = 0.0
        deficit_log = 0.0
        if p.owner == player:
            lost_idx = next((t for t in range(GALCON_TIMELINE_HORIZON) if po[t] <= 0), -1)
            if lost_idx >= 0:
                safe_surplus[i] = 0.0
                is_threatened = 1.0
                threat_urgency = 1.0 / (1.0 + 0.25 * float(lost_idx + 1))
                deficit_log = math.log1p(max(0.0, float(ps[lost_idx]) + 2.0)) / 5.5
            else:
                min_hold = float(ps.min())
                safe_surplus[i] = max(0.0, min(float(p.ships), min_hold))

        min_en_eta = min((etas[i, j] for j in en_indices), default=40.0)
        min_my_eta = min((etas[i, j] for j in my_indices if j != i), default=40.0)
        frontline = (min_my_eta - min_en_eta) / max(10.0, min_my_eta + min_en_eta)

        base_feats = [
            (p.x - cx) / half_span,
            (p.y - cy) / half_span,
            is_mine,
            is_en,
            is_neu,
            math.log1p(max(0.0, p.ships)) / 5.5,
            p.production / 100.0,
            p.radius / 35.0,
            math.log1p(safe_surplus[i]) / 5.5,
            math.log1p(float(my_arr.sum())) / 5.5,
            math.log1p(float(en_arr.sum())) / 5.5,
            float(po[-1]) * (math.log1p(max(0.0, float(ps[-1]))) / 5.5),
            min(1.0, min_en_eta / 35.0),
            min(1.0, min_my_eta / 35.0),
            frontline,
            ship_ratio,
            prod_ratio,
            step_prog,
        ]
        if feature_version >= 2:
            # Explicit rush-threat signal: could the nearest enemy planet all-in kill us?
            rush_threat = 0.0
            if p.owner == player and en_indices:
                nearest_en_idx = min(en_indices, key=lambda j: etas[j, i])
                en_p = planets[nearest_en_idx]
                eta_e = etas[nearest_en_idx, i]
                growth_dur = (p.production / PRODUCTION_DIVISOR) * sim.tick_dt * eta_e
                rush_threat = max(0.0, min(2.0, (en_p.ships * 0.85 - p.ships - growth_dur) / 60.0))
            base_feats.extend([is_threatened, threat_urgency, deficit_log, rush_threat])
        pf[i] = np.array(base_feats, dtype=np.float32)

        for t in range(GALCON_TIMELINE_HORIZON):
            tf[i, t, 0] = math.log1p(float(my_arr[t])) / 5.0
            tf[i, t, 1] = math.log1p(float(en_arr[t])) / 5.0
            tf[i, t, 2] = 1.0 if po[t] > 0 else 0.0
            tf[i, t, 3] = 1.0 if po[t] < 0 else 0.0
            tf[i, t, 4] = 1.0 if po[t] == 0 else 0.0
            tf[i, t, 5] = math.log1p(max(0.0, float(ps[t]))) / 5.5
            tf[i, t, 6] = 1.0 if (t > 0 and po[t] != po[t - 1]) else 0.0
            tf[i, t, 7] = float(t) / float(GALCON_TIMELINE_HORIZON)

    # Edge features (100% D4-invariant!)
    for i in range(n):
        src = planets[i]
        for j in range(n):
            if i == j:
                continue
            legal[i, j] = True
            dst = planets[j]
            eta = etas[i, j]
            _, _, po_j, ps_j = futures[j]
            t_idx = min(GALCON_TIMELINE_HORIZON - 1, max(0, int(eta) - 1))
            owner_eta = float(po_j[t_idx])
            ships_eta = float(ps_j[t_idx])
            extra = (
                (dst.production / PRODUCTION_DIVISOR) * sim.tick_dt * max(0.0, eta - float(GALCON_TIMELINE_HORIZON))
                if owner_eta < 0
                else 0.0
            )
            need = ships_eta + extra + 1.5
            can_capture = 1.0 if (owner_eta <= 0 and src.ships > need) else 0.0
            safe_capture = 1.0 if (owner_eta <= 0 and safe_surplus[i] > need) else 0.0
            steal_flag = 1.0 if (dst.owner == 0 and owner_eta < 0) else 0.0
            defend_flag = 1.0 if (dst.owner == player and owner_eta <= 0) else 0.0
            prod_rate = dst.production / PRODUCTION_DIVISOR
            roi = (prod_rate * 20.0) / max(3.0, need + 0.45 * eta * max(0.3, prod_rate))
            en_eta_src = pf[i, 12]
            en_eta_dst = pf[j, 12]

            edge_vec = [
                min(1.0, eta / 35.0),
                1.0 / (1.0 + 0.12 * eta),
                min(2.0, float(src.ships) / max(1.0, need)),
                can_capture,
                safe_capture,
                steal_flag,
                defend_flag,
                min(2.0, roi),
                owner_eta,
                en_eta_src - en_eta_dst,  # positive => moving toward enemy frontline
            ]
            if feature_version >= 2:
                exact_frac_needed = min(1.0, need / max(1.0, float(src.ships)))
                rush_punish = 1.0 if (dst.owner != 0 and dst.owner != player and dst.production >= 70.0 and src.ships > need) else 0.0
                edge_vec.extend([exact_frac_needed, rush_punish])
            ef[i, j] = np.array(edge_vec, dtype=np.float32)

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
    futures = enc.futures or [
        project_planet_future(sim, p.id, enc.player, horizon=GALCON_TIMELINE_HORIZON) for p in enc.planets
    ]

    # Precompute threatened status per planet
    threat_eta = np.full((n,), -1, dtype=np.int32)
    for j, p in enumerate(enc.planets):
        if p.owner == enc.player:
            _, _, po, _ = futures[j]
            lost = next((t for t in range(GALCON_TIMELINE_HORIZON) if po[t] <= 0), -1)
            if lost >= 0:
                threat_eta[j] = lost + 1

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

    f_cnt = len(my_fleets)
    fleet_ids: List[int] = []
    cur_dst_idx = np.zeros((f_cnt,), dtype=np.int64)
    rf = np.zeros((f_cnt, n, GALCON_REDIR_DIM), dtype=np.float32)

    for fi, fl in enumerate(my_fleets):
        fleet_ids.append(fl.id)
        c_idx = id_to_idx[fl.target]
        cur_dst_idx[fi] = c_idx
        c_dst = enc.planets[c_idx]
        eta_cur = float(sim.eta_ticks(fl.x, fl.y, c_dst))
        _, _, c_po, c_ps = futures[c_idx]
        t_cur = min(GALCON_TIMELINE_HORIZON - 1, max(0, int(eta_cur) - 1))
        wasted_friendly = c_dst.owner == enc.player and c_po[-1] > 0 and c_ps[-1] > fl.ships + 15.0
        doomed_attack = c_dst.owner != enc.player and c_po[t_cur] <= 0 and fl.ships < c_ps[t_cur] * 0.80
        cur_wasted = 1.0 if (wasted_friendly or doomed_attack) else 0.0
        ships_log = math.log1p(max(0.0, fl.ships)) / 5.5

        for j, cand in enumerate(enc.planets):
            eta_j = float(sim.eta_ticks(fl.x, fl.y, cand))
            _, _, po_j, ps_j = futures[j]
            t_j = min(GALCON_TIMELINE_HORIZON - 1, max(0, int(eta_j) - 1))
            owner_eta = float(po_j[t_j])
            ships_eta = float(ps_j[t_j])
            need_j = ships_eta + 1.5
            can_cap = 1.0 if (owner_eta <= 0 and fl.ships > need_j) else 0.0
            can_save = 1.0 if (threat_eta[j] > 0 and eta_j <= threat_eta[j]) else 0.0
            prod_rate = cand.production / PRODUCTION_DIVISOR
            roi_j = (prod_rate * 20.0) / max(3.0, need_j + 0.45 * eta_j * max(0.3, prod_rate))

            rf[fi, j] = np.array(
                [
                    1.0 if j == c_idx else 0.0,
                    min(1.0, eta_j / 35.0),
                    math.tanh((eta_j - eta_cur) / 10.0),
                    ships_log,
                    min(2.0, fl.ships / max(1.0, need_j)),
                    can_cap,
                    can_save,
                    cur_wasted,
                    min(2.0, roi_j),
                    owner_eta,
                ],
                dtype=np.float32,
            )

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

        va = self.backbone.val_attn(h).squeeze(-1).masked_fill(~pad_mask, -1e9)
        vw = F.softmax(va, dim=-1).unsqueeze(-1)
        pooled = (h * vw).sum(dim=1)
        value = torch.tanh(self.backbone.val_head(pooled).squeeze(-1))
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


class GalconTransformerAgent:
    """Fast Galcon neural agent powered by ``PlanetTimelineTransformer`` / ``GalconPlanetTransformer``."""

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
            isinstance(self.model, GalconPlanetTransformer)
            and redir_enc is not None
            and len(redir_enc.fleet_ids) > 0
        )
        want_ms = (
            isinstance(self.model, GalconPlanetTransformer)
            and self.model.enable_multi_send
        )
        if want_redir or want_ms:
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
        probs, mean_fracs, _, redir_probs, ms_probs, ms_fracs = self._predict_all(enc, redir_enc=redir_enc)
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
        time_budget_sec: float = 0.22,
        neural_value_weight: float = 0.70,
        reactive_mid_ply: bool = True,
        opponent_adaptive: bool = True,
        prior_bonus: float = 0.015,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            launch_threshold=launch_threshold,
            redir_threshold=redir_threshold,
            multi_send_threshold=multi_send_threshold,
            use_d4_tta=use_d4_tta,
            use_d4_canonical=use_d4_canonical,
            device=device,
        )
        self.rollout_horizon = max(4, int(rollout_horizon))
        self.num_plies = max(1, min(3, int(num_plies))) if reactive_mid_ply else 1
        self.time_budget_sec = float(time_budget_sec)
        self.neural_value_weight = float(neural_value_weight)
        self.reactive_mid_ply = bool(reactive_mid_ply)
        self.opponent_adaptive = bool(opponent_adaptive)
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
        self, encs: Sequence[EncodedGalconState]
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

        want_ms = isinstance(self.model, GalconPlanetTransformer) and self.model.enable_multi_send
        if want_ms:
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
        pf_b = np.stack([self._prep_pf(e) for e in leaf_encs], axis=0)
        tf_b = np.stack([e.timeline_feats for e in leaf_encs], axis=0)
        ef_b = np.stack([e.edge_feats for e in leaf_encs], axis=0)
        pm_b = np.ones((k, n), dtype=bool)
        _, _, vals = self.model(
            torch.from_numpy(pf_b).to(self.device),
            torch.from_numpy(tf_b).to(self.device),
            torch.from_numpy(ef_b).to(self.device),
            torch.from_numpy(pm_b).to(self.device),
        )
        return vals.cpu().numpy()

    def _build_fanout_plan(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        probs: np.ndarray,
        *,
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
                p_prior = float(probs[i, 1 + j])
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
        deadline = t0 + self.time_budget_sec
        if self.opponent_adaptive:
            self._update_opp_profile(sim, player)
        enc = extract_galcon_features(sim, player, feature_version=self.feature_version)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return [], [], [], 0.0

        redir_enc = (
            extract_galcon_redir_features(sim, enc)
            if (self.use_native_redir and self.has_trained_redir and sim.allow_redirect and sim.fleets)
            else None
        )
        probs, mean_fracs, root_val, redir_probs, ms_probs, ms_fracs = self._predict_all(enc, redir_enc=redir_enc)
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
        fan_cons = self._build_fanout_plan(
            sim, enc, probs, max_targets_per_planet=3, reserve_ratio=0.25, base_redirs=base_redirs
        )
        if fan_cons:
            raw_cands.append(fan_cons)
        fan_agg = self._build_fanout_plan(
            sim, enc, probs, max_targets_per_planet=5, reserve_ratio=0.10, base_redirs=base_redirs
        )
        if fan_agg:
            raw_cands.append(fan_agg)

        # Defensive Hold-All / rush-guard candidate early or when threatened
        any_threat = any(
            enc.planets[i].owner == player and enc.safe_surplus[i] < enc.planets[i].ships * 0.5
            for i in range(n)
        )
        if sim.tick <= 24 or any_threat:
            raw_cands.append([])

        # Smart redirect-augmented vs no-redirect comparison when friendly fleets are flying
        if sim.allow_redirect and any(fl.owner == player for fl in sim.fleets.values()):
            smart_redirs, _ = compute_smart_redirects(sim, enc, verify_rollout=False)
            if smart_redirs:
                base_sends = [o for o in raw_cands[0] if isinstance(o, SendOrder)]
                raw_cands.append(base_sends + list(smart_redirs))
                raw_cands.append(base_sends)

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
        if opp_enc.planet_ids and np.any(opp_enc.owned_mask):
            op, of, _, _, oms_p, oms_f = self._predict_all(opp_enc, use_d4_tta=False)
            opp_orders = self._predict_opponent_reply(sim, opp_enc, op, of, oms_p, oms_f)

        # Build ply schedule (e.g. num_plies=2, horizon=12 -> [6, 12]; num_plies=3, horizon=15 -> [5, 10, 15])
        ply_steps = [
            max(1, int(round(self.rollout_horizon * (k + 1) / self.num_plies)))
            for k in range(self.num_plies)
        ]
        ply_steps[-1] = self.rollout_horizon

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
            m_probs, m_fracs, _, m_msp, m_msf = self._batch_forward_policy_and_val(mid_encs)
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
        best_idx = 0
        best_score = -1e9
        for i in range(len(eval_cands)):
            if done_flags[i]:
                sc = h_vals[i]
            else:
                sc = self.neural_value_weight * nn_map[i] + (1.0 - self.neural_value_weight) * h_vals[i]
            scores.append(sc)
            eff_sc = sc + (self.prior_bonus if i == 0 else 0.0)
            if eff_sc > best_score:
                best_score = eff_sc
                best_idx = i

        return eval_cands[best_idx], eval_cands, scores, scores[best_idx]

    @torch.no_grad()
    def act(self, sim: GalconSim, player: int) -> List[Order]:
        best_plan, _, _, _ = self.act_with_search_info(sim, player)
        return best_plan
