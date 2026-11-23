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
    """Gen 2 Galcon Transformer wrapping ``PlanetTimelineTransformer`` with world-model & ``/REDIR`` heads.

    Directly reuses ``orbit_wars_bot.planet_transformer.PlanetTimelineTransformer`` as its
    core trunk + policy/fraction/value heads (keeping zero modifications to ``orbit_wars_bot/``)
    and adds:
    1. ``aux_future_head``: predicts each planet's ``t+10`` ownership (3-way) + ``log1p(ships)/5.5``.
    2. ``redir_*`` heads: native 8x D4-invariant mid-flight ``/REDIR`` scorer over flying fleets.
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
    ) -> None:
        super().__init__()
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
) -> GalconPlanetTransformer:
    """Instantiate the Gen 2 ~1.28M ``GalconPlanetTransformer`` with world-model & ``/REDIR`` heads."""
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
    )


def load_galcon_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, int, bool]:
    """Load either a v1 ``PlanetTimelineTransformer`` or Gen 2 ``GalconPlanetTransformer`` checkpoint.

    Returns ``(model, feature_version, has_native_redir)``.
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    model_cls = cfg.pop("model_class", None) or ckpt.get("model_class")
    has_redir = bool(ckpt.get("has_trained_redir", False))

    if model_cls == "GalconPlanetTransformer" or "redir_dim" in cfg or any(
        k.startswith("backbone.") for k in ckpt["model_state_dict"].keys()
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
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, fv, has_redir

    fv = 2 if int(cfg.get("planet_dim", 18)) >= 22 else 1
    clean_cfg = {k: v for k, v in cfg.items() if k in ("planet_dim", "timeline_dim", "edge_dim", "d_model", "n_heads", "n_layers", "dim_ff")}
    model = PlanetTimelineTransformer(**clean_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
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
        use_d4_tta: bool = True,
        enable_redirect_overlay: bool = True,
        use_native_redir: bool = True,
        device: str = "cpu",
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.launch_threshold = float(launch_threshold)
        self.redir_threshold = float(redir_threshold)
        self.use_d4_tta = bool(use_d4_tta)
        self.enable_redirect_overlay = bool(enable_redirect_overlay)
        self.use_native_redir = bool(use_native_redir)

        self.model, self.feature_version, self.has_trained_redir = load_galcon_model_from_checkpoint(
            checkpoint_path, device=self.device
        )

    @torch.no_grad()
    def predict_probs_and_value(
        self,
        enc: EncodedGalconState,
        *,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        probs, mean_fracs, val, _ = self._predict_full(enc, redir_enc=None, use_d4_tta=use_d4_tta)
        return probs, mean_fracs, val

    @torch.no_grad()
    def predict_probs_value_and_redir(
        self,
        enc: EncodedGalconState,
        redir_enc: Optional[EncodedRedirState],
        *,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        return self._predict_full(enc, redir_enc=redir_enc, use_d4_tta=use_d4_tta)

    @torch.no_grad()
    def _predict_full(
        self,
        enc: EncodedGalconState,
        *,
        redir_enc: Optional[EncodedRedirState] = None,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        n = len(enc.planet_ids)
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0.0, None
        d4 = self.use_d4_tta if use_d4_tta is None else bool(use_d4_tta)
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
        if want_redir:
            logits, fracs, vals, h = self.model(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
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
            logits, fracs, vals = self.model(pf_t, tf_t, ef_t, pm_t)
            redir_probs = None

        legal = torch.from_numpy(enc.legal_edges).to(self.device)
        mask = torch.cat([torch.ones((n, 1), dtype=torch.bool, device=self.device), legal], dim=-1)
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        mean_fracs = fracs.mean(dim=0).cpu().numpy()
        return probs, mean_fracs, float(vals.mean().item()), redir_probs

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

        # Mid-flight redirects: native learned /REDIR head for Gen 2, upgraded smart overlay fallback
        if include_redirects and sim.allow_redirect and sim.fleets:
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
        probs, mean_fracs, _, redir_probs = self.predict_probs_value_and_redir(enc, redir_enc)
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
    """Fast batched value-guided lookahead search over ``GalconTransformerAgent`` in ``GalconSim``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        redir_threshold: float = 0.55,
        use_d4_tta: bool = True,
        rollout_horizon: int = 12,
        time_budget_sec: float = 0.15,
        neural_value_weight: float = 0.60,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            launch_threshold=launch_threshold,
            redir_threshold=redir_threshold,
            use_d4_tta=use_d4_tta,
            device=device,
        )
        self.rollout_horizon = max(4, int(rollout_horizon))
        self.time_budget_sec = float(time_budget_sec)
        self.neural_value_weight = float(neural_value_weight)

    @torch.no_grad()
    def _batch_evaluate_leaves(self, leaf_encs: Sequence[EncodedGalconState]) -> np.ndarray:
        """Evaluate all non-terminal candidate rollout leaves in ONE batched forward pass."""
        k = len(leaf_encs)
        if k == 0:
            return np.zeros((0,), dtype=np.float32)
        n = len(leaf_encs[0].planet_ids)
        pf_b = np.stack([e.planet_feats for e in leaf_encs], axis=0)
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

    @torch.no_grad()
    def act_with_search_info(
        self,
        sim: GalconSim,
        player: int,
    ) -> Tuple[List[Order], List[List[Order]], List[float], float]:
        """Run batched lookahead search and return ``(best_plan, candidates, scores, best_score)``."""
        t0 = time.perf_counter()
        deadline = t0 + self.time_budget_sec
        enc = extract_galcon_features(sim, player, feature_version=self.feature_version)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return [], [], [], 0.0

        redir_enc = (
            extract_galcon_redir_features(sim, enc)
            if (self.use_native_redir and self.has_trained_redir and sim.allow_redirect and sim.fleets)
            else None
        )
        probs, mean_fracs, root_val, redir_probs = self.predict_probs_value_and_redir(enc, redir_enc)

        raw_cands: List[List[Order]] = [
            self._decode_orders(
                sim, enc, probs, mean_fracs, launch_threshold=thr, redir_enc=redir_enc, redir_probs=redir_probs
            )
            for thr in (
                self.launch_threshold,
                max(0.24, self.launch_threshold - 0.16),
                min(0.78, self.launch_threshold + 0.16),
            )
        ]
        # Fraction-scaled variants (tight exact-capture vs heavy decisive commit)
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
        # Flip top-3 highest-entropy owned planets
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

        # Deduplicate candidates preserving order
        seen = set()
        candidates: List[List[Order]] = []
        for plan in raw_cands:
            key = tuple(repr(o) for o in plan)
            if key not in seen:
                seen.add(key)
                candidates.append(plan)

        if len(candidates) <= 1 or time.perf_counter() >= deadline - 0.015:
            return (candidates[0] if candidates else []), candidates, [root_val], root_val

        # Predict opponent's immediate 1-step response (1x fast forward pass)
        opp = 2 if player == 1 else 1
        opp_enc = extract_galcon_features(sim, opp, feature_version=self.feature_version)
        opp_orders: List[Order] = []
        if opp_enc.planet_ids and np.any(opp_enc.owned_mask):
            op, of, _ = self.predict_probs_and_value(opp_enc, use_d4_tta=False)
            opp_orders = self._decode_orders(
                sim, opp_enc, op, of, launch_threshold=self.launch_threshold, include_redirects=False
            )

        # Roll all candidates forward, then evaluate all non-terminal leaves in a SINGLE batched NN call
        h_vals: List[float] = []
        done_flags: List[bool] = []
        leaf_encs: List[EncodedGalconState] = []
        leaf_pos: List[int] = []
        eval_cands: List[List[Order]] = []

        for idx, plan in enumerate(candidates):
            if idx > 0 and time.perf_counter() >= deadline - 0.012:
                break
            roll = sim.clone()
            roll.step({player: plan, opp: opp_orders})
            for _ in range(1, self.rollout_horizon):
                if roll.done:
                    break
                roll.step({})
            eval_cands.append(plan)
            h_vals.append(evaluate_galcon_state(roll, player))
            done_flags.append(roll.done)
            if not roll.done:
                leaf_pos.append(len(eval_cands) - 1)
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
            if sc > best_score:
                best_score = sc
                best_idx = i

        return eval_cands[best_idx], eval_cands, scores, best_score

    @torch.no_grad()
    def act(self, sim: GalconSim, player: int) -> List[Order]:
        best_plan, _, _, _ = self.act_with_search_info(sim, player)
        return best_plan
