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
from dataclasses import dataclass
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
    SUN_RADIUS,
    build_comet_lookup,
    distance,
    fleet_speed,
    is_orbiting_planet,
    predict_all_planet_tracks,
    predict_fleet_arrivals,
    segment_hits_sun,
    simulate_planet_future,
    solve_intercept,
)

MAX_PLANETS = 44
TIMELINE_HORIZON = 20
MAX_ROUTE_ETA = 20
PLANET_FEAT_DIM = 22
TIMELINE_FEAT_DIM = 8
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


def extract_planet_features(
    obs: Any,
    player: Optional[int] = None,
    max_speed: float = DEFAULT_SHIP_SPEED,
) -> EncodedPlanetState:
    """Extract lossless per-planet, 20-step timeline, and directed edge features."""
    if player is None:
        player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    ang_vel = float(_obs_get(obs, "angular_velocity", 0.0))
    planets = [list(p) for p in (_obs_get(obs, "planets", []) or [])]
    fleets = _obs_get(obs, "fleets", []) or []
    init_planets = _obs_get(obs, "initial_planets", []) or planets
    comets = _obs_get(obs, "comets", []) or []
    comet_ids = set(int(c) for c in (_obs_get(obs, "comet_planet_ids", []) or []))
    comet_lookup = build_comet_lookup(comets)
    initial_by_id = {int(p[0]): p for p in init_planets}

    # Cap to MAX_PLANETS (should always hold: <=36 planets + 4 comets = 40)
    planets = planets[:MAX_PLANETS]
    n = len(planets)
    planet_ids = [int(p[0]) for p in planets]
    id_to_idx = {pid: idx for idx, pid in enumerate(planet_ids)}

    emap, my_ships, tot_ships, my_prod, tot_prod = _enemy_slot_map(planets, fleets, player)
    alive_enemies = len(emap)
    # Detect 4P mode from total players present initially or alive enemies
    init_owners = {int(p[1]) for p in init_planets if int(p[1]) >= 0}
    cur_owners = {int(p[1]) for p in planets if int(p[1]) >= 0} | {int(f[1]) for f in fleets if int(f[1]) >= 0}
    is_4p = 1.0 if (len(init_owners | cur_owners) > 2 or max((init_owners | cur_owners), default=0) >= 2) else 0.0

    tracks = predict_all_planet_tracks(
        planets,
        TIMELINE_HORIZON,
        step,
        ang_vel,
        initial_by_id,
        comet_lookup,
    )
    arrivals = predict_fleet_arrivals(
        fleets,
        planets,
        step,
        ang_vel,
        initial_by_id,
        comet_lookup=comet_lookup,
        tracks=tracks,
        max_horizon=TIMELINE_HORIZON,
        max_speed=max_speed,
    )

    planet_feats = np.zeros((n, PLANET_FEAT_DIM), dtype=np.float32)
    timeline_feats = np.zeros((n, TIMELINE_HORIZON, TIMELINE_FEAT_DIM), dtype=np.float32)
    owned_mask = np.zeros((n,), dtype=bool)
    expiring_comet_indices: List[int] = []

    # Pre-simulate future timeline per planet for both node timelines and edge ROI features
    futures: List[Tuple[Tuple[int, int], ...]] = []
    my_ship_share = my_ships / max(1.0, float(tot_ships))
    my_prod_share = my_prod / max(1.0, float(tot_prod))
    step_norm = min(1.0, step / 500.0)
    ang_vel_norm = ang_vel / 0.05
    alive_en_norm = alive_enemies / 3.0

    for i, p in enumerate(planets):
        pid = int(p[0])
        owner = int(p[1])
        x, y, rad, ships, prod = float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6])
        is_comet = (pid in comet_ids) or (pid in comet_lookup)
        orbiting = (not is_comet) and is_orbiting_planet(x, y, rad)

        # Compute instantaneous velocity (vx, vy) from tracks[pid][0] -> tracks[pid][1]
        tr = tracks.get(pid)
        vx, vy = 0.0, 0.0
        if tr and len(tr) >= 2 and tr[0] is not None and tr[1] is not None:
            vx = (tr[1][0] - tr[0][0]) / 6.0
            vy = (tr[1][1] - tr[0][1]) / 6.0

        comet_rem = 0.0
        if is_comet and pid in comet_lookup:
            cpath, cidx = comet_lookup[pid]
            rem = max(0, len(cpath) - 1 - cidx)
            comet_rem = min(1.0, rem / 40.0)
            if owner == player and ships > 0 and rem <= 1:
                expiring_comet_indices.append(i)

        orb_r = math.hypot(x - CENTER, y - CENTER) / 50.0
        if owner == player and ships > 0:
            owned_mask[i] = True

        # Owner 5-way one-hot: [neutral, me, enemy1, enemy2, enemy3]
        oh = [0.0, 0.0, 0.0, 0.0, 0.0]
        if owner == -1:
            oh[0] = 1.0
        elif owner == player:
            oh[1] = 1.0
        else:
            eslot = emap.get(owner, 2)
            oh[2 + min(eslot, 2)] = 1.0

        planet_feats[i] = (
            (x - CENTER) / 50.0,
            (y - CENTER) / 50.0,
            vx,
            vy,
            orb_r,
            1.0 if orbiting else 0.0,
            1.0 if is_comet else 0.0,
            comet_rem,
            rad / 3.0,
            prod / 5.0,
            math.log1p(max(0, ships)) / 6.0,
            oh[0],
            oh[1],
            oh[2],
            oh[3],
            oh[4],
            step_norm,
            is_4p,
            my_ship_share,
            my_prod_share,
            ang_vel_norm,
            alive_en_norm,
        )

        # Build exact 20-step arrival + future ownership timeline
        arr_list = arrivals.get(pid, ())
        fut = simulate_planet_future(p, arr_list, horizon=TIMELINE_HORIZON)
        futures.append(fut.timeline)

        my_in = np.zeros(TIMELINE_HORIZON + 1, dtype=np.float32)
        en_in = np.zeros(TIMELINE_HORIZON + 1, dtype=np.float32)
        for a in arr_list:
            if 1 <= a.eta <= TIMELINE_HORIZON:
                if a.owner == player:
                    my_in[a.eta] += a.ships
                elif a.owner >= 0:
                    en_in[a.eta] += a.ships

        prev_o = owner
        for dt in range(1, TIMELINE_HORIZON + 1):
            po, ps = fut.timeline[dt]
            t_idx = dt - 1
            timeline_feats[i, t_idx, 0] = 1.0 if po == -1 else 0.0
            timeline_feats[i, t_idx, 1] = 1.0 if po == player else 0.0
            timeline_feats[i, t_idx, 2] = 1.0 if (po >= 0 and po != player) else 0.0
            timeline_feats[i, t_idx, 3] = math.log1p(max(0, ps)) / 6.0
            timeline_feats[i, t_idx, 4] = math.log1p(my_in[dt]) / 6.0
            timeline_feats[i, t_idx, 5] = math.log1p(en_in[dt]) / 6.0
            timeline_feats[i, t_idx, 6] = 1.0 if en_in[dt] > 0.0 else 0.0
            timeline_feats[i, t_idx, 7] = 1.0 if po != prev_o else 0.0
            prev_o = po

    # Build 100% D4-invariant directed edge features (N, N, EDGE_FEAT_DIM)
    edge_feats = np.zeros((n, n, EDGE_FEAT_DIM), dtype=np.float32)
    legal_edges = np.zeros((n, n), dtype=bool)
    pos_list = [(float(p[2]), float(p[3])) for p in planets]
    rad_list = [float(p[4]) for p in planets]
    ships_list = [int(p[5]) for p in planets]
    prod_list = [int(p[6]) for p in planets]
    is_comet_list = [(planet_ids[i] in comet_ids or planet_ids[i] in comet_lookup) for i in range(n)]
    track_list = [tracks[planet_ids[i]] for i in range(n)]

    for i in range(n):
        src_ships = ships_list[i]
        # Compute edge features for every planet with ships >= 1 (or speed at 10 ships for non-owned context)
        spd = fleet_speed(max(1, src_ships), max_speed)
        sx, sy = pos_list[i]
        sr = rad_list[i]
        for j in range(n):
            if i == j:
                continue
            tr_j = track_list[j]
            rj = rad_list[j]
            reach_pad = sr + 0.1 + rj

            # Fast exact-track ETA scan for dt = 1..MAX_ROUTE_ETA
            best_eta = 0
            best_dist = 0.0
            for dt in range(1, MAX_ROUTE_ETA + 1):
                pt = tr_j[dt]
                if pt is None or pt[0] < 0.0:
                    break
                d = math.hypot(pt[0] - sx, pt[1] - sy)
                if d <= dt * spd + reach_pad:
                    if not segment_hits_sun((sx, sy), pt, margin=0.25):
                        best_eta = dt
                        best_dist = d
                    break

            if best_eta == 0:
                continue

            if owned_mask[i]:
                legal_edges[i, j] = True

            po, ps = futures[j][best_eta]
            t_prod = prod_list[j]
            is_t_neutral = 1.0 if po == -1 else 0.0
            is_t_me = 1.0 if po == player else 0.0
            is_t_enemy = 1.0 if (po >= 0 and po != player) else 0.0

            # Signed capture surplus (for friendly target, how much garrison it already has)
            margin = float(src_ships - ps) if not is_t_me else float(src_ships)
            cap_feasible = 1.0 if (is_t_me or src_ships > ps) else 0.0
            cost = 1.0 + (0.0 if is_t_me else float(max(0, ps))) + 0.5 * best_eta
            roi = min(2.0, float(t_prod) / cost)

            edge_feats[i, j] = (
                1.0,  # is_reachable_short_eta
                best_eta / float(MAX_ROUTE_ETA),
                min(1.5, best_dist / 100.0),
                t_prod / 5.0,
                is_t_neutral,
                is_t_me,
                is_t_enemy,
                1.0 if is_comet_list[j] else 0.0,
                math.log1p(max(0, ps)) / 6.0,
                math.tanh(margin / 25.0),
                cap_feasible,
                roi,
            )

    return EncodedPlanetState(
        planet_feats=planet_feats,
        timeline_feats=timeline_feats,
        edge_feats=edge_feats,
        legal_edges=legal_edges,
        owned_mask=owned_mask,
        planet_ids=planet_ids,
        id_to_idx=id_to_idx,
        expiring_comet_indices=expiring_comet_indices,
        planet_rows=planets,
        tracks=tracks,
        initial_by_id=initial_by_id,
        comet_lookup=comet_lookup,
        current_step=step,
        angular_velocity=ang_vel,
        player=player,
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


class _EdgeBiasedTransformerLayer(nn.Module):
    """Pre-LN Multi-Head Self-Attention block with directed pairwise edge bias."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, dim_ff: int = 256, edge_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_bias = nn.Linear(edge_dim, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
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
        attn = attn.masked_fill(key_mask, -1e9)
        probs = F.softmax(attn, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.out_proj(out)
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
    ):
        super().__init__()
        self.config = {
            "planet_dim": planet_dim,
            "timeline_dim": timeline_dim,
            "edge_dim": edge_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dim_ff": dim_ff,
        }

        # 1D-CNN encoder over the 20-step arrival & ownership timeline per planet
        self.timeline_conv = nn.Sequential(
            nn.Conv1d(timeline_dim, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.node_in = nn.Sequential(
            nn.Linear(planet_dim + 128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [_EdgeBiasedTransformerLayer(d_model, n_heads, dim_ff, edge_dim) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Per-planet HOLD logit
        self.hold_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Pairwise target logit & send-fraction head conditioned on (h_src, h_dst, edge_feat)
        self.src_proj = nn.Linear(d_model, d_model // 2)
        self.dst_proj = nn.Linear(d_model, d_model // 2)
        self.edge_proj = nn.Linear(edge_dim, d_model // 2)
        self.pair_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model // 2, 2),  # [target_logit, frac_logit]
        )

        # Global value head (attention pooled over active planets)
        self.val_attn = nn.Linear(d_model, 1)
        self.val_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        planet_feats: torch.Tensor,
        timeline_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            planet_feats: ``(B, N, PLANET_FEAT_DIM)``
            timeline_feats: ``(B, N, T, TIMELINE_FEAT_DIM)``
            edge_feats: ``(B, N, N, EDGE_FEAT_DIM)``
            pad_mask: ``(B, N)`` bool, True for valid planets.

        Returns:
            action_logits: ``(B, N, N + 1)`` where index ``0`` is HOLD and ``1 + j`` is target ``j``.
            frac_preds: ``(B, N, N)`` in ``(0, 1)`` predicted send fraction for ``(i -> j)``.
            value: ``(B,)`` predicted outcome in ``[-1, +1]``.
        """
        B, N, T, Ct = timeline_feats.shape
        tl = timeline_feats.reshape(B * N, T, Ct).transpose(1, 2)  # (B*N, Ct, T)
        tl_h = self.timeline_conv(tl)  # (B*N, 64, T)
        tl_pool = torch.cat([tl_h.mean(dim=-1), tl_h.amax(dim=-1)], dim=-1).view(B, N, 128)

        x = self.node_in(torch.cat([planet_feats, tl_pool], dim=-1))
        for layer in self.layers:
            x = layer(x, edge_feats, pad_mask)
        h = self.final_norm(x)  # (B, N, D)

        hold_logit = self.hold_head(h)  # (B, N, 1)
        pair_in = (
            self.src_proj(h).unsqueeze(2)
            + self.dst_proj(h).unsqueeze(1)
            + self.edge_proj(edge_feats)
        )  # (B, N, N, D/2)
        pair_out = self.pair_head(pair_in)  # (B, N, N, 2)
        target_logits = pair_out[..., 0]  # (B, N, N)
        frac_preds = torch.sigmoid(pair_out[..., 1])  # (B, N, N)

        action_logits = torch.cat([hold_logit, target_logits], dim=-1)  # (B, N, N + 1)

        # Attention-pooled global value head
        va = self.val_attn(h).squeeze(-1).masked_fill(~pad_mask, -1e9)
        vw = F.softmax(va, dim=-1).unsqueeze(-1)
        pooled = (h * vw).sum(dim=1)
        value = torch.tanh(self.val_head(pooled).squeeze(-1))
        return action_logits, frac_preds, value


def pad_encoded_state(enc: EncodedPlanetState, max_n: int = MAX_PLANETS) -> Dict[str, np.ndarray]:
    """Pad an ``EncodedPlanetState`` to fixed ``max_n`` for batched training."""
    n = len(enc.planet_ids)
    pf = np.zeros((max_n, PLANET_FEAT_DIM), dtype=np.float32)
    tf = np.zeros((max_n, TIMELINE_HORIZON, TIMELINE_FEAT_DIM), dtype=np.float32)
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
        force_all_in: bool = False,
        device: str = "cpu",
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.launch_threshold = float(launch_threshold)
        self.use_rot_tta = bool(use_rot_tta)
        self.force_all_in = bool(force_all_in)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        self.model = PlanetTimelineTransformer(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        max_speed = float(_obs_get(config, "shipSpeed", DEFAULT_SHIP_SPEED)) if config else DEFAULT_SHIP_SPEED
        enc = extract_planet_features(obs, max_speed=max_speed)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return []

        if self.use_rot_tta:
            pf_batch = np.stack([apply_d4_planet_feats(enc.planet_feats, k_rot=k) for k in range(4)], axis=0)
            tf_batch = np.broadcast_to(enc.timeline_feats[None], (4, n, TIMELINE_HORIZON, TIMELINE_FEAT_DIM)).copy()
            ef_batch = np.broadcast_to(enc.edge_feats[None], (4, n, n, EDGE_FEAT_DIM)).copy()
            pm_batch = np.ones((4, n), dtype=bool)
        else:
            pf_batch = enc.planet_feats[None]
            tf_batch = enc.timeline_feats[None]
            ef_batch = enc.edge_feats[None]
            pm_batch = np.ones((1, n), dtype=bool)

        logits, fracs, _ = self.model(
            torch.from_numpy(pf_batch).to(self.device),
            torch.from_numpy(tf_batch).to(self.device),
            torch.from_numpy(ef_batch).to(self.device),
            torch.from_numpy(pm_batch).to(self.device),
        )

        # Mask illegal target edges (class 0 = HOLD is always legal)
        legal = torch.from_numpy(enc.legal_edges).to(self.device)  # (N, N)
        mask = torch.cat([torch.ones((n, 1), dtype=torch.bool, device=self.device), legal], dim=-1)  # (N, N+1)
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()  # (N, N+1)
        mean_fracs = fracs.mean(dim=0).cpu().numpy()  # (N, N)

        expiring_set = set(enc.expiring_comet_indices)
        moves: List[List[float]] = []

        for i in range(n):
            if not enc.owned_mask[i]:
                continue
            if not np.any(enc.legal_edges[i]):
                continue
            p_launch = 1.0 - float(probs[i, 0])
            forced = i in expiring_set
            if (p_launch < self.launch_threshold) and (not forced):
                continue

            target_probs = probs[i, 1:]
            best_j = int(np.argmax(target_probs))
            if not enc.legal_edges[i, best_j]:
                continue

            src_row = enc.planet_rows[i]
            dst_row = enc.planet_rows[best_j]
            avail = int(src_row[5])
            if avail <= 0:
                continue

            if self.force_all_in or forced:
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
            if sol is not None:
                moves.append([int(src_row[0]), float(sol.angle), int(send_ships)])

        return moves


def create_planet_transformer_agent(
    checkpoint_path: str | Path,
    *,
    launch_threshold: float = 0.50,
    use_rot_tta: bool = True,
    force_all_in: bool = False,
):
    """Factory returning a Kaggle ``agent(obs, config)`` callable."""
    agent_obj = PlanetTransformerAgent(
        checkpoint_path=checkpoint_path,
        launch_threshold=launch_threshold,
        use_rot_tta=use_rot_tta,
        force_all_in=force_all_in,
    )

    def _kaggle_agent(obs: Any, config: Any = None) -> List[List[float]]:
        return agent_obj.act(obs, config)

    return _kaggle_agent
