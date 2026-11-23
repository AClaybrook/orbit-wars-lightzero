"""Cross-pollinated PlanetTimelineTransformer agent for Galcon & ``gbots``.

Directly reuses ``PlanetTimelineTransformer`` from ``orbit_wars_bot.planet_transformer``
with scale-invariant continuous coordinates, exact 20-step arrival timelines, 100% D4-invariant
pairwise ETA/ROI edges, 8x D4 dihedral TTA, smart mid-flight redirects, and optional
value-guided lookahead search in ``GalconSim``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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


@dataclass
class EncodedGalconState:
    player: int
    planet_ids: List[int]
    planets: List[Planet]
    planet_feats: np.ndarray  # (N, 18)
    timeline_feats: np.ndarray  # (N, 20, 8)
    edge_feats: np.ndarray  # (N, N, 10)
    legal_edges: np.ndarray  # (N, N) bool
    owned_mask: np.ndarray  # (N,) bool
    safe_surplus: np.ndarray  # (N,) float32


def apply_galcon_d4(planet_feats: np.ndarray, k_rot: int = 0, flip_x: bool = False) -> np.ndarray:
    """Exact 8x D4 dihedral symmetry on normalized ``(x, y)`` (cols 0-1).

    Because all timeline and pairwise edge features are 100% D4-invariant, transforming
    only columns 0 and 1 gives exact 8x data augmentation and 8x test-time augmentation
    on any map aspect ratio and any number of planets.
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


def extract_galcon_features(sim: GalconSim, player: int) -> EncodedGalconState:
    """Extract lossless, scale-invariant entity + arrival-timeline + edge features for Galcon."""
    planets = sorted(sim.planets.values(), key=lambda p: p.id)[:GALCON_MAX_PLANETS]
    n = len(planets)
    planet_ids = [p.id for p in planets]

    if n == 0:
        return EncodedGalconState(
            player=player,
            planet_ids=[],
            planets=[],
            planet_feats=np.zeros((0, GALCON_PLANET_DIM), dtype=np.float32),
            timeline_feats=np.zeros((0, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32),
            edge_feats=np.zeros((0, 0, GALCON_EDGE_DIM), dtype=np.float32),
            legal_edges=np.zeros((0, 0), dtype=bool),
            owned_mask=np.zeros((0,), dtype=bool),
            safe_surplus=np.zeros((0,), dtype=np.float32),
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

    pf = np.zeros((n, GALCON_PLANET_DIM), dtype=np.float32)
    tf = np.zeros((n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32)
    ef = np.zeros((n, n, GALCON_EDGE_DIM), dtype=np.float32)
    legal = np.zeros((n, n), dtype=bool)
    owned_mask = np.zeros((n,), dtype=bool)
    safe_surplus = np.zeros((n,), dtype=np.float32)

    # Precompute timelines for all planets
    futures = [project_planet_future(sim, p.id, player, horizon=GALCON_TIMELINE_HORIZON) for p in planets]

    my_indices = [i for i, p in enumerate(planets) if p.owner == player]
    en_indices = [i for i, p in enumerate(planets) if p.owner != 0 and p.owner != player]

    # Pairwise ETAs in ticks
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

        # Safe surplus if owned
        if p.owner == player:
            lost = any(po[t] <= 0 for t in range(GALCON_TIMELINE_HORIZON))
            min_hold = float(ps.min()) if not lost else 0.0
            safe_surplus[i] = max(0.0, min(float(p.ships), min_hold))

        min_en_eta = min((etas[i, j] for j in en_indices), default=40.0)
        min_my_eta = min((etas[i, j] for j in my_indices if j != i), default=40.0)
        frontline = (min_my_eta - min_en_eta) / max(10.0, min_my_eta + min_en_eta)

        pf[i] = np.array(
            [
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
            ],
            dtype=np.float32,
        )

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
            need = ships_eta + 1.5
            can_capture = 1.0 if (owner_eta <= 0 and src.ships > need) else 0.0
            safe_capture = 1.0 if (owner_eta <= 0 and safe_surplus[i] > need) else 0.0
            steal_flag = 1.0 if (dst.owner == 0 and owner_eta < 0) else 0.0
            defend_flag = 1.0 if (dst.owner == player and owner_eta <= 0) else 0.0
            prod_rate = dst.production / PRODUCTION_DIVISOR
            roi = (prod_rate * 20.0) / max(3.0, need + 0.45 * eta * max(0.3, prod_rate))
            en_eta_src = pf[i, 12]
            en_eta_dst = pf[j, 12]

            ef[i, j] = np.array(
                [
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
                ],
                dtype=np.float32,
            )

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
    )


def pad_galcon_state(enc: EncodedGalconState) -> Dict[str, np.ndarray]:
    """Pad variable-N Galcon features to ``GALCON_MAX_PLANETS`` for batched training."""
    n = len(enc.planet_ids)
    pf = np.zeros((GALCON_MAX_PLANETS, GALCON_PLANET_DIM), dtype=np.float32)
    tf = np.zeros((GALCON_MAX_PLANETS, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM), dtype=np.float32)
    ef = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS, GALCON_EDGE_DIM), dtype=np.float32)
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
    """Instantiate our shared ``PlanetTimelineTransformer`` configured for Galcon."""
    return PlanetTimelineTransformer(
        planet_dim=GALCON_PLANET_DIM,
        timeline_dim=GALCON_TIMELINE_DIM,
        edge_dim=GALCON_EDGE_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
    )


class GalconTransformerAgent:
    """Fast Galcon neural agent powered by ``PlanetTimelineTransformer`` with 8x D4 TTA."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        use_d4_tta: bool = True,
        enable_redirect_overlay: bool = True,
        device: str = "cpu",
    ) -> None:
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.launch_threshold = float(launch_threshold)
        self.use_d4_tta = bool(use_d4_tta)
        self.enable_redirect_overlay = bool(enable_redirect_overlay)

        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        self.model = PlanetTimelineTransformer(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_probs_and_value(
        self,
        enc: EncodedGalconState,
        *,
        use_d4_tta: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = len(enc.planet_ids)
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), 0.0
        d4 = self.use_d4_tta if use_d4_tta is None else bool(use_d4_tta)
        syms = [(k, f) for f in (False, True) for k in range(4)] if d4 else [(0, False)]
        k_b = len(syms)
        pf_b = np.stack([apply_galcon_d4(enc.planet_feats, k_rot=k, flip_x=f) for k, f in syms], axis=0)
        tf_b = np.broadcast_to(enc.timeline_feats[None], (k_b, n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM)).copy()
        ef_b = np.broadcast_to(enc.edge_feats[None], (k_b, n, n, GALCON_EDGE_DIM)).copy()
        pm_b = np.ones((k_b, n), dtype=bool)

        logits, fracs, vals = self.model(
            torch.from_numpy(pf_b).to(self.device),
            torch.from_numpy(tf_b).to(self.device),
            torch.from_numpy(ef_b).to(self.device),
            torch.from_numpy(pm_b).to(self.device),
        )
        legal = torch.from_numpy(enc.legal_edges).to(self.device)
        mask = torch.cat([torch.ones((n, 1), dtype=torch.bool, device=self.device), legal], dim=-1)
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        probs = F.softmax(logits, dim=-1).mean(dim=0).cpu().numpy()
        mean_fracs = fracs.mean(dim=0).cpu().numpy()
        return probs, mean_fracs, float(vals.mean().item())

    def _decode_orders(
        self,
        sim: GalconSim,
        enc: EncodedGalconState,
        probs: np.ndarray,
        mean_fracs: np.ndarray,
        *,
        launch_threshold: float,
        overrides: Optional[Dict[int, int]] = None,
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
            f = float(mean_fracs[i, best_j])
            pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
            orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[best_j], pct))

        # Smart mid-flight redirect overlay for wasted over-reinforcement fleets
        if self.enable_redirect_overlay and sim.allow_redirect and sim.fleets:
            orders.extend(self._redirect_wasted_fleets(sim, enc))
        return orders

    def _redirect_wasted_fleets(self, sim: GalconSim, enc: EncodedGalconState) -> List[Order]:
        redirs: List[Order] = []
        player = enc.player
        futures = {p.id: project_planet_future(sim, p.id, player, horizon=24) for p in enc.planets}
        for fl in sim.fleets.values():
            if fl.owner != player:
                continue
            dst = sim.planets.get(fl.target)
            if dst is None:
                continue
            _, _, po, ps = futures[dst.id]
            if dst.owner == player and po[-1] > 0 and ps[-1] > fl.ships + 25.0:
                best_alt = None
                best_sc = 0.0
                for cand in enc.planets:
                    if cand.id == dst.id or cand.owner == player:
                        continue
                    eta = sim.eta_ticks(fl.x, fl.y, cand)
                    _, _, cpo, cps = futures[cand.id]
                    t_idx = min(len(cps) - 1, eta - 1)
                    if cpo[t_idx] <= 0 and fl.ships > cps[t_idx] + 1.5:
                        sc = (cand.production * (2.0 if cand.owner != 0 else 1.2)) / max(2.0, cps[t_idx] + eta * 1.5)
                        if sc > best_sc:
                            best_sc = sc
                            best_alt = cand.id
                if best_alt is not None:
                    redirs.append(RedirOrder(fl.id, best_alt))
        return redirs

    @torch.no_grad()
    def act(self, sim: GalconSim, player: int) -> List[Order]:
        enc = extract_galcon_features(sim, player)
        if not enc.planet_ids or not np.any(enc.owned_mask):
            return []
        probs, mean_fracs, _ = self.predict_probs_and_value(enc)
        return self._decode_orders(sim, enc, probs, mean_fracs, launch_threshold=self.launch_threshold)


class GalconSearchTransformerAgent(GalconTransformerAgent):
    """Value-guided lookahead search over ``GalconTransformerAgent`` using ``GalconSim`` rollouts."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        launch_threshold: float = 0.50,
        use_d4_tta: bool = True,
        rollout_horizon: int = 12,
        time_budget_sec: float = 0.15,
        neural_value_weight: float = 0.60,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            launch_threshold=launch_threshold,
            use_d4_tta=use_d4_tta,
            device=device,
        )
        self.rollout_horizon = max(4, int(rollout_horizon))
        self.time_budget_sec = float(time_budget_sec)
        self.neural_value_weight = float(neural_value_weight)

    @torch.no_grad()
    def act(self, sim: GalconSim, player: int) -> List[Order]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_budget_sec
        enc = extract_galcon_features(sim, player)
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return []

        probs, mean_fracs, _ = self.predict_probs_and_value(enc)
        raw_cands: List[List[Order]] = [
            self._decode_orders(sim, enc, probs, mean_fracs, launch_threshold=thr)
            for thr in (self.launch_threshold, max(0.28, self.launch_threshold - 0.15), min(0.75, self.launch_threshold + 0.15))
        ]
        # Flip top-2 highest-entropy owned planets
        entropies: List[Tuple[float, int, int]] = []
        for i in range(n):
            if not enc.owned_mask[i]:
                continue
            p_row = np.clip(probs[i], 1e-6, 1.0)
            ent = float(-np.sum(p_row * np.log(p_row)))
            top2 = np.argsort(-p_row)[:2]
            best_c, alt_c = int(top2[0]), int(top2[1])
            if p_row[alt_c] >= 0.15:
                entropies.append((ent, i, alt_c))
        entropies.sort(key=lambda x: -x[0])
        for _, p_idx, alt_c in entropies[:2]:
            raw_cands.append(
                self._decode_orders(
                    sim, enc, probs, mean_fracs, launch_threshold=self.launch_threshold, overrides={p_idx: alt_c}
                )
            )

        # Deduplicate candidates
        seen = set()
        candidates: List[List[Order]] = []
        for plan in raw_cands:
            key = tuple(repr(o) for o in plan)
            if key not in seen:
                seen.add(key)
                candidates.append(plan)

        if len(candidates) <= 1 or time.perf_counter() >= deadline - 0.02:
            return candidates[0] if candidates else []

        opp = 2 if player == 1 else 1
        opp_enc = extract_galcon_features(sim, opp)
        opp_orders: List[Order] = []
        if opp_enc.planet_ids and np.any(opp_enc.owned_mask):
            op, of, _ = self.predict_probs_and_value(opp_enc, use_d4_tta=False)
            opp_orders = self._decode_orders(sim, opp_enc, op, of, launch_threshold=self.launch_threshold)

        best_plan = candidates[0]
        best_score = -1e9
        for idx, plan in enumerate(candidates):
            if idx > 0 and time.perf_counter() >= deadline:
                break
            roll = sim.clone()
            roll.step({player: plan, opp: opp_orders})
            for _ in range(1, self.rollout_horizon):
                if roll.done:
                    break
                roll.step({})
            h_val = evaluate_galcon_state(roll, player)
            if roll.done:
                score = h_val
            else:
                fut_enc = extract_galcon_features(roll, player)
                _, _, nn_val = self.predict_probs_and_value(fut_enc, use_d4_tta=False)
                score = self.neural_value_weight * nn_val + (1.0 - self.neural_value_weight) * h_val
            if score > best_score:
                best_score = score
                best_plan = plan
        return best_plan
