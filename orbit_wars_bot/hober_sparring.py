"""Ultra-fast batched in-process `#2 Hober Malloc` opponent & soft-teacher engine.

Reuses ``build_shared_planet_physics`` orbital tracks, fleet arrivals, and 20-step
no-action timelines to build `#2 Hober Malloc`'s exact 10-channel tensor directly
in vectorized NumPy (~0.08 ms/board vs ~3.8 ms for FastOrbitSim + dict parsing)
and runs lockstep batched JAX inference with fixed `(B, N)` shape bucketing to
eliminate JIT recompilation stalls.

Designed for:
  1. Live `#2 Hober Malloc` sparring directly inside lockstep batched PPO rollouts
     (``scripts/rl_finetune_transformer.py --hober-share ...``) at ~8 ms/game-tick.
  2. Fast, exact in-loop Boss validation gates (``--hober-eval-pairs``).
  3. Zero-overhead exact ``ships_for_capture`` soft-label queries for Soft-KD / DAgger.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from orbit_wars_bot.physics import (
    CENTER,
    DEFAULT_SHIP_SPEED,
    solve_intercept,
)
from orbit_wars_bot.planet_transformer import (
    MAX_PLANETS,
    MAX_ROUTE_ETA,
    EncodedPlanetState,
    compute_ships_for_capture,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOBER_DIR = REPO_ROOT / "external" / "opponents" / "simjeg_hober_malloc"
HOBER_WEIGHTS = HOBER_DIR / "checkpoint" / "weight.npz"

_RADIAL_SCALE = np.float32(70.0)
_STEP_SCALE = np.float32(500.0)
_SHIP_SCALE = np.float32(100.0)
_TWO_PI = np.float32(2.0 * math.pi)
_HALF_PI = np.float32(0.5 * math.pi)
_MAX_TIMESTEPS = 20


def is_hober_available() -> bool:
    """Return True if `#2 Hober Malloc`'s local JAX model & weights are present."""
    return HOBER_WEIGHTS.is_file() and (HOBER_DIR / "ow_runtime" / "model.py").is_file()


def _bucket_batch(b: int) -> int:
    for cand in (1, 2, 4, 8, 16, 32):
        if b <= cand:
            return cand
    return ((b + 7) // 8) * 8


def _bucket_bodies(n: int) -> int:
    if n <= 32:
        return 32
    if n <= 44:
        return 44
    return n


def _compute_ships_for_capture(
    shared_physics: Dict[str, Any],
    player: int,
    max_timesteps: int = _MAX_TIMESTEPS,
) -> np.ndarray:
    """Compute exact `ships_for_capture(planet, dt, player)` matching Hober's Rust ledger."""
    return compute_ships_for_capture(shared_physics, player, max_timesteps=max_timesteps)


def build_hober_inputs_fast(
    enc: EncodedPlanetState,
    shared_physics: Dict[str, Any],
    *,
    max_timesteps: int = _MAX_TIMESTEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build `(x[n, T, 10], time_mask[T], body_mask[n], launch_mask[n], target_mask[n, n])` in NumPy.

    Because Hober's ModernBERT architecture is 100% permutation-equivariant across bodies,
    we index bodies directly in `enc.planet_ids` order (zero scatter/gather overhead).
    """
    n = len(enc.planet_ids)
    step = int(shared_physics["step"])
    dts = np.arange(max_timesteps, dtype=np.int64)
    time_mask = (step + dts) <= 500
    if n == 0:
        return (
            np.zeros((0, max_timesteps, 10), dtype=np.float32),
            time_mask,
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
            np.zeros((0, 0), dtype=bool),
        )

    player = int(enc.player)
    feature_dts = np.minimum(dts, max(0, 500 - step))
    step_feat = (step + dts).astype(np.float32) / _STEP_SCALE

    fo = shared_physics["fut_owner"][:, feature_dts]   # (n, T)
    fs = shared_physics["fut_ships"][:, feature_dts]   # (n, T)
    tv = shared_physics["tr_valid"][:, feature_dts]    # (n, T)
    tx = shared_physics["tr_x"][:, feature_dts]        # (n, T)
    ty = shared_physics["tr_y"][:, feature_dts]        # (n, T)
    pprod = shared_physics["pprod"].astype(np.float32) # (n,)
    pown = shared_physics["pown"]                      # (n,)
    is_comet = shared_physics["is_comet_arr"]          # (n,)
    comet_lookup = shared_physics["comet_lookup"]

    # Match Hober's opponent channel mapping: sorted active opponent player IDs -> slots 6, 7, 8
    valid_owners = fo[tv]
    opp_ids = sorted({int(o) for o in np.unique(valid_owners) if int(o) >= 0 and int(o) != player})[:3]

    # Exact ships_for_capture matching Hober's Rust ledger
    cap_full = _compute_ships_for_capture(shared_physics, player, max_timesteps=max_timesteps)
    cap = cap_full[:, feature_dts]

    # Position validity: finite coordinates (non-hidden comets) & time_mask
    pos_ok = tv & np.isfinite(tx) & np.isfinite(ty) & (tx > -50.0) & (ty > -50.0)
    valid = pos_ok & time_mask[None, :]

    x = np.zeros((n, max_timesteps, 10), dtype=np.float32)
    dx = (tx - np.float32(CENTER)).astype(np.float32)
    dy = (ty - np.float32(CENTER)).astype(np.float32)
    radial = np.sqrt(np.maximum(0.0, dx * dx + dy * dy), dtype=np.float32) / _RADIAL_SCALE
    angle = np.arctan2(dy, dx).astype(np.float32)
    angle = np.where(angle < 0.0, angle + _TWO_PI, angle)
    ships_scaled = fs.astype(np.float32) / _SHIP_SCALE
    cap_scaled = cap.astype(np.float32) / _SHIP_SCALE

    x[:, :, 0] = np.where(valid, step_feat[None, :], 0.0)
    x[:, :, 1] = np.where(valid, pprod[:, None], 0.0)
    x[:, :, 2] = np.where(valid, radial, 0.0)
    x[:, :, 3] = np.where(valid, angle, 0.0)
    x[:, :, 4] = np.where(valid & (fo == player), ships_scaled, 0.0)
    x[:, :, 5] = np.where(valid & (fo == -1), ships_scaled, 0.0)
    for k_idx, opp_o in enumerate(opp_ids):
        x[:, :, 6 + k_idx] = np.where(valid & (fo == opp_o), ships_scaled, 0.0)
    x[:, :, 9] = np.where(valid, cap_scaled, 0.0)

    # Match Hober's hidden-comet convention (path_index < 0 -> only step channel populated)
    if np.any(is_comet):
        for i in np.flatnonzero(is_comet):
            pid = enc.planet_ids[i]
            info = comet_lookup.get(pid)
            if info is not None and info[1] < 0:
                x[i, :, :] = 0.0
                v_row = tv[i] & time_mask
                x[i, v_row, 0] = step_feat[v_row]

    body_mask = np.ones((n,), dtype=bool)
    if np.any(is_comet):
        for i in np.flatnonzero(is_comet):
            body_mask[i] = bool(tv[i].any())

    launch_mask = (pown == player)
    target_mask = enc.legal_edges & launch_mask[:, None]
    return x, time_mask, body_mask, launch_mask, target_mask


class FastHoberSparringModel:
    """Batched in-process `#2 Hober Malloc` JAX runner for PPO sparring & fast eval."""

    def __init__(self, *, model_path: Path = HOBER_WEIGHTS) -> None:
        if str(HOBER_DIR) not in sys.path:
            sys.path.insert(0, str(HOBER_DIR))
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault(
            "XLA_FLAGS",
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        )
        from ow_runtime.policy_server import jax_runtime, load_params

        self._jax, self._jnp, self._model_fn = jax_runtime()
        self.params = load_params(Path(model_path))

    def _run_jax_batch(
        self,
        x_np: np.ndarray,
        tm_np: np.ndarray,
        bm_np: np.ndarray,
        lm_np: np.ndarray,
        tgm_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run bucketed JIT-cached JAX inference and return `(launch_logits, target_logits)`."""
        b_real, n_real = x_np.shape[0], x_np.shape[1]
        b_pad = _bucket_batch(b_real)
        n_pad = _bucket_bodies(n_real)

        if b_pad != b_real or n_pad != n_real:
            x_p = np.zeros((b_pad, n_pad, _MAX_TIMESTEPS, 10), dtype=np.float32)
            tm_p = np.zeros((b_pad, _MAX_TIMESTEPS), dtype=bool)
            bm_p = np.zeros((b_pad, n_pad), dtype=bool)
            lm_p = np.zeros((b_pad, n_pad), dtype=bool)
            tgm_p = np.zeros((b_pad, n_pad, n_pad), dtype=bool)
            x_p[:b_real, :n_real] = x_np
            tm_p[:b_real] = tm_np
            bm_p[:b_real, :n_real] = bm_np
            lm_p[:b_real, :n_real] = lm_np
            tgm_p[:b_real, :n_real, :n_real] = tgm_np
        else:
            x_p, tm_p, bm_p, lm_p, tgm_p = x_np, tm_np, bm_np, lm_np, tgm_np

        out = self._model_fn(
            self.params,
            self._jnp.asarray(x_p, dtype=self._jnp.bfloat16),
            time_mask=self._jnp.asarray(tm_p),
            body_mask=self._jnp.asarray(bm_p),
            launch_mask=self._jnp.asarray(lm_p),
            target_mask=self._jnp.asarray(tgm_p),
        )
        ll = np.asarray(out.launch_logits.astype(self._jnp.float32), dtype=np.float32)[:b_real, :n_real]
        tl = np.asarray(out.target_logits.astype(self._jnp.float32), dtype=np.float32)[:b_real, :n_real, :n_real]
        return ll, tl

    def predict_probs_batch(
        self,
        encs: Sequence[EncodedPlanetState],
        shared_phys_list: Sequence[Dict[str, Any]],
        *,
        rotation_tta: bool = False,
        k_rots: Optional[Sequence[int]] = None,
    ) -> List[np.ndarray]:
        """Return exact `(n, n + 1)` action probability matrices for a batch of games."""
        m = len(encs)
        if m == 0:
            return []
        active_indices: List[int] = []
        built: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        max_n = 1
        for idx, (enc, sh) in enumerate(zip(encs, shared_phys_list)):
            n = len(enc.planet_ids)
            if n == 0 or not np.any(enc.owned_mask) or not np.any(enc.legal_edges):
                continue
            item = build_hober_inputs_fast(enc, sh, max_timesteps=_MAX_TIMESTEPS)
            active_indices.append(idx)
            built.append(item)
            if n > max_n:
                max_n = n

        out_probs: List[np.ndarray] = []
        for enc in encs:
            n = len(enc.planet_ids)
            p = np.zeros((n, n + 1), dtype=np.float32)
            if n > 0:
                p[:, 0] = 1.0
            out_probs.append(p)
        if not active_indices:
            return out_probs

        n_views = 4 if rotation_tta else 1
        b_tot = len(active_indices) * n_views
        x_b = np.zeros((b_tot, max_n, _MAX_TIMESTEPS, 10), dtype=np.float32)
        tm_b = np.zeros((b_tot, _MAX_TIMESTEPS), dtype=bool)
        bm_b = np.zeros((b_tot, max_n), dtype=bool)
        lm_b = np.zeros((b_tot, max_n), dtype=bool)
        tgm_b = np.zeros((b_tot, max_n, max_n), dtype=bool)

        for k_pos, (orig_idx, (xi, tmi, bmi, lmi, tgmi)) in enumerate(zip(active_indices, built)):
            ni = xi.shape[0]
            pos_pop = (xi[:, :, 2] != 0.0) | (xi[:, :, 3] != 0.0)
            base_ang = xi[:, :, 3]
            for v_i in range(n_views):
                row = k_pos * n_views + v_i
                rot_k = v_i if rotation_tta else (int(k_rots[orig_idx]) % 4 if k_rots is not None else 0)
                x_b[row, :ni] = xi
                if rot_k != 0:
                    rot_ang = np.mod(base_ang + np.float32(rot_k) * _HALF_PI, _TWO_PI)
                    x_b[row, :ni, :, 3] = np.where(pos_pop, rot_ang, base_ang)
                tm_b[row] = tmi
                bm_b[row, :ni] = bmi
                lm_b[row, :ni] = lmi
                tgm_b[row, :ni, :ni] = tgmi

        ll_raw, tl_raw = self._run_jax_batch(x_b, tm_b, bm_b, lm_b, tgm_b)
        ll_raw = np.clip(ll_raw, -1e4, 1e4)
        tl_raw = np.clip(tl_raw, -1e4, 1e4)
        if n_views > 1:
            ll_all = ll_raw.reshape(len(active_indices), n_views, max_n).mean(axis=1)
            tl_all = tl_raw.reshape(len(active_indices), n_views, max_n, max_n).mean(axis=1)
        else:
            ll_all = ll_raw
            tl_all = tl_raw

        for k_pos, orig_idx in enumerate(active_indices):
            enc = encs[orig_idx]
            n = len(enc.planet_ids)
            ll_g = ll_all[k_pos, :n]
            tl_g = tl_all[k_pos, :n, :n]
            # Sigmoid for launch_prob
            lp_g = 1.0 / (1.0 + np.exp(-np.clip(ll_g, -20.0, 20.0)))
            probs = out_probs[orig_idx]
            for i in range(n):
                if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                    continue
                row_mask = enc.legal_edges[i]
                row_logits = np.where(row_mask, tl_g[i], -np.inf)
                mx = float(np.max(row_logits))
                if not np.isfinite(mx):
                    continue
                exps = np.where(row_mask, np.exp(row_logits - mx), 0.0)
                denom = float(exps.sum())
                if denom <= 0.0 or not np.isfinite(denom):
                    continue
                t_probs = exps / denom
                p_launch = float(np.clip(lp_g[i], 1e-4, 1.0 - 1e-4))
                probs[i, 0] = 1.0 - p_launch
                probs[i, 1:] = p_launch * t_probs
        return out_probs

    def act_batch(
        self,
        encs: Sequence[EncodedPlanetState],
        shared_phys_list: Sequence[Dict[str, Any]],
        *,
        rotation_tta: bool = False,
        launch_threshold: float = 0.53,
        temperature: float = 0.0,
        k_rots: Optional[Sequence[int]] = None,
        rngs: Optional[Sequence[np.random.Generator]] = None,
        max_speed: float = DEFAULT_SHIP_SPEED,
    ) -> List[List[List[float]]]:
        """Decode exact all-in Kaggle moves for a lockstep batch of games against `#2 Hober Malloc`."""
        probs_list = self.predict_probs_batch(
            encs,
            shared_phys_list,
            rotation_tta=rotation_tta,
            k_rots=k_rots,
        )
        moves_batch: List[List[List[float]]] = []
        do_sample = temperature > 1e-4
        for idx, (enc, probs) in enumerate(zip(encs, probs_list)):
            n = len(enc.planet_ids)
            if n == 0 or not np.any(enc.owned_mask):
                moves_batch.append([])
                continue
            exp_set = set(enc.expiring_comet_indices)
            rng_b = rngs[idx] if rngs is not None else None
            moves: List[List[float]] = []
            for i in range(n):
                if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                    continue
                avail = int(enc.planet_rows[i][5])
                if avail <= 0:
                    continue
                forced = i in exp_set
                p_row = probs[i]
                if do_sample:
                    # Tempered sampling around Hober's exact soft distribution for rollout diversity
                    p_safe = np.clip(p_row.astype(np.float64), 1e-7, 1.0)
                    lp_s = np.log(p_safe) / float(temperature)
                    if forced:
                        lp_s[0] = -1e9
                    legal_row = np.concatenate([[True], enc.legal_edges[i]])
                    lp_s = np.where(legal_row, lp_s, -1e9)
                    lp_s -= lp_s.max()
                    w = np.where(legal_row, np.exp(lp_s), 0.0)
                    ws = float(w.sum())
                    if ws <= 0.0:
                        continue
                    w /= ws
                    cls = int(np.searchsorted(np.cumsum(w), rng_b.random() if rng_b is not None else np.random.random()))
                    if cls <= 0 or cls > n:
                        continue
                    dst_j = cls - 1
                else:
                    p_launch = 1.0 - float(p_row[0])
                    if (p_launch < launch_threshold) and (not forced):
                        continue
                    dst_j = int(np.argmax(np.where(enc.legal_edges[i], p_row[1:], -1.0)))
                    if not enc.legal_edges[i, dst_j]:
                        continue
                sol = solve_intercept(
                    enc.planet_rows[i],
                    enc.planet_rows[dst_j],
                    avail,
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
                    moves.append([int(enc.planet_rows[i][0]), float(sol.angle), int(avail)])
            moves_batch.append(moves)
        return moves_batch

    def query_soft_probs_padded(
        self,
        enc: EncodedPlanetState,
        shared_physics: Dict[str, Any],
        *,
        rotation_tta: bool = True,
    ) -> np.ndarray:
        """Return `(MAX_PLANETS, MAX_PLANETS + 1)` exact soft probs for Soft-KD distillation."""
        soft = np.zeros((MAX_PLANETS, MAX_PLANETS + 1), dtype=np.float32)
        soft[:, 0] = 1.0
        n = len(enc.planet_ids)
        if n == 0 or not np.any(enc.owned_mask):
            return soft
        probs = self.predict_probs_batch([enc], [shared_physics], rotation_tta=rotation_tta)[0]
        soft[:n, : n + 1] = probs
        return soft
