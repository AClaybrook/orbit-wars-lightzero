#!/usr/bin/env python3
"""Gen 8 Orbit Wars trainer combining all 3 Top-10 ROI tricks:

1. **Full Soft-Logit Knowledge Distillation (`KL(q_hober || p_ours)`)**:
   Queries `#2 Hober Malloc`'s local JAX policy (`rotation_tta=True` = exact full 2P `1800.1`
   probabilities!) directly in Python to obtain the full soft distribution
   `q_i(HOLD | target_0..N-1)` for every owned planet — ~10x richer signal than 1-hot argmax!
2. **Auxiliary Future-State World-Model Prediction (`#7 Audun` trick)**:
   Trains a lightweight training-only head (`aux_future_head`, zero inference cost) to predict
   every planet's true post-action owner & ship count 10 ticks into the future (`t + 10`).
3. **Dedicated 2P Specialist + All-In Execution (`#5 TonyK` & `#2 SimJeg` tricks)**:
   Trains purely on high-signal 2P boards (Hober-vs-Hober + on-policy student-vs-Hober DAgger)
   for maximum 2P sharpness against `#2 Hober Malloc (1800.1)`.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
HOBER_DIR = REPO_ROOT / "external" / "opponents" / "simjeg_hober_malloc"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orbit_wars_bot.dml_utils import (
    build_optimizer,
    cpu_state_dict,
    one_hot_no_scatter,
    resolve_train_device,
)
from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.planet_transformer import (
    MAX_PLANETS,
    EncodedPlanetState,
    PlanetTimelineTransformer,
    apply_d4_planet_feats,
    extract_planet_features,
    pad_encoded_state,
)


class HoberSoftTeacher:
    """In-process JAX query wrapper for `#2 Hober Malloc`'s exact soft action probabilities."""

    def __init__(self, *, rotation_tta: bool = True, opponent_tta: bool = False) -> None:
        if str(HOBER_DIR) not in sys.path:
            sys.path.insert(0, str(HOBER_DIR))
        os.environ["OW_MODEL_PATH"] = str(HOBER_DIR / "checkpoint" / "weight.npz")
        os.environ["OW_ROTATION_TTA"] = "1" if rotation_tta else "0"
        os.environ["OW_OPPONENT_TTA"] = "1" if opponent_tta else "0"
        os.environ["OW_MAX_TIMESTEPS"] = "20"

        from ow_runtime.policy_server import (
            Policy,
            jax_runtime,
            prepare_request_inputs,
            scatter_compact_outputs,
        )

        self._prepare = prepare_request_inputs
        self._scatter = scatter_compact_outputs
        self._jax, self._jnp, self._model_fn = jax_runtime()
        self.policy = Policy()

    def _build_hober_states(self, obs: Dict[str, Any], config: Any, horizon: int = 20) -> List[List[Any]]:
        """Run an exact 20-step no-action FastOrbitSim rollout to build Hober Malloc's `states` table."""
        sim = FastOrbitSim.from_observation(obs, config)
        all_pids = sorted(int(p[0]) for p in sim.planets)
        owners_map: Dict[int, List[int]] = {pid: [] for pid in all_pids}
        ships_map: Dict[int, List[int]] = {pid: [] for pid in all_pids}
        valid_map: Dict[int, List[bool]] = {pid: [] for pid in all_pids}
        empty_acts = [[] for _ in range(sim.num_agents)]

        for t in range(horizon):
            by_id = {int(p[0]): p for p in sim.planets}
            for pid in all_pids:
                p = by_id.get(pid)
                if p is not None:
                    owners_map[pid].append(int(p[1]))
                    ships_map[pid].append(int(p[5]))
                    valid_map[pid].append(True)
                else:
                    owners_map[pid].append(-1)
                    ships_map[pid].append(0)
                    valid_map[pid].append(False)
            if t + 1 < horizon:
                sim.step(empty_acts)

        return [[pid, owners_map[pid], ships_map[pid], valid_map[pid]] for pid in all_pids]

    def query_soft_probs(
        self,
        obs: Dict[str, Any],
        enc: EncodedPlanetState,
        config: Any = None,
    ) -> np.ndarray:
        """Return exact `(MAX_PLANETS, MAX_PLANETS + 1)` soft teacher probabilities aligned with `enc`."""
        n = len(enc.planet_ids)
        soft = np.zeros((MAX_PLANETS, MAX_PLANETS + 1), dtype=np.float32)
        soft[:, 0] = 1.0
        if n == 0 or not np.any(enc.owned_mask):
            return soft

        # Build legal candidate pairs from enc.legal_edges (ETA <= 20, sun-clear)
        candidates: List[List[int]] = []
        for i in range(n):
            if not enc.owned_mask[i]:
                continue
            src_pid = int(enc.planet_ids[i])
            for j in range(n):
                if enc.legal_edges[i, j]:
                    candidates.append([src_pid, int(enc.planet_ids[j])])
        if not candidates:
            return soft

        req = {
            "id": 1,
            "player": int(enc.player),
            "observation": obs,
            "states": self._build_hober_states(obs, config, horizon=20),
            "candidates": candidates,
        }
        prepared = self._prepare(
            req,
            rotation_tta=self.policy.rotation_tta,
            opponent_tta=self.policy.opponent_tta,
            max_timesteps=self.policy.max_timesteps,
        )
        if prepared.get("status") != "OK":
            return soft

        x = prepared["x"]
        model_body_slots = int(x.shape[1])
        batch_size = int(prepared["batch_size"])
        out = self._model_fn(
            self.policy.params,
            self._jnp.asarray(x, dtype=self._jnp.bfloat16),
            time_mask=self._jnp.asarray(prepared["time_mask"]),
            body_mask=self._jnp.asarray(prepared["body_mask"]),
            launch_mask=self._jnp.asarray(prepared["launch_mask"]),
            target_mask=self._jnp.asarray(prepared["target_mask"]),
        )
        launch_logits = out.launch_logits.astype(self._jnp.float32)
        target_logits = out.target_logits.astype(self._jnp.float32)
        if batch_size > 1:
            launch_logits = launch_logits.reshape((1, batch_size, model_body_slots)).mean(axis=1)
            target_logits = target_logits.reshape((1, batch_size, model_body_slots, model_body_slots)).mean(axis=1)
        else:
            launch_logits = launch_logits.reshape((1, model_body_slots))
            target_logits = target_logits.reshape((1, model_body_slots, model_body_slots))

        launch_prob = np.asarray(self._jax.nn.sigmoid(launch_logits), dtype=np.float32)[0]
        target_logits_np = np.asarray(target_logits, dtype=np.float32)[0]
        lp_full, tl_full = self._scatter(
            launch_prob, target_logits_np, prepared["active_indices"], prepared["n_bodies"]
        )

        hober_indices = prepared["indices"]  # planet_id -> hober_slot_idx
        tmask_2d = np.asarray(prepared["full_target_mask"][0], dtype=bool)
        id_to_enc_idx = {int(pid): idx for idx, pid in enumerate(enc.planet_ids)}

        for i in range(n):
            if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                continue
            src_pid = int(enc.planet_ids[i])
            h_src = hober_indices.get(src_pid)
            if h_src is None or h_src < 0 or h_src >= prepared["n_bodies"]:
                continue

            row_mask = tmask_2d[h_src]
            if not bool(row_mask.any()):
                continue
            row_logits = tl_full[h_src]
            masked = np.where(row_mask, row_logits, -np.inf)
            mx = float(np.max(masked))
            if not np.isfinite(mx):
                continue
            exps = np.where(row_mask, np.exp(masked - mx), 0.0)
            denom = float(np.sum(exps))
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            t_probs = exps / denom

            p_launch = float(np.clip(lp_full[h_src], 1e-4, 1.0 - 1e-4))
            soft[i, :] = 0.0
            soft[i, 0] = 1.0 - p_launch

            # Scatter target probabilities onto our exact enc planet indices
            mapped_sum = 0.0
            for dst_pid, h_dst in hober_indices.items():
                if 0 <= h_dst < prepared["n_bodies"] and row_mask[h_dst]:
                    j = id_to_enc_idx.get(int(dst_pid))
                    if j is not None and enc.legal_edges[i, j]:
                        p_j = float(t_probs[h_dst])
                        soft[i, 1 + j] = p_launch * p_j
                        mapped_sum += p_j
            # Renormalize cleanly in case a comet slot wasn't mapped
            row_tot = float(soft[i, : n + 1].sum())
            if row_tot > 0.0:
                soft[i, : n + 1] /= row_tot
            else:
                soft[i, :] = 0.0
                soft[i, 0] = 1.0

        return soft


def _extract_aux_future_targets(
    enc_t: EncodedPlanetState,
    obs_future: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build true `t + 10` auxiliary world-model targets aligned to `enc_t.planet_ids`."""
    aux_owner = np.full((MAX_PLANETS,), 2, dtype=np.int64)  # 0=mine, 1=enemy, 2=neutral
    aux_ships = np.zeros((MAX_PLANETS,), dtype=np.float32)
    aux_valid = np.zeros((MAX_PLANETS,), dtype=bool)
    if obs_future is None:
        return aux_owner, aux_ships, aux_valid

    fut_by_id = {int(row[0]): (int(row[1]), int(row[5])) for row in obs_future.get("planets", [])}
    player = int(enc_t.player)
    for i, pid in enumerate(enc_t.planet_ids):
        rec = fut_by_id.get(int(pid))
        if rec is None:
            continue
        owner, ships = rec
        aux_owner[i] = 0 if owner == player else (2 if owner < 0 else 1)
        aux_ships[i] = float(np.log1p(max(0, ships)) / 6.0)
        aux_valid[i] = True
    return aux_owner, aux_ships, aux_valid


_WORKER_CACHE: Dict[str, Any] = {}


def _collect_soft_kd_worker(task: Tuple[int, bool, Optional[str], int, int, int]) -> Dict[str, Any]:
    from kaggle_environments import make

    from evaluate import resolve_agent

    seed, is_dagger, student_checkpoint, sample_stride, future_horizon, fast_rollout_opp = task
    if fast_rollout_opp:
        # Speed up the rollout opponent only; HoberSoftTeacher ALWAYS queries full rotation_tta=True (1800.1 2P)!
        os.environ["OW_ROTATION_TTA"] = "0"
        os.environ["OW_OPPONENT_TTA"] = "0"
    if "boss_agent" not in _WORKER_CACHE:
        _WORKER_CACHE["boss_agent"] = resolve_agent("hober_malloc")
    if "teacher" not in _WORKER_CACHE:
        _WORKER_CACHE["teacher"] = HoberSoftTeacher(rotation_tta=True, opponent_tta=False)
    stu_key = f"stu:{student_checkpoint}"
    if is_dagger and student_checkpoint and stu_key not in _WORKER_CACHE:
        _WORKER_CACHE[stu_key] = resolve_agent(f"pt:{student_checkpoint}")

    boss_agent = _WORKER_CACHE["boss_agent"]
    teacher = _WORKER_CACHE["teacher"]
    student_agent = _WORKER_CACHE.get(stu_key) if (is_dagger and student_checkpoint) else None

    random.seed(seed)
    np.random.seed(seed)
    env = make("orbit_wars", configuration={"seed": seed, "actTimeout": 30, "agentTimeout": 60}, debug=False)
    if is_dagger and student_agent is not None:
        stu_slot = seed % 2
        roster = [student_agent, boss_agent] if stu_slot == 0 else [boss_agent, student_agent]
        record_slots = [stu_slot]
    else:
        roster = [boss_agent, boss_agent]
        record_slots = [0, 1]

    steps = env.run(roster)
    n_ticks = len(steps) - 1
    rewards = [steps[-1][0]["reward"], steps[-1][1]["reward"]]

    pf_l, tf_l, ef_l, le_l, om_l, pm_l = [], [], [], [], [], []
    sp_l, vt_l, ao_l, as_l, av_l = [], [], [], [], []
    agree_launch = tot_dec = 0

    for slot in record_slots:
        r_self = rewards[slot] or 0
        r_opp = rewards[1 - slot] or 0
        val_target = 1.0 if r_self > r_opp else (-1.0 if r_self < r_opp else 0.0)
        stride = 1 if (is_dagger and val_target < 0) else sample_stride

        for t in range(0, n_ticks):
            obs_t = steps[t][slot]["observation"]
            has_comets = bool(obs_t.get("comets"))
            in_midgame = 22 <= t <= 78
            if not (in_midgame or has_comets or (t % stride == 0)):
                continue
            obs_t_copy = dict(obs_t)
            obs_t_copy["player"] = slot
            enc = extract_planet_features(obs_t_copy, player=slot)
            if not enc.planet_ids or not np.any(enc.owned_mask):
                continue

            soft_probs = teacher.query_soft_probs(obs_t_copy, enc, env.configuration)
            pad = pad_encoded_state(enc)
            obs_fut = steps[t + future_horizon][slot]["observation"] if (t + future_horizon < len(steps)) else None
            ao, ash, av = _extract_aux_future_targets(enc, obs_fut)

            if not is_dagger:
                actual_acts = steps[t + 1][slot].get("action") or []
                actual_srcs = {int(m[0]) for m in actual_acts}
                for i, pid in enumerate(enc.planet_ids):
                    if enc.owned_mask[i] and np.any(enc.legal_edges[i]):
                        pred_launch = (1.0 - float(soft_probs[i, 0])) >= 0.50
                        actual_launch = int(pid) in actual_srcs
                        tot_dec += 1
                        if pred_launch == actual_launch:
                            agree_launch += 1

            pf_l.append(pad["planet_feats"])
            tf_l.append(pad["timeline_feats"])
            ef_l.append(pad["edge_feats"])
            le_l.append(pad["legal_edges"])
            om_l.append(pad["owned_mask"])
            pm_l.append(pad["pad_mask"])
            sp_l.append(soft_probs.astype(np.float16))
            vt_l.append(val_target)
            ao_l.append(ao)
            as_l.append(ash)
            av_l.append(av)

    return {
        "planet_feats": np.stack(pf_l).astype(np.float32),
        "timeline_feats": np.stack(tf_l).astype(np.float32),
        "edge_feats": np.stack(ef_l).astype(np.float32),
        "legal_edges": np.stack(le_l),
        "owned_mask": np.stack(om_l),
        "pad_mask": np.stack(pm_l),
        "soft_probs": np.stack(sp_l).astype(np.float16),
        "val_targets": np.array(vt_l, dtype=np.float32),
        "aux_owner": np.stack(ao_l).astype(np.int64),
        "aux_ships": np.stack(as_l).astype(np.float32),
        "aux_valid": np.stack(av_l),
        "agree_launch": agree_launch,
        "tot_dec": tot_dec,
    }


def collect_soft_kd_dataset(
    *,
    hober_games: int = 2,
    dagger_games: int = 16,
    student_checkpoint: Optional[str] = None,
    seed: int = 12000,
    sample_stride: int = 3,
    future_horizon: int = 10,
    workers: int = 1,
) -> Dict[str, np.ndarray]:
    """Collect 2P games with exact Full-TTA `#2 Hober Malloc` soft probabilities + `t+10` future targets."""
    import multiprocessing as mp

    tasks: List[Tuple[int, bool, Optional[str], int, int, int]] = []
    if student_checkpoint:
        for g in range(dagger_games):
            tasks.append((seed + 1000 + g, True, student_checkpoint, sample_stride, future_horizon, 1))
    for g in range(hober_games):
        tasks.append((seed + g, False, None, sample_stride, future_horizon, 1))

    t0 = time.time()
    parts: List[Dict[str, Any]] = []
    tot_states = agree_launch = tot_dec = 0
    print(f"[SoftKD Collect] Running {len(tasks)} 2P games (workers={workers}, exact full-TTA soft probs)...", flush=True)

    if workers <= 1:
        for idx, task in enumerate(tasks, 1):
            res = _collect_soft_kd_worker(task)
            parts.append(res)
            tot_states += len(res["val_targets"])
            agree_launch += res["agree_launch"]
            tot_dec += res["tot_dec"]
            pstr = f" | parity={100.0*agree_launch/max(1,tot_dec):.1f}%" if tot_dec else ""
            print(f"  [SoftKD Collect] {idx}/{len(tasks)} ({time.time()-t0:.1f}s) | states={tot_states}{pstr}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for idx, res in enumerate(pool.imap_unordered(_collect_soft_kd_worker, tasks), 1):
                parts.append(res)
                tot_states += len(res["val_targets"])
                agree_launch += res["agree_launch"]
                tot_dec += res["tot_dec"]
                pstr = f" | parity={100.0*agree_launch/max(1,tot_dec):.1f}%" if tot_dec else ""
                print(f"  [SoftKD Collect] {idx}/{len(tasks)} ({time.time()-t0:.1f}s) | states={tot_states}{pstr}", flush=True)

    keys = [k for k in parts[0].keys() if k not in ("agree_launch", "tot_dec")]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def compute_soft_kd_loss(
    model: PlanetTimelineTransformer,
    pf: torch.Tensor,
    tf: torch.Tensor,
    ef: torch.Tensor,
    le: torch.Tensor,
    om: torch.Tensor,
    pm: torch.Tensor,
    soft_probs: torch.Tensor,
    vtar: torch.Tensor,
    aux_owner: torch.Tensor,
    aux_ships: torch.Tensor,
    aux_valid: torch.Tensor,
    *,
    aux_weight: float = 0.20,
    value_weight: float = 0.25,
    launch_boost: float = 1.25,
    midgame_weight: float = 1.0,
    comet_weight: float = 1.0,
    hold_boost: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Soft-KL policy distillation + Auxiliary t+10 world-model + Value loss.

    Gen 9 extensions (default 1.0 for backward compatibility):
      - ``midgame_weight``: multiplies per-board KL weight during ``t in [25, 75]``.
      - ``comet_weight``: multiplies per-board KL weight whenever comets are active.
      - ``hold_boost``: boosts KL weight on midgame decisions where teacher prefers HOLD
        (``q_hold >= 0.65``), curbing impulsive all-in garrison shuffles.
    """
    out = model(pf, tf, ef, pm, return_aux=True)
    if len(out) == 4:
        logits, fracs, val, aux_pred = out
    else:
        logits, fracs, val = out
        aux_pred = None

    B, N, _ = logits.shape
    hold_ok = torch.ones((B, N, 1), dtype=torch.bool, device=logits.device)
    act_mask = torch.cat([hold_ok, le], dim=-1)
    logits = logits.masked_fill(~act_mask, -1e9)
    log_p = F.log_softmax(logits, dim=-1)

    valid_dec = om & le.any(dim=-1)
    valid_f = valid_dec.float()
    valid_cnt = float(valid_f.sum().item())
    if valid_cnt > 0.0:
        # Per-board phase & comet emphasis derived losslessly from planet_feats
        # col 16 = step / 500.0, col 6 = is_comet
        steps = pf[:, 0, 16] * 500.0  # (B,)
        is_mid = (steps >= 24.5) & (steps <= 75.5)
        has_comet = ((pf[:, :, 6] > 0.5) & pm).any(dim=-1)
        board_w = torch.ones((B,), dtype=torch.float32, device=logits.device)
        if midgame_weight != 1.0:
            board_w = torch.where(is_mid, board_w * float(midgame_weight), board_w)
        if comet_weight != 1.0:
            board_w = torch.where(has_comet, board_w * float(comet_weight), board_w)
        node_board_w = board_w.unsqueeze(1).expand(B, N)
        node_is_mid = is_mid.unsqueeze(1).expand(B, N)

        q_all = soft_probs.float()
        q_all = q_all / q_all.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        # Exact KL(q || p) = sum q * (log q - log p)
        kl_per = (q_all * (torch.log(q_all.clamp_min(1e-6)) - log_p)).sum(dim=-1)
        # Per-decision weighting: launch boost, midgame hold calibration, phase & comet weights
        q_hold = q_all[..., 0]
        q_launch = 1.0 - q_hold
        w = (1.0 + (launch_boost - 1.0) * q_launch) * node_board_w
        if hold_boost != 1.0:
            mid_hold = node_is_mid & (q_hold >= 0.65)
            w = torch.where(mid_hold, w * float(hold_boost), w)
        w_masked = w * valid_f
        kd_loss = (kl_per * w_masked).sum() / w_masked.sum().clamp_min(1.0)

        with torch.no_grad():
            pred_cls = log_p.argmax(dim=-1)
            teacher_cls = q_all.argmax(dim=-1)
            acc_all = float(((pred_cls == teacher_cls).float() * valid_f).sum().item() / valid_cnt)
            lm_f = (valid_dec & (teacher_cls > 0)).float()
            lm_cnt = float(lm_f.sum().item())
            if lm_cnt > 0.0:
                acc_launch = float(((pred_cls == teacher_cls).float() * lm_f).sum().item() / lm_cnt)
                tgt_pred = 1 + log_p[..., 1:].argmax(dim=-1)
                acc_tgt = float(((tgt_pred == teacher_cls).float() * lm_f).sum().item() / lm_cnt)
            else:
                acc_launch = acc_tgt = 0.0

        # All-in calibration for fraction head on teacher launches (Hober Malloc always launches 100%)
        q_top_cls = q_all.argmax(dim=-1)
        lmask_f = (valid_dec & (q_top_cls > 0)).float()
        if float(lmask_f.sum().item()) > 0.0:
            dst_idx = (q_top_cls - 1).clamp(0, N - 1)
            dst_oh = one_hot_no_scatter(dst_idx, N, dtype=fracs.dtype)
            f_pred = (fracs * dst_oh).sum(dim=-1)
            frac_loss = (F.smooth_l1_loss(f_pred, torch.ones_like(f_pred), reduction="none") * lmask_f).sum() / lmask_f.sum().clamp_min(1.0)
        else:
            frac_loss = fracs.sum() * 0.0
    else:
        kd_loss = logits.sum() * 0.0
        frac_loss = fracs.sum() * 0.0
        acc_all = acc_launch = acc_tgt = 0.0

    # Auxiliary t+10 world-model loss (#7 Audun trick)
    av_f = aux_valid.float()
    av_cnt = float(av_f.sum().item())
    if aux_pred is not None and av_cnt > 0.0:
        av_denom = av_f.sum().clamp_min(1.0)
        ao_logp = F.log_softmax(aux_pred[..., :3], dim=-1)
        ao_oh = one_hot_no_scatter(aux_owner.clamp(0, 2), 3, dtype=ao_logp.dtype)
        ao_ce = -(ao_logp * ao_oh).sum(dim=-1)
        ao_loss = (ao_ce * av_f).sum() / av_denom
        as_loss = (F.smooth_l1_loss(aux_pred[..., 3], aux_ships, reduction="none") * av_f).sum() / av_denom
        aux_loss = ao_loss + as_loss
        with torch.no_grad():
            aux_owner_acc = float(((aux_pred[..., :3].argmax(dim=-1) == aux_owner).float() * av_f).sum().item() / av_cnt)
    else:
        aux_loss = logits.sum() * 0.0
        aux_owner_acc = 0.0

    val_loss = F.mse_loss(val, vtar)
    vsign = float(((val >= 0) == (vtar >= 0)).float().mean().item())
    total = kd_loss + 0.15 * frac_loss + aux_weight * aux_loss + value_weight * val_loss
    return total, {
        "loss": float(total.item()),
        "kl": float(kd_loss.item()),
        "aux": float(aux_loss.item()),
        "aux_acc": aux_owner_acc,
        "acc_all": acc_all,
        "acc_launch": acc_launch,
        "acc_tgt": acc_tgt,
        "vsign": vsign,
    }


def train_soft_kd(
    data: Dict[str, np.ndarray],
    *,
    resume_from: str,
    out_path: Path,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 3.5e-4,
    aux_weight: float = 0.20,
    launch_boost: float = 1.25,
    midgame_weight: float = 1.0,
    comet_weight: float = 1.0,
    hold_boost: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
    train_threads: int = 4,
) -> Path:
    torch.set_num_threads(max(1, int(train_threads)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_dev, use_dml = resolve_train_device(device)

    n = len(data["val_targets"])
    perm = np.random.permutation(n)
    n_val = max(1, int(0.12 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def make_ds(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(data["planet_feats"][idx]),
            torch.from_numpy(data["timeline_feats"][idx]),
            torch.from_numpy(data["edge_feats"][idx]),
            torch.from_numpy(data["legal_edges"][idx]),
            torch.from_numpy(data["owned_mask"][idx]),
            torch.from_numpy(data["pad_mask"][idx]),
            torch.from_numpy(data["soft_probs"][idx].astype(np.float32)),
            torch.from_numpy(data["val_targets"][idx]),
            torch.from_numpy(data["aux_owner"][idx]),
            torch.from_numpy(data["aux_ships"][idx]),
            torch.from_numpy(data["aux_valid"][idx]),
        )

    tr_loader = DataLoader(make_ds(tr_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_ds(val_idx), batch_size=batch_size, shuffle=False)

    ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    cfg["use_aux_future"] = True
    model = PlanetTimelineTransformer(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(train_dev)
    n_params = sum(p.numel() for p in model.parameters())
    dev_tag = "dml" if use_dml else str(train_dev)
    print(
        f"[SoftKD Train] Resumed {resume_from} -> {n_params/1e3:.1f}K params "
        f"(new aux keys={len(missing)}, unexpected={len(unexpected)}, device={dev_tag}) | "
        f"train={len(tr_idx)} val={len(val_idx)} | epochs={epochs} batch={batch_size} lr={lr} | "
        f"mid_w={midgame_weight} comet_w={comet_weight} hold_b={hold_boost} launch_b={launch_boost}",
        flush=True,
    )

    opt = build_optimizer(model.parameters(), lr=lr, weight_decay=1e-4, use_dml=use_dml)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    best_kl = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_m = []
        for pf, tf, ef, le, om, pm, sp, vt, ao, ash, av in tr_loader:
            k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
            pf_a = torch.from_numpy(apply_d4_planet_feats(pf.numpy(), k_rot=k_rot, flip_x=flip_x)).to(train_dev)
            tf, ef, le, om, pm, sp, vt, ao, ash, av = (
                t.to(train_dev) for t in (tf, ef, le, om, pm, sp, vt, ao, ash, av)
            )
            opt.zero_grad(set_to_none=True)
            loss, m = compute_soft_kd_loss(
                model, pf_a, tf, ef, le, om, pm, sp, vt, ao, ash, av,
                aux_weight=aux_weight,
                launch_boost=launch_boost,
                midgame_weight=midgame_weight,
                comet_weight=comet_weight,
                hold_boost=hold_boost,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=not use_dml)
            opt.step()
            tr_m.append(m)
        sched.step()

        model.eval()
        val_m = []
        with torch.no_grad():
            for batch in val_loader:
                pf, tf, ef, le, om, pm, sp, vt, ao, ash, av = (t.to(train_dev) for t in batch)
                _, m = compute_soft_kd_loss(
                    model, pf, tf, ef, le, om, pm, sp, vt, ao, ash, av,
                    aux_weight=aux_weight,
                    launch_boost=launch_boost,
                    midgame_weight=midgame_weight,
                    comet_weight=comet_weight,
                    hold_boost=hold_boost,
                )
                val_m.append(m)

        def avg(lst, k):
            return float(np.mean([x[k] for x in lst])) if lst else 0.0

        v_kl = avg(val_m, "kl")
        if v_kl < best_kl:
            best_kl = v_kl
            torch.save(
                {
                    "model_config": model.config,
                    "model_state_dict": cpu_state_dict(model),
                    "epoch": ep,
                    "val_kl": v_kl,
                    "val_acc_all": avg(val_m, "acc_all"),
                    "val_acc_launch": avg(val_m, "acc_launch"),
                    "val_acc_target": avg(val_m, "acc_tgt"),
                    "val_aux_owner_acc": avg(val_m, "aux_acc"),
                    "val_sign": avg(val_m, "vsign"),
                },
                out_path,
            )
        print(
            f"  ep {ep:2d}/{epochs} ({time.time()-t0:5.1f}s) | tr_kl={avg(tr_m,'kl'):.3f} val_kl={v_kl:.3f} | "
            f"val_all={100*avg(val_m,'acc_all'):5.1f}% val_launch={100*avg(val_m,'acc_launch'):5.1f}% "
            f"val_tgt={100*avg(val_m,'acc_tgt'):5.1f}% aux_own={100*avg(val_m,'aux_acc'):5.1f}% "
            f"vsign={100*avg(val_m,'vsign'):5.1f}%",
            flush=True,
        )
    print(f"[SoftKD Train] Saved best 2P Specialist -> {out_path} ({out_path.stat().st_size/1024:.1f} KB)", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gen 8/9 Soft-Logit KD + Aux Future World Model + Midgame Calibration.")
    parser.add_argument("--dataset-cache", required=True, help="Path to .npz soft-KD dataset cache.")
    parser.add_argument("--hober-games", type=int, default=16)
    parser.add_argument("--dagger-games", type=int, default=20)
    parser.add_argument("--student-checkpoint", default="checkpoints/planet_transformer_gen8_2p_softkd.pt")
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--append", action="store_true", help="Collect new games and append onto existing dataset-cache.")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume", default="checkpoints/planet_transformer_gen8_2p_softkd.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument("--launch-boost", type=float, default=1.0)
    parser.add_argument("--midgame-weight", type=float, default=2.5)
    parser.add_argument("--comet-weight", type=float, default=2.0)
    parser.add_argument("--hold-boost", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--device", choices=["cpu", "dml", "cuda", "auto"], default="cpu")
    parser.add_argument("--train-threads", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/planet_transformer_gen9_softkd.pt")
    args = parser.parse_args(argv)

    cache = Path(args.dataset_cache)
    existing: Optional[Dict[str, np.ndarray]] = None
    if cache.is_file():
        print(f"[SoftKD] Loading cached dataset from {cache}...", flush=True)
        with np.load(cache) as npz:
            existing = {k: npz[k] for k in npz.files}
        print(f"[SoftKD] Loaded {len(existing['val_targets'])} soft-KD 2P boards.", flush=True)

    if existing is None or args.append:
        fresh = collect_soft_kd_dataset(
            hober_games=args.hober_games,
            dagger_games=args.dagger_games,
            student_checkpoint=args.student_checkpoint,
            seed=args.seed,
            sample_stride=args.sample_stride,
            workers=args.workers,
        )
        if existing is not None:
            data = {k: np.concatenate([existing[k], fresh[k]], axis=0) for k in existing}
        else:
            data = fresh
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
        print(f"[SoftKD] Cached {len(data['val_targets'])} total boards -> {cache} ({cache.stat().st_size/1e6:.2f} MB)", flush=True)
    else:
        data = existing

    if args.collect_only:
        return 0

    train_soft_kd(
        data,
        resume_from=args.resume,
        out_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        aux_weight=args.aux_weight,
        launch_boost=args.launch_boost,
        midgame_weight=args.midgame_weight,
        comet_weight=args.comet_weight,
        hold_boost=args.hold_boost,
        seed=args.seed,
        device=args.device,
        train_threads=args.train_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
