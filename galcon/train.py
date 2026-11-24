"""Train & benchmark Galcon PlanetTimelineTransformer (v1 baseline + Gen 2 Teacher-Free Self-Play Flywheel).

Gen 2 Teacher-Free Self-Play RL Flywheel:
1. **Multi-Style Sparring Suite**: ``GalconRusherBot`` (all-in / blitz / adaptive),
   ``GalconClassicBot``, ``GalconTacticalBot``, and ``GalconMacroBot``.
2. **Stage 1 Warm-Start + World Model + Native ``/REDIR``**: Multi-style games supervising
   the ~1.26M ``GalconPlanetTransformer`` policy, fraction, value, auxiliary ``t+10`` future-state
   world-model head, and rollout-verified mid-flight ``/REDIR`` head.
3. **Stage 2 AlphaZero Search Self-Play Distillation (Expert Iteration)**: Collects multi-map
   games played by ``GalconSearchTransformerAgent`` against itself and the sparring suite,
   distilling Boltzmann soft search plans, exact fractions, rollout-verified redirects, and
   blended game outcomes into the pure ~20 ms neural network.
4. **Stage 3 Large-Scale League PPO + Historical Snapshot Pool + Moving EMA Teacher KL**:
   400-500+ episodes of teacher-free league self-play with GAE, joint target + send-fraction
   optimization, auxiliary ``t+10`` world-model replay, and native ``/REDIR`` co-adaptation.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from galcon.heuristic_bot import (
    GalconClassicBot,
    GalconMacroBot,
    GalconRusherBot,
    GalconTacticalBot,
)
from galcon.sim import GalconSim, Order, RedirOrder, SendOrder, evaluate_galcon_state
from galcon.transformer_bot import (
    GALCON_MAX_PLANETS,
    GALCON_REDIR_DIM,
    GalconPlanetTransformer,
    GalconSearchTransformerAgent,
    GalconTransformerAgent,
    apply_galcon_d4,
    build_galcon_gen2_transformer,
    build_galcon_transformer,
    compute_smart_redirects,
    extract_galcon_features,
    extract_galcon_redir_features,
    pad_galcon_state,
)

MAX_REDIR_FLEETS = 8


def _sample_map_config(rng: random.Random) -> Tuple[int, float, float]:
    """Mix compact tactical maps (8-12 planets) and classic 14-22 planet Galcon maps."""
    if rng.random() < 0.40:
        return rng.randint(8, 12), 240.0, 150.0
    return rng.randint(14, 22), 400.0, 240.0


def _orders_to_targets(enc, orders: Sequence[Order]) -> Tuple[np.ndarray, np.ndarray]:
    tcls = np.zeros((GALCON_MAX_PLANETS,), dtype=np.int64)
    tfrac = np.zeros((GALCON_MAX_PLANETS,), dtype=np.float32)
    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    seen_src = set()
    for o in orders:
        if not isinstance(o, SendOrder):
            continue
        si = id_to_idx.get(o.src_id)
        di = id_to_idx.get(o.dst_id)
        if si is not None and di is not None and enc.owned_mask[si] and enc.legal_edges[si, di]:
            # Keep the primary (highest-priority first) send order per planet
            if si not in seen_src:
                tcls[si] = di + 1
                tfrac[si] = float(o.percent) / 100.0
                seen_src.add(si)
    return tcls, tfrac


def _snapshot_planet_state(sim: GalconSim, planet_ids: Sequence[int], player: int) -> Tuple[np.ndarray, np.ndarray]:
    """Snapshot per-planet ownership class (0=neu, 1=mine, 2=enemy) and log-ships for aux t+10 head."""
    own = np.zeros((GALCON_MAX_PLANETS,), dtype=np.int64)
    shp = np.zeros((GALCON_MAX_PLANETS,), dtype=np.float32)
    for idx, pid in enumerate(planet_ids):
        p = sim.planets[pid]
        own[idx] = 0 if p.owner == 0 else (1 if p.owner == player else 2)
        shp[idx] = math.log1p(max(0.0, float(p.ships))) / 5.5
    return own, shp


def _pad_redir_sample(
    enc,
    renc,
    labels: Dict[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pad fleet redirect features & verified targets up to ``MAX_REDIR_FLEETS``."""
    f_cnt = min(MAX_REDIR_FLEETS, len(renc.fleet_ids))
    if f_cnt == 0:
        return None
    n = len(enc.planet_ids)
    cur_idx = np.zeros((MAX_REDIR_FLEETS,), dtype=np.int64)
    rfeats = np.zeros((MAX_REDIR_FLEETS, GALCON_MAX_PLANETS, GALCON_REDIR_DIM), dtype=np.float32)
    rtar = np.zeros((MAX_REDIR_FLEETS,), dtype=np.int64)
    rmask = np.zeros((MAX_REDIR_FLEETS,), dtype=bool)
    for fi in range(f_cnt):
        fid = renc.fleet_ids[fi]
        cur_idx[fi] = renc.cur_dst_idx[fi]
        rfeats[fi, :n] = renc.redir_feats[fi, :n]
        rtar[fi] = int(labels.get(fid, int(renc.cur_dst_idx[fi])))
        rmask[fi] = True
    return cur_idx, rfeats, rtar, rmask


def _trim_batch_n(pf, tf, ef, le, om, pm, *extra_node_tensors):
    """Dynamically trim padded planet dimension to ``max_n`` in batch (~2x faster CPU attention)."""
    max_n = max(2, int(pm.sum(dim=1).max().item()))
    if max_n >= pf.shape[1]:
        return (pf, tf, ef, le, om, pm, *extra_node_tensors)
    pf = pf[:, :max_n]
    tf = tf[:, :max_n]
    ef = ef[:, :max_n, :max_n]
    le = le[:, :max_n, :max_n]
    om = om[:, :max_n]
    pm = pm[:, :max_n]
    trimmed = []
    for t in extra_node_tensors:
        if t is None:
            trimmed.append(None)
        elif t.ndim == 2:
            trimmed.append(t[:, :max_n])
        elif t.ndim == 3 and t.shape[1] == GALCON_MAX_PLANETS and t.shape[2] == GALCON_MAX_PLANETS + 1:
            # Soft policy targets (B, N, N+1)
            trimmed.append(t[:, :max_n, : max_n + 1])
        elif t.ndim == 4 and t.shape[2] == GALCON_MAX_PLANETS:
            # Redir feats (B, F, N, R)
            trimmed.append(t[:, :, :max_n])
        else:
            trimmed.append(t)
    return (pf, tf, ef, le, om, pm, *trimmed)


# ---------------------------------------------------------------------------
# v1 Legacy Training Functions (preserved for exact apples-to-apples baseline)
# ---------------------------------------------------------------------------

def collect_teacher_dataset(
    *,
    num_games: int = 80,
    seed: int = 1000,
    sample_stride: int = 2,
    dagger_agent: GalconTransformerAgent | None = None,
) -> Dict[str, np.ndarray]:
    """Collect multi-map Galcon states labeled by ``GalconTacticalBot`` (v1 baseline)."""
    rng = random.Random(seed)
    teacher_hi = GalconTacticalBot(aggression=1.05)
    teacher_lo = GalconTacticalBot(aggression=0.90)
    classic = GalconClassicBot()

    pf_l, tf_l, ef_l, le_l, om_l, pm_l, tc_l, fr_l, vt_l = [], [], [], [], [], [], [], [], []
    t0 = time.time()

    for g in range(num_games):
        n_pl, w, h = _sample_map_config(rng)
        sim = GalconSim.create(seed=seed + g, num_planets=n_pl, width=w, height=h, max_ticks=220)
        if dagger_agent is not None:
            student_pid = 1 if (g % 2 == 0) else 2
            opp_bot = teacher_hi
        else:
            student_pid = 0
            opp_bot = teacher_lo if (g % 3 != 2) else classic

        traj: List[Tuple[int, Dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
        while not sim.done:
            if dagger_agent is not None:
                o_stu = dagger_agent.act(sim, student_pid)
                o_opp = opp_bot.act(sim, 2 if student_pid == 1 else 1)
                o1 = o_stu if student_pid == 1 else o_opp
                o2 = o_opp if student_pid == 1 else o_stu
                if sim.tick % sample_stride == 0:
                    oracle_orders = teacher_hi.act(sim, student_pid)
                    enc = extract_galcon_features(sim, student_pid, feature_version=1)
                    if np.any(enc.owned_mask):
                        pad = pad_galcon_state(enc)
                        tc, fr = _orders_to_targets(enc, oracle_orders)
                        traj.append((student_pid, pad, tc, fr))
            else:
                o1 = teacher_hi.act(sim, 1)
                o2 = opp_bot.act(sim, 2)
                for pid_rec, ords in ((1, o1), (2, o2)) if opp_bot is teacher_lo else ((1, o1),):
                    if (sim.tick % sample_stride == 0) or any(isinstance(x, SendOrder) for x in ords):
                        enc = extract_galcon_features(sim, pid_rec, feature_version=1)
                        if np.any(enc.owned_mask):
                            pad = pad_galcon_state(enc)
                            tc, fr = _orders_to_targets(enc, ords)
                            traj.append((pid_rec, pad, tc, fr))
            sim.step({1: o1, 2: o2})

        win = sim.winner()
        for pid_rec, pad, tc, fr in traj:
            v = 1.0 if win == pid_rec else (0.0 if win == 0 else -1.0)
            pf_l.append(pad["planet_feats"])
            tf_l.append(pad["timeline_feats"])
            ef_l.append(pad["edge_feats"])
            le_l.append(pad["legal_edges"])
            om_l.append(pad["owned_mask"])
            pm_l.append(pad["pad_mask"])
            tc_l.append(tc)
            fr_l.append(fr)
            vt_l.append(v)

    data = {
        "planet_feats": np.stack(pf_l).astype(np.float32),
        "timeline_feats": np.stack(tf_l).astype(np.float32),
        "edge_feats": np.stack(ef_l).astype(np.float32),
        "legal_edges": np.stack(le_l),
        "owned_mask": np.stack(om_l),
        "pad_mask": np.stack(pm_l),
        "target_cls": np.stack(tc_l).astype(np.int64),
        "target_frac": np.stack(fr_l).astype(np.float32),
        "val_targets": np.array(vt_l, dtype=np.float32),
    }
    mode = "DAgger" if dagger_agent is not None else "BC"
    print(f"[Galcon v1 {mode}] Collected {len(vt_l)} board states across {num_games} games in {time.time()-t0:.1f}s", flush=True)
    return data


def _batch_loss(model, pf, tf, ef, le, om, pm, tcls, tfrac, vtar, launch_weight: float = 1.35):
    pf, tf, ef, le, om, pm, tcls, tfrac = _trim_batch_n(pf, tf, ef, le, om, pm, tcls, tfrac)
    logits, fracs, val = model(pf, tf, ef, pm)
    B, N, _ = logits.shape
    mask = torch.cat([torch.ones((B, N, 1), dtype=torch.bool, device=logits.device), le], dim=-1)
    logits = logits.masked_fill(~mask, -1e9)
    valid = om & le.any(dim=-1)
    if valid.any():
        fl = logits[valid]
        ft = tcls[valid]
        ce = F.cross_entropy(fl, ft, reduction="none")
        w = torch.where(ft > 0, launch_weight, 1.0)
        pol_loss = (ce * w).sum() / w.sum().clamp_min(1.0)
        preds = fl.argmax(dim=-1)
        acc_all = float((preds == ft).float().mean().item())
        lm = ft > 0
        acc_tgt = float((1 + fl[lm, 1:].argmax(dim=-1) == ft[lm]).float().mean().item()) if lm.any() else 0.0
    else:
        pol_loss = logits.sum() * 0.0
        acc_all = acc_tgt = 0.0

    ldec = valid & (tcls > 0)
    if ldec.any():
        dst = (tcls[ldec] - 1).clamp(0, N - 1)
        pf_pred = fracs[ldec].gather(1, dst.unsqueeze(1)).squeeze(1)
        frac_loss = F.smooth_l1_loss(pf_pred, tfrac[ldec])
    else:
        frac_loss = fracs.sum() * 0.0

    v_loss = F.mse_loss(val, vtar)
    vsign = float(((val >= 0) == (vtar >= 0)).float().mean().item())
    total = pol_loss + 0.5 * frac_loss + 0.30 * v_loss
    return total, {"pol": float(pol_loss.item()), "acc_all": acc_all, "acc_tgt": acc_tgt, "vsign": vsign}


def train_supervised(
    data: Dict[str, np.ndarray],
    *,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1.2e-3,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    dim_ff: int = 256,
    out_path: Path,
    resume_from: Path | None = None,
    seed: int = 42,
) -> Path:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = len(data["val_targets"])
    perm = np.random.permutation(n)
    n_val = max(1, int(0.12 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def make_ds(idx):
        return TensorDataset(
            torch.from_numpy(data["planet_feats"][idx]),
            torch.from_numpy(data["timeline_feats"][idx]),
            torch.from_numpy(data["edge_feats"][idx]),
            torch.from_numpy(data["legal_edges"][idx]),
            torch.from_numpy(data["owned_mask"][idx]),
            torch.from_numpy(data["pad_mask"][idx]),
            torch.from_numpy(data["target_cls"][idx]),
            torch.from_numpy(data["target_frac"][idx]),
            torch.from_numpy(data["val_targets"][idx]),
        )

    tr_loader = DataLoader(make_ds(tr_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_ds(val_idx), batch_size=batch_size, shuffle=False)

    if resume_from is not None and resume_from.is_file():
        ckpt = torch.load(str(resume_from), map_location="cpu", weights_only=False)
        model = build_galcon_transformer(**{k: ckpt["model_config"][k] for k in ("d_model", "n_heads", "n_layers", "dim_ff")})
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = build_galcon_transformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dim_ff=dim_ff)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    best_pol = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for pf, tf, ef, le, om, pm, tc, fr, vt in tr_loader:
            k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
            pf_aug = torch.from_numpy(apply_galcon_d4(pf.numpy(), k_rot=k_rot, flip_x=flip_x))
            opt.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, pf_aug, tf, ef, le, om, pm, tc, fr, vt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        vm = []
        with torch.no_grad():
            for pf, tf, ef, le, om, pm, tc, fr, vt in val_loader:
                _, m = _batch_loss(model, pf, tf, ef, le, om, pm, tc, fr, vt)
                vm.append(m)
        v_pol = float(np.mean([x["pol"] for x in vm]))
        if v_pol < best_pol:
            best_pol = v_pol
            torch.save({"model_config": model.config, "model_state_dict": model.state_dict(), "val_pol": v_pol}, out_path)
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(
                f"  ep {ep:2d}/{epochs} ({time.time()-t0:4.1f}s) | val_pol={v_pol:.3f} "
                f"val_all={100*np.mean([x['acc_all'] for x in vm]):5.1f}% "
                f"val_tgt={100*np.mean([x['acc_tgt'] for x in vm]):5.1f}% "
                f"val_vsign={100*np.mean([x['vsign'] for x in vm]):5.1f}%",
                flush=True,
            )
    return out_path


def rl_selfplay_ppo(
    ckpt_path: Path,
    out_path: Path,
    *,
    iters: int = 4,
    episodes_per_iter: int = 12,
    lr: float = 1.2e-4,
    kl_coef: float = 0.05,
    seed: int = 5000,
) -> Path:
    """v1 On-policy Self-Play PPO + fixed Anchor-KL RL (kept for v1 baseline comparison)."""
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    policy = build_galcon_transformer(**{k: cfg[k] for k in ("d_model", "n_heads", "n_layers", "dim_ff")})
    policy.load_state_dict(ckpt["model_state_dict"])
    anchor = copy.deepcopy(policy)
    anchor.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    tactical = GalconTacticalBot()
    t0 = time.time()

    for it in range(1, iters + 1):
        policy.eval()
        buf = {k: [] for k in ("pf", "tf", "ef", "le", "om", "pm", "acts", "old_lp", "adv", "ret")}
        wins = 0
        for ep in range(episodes_per_iter):
            n_pl, w, h = _sample_map_config(rng)
            sim = GalconSim.create(seed=seed + it * 100 + ep, num_planets=n_pl, width=w, height=h, max_ticks=200)
            ours_pid = 1 if (ep % 2 == 0) else 2
            opp_pid = 2 if ours_pid == 1 else 1
            use_selfplay = (ep % 2 == 1)

            ep_steps = []
            while not sim.done:
                if sim.tick % 2 == 0:
                    enc = extract_galcon_features(sim, ours_pid, feature_version=1)
                    pad = pad_galcon_state(enc)
                    n = len(enc.planet_ids)
                    with torch.no_grad():
                        logits, fracs, val = policy(
                            torch.from_numpy(enc.planet_feats[None]),
                            torch.from_numpy(enc.timeline_feats[None]),
                            torch.from_numpy(enc.edge_feats[None]),
                            torch.ones((1, n), dtype=torch.bool),
                        )
                        le_t = torch.from_numpy(enc.legal_edges[None])
                        mask = torch.cat([torch.ones((1, n, 1), dtype=torch.bool), le_t], dim=-1)
                        logits = logits.masked_fill(~mask, -1e9)
                        dist = torch.distributions.Categorical(logits=logits[0])
                        act_short = dist.sample()
                        om_t = torch.from_numpy(enc.owned_mask)
                        lp = (dist.log_prob(act_short) * om_t.float()).sum().item()
                        fracs_np = fracs[0].numpy()
                        v_pred = float(val[0].item())

                    act_pad = np.zeros((GALCON_MAX_PLANETS,), dtype=np.int64)
                    act_pad[:n] = act_short.numpy()
                    o_ours: List[Order] = []
                    for i in range(n):
                        if not enc.owned_mask[i]:
                            continue
                        a = int(act_pad[i])
                        if a > 0:
                            j = a - 1
                            if j < n and enc.legal_edges[i, j]:
                                f = float(fracs_np[i, j])
                                pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
                                o_ours.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                    ep_steps.append((pad, act_pad, lp, v_pred))

                    if use_selfplay:
                        o_enc = extract_galcon_features(sim, opp_pid, feature_version=1)
                        on = len(o_enc.planet_ids)
                        with torch.no_grad():
                            ol, of, _ = anchor(
                                torch.from_numpy(o_enc.planet_feats[None]),
                                torch.from_numpy(o_enc.timeline_feats[None]),
                                torch.from_numpy(o_enc.edge_feats[None]),
                                torch.ones((1, on), dtype=torch.bool),
                            )
                            ole = torch.from_numpy(o_enc.legal_edges[None])
                            omk = torch.cat([torch.ones((1, on, 1), dtype=torch.bool), ole], dim=-1)
                            ol = ol.masked_fill(~omk, -1e9)
                            op_np = F.softmax(ol[0], dim=-1).numpy()
                            of_np = of[0].numpy()
                        o_opp: List[Order] = []
                        for i in range(on):
                            if o_enc.owned_mask[i] and (1.0 - op_np[i, 0]) >= 0.50:
                                j = int(np.argmax(op_np[i, 1:]))
                                if o_enc.legal_edges[i, j]:
                                    f = float(of_np[i, j])
                                    pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
                                    o_opp.append(SendOrder(o_enc.planet_ids[i], o_enc.planet_ids[j], pct))
                    else:
                        o_opp = tactical.act(sim, opp_pid)
                    sim.step({ours_pid: o_ours, opp_pid: o_opp})
                else:
                    sim.step({})

            win = sim.winner()
            rew = 1.0 if win == ours_pid else (0.0 if win == 0 else -1.0)
            if rew > 0:
                wins += 1
            if not ep_steps:
                continue
            T = len(ep_steps)
            for idx_t, (pad, acts_np, old_lp, v_pred) in enumerate(ep_steps):
                disc = 0.99 ** (T - 1 - idx_t)
                ret = rew * disc
                buf["pf"].append(pad["planet_feats"])
                buf["tf"].append(pad["timeline_feats"])
                buf["ef"].append(pad["edge_feats"])
                buf["le"].append(pad["legal_edges"])
                buf["om"].append(pad["owned_mask"])
                buf["pm"].append(pad["pad_mask"])
                buf["acts"].append(acts_np)
                buf["old_lp"].append(old_lp)
                buf["ret"].append(ret)
                buf["adv"].append(ret - v_pred)

        policy.train()
        adv_arr = np.array(buf["adv"], dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)
        ds = TensorDataset(
            torch.from_numpy(np.stack(buf["pf"])),
            torch.from_numpy(np.stack(buf["tf"])),
            torch.from_numpy(np.stack(buf["ef"])),
            torch.from_numpy(np.stack(buf["le"])),
            torch.from_numpy(np.stack(buf["om"])),
            torch.from_numpy(np.stack(buf["pm"])),
            torch.from_numpy(np.stack(buf["acts"])),
            torch.from_numpy(np.array(buf["old_lp"], dtype=np.float32)),
            torch.from_numpy(adv_arr),
            torch.from_numpy(np.array(buf["ret"], dtype=np.float32)),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for _ in range(2):
            for pf, tf, ef, le, om, pm, acts, old_lp, adv, ret in loader:
                pf, tf, ef, le, om, pm, acts = _trim_batch_n(pf, tf, ef, le, om, pm, acts)
                k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
                pf_a = torch.from_numpy(apply_galcon_d4(pf.numpy(), k_rot=k_rot, flip_x=flip_x))
                logits, _, val = policy(pf_a, tf, ef, pm)
                mask = torch.cat([torch.ones((logits.size(0), logits.size(1), 1), dtype=torch.bool), le], dim=-1)
                logits = logits.masked_fill(~mask, -1e9)
                logp = F.log_softmax(logits, dim=-1)
                act_lp = logp.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
                new_lp = (act_lp * om.float()).sum(dim=-1)
                ratio = torch.exp((new_lp - old_lp).clamp(-10.0, 10.0))
                surr = torch.min(ratio * adv, torch.clamp(ratio, 0.85, 1.15) * adv).mean()

                with torch.no_grad():
                    a_logits, _, _ = anchor(pf_a, tf, ef, pm)
                    a_logits = a_logits.masked_fill(~mask, -1e9)
                    a_logp = F.log_softmax(a_logits, dim=-1)
                    a_p = a_logp.exp()
                kl = (a_p * (a_logp - logp)).sum(dim=-1)
                kl_mean = (kl * om.float()).sum() / om.float().sum().clamp_min(1.0)
                v_loss = F.mse_loss(val, ret)
                loss = -surr + 0.35 * v_loss + kl_coef * kl_mean
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
                opt.step()

        print(f"[Galcon v1 PPO] iter {it}/{iters} ({time.time()-t0:.1f}s) | eps_wins={wins}/{episodes_per_iter} | states={len(buf['ret'])}", flush=True)

    torch.save({"model_config": policy.config, "model_state_dict": policy.state_dict()}, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Gen 2 Teacher-Free Self-Play RL Flywheel
# ---------------------------------------------------------------------------

def _make_sparring_bot(rng: random.Random, kind: Optional[str] = None):
    if kind is None:
        r = rng.random()
        if r < 0.35:
            kind = "rusher"
        elif r < 0.65:
            kind = "tactical"
        elif r < 0.82:
            kind = "classic"
        else:
            kind = "macro"
    if kind == "rusher":
        style = rng.choice(["all_in", "blitz", "adaptive"])
        return GalconRusherBot(style=style, aggression=rng.uniform(0.90, 1.15))
    if kind == "tactical":
        return GalconTacticalBot(aggression=rng.uniform(0.88, 1.15))
    if kind == "classic":
        return GalconClassicBot(send_threshold=rng.uniform(12.0, 18.0))
    return GalconMacroBot(expand_bias=rng.uniform(1.15, 1.50))


def collect_gen2_warmstart_dataset(
    *,
    num_games: int = 90,
    seed: int = 2000,
    sample_stride: int = 2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Collect multi-style warm-start games (Tactical / Rusher / Classic / Macro) with t+10 & /REDIR labels.

    Records ONLY the winning side's decisions from diverse matches (winner-filtered multi-style BC)
    so the warm-start policy immediately learns rush defense, blitz punish, and macro expansion
    without inheriting any single heuristic's blind spots.
    """
    rng = random.Random(seed)
    pf_l, tf_l, ef_l, le_l, om_l, pm_l = [], [], [], [], [], []
    soft_l, fr_l, vt_l, aux_o_l, aux_s_l = [], [], [], [], []
    r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk = [], [], [], [], [], [], [], []
    t0 = time.time()

    matchup_pool = [
        ("tactical", "rusher"),
        ("rusher", "tactical"),
        ("tactical", "classic"),
        ("rusher", "classic"),
        ("tactical", "macro"),
        ("rusher", "macro"),
        ("tactical", "tactical"),
        ("rusher", "rusher"),
    ]

    for g in range(num_games):
        n_pl, w, h = _sample_map_config(rng)
        sim = GalconSim.create(seed=seed + g, num_planets=n_pl, width=w, height=h, max_ticks=220)
        k1, k2 = matchup_pool[g % len(matchup_pool)]
        b1 = _make_sparring_bot(rng, k1)
        b2 = _make_sparring_bot(rng, k2)

        # Track per-player trajectories for winner filtering + t+10 world-model lookup
        traj_by_p: Dict[int, List[Tuple]] = {1: [], 2: []}
        snaps_by_p: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {1: [], 2: []}

        while not sim.done:
            if sim.tick % 2 == 0:
                o1 = b1.act(sim, 1)
                o2 = b2.act(sim, 2)
                for pid, ords in ((1, o1), (2, o2)):
                    enc = extract_galcon_features(sim, pid, feature_version=2)
                    so, ss = _snapshot_planet_state(sim, enc.planet_ids, pid)
                    snaps_by_p[pid].append((sim.tick, so, ss))
                    if np.any(enc.owned_mask) and ((sim.tick // 2) % sample_stride == 0 or any(isinstance(x, SendOrder) for x in ords)):
                        pad = pad_galcon_state(enc)
                        tc, fr = _orders_to_targets(enc, ords)
                        # Build one-hot soft target (GALCON_MAX_PLANETS, GALCON_MAX_PLANETS + 1)
                        soft = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS + 1), dtype=np.float32)
                        soft[np.arange(GALCON_MAX_PLANETS), tc] = 1.0

                        redir_pack = None
                        if sim.fleets and (sim.tick % 4 == 0):
                            renc = extract_galcon_redir_features(sim, enc)
                            if renc.fleet_ids:
                                _, rlabels = compute_smart_redirects(sim, enc, verify_rollout=True)
                                redir_pack = _pad_redir_sample(enc, renc, rlabels)
                        traj_by_p[pid].append((sim.tick, pad, soft, fr, redir_pack))
                sim.step({1: o1, 2: o2})
            else:
                sim.step({})

        win = sim.winner()
        # Final snapshots
        for pid in (1, 2):
            enc_f = extract_galcon_features(sim, pid, feature_version=2)
            so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, pid)
            snaps_by_p[pid].append((sim.tick, so, ss))

        # Record winner's policy + balanced value/aux from BOTH sides (masking owned_mask on losing side!)
        for pid in (1, 2):
            snaps = snaps_by_p[pid]
            v_game = 1.0 if win == pid else (0.0 if win == 0 else -1.0)
            is_winner_side = (win == pid) or (win == 0)
            for tick_rec, pad, soft, fr, redir_pack in traj_by_p[pid]:
                target_tick = tick_rec + 10
                fut_o, fut_s = snaps[-1][1], snaps[-1][2]
                for st, so, ss in snaps:
                    if st >= target_tick:
                        fut_o, fut_s = so, ss
                        break
                pf_l.append(pad["planet_feats"])
                tf_l.append(pad["timeline_feats"])
                ef_l.append(pad["edge_feats"])
                le_l.append(pad["legal_edges"])
                # Policy loss only trains on winning/draw side; value & aux t+10 train on both sides!
                om_l.append(pad["owned_mask"] if is_winner_side else np.zeros_like(pad["owned_mask"]))
                pm_l.append(pad["pad_mask"])
                soft_l.append(soft)
                fr_l.append(fr)
                vt_l.append(v_game)
                aux_o_l.append(fut_o)
                aux_s_l.append(fut_s)
                if is_winner_side and redir_pack is not None:
                    r_pf.append(pad["planet_feats"])
                    r_tf.append(pad["timeline_feats"])
                    r_ef.append(pad["edge_feats"])
                    r_pm.append(pad["pad_mask"])
                    r_cur.append(redir_pack[0])
                    r_feat.append(redir_pack[1])
                    r_tar.append(redir_pack[2])
                    r_msk.append(redir_pack[3])

    data = {
        "planet_feats": np.stack(pf_l).astype(np.float32),
        "timeline_feats": np.stack(tf_l).astype(np.float32),
        "edge_feats": np.stack(ef_l).astype(np.float32),
        "legal_edges": np.stack(le_l),
        "owned_mask": np.stack(om_l),
        "pad_mask": np.stack(pm_l),
        "soft_target": np.stack(soft_l).astype(np.float32),
        "target_frac": np.stack(fr_l).astype(np.float32),
        "val_targets": np.array(vt_l, dtype=np.float32),
        "aux_owner": np.stack(aux_o_l).astype(np.int64),
        "aux_ships": np.stack(aux_s_l).astype(np.float32),
    }
    redir_data = _pack_redir_dict(r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk)
    print(
        f"[Galcon Gen2 WarmStart] Collected {len(vt_l)} balanced states (winner-filtered policy) + "
        f"{len(r_pf)} redir states across {num_games} games in {time.time()-t0:.1f}s",
        flush=True,
    )
    return data, redir_data


def _pack_redir_dict(r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk) -> Dict[str, np.ndarray]:
    if not r_pf:
        return {}
    return {
        "planet_feats": np.stack(r_pf).astype(np.float32),
        "timeline_feats": np.stack(r_tf).astype(np.float32),
        "edge_feats": np.stack(r_ef).astype(np.float32),
        "pad_mask": np.stack(r_pm),
        "cur_dst_idx": np.stack(r_cur).astype(np.int64),
        "redir_feats": np.stack(r_feat).astype(np.float32),
        "redir_target": np.stack(r_tar).astype(np.int64),
        "redir_mask": np.stack(r_msk),
    }


def collect_alphazero_search_dataset(
    ckpt_path: Path,
    *,
    num_games: int = 72,
    seed: int = 4000,
    temperature: float = 0.06,
    rollout_horizon: int = 12,
    use_d4_tta: bool = False,
    lua_opp_path: Optional[Path] = None,
    gen3_maps: bool = False,
    filter_losing_policy: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """AlphaZero Reactive-Search Self-Play Distillation (Expert Iteration — zero external teacher!).

    Collects trajectories played by our own ``GalconSearchTransformerAgent`` (2-ply reactive
    neural search) against itself, our heuristic suite, and optionally an external Lua sparring
    opponent (never imitating the opponent — only distilling our own search agent's winning/drawing
    plans while training the value & ``t+10`` world-model heads on all outcomes).
    """
    rng = random.Random(seed)
    lua_opp = None
    if lua_opp_path is not None and lua_opp_path.is_file():
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp = GalconLuaBotBridge(lua_opp_path)

    search_ag = GalconSearchTransformerAgent(
        ckpt_path,
        use_d4_tta=use_d4_tta,
        rollout_horizon=rollout_horizon,
        time_budget_sec=0.14 if gen3_maps else 0.10,
        neural_value_weight=0.85,
        reactive_mid_ply=True,
        prior_bonus=0.025,
    )
    pf_l, tf_l, ef_l, le_l, om_l, pm_l = [], [], [], [], [], []
    soft_l, fr_l, vt_l, aux_o_l, aux_s_l = [], [], [], [], []
    r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk = [], [], [], [], [], [], [], []
    wins_vs_bots = 0
    draws_vs_bots = 0
    games_vs_bots = 0
    wins_vs_boss = 0
    draws_vs_boss = 0
    games_vs_boss = 0
    t0 = time.time()

    opp_cycle = ["rusher", "tactical", "rusher", "classic", "tactical", "macro"]
    bot_game_idx = 0
    boss_game_idx = 0

    try:
        for g in range(num_games):
            if gen3_maps:
                n_pl, w, h, sym = _sample_gen3_map_config(rng)
                max_t = 220 if n_pl >= 22 else 200
            else:
                n_pl, w, h = _sample_map_config(rng)
                sym = True
                max_t = 180
            sim = GalconSim.create(seed=seed + g, num_planets=n_pl, width=w, height=h, max_ticks=max_t, symmetric=sym)

            is_selfplay = False
            is_boss = False
            if lua_opp is not None:
                # 50% vs Lua boss, 25% Search self-play, 25% vs heuristic suite
                slot = g % 4
                if slot in (0, 1):
                    is_boss = True
                    boss_game_idx += 1
                    opp_bot = lua_opp
                    our_pid = 1 if (boss_game_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 2:
                    is_selfplay = True
                    opp_bot = None
                    our_pid = opp_pid = 0
                else:
                    opp_kind = opp_cycle[bot_game_idx % len(opp_cycle)]
                    bot_game_idx += 1
                    opp_bot = _make_sparring_bot(rng, opp_kind)
                    our_pid = 1 if (bot_game_idx % 2 == 0) else 2
                    opp_pid = 2 if our_pid == 1 else 1
            else:
                is_selfplay = (g % 5) < 2  # Exact 40% Search vs Search, 60% vs diverse bots
                if not is_selfplay:
                    opp_kind = opp_cycle[bot_game_idx % len(opp_cycle)]
                    bot_game_idx += 1
                    opp_bot = _make_sparring_bot(rng, opp_kind)
                    our_pid = 1 if (bot_game_idx % 2 == 0) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                else:
                    opp_bot = None
                    our_pid = opp_pid = 0

            traj_by_p: Dict[int, List[Tuple]] = {1: [], 2: []}
            snaps_by_p: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {1: [], 2: []}

            while not sim.done:
                if sim.tick % 2 == 0:
                    orders_map: Dict[int, List[Order]] = {}
                    search_pids = (1, 2) if is_selfplay else (our_pid,)
                    for pid in search_pids:
                        enc = extract_galcon_features(sim, pid, feature_version=2)
                        so, ss = _snapshot_planet_state(sim, enc.planet_ids, pid)
                        snaps_by_p[pid].append((sim.tick, so, ss))
                        best_plan, cands, scores, best_sc = search_ag.act_with_search_info(sim, pid)
                        orders_map[pid] = best_plan

                        if np.any(enc.owned_mask):
                            pad = pad_galcon_state(enc)
                            sc_arr = np.array(scores, dtype=np.float64) if scores else np.array([0.0])
                            logits_c = (sc_arr - sc_arr.max()) / max(1e-3, temperature)
                            w_c = np.exp(np.clip(logits_c, -20.0, 0.0))
                            w_c = w_c / max(1e-8, w_c.sum())

                            soft = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS + 1), dtype=np.float32)
                            if cands:
                                for wk, plan_k in zip(w_c, cands):
                                    if wk < 1e-4:
                                        continue
                                    tc_k, _ = _orders_to_targets(enc, plan_k)
                                    soft[np.arange(GALCON_MAX_PLANETS), tc_k] += float(wk)
                            else:
                                tc_b, _ = _orders_to_targets(enc, best_plan)
                                soft[np.arange(GALCON_MAX_PLANETS), tc_b] = 1.0
                            row_s = soft.sum(axis=-1, keepdims=True)
                            soft = np.where(row_s > 0, soft / np.maximum(1e-8, row_s), soft)
                            _, fr_best = _orders_to_targets(enc, best_plan)

                            redir_pack = None
                            if sim.fleets and (sim.tick % 6 == 0):
                                renc = extract_galcon_redir_features(sim, enc)
                                if renc.fleet_ids:
                                    _, rlabels = compute_smart_redirects(sim, enc, verify_rollout=True)
                                    redir_pack = _pad_redir_sample(enc, renc, rlabels)
                            traj_by_p[pid].append((sim.tick, pad, soft, fr_best, float(best_sc), redir_pack))

                    if not is_selfplay and opp_bot is not None:
                        orders_map[opp_pid] = opp_bot.act(sim, opp_pid)
                    sim.step(orders_map)
                else:
                    sim.step({})

            win = sim.winner()
            if is_boss:
                games_vs_boss += 1
                if win == our_pid:
                    wins_vs_boss += 1
                elif win == 0:
                    draws_vs_boss += 1
            elif not is_selfplay:
                games_vs_bots += 1
                if win == our_pid:
                    wins_vs_bots += 1
                elif win == 0:
                    draws_vs_bots += 1

            rec_pids = (1, 2) if is_selfplay else (our_pid,)
            for pid in rec_pids:
                enc_f = extract_galcon_features(sim, pid, feature_version=2)
                so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, pid)
                snaps_by_p[pid].append((sim.tick, so, ss))
                snaps = snaps_by_p[pid]
                z = 1.0 if win == pid else (0.0 if win == 0 else -1.0)
                keep_policy = (not filter_losing_policy) or (z >= 0.0)
                for tick_rec, pad, soft, fr_best, best_sc, redir_pack in traj_by_p[pid]:
                    target_tick = tick_rec + 10
                    fut_o, fut_s = snaps[-1][1], snaps[-1][2]
                    for st, so, ss in snaps:
                        if st >= target_tick:
                            fut_o, fut_s = so, ss
                            break
                    pf_l.append(pad["planet_feats"])
                    tf_l.append(pad["timeline_feats"])
                    ef_l.append(pad["edge_feats"])
                    le_l.append(pad["legal_edges"])
                    om_l.append(pad["owned_mask"] if keep_policy else np.zeros_like(pad["owned_mask"]))
                    pm_l.append(pad["pad_mask"])
                    soft_l.append(soft)
                    fr_l.append(fr_best)
                    vt_l.append(float(np.clip(0.70 * z + 0.30 * best_sc, -1.0, 1.0)))
                    aux_o_l.append(fut_o)
                    aux_s_l.append(fut_s)
                    if redir_pack is not None:
                        r_pf.append(pad["planet_feats"])
                        r_tf.append(pad["timeline_feats"])
                        r_ef.append(pad["edge_feats"])
                        r_pm.append(pad["pad_mask"])
                        r_cur.append(redir_pack[0])
                        r_feat.append(redir_pack[1])
                        r_tar.append(redir_pack[2])
                        r_msk.append(redir_pack[3])
    finally:
        if lua_opp is not None:
            lua_opp.close()

    data = {
        "planet_feats": np.stack(pf_l).astype(np.float32),
        "timeline_feats": np.stack(tf_l).astype(np.float32),
        "edge_feats": np.stack(ef_l).astype(np.float32),
        "legal_edges": np.stack(le_l),
        "owned_mask": np.stack(om_l),
        "pad_mask": np.stack(pm_l),
        "soft_target": np.stack(soft_l).astype(np.float32),
        "target_frac": np.stack(fr_l).astype(np.float32),
        "val_targets": np.array(vt_l, dtype=np.float32),
        "aux_owner": np.stack(aux_o_l).astype(np.int64),
        "aux_ships": np.stack(aux_s_l).astype(np.float32),
    }
    redir_data = _pack_redir_dict(r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk)
    wr = 100.0 * wins_vs_bots / max(1, games_vs_bots)
    boss_msg = (
        f", boss={wins_vs_boss}-{games_vs_boss - wins_vs_boss - draws_vs_boss}-{draws_vs_boss}"
        if games_vs_boss > 0
        else ""
    )
    print(
        f"[Galcon AlphaZero Reactive Search] Collected {len(vt_l)} states + "
        f"{len(r_pf)} redir states across {num_games} games "
        f"(bots={wins_vs_bots}/{games_vs_bots}={wr:.1f}%{boss_msg}) "
        f"in {time.time()-t0:.1f}s",
        flush=True,
    )
    return data, redir_data


def _gen2_supervised_loss(
    model: GalconPlanetTransformer,
    pf,
    tf,
    ef,
    le,
    om,
    pm,
    soft_tar,
    tfrac,
    vtar,
    aux_own,
    aux_shp,
    launch_weight: float = 1.35,
):
    pf, tf, ef, le, om, pm, soft_tar, tfrac, aux_own, aux_shp = _trim_batch_n(
        pf, tf, ef, le, om, pm, soft_tar, tfrac, aux_own, aux_shp
    )
    logits, fracs, val, aux_pred = model(pf, tf, ef, pm, return_aux=True)
    B, N, _ = logits.shape
    mask = torch.cat([torch.ones((B, N, 1), dtype=torch.bool, device=logits.device), le], dim=-1)
    logits = logits.masked_fill(~mask, -1e9)
    valid = om & le.any(dim=-1)
    if valid.any():
        fl = logits[valid]
        fs = soft_tar[valid]
        fs = fs / fs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        logp = F.log_softmax(fl, dim=-1)
        ce = -(fs * logp).sum(dim=-1)
        ft = fs.argmax(dim=-1)
        w = torch.where(ft > 0, launch_weight, 1.0)
        pol_loss = (ce * w).sum() / w.sum().clamp_min(1.0)
        preds = fl.argmax(dim=-1)
        acc_all = float((preds == ft).float().mean().item())
        lm = ft > 0
        acc_tgt = float((1 + fl[lm, 1:].argmax(dim=-1) == ft[lm]).float().mean().item()) if lm.any() else 0.0
    else:
        pol_loss = logits.sum() * 0.0
        acc_all = acc_tgt = 0.0

    tcls_hard = soft_tar.argmax(dim=-1)
    ldec = valid & (tcls_hard > 0) & (tfrac > 0.01)
    if ldec.any():
        dst = (tcls_hard[ldec] - 1).clamp(0, N - 1)
        pf_pred = fracs[ldec].gather(1, dst.unsqueeze(1)).squeeze(1)
        frac_loss = F.smooth_l1_loss(pf_pred, tfrac[ldec])
    else:
        frac_loss = fracs.sum() * 0.0

    v_loss = F.mse_loss(val, vtar)
    vsign = float(((val >= 0) == (vtar >= 0)).float().mean().item())

    if pm.any():
        aux_own_loss = F.cross_entropy(aux_pred[pm, :3], aux_own[pm])
        aux_shp_loss = F.smooth_l1_loss(aux_pred[pm, 3], aux_shp[pm])
        aux_loss = aux_own_loss + 0.5 * aux_shp_loss
        aux_acc = float((aux_pred[pm, :3].argmax(dim=-1) == aux_own[pm]).float().mean().item())
    else:
        aux_loss = aux_pred.sum() * 0.0
        aux_acc = 0.0

    total = pol_loss + 0.55 * frac_loss + 0.35 * v_loss + 0.22 * aux_loss
    return total, {
        "pol": float(pol_loss.item()),
        "acc_all": acc_all,
        "acc_tgt": acc_tgt,
        "vsign": vsign,
        "aux_acc": aux_acc,
    }


def _redir_batch_loss(
    model: GalconPlanetTransformer,
    pf,
    tf,
    ef,
    pm,
    cur_idx,
    rfeats,
    rtar,
    rmask,
    redir_pos_weight: float = 3.5,
):
    max_n = max(2, int(pm.sum(dim=1).max().item()))
    if max_n < pf.shape[1]:
        pf = pf[:, :max_n]
        tf = tf[:, :max_n]
        ef = ef[:, :max_n, :max_n]
        pm = pm[:, :max_n]
        rfeats = rfeats[:, :, :max_n]
    h = model.encode_planets(pf, tf, ef, pm)
    rlogits = model.predict_redir_logits(h, cur_idx, rfeats, pm)
    if not rmask.any():
        return rlogits.sum() * 0.0, 0.0, 0.0
    flat_l = rlogits[rmask]
    flat_t = rtar[rmask].clamp(0, max_n - 1)
    flat_c = cur_idx[rmask].clamp(0, max_n - 1)
    ce = F.cross_entropy(flat_l, flat_t, reduction="none")
    is_redir = flat_t != flat_c
    w = torch.where(is_redir, redir_pos_weight, 1.0)
    loss = (ce * w).sum() / w.sum().clamp_min(1.0)
    preds = flat_l.argmax(dim=-1)
    acc = float((preds == flat_t).float().mean().item())
    recall = float((preds[is_redir] == flat_t[is_redir]).float().mean().item()) if is_redir.any() else 1.0
    return loss, acc, recall


def train_gen2_supervised(
    data: Dict[str, np.ndarray],
    redir_data: Dict[str, np.ndarray],
    *,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1.0e-3,
    d_model: int = 160,
    n_heads: int = 4,
    n_layers: int = 5,
    dim_ff: int = 384,
    out_path: Path,
    resume_from: Path | None = None,
    seed: int = 42,
    label: str = "Gen2-Sup",
) -> Path:
    """Train Gen 2 ``GalconPlanetTransformer`` jointly on policy, fraction, value, aux ``t+10``, and ``/REDIR``."""
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = len(data["val_targets"])
    perm = np.random.permutation(n)
    n_val = max(1, int(0.10 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def make_ds(idx):
        return TensorDataset(
            torch.from_numpy(data["planet_feats"][idx]),
            torch.from_numpy(data["timeline_feats"][idx]),
            torch.from_numpy(data["edge_feats"][idx]),
            torch.from_numpy(data["legal_edges"][idx]),
            torch.from_numpy(data["owned_mask"][idx]),
            torch.from_numpy(data["pad_mask"][idx]),
            torch.from_numpy(data["soft_target"][idx]),
            torch.from_numpy(data["target_frac"][idx]),
            torch.from_numpy(data["val_targets"][idx]),
            torch.from_numpy(data["aux_owner"][idx]),
            torch.from_numpy(data["aux_ships"][idx]),
        )

    tr_loader = DataLoader(make_ds(tr_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_ds(val_idx), batch_size=batch_size, shuffle=False)

    redir_loader = None
    if redir_data and len(redir_data.get("planet_feats", [])) >= 16:
        rds = TensorDataset(
            torch.from_numpy(redir_data["planet_feats"]),
            torch.from_numpy(redir_data["timeline_feats"]),
            torch.from_numpy(redir_data["edge_feats"]),
            torch.from_numpy(redir_data["pad_mask"]),
            torch.from_numpy(redir_data["cur_dst_idx"]),
            torch.from_numpy(redir_data["redir_feats"]),
            torch.from_numpy(redir_data["redir_target"]),
            torch.from_numpy(redir_data["redir_mask"]),
        )
        redir_loader = DataLoader(rds, batch_size=32, shuffle=True)

    if resume_from is not None and resume_from.is_file():
        ckpt = torch.load(str(resume_from), map_location="cpu", weights_only=False)
        cfg = ckpt["model_config"]
        model = build_galcon_gen2_transformer(
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            dim_ff=int(cfg["dim_ff"]),
            feature_version=int(cfg.get("feature_version", 2)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = build_galcon_gen2_transformer(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dim_ff=dim_ff, feature_version=2
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)
    best_pol = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        r_iter = iter(redir_loader) if redir_loader is not None else None
        for b_idx, (pf, tf, ef, le, om, pm, soft, fr, vt, ao, ashp) in enumerate(tr_loader):
            k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
            pf_aug = torch.from_numpy(apply_galcon_d4(pf.numpy(), k_rot=k_rot, flip_x=flip_x))
            opt.zero_grad(set_to_none=True)
            loss, _ = _gen2_supervised_loss(model, pf_aug, tf, ef, le, om, pm, soft, fr, vt, ao, ashp)
            if r_iter is not None and (b_idx % 4 == 0):
                try:
                    r_batch = next(r_iter)
                except StopIteration:
                    r_iter = iter(redir_loader)
                    r_batch = next(r_iter)
                rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk = r_batch
                kr, fx = random.randrange(4), bool(random.randrange(2))
                rpf_a = torch.from_numpy(apply_galcon_d4(rpf.numpy(), k_rot=kr, flip_x=fx))
                rloss, _, _ = _redir_batch_loss(model, rpf_a, rtf, ref, rpm, rcur, rfeat, rtar, rmsk)
                loss = loss + 0.35 * rloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        vm = []
        with torch.no_grad():
            for pf, tf, ef, le, om, pm, soft, fr, vt, ao, ashp in val_loader:
                _, m = _gen2_supervised_loss(model, pf, tf, ef, le, om, pm, soft, fr, vt, ao, ashp)
                vm.append(m)
        v_pol = float(np.mean([x["pol"] for x in vm]))
        if v_pol < best_pol:
            best_pol = v_pol
            torch.save(
                {
                    "model_config": model.config,
                    "model_state_dict": model.state_dict(),
                    "has_trained_redir": redir_loader is not None,
                    "val_pol": v_pol,
                },
                out_path,
            )
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(
                f"  [{label}] ep {ep:2d}/{epochs} ({time.time()-t0:4.1f}s) | val_pol={v_pol:.3f} "
                f"val_all={100*np.mean([x['acc_all'] for x in vm]):5.1f}% "
                f"val_tgt={100*np.mean([x['acc_tgt'] for x in vm]):5.1f}% "
                f"val_aux={100*np.mean([x['aux_acc'] for x in vm]):5.1f}% "
                f"val_vsign={100*np.mean([x['vsign'] for x in vm]):5.1f}%",
                flush=True,
            )
    return out_path


def _fast_neural_act(model: GalconPlanetTransformer, sim: GalconSim, player: int, *, launch_threshold: float = 0.50) -> List[Order]:
    """Fast 1x forward pass action decoder for historical league snapshot opponents."""
    enc = extract_galcon_features(sim, player, feature_version=2)
    n = len(enc.planet_ids)
    if n == 0 or not np.any(enc.owned_mask):
        return []
    with torch.no_grad():
        pf_t = torch.from_numpy(enc.planet_feats[None])
        tf_t = torch.from_numpy(enc.timeline_feats[None])
        ef_t = torch.from_numpy(enc.edge_feats[None])
        pm_t = torch.ones((1, n), dtype=torch.bool)
        logits, fracs, _, h = model(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
        le_t = torch.from_numpy(enc.legal_edges[None])
        mask = torch.cat([torch.ones((1, n, 1), dtype=torch.bool), le_t], dim=-1)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits[0], dim=-1).numpy()
        fracs_np = fracs[0].numpy()

        orders: List[Order] = []
        for i in range(n):
            if enc.owned_mask[i] and (1.0 - float(probs[i, 0])) >= launch_threshold:
                j = int(np.argmax(probs[i, 1:]))
                if enc.legal_edges[i, j]:
                    f = float(fracs_np[i, j])
                    pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
                    orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))

        # Native /REDIR head for league snapshots
        if sim.allow_redirect and sim.fleets:
            renc = extract_galcon_redir_features(sim, enc)
            if renc.fleet_ids:
                rlogits = model.predict_redir_logits(
                    h,
                    torch.from_numpy(renc.cur_dst_idx[None]),
                    torch.from_numpy(renc.redir_feats[None]),
                    pm_t,
                )
                rprobs = F.softmax(rlogits[0], dim=-1).numpy()
                for fi, fid in enumerate(renc.fleet_ids):
                    c_idx = int(renc.cur_dst_idx[fi])
                    bj = int(np.argmax(rprobs[fi]))
                    if bj != c_idx and float(rprobs[fi, bj]) >= 0.55:
                        orders.append(RedirOrder(fid, enc.planet_ids[bj]))
    return orders


def rl_league_ppo_gen2(
    ckpt_path: Path,
    out_path: Path,
    *,
    redir_replay: Optional[Dict[str, np.ndarray]] = None,
    v1_ckpt_path: Optional[Path] = None,
    iters: int = 26,
    episodes_per_iter: int = 16,
    lr: float = 1.0e-4,
    kl_coef: float = 0.025,
    ema_tau: float = 0.08,
    gamma: float = 0.985,
    gae_lam: float = 0.95,
    frac_std: float = 0.075,
    draw_penalty: float = -0.35,
    eval_every: int = 5,
    seed: int = 6000,
) -> Path:
    """Large-Scale Teacher-Free League PPO + Historical Snapshot Pool + Moving EMA Teacher KL.

    Zero external teacher ceiling:
    - Opponent mix: Historical Snapshot Pool + Rusher + Tactical + Classic + Macro (+ v1 if provided).
    - Moving EMA teacher KL regularizes smoothly without anchoring to a weak BC model.
    - Penalizes stalemate draws (``draw_penalty``) so the bot plays for decisive knockouts.
    - Jointly optimizes discrete planet targets AND continuous send fractions via GAE + potential shaping.
    - Keeps auxiliary ``t+10`` world-model head and native ``/REDIR`` head sharp throughout RL.
    """
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    v1_opp = (
        GalconTransformerAgent(v1_ckpt_path, use_d4_tta=False)
        if (v1_ckpt_path is not None and v1_ckpt_path.is_file())
        else None
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    policy = build_galcon_gen2_transformer(
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        dim_ff=int(cfg["dim_ff"]),
        feature_version=int(cfg.get("feature_version", 2)),
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    has_redir = bool(ckpt.get("has_trained_redir", False))

    # Moving EMA teacher (zero fixed BC ceiling!)
    ema_teacher = copy.deepcopy(policy)
    ema_teacher.eval()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # Historical snapshot pool
    snapshot_pool: List[GalconPlanetTransformer] = [copy.deepcopy(policy).eval()]

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters), eta_min=lr * 0.20)

    redir_loader = None
    if redir_replay and len(redir_replay.get("planet_feats", [])) >= 16:
        rds = TensorDataset(
            torch.from_numpy(redir_replay["planet_feats"]),
            torch.from_numpy(redir_replay["timeline_feats"]),
            torch.from_numpy(redir_replay["edge_feats"]),
            torch.from_numpy(redir_replay["pad_mask"]),
            torch.from_numpy(redir_replay["cur_dst_idx"]),
            torch.from_numpy(redir_replay["redir_feats"]),
            torch.from_numpy(redir_replay["redir_target"]),
            torch.from_numpy(redir_replay["redir_mask"]),
        )
        redir_loader = DataLoader(rds, batch_size=32, shuffle=True)

    best_gauntlet_wr = -1.0
    best_state = copy.deepcopy(policy.state_dict())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_eps = 0

    def quick_gauntlet(model_eval: GalconPlanetTransformer) -> float:
        model_eval.eval()
        opps = [
            GalconTacticalBot(),
            GalconClassicBot(),
            GalconRusherBot(style="all_in"),
            GalconRusherBot(style="adaptive"),
            GalconMacroBot(),
        ]
        wins_g = 0
        tot_g = 0
        for oi, obot in enumerate(opps):
            for k in range(4):
                g_rng = random.Random(19000 + oi * 100 + k)
                n_p, ww, hh = _sample_map_config(g_rng)
                s = GalconSim.create(seed=19000 + oi * 100 + k, num_planets=n_p, width=ww, height=hh, max_ticks=200)
                a_p = 1 if (k % 2 == 0) else 2
                b_p = 2 if a_p == 1 else 1
                while not s.done:
                    if s.tick % 2 == 0:
                        oa = _fast_neural_act(model_eval, s, a_p)
                        ob = obot.act(s, b_p)
                        s.step({a_p: oa, b_p: ob})
                    else:
                        s.step({})
                tot_g += 1
                if s.winner() == a_p:
                    wins_g += 1
        return wins_g / max(1, tot_g)

    torch.set_num_threads(1)
    init_wr = quick_gauntlet(policy)
    best_gauntlet_wr = init_wr
    print(f"[Galcon Gen2 League PPO] Start 20g validation gauntlet WR = {100*init_wr:.1f}% (pool=1)", flush=True)
    t0 = time.time()

    for it in range(1, iters + 1):
        torch.set_num_threads(1)
        policy.eval()
        buf = {
            k: []
            for k in (
                "pf",
                "tf",
                "ef",
                "le",
                "om",
                "pm",
                "acts",
                "f_acts",
                "old_lp",
                "adv",
                "ret",
                "aux_o",
                "aux_s",
            )
        }
        wins = 0
        sp_wins = 0
        sp_eps = 0
        bot_wins = 0
        bot_eps = 0

        for ep in range(episodes_per_iter):
            total_eps += 1
            n_pl, w, h = _sample_map_config(rng)
            sim = GalconSim.create(seed=seed + it * 200 + ep, num_planets=n_pl, width=w, height=h, max_ticks=200)
            ours_pid = 1 if (ep % 2 == 0) else 2
            opp_pid = 2 if ours_pid == 1 else 1

            # Opponent League Selection
            u = rng.random()
            opp_snap: Optional[GalconPlanetTransformer] = None
            opp_bot = None
            if u < 0.35:
                # Sample from historical snapshot pool (bias slightly toward latest half)
                if len(snapshot_pool) >= 2 and rng.random() < 0.60:
                    opp_snap = rng.choice(snapshot_pool[len(snapshot_pool) // 2 :])
                else:
                    opp_snap = rng.choice(snapshot_pool)
            elif u < 0.57:
                opp_bot = _make_sparring_bot(rng, "tactical")
            elif u < 0.77:
                opp_bot = _make_sparring_bot(rng, "rusher")
            elif u < 0.90 and v1_opp is not None:
                opp_bot = v1_opp
            elif u < 0.95:
                opp_bot = _make_sparring_bot(rng, "classic")
            else:
                opp_bot = _make_sparring_bot(rng, "macro")

            ep_steps = []
            snaps: List[Tuple[int, np.ndarray, np.ndarray]] = []
            prev_pot = evaluate_galcon_state(sim, ours_pid)

            while not sim.done:
                if sim.tick % 2 == 0:
                    enc = extract_galcon_features(sim, ours_pid, feature_version=2)
                    n = len(enc.planet_ids)
                    so, ss = _snapshot_planet_state(sim, enc.planet_ids, ours_pid)
                    snaps.append((sim.tick, so, ss))
                    pad = pad_galcon_state(enc)

                    with torch.no_grad():
                        pf_t = torch.from_numpy(enc.planet_feats[None])
                        tf_t = torch.from_numpy(enc.timeline_feats[None])
                        ef_t = torch.from_numpy(enc.edge_feats[None])
                        pm_t = torch.ones((1, n), dtype=torch.bool)
                        logits, fracs, val, h_emb = policy(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
                        le_t = torch.from_numpy(enc.legal_edges[None])
                        mask = torch.cat([torch.ones((1, n, 1), dtype=torch.bool), le_t], dim=-1)
                        logits = logits.masked_fill(~mask, -1e9)
                        dist = torch.distributions.Categorical(logits=logits[0])
                        act_short = dist.sample()
                        om_t = torch.from_numpy(enc.owned_mask)
                        lp_disc = (dist.log_prob(act_short) * om_t.float()).sum()

                        # Sample continuous send fraction around predicted mean for launched planets
                        f_mean_all = fracs[0]  # (n, n)
                        f_act_short = torch.zeros((n,), dtype=torch.float32)
                        lp_frac = torch.tensor(0.0)
                        for i in range(n):
                            if enc.owned_mask[i] and int(act_short[i].item()) > 0:
                                j = int(act_short[i].item()) - 1
                                mu = f_mean_all[i, j].clamp(0.08, 0.98)
                                f_dist = torch.distributions.Normal(mu, frac_std)
                                f_s = f_dist.sample().clamp(0.08, 1.0)
                                f_act_short[i] = f_s
                                lp_frac = lp_frac + f_dist.log_prob(f_s)

                        total_lp = float((lp_disc + 0.25 * lp_frac).item())
                        v_pred = float(val[0].item())

                        # Decode orders + native /REDIR
                        o_ours: List[Order] = []
                        for i in range(n):
                            if not enc.owned_mask[i]:
                                continue
                            a = int(act_short[i].item())
                            if a > 0:
                                j = a - 1
                                if j < n and enc.legal_edges[i, j]:
                                    f_val = float(f_act_short[i].item())
                                    pct = 100 if f_val >= 0.90 else max(5, min(100, int(round(f_val * 100.0))))
                                    o_ours.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))

                        if has_redir and sim.allow_redirect and sim.fleets:
                            renc = extract_galcon_redir_features(sim, enc)
                            if renc.fleet_ids:
                                rlogits = policy.predict_redir_logits(
                                    h_emb,
                                    torch.from_numpy(renc.cur_dst_idx[None]),
                                    torch.from_numpy(renc.redir_feats[None]),
                                    pm_t,
                                )
                                rprobs = F.softmax(rlogits[0], dim=-1).numpy()
                                for fi, fid in enumerate(renc.fleet_ids):
                                    c_idx = int(renc.cur_dst_idx[fi])
                                    bj = int(np.argmax(rprobs[fi]))
                                    if bj != c_idx and float(rprobs[fi, bj]) >= 0.55:
                                        o_ours.append(RedirOrder(fid, enc.planet_ids[bj]))

                    act_pad = np.zeros((GALCON_MAX_PLANETS,), dtype=np.int64)
                    act_pad[:n] = act_short.numpy()
                    f_pad = np.zeros((GALCON_MAX_PLANETS,), dtype=np.float32)
                    f_pad[:n] = f_act_short.numpy()

                    if opp_snap is not None:
                        o_opp = _fast_neural_act(opp_snap, sim, opp_pid)
                    else:
                        o_opp = opp_bot.act(sim, opp_pid)

                    cur_tick = sim.tick
                    sim.step({ours_pid: o_ours, opp_pid: o_opp})
                    if not sim.done:
                        sim.step({})

                    new_pot = evaluate_galcon_state(sim, ours_pid)
                    step_rew = 0.06 * (new_pot - prev_pot)
                    prev_pot = new_pot
                    ep_steps.append((cur_tick, pad, act_pad, f_pad, total_lp, v_pred, step_rew))
                else:
                    sim.step({})

            # Terminal outcome + final snapshot
            enc_f = extract_galcon_features(sim, ours_pid, feature_version=2)
            so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, ours_pid)
            snaps.append((sim.tick, so, ss))

            win = sim.winner()
            speed_bonus = 0.10 * max(0.0, 1.0 - float(sim.tick) / max(1.0, float(sim.max_ticks)))
            term_rew = (1.0 + speed_bonus) if win == ours_pid else (draw_penalty if win == 0 else -1.0)
            if win == ours_pid:
                wins += 1
                if opp_snap is not None:
                    sp_wins += 1
                else:
                    bot_wins += 1
            if opp_snap is not None:
                sp_eps += 1
            else:
                bot_eps += 1

            if not ep_steps:
                continue

            # Exact GAE(gamma, gae_lam)
            T = len(ep_steps)
            rews = [s[6] for s in ep_steps]
            rews[-1] += term_rew
            vals = [s[5] for s in ep_steps] + [0.0]
            advs = np.zeros((T,), dtype=np.float32)
            rets = np.zeros((T,), dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = rews[t] + gamma * vals[t + 1] - vals[t]
                gae = delta + gamma * gae_lam * gae
                advs[t] = gae
                rets[t] = float(np.clip(gae + vals[t], -1.15, 1.15))

            for idx_t, (cur_tick, pad, act_pad, f_pad, old_lp, _, _) in enumerate(ep_steps):
                target_tick = cur_tick + 10
                fut_o, fut_s = snaps[-1][1], snaps[-1][2]
                for st, so_s, ss_s in snaps:
                    if st >= target_tick:
                        fut_o, fut_s = so_s, ss_s
                        break
                buf["pf"].append(pad["planet_feats"])
                buf["tf"].append(pad["timeline_feats"])
                buf["ef"].append(pad["edge_feats"])
                buf["le"].append(pad["legal_edges"])
                buf["om"].append(pad["owned_mask"])
                buf["pm"].append(pad["pad_mask"])
                buf["acts"].append(act_pad)
                buf["f_acts"].append(f_pad)
                buf["old_lp"].append(old_lp)
                buf["ret"].append(rets[idx_t])
                buf["adv"].append(advs[idx_t])
                buf["aux_o"].append(fut_o)
                buf["aux_s"].append(fut_s)

        # PPO update with Moving EMA Teacher KL + Joint Fraction + Aux t+10 + /REDIR replay
        policy.train()
        adv_arr = np.array(buf["adv"], dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)
        ds = TensorDataset(
            torch.from_numpy(np.stack(buf["pf"])),
            torch.from_numpy(np.stack(buf["tf"])),
            torch.from_numpy(np.stack(buf["ef"])),
            torch.from_numpy(np.stack(buf["le"])),
            torch.from_numpy(np.stack(buf["om"])),
            torch.from_numpy(np.stack(buf["pm"])),
            torch.from_numpy(np.stack(buf["acts"])),
            torch.from_numpy(np.stack(buf["f_acts"])),
            torch.from_numpy(np.array(buf["old_lp"], dtype=np.float32)),
            torch.from_numpy(adv_arr),
            torch.from_numpy(np.array(buf["ret"], dtype=np.float32)),
            torch.from_numpy(np.stack(buf["aux_o"])),
            torch.from_numpy(np.stack(buf["aux_s"])),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        ent_coef = 0.008 * (1.0 - 0.65 * (it / max(1, iters)))
        r_iter = iter(redir_loader) if redir_loader is not None else None

        torch.set_num_threads(2)
        for _ in range(2):
            for b_idx, (pf, tf, ef, le, om, pm, acts, f_acts, old_lp, adv, ret, aux_o, aux_s) in enumerate(loader):
                pf, tf, ef, le, om, pm, acts, f_acts, aux_o, aux_s = _trim_batch_n(
                    pf, tf, ef, le, om, pm, acts, f_acts, aux_o, aux_s
                )
                k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
                pf_a = torch.from_numpy(apply_galcon_d4(pf.numpy(), k_rot=k_rot, flip_x=flip_x))

                logits, fracs, val, aux_pred = policy(pf_a, tf, ef, pm, return_aux=True)
                B, N, _ = logits.shape
                mask = torch.cat([torch.ones((B, N, 1), dtype=torch.bool), le], dim=-1)
                logits = logits.masked_fill(~mask, -1e9)
                logp = F.log_softmax(logits, dim=-1)
                probs = logp.exp()
                act_lp = logp.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
                new_lp_disc = (act_lp * om.float()).sum(dim=-1)

                # Continuous send-fraction Gaussian log-prob on launched planets
                launch_m = om & (acts > 0)
                dst_idx = (acts - 1).clamp(0, N - 1)
                chosen_f_mu = fracs.gather(2, dst_idx.unsqueeze(-1)).squeeze(-1).clamp(0.08, 0.98)
                f_dist = torch.distributions.Normal(chosen_f_mu, frac_std)
                new_lp_frac = (f_dist.log_prob(f_acts) * launch_m.float()).sum(dim=-1)
                new_lp = new_lp_disc + 0.25 * new_lp_frac

                ratio = torch.exp((new_lp - old_lp).clamp(-8.0, 8.0))
                surr = torch.min(ratio * adv, torch.clamp(ratio, 0.85, 1.15) * adv).mean()

                # Entropy bonus
                ent = -(probs * logp).sum(dim=-1)
                ent_mean = (ent * om.float()).sum() / om.float().sum().clamp_min(1.0)

                # Moving EMA Teacher KL
                with torch.no_grad():
                    e_logits, _, _ = ema_teacher(pf_a, tf, ef, pm)
                    e_logits = e_logits.masked_fill(~mask, -1e9)
                    e_logp = F.log_softmax(e_logits, dim=-1)
                    e_p = e_logp.exp()
                kl = (e_p * (e_logp - logp)).sum(dim=-1)
                kl_mean = (kl * om.float()).sum() / om.float().sum().clamp_min(1.0)

                v_loss = F.mse_loss(val, ret)
                aux_loss = F.cross_entropy(aux_pred[pm, :3], aux_o[pm]) + 0.5 * F.smooth_l1_loss(aux_pred[pm, 3], aux_s[pm])

                loss = -surr + 0.40 * v_loss + kl_coef * kl_mean - ent_coef * ent_mean + 0.14 * aux_loss

                # Co-adapt native /REDIR head with verified replay every 4th batch
                if r_iter is not None and (b_idx % 4 == 0):
                    try:
                        rb = next(r_iter)
                    except StopIteration:
                        r_iter = iter(redir_loader)
                        rb = next(r_iter)
                    rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk = rb
                    kr, fx = random.randrange(4), bool(random.randrange(2))
                    rpf_a = torch.from_numpy(apply_galcon_d4(rpf.numpy(), k_rot=kr, flip_x=fx))
                    rloss, _, _ = _redir_batch_loss(policy, rpf_a, rtf, ref, rpm, rcur, rfeat, rtar, rmsk)
                    loss = loss + 0.25 * rloss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
                opt.step()

        sched.step()
        torch.set_num_threads(1)

        # Update Moving EMA Teacher weights
        with torch.no_grad():
            for p_ema, p_cur in zip(ema_teacher.parameters(), policy.parameters()):
                p_ema.data.mul_(1.0 - ema_tau).add_(p_cur.data, alpha=ema_tau)

        # Grow Historical Snapshot Pool every 3 iterations
        if it % 3 == 0:
            snap_copy = copy.deepcopy(policy).eval()
            for p in snap_copy.parameters():
                p.requires_grad_(False)
            snapshot_pool.append(snap_copy)
            if len(snapshot_pool) > 10:
                # Keep snapshot #0 (initial anchor) + 9 most recent
                snapshot_pool = [snapshot_pool[0]] + snapshot_pool[-9:]

        log_msg = (
            f"[Galcon Gen2 League PPO] iter {it:2d}/{iters} (eps={total_eps}, {time.time()-t0:5.1f}s) | "
            f"wins={wins}/{episodes_per_iter} (sp={sp_wins}/{sp_eps}, bots={bot_wins}/{bot_eps}) | "
            f"states={len(buf['ret'])} | pool={len(snapshot_pool)}"
        )
        if it % eval_every == 0 or it == iters:
            g_wr = quick_gauntlet(policy)
            log_msg += f" | val_gauntlet={100*g_wr:.1f}%"
            if g_wr >= best_gauntlet_wr:
                best_gauntlet_wr = g_wr
                best_state = copy.deepcopy(policy.state_dict())
                torch.save(
                    {
                        "model_config": policy.config,
                        "model_state_dict": best_state,
                        "has_trained_redir": has_redir or (redir_loader is not None),
                        "val_gauntlet_wr": best_gauntlet_wr,
                        "ppo_episodes": total_eps,
                    },
                    out_path,
                )
                log_msg += " *BEST*"
        print(log_msg, flush=True)

    # Save final checkpoint too & ensure out_path has best_state
    final_path = out_path.with_name(out_path.stem + "_final.pt")
    torch.save(
        {
            "model_config": policy.config,
            "model_state_dict": policy.state_dict(),
            "has_trained_redir": has_redir or (redir_loader is not None),
            "ppo_episodes": total_eps,
        },
        final_path,
    )
    torch.save(
        {
            "model_config": policy.config,
            "model_state_dict": best_state,
            "has_trained_redir": has_redir or (redir_loader is not None),
            "val_gauntlet_wr": best_gauntlet_wr,
            "ppo_episodes": total_eps,
        },
        out_path,
    )
    return out_path


def _sample_gen3_map_config(rng: random.Random) -> Tuple[int, float, float, bool]:
    """Gen 3 multi-scale map distribution: compact + classic + large asymmetric gbots arenas."""
    u = rng.random()
    if u < 0.25:
        return rng.randint(8, 12), 240.0, 150.0, True
    if u < 0.70:
        return rng.randint(14, 22), 400.0, 240.0, (rng.random() < 0.80)
    # Large gbots arena (22-28 planets, 520x320 to 600x360, 35% asymmetric!)
    w = rng.choice([520.0, 560.0, 600.0])
    h = round(w * 0.60, 1)
    return rng.randint(22, 28), w, h, (rng.random() < 0.65)


def evaluate_large_arena_matchup(
    name_a: str,
    bot_a,
    name_b: str,
    bot_b,
    *,
    num_pairs: int = 8,
    seed: int = 18000,
) -> Tuple[int, int, int, float]:
    """Paired-mirror stress test on large asymmetric 22-28 planet gbots arenas (560x340)."""
    rng = random.Random(seed)
    w = l = d = 0
    ticks = []
    for k in range(num_pairs):
        n_pl = rng.randint(22, 28)
        sym = (k % 2 == 0)  # 50% symmetric, 50% asymmetric (mirrored P1/P2 ensures 100% fairness!)
        for a_pid in (1, 2):
            b_pid = 2 if a_pid == 1 else 1
            sim = GalconSim.create(
                seed=seed + k, num_planets=n_pl, width=560.0, height=340.0, max_ticks=240, symmetric=sym
            )
            while not sim.done:
                if sim.tick % 2 == 0:
                    sim.step({a_pid: bot_a.act(sim, a_pid), b_pid: bot_b.act(sim, b_pid)})
                else:
                    sim.step({})
            win = sim.winner()
            ticks.append(sim.tick)
            if win == a_pid:
                w += 1
            elif win == b_pid:
                l += 1
            else:
                d += 1
    tot = 2 * num_pairs
    avg_t = float(np.mean(ticks))
    print(
        f"[Galcon Large-Arena 22-28pl] {name_a} vs {name_b}: {w}-{l}-{d} ({100.0*w/tot:.1f}%) | avg_ticks={avg_t:.1f}",
        flush=True,
    )
    return w, l, d, avg_t


def rl_mirror_league_ppo_gen3(
    ckpt_path: Path,
    out_path: Path,
    *,
    redir_replay: Optional[Dict[str, np.ndarray]] = None,
    gen2_anchor_path: Optional[Path] = None,
    lua_opp_path: Optional[Path] = None,
    iters: int = 14,
    pairs_per_iter: int = 10,
    lr: float = 4.5e-5,
    kl_coef: float = 0.025,
    ema_tau: float = 0.07,
    gamma: float = 0.985,
    gae_lam: float = 0.95,
    frac_std: float = 0.065,
    draw_penalty: float = -0.40,
    eval_every: int = 5,
    seed: int = 20000,
) -> Path:
    """Gen 3 Paired-Seed Mirror League PPO (GRPO-style variance reduction) + Large-Arena Generalization.

    For every sampled map ``(map_seed, n_pl, w, h, symmetric)`` and opponent, plays BOTH
    P1 and P2 sides in the same iteration and centers split-pair advantages against the paired
    baseline — eliminating map-luck / asymmetry variance while training across compact, classic,
    and large 22-28 planet asymmetric ``gbots`` arenas against Gen 2/Gen 3 snapshots, heuristics,
    and optionally an external Galcon 2 Lua boss (e.g. ``esparano``'s ``Sparky v1.6``).
    """
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    lua_opp = None
    if lua_opp_path is not None and lua_opp_path.is_file():
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp = GalconLuaBotBridge(lua_opp_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    policy = build_galcon_gen2_transformer(
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        dim_ff=int(cfg["dim_ff"]),
        feature_version=int(cfg.get("feature_version", 2)),
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    has_redir = bool(ckpt.get("has_trained_redir", True))

    ema_teacher = copy.deepcopy(policy).eval()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    # Historical Snapshot Pool always preserves Gen 2 Champion (1628 Elo) at index 0!
    snapshot_pool: List[GalconPlanetTransformer] = []
    if gen2_anchor_path is not None and gen2_anchor_path.is_file():
        g2_ck = torch.load(str(gen2_anchor_path), map_location="cpu", weights_only=False)
        g2_cfg = g2_ck["model_config"]
        g2_m = build_galcon_gen2_transformer(
            d_model=int(g2_cfg["d_model"]),
            n_heads=int(g2_cfg["n_heads"]),
            n_layers=int(g2_cfg["n_layers"]),
            dim_ff=int(g2_cfg["dim_ff"]),
            feature_version=int(g2_cfg.get("feature_version", 2)),
        )
        g2_m.load_state_dict(g2_ck["model_state_dict"])
        g2_m.eval()
        for p in g2_m.parameters():
            p.requires_grad_(False)
        snapshot_pool.append(g2_m)
    snapshot_pool.append(copy.deepcopy(policy).eval())
    for p in snapshot_pool[-1].parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters), eta_min=lr * 0.25)

    redir_loader = None
    if redir_replay and len(redir_replay.get("planet_feats", [])) >= 16:
        rds = TensorDataset(
            torch.from_numpy(redir_replay["planet_feats"]),
            torch.from_numpy(redir_replay["timeline_feats"]),
            torch.from_numpy(redir_replay["edge_feats"]),
            torch.from_numpy(redir_replay["pad_mask"]),
            torch.from_numpy(redir_replay["cur_dst_idx"]),
            torch.from_numpy(redir_replay["redir_feats"]),
            torch.from_numpy(redir_replay["redir_target"]),
            torch.from_numpy(redir_replay["redir_mask"]),
        )
        redir_loader = DataLoader(rds, batch_size=32, shuffle=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_state = copy.deepcopy(policy.state_dict())
    total_eps = 0

    def quick_gen3_val(model_eval: GalconPlanetTransformer) -> float:
        """24-game validation across standard (Tactical, Rusher, Gen2) + Large 24pl Arena."""
        model_eval.eval()
        g2_ref = snapshot_pool[0]
        score_g = 0.0
        tot_g = 0
        # 16 standard games (Tactical x4, Rusher x4, Gen2-Champ x4, Sparky/Tactical x4)
        std_opps = [
            ("tac", GalconTacticalBot(), 4),
            ("rush", GalconRusherBot(style="all_in"), 4),
            ("g2", g2_ref, 4),
            ("boss", lua_opp if lua_opp is not None else GalconTacticalBot(), 4),
        ]
        for oi, (tag, ob, ng) in enumerate(std_opps):
            for k in range(ng):
                g_rng = random.Random(29000 + oi * 100 + k)
                n_p, ww, hh = _sample_map_config(g_rng)
                s = GalconSim.create(seed=29000 + oi * 100 + k, num_planets=n_p, width=ww, height=hh, max_ticks=200)
                a_p = 1 if (k % 2 == 0) else 2
                b_p = 2 if a_p == 1 else 1
                while not s.done:
                    if s.tick % 2 == 0:
                        oa = _fast_neural_act(model_eval, s, a_p)
                        ob_act = _fast_neural_act(ob, s, b_p) if tag == "g2" else ob.act(s, b_p)
                        s.step({a_p: oa, b_p: ob_act})
                    else:
                        s.step({})
                tot_g += 1
                w_id = s.winner()
                score_g += 1.0 if w_id == a_p else (0.5 if w_id == 0 else 0.0)
        # 8 large-arena games (24-26pl 560x340 vs Boss/Tactical & Gen2)
        for k in range(4):
            for a_p in (1, 2):
                b_p = 2 if a_p == 1 else 1
                ob = (lua_opp if lua_opp is not None else GalconTacticalBot()) if (k % 2 == 0) else g2_ref
                is_nn = (k % 2 == 1)
                s = GalconSim.create(
                    seed=29500 + k, num_planets=24 + (k % 3), width=560.0, height=340.0, max_ticks=220, symmetric=(k % 2 == 0)
                )
                while not s.done:
                    if s.tick % 2 == 0:
                        oa = _fast_neural_act(model_eval, s, a_p)
                        ob_act = _fast_neural_act(ob, s, b_p) if is_nn else ob.act(s, b_p)
                        s.step({a_p: oa, b_p: ob_act})
                    else:
                        s.step({})
                tot_g += 1
                w_id = s.winner()
                score_g += 1.0 if w_id == a_p else (0.5 if w_id == 0 else 0.0)
        return score_g / max(1, tot_g)

    def run_single_episode(
        map_seed: int,
        n_pl: int,
        w: float,
        h: float,
        sym: bool,
        ours_pid: int,
        opp_snap: Optional[GalconPlanetTransformer],
        opp_bot,
        is_boss: bool = False,
    ):
        opp_pid = 2 if ours_pid == 1 else 1
        sim = GalconSim.create(
            seed=map_seed, num_planets=n_pl, width=w, height=h, max_ticks=220 if n_pl >= 22 else 200, symmetric=sym
        )
        ep_steps = []
        snaps: List[Tuple[int, np.ndarray, np.ndarray]] = []
        prev_pot = evaluate_galcon_state(sim, ours_pid)
        while not sim.done:
            if sim.tick % 2 == 0:
                enc = extract_galcon_features(sim, ours_pid, feature_version=2)
                n = len(enc.planet_ids)
                so, ss = _snapshot_planet_state(sim, enc.planet_ids, ours_pid)
                snaps.append((sim.tick, so, ss))
                pad = pad_galcon_state(enc)

                with torch.no_grad():
                    pf_t = torch.from_numpy(enc.planet_feats[None])
                    tf_t = torch.from_numpy(enc.timeline_feats[None])
                    ef_t = torch.from_numpy(enc.edge_feats[None])
                    pm_t = torch.ones((1, n), dtype=torch.bool)
                    logits, fracs, val, h_emb = policy(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
                    le_t = torch.from_numpy(enc.legal_edges[None])
                    mask = torch.cat([torch.ones((1, n, 1), dtype=torch.bool), le_t], dim=-1)
                    logits = logits.masked_fill(~mask, -1e9)
                    dist = torch.distributions.Categorical(logits=logits[0])
                    act_short = dist.sample()
                    om_t = torch.from_numpy(enc.owned_mask)
                    lp_disc = (dist.log_prob(act_short) * om_t.float()).sum()

                    f_mean_all = fracs[0]
                    f_act_short = torch.zeros((n,), dtype=torch.float32)
                    lp_frac = torch.tensor(0.0)
                    for i in range(n):
                        if enc.owned_mask[i] and int(act_short[i].item()) > 0:
                            j = int(act_short[i].item()) - 1
                            mu = f_mean_all[i, j].clamp(0.08, 0.98)
                            f_dist = torch.distributions.Normal(mu, frac_std)
                            f_s = f_dist.sample().clamp(0.08, 1.0)
                            f_act_short[i] = f_s
                            lp_frac = lp_frac + f_dist.log_prob(f_s)

                    total_lp = float((lp_disc + 0.25 * lp_frac).item())
                    v_pred = float(val[0].item())

                    o_ours: List[Order] = []
                    for i in range(n):
                        if not enc.owned_mask[i]:
                            continue
                        a = int(act_short[i].item())
                        if a > 0:
                            j = a - 1
                            if j < n and enc.legal_edges[i, j]:
                                f_val = float(f_act_short[i].item())
                                pct = 100 if f_val >= 0.90 else max(5, min(100, int(round(f_val * 100.0))))
                                o_ours.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))

                    if has_redir and sim.allow_redirect and sim.fleets:
                        renc = extract_galcon_redir_features(sim, enc)
                        if renc.fleet_ids:
                            rlogits = policy.predict_redir_logits(
                                h_emb,
                                torch.from_numpy(renc.cur_dst_idx[None]),
                                torch.from_numpy(renc.redir_feats[None]),
                                pm_t,
                            )
                            rprobs = F.softmax(rlogits[0], dim=-1).numpy()
                            for fi, fid in enumerate(renc.fleet_ids):
                                c_idx = int(renc.cur_dst_idx[fi])
                                bj = int(np.argmax(rprobs[fi]))
                                if bj != c_idx and float(rprobs[fi, bj]) >= 0.55:
                                    o_ours.append(RedirOrder(fid, enc.planet_ids[bj]))

                act_pad = np.zeros((GALCON_MAX_PLANETS,), dtype=np.int64)
                act_pad[:n] = act_short.numpy()
                f_pad = np.zeros((GALCON_MAX_PLANETS,), dtype=np.float32)
                f_pad[:n] = f_act_short.numpy()

                o_opp = _fast_neural_act(opp_snap, sim, opp_pid) if opp_snap is not None else opp_bot.act(sim, opp_pid)
                cur_tick = sim.tick
                sim.step({ours_pid: o_ours, opp_pid: o_opp})
                if not sim.done:
                    sim.step({})
                new_pot = evaluate_galcon_state(sim, ours_pid)
                step_rew = 0.05 * (new_pot - prev_pot)
                prev_pot = new_pot
                ep_steps.append((cur_tick, pad, act_pad, f_pad, total_lp, v_pred, step_rew))
            else:
                sim.step({})

        enc_f = extract_galcon_features(sim, ours_pid, feature_version=2)
        so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, ours_pid)
        snaps.append((sim.tick, so, ss))
        win = sim.winner()
        speed_bonus = 0.10 * max(0.0, 1.0 - float(sim.tick) / max(1.0, float(sim.max_ticks)))
        d_rew = 0.10 if is_boss else draw_penalty
        term_rew = (1.0 + speed_bonus) if win == ours_pid else (d_rew if win == 0 else -1.0)
        won = 1 if win == ours_pid else 0

        T = len(ep_steps)
        advs = np.zeros((T,), dtype=np.float32)
        rets = np.zeros((T,), dtype=np.float32)
        if T > 0:
            rews = [s[6] for s in ep_steps]
            rews[-1] += term_rew
            vals = [s[5] for s in ep_steps] + [0.0]
            gae = 0.0
            for t in reversed(range(T)):
                delta = rews[t] + gamma * vals[t + 1] - vals[t]
                gae = delta + gamma * gae_lam * gae
                advs[t] = gae
                rets[t] = float(np.clip(gae + vals[t], -1.15, 1.15))
        return ep_steps, snaps, advs, rets, term_rew, won

    try:
        torch.set_num_threads(1)
        best_score = quick_gen3_val(policy)
        boss_tag = f"+LuaBoss({lua_opp_path.stem})" if lua_opp is not None else ""
        print(
            f"[Galcon Gen3 Mirror League PPO{boss_tag}] Start 24g std+large val score = {100*best_score:.1f}% (pool={len(snapshot_pool)})",
            flush=True,
        )
        t0 = time.time()

        for it in range(1, iters + 1):
            torch.set_num_threads(1)
            policy.eval()
            buf = {k: [] for k in ("pf", "tf", "ef", "le", "om", "pm", "acts", "f_acts", "old_lp", "adv", "ret", "aux_o", "aux_s")}
            wins = sp_wins = sp_eps = bot_wins = bot_eps = boss_wins = boss_eps = 0
            split_pairs = 0

            for k in range(pairs_per_iter):
                n_pl, w, h, sym = _sample_gen3_map_config(rng)
                map_seed = seed + it * 500 + k

                u = rng.random()
                opp_snap: Optional[GalconPlanetTransformer] = None
                opp_bot = None
                is_boss = False
                if lua_opp is not None and u < 0.25:
                    opp_bot = lua_opp
                    is_boss = True
                elif u < (0.60 if lua_opp is not None else 0.40):
                    opp_snap = snapshot_pool[0] if rng.random() < 0.45 else rng.choice(snapshot_pool)
                elif u < (0.76 if lua_opp is not None else 0.65):
                    opp_bot = _make_sparring_bot(rng, "tactical")
                elif u < (0.86 if lua_opp is not None else 0.80):
                    opp_bot = _make_sparring_bot(rng, "rusher")
                elif u < (0.93 if lua_opp is not None else 0.90):
                    opp_bot = _make_sparring_bot(rng, "classic")
                else:
                    opp_bot = _make_sparring_bot(rng, "macro")

                ep_a = run_single_episode(map_seed, n_pl, w, h, sym, 1, opp_snap, opp_bot, is_boss=is_boss)
                ep_b = run_single_episode(map_seed, n_pl, w, h, sym, 2, opp_snap, opp_bot, is_boss=is_boss)
                total_eps += 2

                term_a, won_a = ep_a[4], ep_a[5]
                term_b, won_b = ep_b[4], ep_b[5]
                wins += won_a + won_b
                if is_boss:
                    boss_eps += 2
                    boss_wins += won_a + won_b
                elif opp_snap is not None:
                    sp_eps += 2
                    sp_wins += won_a + won_b
                else:
                    bot_eps += 2
                    bot_wins += won_a + won_b

                pair_shift = 0.0
                if won_a != won_b:
                    split_pairs += 1
                    pair_shift = 0.30 * 0.5 * (term_a + term_b)

                for ep_steps, snaps, advs, rets, _, _ in (ep_a, ep_b):
                    for idx_t, (cur_tick, pad, act_pad, f_pad, old_lp, _, _) in enumerate(ep_steps):
                        target_tick = cur_tick + 10
                        fut_o, fut_s = snaps[-1][1], snaps[-1][2]
                        for st, so_s, ss_s in snaps:
                            if st >= target_tick:
                                fut_o, fut_s = so_s, ss_s
                                break
                        buf["pf"].append(pad["planet_feats"])
                        buf["tf"].append(pad["timeline_feats"])
                        buf["ef"].append(pad["edge_feats"])
                        buf["le"].append(pad["legal_edges"])
                        buf["om"].append(pad["owned_mask"])
                        buf["pm"].append(pad["pad_mask"])
                        buf["acts"].append(act_pad)
                        buf["f_acts"].append(f_pad)
                        buf["old_lp"].append(old_lp)
                        buf["ret"].append(rets[idx_t])
                        buf["adv"].append(advs[idx_t] - pair_shift)
                        buf["aux_o"].append(fut_o)
                        buf["aux_s"].append(fut_s)

            policy.train()
            adv_arr = np.array(buf["adv"], dtype=np.float32)
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)
            ds = TensorDataset(
                torch.from_numpy(np.stack(buf["pf"])),
                torch.from_numpy(np.stack(buf["tf"])),
                torch.from_numpy(np.stack(buf["ef"])),
                torch.from_numpy(np.stack(buf["le"])),
                torch.from_numpy(np.stack(buf["om"])),
                torch.from_numpy(np.stack(buf["pm"])),
                torch.from_numpy(np.stack(buf["acts"])),
                torch.from_numpy(np.stack(buf["f_acts"])),
                torch.from_numpy(np.array(buf["old_lp"], dtype=np.float32)),
                torch.from_numpy(adv_arr),
                torch.from_numpy(np.array(buf["ret"], dtype=np.float32)),
                torch.from_numpy(np.stack(buf["aux_o"])),
                torch.from_numpy(np.stack(buf["aux_s"])),
            )
            loader = DataLoader(ds, batch_size=64, shuffle=True)
            ent_coef = 0.005 * (1.0 - 0.50 * (it / max(1, iters)))
            r_iter = iter(redir_loader) if redir_loader is not None else None

            torch.set_num_threads(2)
            for _ in range(2):
                for b_idx, (pf, tf, ef, le, om, pm, acts, f_acts, old_lp, adv, ret, aux_o, aux_s) in enumerate(loader):
                    pf, tf, ef, le, om, pm, acts, f_acts, aux_o, aux_s = _trim_batch_n(
                        pf, tf, ef, le, om, pm, acts, f_acts, aux_o, aux_s
                    )
                    k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
                    pf_a = torch.from_numpy(apply_galcon_d4(pf.numpy(), k_rot=k_rot, flip_x=flip_x))

                    logits, fracs, val, aux_pred = policy(pf_a, tf, ef, pm, return_aux=True)
                    B, N, _ = logits.shape
                    mask = torch.cat([torch.ones((B, N, 1), dtype=torch.bool), le], dim=-1)
                    logits = logits.masked_fill(~mask, -1e9)
                    logp = F.log_softmax(logits, dim=-1)
                    probs = logp.exp()
                    act_lp = logp.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
                    new_lp_disc = (act_lp * om.float()).sum(dim=-1)

                    launch_m = om & (acts > 0)
                    dst_idx = (acts - 1).clamp(0, N - 1)
                    chosen_f_mu = fracs.gather(2, dst_idx.unsqueeze(-1)).squeeze(-1).clamp(0.08, 0.98)
                    f_dist = torch.distributions.Normal(chosen_f_mu, frac_std)
                    new_lp_frac = (f_dist.log_prob(f_acts) * launch_m.float()).sum(dim=-1)
                    new_lp = new_lp_disc + 0.25 * new_lp_frac

                    ratio = torch.exp((new_lp - old_lp).clamp(-8.0, 8.0))
                    surr = torch.min(ratio * adv, torch.clamp(ratio, 0.85, 1.15) * adv).mean()

                    ent = -(probs * logp).sum(dim=-1)
                    ent_mean = (ent * om.float()).sum() / om.float().sum().clamp_min(1.0)

                    with torch.no_grad():
                        e_logits, _, _ = ema_teacher(pf_a, tf, ef, pm)
                        e_logits = e_logits.masked_fill(~mask, -1e9)
                        e_logp = F.log_softmax(e_logits, dim=-1)
                        e_p = e_logp.exp()
                    kl = (e_p * (e_logp - logp)).sum(dim=-1)
                    kl_mean = (kl * om.float()).sum() / om.float().sum().clamp_min(1.0)

                    v_loss = F.mse_loss(val, ret)
                    aux_loss = F.cross_entropy(aux_pred[pm, :3], aux_o[pm]) + 0.5 * F.smooth_l1_loss(aux_pred[pm, 3], aux_s[pm])
                    loss = -surr + 0.40 * v_loss + kl_coef * kl_mean - ent_coef * ent_mean + 0.14 * aux_loss

                    if r_iter is not None and (b_idx % 4 == 0):
                        try:
                            rb = next(r_iter)
                        except StopIteration:
                            r_iter = iter(redir_loader)
                            rb = next(r_iter)
                        rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk = rb
                        kr, fx = random.randrange(4), bool(random.randrange(2))
                        rpf_a = torch.from_numpy(apply_galcon_d4(rpf.numpy(), k_rot=kr, flip_x=fx))
                        rloss, _, _ = _redir_batch_loss(policy, rpf_a, rtf, ref, rpm, rcur, rfeat, rtar, rmsk)
                        loss = loss + 0.25 * rloss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
                    opt.step()

            sched.step()
            torch.set_num_threads(1)

            with torch.no_grad():
                for p_ema, p_cur in zip(ema_teacher.parameters(), policy.parameters()):
                    p_ema.data.mul_(1.0 - ema_tau).add_(p_cur.data, alpha=ema_tau)

            if it % 3 == 0:
                snap_copy = copy.deepcopy(policy).eval()
                for p in snap_copy.parameters():
                    p.requires_grad_(False)
                snapshot_pool.append(snap_copy)
                if len(snapshot_pool) > 10:
                    snapshot_pool = [snapshot_pool[0]] + snapshot_pool[-9:]

            boss_str = f", boss={boss_wins}/{boss_eps}" if boss_eps > 0 else ""
            log_msg = (
                f"[Galcon Gen3 Mirror PPO] iter {it:2d}/{iters} (eps={total_eps}, {time.time()-t0:5.1f}s) | "
                f"wins={wins}/{2*pairs_per_iter} (sp={sp_wins}/{sp_eps}, bots={bot_wins}/{bot_eps}{boss_str}, splits={split_pairs}) | "
                f"states={len(buf['ret'])} | pool={len(snapshot_pool)}"
            )
            if it % eval_every == 0 or it == iters:
                g_sc = quick_gen3_val(policy)
                log_msg += f" | val_24g={100*g_sc:.1f}%"
                if g_sc >= best_score:
                    best_score = g_sc
                    best_state = copy.deepcopy(policy.state_dict())
                    torch.save(
                        {
                            "model_config": policy.config,
                            "model_state_dict": best_state,
                            "has_trained_redir": has_redir or (redir_loader is not None),
                            "val_gen3_score": best_score,
                            "ppo_episodes": total_eps,
                        },
                        out_path,
                    )
                    log_msg += " *BEST*"
            print(log_msg, flush=True)

        final_path = out_path.with_name(out_path.stem + "_final.pt")
        torch.save(
            {
                "model_config": policy.config,
                "model_state_dict": policy.state_dict(),
                "has_trained_redir": has_redir or (redir_loader is not None),
                "ppo_episodes": total_eps,
            },
            final_path,
        )
        torch.save(
            {
                "model_config": policy.config,
                "model_state_dict": best_state,
                "has_trained_redir": has_redir or (redir_loader is not None),
                "val_gen3_score": best_score,
                "ppo_episodes": total_eps,
            },
            out_path,
        )
        return out_path
    finally:
        if lua_opp is not None:
            lua_opp.close()


def evaluate_matchup(
    name_a: str,
    bot_a,
    name_b: str,
    bot_b,
    *,
    num_games: int = 12,
    seed: int = 9000,
) -> Tuple[int, int, int, float]:
    rng = random.Random(seed)
    w = l = d = 0
    ticks = []
    for g in range(num_games):
        n_pl, width, height = _sample_map_config(rng)
        sim = GalconSim.create(seed=seed + g, num_planets=n_pl, width=width, height=height, max_ticks=220)
        a_pid = 1 if (g % 2 == 0) else 2
        b_pid = 2 if a_pid == 1 else 1
        while not sim.done:
            if sim.tick % 2 == 0:
                oa = bot_a.act(sim, a_pid)
                ob = bot_b.act(sim, b_pid)
                sim.step({a_pid: oa, b_pid: ob})
            else:
                sim.step({})
        win = sim.winner()
        ticks.append(sim.tick)
        if win == a_pid:
            w += 1
        elif win == b_pid:
            l += 1
        else:
            d += 1
    avg_t = float(np.mean(ticks))
    print(f"[Galcon Eval] {name_a} vs {name_b}: {w}-{l}-{d} ({100.0*w/num_games:.1f}%) | avg_ticks={avg_t:.1f}", flush=True)
    return w, l, d, avg_t


def _merge_datasets(ds_list: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    valid = [d for d in ds_list if d and len(next(iter(d.values()))) > 0]
    if not valid:
        return {}
    return {k: np.concatenate([d[k] for d in valid], axis=0) for k in valid[0].keys()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train GalconPlanetTransformer (v1, Gen 2, or Gen 3 Mirror League).")
    parser.add_argument("--mode", choices=["gen2", "v1"], default="gen2")
    parser.add_argument("--stage", choices=["all", "ei", "ppo", "mirror_ppo", "eval"], default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--gen2-anchor", default="checkpoints/galcon_gen2_champion.pt")
    parser.add_argument("--lua-opp", default="")
    parser.add_argument("--redir-npz", default="checkpoints/galcon_gen2_redir.npz")
    parser.add_argument("--bc-games", type=int, default=80)
    parser.add_argument("--dagger-games", type=int, default=30)
    parser.add_argument("--search-games", type=int, default=64)
    parser.add_argument("--search-rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ppo-iters", type=int, default=26)
    parser.add_argument("--ppo-eps", type=int, default=16)
    parser.add_argument("--mirror-pairs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--out", default="checkpoints/galcon_transformer_best.pt")
    parser.add_argument("--v1-ckpt", default="checkpoints/galcon_v1_baseline.pt")
    args = parser.parse_args(argv)

    out_path = Path(args.out)

    if args.mode == "v1":
        bc_path = out_path.with_name("galcon_v1_bc.pt")
        dagger_path = out_path.with_name("galcon_v1_dagger.pt")
        ppo_path = out_path.with_name("galcon_v1_ppo.pt")

        bc_data = collect_teacher_dataset(num_games=args.bc_games, seed=1000)
        train_supervised(bc_data, epochs=args.epochs, lr=1.4e-3, out_path=bc_path, seed=42)

        stu = GalconTransformerAgent(bc_path, use_d4_tta=False)
        dg_data = collect_teacher_dataset(num_games=args.dagger_games, seed=3000, dagger_agent=stu)
        merged = {k: np.concatenate([bc_data[k], dg_data[k], dg_data[k]], axis=0) for k in bc_data}
        train_supervised(merged, epochs=max(4, args.epochs // 2), lr=5e-4, out_path=dagger_path, resume_from=bc_path, seed=43)

        rl_selfplay_ppo(dagger_path, ppo_path, iters=args.ppo_iters, episodes_per_iter=args.ppo_eps)
        out_path.write_bytes(ppo_path.read_bytes())
        print(f"[Galcon v1] Saved baseline -> {out_path}", flush=True)
        return 0

    redir_npz = Path(args.redir_npz)

    def _load_redir_npz() -> Dict[str, np.ndarray]:
        if redir_npz.is_file():
            z = np.load(str(redir_npz))
            return {k: z[k] for k in z.files}
        return {}

    def _save_redir_npz(rd: Dict[str, np.ndarray]) -> None:
        if rd:
            redir_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(redir_npz), **rd)

    # Modular single-stage execution
    if args.stage == "ei":
        resume_p = Path(args.resume) if args.resume else out_path.with_name("galcon_gen2_ei_r1.pt")
        lua_p = Path(args.lua_opp) if args.lua_opp else None
        s_data, s_redir = collect_alphazero_search_dataset(
            resume_p,
            num_games=args.search_games,
            seed=args.seed,
            use_d4_tta=False,
            lua_opp_path=lua_p,
            gen3_maps=(lua_p is not None),
        )
        old_r = _load_redir_npz()
        merged_r = _merge_datasets([old_r, s_redir]) if old_r else s_redir
        _save_redir_npz(merged_r)
        train_gen2_supervised(
            s_data,
            merged_r,
            epochs=args.epochs,
            lr=args.lr if args.lr > 0 else 2.2e-4,
            out_path=out_path,
            resume_from=resume_p,
            seed=args.seed,
            label="Gen3-ReactiveEI" if lua_p is not None else "Gen2-EI",
        )
        return 0

    if args.stage == "ppo":
        resume_p = Path(args.resume) if args.resume else out_path.with_name("galcon_gen2_ei_r2.pt")
        redir_replay = _load_redir_npz()
        rl_league_ppo_gen2(
            resume_p,
            out_path,
            redir_replay=redir_replay,
            v1_ckpt_path=Path(args.v1_ckpt) if args.v1_ckpt else None,
            iters=args.ppo_iters,
            episodes_per_iter=args.ppo_eps,
            lr=args.lr if args.lr > 0 else 9.0e-5,
            seed=args.seed,
        )
        return 0

    if args.stage == "mirror_ppo":
        resume_p = Path(args.resume) if args.resume else Path("checkpoints/galcon_gen2_champion.pt")
        redir_replay = _load_redir_npz()
        rl_mirror_league_ppo_gen3(
            resume_p,
            out_path,
            redir_replay=redir_replay,
            gen2_anchor_path=Path(args.gen2_anchor) if args.gen2_anchor else None,
            lua_opp_path=Path(args.lua_opp) if args.lua_opp else None,
            iters=args.ppo_iters,
            pairs_per_iter=args.mirror_pairs,
            lr=args.lr if args.lr > 0 else 4.5e-5,
            seed=args.seed,
        )
        return 0

    if args.stage == "eval":
        tactical = GalconTacticalBot()
        classic = GalconClassicBot()
        rusher_ai = GalconRusherBot(style="all_in")
        rusher_ad = GalconRusherBot(style="adaptive")
        macro = GalconMacroBot()

        ag_best = GalconTransformerAgent(out_path, use_d4_tta=True)
        ag_search = GalconSearchTransformerAgent(out_path, use_d4_tta=True)
        print(f"\n=== Held-Out Multi-Map Gauntlet ({out_path}) ===", flush=True)
        for name, opp, sd in [
            ("GalconTacticalBot", tactical, 12000),
            ("GalconClassicBot", classic, 12100),
            ("GalconRusherBot (all_in)", rusher_ai, 12200),
            ("GalconRusherBot (adaptive)", rusher_ad, 12300),
            ("GalconMacroBot", macro, 12400),
        ]:
            evaluate_matchup("GalconPT-Gen2 Pure (8x D4)", ag_best, name, opp, num_games=args.eval_games, seed=sd)
        if Path(args.v1_ckpt).is_file():
            ag_v1 = GalconTransformerAgent(args.v1_ckpt, use_d4_tta=True)
            ag_v1_s = GalconSearchTransformerAgent(args.v1_ckpt, use_d4_tta=True)
            evaluate_matchup("GalconPT-Gen2 Pure (8x D4)", ag_best, "GalconPT-v1 Pure (8x D4)", ag_v1, num_games=args.eval_games, seed=12500)
            evaluate_matchup("GalconPT-Gen2 + Search", ag_search, "GalconPT-v1 + Search", ag_v1_s, num_games=args.eval_games, seed=12600)
        for name, opp, sd in [
            ("GalconTacticalBot", tactical, 12000),
            ("GalconClassicBot", classic, 12100),
            ("GalconRusherBot (all_in)", rusher_ai, 12200),
            ("GalconMacroBot", macro, 12400),
        ]:
            evaluate_matchup("GalconPT-Gen2 + Search", ag_search, name, opp, num_games=args.eval_games, seed=sd)
        return 0

    # --- Full Gen 2 Teacher-Free Self-Play RL Flywheel ---
    ws_path = out_path.with_name("galcon_gen2_warmstart.pt")
    ei_path = out_path.with_name("galcon_gen2_ei.pt")
    ppo_path = out_path.with_name("galcon_gen2_league_ppo.pt")

    # Stage 1: Multi-Style Winner-Filtered Warm-Start + Aux t+10 + /REDIR Bootstrap
    ws_data, ws_redir = collect_gen2_warmstart_dataset(num_games=args.bc_games, seed=2000)
    train_gen2_supervised(
        ws_data,
        ws_redir,
        epochs=args.epochs,
        lr=1.1e-3,
        out_path=ws_path,
        seed=42,
        label="Gen2-WarmStart",
    )

    # Stage 2: AlphaZero Search Self-Play Distillation (Expert Iteration — zero teacher ceiling!)
    cur_ckpt = ws_path
    all_search_data: List[Dict[str, np.ndarray]] = []
    all_redir_data: List[Dict[str, np.ndarray]] = [ws_redir]
    for rnd in range(1, args.search_rounds + 1):
        s_data, s_redir = collect_alphazero_search_dataset(
            cur_ckpt,
            num_games=args.search_games,
            seed=4000 + rnd * 500,
            use_d4_tta=False,
        )
        all_search_data.append(s_data)
        if s_redir:
            all_redir_data.append(s_redir)
        if rnd == 1:
            merged_d = _merge_datasets([ws_data, s_data, s_data])
        else:
            merged_d = _merge_datasets(all_search_data)
        merged_r = _merge_datasets(all_redir_data)
        _save_redir_npz(merged_r)
        rnd_out = out_path.with_name(f"galcon_gen2_ei_r{rnd}.pt")
        train_gen2_supervised(
            merged_d,
            merged_r,
            epochs=max(6, args.epochs - 2),
            lr=5.5e-4 if rnd == 1 else 3.5e-4,
            out_path=rnd_out,
            resume_from=cur_ckpt,
            seed=100 + rnd,
            label=f"Gen2-EI-R{rnd}",
        )
        cur_ckpt = rnd_out
    ei_path.write_bytes(cur_ckpt.read_bytes())

    # Stage 3: Large-Scale Teacher-Free League PPO + Snapshot Pool + Moving EMA KL
    merged_redir = _merge_datasets(all_redir_data)
    rl_league_ppo_gen2(
        ei_path,
        ppo_path,
        redir_replay=merged_redir,
        iters=args.ppo_iters,
        episodes_per_iter=args.ppo_eps,
        seed=6000,
    )

    # Stage 4: Full Held-Out Multi-Map Gauntlet Benchmark & Promotion
    tactical = GalconTacticalBot()
    classic = GalconClassicBot()
    rusher_ai = GalconRusherBot(style="all_in")
    rusher_ad = GalconRusherBot(style="adaptive")
    macro = GalconMacroBot()

    ag_ei = GalconTransformerAgent(ei_path, use_d4_tta=True)
    ag_ppo = GalconTransformerAgent(ppo_path, use_d4_tta=True)

    print("\n=== Held-Out Multi-Map Gauntlet (Gen 2 EI vs Gen 2 League PPO) ===", flush=True)
    ei_tot = 0
    ppo_tot = 0
    for name, opp, sd in [
        ("GalconTacticalBot", tactical, 12000),
        ("GalconClassicBot", classic, 12100),
        ("GalconRusherBot (all_in)", rusher_ai, 12200),
        ("GalconRusherBot (adaptive)", rusher_ad, 12300),
        ("GalconMacroBot", macro, 12400),
    ]:
        w_ei, _, _, _ = evaluate_matchup("GalconPT-Gen2 EI (8x D4)", ag_ei, name, opp, num_games=args.eval_games, seed=sd)
        w_ppo, _, _, _ = evaluate_matchup("GalconPT-Gen2 PPO (8x D4)", ag_ppo, name, opp, num_games=args.eval_games, seed=sd)
        ei_tot += w_ei
        ppo_tot += w_ppo

    best_src = ppo_path if ppo_tot >= ei_tot else ei_path
    out_path.write_bytes(best_src.read_bytes())
    print(f"\n[Galcon Gen2] Promoted {best_src} -> {out_path} (PPO={ppo_tot} vs EI={ei_tot} / {5*args.eval_games})", flush=True)

    ag_best = GalconTransformerAgent(out_path, use_d4_tta=True)
    ag_search = GalconSearchTransformerAgent(out_path, use_d4_tta=True)

    if Path(args.v1_ckpt).is_file():
        ag_v1 = GalconTransformerAgent(args.v1_ckpt, use_d4_tta=True)
        ag_v1_s = GalconSearchTransformerAgent(args.v1_ckpt, use_d4_tta=True)
        evaluate_matchup("GalconPT-Gen2 Pure (8x D4)", ag_best, "GalconPT-v1 Pure (8x D4)", ag_v1, num_games=args.eval_games, seed=12500)
        evaluate_matchup("GalconPT-Gen2 + Search", ag_search, "GalconPT-v1 + Search", ag_v1_s, num_games=args.eval_games, seed=12600)

    for name, opp, sd in [
        ("GalconTacticalBot", tactical, 12000),
        ("GalconClassicBot", classic, 12100),
        ("GalconRusherBot (all_in)", rusher_ai, 12200),
        ("GalconMacroBot", macro, 12400),
    ]:
        evaluate_matchup("GalconPT-Gen2 + Search", ag_search, name, opp, num_games=args.eval_games, seed=sd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
