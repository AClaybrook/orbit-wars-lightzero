"""Train & benchmark our Galcon PlanetTimelineTransformer (BC + DAgger + Self-Play PPO RL).

Trains across diverse continuous Galcon maps (8..22 planets, 200x125 to 400x240) using:
1. Behavior Cloning + DAgger against our timeline-aware ``GalconTacticalBot`` solver,
2. Exact 8x D4 dihedral symmetry augmentation,
3. On-policy Self-Play PPO + Anchor-KL RL in ``GalconSim`` so the neural bot surpasses its teacher!
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from galcon.heuristic_bot import GalconClassicBot, GalconTacticalBot
from galcon.sim import GalconSim, Order, SendOrder, evaluate_galcon_state
from galcon.transformer_bot import (
    GALCON_MAX_PLANETS,
    GalconSearchTransformerAgent,
    GalconTransformerAgent,
    apply_galcon_d4,
    build_galcon_transformer,
    extract_galcon_features,
    pad_galcon_state,
)


def _sample_map_config(rng: random.Random) -> Tuple[int, float, float]:
    """Mix compact tactical maps (Evan's 8-12 planet scale) and classic 16-22 planet Galcon maps."""
    if rng.random() < 0.40:
        return rng.randint(8, 12), 240.0, 150.0
    return rng.randint(14, 22), 400.0, 240.0


def _orders_to_targets(enc, orders: Sequence[Order]) -> Tuple[np.ndarray, np.ndarray]:
    tcls = np.zeros((GALCON_MAX_PLANETS,), dtype=np.int64)
    tfrac = np.zeros((GALCON_MAX_PLANETS,), dtype=np.float32)
    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    for o in orders:
        if not isinstance(o, SendOrder):
            continue
        si = id_to_idx.get(o.src_id)
        di = id_to_idx.get(o.dst_id)
        if si is not None and di is not None and enc.owned_mask[si] and enc.legal_edges[si, di]:
            tcls[si] = di + 1
            tfrac[si] = float(o.percent) / 100.0
    return tcls, tfrac


def collect_teacher_dataset(
    *,
    num_games: int = 80,
    seed: int = 1000,
    sample_stride: int = 2,
    dagger_agent: GalconTransformerAgent | None = None,
) -> Dict[str, np.ndarray]:
    """Collect multi-map Galcon states labeled by ``GalconTacticalBot``."""
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
                # Query tactical oracle on student-visited states!
                if sim.tick % sample_stride == 0:
                    oracle_orders = teacher_hi.act(sim, student_pid)
                    enc = extract_galcon_features(sim, student_pid)
                    if np.any(enc.owned_mask):
                        pad = pad_galcon_state(enc)
                        tc, fr = _orders_to_targets(enc, oracle_orders)
                        traj.append((student_pid, pad, tc, fr))
            else:
                o1 = teacher_hi.act(sim, 1)
                o2 = opp_bot.act(sim, 2)
                for pid_rec, ords in ((1, o1), (2, o2)) if opp_bot is teacher_lo else ((1, o1),):
                    if (sim.tick % sample_stride == 0) or any(isinstance(x, SendOrder) for x in ords):
                        enc = extract_galcon_features(sim, pid_rec)
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
    print(f"[Galcon {mode}] Collected {len(vt_l)} board states across {num_games} games in {time.time()-t0:.1f}s", flush=True)
    return data


def _batch_loss(model, pf, tf, ef, le, om, pm, tcls, tfrac, vtar, launch_weight: float = 1.35):
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
    """On-policy Self-Play PPO + Anchor-KL RL in ``GalconSim`` to surpass ``GalconTacticalBot``."""
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
                # Sample our action on every 2nd tick (matches human/bot tempo & keeps PPO crisp)
                if sim.tick % 2 == 0:
                    enc = extract_galcon_features(sim, ours_pid)
                    pad = pad_galcon_state(enc)
                    n = len(enc.planet_ids)
                    with torch.no_grad():
                        logits, fracs, val = policy(
                            torch.from_numpy(pad["planet_feats"][None]),
                            torch.from_numpy(pad["timeline_feats"][None]),
                            torch.from_numpy(pad["edge_feats"][None]),
                            torch.from_numpy(pad["pad_mask"][None]),
                        )
                        le_t = torch.from_numpy(pad["legal_edges"][None])
                        mask = torch.cat([torch.ones((1, GALCON_MAX_PLANETS, 1), dtype=torch.bool), le_t], dim=-1)
                        logits = logits.masked_fill(~mask, -1e9)
                        dist = torch.distributions.Categorical(logits=logits[0])
                        act_vec = dist.sample()
                        om_t = torch.from_numpy(pad["owned_mask"])
                        lp = (dist.log_prob(act_vec) * om_t.float()).sum().item()
                        fracs_np = fracs[0].numpy()
                        v_pred = float(val[0].item())

                    o_ours: List[Order] = []
                    for i in range(n):
                        if not enc.owned_mask[i]:
                            continue
                        a = int(act_vec[i].item())
                        if a > 0:
                            j = a - 1
                            if j < n and enc.legal_edges[i, j]:
                                f = float(fracs_np[i, j])
                                pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
                                o_ours.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                    ep_steps.append((pad, act_vec.numpy(), lp, v_pred))

                    if use_selfplay:
                        o_enc = extract_galcon_features(sim, opp_pid)
                        o_pad = pad_galcon_state(o_enc)
                        with torch.no_grad():
                            ol, of, _ = anchor(
                                torch.from_numpy(o_pad["planet_feats"][None]),
                                torch.from_numpy(o_pad["timeline_feats"][None]),
                                torch.from_numpy(o_pad["edge_feats"][None]),
                                torch.from_numpy(o_pad["pad_mask"][None]),
                            )
                            ole = torch.from_numpy(o_pad["legal_edges"][None])
                            omk = torch.cat([torch.ones((1, GALCON_MAX_PLANETS, 1), dtype=torch.bool), ole], dim=-1)
                            ol = ol.masked_fill(~omk, -1e9)
                            op_np = F.softmax(ol[0], dim=-1).numpy()
                            of_np = of[0].numpy()
                        o_opp: List[Order] = []
                        for i in range(len(o_enc.planet_ids)):
                            if o_enc.owned_mask[i] and (1.0 - op_np[i, 0]) >= 0.50:
                                j = int(np.argmax(op_np[i, 1 : len(o_enc.planet_ids) + 1]))
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
            # Discounted terminal returns + GAE-like advantage
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

        # PPO update with Anchor-KL regularization
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
                k_rot, flip_x = random.randrange(4), bool(random.randrange(2))
                pf_a = torch.from_numpy(apply_galcon_d4(pf.numpy(), k_rot=k_rot, flip_x=flip_x))
                logits, _, val = policy(pf_a, tf, ef, pm)
                mask = torch.cat([torch.ones((logits.size(0), GALCON_MAX_PLANETS, 1), dtype=torch.bool), le], dim=-1)
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

        print(f"[Galcon PPO] iter {it}/{iters} ({time.time()-t0:.1f}s) | eps_wins={wins}/{episodes_per_iter} | states={len(buf['ret'])}", flush=True)

    torch.save({"model_config": policy.config, "model_state_dict": policy.state_dict()}, out_path)
    return out_path


def evaluate_matchup(name_a, bot_a, name_b, bot_b, *, num_games: int = 12, seed: int = 9000) -> Tuple[int, int, int, float]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train GalconPlanetTransformer (BC + DAgger + Self-Play PPO).")
    parser.add_argument("--bc-games", type=int, default=70)
    parser.add_argument("--dagger-games", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ppo-iters", type=int, default=4)
    parser.add_argument("--ppo-eps", type=int, default=12)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--out", default="checkpoints/galcon_transformer_best.pt")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    bc_path = out_path.with_name("galcon_transformer_bc.pt")
    dagger_path = out_path.with_name("galcon_transformer_dagger.pt")
    ppo_path = out_path.with_name("galcon_transformer_ppo.pt")

    # 1. BC on GalconTacticalBot
    bc_data = collect_teacher_dataset(num_games=args.bc_games, seed=1000)
    train_supervised(bc_data, epochs=args.epochs, lr=1.4e-3, out_path=bc_path, seed=42)

    # 2. DAgger on student-visited states
    stu = GalconTransformerAgent(bc_path, use_d4_tta=False)
    dg_data = collect_teacher_dataset(num_games=args.dagger_games, seed=3000, dagger_agent=stu)
    merged = {k: np.concatenate([bc_data[k], dg_data[k], dg_data[k]], axis=0) for k in bc_data}
    train_supervised(merged, epochs=max(4, args.epochs // 2), lr=5e-4, out_path=dagger_path, resume_from=bc_path, seed=43)

    # 3. Self-Play PPO + Anchor-KL RL
    rl_selfplay_ppo(dagger_path, ppo_path, iters=args.ppo_iters, episodes_per_iter=args.ppo_eps)

    # 4. Benchmark all stages & promote winner!
    classic = GalconClassicBot()
    tactical = GalconTacticalBot()
    ag_dg = GalconTransformerAgent(dagger_path, use_d4_tta=True)
    ag_ppo = GalconTransformerAgent(ppo_path, use_d4_tta=True)
    ag_search = GalconSearchTransformerAgent(ppo_path, use_d4_tta=True)

    evaluate_matchup("GalconTacticalBot", tactical, "GalconClassicBot", classic, num_games=args.eval_games, seed=8000)
    w_dg, _, _, _ = evaluate_matchup("GalconPT (DAgger 8x D4)", ag_dg, "GalconTacticalBot", tactical, num_games=args.eval_games, seed=8100)
    w_ppo, _, _, _ = evaluate_matchup("GalconPT (PPO 8x D4)", ag_ppo, "GalconTacticalBot", tactical, num_games=args.eval_games, seed=8100)
    evaluate_matchup("GalconPT (PPO 8x D4)", ag_ppo, "GalconClassicBot", classic, num_games=args.eval_games, seed=8200)
    evaluate_matchup("GalconPT (PPO) + Search", ag_search, "GalconTacticalBot", tactical, num_games=args.eval_games, seed=8100)

    best_src = ppo_path if w_ppo >= w_dg else dagger_path
    out_path.write_bytes(best_src.read_bytes())
    print(f"[Galcon] Promoted {best_src} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
