#!/usr/bin/env python3
"""Train ``PlanetTimelineTransformer`` via Behavior Cloning + Value Learning on elite RL replays.

Workflow:
  1. Collects diverse official ``kaggle_environments`` replays where ``hober_malloc``
     (10B-step ModernBERT RL) plays 2P mirrors, 2P vs heuristics, and mixed 4P FFA tables.
  2. Extracts lossless per-planet features, 20-step projected arrival timelines, and
     D4-invariant short-ETA edge features, recovering exact ``(source -> target, frac)``
     labels for every owned planet via ``physics.predict_fleet_arrivals``.
  3. Trains ``PlanetTimelineTransformer`` with 8x D4 symmetry augmentation, masked
     per-planet cross-entropy, send-fraction Huber loss, and game-outcome value MSE.

Example:
    python scripts/train_planet_transformer.py --games-2p 24 --games-4p 10 --epochs 25 \
        --dataset-cache scratch/pt_dataset.npz --out checkpoints/planet_transformer_best.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orbit_wars_bot.planet_transformer import (  # noqa: E402
    EDGE_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    TIMELINE_FEAT_DIM,
    TIMELINE_HORIZON,
    PlanetTimelineTransformer,
    apply_d4_planet_feats,
    extract_planet_features,
    match_action_to_target,
    pad_encoded_state,
)

# Per-worker agent cache so each multiprocessing worker spawns Hober Malloc only once.
_WORKER_AGENTS: Dict[str, Any] = {}


def _get_worker_agent(spec: str):
    if spec not in _WORKER_AGENTS:
        from evaluate import resolve_agent

        _WORKER_AGENTS[spec] = resolve_agent(spec)
    return _WORKER_AGENTS[spec]


def _collect_one_game(task: Tuple[List[str], int, List[int], int, bool]) -> Dict[str, Any]:
    """Play one official game and extract teacher samples for ``teacher_slots``."""
    bot_specs, seed, teacher_slots, sample_stride, fast_teacher = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if fast_teacher:
        # Turning off 8x JAX TTA inside Hober Malloc speeds up teacher replay collection ~5-6x
        # while keeping the full 10B-step ModernBERT RL policy intact.
        os.environ["OW_ROTATION_TTA"] = "0"
        os.environ["OW_OPPONENT_TTA"] = "0"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    from kaggle_environments import make

    agents = [_get_worker_agent(s) for s in bot_specs]
    random.seed(seed)
    np.random.seed(seed)
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "actTimeout": 30, "agentTimeout": 60},
        debug=False,
    )
    steps = env.run(agents)
    n_players = len(bot_specs)
    final = steps[-1]
    rewards = [
        float(final[i]["reward"]) if final[i]["reward"] is not None else -1.0
        for i in range(n_players)
    ]

    # Compute survival-graduated value targets in [-1, +1] for 4P and {-1, +1} for 2P
    if n_players == 2:
        val_targets = [1.0 if r > 0 else (-1.0 if r < 0 else 0.0) for r in rewards]
    else:
        last_alive = [0] * n_players
        for t_idx, st in enumerate(steps):
            obs = st[0]["observation"]
            alive = [0] * n_players
            for p in obs.get("planets", []):
                if 0 <= int(p[1]) < n_players:
                    alive[int(p[1])] += int(p[5])
            for fl in obs.get("fleets", []):
                if 0 <= int(fl[1]) < n_players:
                    alive[int(fl[1])] += int(fl[6])
            for i in range(n_players):
                if alive[i] > 0:
                    last_alive[i] = t_idx
        order = sorted(range(n_players), key=lambda i: (rewards[i], last_alive[i]), reverse=True)
        grad = [1.0, 0.25, -0.35, -1.0]
        val_targets = [0.0] * n_players
        for rank_idx, pid in enumerate(order):
            val_targets[pid] = grad[rank_idx]

    pf_list: List[np.ndarray] = []
    tf_list: List[np.ndarray] = []
    ef_list: List[np.ndarray] = []
    le_list: List[np.ndarray] = []
    om_list: List[np.ndarray] = []
    pm_list: List[np.ndarray] = []
    act_list: List[np.ndarray] = []
    frac_list: List[np.ndarray] = []
    val_list: List[float] = []

    matched_moves = 0
    total_moves = 0

    for t in range(len(steps) - 1):
        obs_t = steps[t][0]["observation"]
        nxt = steps[t + 1]

        for pid in teacher_slots:
            moves = nxt[pid].get("action") or []
            # Keep ALL active launch turns + every sample_stride-th quiet turn so hold/launch balance stays sharp
            if not moves and (t % sample_stride != 0):
                continue

            enc = extract_planet_features(obs_t, player=pid)
            if not np.any(enc.owned_mask):
                continue

            padded = pad_encoded_state(enc, MAX_PLANETS)
            # Default action for every owned planet is 0 (HOLD)
            target_cls = np.zeros((MAX_PLANETS,), dtype=np.int64)
            target_frac = np.ones((MAX_PLANETS,), dtype=np.float32)

            for mv in moves:
                total_moves += 1
                rec = match_action_to_target(mv, enc)
                if rec is None:
                    continue
                src_i, dst_i, frac = rec
                if padded["legal_edges"][src_i, dst_i]:
                    target_cls[src_i] = 1 + dst_i
                    target_frac[src_i] = frac
                    matched_moves += 1

            pf_list.append(padded["planet_feats"])
            tf_list.append(padded["timeline_feats"])
            ef_list.append(padded["edge_feats"])
            le_list.append(padded["legal_edges"])
            om_list.append(padded["owned_mask"])
            pm_list.append(padded["pad_mask"])
            act_list.append(target_cls)
            frac_list.append(target_frac)
            val_list.append(val_targets[pid])

    return {
        "n_samples": len(val_list),
        "matched_moves": matched_moves,
        "total_moves": total_moves,
        "steps": len(steps) - 1,
        "planet_feats": np.stack(pf_list, axis=0) if pf_list else np.zeros((0, MAX_PLANETS, PLANET_FEAT_DIM), np.float32),
        "timeline_feats": np.stack(tf_list, axis=0) if tf_list else np.zeros((0, MAX_PLANETS, TIMELINE_HORIZON, TIMELINE_FEAT_DIM), np.float32),
        "edge_feats": np.stack(ef_list, axis=0) if ef_list else np.zeros((0, MAX_PLANETS, MAX_PLANETS, EDGE_FEAT_DIM), np.float32),
        "legal_edges": np.stack(le_list, axis=0) if le_list else np.zeros((0, MAX_PLANETS, MAX_PLANETS), bool),
        "owned_mask": np.stack(om_list, axis=0) if om_list else np.zeros((0, MAX_PLANETS), bool),
        "pad_mask": np.stack(pm_list, axis=0) if pm_list else np.zeros((0, MAX_PLANETS), bool),
        "target_cls": np.stack(act_list, axis=0) if act_list else np.zeros((0, MAX_PLANETS), np.int64),
        "target_frac": np.stack(frac_list, axis=0) if frac_list else np.zeros((0, MAX_PLANETS), np.float32),
        "val_targets": np.asarray(val_list, dtype=np.float32),
    }


def collect_dataset(
    games_2p: int,
    games_4p: int,
    workers: int,
    seed: int,
    sample_stride: int = 2,
    fast_teacher: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate a diverse 2P & 4P Behavior-Cloning dataset from Hober Malloc."""
    tasks: List[Tuple[List[str], int, List[int], int, bool]] = []
    # Mix of 2P Hober Malloc mirrors + Hober Malloc vs strong heuristics
    opp_2p_cycle = ["hober_malloc", "subham_v14", "heuristic", "wd_bestv6", "hober_malloc", "subham_v14"]
    for g in range(games_2p):
        opp = opp_2p_cycle[g % len(opp_2p_cycle)]
        if opp == "hober_malloc":
            tasks.append((["hober_malloc", "hober_malloc"], seed + g, [0, 1], sample_stride, fast_teacher))
        elif g % 2 == 0:
            tasks.append((["hober_malloc", opp], seed + g, [0], sample_stride, fast_teacher))
        else:
            tasks.append(([opp, "hober_malloc"], seed + g, [1], sample_stride, fast_teacher))

    # Diverse 4P FFA tables with 1-2 Hober Malloc bots vs strong heuristic/CMA-ES bots
    for g in range(games_4p):
        s = seed + 5000 + g
        if g % 2 == 0:
            # 2 Hober Malloc + Heuristic v2 + subham_v14 (collect both Hober perspectives)
            roster = ["hober_malloc", "subham_v14", "hober_malloc", "heuristic"]
            shift = g % 4
            roster = roster[shift:] + roster[:shift]
            t_slots = [i for i, b in enumerate(roster) if b == "hober_malloc"]
        else:
            # 1 Hober Malloc vs 3 diverse strong opponents
            roster = ["hober_malloc", "subham_v14", "heuristic", "wd_bestv6"]
            shift = g % 4
            roster = roster[shift:] + roster[:shift]
            t_slots = [i for i, b in enumerate(roster) if b == "hober_malloc"]
        tasks.append((roster, s, t_slots, sample_stride, fast_teacher))

    print(
        f"[Collect] Generating {games_2p} 2P + {games_4p} 4P Hober Malloc teacher games "
        f"(workers={workers}, fast_teacher={fast_teacher})...",
        flush=True,
    )
    t0 = time.time()
    parts: List[Dict[str, Any]] = []
    tot_matched = 0
    tot_moves = 0

    if workers <= 1:
        for idx, task in enumerate(tasks, 1):
            res = _collect_one_game(task)
            parts.append(res)
            tot_matched += res["matched_moves"]
            tot_moves += res["total_moves"]
            if idx % 5 == 0 or idx == len(tasks):
                print(
                    f"  collected {idx}/{len(tasks)} games ({time.time() - t0:.1f}s) | "
                    f"move match={100*tot_matched/max(1,tot_moves):.1f}%",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_collect_one_game, t) for t in tasks]
            for idx, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                parts.append(res)
                tot_matched += res["matched_moves"]
                tot_moves += res["total_moves"]
                if idx % 5 == 0 or idx == len(tasks):
                    print(
                        f"  collected {idx}/{len(tasks)} games ({time.time() - t0:.1f}s) | "
                        f"move match={100*tot_matched/max(1,tot_moves):.1f}%",
                        flush=True,
                    )

    keys = [
        "planet_feats",
        "timeline_feats",
        "edge_feats",
        "legal_edges",
        "owned_mask",
        "pad_mask",
        "target_cls",
        "target_frac",
        "val_targets",
    ]
    data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    n_samples = len(data["val_targets"])
    n_owned = int(data["owned_mask"].sum())
    n_launch = int(((data["target_cls"] > 0) & data["owned_mask"]).sum())
    print(
        f"[Collect] Done in {time.time() - t0:.1f}s: {n_samples} board states, "
        f"{n_owned} owned-planet decisions ({n_launch} launches = {100*n_launch/max(1,n_owned):.1f}%), "
        f"replay target recovery={100*tot_matched/max(1,tot_moves):.1f}%.",
        flush=True,
    )
    return data


def _compute_batch_loss(
    model: PlanetTimelineTransformer,
    pf: torch.Tensor,
    tf: torch.Tensor,
    ef: torch.Tensor,
    le: torch.Tensor,
    om: torch.Tensor,
    pm: torch.Tensor,
    tcls: torch.Tensor,
    tfrac: torch.Tensor,
    vtar: torch.Tensor,
    launch_weight: float = 1.0,
    value_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits, fracs, val = model(pf, tf, ef, pm)  # logits: (B, N, N+1)
    B, N, _ = logits.shape
    hold_ok = torch.ones((B, N, 1), dtype=torch.bool, device=logits.device)
    act_mask = torch.cat([hold_ok, le], dim=-1)  # (B, N, N+1)
    logits = logits.masked_fill(~act_mask, -1e9)

    # Masked per-planet cross-entropy over owned planets that have >= 1 legal route
    has_route = le.any(dim=-1)
    valid_dec = om & has_route  # (B, N)
    if valid_dec.any():
        flat_logits = logits[valid_dec]  # (M, N+1)
        flat_tcls = tcls[valid_dec]  # (M,)
        per_loss = F.cross_entropy(flat_logits, flat_tcls, reduction="none")
        weights = torch.where(flat_tcls > 0, launch_weight, 1.0)
        policy_loss = (per_loss * weights).sum() / weights.sum().clamp_min(1.0)

        preds = flat_logits.argmax(dim=-1)
        acc_all = float((preds == flat_tcls).float().mean().item())
        launch_mask = flat_tcls > 0
        if launch_mask.any():
            acc_launch = float((preds[launch_mask] == flat_tcls[launch_mask]).float().mean().item())
            # Target accuracy conditioned on launching (argmax over 1..N)
            tgt_preds = 1 + flat_logits[launch_mask, 1:].argmax(dim=-1)
            acc_target = float((tgt_preds == flat_tcls[launch_mask]).float().mean().item())
        else:
            acc_launch = 0.0
            acc_target = 0.0
    else:
        policy_loss = logits.sum() * 0.0
        acc_all = acc_launch = acc_target = 0.0

    # Fraction regression loss on actual launch pairs
    launch_dec = valid_dec & (tcls > 0)
    if launch_dec.any():
        dst_idx = (tcls[launch_dec] - 1).clamp(min=0, max=N - 1)
        pred_f = fracs[launch_dec].gather(1, dst_idx.unsqueeze(1)).squeeze(1)
        frac_loss = F.smooth_l1_loss(pred_f, tfrac[launch_dec])
    else:
        frac_loss = fracs.sum() * 0.0

    val_loss = F.mse_loss(val, vtar)
    val_sign_acc = float(((val >= 0) == (vtar >= 0)).float().mean().item())
    total = policy_loss + 0.5 * frac_loss + value_weight * val_loss
    return total, {
        "loss": float(total.item()),
        "pol_loss": float(policy_loss.item()),
        "val_loss": float(val_loss.item()),
        "acc_all": acc_all,
        "acc_launch": acc_launch,
        "acc_target": acc_target,
        "val_sign": val_sign_acc,
    }


def train_model(
    data: Dict[str, np.ndarray],
    *,
    epochs: int = 24,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    dim_ff: int = 256,
    out_path: Path,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = len(data["val_targets"])
    perm = np.random.permutation(n)
    n_val = max(1, int(0.12 * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    def make_ds(indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(data["planet_feats"][indices]),
            torch.from_numpy(data["timeline_feats"][indices]),
            torch.from_numpy(data["edge_feats"][indices]),
            torch.from_numpy(data["legal_edges"][indices]),
            torch.from_numpy(data["owned_mask"][indices]),
            torch.from_numpy(data["pad_mask"][indices]),
            torch.from_numpy(data["target_cls"][indices]),
            torch.from_numpy(data["target_frac"][indices]),
            torch.from_numpy(data["val_targets"][indices]),
        )

    tr_loader = DataLoader(make_ds(tr_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(make_ds(val_idx), batch_size=batch_size, shuffle=False, drop_last=False)

    model = PlanetTimelineTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dim_ff=dim_ff
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Train] PlanetTimelineTransformer ({n_params/1e3:.1f}K params) | "
        f"train={len(tr_idx)} val={len(val_idx)} | epochs={epochs} batch={batch_size} lr={lr}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)

    best_val_loss = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_metrics: List[Dict[str, float]] = []
        for batch in tr_loader:
            pf, tf, ef, le, om, pm, tcls, tfrac, vtar = batch
            # Exact 8x D4 dihedral symmetry augmentation on every batch!
            k_rot = random.randrange(4)
            flip_x = bool(random.randrange(2))
            pf_aug = torch.from_numpy(apply_d4_planet_feats(pf.numpy(), k_rot=k_rot, flip_x=flip_x))

            opt.zero_grad(set_to_none=True)
            loss, m = _compute_batch_loss(model, pf_aug, tf, ef, le, om, pm, tcls, tfrac, vtar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_metrics.append(m)
        sched.step()

        model.eval()
        val_metrics: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                pf, tf, ef, le, om, pm, tcls, tfrac, vtar = batch
                _, m = _compute_batch_loss(model, pf, tf, ef, le, om, pm, tcls, tfrac, vtar)
                val_metrics.append(m)

        def avg(lst: List[Dict[str, float]], key: str) -> float:
            return float(np.mean([x[key] for x in lst])) if lst else 0.0

        v_pol = avg(val_metrics, "pol_loss")
        v_loss = avg(val_metrics, "loss")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(
                {
                    "model_config": model.config,
                    "model_state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_loss": v_loss,
                    "val_pol_loss": v_pol,
                    "val_acc_all": avg(val_metrics, "acc_all"),
                    "val_acc_target": avg(val_metrics, "acc_target"),
                    "val_sign": avg(val_metrics, "val_sign"),
                },
                out_path,
            )

        if ep == 1 or ep % 4 == 0 or ep == epochs:
            print(
                f"  ep {ep:2d}/{epochs} ({time.time()-t0:5.1f}s) | "
                f"tr_pol={avg(tr_metrics,'pol_loss'):.3f} val_pol={v_pol:.3f} | "
                f"val_all={100*avg(val_metrics,'acc_all'):5.1f}% "
                f"val_launch={100*avg(val_metrics,'acc_launch'):5.1f}% "
                f"val_tgt={100*avg(val_metrics,'acc_target'):5.1f}% "
                f"val_vsign={100*avg(val_metrics,'val_sign'):5.1f}%",
                flush=True,
            )

    print(f"[Train] Saved best checkpoint -> {out_path} ({out_path.stat().st_size/1024:.1f} KB)", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PlanetTimelineTransformer on Hober Malloc replays.")
    parser.add_argument("--games-2p", type=int, default=20, help="Number of 2P teacher games to collect.")
    parser.add_argument("--games-4p", type=int, default=8, help="Number of 4P FFA teacher games to collect.")
    parser.add_argument("--sample-stride", type=int, default=2, help="Stride for quiet (non-launch) turns.")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--dataset-cache", default=None, help="Optional .npz cache path to save/load dataset.")
    parser.add_argument("--append-cache", action="store_true", help="Collect new games and append to existing cache.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect/cache dataset without training.")
    parser.add_argument("--full-tta-teacher", action="store_true", help="Keep Hober Malloc 8x JAX TTA enabled (~5x slower).")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--out", default="checkpoints/planet_transformer_best.pt")
    parser.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args(argv)

    cache = Path(args.dataset_cache) if args.dataset_cache else None
    if cache is not None and cache.is_file() and not args.append_cache:
        print(f"[Collect] Loading cached dataset from {cache}...", flush=True)
        with np.load(cache) as npz:
            data = {k: npz[k] for k in npz.files}
    else:
        new_data = collect_dataset(
            games_2p=args.games_2p,
            games_4p=args.games_4p,
            workers=args.workers,
            seed=args.seed,
            sample_stride=args.sample_stride,
            fast_teacher=not args.full_tta_teacher,
        )
        if cache is not None and cache.is_file() and args.append_cache:
            with np.load(cache) as npz:
                prev = {k: npz[k] for k in npz.files}
            data = {k: np.concatenate([prev[k], new_data[k]], axis=0) for k in new_data}
            print(f"[Collect] Appended to cache -> {len(data['val_targets'])} total samples.", flush=True)
        else:
            data = new_data
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **data)
            print(f"[Collect] Cached dataset -> {cache} ({cache.stat().st_size/1e6:.2f} MB)", flush=True)

    if args.collect_only:
        return 0

    train_model(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        out_path=Path(args.out),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
