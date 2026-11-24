#!/usr/bin/env python3
"""Stage 1 Gen 10 Self-Distillation: scale PT-Gen9 (Ens-2x) into a fresh 4.35M Transformer on DirectML.

Workflow:
  1. Collects diverse 2P FastOrbitSim self-play & mixed-opponent games across CPU workers.
  2. Labels every recorded board with exact 8x D4 TTA ensemble soft action probabilities
     and blended ensemble+outcome values batched on the DirectML/CUDA/CPU device.
  3. Trains a fresh ~4.35M PlanetTimelineTransformer (d_model=256, n_heads=8, n_layers=7,
     dim_ff=640, use_aux_future=True) using zero-stall DMLAdamW Soft-KD + t+10 world-model loss.
  4. Runs paired-seed deterministic sanity evaluation against the Gen 9 teachers.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from distill_soft_kd import train_soft_kd  # noqa: E402
from orbit_wars_bot.dml_utils import resolve_train_device  # noqa: E402
from orbit_wars_bot.fast_sim import FastOrbitSim  # noqa: E402
from orbit_wars_bot.heuristic_bot import heuristic_moves  # noqa: E402
from orbit_wars_bot.planet_transformer import (  # noqa: E402
    MAX_PLANETS,
    PlanetTimelineTransformer,
    apply_d4_planet_feats,
    extract_planet_features,
    pad_encoded_state,
)
from rl_finetune_transformer import _eval_vs_anchors_parallel, _policy_step  # noqa: E402

_WORKER_MODELS: Dict[str, PlanetTimelineTransformer] = {}


def _get_cpu_model(ckpt_path: str) -> PlanetTimelineTransformer:
    m = _WORKER_MODELS.get(ckpt_path)
    if m is None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {})
        m = PlanetTimelineTransformer(**cfg)
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        m.eval()
        _WORKER_MODELS[ckpt_path] = m
    return m


def _collect_game_worker(
    task: Tuple[int, Optional[str], Optional[str], float, float, int, int, int]
) -> Dict[str, np.ndarray]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    seed, path0, path1, temp0, temp1, sample_stride, max_steps, future_horizon = task
    m0 = _get_cpu_model(path0) if path0 else None
    m1 = _get_cpu_model(path1) if path1 else None
    models = [m0, m1]
    temps = [temp0, temp1]

    rng = np.random.default_rng(seed)
    sim = FastOrbitSim.from_seed(seed=seed, num_agents=2, max_steps=max_steps)

    rec_pf: List[List[np.ndarray]] = [[], []]
    rec_tf: List[List[np.ndarray]] = [[], []]
    rec_ef: List[List[np.ndarray]] = [[], []]
    rec_le: List[List[np.ndarray]] = [[], []]
    rec_om: List[List[np.ndarray]] = [[], []]
    rec_pm: List[List[np.ndarray]] = [[], []]
    rec_steps: List[List[int]] = [[], []]
    rec_pids: List[List[List[int]]] = [[], []]
    planet_snaps: Dict[int, Dict[int, Tuple[int, int]]] = {}
    early_winner: Optional[int] = None

    while not sim.done:
        t = int(sim.step_count)
        planet_snaps[t] = {int(p[0]): (int(p[1]), int(p[5])) for p in sim.planets}
        has_comets = bool(sim.comets)
        keep_tick = (22 <= t <= 80) or has_comets or (t % sample_stride == 0)

        acts: List[List[List[float]]] = [[], []]
        for pid in range(2):
            m = models[pid]
            if m is None:
                acts[pid] = heuristic_moves(sim, pid)
                continue
            obs = sim.to_observation(player=pid)
            enc = extract_planet_features(obs, player=pid, max_speed=sim.ship_speed)
            if not np.any(enc.owned_mask):
                continue
            tp = temps[pid]
            mv, _, _, _ = _policy_step(
                m, enc, sample=(tp > 1e-4), temperature=(tp if tp > 1e-4 else 1.0), force_all_in=True, rng=rng
            )
            acts[pid] = mv
            if keep_tick:
                pad = pad_encoded_state(enc, MAX_PLANETS)
                rec_pf[pid].append(pad["planet_feats"].astype(np.float16))
                rec_tf[pid].append(pad["timeline_feats"].astype(np.float16))
                rec_ef[pid].append(pad["edge_feats"].astype(np.float16))
                rec_le[pid].append(pad["legal_edges"])
                rec_om[pid].append(pad["owned_mask"])
                rec_pm[pid].append(pad["pad_mask"])
                rec_steps[pid].append(t)
                rec_pids[pid].append(list(enc.planet_ids))

        sim.step(acts)
        if not sim.done and sim.step_count >= 12:
            totals = sim.scores()
            p_counts = [sim.planet_count(0), sim.planet_count(1)]
            alive = [p for p, c in enumerate(p_counts) if c > 0]
            if len(alive) == 1:
                sole = alive[0]
                if totals[sole] >= max(totals) and sum(totals) > 0:
                    early_winner = sole
                    break

    planet_snaps[int(sim.step_count)] = {int(p[0]): (int(p[1]), int(p[5])) for p in sim.planets}
    totals = sim.scores()
    if early_winner is not None:
        win_pid = early_winner
    elif sim.done:
        win_pid = sim.winner()
    else:
        win_pid = 0 if totals[0] > totals[1] else (1 if totals[1] > totals[0] else -1)

    pf_all, tf_all, ef_all, le_all, om_all, pm_all = [], [], [], [], [], []
    vt_all, ao_all, ash_all, av_all = [], [], [], []

    for pid in range(2):
        if models[pid] is None or not rec_steps[pid]:
            continue
        v_target = 1.0 if win_pid == pid else (0.0 if win_pid < 0 else -1.0)
        for idx_s, step_t in enumerate(rec_steps[pid]):
            ao = np.full((MAX_PLANETS,), 2, dtype=np.int64)
            ash = np.zeros((MAX_PLANETS,), dtype=np.float32)
            av = np.zeros((MAX_PLANETS,), dtype=bool)
            fut_map = planet_snaps.get(step_t + future_horizon)
            if fut_map is not None:
                for i_p, p_id in enumerate(rec_pids[pid][idx_s]):
                    rec = fut_map.get(int(p_id))
                    if rec is None:
                        continue
                    ow, sh = rec
                    ao[i_p] = 0 if ow == pid else (2 if ow < 0 else 1)
                    ash[i_p] = float(np.log1p(max(0, sh)) / 6.0)
                    av[i_p] = True

            pf_all.append(rec_pf[pid][idx_s])
            tf_all.append(rec_tf[pid][idx_s])
            ef_all.append(rec_ef[pid][idx_s])
            le_all.append(rec_le[pid][idx_s])
            om_all.append(rec_om[pid][idx_s])
            pm_all.append(rec_pm[pid][idx_s])
            vt_all.append(v_target)
            ao_all.append(ao)
            ash_all.append(ash)
            av_all.append(av)

    if not vt_all:
        return {
            "planet_feats": np.zeros((0, MAX_PLANETS, 22), np.float16),
            "timeline_feats": np.zeros((0, MAX_PLANETS, 20, 8), np.float16),
            "edge_feats": np.zeros((0, MAX_PLANETS, MAX_PLANETS, 12), np.float16),
            "legal_edges": np.zeros((0, MAX_PLANETS, MAX_PLANETS), bool),
            "owned_mask": np.zeros((0, MAX_PLANETS), bool),
            "pad_mask": np.zeros((0, MAX_PLANETS), bool),
            "val_targets": np.zeros((0,), np.float32),
            "aux_owner": np.zeros((0, MAX_PLANETS), np.int64),
            "aux_ships": np.zeros((0, MAX_PLANETS), np.float32),
            "aux_valid": np.zeros((0, MAX_PLANETS), bool),
        }

    return {
        "planet_feats": np.stack(pf_all, axis=0),
        "timeline_feats": np.stack(tf_all, axis=0),
        "edge_feats": np.stack(ef_all, axis=0),
        "legal_edges": np.stack(le_all, axis=0),
        "owned_mask": np.stack(om_all, axis=0),
        "pad_mask": np.stack(pm_all, axis=0),
        "val_targets": np.asarray(vt_all, dtype=np.float32),
        "aux_owner": np.stack(ao_all, axis=0),
        "aux_ships": np.stack(ash_all, axis=0),
        "aux_valid": np.stack(av_all, axis=0),
    }


def collect_unlabeled_boards(
    *,
    teacher_paths: Sequence[str],
    extra_opp_path: Optional[str] = None,
    games: int = 120,
    sample_stride: int = 2,
    max_steps: int = 260,
    future_horizon: int = 10,
    workers: int = 8,
    seed: int = 20000,
) -> Dict[str, np.ndarray]:
    """Collect diverse 2P FastOrbitSim boards with exact t+10 world-model targets."""
    t0_path = teacher_paths[0]
    t1_path = teacher_paths[1] if len(teacher_paths) > 1 else teacher_paths[0]
    tasks: List[Tuple[int, Optional[str], Optional[str], float, float, int, int, int]] = []

    for g in range(games):
        m = g % 20
        s = seed + g
        if m < 9:  # 45% Gen9-A vs Gen9-B
            p0, p1 = (t0_path, t1_path) if (g % 2 == 0) else (t1_path, t0_path)
            t0_val, t1_val = 0.40, 0.40
        elif m < 13:  # 20% Gen9-A mirror
            p0, p1, t0_val, t1_val = t0_path, t0_path, 0.55, 0.55
        elif m < 17:  # 20% Gen9-B mirror
            p0, p1, t0_val, t1_val = t1_path, t1_path, 0.55, 0.55
        elif m < 19:  # 10% vs extra checkpoint (e.g. 4P specialist)
            ta = teacher_paths[g % len(teacher_paths)]
            tb = extra_opp_path if extra_opp_path else t0_path
            p0, p1 = (ta, tb) if (g % 2 == 0) else (tb, ta)
            t0_val, t1_val = 0.40, 0.40
        else:  # 5% vs heuristic
            p0, p1 = (t0_path, None) if (g % 2 == 0) else (None, t0_path)
            t0_val, t1_val = 0.40, 0.40
        tasks.append((s, p0, p1, t0_val, t1_val, sample_stride, max_steps, future_horizon))

    t0 = time.time()
    print(f"[Ensemble4M Collect] Running {games} diverse 2P games across {workers} workers...", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_collect_game_worker, tasks))
    else:
        parts = [_collect_game_worker(t) for t in tasks]

    parts = [p for p in parts if len(p["val_targets"]) > 0]
    keys = list(parts[0].keys())
    data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    print(
        f"[Ensemble4M Collect] Collected {len(data['val_targets'])} boards in {time.time()-t0:.1f}s.",
        flush=True,
    )
    return data


@torch.no_grad()
def label_boards_with_ensemble_gpu(
    data: Dict[str, np.ndarray],
    teacher_models: Sequence[Tuple[PlanetTimelineTransformer, float]],
    *,
    device: torch.device,
    batch_size: int = 64,
    val_blend_outcome: float = 0.50,
) -> Dict[str, np.ndarray]:
    """Label boards with exact 8x D4 TTA ensemble soft probabilities and blended values on GPU."""
    tot_w = sum(max(0.0, float(w)) for _, w in teacher_models)
    if tot_w <= 0.0:
        raise ValueError("Teacher weights must sum to a positive value.")
    norm_teachers = [(m.eval(), float(w) / tot_w) for m, w in teacher_models]

    n_total = len(data["val_targets"])
    soft_probs = np.zeros((n_total, MAX_PLANETS, MAX_PLANETS + 1), dtype=np.float16)
    soft_probs[..., 0] = 1.0
    blended_val = np.zeros((n_total,), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_total, batch_size):
        end = min(n_total, start + batch_size)
        b_sz = end - start
        pm_np = data["pad_mask"][start:end]
        max_n = max(2, int(pm_np.sum(axis=1).max()))

        pf_base = data["planet_feats"][start:end, :max_n].astype(np.float32)
        tf_d = torch.from_numpy(np.ascontiguousarray(data["timeline_feats"][start:end, :max_n].astype(np.float32))).to(device)
        ef_d = torch.from_numpy(np.ascontiguousarray(data["edge_feats"][start:end, :max_n, :max_n].astype(np.float32))).to(device)
        le_d = torch.from_numpy(np.ascontiguousarray(data["legal_edges"][start:end, :max_n, :max_n])).to(device)
        pm_d = torch.from_numpy(np.ascontiguousarray(pm_np[:, :max_n])).to(device)

        hold_ok = torch.ones((b_sz, max_n, 1), dtype=torch.bool, device=device)
        act_mask = torch.cat([hold_ok, le_d], dim=-1)
        probs_acc = torch.zeros((b_sz, max_n, max_n + 1), dtype=torch.float32, device=device)
        val_acc = torch.zeros((b_sz,), dtype=torch.float32, device=device)

        for k_rot in range(4):
            for flip_x in (False, True):
                pf_sym = (
                    apply_d4_planet_feats(pf_base, k_rot=k_rot, flip_x=flip_x)
                    if (k_rot != 0 or flip_x)
                    else pf_base
                )
                pf_d = torch.from_numpy(np.ascontiguousarray(pf_sym)).to(device)
                for model, w in norm_teachers:
                    logits, _, val = model(pf_d, tf_d, ef_d, pm_d)
                    logits = logits.masked_fill(~act_mask, -1e9)
                    probs_acc = probs_acc + (w / 8.0) * F.softmax(logits, dim=-1)
                    val_acc = val_acc + (w / 8.0) * val

        soft_probs[start:end, :max_n, : max_n + 1] = probs_acc.cpu().numpy().astype(np.float16)
        val_np = val_acc.cpu().numpy()
        outcome_np = data["val_targets"][start:end].astype(np.float32)
        blended_val[start:end] = (1.0 - val_blend_outcome) * val_np + val_blend_outcome * outcome_np

    out = dict(data)
    out["soft_probs"] = soft_probs
    out["val_targets"] = blended_val
    print(f"[Ensemble4M Label] Labeled {n_total} boards (8x D4 TTA x {len(norm_teachers)} teachers) in {time.time()-t0:.1f}s.", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distill PT-Gen9 Ensemble into a 4.35M PlanetTimelineTransformer.")
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=[
            "checkpoints/planet_transformer_gen9_balanced.pt",
            "checkpoints/planet_transformer_gen9_softkd.pt",
        ],
    )
    parser.add_argument("--weights", nargs="*", type=float, default=[0.55, 0.45])
    parser.add_argument("--extra-opp", default="checkpoints/planet_transformer_4p_best.pt")
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-threads", type=int, default=8)
    parser.add_argument("--label-batch-size", type=int, default=64)
    parser.add_argument("--dataset-cache", default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=7)
    parser.add_argument("--dim-ff", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument("--launch-boost", type=float, default=1.05)
    parser.add_argument("--midgame-weight", type=float, default=2.0)
    parser.add_argument("--comet-weight", type=float, default=2.0)
    parser.add_argument("--hold-boost", type=float, default=1.0)
    parser.add_argument("--eval-pairs", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "dml", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--out", default="checkpoints/planet_transformer_gen10_4m_base.pt")
    args = parser.parse_args(argv)

    train_dev, use_dml = resolve_train_device(args.device)
    dev_tag = "dml" if use_dml else str(train_dev)
    print(f"[Ensemble4M] Device={dev_tag} | teachers={args.teachers} weights={args.weights}", flush=True)

    cache_p = Path(args.dataset_cache) if args.dataset_cache else None
    if cache_p is not None and cache_p.is_file():
        print(f"[Ensemble4M] Loading cached labeled dataset from {cache_p}...", flush=True)
        with np.load(cache_p) as npz:
            labeled = {k: npz[k] for k in npz.files}
    else:
        extra_opp = args.extra_opp if (args.extra_opp and Path(args.extra_opp).is_file()) else None
        raw = collect_unlabeled_boards(
            teacher_paths=args.teachers,
            extra_opp_path=extra_opp,
            games=args.games,
            sample_stride=args.sample_stride,
            max_steps=args.max_steps,
            workers=args.workers,
            seed=args.seed,
        )
        t_weights = args.weights if len(args.weights) == len(args.teachers) else [1.0] * len(args.teachers)
        gpu_teachers: List[Tuple[PlanetTimelineTransformer, float]] = []
        for tp, w in zip(args.teachers, t_weights):
            m = _get_cpu_model(tp).to(train_dev).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            gpu_teachers.append((m, w))

        labeled = label_boards_with_ensemble_gpu(
            raw,
            gpu_teachers,
            device=train_dev,
            batch_size=args.label_batch_size,
        )
        del gpu_teachers, raw
        if cache_p is not None:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_p, **labeled)
            print(f"[Ensemble4M] Saved labeled dataset cache -> {cache_p}", flush=True)

    out_path = Path(args.out)
    train_soft_kd(
        labeled,
        resume_from=None,
        out_path=out_path,
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
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
    )

    if args.eval_pairs > 0:
        student = _get_cpu_model(str(out_path))
        eval_anchors = [
            (Path(tp).stem.replace("planet_transformer_", "")[:14], tp, _get_cpu_model(tp))
            for tp in args.teachers
            if Path(tp).is_file()
        ]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool_exec:
                ev = _eval_vs_anchors_parallel(
                    cpu_policy=student,
                    policy_path=str(out_path),
                    policy_version=1,
                    eval_anchors=eval_anchors,
                    pool_exec=pool_exec,
                    seed_base=args.seed + 70000,
                    eval_pairs=args.eval_pairs,
                    two_player_only=True,
                    max_steps=max(240, args.max_steps),
                    force_all_in=True,
                )
        else:
            ev = _eval_vs_anchors_parallel(
                cpu_policy=student,
                policy_path=None,
                policy_version=1,
                eval_anchors=eval_anchors,
                pool_exec=None,
                seed_base=args.seed + 70000,
                eval_pairs=args.eval_pairs,
                two_player_only=True,
                max_steps=max(240, args.max_steps),
                force_all_in=True,
            )
        tag_str = " ".join(f"{t}={100*v:.1f}%" for t, v in ev["per_tag_2p"].items())
        print(
            f"[Ensemble4M Eval({ev['total_games']}g)] {tag_str} avg={100*ev['avg_2p']:.1f}%",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
