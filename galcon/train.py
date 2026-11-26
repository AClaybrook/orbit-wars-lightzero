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
import atexit
from concurrent.futures import ProcessPoolExecutor
import copy
import math
import multiprocessing as mp
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset


class _LengthBucketBatchSampler(Sampler[List[int]]):
    """Groups samples with similar planet counts ``N`` into minibatches so ``_trim_batch_n`` avoids O(N^2) padding waste."""

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.lengths = [int(x) for x in lengths]
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed)

    def __iter__(self):
        buckets: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for idx, n_pl in enumerate(self.lengths):
            if n_pl <= 14:
                buckets[0].append(idx)
            elif n_pl <= 18:
                buckets[1].append(idx)
            else:
                buckets[2].append(idx)
        batches: List[List[int]] = []
        for bk in (0, 1, 2):
            idxs = buckets[bk]
            if not idxs:
                continue
            if self.shuffle:
                self._rng.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                batches.append(idxs[s : s + self.batch_size])
        if self.shuffle:
            self._rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        cnt = 0
        b0 = sum(1 for n_pl in self.lengths if n_pl <= 14)
        b1 = sum(1 for n_pl in self.lengths if 14 < n_pl <= 18)
        b2 = len(self.lengths) - b0 - b1
        for sz in (b0, b1, b2):
            if sz > 0:
                cnt += (sz + self.batch_size - 1) // self.batch_size
        return cnt


_WORKER_SNAP_CACHE: Dict[str, object] = {}
_WORKER_LUA_POOLS: Dict[Tuple[str, int], List[object]] = {}
_WORKER_ATEXIT_REGISTERED = False


def _close_worker_lua_pools() -> None:
    for pool in list(_WORKER_LUA_POOLS.values()):
        for lb in pool:
            try:
                lb.close()
            except Exception:
                pass
    _WORKER_LUA_POOLS.clear()


def _get_worker_lua_pool(lua_spec: str, count: int) -> List[object]:
    """Return persistent per-worker ``GalconLuaBotBridge`` instances to avoid respawning lua5.3 every iter."""
    global _WORKER_ATEXIT_REGISTERED
    if not lua_spec or count <= 0:
        return []
    key = (str(lua_spec), int(count))
    pool = _WORKER_LUA_POOLS.get(key)
    if pool is not None:
        return pool
    from galcon.lua_bridge import GalconLuaBotBridge

    if not _WORKER_ATEXIT_REGISTERED:
        atexit.register(_close_worker_lua_pools)
        _WORKER_ATEXIT_REGISTERED = True
    pool = [GalconLuaBotBridge(lua_spec) for _ in range(int(count))]
    _WORKER_LUA_POOLS[key] = pool
    return pool


def _get_worker_cached_snap(ckpt_str: str) -> object:
    """Cache immutable snapshot checkpoints inside persistent rollout/validation workers."""
    m = _WORKER_SNAP_CACHE.get(ckpt_str)
    if m is not None:
        return m
    m = _load_gen4_policy_from_ckpt(Path(ckpt_str), allow_non_ms=True).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    _WORKER_SNAP_CACHE[ckpt_str] = m
    return m

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
    EnsembleGalconSearchAgent,
    GalconSearchTransformerAgent,
    GalconTransformerAgent,
    apply_galcon_d4,
    build_galcon_gen2_transformer,
    build_galcon_gen4_transformer,
    build_galcon_transformer,
    canonicalize_galcon_d4,
    compute_smart_redirects,
    extract_galcon_features,
    extract_galcon_redir_features,
    pad_galcon_state,
)

MAX_REDIR_FLEETS = 8


class DMLAdamW(torch.optim.Optimizer):
    """DirectML-native decoupled AdamW avoiding ``aten::lerp_`` CPU fallback warnings."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                m, v = st["m"], st["v"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                m.mul_(b1).add_(grad, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                bc1 = 1.0 - b1 ** st["step"]
                bc2 = 1.0 - b2 ** st["step"]
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


def resolve_train_device(device_arg: str = "cpu") -> Tuple[torch.device, bool]:
    """Resolve training device (``cpu``, ``dml``, ``cuda``, or ``auto``) and return ``(device, is_dml)``."""
    req = (device_arg or "cpu").strip().lower()
    if req == "cpu":
        return torch.device("cpu"), False
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda"), False
    if req == "dml":
        import torch_directml

        if torch_directml.device_count() <= 0:
            raise RuntimeError("Requested --device dml, but no DirectML GPU was detected.")
        return torch_directml.device(), True
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), False
        try:
            import torch_directml

            if torch_directml.device_count() > 0:
                return torch_directml.device(), True
        except Exception:
            pass
        return torch.device("cpu"), False
    raise ValueError(f"Unsupported device: {device_arg!r}")


def _build_optimizer(params, *, lr: float = 1e-3, weight_decay: float = 1e-4, use_dml: bool = False):
    if use_dml:
        return DMLAdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _stable_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Numerically stable elementwise BCE-with-logits avoiding DirectML's unsupported ``aten::log_sigmoid_forward``."""
    max_val = logits.clamp_min(0.0)
    log_term = torch.log1p(torch.exp(-logits.abs()))
    if pos_weight is None:
        return max_val - logits * targets + log_term
    # Equivalent to PyTorch's pos_weight BCE: (1 - y) * x + (1 + (pw - 1) * y) * (log(1 + exp(-|x|)) + max(-x, 0))
    log_weight = (pos_weight - 1.0) * targets + 1.0
    return (1.0 - targets) * logits + log_weight * (log_term + (-logits).clamp_min(0.0))


def _bernoulli_log_prob_and_entropy(logits: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """DirectML-native Bernoulli ``(log_prob, entropy)`` without ``aten::log_sigmoid_forward`` CPU fallback."""
    softplus = logits.clamp_min(0.0) + torch.log1p(torch.exp(-logits.abs()))
    log_prob = logits * actions - softplus
    probs = torch.sigmoid(logits)
    entropy = softplus - logits * probs
    return log_prob, entropy


def _sample_map_config(rng: random.Random) -> Tuple[int, float, float]:
    """Mix compact tactical maps (8-12 even planets) and classic 14-22 even-planet Galcon maps."""
    if rng.random() < 0.40:
        return rng.randrange(8, 13, 2), 240.0, 150.0
    return rng.randrange(14, 23, 2), 400.0, 240.0


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


def _orders_to_multi_send_targets(enc, orders: Sequence[Order]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert simultaneous ``SendOrder``s into exact ``(ms_target, ms_frac)`` matrices of shape ``(MAX, MAX)``.

    Tracks sequential ship deductions on each source planet so ``ms_frac[i, j]`` records the exact
    fraction of the source planet's initial ships sent to target ``j``.
    """
    ms_tar = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
    ms_frac = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    rem_by_src = {idx: float(p.ships) for idx, p in enumerate(enc.planets)}
    orig_by_src = {idx: max(1.0, float(p.ships)) for idx, p in enumerate(enc.planets)}
    for o in orders:
        if not isinstance(o, SendOrder):
            continue
        si = id_to_idx.get(o.src_id)
        di = id_to_idx.get(o.dst_id)
        if si is not None and di is not None and enc.owned_mask[si] and enc.legal_edges[si, di]:
            rem = rem_by_src.get(si, 0.0)
            pct = max(1, min(100, int(o.percent)))
            sent = math.floor(rem * (pct / 100.0) + 1e-9)
            if sent < 1:
                continue
            rem_by_src[si] = max(0.0, rem - sent)
            ms_tar[si, di] = 1.0
            ms_frac[si, di] = min(1.0, ms_frac[si, di] + float(sent) / orig_by_src[si])
    return ms_tar, ms_frac


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
        elif t.ndim == 3 and t.shape[1] == GALCON_MAX_PLANETS and t.shape[2] == GALCON_MAX_PLANETS:
            # Pairwise multi-send targets/fractions (B, N, N)
            trimmed.append(t[:, :max_n, :max_n])
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


def _sample_gen3_map_config(rng: random.Random, large_prob: float = 0.30) -> Tuple[int, float, float, bool]:
    """Gen 3/8 multi-scale map distribution preferring symmetric even-planet 1v1 maps."""
    lp = max(0.0, min(0.80, float(large_prob)))
    u = rng.random()
    if u >= 1.0 - lp:
        # Large gbots arena (22-28 even planets, 520x320 to 600x360, 85% symmetric)
        w = rng.choice([520.0, 560.0, 600.0])
        h = round(w * 0.60, 1)
        return rng.randrange(22, 29, 2), w, h, (rng.random() < 0.85)
    # Split remaining (1 - lp) roughly 35% compact / 65% classic 14-22 even planets
    rem_u = u / max(1e-6, 1.0 - lp)
    if rem_u < 0.35:
        return rng.randrange(8, 13, 2), 240.0, 150.0, True
    return rng.randrange(14, 23, 2), 400.0, 240.0, (rng.random() < 0.90)


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
    train_threads: int = 2,
) -> Path:
    """Gen 3 Paired-Seed Mirror League PPO (GRPO-style variance reduction) + Large-Arena Generalization.

    For every sampled map ``(map_seed, n_pl, w, h, symmetric)`` and opponent, plays BOTH
    P1 and P2 sides in the same iteration and centers split-pair advantages against the paired
    baseline — eliminating map-luck / asymmetry variance while training across compact, classic,
    and large 22-28 planet asymmetric ``gbots`` arenas against Gen 2/Gen 3 snapshots, heuristics,
    and optionally an external Galcon 2 Lua boss (e.g. ``esparano``'s ``Sparky v1.6``).
    """
    torch.set_num_threads(max(1, int(train_threads)))
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

            torch.set_num_threads(max(1, int(train_threads)))
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


# ---------------------------------------------------------------------------
# Gen 4: Native Multi-Target Pairwise Send Head + 1x Canonical D4 Flywheel
# ---------------------------------------------------------------------------


def _lua_spec_exists(spec: Optional[str | Path]) -> bool:
    if spec is None:
        return False
    raw = str(spec).strip()
    if not raw:
        return False
    path_part = raw.rsplit("::", 1)[0] if "::" in raw else raw
    return Path(path_part).is_file()


def _lua_spec_resolve(spec: Optional[str | Path]) -> str:
    if not _lua_spec_exists(spec):
        return ""
    raw = str(spec).strip()
    if "::" in raw:
        path_part, fn_part = raw.rsplit("::", 1)
        return f"{Path(path_part).resolve()}::{fn_part.strip()}"
    return str(Path(raw).resolve())


def _lua_spec_label(spec: Optional[str | Path]) -> str:
    if spec is None:
        return ""
    raw = str(spec).strip()
    if "::" in raw:
        _, fn_part = raw.rsplit("::", 1)
        return fn_part.strip() or Path(raw.rsplit("::", 1)[0]).stem
    return Path(raw).stem


def _gen4_ei_worker(
    ckpt_path_str: str,
    num_games: int,
    seed: int,
    rollout_horizon: int,
    lua_opp_str: str,
    filter_losing_policy: bool,
    lua_opp2_str: str = "",
    ckpt_path2_str: str = "",
    large_map_prob: float = 0.30,
    lua_opp3_str: str = "",
    filter_draw_policy: bool = False,
    act_every: int = 2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    return collect_gen4_multisend_ei_dataset(
        Path(ckpt_path_str),
        num_games=num_games,
        seed=seed,
        rollout_horizon=rollout_horizon,
        lua_opp_path=lua_opp_str if lua_opp_str else None,
        lua_opp2_path=lua_opp2_str if lua_opp2_str else None,
        lua_opp3_path=lua_opp3_str if lua_opp3_str else None,
        ckpt_path2=Path(ckpt_path2_str) if ckpt_path2_str else None,
        large_map_prob=large_map_prob,
        filter_losing_policy=filter_losing_policy,
        filter_draw_policy=filter_draw_policy,
        act_every=act_every,
        workers=1,
    )


def collect_gen4_multisend_ei_dataset(
    ckpt_path: Path,
    *,
    num_games: int = 56,
    seed: int = 30000,
    rollout_horizon: int = 16,
    lua_opp_path: Optional[str | Path] = None,
    lua_opp2_path: Optional[str | Path] = None,
    lua_opp3_path: Optional[str | Path] = None,
    ckpt_path2: Optional[Path] = None,
    large_map_prob: float = 0.30,
    filter_losing_policy: bool = True,
    filter_draw_policy: bool = False,
    act_every: int = 2,
    workers: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Collect winning/drawing multi-send trajectories from our Opponent-Adaptive Reactive Search.

    Supports up to 3 external Lua bosses (e.g. ``Sparky v1.6`` + ``EsparanBot v0.2`` + ``Waffle3z-GA`` via
    ``path.lua::bots_waffle3z_optimized``). When ``ckpt_path2`` is supplied, uses ``EnsembleGalconSearchAgent``
    as the teacher to distill 2-model Ensemble Search directly into a single fast student.
    When ``filter_draw_policy=True``, boss stalemate draws still supervise the value & aux heads while
    masking passive draw trajectories from policy updates.
    Set ``act_every=1`` to play full-speed 1-tick tournament cadence (recording alternating even/odd ticks).
    """
    act_every = max(1, int(act_every))
    if workers > 1 and num_games >= 2:
        n_w = min(int(workers), int(num_games))
        base = num_games // n_w
        rem = num_games % n_w
        shards = [base + (1 if i < rem else 0) for i in range(n_w)]
        t_par = time.time()
        lua_str = _lua_spec_resolve(lua_opp_path)
        lua2_str = _lua_spec_resolve(lua_opp2_path)
        lua3_str = _lua_spec_resolve(lua_opp3_path)
        ckpt2_str = str(ckpt_path2.resolve()) if (ckpt_path2 is not None and ckpt_path2.is_file()) else ""
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as ex:
            futs = [
                ex.submit(
                    _gen4_ei_worker,
                    str(ckpt_path.resolve()),
                    cnt,
                    seed + idx * 1000,
                    rollout_horizon,
                    lua_str,
                    filter_losing_policy,
                    lua2_str,
                    ckpt2_str,
                    large_map_prob,
                    lua3_str,
                    filter_draw_policy,
                    act_every,
                )
                for idx, cnt in enumerate(shards)
            ]
            results = [f.result() for f in futs]
        merged_d = _merge_datasets([r[0] for r in results])
        merged_r = _merge_datasets([r[1] for r in results])
        ens_tag = " [EnsembleTeacher]" if ckpt2_str else ""
        print(
            f"[Galcon Gen4 Multi-Send EI{ens_tag} {act_every}t x{n_w} workers] Merged {len(merged_d.get('val_targets', []))} states "
            f"+ {len(merged_r.get('planet_feats', []))} redir states across {num_games} games in {time.time()-t_par:.1f}s",
            flush=True,
        )
        return merged_d, merged_r

    rng = random.Random(seed)
    lua_opp = None
    lua_opp2 = None
    lua_opp3 = None
    if _lua_spec_exists(lua_opp_path):
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp = GalconLuaBotBridge(_lua_spec_resolve(lua_opp_path))
    if _lua_spec_exists(lua_opp2_path):
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp2 = GalconLuaBotBridge(_lua_spec_resolve(lua_opp2_path))
    if _lua_spec_exists(lua_opp3_path):
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp3 = GalconLuaBotBridge(_lua_spec_resolve(lua_opp3_path))

    if ckpt_path2 is not None and ckpt_path2.is_file():
        search_ag = EnsembleGalconSearchAgent(
            [ckpt_path, ckpt_path2],
            use_d4_canonical=True,
            use_d4_tta=False,
            rollout_horizon=rollout_horizon,
            num_plies=2,
            time_budget_sec=0.28,
            neural_value_weight=0.70,
            reactive_mid_ply=True,
            opponent_adaptive=True,
            prior_bonus=0.010,
        )
    else:
        from galcon.gen5_model import GalconGen5Transformer

        search_ag = GalconSearchTransformerAgent(
            ckpt_path,
            use_d4_canonical=True,
            rollout_horizon=rollout_horizon,
            num_plies=2,
            time_budget_sec=0.24,
            neural_value_weight=0.70,
            reactive_mid_ply=True,
            opponent_adaptive=True,
            enable_def_stream=True,
            enable_opening_def_guard=True,
            prior_bonus=0.010,
        )
        if isinstance(search_ag.model, GalconGen5Transformer):
            search_ag.neural_value_weight = 0.75
            search_ag.enable_rescue_snipe = True
            search_ag.enable_opp_direct_rush = True

    pf_l, tf_l, ef_l, le_l, om_l, pm_l = [], [], [], [], [], []
    ms_tar_l, ms_fr_l, t_msp_l, vt_l, aux_o_l, aux_s_l = [], [], [], [], [], []
    r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk = [], [], [], [], [], [], [], []
    wins_bots = draws_bots = games_bots = 0
    wins_boss = draws_boss = games_boss = 0
    multi_send_events = 0
    total_send_ticks = 0
    t0 = time.time()

    opp_cycle = ["rusher", "tactical", "classic", "macro", "rusher", "tactical"]
    bot_idx = 0
    boss_idx = 0
    boss2_idx = 0
    boss3_idx = 0

    try:
        for g in range(num_games):
            n_pl, w, h, sym = _sample_gen3_map_config(rng, large_prob=large_map_prob)
            max_t = 220 if n_pl >= 22 else 200
            sim = GalconSim.create(seed=seed + g, num_planets=n_pl, width=w, height=h, max_ticks=max_t, symmetric=sym)

            is_selfplay = False
            is_boss = False
            if lua_opp is not None and lua_opp2 is not None and lua_opp3 is not None:
                slot = g % 6
                if slot in (0, 1):
                    is_boss = True
                    boss_idx += 1
                    opp_bot = lua_opp
                    our_pid = 1 if (boss_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 2:
                    is_boss = True
                    boss3_idx += 1
                    opp_bot = lua_opp3
                    our_pid = 1 if (boss3_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 3:
                    is_boss = True
                    boss2_idx += 1
                    opp_bot = lua_opp2
                    our_pid = 1 if (boss2_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 4:
                    is_selfplay = True
                    opp_bot = None
                    our_pid = opp_pid = 0
                else:
                    opp_kind = opp_cycle[bot_idx % len(opp_cycle)]
                    bot_idx += 1
                    opp_bot = _make_sparring_bot(rng, opp_kind)
                    our_pid = 1 if (bot_idx % 2 == 0) else 2
                    opp_pid = 2 if our_pid == 1 else 1
            elif lua_opp is not None and (lua_opp2 is not None or lua_opp3 is not None):
                sec_boss = lua_opp3 if lua_opp3 is not None else lua_opp2
                slot = g % 5
                if slot in (0, 1):
                    is_boss = True
                    boss_idx += 1
                    opp_bot = lua_opp
                    our_pid = 1 if (boss_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 2:
                    is_boss = True
                    boss2_idx += 1
                    opp_bot = sec_boss
                    our_pid = 1 if (boss2_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 3:
                    is_selfplay = True
                    opp_bot = None
                    our_pid = opp_pid = 0
                else:
                    opp_kind = opp_cycle[bot_idx % len(opp_cycle)]
                    bot_idx += 1
                    opp_bot = _make_sparring_bot(rng, opp_kind)
                    our_pid = 1 if (bot_idx % 2 == 0) else 2
                    opp_pid = 2 if our_pid == 1 else 1
            elif lua_opp is not None:
                slot = g % 4
                if slot in (0, 1):
                    is_boss = True
                    boss_idx += 1
                    opp_bot = lua_opp
                    our_pid = 1 if (boss_idx % 2 == 1) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                elif slot == 2:
                    is_selfplay = True
                    opp_bot = None
                    our_pid = opp_pid = 0
                else:
                    opp_kind = opp_cycle[bot_idx % len(opp_cycle)]
                    bot_idx += 1
                    opp_bot = _make_sparring_bot(rng, opp_kind)
                    our_pid = 1 if (bot_idx % 2 == 0) else 2
                    opp_pid = 2 if our_pid == 1 else 1
            else:
                is_selfplay = (g % 3) == 0
                if not is_selfplay:
                    opp_kind = opp_cycle[bot_idx % len(opp_cycle)]
                    bot_idx += 1
                    opp_bot = _make_sparring_bot(rng, opp_kind)
                    our_pid = 1 if (bot_idx % 2 == 0) else 2
                    opp_pid = 2 if our_pid == 1 else 1
                else:
                    opp_bot = None
                    our_pid = opp_pid = 0

            traj_by_p: Dict[int, List[Tuple]] = {1: [], 2: []}
            snaps_by_p: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {1: [], 2: []}

            while not sim.done:
                if sim.tick % act_every == 0:
                    orders_map: Dict[int, List[Order]] = {}
                    search_pids = (1, 2) if is_selfplay else (our_pid,)
                    # When playing full 1-tick speed, record alternating even/odd ticks (50/50 split) to stay lean
                    rec_tick = (act_every >= 2) or (sim.tick % 2 == (g % 2))
                    for pid in search_pids:
                        enc = extract_galcon_features(sim, pid, feature_version=2)
                        so, ss = _snapshot_planet_state(sim, enc.planet_ids, pid)
                        snaps_by_p[pid].append((sim.tick, so, ss))
                        best_plan, _, _, best_sc = search_ag.act_with_search_info(sim, pid)
                        orders_map[pid] = best_plan

                        if rec_tick and np.any(enc.owned_mask):
                            # Store canonical D4 planet features so Gen 4/5 trains directly in canonical frame
                            enc_c = copy.copy(enc)
                            enc_c.planet_feats = canonicalize_galcon_d4(enc.planet_feats)
                            pad = pad_galcon_state(enc_c)
                            ms_tar, ms_fr = _orders_to_multi_send_targets(enc, best_plan)
                            t_msp_raw = getattr(search_ag, "_last_root_ms_probs", None)
                            t_val = float(getattr(search_ag, "_last_root_val", best_sc))
                            t_msp_pad = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
                            if t_msp_raw is not None:
                                tn = min(GALCON_MAX_PLANETS, t_msp_raw.shape[0])
                                t_msp_pad[:tn, :tn] = np.clip(t_msp_raw[:tn, :tn], 1e-4, 1.0 - 1e-4)
                            else:
                                t_msp_pad = ms_tar.copy()
                            row_cnts = ms_tar.sum(axis=-1)
                            if float(row_cnts.max()) >= 1.0:
                                total_send_ticks += 1
                                if float(row_cnts.max()) >= 2.0:
                                    multi_send_events += 1

                            redir_pack = None
                            if sim.fleets and (sim.tick % 6 == 0):
                                renc = extract_galcon_redir_features(sim, enc)
                                if renc.fleet_ids:
                                    _, rlabels = compute_smart_redirects(sim, enc, verify_rollout=True)
                                    redir_pack = _pad_redir_sample(enc_c, renc, rlabels)
                            traj_by_p[pid].append((sim.tick, pad, ms_tar, ms_fr, t_msp_pad, float(best_sc), t_val, redir_pack))

                    if not is_selfplay and opp_bot is not None:
                        orders_map[opp_pid] = opp_bot.act(sim, opp_pid)
                    sim.step(orders_map)
                else:
                    sim.step({})

            win = sim.winner()
            if is_boss:
                games_boss += 1
                if win == our_pid:
                    wins_boss += 1
                elif win == 0:
                    draws_boss += 1
            elif not is_selfplay:
                games_bots += 1
                if win == our_pid:
                    wins_bots += 1
                elif win == 0:
                    draws_bots += 1

            rec_pids = (1, 2) if is_selfplay else (our_pid,)
            for pid in rec_pids:
                enc_f = extract_galcon_features(sim, pid, feature_version=2)
                so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, pid)
                snaps_by_p[pid].append((sim.tick, so, ss))
                snaps = snaps_by_p[pid]
                z = 1.0 if win == pid else (0.0 if win == 0 else -1.0)
                if not filter_losing_policy:
                    keep_policy = True
                elif filter_draw_policy and is_boss:
                    keep_policy = z > 0.0
                else:
                    keep_policy = z >= 0.0
                for tick_rec, pad, ms_tar, ms_fr, t_msp_pad, best_sc, t_val, redir_pack in traj_by_p[pid]:
                    target_tick = tick_rec + 10
                    fut_o, fut_s = snaps[-1][1], snaps[-1][2]
                    for st, so_s, ss_s in snaps:
                        if st >= target_tick:
                            fut_o, fut_s = so_s, ss_s
                            break
                    pf_l.append(pad["planet_feats"])
                    tf_l.append(pad["timeline_feats"])
                    ef_l.append(pad["edge_feats"])
                    le_l.append(pad["legal_edges"])
                    om_l.append(pad["owned_mask"] if keep_policy else np.zeros_like(pad["owned_mask"]))
                    pm_l.append(pad["pad_mask"])
                    ms_tar_l.append(ms_tar)
                    ms_fr_l.append(ms_fr)
                    t_msp_l.append(t_msp_pad)
                    vt_l.append(float(np.clip(0.55 * z + 0.25 * best_sc + 0.20 * t_val, -1.0, 1.0)))
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
        if lua_opp2 is not None:
            lua_opp2.close()
        if lua_opp3 is not None:
            lua_opp3.close()

    data = {
        "planet_feats": np.stack(pf_l).astype(np.float32),
        "timeline_feats": np.stack(tf_l).astype(np.float32),
        "edge_feats": np.stack(ef_l).astype(np.float32),
        "legal_edges": np.stack(le_l),
        "owned_mask": np.stack(om_l),
        "pad_mask": np.stack(pm_l),
        "ms_target": np.stack(ms_tar_l).astype(np.float32),
        "ms_frac": np.stack(ms_fr_l).astype(np.float32),
        "teacher_ms_prob": np.stack(t_msp_l).astype(np.float32),
        "val_targets": np.array(vt_l, dtype=np.float32),
        "aux_owner": np.stack(aux_o_l).astype(np.int64),
        "aux_ships": np.stack(aux_s_l).astype(np.float32),
    }
    redir_data = _pack_redir_dict(r_pf, r_tf, r_ef, r_pm, r_cur, r_feat, r_tar, r_msk)
    boss_msg = (
        f", boss={wins_boss}-{games_boss - wins_boss - draws_boss}-{draws_boss}"
        if games_boss > 0
        else ""
    )
    print(
        f"[Galcon Gen4 Multi-Send EI] Collected {len(vt_l)} states ({multi_send_events}/{max(1, total_send_ticks)} multi-send ticks) "
        f"+ {len(r_pf)} redir states across {num_games} games "
        f"(bots={wins_bots}-{games_bots - wins_bots - draws_bots}-{draws_bots}{boss_msg}) in {time.time()-t0:.1f}s",
        flush=True,
    )
    return data, redir_data


def _gen4_multisend_loss(
    model: GalconPlanetTransformer,
    pf,
    tf,
    ef,
    le,
    om,
    pm,
    ms_tar,
    ms_fr,
    vtar,
    aux_own,
    aux_shp,
    pos_weight: float = 4.5,
    t_msp=None,
    kd_alpha: float = 0.0,
):
    if t_msp is not None:
        pf, tf, ef, le, om, pm, ms_tar, ms_fr, aux_own, aux_shp, t_msp = _trim_batch_n(
            pf, tf, ef, le, om, pm, ms_tar, ms_fr, aux_own, aux_shp, t_msp
        )
    else:
        pf, tf, ef, le, om, pm, ms_tar, ms_fr, aux_own, aux_shp = _trim_batch_n(
            pf, tf, ef, le, om, pm, ms_tar, ms_fr, aux_own, aux_shp
        )
    has_wdl = hasattr(model, "decode_wdl_from_hidden")
    if has_wdl:
        ms_logits, ms_frac_pred, val, wdl_logits, aux_pred = model.forward_multi_send(
            pf, tf, ef, pm, return_aux=True, return_wdl=True
        )
    else:
        ms_logits, ms_frac_pred, val, aux_pred = model.forward_multi_send(pf, tf, ef, pm, return_aux=True)
        wdl_logits = None

    edge_valid = om.unsqueeze(-1) & le
    ev_f = edge_valid.float()
    ev_sum = ev_f.sum()
    if float(ev_sum.item()) > 0.0:
        pw = torch.tensor(pos_weight, device=ms_logits.device, dtype=ms_logits.dtype)
        bce_all = _stable_bce_with_logits(ms_logits, ms_tar, pos_weight=pw)
        ms_bce = (bce_all * ev_f).sum() / ev_sum.clamp_min(1.0)
        if t_msp is not None and kd_alpha > 0.0:
            soft_all = _stable_bce_with_logits(ms_logits, t_msp.clamp(1e-4, 1.0 - 1e-4))
            soft_bce = (soft_all * ev_f).sum() / ev_sum.clamp_min(1.0)
            pol_loss = (1.0 - kd_alpha) * ms_bce + kd_alpha * soft_bce
        else:
            pol_loss = ms_bce
        with torch.no_grad():
            preds = (torch.sigmoid(ms_logits) >= 0.45).float() * ev_f
            pos_f = (ms_tar > 0.5).float() * ev_f
            pos_tot = float(pos_f.sum().item())
            pred_tot = float(preds.sum().item())
            tp = float((preds * pos_f).sum().item())
            rec = tp / pos_tot if pos_tot > 0.0 else 0.0
            prec = tp / pred_tot if pred_tot > 0.0 else 0.0
    else:
        ms_bce = pol_loss = ms_logits.sum() * 0.0
        rec = prec = 0.0

    pos_edges_f = (edge_valid & (ms_tar > 0.5) & (ms_fr > 0.01)).float()
    pe_sum = pos_edges_f.sum()
    if float(pe_sum.item()) > 0.0:
        frac_loss = (F.smooth_l1_loss(ms_frac_pred, ms_fr, reduction="none") * pos_edges_f).sum() / pe_sum.clamp_min(1.0)
    else:
        frac_loss = ms_frac_pred.sum() * 0.0

    if wdl_logits is not None:
        # Calibrated exact-mean-preserving quadratic soft-WDL targets:
        # t_draw = 0.65 * (1 - |v|)^2, t_win = (1 - t_draw + v)/2, t_loss = (1 - t_draw - v)/2
        # Guarantees t_win + t_draw + t_loss == 1 AND t_win - t_loss == v while avoiding bogus draw mass on wins.
        vt_c = vtar.clamp(-1.0, 1.0)
        t_draw = 0.65 * (1.0 - vt_c.abs()).pow(2)
        t_win = (1.0 - t_draw + vt_c).mul(0.5).clamp(0.0, 1.0)
        t_loss = (1.0 - t_draw - vt_c).mul(0.5).clamp(0.0, 1.0)
        t_wdl = torch.stack([t_loss, t_draw, t_win], dim=-1)
        wdl_ce = -(t_wdl * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()
        v_loss = 0.65 * wdl_ce + 0.50 * F.mse_loss(val, vt_c)
    else:
        v_loss = F.mse_loss(val, vtar)
    vsign = float(((val >= 0) == (vtar >= 0)).float().mean().item())
    pm_f = pm.float()
    pm_sum = pm_f.sum()
    if float(pm_sum.item()) > 0.0:
        aux_ce = F.cross_entropy(
            aux_pred[..., :3].reshape(-1, 3),
            aux_own.reshape(-1),
            reduction="none",
        ).reshape_as(pm_f)
        aux_own_loss = (aux_ce * pm_f).sum() / pm_sum.clamp_min(1.0)
        aux_shp_loss = (F.smooth_l1_loss(aux_pred[..., 3], aux_shp, reduction="none") * pm_f).sum() / pm_sum.clamp_min(1.0)
        aux_loss = aux_own_loss + 0.5 * aux_shp_loss
        with torch.no_grad():
            aux_acc = float(((aux_pred[..., :3].argmax(dim=-1) == aux_own).float() * pm_f).sum().item() / pm_sum.clamp_min(1.0).item())
    else:
        aux_loss = aux_pred.sum() * 0.0
        aux_acc = 0.0

    total = pol_loss + 0.85 * frac_loss + 0.35 * v_loss + 0.22 * aux_loss
    return total, {
        "bce": float(pol_loss.item()),
        "v_loss": float(v_loss.item()),
        "rec": rec,
        "prec": prec,
        "vsign": vsign,
        "aux": aux_acc,
    }


def train_gen4_multisend_supervised(
    data: Dict[str, np.ndarray],
    redir_data: Dict[str, np.ndarray],
    *,
    resume_from: Optional[Path],
    out_path: Path,
    epochs: int = 8,
    batch_size: int = 96,
    lr: float = 3.5e-4,
    kd_alpha: float = 0.30,
    arch: str = "gen4",
    seed: int = 42,
    label: str = "Gen4-MultiSend-EI",
    train_threads: int = 2,
    device: str = "cpu",
) -> Path:
    """Warm-start ``GalconGen4Transformer`` or train ``GalconGen5Transformer`` with Hybrid Search-EI + Soft-KD."""
    from galcon.gen5_model import GalconGen5Transformer, build_galcon_gen5_transformer

    torch.set_num_threads(max(1, int(train_threads)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_dev, use_dml = resolve_train_device(device)

    n = len(data["val_targets"])
    # Contiguous game-block held-out validation split (~64 states/block) to prevent adjacent-tick leakage
    if n >= 256:
        block_sz = 64
        n_blocks = (n + block_sz - 1) // block_sz
        block_perm = np.random.permutation(n_blocks)
        n_val_blocks = max(1, int(round(0.10 * n_blocks)))
        val_blocks = set(block_perm[:n_val_blocks].tolist())
        val_idx = np.array([i for i in range(n) if (i // block_sz) in val_blocks], dtype=np.int64)
        tr_idx = np.array([i for i in range(n) if (i // block_sz) not in val_blocks], dtype=np.int64)
    else:
        perm = np.random.permutation(n)
        n_val = max(1, int(0.10 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    has_kd = "teacher_ms_prob" in data and kd_alpha > 0.0

    # Zero-copy TensorDataset + Subset avoids duplicating GBs of padded edge tensors in RAM
    full_tensors = [
        torch.from_numpy(data["planet_feats"]),
        torch.from_numpy(data["timeline_feats"]),
        torch.from_numpy(data["edge_feats"]),
        torch.from_numpy(data["legal_edges"]),
        torch.from_numpy(data["owned_mask"]),
        torch.from_numpy(data["pad_mask"]),
        torch.from_numpy(data["ms_target"]),
        torch.from_numpy(data["ms_frac"]),
        torch.from_numpy(data["val_targets"]),
        torch.from_numpy(data["aux_owner"]),
        torch.from_numpy(data["aux_ships"]),
    ]
    if has_kd:
        full_tensors.append(torch.from_numpy(data["teacher_ms_prob"]))
    full_ds = TensorDataset(*full_tensors)

    tr_lens = data["pad_mask"][tr_idx].sum(axis=1)
    val_lens = data["pad_mask"][val_idx].sum(axis=1)
    tr_loader = DataLoader(
        torch.utils.data.Subset(full_ds, tr_idx.tolist()),
        batch_sampler=_LengthBucketBatchSampler(tr_lens, batch_size, shuffle=True, seed=seed),
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_ds, val_idx.tolist()),
        batch_sampler=_LengthBucketBatchSampler(val_lens, batch_size, shuffle=False, seed=seed),
    )

    redir_loader = None
    if redir_data and len(redir_data.get("planet_feats", [])) >= 16:
        n_r = len(redir_data["planet_feats"])
        r_sl = slice(max(0, n_r - 4000), n_r)
        rds = TensorDataset(
            torch.from_numpy(redir_data["planet_feats"][r_sl]),
            torch.from_numpy(redir_data["timeline_feats"][r_sl]),
            torch.from_numpy(redir_data["edge_feats"][r_sl]),
            torch.from_numpy(redir_data["pad_mask"][r_sl]),
            torch.from_numpy(redir_data["cur_dst_idx"][r_sl]),
            torch.from_numpy(redir_data["redir_feats"][r_sl]),
            torch.from_numpy(redir_data["redir_target"][r_sl]),
            torch.from_numpy(redir_data["redir_mask"][r_sl]),
        )
        redir_loader = DataLoader(rds, batch_size=32, shuffle=True)

    ckpt = torch.load(str(resume_from), map_location="cpu", weights_only=False) if (resume_from is not None and resume_from.is_file()) else None
    is_gen5_ckpt = bool(ckpt is not None and ckpt.get("model_config", {}).get("model_class") == "GalconGen5Transformer")
    use_gen5 = (arch == "gen5") or is_gen5_ckpt
    dev_tag = "dml" if use_dml else str(train_dev)
    kd_tag = f", kd_alpha={kd_alpha:.2f}" if has_kd else ""

    if use_gen5:
        label = "Gen5-EdgeGated-WDL-EI"
        if is_gen5_ckpt and ckpt is not None:
            cfg = ckpt["model_config"]
            model: nn.Module = GalconGen5Transformer(
                d_model=int(cfg.get("d_model", 192)),
                n_heads=int(cfg.get("n_heads", 6)),
                n_layers=int(cfg.get("n_layers", 6)),
                dim_ff=int(cfg.get("dim_ff", 384)),
                edge_hidden=int(cfg.get("edge_hidden", 24)),
                num_val_queries=int(cfg.get("num_val_queries", 4)),
                feature_version=int(cfg.get("feature_version", 2)),
                dropout=float(cfg.get("dropout", 0.08)),
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            src_desc = f"Warm-started Gen5 from {resume_from.name}" if resume_from is not None else "Gen5"
            param_groups = [{"params": list(model.parameters()), "lr": lr}]
        else:
            model = build_galcon_gen5_transformer(dropout=0.08)
            src_desc = f"Fresh Gen5 ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params, drop=0.08)"
            param_groups = [{"params": list(model.parameters()), "lr": lr}]
        model = model.to(train_dev)
        print(f"  [{label}] {src_desc} (device={dev_tag}{kd_tag}, block_val={len(val_idx)}/{n})", flush=True)
    else:
        assert ckpt is not None and resume_from is not None
        cfg = ckpt["model_config"]
        model = build_galcon_gen4_transformer(
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            dim_ff=int(cfg["dim_ff"]),
            feature_version=int(cfg.get("feature_version", 2)),
        )
        missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(train_dev)
        print(f"  [{label}] Warm-started from {resume_from.name} (new multi-send params: {len(missing)} tensors, device={dev_tag}{kd_tag})", flush=True)
        # Give the newly initialized multi-send head 3x higher LR than the warm-started trunk
        ms_params = [p for k, p in model.named_parameters() if k.startswith("ms_")]
        base_params = [p for k, p in model.named_parameters() if not k.startswith("ms_")]
        param_groups = [{"params": base_params, "lr": lr * 0.45}, {"params": ms_params, "lr": lr * 2.5}]

    ms_pos_weight = 2.2 if use_gen5 else 4.5
    opt = _build_optimizer(
        param_groups,
        lr=lr,
        weight_decay=(2.5e-3 if use_gen5 else 1e-4),
        use_dml=use_dml,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.08)
    best_metric = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        r_iter = iter(redir_loader) if redir_loader is not None else None
        for b_idx, batch in enumerate(tr_loader):
            bt = [t.to(train_dev) for t in batch]
            pf, tf, ef, le, om, pm, mst, msf, vt, ao, ashp = bt[:11]
            t_msp = bt[11] if has_kd else None
            opt.zero_grad(set_to_none=True)
            loss, _ = _gen4_multisend_loss(
                model,
                pf,
                tf,
                ef,
                le,
                om,
                pm,
                mst,
                msf,
                vt,
                ao,
                ashp,
                pos_weight=ms_pos_weight,
                t_msp=t_msp,
                kd_alpha=(kd_alpha if has_kd else 0.0),
            )
            if r_iter is not None and (b_idx % 4 == 0):
                try:
                    rb = next(r_iter)
                except StopIteration:
                    r_iter = iter(redir_loader)
                    rb = next(r_iter)
                rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk = (t.to(train_dev) for t in rb)
                rloss, _, _ = _redir_batch_loss(model, rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk)
                loss = loss + 0.30 * rloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=not use_dml)
            opt.step()
        sched.step()

        model.eval()
        vm = []
        with torch.no_grad():
            for batch in val_loader:
                bt = [t.to(train_dev) for t in batch]
                pf, tf, ef, le, om, pm, mst, msf, vt, ao, ashp = bt[:11]
                t_msp = bt[11] if has_kd else None
                _, m = _gen4_multisend_loss(
                    model,
                    pf,
                    tf,
                    ef,
                    le,
                    om,
                    pm,
                    mst,
                    msf,
                    vt,
                    ao,
                    ashp,
                    pos_weight=ms_pos_weight,
                    t_msp=t_msp,
                    kd_alpha=(kd_alpha if has_kd else 0.0),
                )
                vm.append(m)
        v_bce = float(np.mean([x["bce"] for x in vm]))
        v_vloss = float(np.mean([x["v_loss"] for x in vm]))
        sel_metric = (v_bce + 0.15 * v_vloss) if use_gen5 else v_bce
        if sel_metric < best_metric:
            best_metric = sel_metric
            torch.save(
                {
                    "model_config": model.config,
                    "model_state_dict": _cpu_state_dict(model),
                    "has_trained_redir": True,
                    "has_trained_multi_send": True,
                    "use_canonical_d4": True,
                    "val_ms_bce": v_bce,
                    "val_v_loss": v_vloss,
                },
                out_path,
            )
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(
                f"  [{label}] ep {ep:2d}/{epochs} ({time.time()-t0:4.1f}s) | val_bce={v_bce:.3f} vloss={v_vloss:.3f} "
                f"rec={100*np.mean([x['rec'] for x in vm]):5.1f}% prec={100*np.mean([x['prec'] for x in vm]):5.1f}% "
                f"aux={100*np.mean([x['aux'] for x in vm]):5.1f}% vsign={100*np.mean([x['vsign'] for x in vm]):5.1f}%",
                flush=True,
            )
    return out_path


def _fast_gen4_neural_act(
    model: GalconPlanetTransformer,
    sim: GalconSim,
    player: int,
    *,
    threshold: float = 0.45,
    max_sends: int = 6,
) -> List[Order]:
    """Fast 1x Canonical D4 multi-send + native /REDIR actor for Gen 4 league snapshots."""
    enc = extract_galcon_features(sim, player, feature_version=2)
    n = len(enc.planet_ids)
    if n == 0 or not np.any(enc.owned_mask):
        return []
    with torch.no_grad():
        pf_c = canonicalize_galcon_d4(enc.planet_feats)
        pf_t = torch.from_numpy(pf_c[None])
        tf_t = torch.from_numpy(enc.timeline_feats[None])
        ef_t = torch.from_numpy(enc.edge_feats[None])
        pm_t = torch.ones((1, n), dtype=torch.bool)
        orders: List[Order] = []
        if model.enable_multi_send:
            ms_l, ms_f, _, h = model.forward_multi_send(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
            le_t = torch.from_numpy(enc.legal_edges[None])
            ms_l = ms_l.masked_fill(~le_t, -1e9)
            ms_p = torch.sigmoid(ms_l[0]).numpy()
            ms_f_np = ms_f[0].numpy()
            for i in range(n):
                if not enc.owned_mask[i] or not np.any(enc.legal_edges[i]):
                    continue
                orig_s = float(enc.planets[i].ships)
                if orig_s < 2.0:
                    continue
                sel = [(float(ms_p[i, j]), j) for j in range(n) if enc.legal_edges[i, j] and float(ms_p[i, j]) >= threshold]
                if not sel:
                    continue
                sel.sort(key=lambda x: -x[0])
                rem = orig_s
                for _, j in sel[:max_sends]:
                    if rem < 2.0:
                        break
                    want = min(rem, max(1.0, float(ms_f_np[i, j]) * orig_s))
                    ratio = want / max(1.0, rem)
                    pct = 100 if ratio >= 0.92 else max(5, min(100, int(round(ratio * 100.0))))
                    sent = math.floor(rem * (pct / 100.0) + 1e-9)
                    if sent >= 1:
                        orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                        rem = max(0.0, rem - sent)
        else:
            logits, fracs, _, h = model(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
            le_t = torch.from_numpy(enc.legal_edges[None])
            mask = torch.cat([torch.ones((1, n, 1), dtype=torch.bool), le_t], dim=-1)
            probs = F.softmax(logits.masked_fill(~mask, -1e9)[0], dim=-1).numpy()
            fracs_np = fracs[0].numpy()
            for i in range(n):
                if enc.owned_mask[i] and (1.0 - float(probs[i, 0])) >= 0.50:
                    j = int(np.argmax(probs[i, 1:]))
                    if enc.legal_edges[i, j]:
                        f = float(fracs_np[i, j])
                        pct = 100 if f >= 0.90 else max(5, min(100, int(round(f * 100.0))))
                        orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))

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


def _load_gen4_policy_from_ckpt(ckpt_path: Path, *, allow_non_ms: bool = False):
    """Instantiate a ``GalconPlanetTransformer`` or ``GalconGen5Transformer`` directly from a saved checkpoint dict."""
    from galcon.gen5_model import GalconGen5Transformer

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    if cfg.get("model_class") == "GalconGen5Transformer" or any(
        k.startswith("val_wdl_head.") for k in ckpt["model_state_dict"]
    ):
        model = GalconGen5Transformer(
            d_model=int(cfg.get("d_model", 192)),
            n_heads=int(cfg.get("n_heads", 6)),
            n_layers=int(cfg.get("n_layers", 6)),
            dim_ff=int(cfg.get("dim_ff", 384)),
            edge_hidden=int(cfg.get("edge_hidden", 24)),
            num_val_queries=int(cfg.get("num_val_queries", 4)),
            feature_version=int(cfg.get("feature_version", 2)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model
    enable_ms = (
        bool(cfg.get("enable_multi_send", False) or any(k.startswith("ms_head.") for k in ckpt["model_state_dict"]))
        if allow_non_ms
        else True
    )
    model = build_galcon_gen2_transformer(
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        dim_ff=int(cfg["dim_ff"]),
        feature_version=int(cfg.get("feature_version", 2)),
        enable_multi_send=enable_ms,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _run_gen4_ppo_episode(
    policy: GalconPlanetTransformer,
    map_seed: int,
    n_pl: int,
    w: float,
    h: float,
    sym: bool,
    ours_pid: int,
    opp_snap: Optional[GalconPlanetTransformer],
    opp_bot,
    *,
    is_boss: bool,
    gamma: float,
    gae_lam: float,
    frac_std: float,
    draw_penalty: float,
    boss_draw_penalty: float,
    max_sends_per_planet: int,
    act_every: int = 2,
):
    """Run one Gen 4/5 multi-send PPO rollout episode and compute GAE advantages/returns."""
    act_every = max(1, int(act_every))
    opp_pid = 2 if ours_pid == 1 else 1
    sim = GalconSim.create(
        seed=map_seed, num_planets=n_pl, width=w, height=h, max_ticks=220 if n_pl >= 22 else 200, symmetric=sym
    )
    ep_steps = []
    snaps: List[Tuple[int, np.ndarray, np.ndarray]] = []
    prev_pot = evaluate_galcon_state(sim, ours_pid)
    while not sim.done:
        if sim.tick % act_every == 0:
            enc = extract_galcon_features(sim, ours_pid, feature_version=2)
            n = len(enc.planet_ids)
            so, ss = _snapshot_planet_state(sim, enc.planet_ids, ours_pid)
            snaps.append((sim.tick, so, ss))
            enc_c = copy.copy(enc)
            enc_c.planet_feats = canonicalize_galcon_d4(enc.planet_feats)
            pad = pad_galcon_state(enc_c)

            with torch.no_grad():
                pf_t = torch.from_numpy(enc_c.planet_feats[None])
                tf_t = torch.from_numpy(enc_c.timeline_feats[None])
                ef_t = torch.from_numpy(enc_c.edge_feats[None])
                pm_t = torch.ones((1, n), dtype=torch.bool)
                ms_l, ms_f, val, h_emb = policy.forward_multi_send(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
                le_t = torch.from_numpy(enc.legal_edges)
                om_t = torch.from_numpy(enc.owned_mask)
                edge_m = om_t.unsqueeze(-1) & le_t

                ms_l0 = ms_l[0].masked_fill(~edge_m, -1e9)
                b_dist = torch.distributions.Bernoulli(logits=ms_l0)
                a_edge = b_dist.sample() * edge_m.float()
                ms_probs_np = torch.sigmoid(ms_l0).numpy()
                a_np = a_edge.numpy()
                for i in range(n):
                    if enc.owned_mask[i]:
                        act_js = np.where(a_np[i] > 0.5)[0]
                        if len(act_js) > max_sends_per_planet:
                            keep = act_js[np.argsort(-ms_probs_np[i, act_js])[:max_sends_per_planet]]
                            a_np[i] = 0.0
                            a_np[i, keep] = 1.0
                a_edge = torch.from_numpy(a_np)
                lp_bern = (b_dist.log_prob(a_edge) * edge_m.float()).sum()

                f_mu = ms_f[0].clamp(0.06, 0.98)
                f_dist = torch.distributions.Normal(f_mu, frac_std)
                f_samp = f_dist.sample().clamp(0.06, 1.0)
                lp_frac = (f_dist.log_prob(f_samp) * a_edge).sum()
                total_lp = float((lp_bern + 0.20 * lp_frac).item())
                v_pred = float(val[0].item())

                o_ours: List[Order] = []
                f_np = f_samp.numpy()
                for i in range(n):
                    if not enc.owned_mask[i]:
                        continue
                    orig_s = float(enc.planets[i].ships)
                    if orig_s < 2.0:
                        continue
                    act_js = [int(j) for j in np.where(a_np[i] > 0.5)[0]]
                    if not act_js:
                        continue
                    act_js.sort(key=lambda j: -float(ms_probs_np[i, j]))
                    rem = orig_s
                    for j in act_js:
                        if rem < 2.0:
                            break
                        want = min(rem, max(1.0, float(f_np[i, j]) * orig_s))
                        ratio = want / max(1.0, rem)
                        pct = 100 if ratio >= 0.92 else max(5, min(100, int(round(ratio * 100.0))))
                        sent = math.floor(rem * (pct / 100.0) + 1e-9)
                        if sent >= 1:
                            o_ours.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                            rem = max(0.0, rem - sent)

                if sim.allow_redirect and sim.fleets:
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

            a_pad = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
            a_pad[:n, :n] = a_np
            f_pad = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
            f_pad[:n, :n] = f_np

            o_opp = _fast_gen4_neural_act(opp_snap, sim, opp_pid) if opp_snap is not None else opp_bot.act(sim, opp_pid)
            cur_tick = sim.tick
            sim.step({ours_pid: o_ours, opp_pid: o_opp})
            for _ in range(act_every - 1):
                if not sim.done:
                    sim.step({})
            new_pot = evaluate_galcon_state(sim, ours_pid)
            # Uniform 0.05 potential shaping (Wave 9's 0.085 opening boost caused greedy early over-expansion vs Sparky/Waffle3z)
            step_rew = 0.05 * (new_pot - prev_pot)
            prev_pot = new_pot
            ep_steps.append((cur_tick, pad, a_pad, f_pad, total_lp, v_pred, step_rew))
        else:
            sim.step({})

    enc_f = extract_galcon_features(sim, ours_pid, feature_version=2)
    so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, ours_pid)
    snaps.append((sim.tick, so, ss))
    win = sim.winner()
    speed_bonus = 0.10 * max(0.0, 1.0 - float(sim.tick) / max(1.0, float(sim.max_ticks)))
    d_rew = boss_draw_penalty if is_boss else draw_penalty
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


def _append_gen4_pair_to_buf(buf: Dict[str, List], ep_a, ep_b) -> int:
    """Append paired P1/P2 rollout trajectories with GRPO split-pair centering to buffer."""
    term_a, won_a = ep_a[4], ep_a[5]
    term_b, won_b = ep_b[4], ep_b[5]
    pair_shift = 0.0
    split = 0
    if won_a != won_b:
        split = 1
        pair_shift = 0.30 * 0.5 * (term_a + term_b)
    for ep_steps, snaps, advs, rets, _, _ in (ep_a, ep_b):
        for idx_t, (cur_tick, pad, a_pad, f_pad, old_lp, _, _) in enumerate(ep_steps):
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
            buf["acts"].append(a_pad)
            buf["f_acts"].append(f_pad)
            buf["old_lp"].append(old_lp)
            buf["ret"].append(rets[idx_t])
            buf["adv"].append(advs[idx_t] - pair_shift)
            buf["aux_o"].append(fut_o)
            buf["aux_s"].append(fut_s)
    return split


def _run_gen4_ppo_episodes_batched(
    policy: GalconPlanetTransformer,
    ep_specs: Sequence[Tuple[int, int, float, float, bool, int, Optional[GalconPlanetTransformer], object, bool]],
    *,
    gamma: float,
    gae_lam: float,
    frac_std: float,
    draw_penalty: float,
    boss_draw_penalty: float,
    max_sends_per_planet: int,
    act_every: int = 2,
):
    """Run multiple Gen 4/5 PPO rollout episodes in lockstep with batched policy inference.

    Each element of ``ep_specs`` is ``(map_seed, n_pl, w, h, sym, ours_pid, opp_snap, opp_bot, is_boss)``.
    No two elements in the same batch should share a stateful ``GalconLuaBotBridge`` instance.
    """
    act_every = max(1, int(act_every))
    m = len(ep_specs)
    if m == 0:
        return []
    if m == 1:
        s0 = ep_specs[0]
        return [
            _run_gen4_ppo_episode(
                policy, s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7],
                is_boss=s0[8], gamma=gamma, gae_lam=gae_lam, frac_std=frac_std,
                draw_penalty=draw_penalty, boss_draw_penalty=boss_draw_penalty,
                max_sends_per_planet=max_sends_per_planet, act_every=act_every,
            )
        ]

    sims: List[GalconSim] = []
    ours_pids: List[int] = []
    opp_pids: List[int] = []
    opp_snaps: List[Optional[GalconPlanetTransformer]] = []
    opp_bots: List[object] = []
    is_bosses: List[bool] = []
    ep_steps_list: List[List] = [[] for _ in range(m)]
    snaps_list: List[List[Tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in range(m)]
    prev_pots: List[float] = []

    for map_seed, n_pl, w, h, sym, ours_pid, opp_snap, opp_bot, is_boss in ep_specs:
        sim = GalconSim.create(
            seed=map_seed, num_planets=n_pl, width=w, height=h, max_ticks=220 if n_pl >= 22 else 200, symmetric=sym
        )
        sims.append(sim)
        ours_pids.append(ours_pid)
        opp_pids.append(2 if ours_pid == 1 else 1)
        opp_snaps.append(opp_snap)
        opp_bots.append(opp_bot)
        is_bosses.append(is_boss)
        prev_pots.append(evaluate_galcon_state(sim, ours_pid))

    while True:
        active = [idx for idx in range(m) if not sims[idx].done]
        if not active:
            break
        even_act = [idx for idx in active if sims[idx].tick % act_every == 0]
        for idx in active:
            if sims[idx].tick % act_every != 0:
                sims[idx].step({})
        if not even_act:
            continue

        encs = []
        encs_c = []
        pads = []
        ns = []
        for idx in even_act:
            sim = sims[idx]
            ours_pid = ours_pids[idx]
            enc = extract_galcon_features(sim, ours_pid, feature_version=2)
            n = len(enc.planet_ids)
            so, ss = _snapshot_planet_state(sim, enc.planet_ids, ours_pid)
            snaps_list[idx].append((sim.tick, so, ss))
            enc_c = copy.copy(enc)
            enc_c.planet_feats = canonicalize_galcon_d4(enc.planet_feats)
            pads.append(pad_galcon_state(enc_c))
            encs.append(enc)
            encs_c.append(enc_c)
            ns.append(n)

        k_b = len(even_act)
        max_n = max(ns)
        pf_dim = encs_c[0].planet_feats.shape[-1]
        tf_tail = encs_c[0].timeline_feats.shape[1:]
        ef_dim = encs_c[0].edge_feats.shape[-1]
        pf_b = np.zeros((k_b, max_n, pf_dim), dtype=np.float32)
        tf_b = np.zeros((k_b, max_n, *tf_tail), dtype=np.float32)
        ef_b = np.zeros((k_b, max_n, max_n, ef_dim), dtype=np.float32)
        pm_b = np.zeros((k_b, max_n), dtype=bool)
        for bi, (enc_c, n) in enumerate(zip(encs_c, ns)):
            pf_b[bi, :n] = enc_c.planet_feats
            tf_b[bi, :n] = enc_c.timeline_feats
            ef_b[bi, :n, :n] = enc_c.edge_feats
            pm_b[bi, :n] = True

        with torch.no_grad():
            pf_t = torch.from_numpy(pf_b)
            tf_t = torch.from_numpy(tf_b)
            ef_t = torch.from_numpy(ef_b)
            pm_t = torch.from_numpy(pm_b)
            ms_l_b, ms_f_b, val_b, h_emb_b = policy.forward_multi_send(pf_t, tf_t, ef_t, pm_t, return_hidden=True)

            for bi, idx in enumerate(even_act):
                sim = sims[idx]
                ours_pid = ours_pids[idx]
                opp_pid = opp_pids[idx]
                enc = encs[bi]
                pad = pads[bi]
                n = ns[bi]

                le_t = torch.from_numpy(enc.legal_edges)
                om_t = torch.from_numpy(enc.owned_mask)
                edge_m = om_t.unsqueeze(-1) & le_t
                ms_l0 = ms_l_b[bi, :n, :n].masked_fill(~edge_m, -1e9)
                b_dist = torch.distributions.Bernoulli(logits=ms_l0)
                a_edge = b_dist.sample() * edge_m.float()
                ms_probs_np = torch.sigmoid(ms_l0).numpy()
                a_np = a_edge.numpy()
                for i in range(n):
                    if enc.owned_mask[i]:
                        act_js = np.where(a_np[i] > 0.5)[0]
                        if len(act_js) > max_sends_per_planet:
                            keep = act_js[np.argsort(-ms_probs_np[i, act_js])[:max_sends_per_planet]]
                            a_np[i] = 0.0
                            a_np[i, keep] = 1.0
                a_edge = torch.from_numpy(a_np)
                lp_bern = (b_dist.log_prob(a_edge) * edge_m.float()).sum()

                f_mu = ms_f_b[bi, :n, :n].clamp(0.06, 0.98)
                f_dist = torch.distributions.Normal(f_mu, frac_std)
                f_samp = f_dist.sample().clamp(0.06, 1.0)
                lp_frac = (f_dist.log_prob(f_samp) * a_edge).sum()
                total_lp = float((lp_bern + 0.20 * lp_frac).item())
                v_pred = float(val_b[bi].item())

                o_ours: List[Order] = []
                f_np = f_samp.numpy()
                for i in range(n):
                    if not enc.owned_mask[i]:
                        continue
                    orig_s = float(enc.planets[i].ships)
                    if orig_s < 2.0:
                        continue
                    act_js = [int(j) for j in np.where(a_np[i] > 0.5)[0]]
                    if not act_js:
                        continue
                    act_js.sort(key=lambda j: -float(ms_probs_np[i, j]))
                    rem = orig_s
                    for j in act_js:
                        if rem < 2.0:
                            break
                        want = min(rem, max(1.0, float(f_np[i, j]) * orig_s))
                        ratio = want / max(1.0, rem)
                        pct = 100 if ratio >= 0.92 else max(5, min(100, int(round(ratio * 100.0))))
                        sent = math.floor(rem * (pct / 100.0) + 1e-9)
                        if sent >= 1:
                            o_ours.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                            rem = max(0.0, rem - sent)

                if sim.allow_redirect and sim.fleets:
                    renc = extract_galcon_redir_features(sim, enc)
                    if renc.fleet_ids:
                        rlogits = policy.predict_redir_logits(
                            h_emb_b[bi : bi + 1, :n],
                            torch.from_numpy(renc.cur_dst_idx[None]),
                            torch.from_numpy(renc.redir_feats[None]),
                            torch.ones((1, n), dtype=torch.bool),
                        )
                        rprobs = F.softmax(rlogits[0], dim=-1).numpy()
                        for fi, fid in enumerate(renc.fleet_ids):
                            c_idx = int(renc.cur_dst_idx[fi])
                            bj = int(np.argmax(rprobs[fi]))
                            if bj != c_idx and float(rprobs[fi, bj]) >= 0.55:
                                o_ours.append(RedirOrder(fid, enc.planet_ids[bj]))

                a_pad = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
                a_pad[:n, :n] = a_np
                f_pad = np.zeros((GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=np.float32)
                f_pad[:n, :n] = f_np

                opp_snap = opp_snaps[idx]
                opp_bot = opp_bots[idx]
                o_opp = _fast_gen4_neural_act(opp_snap, sim, opp_pid) if opp_snap is not None else opp_bot.act(sim, opp_pid)
                cur_tick = sim.tick
                sim.step({ours_pid: o_ours, opp_pid: o_opp})
                for _ in range(act_every - 1):
                    if not sim.done:
                        sim.step({})
                new_pot = evaluate_galcon_state(sim, ours_pid)
                step_rew = 0.05 * (new_pot - prev_pots[idx])
                prev_pots[idx] = new_pot
                ep_steps_list[idx].append((cur_tick, pad, a_pad, f_pad, total_lp, v_pred, step_rew))

    results = []
    for idx in range(m):
        sim = sims[idx]
        ours_pid = ours_pids[idx]
        enc_f = extract_galcon_features(sim, ours_pid, feature_version=2)
        so, ss = _snapshot_planet_state(sim, enc_f.planet_ids, ours_pid)
        snaps_list[idx].append((sim.tick, so, ss))
        win = sim.winner()
        speed_bonus = 0.10 * max(0.0, 1.0 - float(sim.tick) / max(1.0, float(sim.max_ticks)))
        d_rew = boss_draw_penalty if is_bosses[idx] else draw_penalty
        term_rew = (1.0 + speed_bonus) if win == ours_pid else (d_rew if win == 0 else -1.0)
        won = 1 if win == ours_pid else 0

        ep_steps = ep_steps_list[idx]
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
        results.append((ep_steps, snaps_list[idx], advs, rets, term_rew, won))
    return results


def _gen4_ppo_rollout_worker(args_tuple):
    """Worker entrypoint for parallel paired-mirror Gen 4 PPO rollout collection (batched lockstep + cached bridges)."""
    (
        policy_ckpt_str,
        snap_ckpt_strs,
        pair_jobs,
        lua_opp_str,
        lua_opp2_str,
        lua_opp3_str,
        gamma,
        gae_lam,
        frac_std,
        draw_penalty,
        boss_draw_penalty,
        max_sends_per_planet,
        worker_seed,
        act_every,
    ) = args_tuple
    torch.set_num_threads(1)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed % (2**31 - 1))
    w_rng = random.Random(worker_seed)

    policy = _load_gen4_policy_from_ckpt(Path(policy_ckpt_str)).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    needed_snaps = {opp_idx for (_, _, _, _, _, ok, opp_idx) in pair_jobs if ok == "snap"}
    snap_map: Dict[int, GalconPlanetTransformer] = {}
    for s_idx in needed_snaps:
        snap_map[s_idx] = _get_worker_cached_snap(snap_ckpt_strs[s_idx])  # type: ignore[assignment]

    lua_pools: Dict[str, List] = {"boss": [], "boss2": [], "boss3": []}
    buf: Dict[str, List] = {
        k: [] for k in ("pf", "tf", "ef", "le", "om", "pm", "acts", "f_acts", "old_lp", "adv", "ret", "aux_o", "aux_s")
    }
    wins = sp_wins = sp_eps = bot_wins = bot_eps = 0
    boss_wins = boss_eps = boss2_wins = boss2_eps = boss3_wins = boss3_eps = 0
    split_pairs = total_eps = 0

    need_b1 = any(j[5] == "boss" for j in pair_jobs)
    need_b2 = any(j[5] == "boss2" for j in pair_jobs)
    need_b3 = any(j[5] == "boss3" for j in pair_jobs)
    if lua_opp_str and need_b1:
        lua_pools["boss"] = _get_worker_lua_pool(lua_opp_str, 3)
    if lua_opp2_str and need_b2:
        lua_pools["boss2"] = _get_worker_lua_pool(lua_opp2_str, 3)
    if lua_opp3_str and need_b3:
        lua_pools["boss3"] = _get_worker_lua_pool(lua_opp3_str, 3)

    # Build all episode tasks and pack into conflict-free waves of up to 10 episodes per batched forward pass
    pending = []
    pair_meta = []
    for p_idx, (map_seed, n_pl, w, h, sym, opp_kind, opp_idx) in enumerate(pair_jobs):
        opp_snap = snap_map[opp_idx] if opp_kind == "snap" else None
        is_boss = False
        is_boss2 = False
        is_boss3 = False
        lua_key = None
        heur_bot = None
        if opp_kind == "boss" and lua_pools["boss"]:
            lua_key = "boss"
            is_boss = True
        elif opp_kind == "boss2":
            if lua_pools["boss2"]:
                lua_key = "boss2"
                is_boss = True
                is_boss2 = True
            else:
                heur_bot = _make_sparring_bot(w_rng, "tactical")
        elif opp_kind == "boss3":
            if lua_pools["boss3"]:
                lua_key = "boss3"
                is_boss = True
                is_boss3 = True
            else:
                heur_bot = _make_sparring_bot(w_rng, "tactical")
        elif opp_kind != "snap":
            heur_bot = _make_sparring_bot(w_rng, opp_kind)
        pair_meta.append((opp_snap is not None, is_boss, is_boss2, is_boss3))
        for side_idx, ours_pid in enumerate((1, 2)):
            pending.append((p_idx, side_idx, lua_key, (map_seed, n_pl, w, h, sym, ours_pid, opp_snap, heur_bot, is_boss)))

    pair_results: Dict[int, List] = {i: [None, None] for i in range(len(pair_jobs))}
    while pending:
        wave_specs = []
        wave_refs = []
        used_lua: Dict[str, int] = {"boss": 0, "boss2": 0, "boss3": 0}
        still = []
        for item in pending:
            p_idx, side_idx, lua_key, spec = item
            if len(wave_specs) >= 10:
                still.append(item)
                continue
            if lua_key is not None:
                slot = used_lua[lua_key]
                if slot >= len(lua_pools[lua_key]):
                    still.append(item)
                    continue
                used_lua[lua_key] = slot + 1
                bot_obj = lua_pools[lua_key][slot]
                spec = (spec[0], spec[1], spec[2], spec[3], spec[4], spec[5], spec[6], bot_obj, spec[8])
            wave_specs.append(spec)
            wave_refs.append((p_idx, side_idx))
        pending = still

        outs = _run_gen4_ppo_episodes_batched(
            policy,
            wave_specs,
            gamma=gamma,
            gae_lam=gae_lam,
            frac_std=frac_std,
            draw_penalty=draw_penalty,
            boss_draw_penalty=boss_draw_penalty,
            max_sends_per_planet=max_sends_per_planet,
            act_every=act_every,
        )
        for (p_idx, side_idx), ep_out in zip(wave_refs, outs):
            pair_results[p_idx][side_idx] = ep_out

    for p_idx in range(len(pair_jobs)):
        ep_a, ep_b = pair_results[p_idx]
        is_snap, is_boss, is_boss2, is_boss3 = pair_meta[p_idx]
        total_eps += 2
        won_a, won_b = ep_a[5], ep_b[5]
        wins += won_a + won_b
        if is_boss3:
            boss3_eps += 2
            boss3_wins += won_a + won_b
        elif is_boss2:
            boss2_eps += 2
            boss2_wins += won_a + won_b
        elif is_boss:
            boss_eps += 2
            boss_wins += won_a + won_b
        elif is_snap:
            sp_eps += 2
            sp_wins += won_a + won_b
        else:
            bot_eps += 2
            bot_wins += won_a + won_b
        split_pairs += _append_gen4_pair_to_buf(buf, ep_a, ep_b)

    arr_buf = {k: np.stack(v) for k, v in buf.items()} if buf["ret"] else {}
    return (
        arr_buf,
        wins,
        sp_wins,
        sp_eps,
        bot_wins,
        bot_eps,
        boss_wins,
        boss_eps,
        boss2_wins,
        boss2_eps,
        boss3_wins,
        boss3_eps,
        split_pairs,
        total_eps,
    )


def _build_stratified_gen4_opp_schedule(
    pairs_per_iter: int,
    *,
    has_lua1: bool,
    has_lua2: bool,
    pool_size: int,
    rng: random.Random,
    sparky_share: float = 0.42,
    esparan_share: float = 0.17,
    has_lua3: bool = False,
    waffle_share: float = 0.15,
) -> List[Tuple[str, int]]:
    """Allocate deterministic, low-variance opponent pairs per PPO iteration.

    Prevents RNG starvation where Sparky v1.6 could drop to 1/12 pairs in an iteration
    while keeping Waffle3z-GA, EsparanBot v0.2, snapshot self-play, and heuristics balanced.
    """
    n = max(1, int(pairs_per_iter))
    heur_cycle = ["tactical", "macro", "rusher", "classic"]
    rng.shuffle(heur_cycle)
    sched: List[Tuple[str, int]] = []

    def _pick_snap() -> Tuple[str, int]:
        if pool_size <= 1:
            return ("snap", 0)
        idx = 0 if rng.random() < 0.45 else rng.randrange(pool_size)
        return ("snap", idx)

    if has_lua1 and has_lua2 and has_lua3:
        n_b1 = min(n, max(1, int(round(n * max(0.0, sparky_share)))))
        n_b3 = min(max(0, n - n_b1), max(1, int(round(n * max(0.0, waffle_share)))) if waffle_share > 0 else 0)
        n_b2 = min(max(0, n - n_b1 - n_b3), max(1, int(round(n * max(0.0, esparan_share)))) if esparan_share > 0 else 0)
        rem = max(0, n - n_b1 - n_b2 - n_b3)
        n_snap = int(round(rem * 0.60))
        n_heur = max(0, rem - n_snap)
        sched.extend([("boss", -1)] * n_b1)
        sched.extend([("boss3", -1)] * n_b3)
        sched.extend([("boss2", -1)] * n_b2)
        sched.extend([_pick_snap() for _ in range(n_snap)])
        sched.extend([(heur_cycle[i % len(heur_cycle)], -1) for i in range(n_heur)])
    elif has_lua1 and has_lua3:
        n_b1 = min(n, max(1, int(round(n * max(0.0, sparky_share)))))
        n_b3 = min(max(0, n - n_b1), max(1, int(round(n * max(0.0, waffle_share)))) if waffle_share > 0 else 0)
        rem = max(0, n - n_b1 - n_b3)
        n_snap = int(round(rem * 0.60))
        n_heur = max(0, rem - n_snap)
        sched.extend([("boss", -1)] * n_b1)
        sched.extend([("boss3", -1)] * n_b3)
        sched.extend([_pick_snap() for _ in range(n_snap)])
        sched.extend([(heur_cycle[i % len(heur_cycle)], -1) for i in range(n_heur)])
    elif has_lua1 and has_lua2:
        n_b1 = min(n, max(1, int(round(n * max(0.0, sparky_share)))))
        n_b2 = min(max(0, n - n_b1), max(1, int(round(n * max(0.0, esparan_share)))) if esparan_share > 0 else 0)
        rem = max(0, n - n_b1 - n_b2)
        n_snap = int(round(rem * 0.60))
        n_heur = max(0, rem - n_snap)
        sched.extend([("boss", -1)] * n_b1)
        sched.extend([("boss2", -1)] * n_b2)
        sched.extend([_pick_snap() for _ in range(n_snap)])
        sched.extend([(heur_cycle[i % len(heur_cycle)], -1) for i in range(n_heur)])
    elif has_lua1:
        n_b1 = min(n, max(1, int(round(n * max(0.0, sparky_share)))))
        rem = max(0, n - n_b1)
        n_snap = int(round(rem * 0.55))
        n_heur = max(0, rem - n_snap)
        sched.extend([("boss", -1)] * n_b1)
        sched.extend([_pick_snap() for _ in range(n_snap)])
        sched.extend([(heur_cycle[i % len(heur_cycle)], -1) for i in range(n_heur)])
    else:
        n_snap = int(round(n * 0.45))
        n_heur = max(0, n - n_snap)
        sched.extend([_pick_snap() for _ in range(n_snap)])
        sched.extend([(heur_cycle[i % len(heur_cycle)], -1) for i in range(n_heur)])

    rng.shuffle(sched)
    return sched


def _compute_gen4_val_gate(
    buckets: Dict[str, Tuple[float, int]],
    *,
    has_lua1: bool,
    has_lua2: bool,
    has_lua3: bool = False,
) -> Dict[str, float]:
    """Compute raw and multi-boss weighted composite validation scores from per-bucket (score, games)."""
    tot_sc = sum(v[0] for v in buckets.values())
    tot_g = max(1, sum(v[1] for v in buckets.values()))
    raw = tot_sc / tot_g
    spky = buckets["boss"][0] / max(1, buckets["boss"][1]) if "boss" in buckets and buckets["boss"][1] > 0 else raw
    esp = buckets["boss2"][0] / max(1, buckets["boss2"][1]) if "boss2" in buckets and buckets["boss2"][1] > 0 else raw
    waff = buckets["boss3"][0] / max(1, buckets["boss3"][1]) if "boss3" in buckets and buckets["boss3"][1] > 0 else raw
    anc = buckets["anc"][0] / max(1, buckets["anc"][1]) if "anc" in buckets and buckets["anc"][1] > 0 else raw
    heur = buckets["heur"][0] / max(1, buckets["heur"][1]) if "heur" in buckets and buckets["heur"][1] > 0 else raw
    if has_lua1 and has_lua2 and has_lua3:
        comp = 0.40 * spky + 0.25 * waff + 0.15 * esp + 0.10 * anc + 0.10 * heur
    elif has_lua1 and has_lua3:
        comp = 0.45 * spky + 0.25 * waff + 0.15 * anc + 0.15 * heur
    elif has_lua1 and has_lua2:
        comp = 0.45 * spky + 0.25 * esp + 0.15 * anc + 0.15 * heur
    elif has_lua1:
        comp = 0.55 * spky + 0.20 * anc + 0.25 * heur
    else:
        comp = raw
    return {
        "raw": float(raw),
        "comp": float(comp),
        "spky": float(spky),
        "waff": float(waff),
        "esp": float(esp),
        "anc": float(anc),
        "heur": float(heur),
        "games": float(tot_g),
    }


def _should_promote_gen4_val(
    cur: Dict[str, float],
    best: Dict[str, float],
    start: Dict[str, float],
    *,
    has_lua1: bool,
    has_lua3: bool = False,
    sparky_slack: float = 0.032,
) -> bool:
    """Strictly gate checkpoint promotion so heuristic gains cannot mask a Sparky/Waffle3z regression."""
    if has_lua1:
        floor = max(best["spky"] - sparky_slack, start["spky"] - sparky_slack)
        if cur["spky"] + 1e-9 < floor:
            return False
    if has_lua1 and has_lua3:
        cur_duo = 0.60 * cur["spky"] + 0.40 * cur["waff"]
        best_duo = 0.60 * best["spky"] + 0.40 * best["waff"]
        start_duo = 0.60 * start["spky"] + 0.40 * start["waff"]
        duo_floor = max(best_duo - sparky_slack, start_duo - sparky_slack)
        if cur_duo + 1e-9 < duo_floor:
            return False
    if cur["comp"] > best["comp"] + 1e-6:
        return True
    if abs(cur["comp"] - best["comp"]) <= 1e-6 and has_lua1 and cur["spky"] > best["spky"] + 1e-6:
        return True
    return False


def _decode_gen4_ms_orders_fast(
    sim: GalconSim,
    enc,
    ms_probs: np.ndarray,
    ms_fracs: np.ndarray,
    h_row: Optional[torch.Tensor],
    model: GalconPlanetTransformer,
    *,
    threshold: float = 0.45,
    max_sends_per_planet: int = 6,
) -> List[Order]:
    """Fast greedy Gen 4 multi-send + native /REDIR decoder from precomputed batched tensors."""
    n = len(enc.planet_ids)
    orders: List[Order] = []
    for i in range(n):
        if not enc.owned_mask[i]:
            continue
        orig_s = float(enc.planets[i].ships)
        if orig_s < 2.0:
            continue
        sel = [
            (float(ms_probs[i, j]), j)
            for j in range(n)
            if j != i and enc.legal_edges[i, j] and float(ms_probs[i, j]) >= threshold
        ]
        if not sel:
            continue
        sel.sort(key=lambda x: -x[0])
        rem = orig_s
        for _, j in sel[:max_sends_per_planet]:
            if rem < 2.0:
                break
            want = min(rem, max(1.0, float(ms_fracs[i, j]) * orig_s))
            ratio = want / max(1.0, rem)
            pct = 100 if ratio >= 0.92 else max(5, min(100, int(round(ratio * 100.0))))
            sent = math.floor(rem * (pct / 100.0) + 1e-9)
            if sent >= 1:
                orders.append(SendOrder(enc.planet_ids[i], enc.planet_ids[j], pct))
                rem = max(0.0, rem - sent)
    if sim.allow_redirect and sim.fleets and h_row is not None:
        renc = extract_galcon_redir_features(sim, enc)
        if renc.fleet_ids:
            rlogits = model.predict_redir_logits(
                h_row,
                torch.from_numpy(renc.cur_dst_idx[None]),
                torch.from_numpy(renc.redir_feats[None]),
                torch.ones((1, n), dtype=torch.bool),
            )
            rprobs = F.softmax(rlogits[0], dim=-1).numpy()
            for fi, fid in enumerate(renc.fleet_ids):
                c_idx = int(renc.cur_dst_idx[fi])
                bj = int(np.argmax(rprobs[fi]))
                if bj != c_idx and float(rprobs[fi, bj]) >= 0.55:
                    orders.append(RedirOrder(fid, enc.planet_ids[bj]))
    return orders


def _run_gen4_val_wave_batched(
    model_eval: GalconPlanetTransformer,
    wave_jobs: Sequence[Tuple[str, str, int, int, float, float, int, bool, int, object, bool]],
    *,
    act_every: int = 2,
) -> List[Tuple[str, float]]:
    """Run a batch of up to 10 validation games in lockstep with one batched ``forward_multi_send`` call per tick."""
    act_every = max(1, int(act_every))
    m = len(wave_jobs)
    if m == 0:
        return []
    sims = [
        GalconSim.create(seed=j[2], num_planets=j[3], width=j[4], height=j[5], max_ticks=j[6], symmetric=j[7])
        for j in wave_jobs
    ]
    a_pids = [j[8] for j in wave_jobs]
    b_pids = [2 if a == 1 else 1 for a in a_pids]
    opp_objs = [j[9] for j in wave_jobs]
    opp_is_nn = [j[10] for j in wave_jobs]

    while True:
        active = [idx for idx in range(m) if not sims[idx].done]
        if not active:
            break
        even_act = [idx for idx in active if sims[idx].tick % act_every == 0]
        for idx in active:
            if sims[idx].tick % act_every != 0:
                sims[idx].step({})
        if not even_act:
            continue

        encs = []
        pfs = []
        ns = []
        for idx in even_act:
            enc = extract_galcon_features(sims[idx], a_pids[idx], feature_version=2)
            encs.append(enc)
            pfs.append(canonicalize_galcon_d4(enc.planet_feats))
            ns.append(len(enc.planet_ids))

        k_b = len(even_act)
        max_n = max(ns)
        pf_dim = pfs[0].shape[-1]
        tf_tail = encs[0].timeline_feats.shape[1:]
        ef_dim = encs[0].edge_feats.shape[-1]
        pf_b = np.zeros((k_b, max_n, pf_dim), dtype=np.float32)
        tf_b = np.zeros((k_b, max_n, *tf_tail), dtype=np.float32)
        ef_b = np.zeros((k_b, max_n, max_n, ef_dim), dtype=np.float32)
        pm_b = np.zeros((k_b, max_n), dtype=bool)
        for bi, (pf_c, enc, n) in enumerate(zip(pfs, encs, ns)):
            pf_b[bi, :n] = pf_c
            tf_b[bi, :n] = enc.timeline_feats
            ef_b[bi, :n, :n] = enc.edge_feats
            pm_b[bi, :n] = True

        with torch.no_grad():
            pf_t = torch.from_numpy(pf_b)
            tf_t = torch.from_numpy(tf_b)
            ef_t = torch.from_numpy(ef_b)
            pm_t = torch.from_numpy(pm_b)
            ms_l_b, ms_f_b, _, h_b = model_eval.forward_multi_send(pf_t, tf_t, ef_t, pm_t, return_hidden=True)
            for bi, idx in enumerate(even_act):
                enc = encs[bi]
                n = ns[bi]
                le_t = torch.from_numpy(enc.legal_edges)
                ms_p = torch.sigmoid(ms_l_b[bi, :n, :n].masked_fill(~le_t, -1e9)).numpy()
                ms_f = ms_f_b[bi, :n, :n].numpy()
                oa = _decode_gen4_ms_orders_fast(
                    sims[idx], enc, ms_p, ms_f, h_b[bi : bi + 1, :n], model_eval
                )
                ob = opp_objs[idx]
                ob_act = _fast_gen4_neural_act(ob, sims[idx], b_pids[idx]) if opp_is_nn[idx] else ob.act(sims[idx], b_pids[idx])
                sims[idx].step({a_pids[idx]: oa, b_pids[idx]: ob_act})

    out: List[Tuple[str, float]] = []
    for idx in range(m):
        w_id = sims[idx].winner()
        pts = 1.0 if w_id == a_pids[idx] else (0.5 if w_id == 0 else 0.0)
        out.append((wave_jobs[idx][0], pts))
    return out


def _gen4_val_worker(args_tuple):
    """Worker entrypoint for parallel batched lockstep ``val_44g`` / ``val_56g`` shards (with cached bridges)."""
    if len(args_tuple) >= 7:
        policy_ckpt_str, anchor_ckpt_str, val_jobs, lua1_str, lua2_str, lua3_str, act_every = args_tuple
    else:
        policy_ckpt_str, anchor_ckpt_str, val_jobs, lua1_str, lua2_str, lua3_str = args_tuple
        act_every = 2
    torch.set_num_threads(1)
    model_eval = _load_gen4_policy_from_ckpt(Path(policy_ckpt_str)).eval()
    for p in model_eval.parameters():
        p.requires_grad_(False)
    anchor_ref = None
    if any(j[1] == "anc" for j in val_jobs):
        anchor_ref = _get_worker_cached_snap(anchor_ckpt_str)

    lua_pools: Dict[str, List] = {"boss": [], "boss2": [], "boss3": []}
    needed_kinds = {j[1] for j in val_jobs}
    if "boss" in needed_kinds and lua1_str:
        lua_pools["boss"] = _get_worker_lua_pool(lua1_str, 3)
    if "boss2" in needed_kinds and lua2_str:
        lua_pools["boss2"] = _get_worker_lua_pool(lua2_str, 3)
    if "boss3" in needed_kinds and lua3_str:
        lua_pools["boss3"] = _get_worker_lua_pool(lua3_str, 3)

    heur_bots = {
        "tactical": GalconTacticalBot(),
        "rusher": GalconRusherBot(style="all_in"),
        "macro": GalconMacroBot(),
    }
    b_sc: Dict[str, float] = {"heur": 0.0, "anc": 0.0, "boss": 0.0, "boss2": 0.0, "boss3": 0.0}
    b_cnt: Dict[str, int] = {"heur": 0, "anc": 0, "boss": 0, "boss2": 0, "boss3": 0}

    pending = list(val_jobs)
    while pending:
        wave = []
        used_lua: Dict[str, int] = {"boss": 0, "boss2": 0, "boss3": 0}
        still = []
        for job in pending:
            bucket, opp_kind, g_seed, n_p, ww, hh, max_t, sym, a_p = job
            if len(wave) >= 10:
                still.append(job)
                continue
            if opp_kind in lua_pools and lua_pools[opp_kind]:
                slot = used_lua[opp_kind]
                if slot >= len(lua_pools[opp_kind]):
                    still.append(job)
                    continue
                used_lua[opp_kind] = slot + 1
                wave.append((bucket, opp_kind, g_seed, n_p, ww, hh, max_t, sym, a_p, lua_pools[opp_kind][slot], False))
            elif opp_kind == "anc":
                wave.append((bucket, opp_kind, g_seed, n_p, ww, hh, max_t, sym, a_p, anchor_ref, True))
            else:
                wave.append((bucket, opp_kind, g_seed, n_p, ww, hh, max_t, sym, a_p, heur_bots.get(opp_kind, heur_bots["tactical"]), False))
        pending = still
        for bucket, pts in _run_gen4_val_wave_batched(model_eval, wave, act_every=act_every):
            b_sc[bucket] += pts
            b_cnt[bucket] += 1
    return b_sc, b_cnt


def rl_mirror_league_ppo_gen4(
    ckpt_path: Path,
    out_path: Path,
    *,
    redir_replay: Optional[Dict[str, np.ndarray]] = None,
    anchor_ckpt_path: Optional[Path] = None,
    anchor2_ckpt_path: Optional[Path] = None,
    lua_opp_path: Optional[str | Path] = None,
    lua_opp2_path: Optional[str | Path] = None,
    lua_opp3_path: Optional[str | Path] = None,
    iters: int = 12,
    pairs_per_iter: int = 8,
    lr: float = 4.5e-5,
    kl_coef: float = 0.030,
    v_anchor_coef: float = 0.0,
    ema_tau: float = 0.07,
    gamma: float = 0.985,
    gae_lam: float = 0.95,
    frac_std: float = 0.065,
    draw_penalty: float = -0.40,
    boss_draw_penalty: float = 0.10,
    sparky_share: float = 0.42,
    esparan_share: float = 0.17,
    waffle_share: float = 0.15,
    large_map_prob: float = 0.30,
    max_sends_per_planet: int = 6,
    act_every: int = 2,
    eval_every: int = 4,
    seed: int = 34000,
    train_threads: int = 2,
    workers: int = 1,
    device: str = "cpu",
) -> Path:
    """Gen 4/6 Native Multi-Send Paired-Seed Mirror League PPO in 1x Canonical D4 space.

    Samples independent Bernoulli multi-edge actions + continuous send fractions directly on
    canonicalized coordinates, optimizing joint Bernoulli+Normal PPO with Moving EMA Teacher KL,
    GRPO paired-seed variance reduction, large 22-28 planet asymmetric arenas, and native ``/REDIR``.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    train_dev, use_dml = resolve_train_device(device)

    has_lua1 = _lua_spec_exists(lua_opp_path)
    lua_opp = None
    if has_lua1:
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp = GalconLuaBotBridge(lua_opp_path)
    has_lua2 = _lua_spec_exists(lua_opp2_path)
    lua_opp2 = None
    if has_lua2:
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp2 = GalconLuaBotBridge(lua_opp2_path)
    has_lua3 = _lua_spec_exists(lua_opp3_path)
    lua_opp3 = None
    if has_lua3:
        from galcon.lua_bridge import GalconLuaBotBridge

        lua_opp3 = GalconLuaBotBridge(lua_opp3_path)

    cpu_policy = _load_gen4_policy_from_ckpt(ckpt_path)
    policy = copy.deepcopy(cpu_policy).to(train_dev)
    ema_teacher = copy.deepcopy(cpu_policy).to(train_dev).eval()
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    snapshot_pool: List[GalconPlanetTransformer] = []
    if anchor_ckpt_path is not None and anchor_ckpt_path.is_file():
        a_m = _load_gen4_policy_from_ckpt(anchor_ckpt_path, allow_non_ms=True).eval()
        for p in a_m.parameters():
            p.requires_grad_(False)
        snapshot_pool.append(a_m)
    if anchor2_ckpt_path is not None and anchor2_ckpt_path.is_file():
        a_m2 = _load_gen4_policy_from_ckpt(anchor2_ckpt_path, allow_non_ms=True).eval()
        for p in a_m2.parameters():
            p.requires_grad_(False)
        snapshot_pool.append(a_m2)
    snapshot_pool.append(copy.deepcopy(cpu_policy).eval())
    for p in snapshot_pool[-1].parameters():
        p.requires_grad_(False)

    opt = _build_optimizer(policy.parameters(), lr=lr, weight_decay=1e-4, use_dml=use_dml)
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
    best_state = _cpu_state_dict(policy)
    total_eps = 0

    lua1_str = _lua_spec_resolve(lua_opp_path)
    lua2_str = _lua_spec_resolve(lua_opp2_path)
    lua3_str = _lua_spec_resolve(lua_opp3_path)

    def _build_val_jobs() -> List[Tuple[str, str, int, int, float, float, int, bool, int]]:
        jobs: List[Tuple[str, str, int, int, float, float, int, bool, int]] = []
        std_specs = [
            ("heur", "tactical", 4),
            ("heur", "rusher", 4),
            ("heur", "macro", 4),
            ("anc", "anc", 4),
            ("boss" if has_lua1 else "heur", "boss" if has_lua1 else "tactical", 12 if has_lua1 else 6),
        ]
        if has_lua2:
            std_specs.append(("boss2", "boss2", 8))
        if has_lua3:
            std_specs.append(("boss3", "boss3", 8))
        for oi, (bucket, opp_kind, ng) in enumerate(std_specs):
            for k in range(ng):
                if bucket in ("boss", "boss3") and k < (ng // 2):
                    n_p, ww, hh = 18, 400.0, 240.0
                    g_seed = 39000 + oi * 100 + (k // 2)
                else:
                    g_seed = 39000 + oi * 100 + k
                    g_rng = random.Random(g_seed)
                    n_p, ww, hh = _sample_map_config(g_rng)
                a_p = 1 if (k % 2 == 0) else 2
                jobs.append((bucket, opp_kind, g_seed, n_p, ww, hh, 200, True, a_p))
        for k in range(4):
            for a_p in (1, 2):
                is_boss_lg = (k % 2 == 0)
                if is_boss_lg:
                    bucket = "boss" if has_lua1 else "heur"
                    opp_kind = "boss" if has_lua1 else "tactical"
                else:
                    bucket = "anc"
                    opp_kind = "anc"
                jobs.append((bucket, opp_kind, 39500 + k, 24 + (k % 3), 560.0, 340.0, 220, (k % 2 == 0), a_p))
        if has_lua3:
            for k in range(2):
                for a_p in (1, 2):
                    jobs.append(("boss3", "boss3", 39700 + k, 24 + (k % 3), 560.0, 340.0, 220, (k % 2 == 0), a_p))
        return jobs

    val_jobs_template = _build_val_jobs()

    tmp_dir_ctx = None
    pool_exec: Optional[ProcessPoolExecutor] = None
    snap_ckpt_paths: List[str] = []
    snap_seq = 0
    use_parallel = workers > 1 and pairs_per_iter >= 2

    def quick_gen4_val(model_eval: GalconPlanetTransformer) -> Dict[str, float]:
        model_eval.eval()
        b_sc: Dict[str, float] = {"heur": 0.0, "anc": 0.0, "boss": 0.0, "boss2": 0.0, "boss3": 0.0}
        b_cnt: Dict[str, int] = {"heur": 0, "anc": 0, "boss": 0, "boss2": 0, "boss3": 0}
        if use_parallel and pool_exec is not None and tmp_dir_ctx is not None and snap_ckpt_paths:
            val_pol_p = Path(tmp_dir_ctx.name) / "val_policy_cur.pt"
            torch.save({"model_config": model_eval.config, "model_state_dict": model_eval.state_dict()}, val_pol_p)
            n_w = min(int(workers), len(val_jobs_template))
            shards = [val_jobs_template[w_i::n_w] for w_i in range(n_w)]
            v_args = [
                (str(val_pol_p), snap_ckpt_paths[0], sh, lua1_str, lua2_str, lua3_str, act_every)
                for sh in shards
                if sh
            ]
            for s_sc, s_cnt in pool_exec.map(_gen4_val_worker, v_args):
                for bk in b_sc:
                    b_sc[bk] += s_sc[bk]
                    b_cnt[bk] += s_cnt[bk]
        else:
            anchor_ref = snapshot_pool[0]
            heur_bots = {
                "tactical": GalconTacticalBot(),
                "rusher": GalconRusherBot(style="all_in"),
                "macro": GalconMacroBot(),
            }
            lua_local = {"boss": lua_opp, "boss2": lua_opp2, "boss3": lua_opp3}
            for bucket, opp_kind, g_seed, n_p, ww, hh, max_t, sym, a_p in val_jobs_template:
                b_p = 2 if a_p == 1 else 1
                s = GalconSim.create(seed=g_seed, num_planets=n_p, width=ww, height=hh, max_ticks=max_t, symmetric=sym)
                is_nn = opp_kind == "anc"
                if is_nn:
                    ob = anchor_ref
                elif opp_kind in lua_local and lua_local[opp_kind] is not None:
                    ob = lua_local[opp_kind]
                else:
                    ob = heur_bots.get(opp_kind, heur_bots["tactical"])
                while not s.done:
                    if s.tick % act_every == 0:
                        oa = _fast_gen4_neural_act(model_eval, s, a_p)
                        ob_act = _fast_gen4_neural_act(ob, s, b_p) if is_nn else ob.act(s, b_p)
                        s.step({a_p: oa, b_p: ob_act})
                    else:
                        s.step({})
                w_id = s.winner()
                pts = 1.0 if w_id == a_p else (0.5 if w_id == 0 else 0.0)
                b_sc[bucket] += pts
                b_cnt[bucket] += 1
        buckets = {k: (b_sc[k], b_cnt[k]) for k in b_sc if b_cnt[k] > 0}
        return _compute_gen4_val_gate(buckets, has_lua1=has_lua1, has_lua2=has_lua2, has_lua3=has_lua3)

    def _fmt_val(vm: Dict[str, float]) -> str:
        vg = int(round(vm["games"]))
        if has_lua1 and has_lua2 and has_lua3:
            return (
                f"val_{vg}g={100*vm['raw']:.1f}% "
                f"[comp={100*vm['comp']:.1f}% | spky={100*vm['spky']:.1f}% waff={100*vm['waff']:.1f}% "
                f"esp={100*vm['esp']:.1f}% anc={100*vm['anc']:.1f}% heur={100*vm['heur']:.1f}%]"
            )
        if has_lua1 and has_lua3:
            return (
                f"val_{vg}g={100*vm['raw']:.1f}% "
                f"[comp={100*vm['comp']:.1f}% | spky={100*vm['spky']:.1f}% waff={100*vm['waff']:.1f}% "
                f"anc={100*vm['anc']:.1f}% heur={100*vm['heur']:.1f}%]"
            )
        if has_lua1 and has_lua2:
            return (
                f"val_{vg}g={100*vm['raw']:.1f}% "
                f"[comp={100*vm['comp']:.1f}% | spky={100*vm['spky']:.1f}% esp={100*vm['esp']:.1f}% anc={100*vm['anc']:.1f}% heur={100*vm['heur']:.1f}%]"
            )
        if has_lua1:
            return (
                f"val_{vg}g={100*vm['raw']:.1f}% "
                f"[comp={100*vm['comp']:.1f}% | spky={100*vm['spky']:.1f}% anc={100*vm['anc']:.1f}% heur={100*vm['heur']:.1f}%]"
            )
        return f"val_{vg}g={100*vm['raw']:.1f}% [anc={100*vm['anc']:.1f}% heur={100*vm['heur']:.1f}%]"

    try:
        torch.set_num_threads(1)
        if use_parallel:
            tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="galcon_gen4_ppo_")
            tmp_root = Path(tmp_dir_ctx.name)
            for s_m in snapshot_pool:
                sp_p = tmp_root / f"snap_{snap_seq}.pt"
                snap_seq += 1
                torch.save({"model_config": s_m.config, "model_state_dict": s_m.state_dict()}, sp_p)
                snap_ckpt_paths.append(str(sp_p))
            n_w = min(int(workers), max(int(pairs_per_iter), 2))
            pool_exec = ProcessPoolExecutor(max_workers=n_w, mp_context=mp.get_context("spawn"))

        start_val = quick_gen4_val(cpu_policy)
        best_val = dict(start_val)
        best_score = float(best_val["comp"])

        boss_names = [
            _lua_spec_label(p)
            for p, ok in ((lua_opp_path, has_lua1), (lua_opp2_path, has_lua2), (lua_opp3_path, has_lua3))
            if ok
        ]
        boss_tag = f"+LuaBoss({'+'.join(boss_names)})" if boss_names else ""
        par_tag = f" x{min(int(workers), int(pairs_per_iter))}w" if use_parallel else ""
        dev_tag = " dml" if use_dml else (f" {train_dev}" if train_dev.type != "cpu" else "")
        print(
            f"[Galcon Gen4 Multi-Send Mirror PPO{boss_tag}{par_tag}{dev_tag}] Start {_fmt_val(start_val)} (pool={len(snapshot_pool)})",
            flush=True,
        )
        t0 = time.time()

        for it in range(1, iters + 1):
            torch.set_num_threads(1)
            cpu_policy.eval()

            opp_sched = _build_stratified_gen4_opp_schedule(
                pairs_per_iter,
                has_lua1=(lua_opp is not None),
                has_lua2=has_lua2,
                has_lua3=has_lua3,
                pool_size=len(snapshot_pool),
                rng=rng,
                sparky_share=sparky_share,
                esparan_share=esparan_share,
                waffle_share=waffle_share,
            )
            pair_jobs = []
            for k, (opp_kind, opp_idx) in enumerate(opp_sched):
                n_pl, w, h, sym = _sample_gen3_map_config(rng, large_prob=large_map_prob)
                map_seed = seed + it * 500 + k
                pair_jobs.append((map_seed, n_pl, w, h, sym, opp_kind, opp_idx))

            wins = sp_wins = sp_eps = bot_wins = bot_eps = 0
            boss_wins = boss_eps = boss2_wins = boss2_eps = boss3_wins = boss3_eps = split_pairs = 0
            t_roll = time.time()
            if use_parallel and pool_exec is not None and tmp_dir_ctx is not None:
                pol_p = Path(tmp_dir_ctx.name) / "policy_cur.pt"
                torch.save({"model_config": cpu_policy.config, "model_state_dict": cpu_policy.state_dict()}, pol_p)
                n_w = min(int(workers), len(pair_jobs))
                shards = [pair_jobs[w_i::n_w] for w_i in range(n_w)]
                worker_args = [
                    (
                        str(pol_p),
                        list(snap_ckpt_paths),
                        sh,
                        lua1_str,
                        lua2_str,
                        lua3_str,
                        gamma,
                        gae_lam,
                        frac_std,
                        draw_penalty,
                        boss_draw_penalty,
                        max_sends_per_planet,
                        seed + it * 10000 + w_i,
                        act_every,
                    )
                    for w_i, sh in enumerate(shards)
                    if sh
                ]
                w_outs = list(pool_exec.map(_gen4_ppo_rollout_worker, worker_args))
                merged = _merge_datasets([o[0] for o in w_outs if o[0]])
                for o in w_outs:
                    wins += o[1]
                    sp_wins += o[2]
                    sp_eps += o[3]
                    bot_wins += o[4]
                    bot_eps += o[5]
                    boss_wins += o[6]
                    boss_eps += o[7]
                    boss2_wins += o[8]
                    boss2_eps += o[9]
                    boss3_wins += o[10]
                    boss3_eps += o[11]
                    split_pairs += o[12]
                    total_eps += o[13]
                adv_arr = merged["adv"].astype(np.float32)
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)
                ds = TensorDataset(
                    torch.from_numpy(merged["pf"]),
                    torch.from_numpy(merged["tf"]),
                    torch.from_numpy(merged["ef"]),
                    torch.from_numpy(merged["le"]),
                    torch.from_numpy(merged["om"]),
                    torch.from_numpy(merged["pm"]),
                    torch.from_numpy(merged["acts"]),
                    torch.from_numpy(merged["f_acts"]),
                    torch.from_numpy(merged["old_lp"].astype(np.float32)),
                    torch.from_numpy(adv_arr),
                    torch.from_numpy(merged["ret"].astype(np.float32)),
                    torch.from_numpy(merged["aux_o"]),
                    torch.from_numpy(merged["aux_s"]),
                )
            else:
                buf = {k: [] for k in ("pf", "tf", "ef", "le", "om", "pm", "acts", "f_acts", "old_lp", "adv", "ret", "aux_o", "aux_s")}
                for map_seed, n_pl, w, h, sym, opp_kind, opp_idx in pair_jobs:
                    opp_snap = snapshot_pool[opp_idx] if opp_kind == "snap" else None
                    opp_bot = None
                    is_boss = False
                    is_boss2 = False
                    is_boss3 = False
                    if opp_kind == "boss":
                        opp_bot = lua_opp
                        is_boss = True
                    elif opp_kind == "boss2":
                        if lua_opp2 is not None:
                            opp_bot = lua_opp2
                            is_boss = True
                            is_boss2 = True
                        else:
                            opp_bot = _make_sparring_bot(rng, "tactical")
                    elif opp_kind == "boss3":
                        if lua_opp3 is not None:
                            opp_bot = lua_opp3
                            is_boss = True
                            is_boss3 = True
                        else:
                            opp_bot = _make_sparring_bot(rng, "tactical")
                    elif opp_kind != "snap":
                        opp_bot = _make_sparring_bot(rng, opp_kind)

                    ep_a = _run_gen4_ppo_episode(
                        cpu_policy, map_seed, n_pl, w, h, sym, 1, opp_snap, opp_bot,
                        is_boss=is_boss, gamma=gamma, gae_lam=gae_lam, frac_std=frac_std,
                        draw_penalty=draw_penalty, boss_draw_penalty=boss_draw_penalty,
                        max_sends_per_planet=max_sends_per_planet, act_every=act_every,
                    )
                    ep_b = _run_gen4_ppo_episode(
                        cpu_policy, map_seed, n_pl, w, h, sym, 2, opp_snap, opp_bot,
                        is_boss=is_boss, gamma=gamma, gae_lam=gae_lam, frac_std=frac_std,
                        draw_penalty=draw_penalty, boss_draw_penalty=boss_draw_penalty,
                        max_sends_per_planet=max_sends_per_planet, act_every=act_every,
                    )
                    total_eps += 2
                    won_a, won_b = ep_a[5], ep_b[5]
                    wins += won_a + won_b
                    if is_boss3:
                        boss3_eps += 2
                        boss3_wins += won_a + won_b
                    elif is_boss2:
                        boss2_eps += 2
                        boss2_wins += won_a + won_b
                    elif is_boss:
                        boss_eps += 2
                        boss_wins += won_a + won_b
                    elif opp_snap is not None:
                        sp_eps += 2
                        sp_wins += won_a + won_b
                    else:
                        bot_eps += 2
                        bot_wins += won_a + won_b
                    split_pairs += _append_gen4_pair_to_buf(buf, ep_a, ep_b)

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
            roll_sec = time.time() - t_roll

            t_upd = time.time()
            torch.set_num_threads(max(1, int(train_threads)))
            # Wave 11/12 speedup: length-bucketed frozen ema_teacher precompute + forward_multi_send (skips dead legacy pairwise head)
            ds_lens = ds.tensors[5].sum(dim=1).numpy()
            n_ds = len(ds)
            full_n_ds = int(ds.tensors[0].shape[1])
            e_p_all = torch.full((n_ds, full_n_ds, full_n_ds), 1e-5, dtype=torch.float32)
            e_v_all = torch.zeros((n_ds,), dtype=torch.float32)
            with torch.no_grad():
                pre_sampler = _LengthBucketBatchSampler(ds_lens, 128, shuffle=False, seed=seed + it)
                for b_indices in pre_sampler:
                    idx_t = torch.tensor(b_indices, dtype=torch.long)
                    p_pf, p_tf, p_ef, p_le, p_om, p_pm = (ds.tensors[k][idx_t].to(train_dev) for k in range(6))
                    t_pf, t_tf, t_ef, t_le, t_om, t_pm = _trim_batch_n(p_pf, p_tf, p_ef, p_le, p_om, p_pm)
                    eml_c, _, ev_c = ema_teacher.forward_multi_send(t_pf, t_tf, t_ef, t_pm)
                    em_c = t_om.unsqueeze(-1) & t_le
                    ep_c = torch.sigmoid(eml_c.masked_fill(~em_c, -1e9)).clamp(1e-5, 1.0 - 1e-5).cpu()
                    tn = t_pf.shape[1]
                    e_p_all[idx_t, :tn, :tn] = ep_c
                    e_v_all[idx_t] = ev_c.cpu()
            ds_with_ema = TensorDataset(*ds.tensors, e_p_all, e_v_all)

            policy.train()
            loader = DataLoader(
                ds_with_ema,
                batch_sampler=_LengthBucketBatchSampler(ds_lens, 96, shuffle=True, seed=seed + it * 31),
            )
            ent_coef = 0.004 * (1.0 - 0.45 * (it / max(1, iters)))
            r_iter = iter(redir_loader) if redir_loader is not None else None

            for _ in range(2):
                for b_idx, batch in enumerate(loader):
                    pf, tf, ef, le, om, pm, acts, f_acts, old_lp, adv, ret, aux_o, aux_s, e_p, e_val = (
                        t.to(train_dev) for t in batch
                    )
                    pf, tf, ef, le, om, pm, acts, f_acts, aux_o, aux_s, e_p = _trim_batch_n(
                        pf, tf, ef, le, om, pm, acts, f_acts, aux_o, aux_s, e_p
                    )
                    is_gen5_ppo = hasattr(policy, "decode_wdl_from_hidden")
                    if is_gen5_ppo:
                        ms_l, ms_f, val, wdl_logits, aux_pred = policy.forward_multi_send(
                            pf, tf, ef, pm, return_aux=True, return_wdl=True
                        )
                    else:
                        ms_l, ms_f, val, aux_pred = policy.forward_multi_send(pf, tf, ef, pm, return_aux=True)
                        wdl_logits = None
                    edge_m = om.unsqueeze(-1) & le
                    edge_f = edge_m.float()
                    edge_sum = edge_f.sum().clamp_min(1.0)
                    ms_l = ms_l.masked_fill(~edge_m, -1e9)
                    bern_lp, ent = _bernoulli_log_prob_and_entropy(ms_l, acts)
                    new_lp_bern = (bern_lp * edge_f).sum(dim=(-1, -2))

                    f_mu = ms_f.clamp(0.06, 0.98)
                    f_dist = torch.distributions.Normal(f_mu, frac_std)
                    new_lp_frac = (f_dist.log_prob(f_acts) * acts * edge_f).sum(dim=(-1, -2))
                    new_lp = new_lp_bern + 0.20 * new_lp_frac

                    ratio = torch.exp((new_lp - old_lp).clamp(-8.0, 8.0))
                    surr = torch.min(ratio * adv, torch.clamp(ratio, 0.85, 1.15) * adv).mean()

                    ent_mean = (ent * edge_f).sum() / edge_sum

                    cur_p = torch.sigmoid(ms_l).clamp(1e-5, 1.0 - 1e-5)
                    kl = e_p * torch.log(e_p / cur_p) + (1.0 - e_p) * torch.log((1.0 - e_p) / (1.0 - cur_p))
                    kl_mean = (kl * edge_f).sum() / edge_sum

                    if wdl_logits is not None:
                        v_clamped = ret.clamp(-1.0, 1.0)
                        t_draw = 0.65 * (1.0 - v_clamped.abs()).pow(2)
                        t_win = (1.0 - t_draw + v_clamped) * 0.5
                        t_loss = (1.0 - t_draw - v_clamped) * 0.5
                        wdl_t = torch.stack([t_loss, t_draw, t_win], dim=-1).clamp(1e-4, 1.0)
                        wdl_t = wdl_t / wdl_t.sum(dim=-1, keepdim=True)
                        v_loss = -(wdl_t * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean() + 0.25 * F.mse_loss(val, ret)
                    else:
                        v_loss = F.mse_loss(val, ret)
                    v_anc = F.mse_loss(val, e_val) if v_anchor_coef > 0.0 else 0.0
                    pm_f = pm.float()
                    pm_sum = pm_f.sum().clamp_min(1.0)
                    aux_ce = F.cross_entropy(
                        aux_pred[..., :3].reshape(-1, 3),
                        aux_o.reshape(-1),
                        reduction="none",
                    ).reshape_as(pm_f)
                    aux_own_loss = (aux_ce * pm_f).sum() / pm_sum
                    aux_shp_loss = (F.smooth_l1_loss(aux_pred[..., 3], aux_s, reduction="none") * pm_f).sum() / pm_sum
                    aux_loss = aux_own_loss + 0.5 * aux_shp_loss
                    loss = -surr + 0.40 * v_loss + v_anchor_coef * v_anc + kl_coef * kl_mean - ent_coef * ent_mean + 0.14 * aux_loss

                    if r_iter is not None and (b_idx % 4 == 0):
                        try:
                            rb = next(r_iter)
                        except StopIteration:
                            r_iter = iter(redir_loader)
                            rb = next(r_iter)
                        rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk = (t.to(train_dev) for t in rb)
                        rloss, _, _ = _redir_batch_loss(policy, rpf, rtf, ref, rpm, rcur, rfeat, rtar, rmsk)
                        loss = loss + 0.25 * rloss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.8, foreach=not use_dml)
                    opt.step()

            sched.step()
            torch.set_num_threads(1)

            with torch.no_grad():
                for p_ema, p_cur in zip(ema_teacher.parameters(), policy.parameters()):
                    p_ema.data.mul_(1.0 - ema_tau).add_(p_cur.data, alpha=ema_tau)

            cpu_sd = _cpu_state_dict(policy)
            cpu_policy.load_state_dict(cpu_sd)
            upd_sec = time.time() - t_upd

            if it % 3 == 0:
                snap_copy = copy.deepcopy(cpu_policy).eval()
                for p in snap_copy.parameters():
                    p.requires_grad_(False)
                snapshot_pool.append(snap_copy)
                if use_parallel and tmp_dir_ctx is not None:
                    sp_p = Path(tmp_dir_ctx.name) / f"snap_{snap_seq}.pt"
                    snap_seq += 1
                    torch.save({"model_config": snap_copy.config, "model_state_dict": snap_copy.state_dict()}, sp_p)
                    snap_ckpt_paths.append(str(sp_p))
                if len(snapshot_pool) > 10:
                    snapshot_pool = [snapshot_pool[0]] + snapshot_pool[-9:]
                    if use_parallel:
                        snap_ckpt_paths = [snap_ckpt_paths[0]] + snap_ckpt_paths[-9:]

            boss_str = f", boss={boss_wins}/{boss_eps}" if boss_eps > 0 else ""
            boss2_str = f", boss2={boss2_wins}/{boss2_eps}" if boss2_eps > 0 else ""
            boss3_str = f", boss3={boss3_wins}/{boss3_eps}" if boss3_eps > 0 else ""
            log_msg = (
                f"[Galcon Gen4 Multi-Send PPO] iter {it:2d}/{iters} (eps={total_eps}, {time.time()-t0:5.1f}s "
                f"[roll={roll_sec:.1f}s upd={upd_sec:.1f}s]) | "
                f"wins={wins}/{2*pairs_per_iter} (sp={sp_wins}/{sp_eps}, bots={bot_wins}/{bot_eps}{boss_str}{boss2_str}{boss3_str}, splits={split_pairs}) | "
                f"states={len(ds)} | pool={len(snapshot_pool)}"
            )
            if it % eval_every == 0 or it == iters:
                cur_val = quick_gen4_val(cpu_policy)
                log_msg += f" | {_fmt_val(cur_val)}"
                if _should_promote_gen4_val(cur_val, best_val, start_val, has_lua1=(lua_opp is not None), has_lua3=(lua_opp3 is not None)):
                    best_val = dict(cur_val)
                    best_score = float(best_val["comp"])
                    best_state = copy.deepcopy(cpu_sd)
                    torch.save(
                        {
                            "model_config": cpu_policy.config,
                            "model_state_dict": best_state,
                            "has_trained_redir": True,
                            "has_trained_multi_send": True,
                            "use_canonical_d4": True,
                            "val_gen4_score": best_score,
                            "val_gen4_breakdown": best_val,
                            "ppo_episodes": total_eps,
                        },
                        out_path,
                    )
                    log_msg += " *BEST*"
            print(log_msg, flush=True)

        final_path = out_path.with_name(out_path.stem + "_final.pt")
        torch.save(
            {
                "model_config": cpu_policy.config,
                "model_state_dict": _cpu_state_dict(policy),
                "has_trained_redir": True,
                "has_trained_multi_send": True,
                "use_canonical_d4": True,
                "ppo_episodes": total_eps,
            },
            final_path,
        )
        torch.save(
            {
                "model_config": cpu_policy.config,
                "model_state_dict": best_state,
                "has_trained_redir": True,
                "has_trained_multi_send": True,
                "use_canonical_d4": True,
                "val_gen4_score": best_score,
                "val_gen4_breakdown": best_val,
                "ppo_episodes": total_eps,
            },
            out_path,
        )
        return out_path
    finally:
        if pool_exec is not None:
            pool_exec.shutdown(wait=True)
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.cleanup()
        if lua_opp is not None:
            lua_opp.close()
        if lua_opp2 is not None:
            lua_opp2.close()
        if lua_opp3 is not None:
            lua_opp3.close()


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
    common_keys = [k for k in valid[0].keys() if all(k in d for d in valid)]
    return {k: np.concatenate([d[k] for d in valid], axis=0) for k in common_keys}


def save_ei_dataset_npz(path: str | Path, data: Dict[str, np.ndarray]) -> None:
    """Persist a Gen 4 Search-EI dataset dict to a compressed ``.npz`` replay buffer file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), **data)


def load_ei_dataset_npz(paths_spec: str | Sequence[str | Path]) -> Dict[str, np.ndarray]:
    """Load and merge one or more comma-separated ``.npz`` EI replay buffer files."""
    if isinstance(paths_spec, (str, Path)):
        raw_paths = [Path(s.strip()) for s in str(paths_spec).split(",") if s.strip()]
    else:
        raw_paths = [Path(s) for s in paths_spec if str(s).strip()]
    loaded: List[Dict[str, np.ndarray]] = []
    for p in raw_paths:
        if not p.is_file():
            raise FileNotFoundError(f"EI replay buffer file not found: {p}")
        with np.load(str(p)) as z:
            loaded.append({k: z[k] for k in z.files})
    return _merge_datasets(loaded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train GalconPlanetTransformer (v1, Gen 2, Gen 3, or Gen 4 Multi-Send).")
    parser.add_argument("--mode", choices=["gen2", "v1"], default="gen2")
    parser.add_argument("--stage", choices=["all", "ei", "ppo", "mirror_ppo", "gen4_ei", "gen4_ppo", "eval"], default="all")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume2", default="")
    parser.add_argument("--student-init", default="")
    parser.add_argument("--gen2-anchor", default="checkpoints/galcon_gen2_champion.pt")
    parser.add_argument("--gen2-anchor2", default="")
    parser.add_argument("--lua-opp", default="")
    parser.add_argument("--lua-opp2", default="")
    parser.add_argument("--lua-opp3", default="")
    parser.add_argument("--boss-draw-penalty", type=float, default=0.10)
    parser.add_argument("--sparky-share", type=float, default=0.42)
    parser.add_argument("--esparan-share", type=float, default=0.17)
    parser.add_argument("--waffle-share", type=float, default=0.15)
    parser.add_argument("--large-map-prob", type=float, default=0.30)
    parser.add_argument("--kl-coef", type=float, default=0.030)
    parser.add_argument("--v-anchor-coef", type=float, default=0.0)
    parser.add_argument("--kd-alpha", type=float, default=0.30)
    parser.add_argument("--arch", choices=["gen4", "gen5"], default="gen4", help="Model architecture for gen4_ei supervised/KD training (gen4=1.29M, gen5=2.93M Edge-Gated WDL).")
    parser.add_argument("--filter-draws", action="store_true", help="Mask boss stalemate draws from EI policy loss while keeping value/aux supervision.")
    parser.add_argument("--act-every", type=int, default=2, help="Action cadence during EI game collection (1=full-speed tournament cadence, 2=classic 2-tick).")
    parser.add_argument("--save-ei-npz", default="", help="Optional path to save newly collected Gen 4 Search-EI dataset (.npz) for fast replay/pooling.")
    parser.add_argument("--load-ei-npz", default="", help="Optional comma-separated .npz EI replay buffer paths to pool with newly collected games (or train directly if --search-games 0).")
    parser.add_argument("--eval-every", type=int, default=4)
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
    parser.add_argument("--train-threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "dml", "cuda", "auto"], default="cpu")
    parser.add_argument("--out", default="checkpoints/galcon_transformer_best.pt")
    parser.add_argument("--v1-ckpt", default="checkpoints/galcon_v1_baseline.pt")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.mode == "v1":
        bc_path = out_path.with_name("galcon_v1_bc.pt")
        dagger_path = out_path.with_name("galcon_v1_dagger.pt")
        ppo_path = out_path.with_name("galcon_v1_ppo.pt")

        bc_data = collect_teacher_dataset(num_games=args.bc_games, seed=2000)
        train_supervised(bc_data, epochs=args.epochs, lr=1.2e-3, out_path=bc_path, seed=42)

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
            train_threads=args.train_threads,
        )
        return 0

    if args.stage == "gen4_ei":
        resume_p = Path(args.resume) if args.resume else Path("checkpoints/galcon_transformer_best.pt")
        resume_p2 = Path(args.resume2) if args.resume2 else None
        student_p = Path(args.student_init) if args.student_init else (None if args.arch == "gen5" and not args.student_init else resume_p)
        lua_p = args.lua_opp or None
        lua_p2 = args.lua_opp2 or None
        lua_p3 = args.lua_opp3 or None

        datasets_to_pool: List[Dict[str, np.ndarray]] = []
        if args.load_ei_npz:
            cached = load_ei_dataset_npz(args.load_ei_npz)
            if cached:
                n_cached = len(next(iter(cached.values())))
                print(f"[Galcon-Gen4-EI] Loaded {n_cached} cached replay states from {args.load_ei_npz}", flush=True)
                datasets_to_pool.append(cached)

        g4_redir: Dict[str, np.ndarray] = {}
        if args.search_games > 0:
            fresh_data, g4_redir = collect_gen4_multisend_ei_dataset(
                resume_p,
                num_games=args.search_games,
                seed=args.seed,
                lua_opp_path=lua_p,
                lua_opp2_path=lua_p2,
                lua_opp3_path=lua_p3,
                ckpt_path2=resume_p2,
                large_map_prob=args.large_map_prob,
                filter_draw_policy=args.filter_draws,
                act_every=args.act_every,
                workers=args.workers,
            )
            if args.save_ei_npz and fresh_data:
                save_ei_dataset_npz(args.save_ei_npz, fresh_data)
                print(
                    f"[Galcon-Gen4-EI] Saved {len(next(iter(fresh_data.values())))} EI replay states -> {args.save_ei_npz}",
                    flush=True,
                )
            datasets_to_pool.append(fresh_data)

        g4_data = datasets_to_pool[0] if len(datasets_to_pool) == 1 else _merge_datasets(datasets_to_pool)
        datasets_to_pool.clear()
        if not g4_data:
            raise ValueError("gen4_ei requires either --search-games > 0 or a non-empty --load-ei-npz buffer.")
        old_r = _load_redir_npz()
        merged_r = _merge_datasets([old_r, g4_redir]) if (old_r and g4_redir) else (g4_redir or old_r)
        if g4_redir:
            _save_redir_npz(merged_r)
        del old_r, g4_redir
        train_gen4_multisend_supervised(
            g4_data,
            merged_r,
            resume_from=student_p,
            out_path=out_path,
            epochs=args.epochs,
            lr=args.lr if args.lr > 0 else (4.5e-4 if args.arch == "gen5" else 3.5e-4),
            kd_alpha=args.kd_alpha,
            arch=args.arch,
            seed=args.seed,
            train_threads=args.train_threads,
            device=args.device,
        )
        return 0

    if args.stage == "gen4_ppo":
        resume_p = Path(args.resume) if args.resume else Path("checkpoints/galcon_gen4_ei_r1.pt")
        redir_replay = _load_redir_npz()
        rl_mirror_league_ppo_gen4(
            resume_p,
            out_path,
            redir_replay=redir_replay,
            anchor_ckpt_path=Path(args.gen2_anchor) if args.gen2_anchor else None,
            anchor2_ckpt_path=Path(args.gen2_anchor2) if args.gen2_anchor2 else None,
            lua_opp_path=args.lua_opp or None,
            lua_opp2_path=args.lua_opp2 or None,
            lua_opp3_path=args.lua_opp3 or None,
            iters=args.ppo_iters,
            pairs_per_iter=args.mirror_pairs,
            lr=args.lr if args.lr > 0 else 4.5e-5,
            kl_coef=args.kl_coef,
            v_anchor_coef=args.v_anchor_coef,
            boss_draw_penalty=args.boss_draw_penalty,
            sparky_share=args.sparky_share,
            esparan_share=args.esparan_share,
            waffle_share=args.waffle_share,
            large_map_prob=args.large_map_prob,
            act_every=args.act_every,
            eval_every=max(1, args.eval_every),
            seed=args.seed,
            train_threads=args.train_threads,
            workers=args.workers,
            device=args.device,
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
