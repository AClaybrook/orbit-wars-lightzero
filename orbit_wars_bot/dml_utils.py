"""Shared PyTorch DirectML utilities for zero-fallback GPU training on AMD/Windows/WSL2."""

from __future__ import annotations

import math
import queue
import threading
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn

_T = TypeVar("_T")


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
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
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
                denom = v.sqrt().mul_(1.0 / math.sqrt(bc2)).add_(eps)
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


def build_optimizer(
    params,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_dml: bool = False,
) -> torch.optim.Optimizer:
    """Build ``DMLAdamW`` for DirectML or ``torch.optim.AdamW`` otherwise."""
    if use_dml:
        return DMLAdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detach and clone ``model.state_dict()`` onto CPU for portable saving and workers."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def one_hot_no_scatter(
    indices: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Broadcast-comparison one-hot avoiding DirectML's scatter restriction."""
    cls = torch.arange(num_classes, device=indices.device, dtype=indices.dtype)
    return (indices.unsqueeze(-1) == cls).to(dtype)


def apply_d4_batch_gpu(
    pf: torch.Tensor,
    tf: torch.Tensor,
    *,
    k_rot: int = 0,
    flip_x: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-fallback GPU D4 symmetry transform for batched planet & timeline tensors.

    Mirrors ``apply_d4_planet_feats`` / ``apply_d4_timeline_feats`` bit-for-bit using
    only native ``neg`` and ``cat`` ops (no CPU roundtrip, no ``index_put`` fallback).
    """
    k = int(k_rot) % 4
    if k == 0 and not flip_x:
        return pf, tf

    x = pf[..., 0:1]
    y = pf[..., 1:2]
    vx = pf[..., 2:3]
    vy = pf[..., 3:4]
    if flip_x:
        x = -x
        vx = -vx
    if k == 1:
        x, y = -y, x
        vx, vy = -vy, vx
    elif k == 2:
        x, y = -x, -y
        vx, vy = -vx, -vy
    elif k == 3:
        x, y = y, -x
        vx, vy = vy, -vx
    pf_out = torch.cat([x, y, vx, vy, pf[..., 4:]], dim=-1)

    if tf.shape[-1] >= 10:
        tx = tf[..., 8:9]
        ty = tf[..., 9:10]
        if flip_x:
            tx = -tx
        if k == 1:
            tx, ty = -ty, tx
        elif k == 2:
            tx, ty = -tx, -ty
        elif k == 3:
            tx, ty = ty, -tx
        tf_out = torch.cat([tf[..., :8], tx, ty, tf[..., 10:]], dim=-1)
    else:
        tf_out = tf
    return pf_out, tf_out


def snap_bucket_batches(
    planet_counts: np.ndarray,
    batch_size: int,
    *,
    buckets: Tuple[int, ...] = (24, 32, 44),
    max_cap: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """Group board indices into fixed discrete planet-count ceilings (e.g. 24, 32, 44).

    On DirectML, arbitrary per-batch ``max_n`` values trigger metacommand/shader cache
    misses. Snapping every batch to 3 fixed ceilings keeps 100% DirectML cache hit rate
    while still cutting average O(N^2) attention FLOPs by ~1.7x (zero dropped samples).
    """
    n = int(len(planet_counts))
    if n == 0:
        return []
    bs = max(1, int(batch_size))
    cap = int(max_cap) if max_cap is not None else int(max(int(planet_counts.max()), max(buckets)))
    clean_buckets = sorted({min(cap, int(b)) for b in buckets if int(b) > 0} | {cap})
    gen = rng if rng is not None else np.random.default_rng()

    out: List[Tuple[np.ndarray, int]] = []
    lo = -1
    for hi in clean_buckets:
        idxs = np.flatnonzero((planet_counts > lo) & (planet_counts <= hi))
        lo = hi
        if len(idxs) == 0:
            continue
        if shuffle:
            idxs = gen.permutation(idxs)
        mn = max(2, min(cap, hi))
        for i in range(0, len(idxs), bs):
            out.append((idxs[i : i + bs], mn))
    if shuffle and len(out) > 1:
        perm = gen.permutation(len(out))
        out = [out[int(i)] for i in perm]
    return out


def bucket_batch_indices(
    planet_counts: np.ndarray,
    batch_size: int,
    *,
    rng: Optional[np.random.Generator] = None,
    shuffle: bool = True,
    jitter: float = 1.5,
) -> List[np.ndarray]:
    """Group board indices into length-homogeneous minibatches to cut O(N^2) padding FLOPs.

    Sorts by ``planet_counts + small uniform jitter`` (preserving stochastic batch
    composition across epochs), slices batches of size ``batch_size``, then randomly
    shuffles minibatch order.
    """
    n = int(len(planet_counts))
    if n == 0:
        return []
    bs = max(1, int(batch_size))
    if shuffle:
        gen = rng if rng is not None else np.random.default_rng()
        keys = planet_counts.astype(np.float32) + gen.uniform(-jitter, jitter, size=n).astype(np.float32)
        order = np.argsort(keys, kind="mergesort")
    else:
        order = np.argsort(planet_counts, kind="mergesort")
    batches = [order[i : i + bs] for i in range(0, n, bs)]
    if shuffle and len(batches) > 1:
        gen = rng if rng is not None else np.random.default_rng()
        b_perm = gen.permutation(len(batches))
        batches = [batches[int(i)] for i in b_perm]
    return batches


class ThreadedPrefetcher:
    """Overlap CPU/NVMe batch preparation with GPU compute using a bounded daemon queue."""

    def __init__(self, source: Iterable[_T], maxsize: int = 3) -> None:
        self._source = source
        self._maxsize = max(1, int(maxsize))

    def __iter__(self) -> Iterator[_T]:
        sentinel = object()
        q: queue.Queue[object] = queue.Queue(maxsize=self._maxsize)
        err_box: List[BaseException] = []
        stop_ev = threading.Event()

        def _producer() -> None:
            try:
                for item in self._source:
                    while not stop_ev.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop_ev.is_set():
                        break
            except BaseException as exc:
                err_box.append(exc)
            finally:
                while True:
                    try:
                        q.put(sentinel, timeout=0.1)
                        break
                    except queue.Full:
                        if stop_ev.is_set():
                            break

        th = threading.Thread(target=_producer, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is sentinel:
                    break
                yield item  # type: ignore[misc]
            if err_box:
                raise err_box[0]
        finally:
            stop_ev.set()
            th.join(timeout=2.0)
