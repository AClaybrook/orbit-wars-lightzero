"""Shared PyTorch DirectML utilities for zero-fallback GPU training on AMD/Windows/WSL2."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


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
