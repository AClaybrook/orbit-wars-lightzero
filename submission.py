"""Official Kaggle Orbit Wars submission entry point.

Kaggle invokes ``agent(obs, config)`` on every tick with a 500ms per-turn
budget. This module lazily initializes an ``OnlineMCTSAgent`` (optionally
loading a trained LightZero AlphaZero/MuZero checkpoint when available) and
enforces a strict wall-clock time budget so turns never time out.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from orbit_wars_bot.mcts_agent import OnlineMCTSAgent  # noqa: E402

_AGENT: OnlineMCTSAgent | None = None


def _discover_checkpoint() -> str | None:
    env_ckpt = os.environ.get("ORBIT_WARS_CHECKPOINT")
    if env_ckpt and Path(env_ckpt).is_file():
        return env_ckpt
    candidates = [
        _ROOT / "checkpoints" / "ckpt_best.pth.tar",
        _ROOT / "checkpoints" / "best.pth.tar",
        _ROOT / "ckpt_best.pth.tar",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _get_agent() -> OnlineMCTSAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = OnlineMCTSAgent(
            checkpoint_path=_discover_checkpoint(),
            num_simulations=24,
            time_budget_sec=0.35,
            rollout_horizon=12,
        )
    return _AGENT


def agent(obs: Any, config: Any = None) -> List[List[float]]:
    """Kaggle Environments agent callback."""
    return _get_agent().act(obs, config)
