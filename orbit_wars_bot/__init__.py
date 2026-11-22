"""Orbit Wars LightZero package.

Provides an exact fast simulator, physics/intercept solvers, spatial feature
encoding, tactical macro-actions, rule-based heuristic baseline, LightZero
DI-engine environment integration, and an online MCTS Kaggle submission agent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_lightzero_on_path() -> Path | None:
    """Ensure ``external/LightZero`` is importable when present in the repo."""
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "external" / "LightZero"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate
    env_path = os.environ.get("LIGHTZERO_PATH")
    if env_path and os.path.isdir(env_path) and env_path not in sys.path:
        sys.path.insert(0, env_path)
        return Path(env_path)
    return None


__all__ = ["ensure_lightzero_on_path"]
