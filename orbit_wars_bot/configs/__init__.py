"""LightZero training configuration factories for Orbit Wars."""

from __future__ import annotations

from orbit_wars_bot.configs.alphazero_config import make_alphazero_config
from orbit_wars_bot.configs.gumbel_muzero_config import make_gumbel_muzero_config
from orbit_wars_bot.configs.muzero_config import make_muzero_config

__all__ = [
    "make_alphazero_config",
    "make_muzero_config",
    "make_gumbel_muzero_config",
]
