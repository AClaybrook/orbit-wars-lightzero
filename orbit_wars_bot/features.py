"""Player-canonicalized spatial feature encoder and heuristic state evaluator.

Encodes any Orbit Wars state (``FastOrbitSim`` or Kaggle ``obs`` dict) into a
``(12, 16, 16)`` float32 tensor suitable for LightZero convolutional AlphaZero /
MuZero / Gumbel MuZero models, plus a fast scalar evaluator in ``[-1, 1]``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np

from orbit_wars_bot.physics import (
    BOARD_SIZE,
    CENTER,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_SHIP_SPEED,
    SUN_RADIUS,
    fleet_speed,
    is_orbiting_planet,
)

GRID_SIZE: int = 16
NUM_CHANNELS: int = 12
OBS_SHAPE: Tuple[int, int, int] = (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

_CELL_SIZE: float = BOARD_SIZE / GRID_SIZE


def _precompute_sun_plane() -> np.ndarray:
    plane = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for row in range(GRID_SIZE):
        cy = (row + 0.5) * _CELL_SIZE
        for col in range(GRID_SIZE):
            cx = (col + 0.5) * _CELL_SIZE
            d = math.hypot(cx - CENTER, cy - CENTER)
            if d <= SUN_RADIUS + _CELL_SIZE * 0.5:
                plane[row, col] = max(0.0, 1.0 - d / (SUN_RADIUS + _CELL_SIZE * 0.5))
    return plane


_SUN_PLANE: np.ndarray = _precompute_sun_plane()


def _extract_fields(state: Any) -> Tuple[Sequence, Sequence, set, int, int, float]:
    """Extract (planets, fleets, comet_ids, step, max_steps, ship_speed) from sim or obs."""
    if hasattr(state, "step_count"):
        return (
            state.planets,
            state.fleets,
            set(state.comet_planet_ids),
            int(state.step_count),
            int(state.max_steps),
            float(state.ship_speed),
        )
    if isinstance(state, dict):
        return (
            state.get("planets", ()),
            state.get("fleets", ()),
            set(state.get("comet_planet_ids", ()) or ()),
            int(state.get("step", 0)),
            DEFAULT_EPISODE_STEPS,
            DEFAULT_SHIP_SPEED,
        )
    return (
        getattr(state, "planets", ()),
        getattr(state, "fleets", ()),
        set(getattr(state, "comet_planet_ids", ()) or ()),
        int(getattr(state, "step", 0)),
        DEFAULT_EPISODE_STEPS,
        DEFAULT_SHIP_SPEED,
    )


def _canonical_xy(x: float, y: float, player: int) -> Tuple[float, float]:
    """Apply 180-degree rotational symmetry canonicalization for player 1."""
    if player == 1:
        return BOARD_SIZE - x, BOARD_SIZE - y
    return x, y


def _to_grid_rc(x: float, y: float) -> Tuple[int, int]:
    col = int(x / _CELL_SIZE)
    row = int(y / _CELL_SIZE)
    if col < 0:
        col = 0
    elif col >= GRID_SIZE:
        col = GRID_SIZE - 1
    if row < 0:
        row = 0
    elif row >= GRID_SIZE:
        row = GRID_SIZE - 1
    return row, col


def encode_observation(
    state: Any,
    player: int = 0,
    channel_last: bool = False,
) -> np.ndarray:
    """Encode an Orbit Wars state into a player-canonicalized ``(12, 16, 16)`` tensor.

    Channels:
        0: Friendly planet ships (``log1p(ships) / 6.0``)
        1: Enemy planet ships (``log1p(ships) / 6.0``)
        2: Neutral planet ships (``log1p(ships) / 6.0``)
        3: Friendly planet production (``prod / 5.0``)
        4: Enemy planet production (``prod / 5.0``)
        5: Neutral planet production (``prod / 5.0``)
        6: Motion type (``0.5`` orbiting planet, ``1.0`` comet)
        7: Friendly fleet ships (``log1p(ships) / 6.0``)
        8: Enemy fleet ships (``log1p(ships) / 6.0``)
        9: Net canonical fleet momentum x (friendly positive, enemy negative)
        10: Net canonical fleet momentum y (friendly positive, enemy negative)
        11: Global time progress + central sun spatial mask
    """
    planets, fleets, comet_ids, step, max_steps, ship_speed = _extract_fields(state)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)

    for p in planets:
        pid = int(p[0])
        owner = int(p[1])
        px = float(p[2])
        py = float(p[3])
        if px < 0.0 or px > BOARD_SIZE or py < 0.0 or py > BOARD_SIZE:
            continue
        cx, cy = _canonical_xy(px, py, player)
        r, c = _to_grid_rc(cx, cy)
        ships_val = min(1.5, math.log1p(max(0.0, float(p[5]))) / 6.0)
        prod_val = float(p[6]) / 5.0

        if owner == player:
            obs[0, r, c] += ships_val
            obs[3, r, c] += prod_val
        elif owner == -1:
            obs[2, r, c] += ships_val
            obs[5, r, c] += prod_val
        else:
            obs[1, r, c] += ships_val
            obs[4, r, c] += prod_val

        if pid in comet_ids:
            obs[6, r, c] = max(obs[6, r, c], 1.0)
        elif is_orbiting_planet(px, py, float(p[4])):
            obs[6, r, c] = max(obs[6, r, c], 0.5)

    vel_sign_flip = -1.0 if player == 1 else 1.0
    for f in fleets:
        owner = int(f[1])
        fx = float(f[2])
        fy = float(f[3])
        if fx < 0.0 or fx > BOARD_SIZE or fy < 0.0 or fy > BOARD_SIZE:
            continue
        cx, cy = _canonical_xy(fx, fy, player)
        r, c = _to_grid_rc(cx, cy)
        ships = int(f[6])
        ships_val = min(1.5, math.log1p(max(0, ships)) / 6.0)
        spd = fleet_speed(ships, ship_speed) / max(1.0, ship_speed)
        ang = float(f[4])
        vx = math.cos(ang) * spd * vel_sign_flip
        vy = math.sin(ang) * spd * vel_sign_flip

        if owner == player:
            obs[7, r, c] += ships_val
            obs[9, r, c] += vx * ships_val
            obs[10, r, c] += vy * ships_val
        else:
            obs[8, r, c] += ships_val
            obs[9, r, c] -= vx * ships_val
            obs[10, r, c] -= vy * ships_val

    # Squash momentum planes smoothly to [-1, 1]
    obs[9] = np.tanh(obs[9])
    obs[10] = np.tanh(obs[10])

    # Channel 11: D4-invariant global time progress + central sun spatial cue + power balance
    progress = float(min(1.0, max(0.0, step / max(1, max_steps))))
    bal = 0.5 * (1.0 + evaluate_state(state, player=player))
    obs[11] = progress * 0.30 + _SUN_PLANE * 0.40 + bal * 0.30

    if channel_last:
        return np.transpose(obs, (1, 2, 0)).copy()
    return obs


def augment_obs_d4(obs: np.ndarray, sym_idx: int) -> np.ndarray:
    """Apply one of the 8 exact D4 dihedral board symmetries (``sym_idx in 0..7``) to a ``(12, 16, 16)`` tensor.

    Transforms scalar channels ``0..8, 11`` spatially and rotates/reflects the 2D
    fleet momentum vector field ``(obs[9]=vx, obs[10]=vy)`` consistently.
    """
    k = sym_idx % 4
    do_flip = (sym_idx // 4) % 2 == 1
    out = obs.copy()

    if k:
        out = np.rot90(out, k=k, axes=(1, 2)).copy()
        # Grid axes are (row=y, col=x). Each np.rot90(..., k=1) maps (r, c) -> (N-1-c, r),
        # i.e. (new_x, new_y) = (old_y, -old_x) in grid coordinates.
        for _ in range(k):
            vx = out[9].copy()
            vy = out[10].copy()
            out[9] = vy
            out[10] = -vx

    if do_flip:
        # Horizontal flip along col=x axis: (r, c) -> (r, N-1-c) => vx -> -vx
        out = out[:, :, ::-1].copy()
        out[9] = -out[9]

    return out


def evaluate_state(state: Any, player: int = 0) -> float:
    """Fast heuristic state value estimate in ``[-1.0, 1.0]`` from ``player``'s view.

    Works accurately in both 2-player 1v1 (zero-sum symmetric) and 4-player FFA
    (compares ``player`` against the strongest rival and active field average).
    """
    if hasattr(state, "done") and state.done:
        w = state.winner()
        if w == player:
            return 1.0
        if w == -1:
            return 0.0
        return -1.0

    planets, fleets, comet_ids, step, max_steps, _ = _extract_fields(state)
    ships_by_p: dict[int, float] = {}
    prod_by_p: dict[int, float] = {}
    planets_by_p: dict[int, int] = {}

    for p in planets:
        owner = int(p[1])
        if owner < 0:
            continue
        ships = float(p[5])
        prod = float(p[6])
        is_comet = int(p[0]) in comet_ids
        # Comets eventually expire so discount their long-term production
        eff_prod = prod * (0.35 if is_comet else 1.0)
        ships_by_p[owner] = ships_by_p.get(owner, 0.0) + ships
        prod_by_p[owner] = prod_by_p.get(owner, 0.0) + eff_prod
        planets_by_p[owner] = planets_by_p.get(owner, 0) + 1

    for f in fleets:
        owner = int(f[1])
        if owner < 0:
            continue
        ships_by_p[owner] = ships_by_p.get(owner, 0.0) + float(f[6])

    my_ships = ships_by_p.get(player, 0.0)
    opp_ids = [pid for pid, s in ships_by_p.items() if pid != player and s > 0.0]

    if my_ships <= 0.0 and opp_ids:
        return -1.0
    if not opp_ids and my_ships > 0.0:
        return 1.0
    if my_ships <= 0.0 and not opp_ids:
        return 0.0

    remaining = max(0, max_steps - step)
    # Effective value of 1 point of production scales down as the match nears step 500
    prod_horizon = min(36.0, float(remaining) * 0.65)

    def _power(pid: int) -> float:
        return (
            ships_by_p.get(pid, 0.0)
            + prod_by_p.get(pid, 0.0) * prod_horizon
            + planets_by_p.get(pid, 0) * 2.5
        )

    my_power = _power(player)
    opp_powers = [_power(pid) for pid in opp_ids]
    max_opp_power = max(opp_powers)
    if len(opp_powers) == 1:
        # Exact zero-sum symmetry in 2-player games
        ref_opp_power = max_opp_power
    else:
        # In 4P FFA, winning requires beating the strongest rival while staying ahead of the field
        mean_opp_power = sum(opp_powers) / len(opp_powers)
        ref_opp_power = 0.72 * max_opp_power + 0.28 * mean_opp_power

    denom = max(10.0, my_power + ref_opp_power)
    advantage = (my_power - ref_opp_power) / denom
    # Scale so a ~2:1 advantage maps around ~0.6-0.75 without saturating too early
    return float(math.tanh(2.4 * advantage))
