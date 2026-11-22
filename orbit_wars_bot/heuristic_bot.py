"""Strong rule-based Orbit Wars baseline bot.

Combines:
1. Exact orbital & comet lead intercepts (via ``orbit_wars_bot.physics``).
2. Threat detection and urgent defense / doomed-planet evacuation.
3. Vulture sniping of neutrals weakened by enemy fleets.
4. Payback-horizon ROI expansion across static, orbiting, and comet bodies.
5. Coordinated enemy strikes and backline-to-frontline stack feeding.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional

from orbit_wars_bot.macro_actions import (
    MacroAction,
    TacticalContext,
    _do_attack_enemy,
    _do_defend_urgent,
    _do_evacuate_doomed,
    _do_expand_neutrals,
    _do_feed_frontline,
    _do_vulture_snipe,
    build_tactical_context,
    get_legal_action_mask,
)
from orbit_wars_bot.physics import CENTER, ROTATION_RADIUS_LIMIT


def heuristic_moves(
    state: Any,
    player: int,
    ctx: Optional[TacticalContext] = None,
) -> List[List[float]]:
    """Compute strong rule-based launch orders for ``player``."""
    if ctx is None:
        ctx = build_tactical_context(state, player)
    if not ctx.my_planets:
        return []

    available = dict(ctx.surplus)
    moves: List[List[float]] = []

    # 1. Defend threatened friendly planets first
    _do_defend_urgent(ctx, available, moves)

    # 2. Evacuate any planet still doomed to fall immediately
    _do_evacuate_doomed(ctx, available, moves)

    # 3. Vulture-snipe neutrals about to be weakened/captured by the enemy
    _do_vulture_snipe(ctx, available, moves)

    # 4. Endgame push: in the final 45 ticks, focus on ships & enemy captures
    if ctx.remaining_steps <= 45:
        if ctx.my_total_ships > ctx.enemy_total_ships * 1.15:
            _do_attack_enemy(ctx, available, moves, target_mode="weakest", max_targets=2)
        else:
            _do_attack_enemy(ctx, available, moves, target_mode="closest", max_targets=2)
        return moves

    # 5. Early & mid-game ROI expansion (static + orbiting + comets)
    # Wait slightly on tick 0-2 if we have few ships so we launch a faster stack
    total_surplus = sum(available.values())
    if total_surplus >= 4:
        max_expansions = 3 if len(ctx.my_planets) >= 3 else 2
        _do_expand_neutrals(ctx, available, moves, max_targets=max_expansions, filter_mode="roi")

    # 6. Attack enemy when ahead or when an enemy planet is vulnerable
    rem_surplus = sum(available.values())
    if ctx.enemy_planets and rem_surplus >= 6:
        if ctx.my_total_ships > ctx.enemy_total_ships * 1.45 and len(ctx.my_planets) >= 3:
            _do_attack_enemy(ctx, available, moves, target_mode="weakest", all_in=True)
        elif ctx.my_total_ships >= ctx.enemy_total_ships * 0.95:
            _do_attack_enemy(ctx, available, moves, target_mode="high_prod", max_targets=1)
            _do_attack_enemy(ctx, available, moves, target_mode="weakest", max_targets=1)
        else:
            _do_attack_enemy(ctx, available, moves, target_mode="weakest", max_targets=1)

    # 7. Feed backline surplus forward to frontline planets
    _do_feed_frontline(ctx, available, moves)

    return moves


def choose_heuristic_macro_action(
    state: Any,
    player: int,
    ctx: Optional[TacticalContext] = None,
) -> int:
    """Select the highest-priority legal ``MacroAction`` for ``player``."""
    if ctx is None:
        ctx = build_tactical_context(state, player)
    mask = get_legal_action_mask(state, player, ctx=ctx)

    if mask[MacroAction.DEFEND_URGENT]:
        return int(MacroAction.DEFEND_URGENT)
    if mask[MacroAction.COUNTER_EVACUATE]:
        return int(MacroAction.COUNTER_EVACUATE)
    if mask[MacroAction.VULTURE_SNIPE]:
        return int(MacroAction.VULTURE_SNIPE)

    total_surplus = sum(ctx.surplus.values())
    if total_surplus < 6:
        return int(MacroAction.HOLD_ACCUMULATE)

    if ctx.remaining_steps <= 55 and mask[MacroAction.COORDINATED_ALL_IN]:
        return int(MacroAction.COORDINATED_ALL_IN)

    if ctx.my_total_ships > ctx.enemy_total_ships * 1.4 and mask[MacroAction.COORDINATED_ALL_IN]:
        return int(MacroAction.COORDINATED_ALL_IN)

    if mask[MacroAction.BALANCED_MACRO]:
        return int(MacroAction.BALANCED_MACRO)
    if mask[MacroAction.EXPAND_BEST_ROI_MULTI]:
        return int(MacroAction.EXPAND_BEST_ROI_MULTI)
    if mask[MacroAction.ATTACK_WEAKEST_ENEMY]:
        return int(MacroAction.ATTACK_WEAKEST_ENEMY)
    return int(MacroAction.HOLD_ACCUMULATE)


def starter_moves(state: Any, player: int) -> List[List[float]]:
    """Mirror Kaggle's built-in ``starter`` agent on a ``FastOrbitSim`` or ``obs``."""
    planets = state.planets if hasattr(state, "planets") else state.get("planets", [])
    static_targets = []
    my_planets = []
    for p in planets:
        if p[1] == player:
            my_planets.append(p)
        else:
            orbital_r = math.hypot(p[2] - CENTER, p[3] - CENTER)
            if orbital_r + p[4] >= ROTATION_RADIUS_LIMIT:
                static_targets.append(p)

    moves: List[List[float]] = []
    for mp in my_planets:
        if mp[5] <= 0:
            continue
        closest = None
        min_dist = float("inf")
        for t in static_targets:
            d = math.hypot(mp[2] - t[2], mp[3] - t[3])
            if d < min_dist:
                min_dist = d
                closest = t
        if closest is not None:
            ships = int(mp[5]) // 2
            if ships >= 20:
                angle = math.atan2(closest[3] - mp[3], closest[2] - mp[2])
                moves.append([int(mp[0]), float(angle), ships])
    return moves


def random_moves(state: Any, player: int, rng: Optional[random.Random] = None) -> List[List[float]]:
    """Mirror Kaggle's built-in ``random`` agent on a ``FastOrbitSim`` or ``obs``."""
    rng_obj = rng if rng is not None else random
    planets = state.planets if hasattr(state, "planets") else state.get("planets", [])
    moves: List[List[float]] = []
    for p in planets:
        if p[1] == player and p[5] > 0:
            ships = int(p[5]) // 2
            if ships >= 20:
                moves.append([int(p[0]), rng_obj.uniform(0.0, 2.0 * math.pi), ships])
    return moves


def heuristic_agent(obs: Any, config: Any = None) -> List[List[float]]:
    """Official Kaggle-compatible callback for the heuristic baseline bot."""
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    return heuristic_moves(obs, int(player))
