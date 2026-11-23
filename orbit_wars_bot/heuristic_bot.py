"""Strong rule-based Orbit Wars baseline bot (2P 1v1 and 4P FFA).

Combines:
1. Exact orbital & comet lead intercepts (via ``orbit_wars_bot.physics``).
2. Atomic threat defense (never double-retaking or feeding lost fights) + last-tick evacuation.
3. Neutral & enemy-vs-enemy crash-exploit vulture sniping.
4. Unified neutral + enemy ROI allocation with static-planet durability, sector safety,
   opening reservation patience, and massive opponent-elimination priority.
5. Mode-aware combat (disciplined 4P FFA diplomacy vs decisive 2P pressure).
6. Short-hop anti-ping-pong backline-to-frontline stack consolidation.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Set

from orbit_wars_bot.macro_actions import (
    MacroAction,
    TacticalContext,
    _do_attack_enemy,
    _do_defend_urgent,
    _do_evacuate_doomed,
    _do_feed_frontline,
    _do_unified_expand_and_attack,
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
    raw_available = dict(ctx.raw_surplus)
    moves: List[List[float]] = []
    claimed: Set[int] = set()

    # 1. Atomically defend saveable threatened planets (never double-retake or trickle into lost fights)
    saved_pids = _do_defend_urgent(
        ctx, available, moves, raw_available=raw_available, claimed=claimed
    )

    # 2. Evacuate unsaveable doomed planets ONLY on the final tick before impact
    _do_evacuate_doomed(
        ctx, available, moves, saved_pids=saved_pids, raw_available=raw_available
    )

    # 3. Vulture-snipe neutrals and enemy-vs-enemy battles 1-4 ticks after impact
    _do_vulture_snipe(
        ctx, available, moves, raw_available=raw_available, claimed=claimed
    )

    strength_ratio = ctx.my_total_ships / max(1, ctx.leader_ships)

    # 4. Endgame push: in the final 42 ticks, protect ship lead or launch high-ROI close steals
    if ctx.remaining_steps <= 42:
        if ctx.enemy_planets:
            if strength_ratio > 1.15:
                _do_attack_enemy(
                    ctx,
                    available,
                    moves,
                    target_mode="roi",
                    max_targets=2,
                    min_roi=1.05,
                    raw_available=raw_available,
                    claimed=claimed,
                )
            else:
                _do_attack_enemy(
                    ctx,
                    available,
                    moves,
                    target_mode="closest",
                    max_targets=2,
                    min_roi=0.75,
                    raw_available=raw_available,
                    claimed=claimed,
                )
        return moves

    # 5. Unified neutral + enemy ROI allocation (mode-aware for 4P FFA vs 2P 1v1)
    if ctx.is_ffa:
        if ctx.step < 28:
            enemy_mult = 0.72
            min_enemy_roi = 2.15
            max_targets = 2 if len(ctx.my_planets) <= 2 else 3
        elif strength_ratio >= 1.32 and ctx.my_prod >= ctx.leader_prod and len(ctx.my_planets) >= 4:
            enemy_mult = 1.25
            min_enemy_roi = 1.05
            max_targets = 4
        elif strength_ratio >= 0.90:
            enemy_mult = 1.02
            min_enemy_roi = 1.38
            max_targets = 3
        else:
            # Behind in 4P FFA: prioritize safe static expansion & cheap cleanups/eliminations
            enemy_mult = 0.85
            min_enemy_roi = 1.75
            max_targets = 3
    else:
        # 2P 1v1 (or final 2 survivors of 4P FFA)
        if strength_ratio >= 1.45 and len(ctx.my_planets) >= 3 and sum(raw_available.values()) >= 22:
            _do_attack_enemy(
                ctx,
                available,
                moves,
                target_mode="roi",
                all_in=True,
                min_roi=0.80,
                raw_available=raw_available,
                claimed=claimed,
            )
            enemy_mult = 1.20
            min_enemy_roi = 1.00
            max_targets = 2
        elif strength_ratio >= 0.92:
            enemy_mult = 1.18
            min_enemy_roi = 1.05
            max_targets = 3 if len(ctx.my_planets) >= 3 else 2
        else:
            enemy_mult = 0.95
            min_enemy_roi = 1.40
            max_targets = 3 if len(ctx.my_planets) >= 3 else 2

    _do_unified_expand_and_attack(
        ctx,
        available,
        raw_available,
        moves,
        claimed,
        max_total_targets=max_targets,
        enemy_mult=enemy_mult,
        min_enemy_roi=min_enemy_roi,
        enemy_target_mode="roi",
        allow_reserve_wait=True,
    )

    # 6. Short-hop anti-ping-pong backline-to-frontline consolidation
    _do_feed_frontline(ctx, available, moves, raw_available=raw_available, min_stack=24)

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

    total_surplus = sum(ctx.raw_surplus.values())
    if total_surplus < 3 and not ctx.doomed_my:
        return int(MacroAction.HOLD_ACCUMULATE)

    if mask[MacroAction.BALANCED_MACRO]:
        return int(MacroAction.BALANCED_MACRO)

    if mask[MacroAction.DEFEND_URGENT]:
        return int(MacroAction.DEFEND_URGENT)
    if mask[MacroAction.COUNTER_EVACUATE]:
        return int(MacroAction.COUNTER_EVACUATE)
    if mask[MacroAction.VULTURE_SNIPE]:
        return int(MacroAction.VULTURE_SNIPE)

    strength_ratio = ctx.my_total_ships / max(1, ctx.leader_ships)
    if not ctx.is_ffa and strength_ratio > 1.45 and mask[MacroAction.COORDINATED_ALL_IN]:
        return int(MacroAction.COORDINATED_ALL_IN)

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
