"""Tests for spatial feature encoding, heuristic evaluation, macro-actions, and baseline bot."""

from __future__ import annotations

import numpy as np

from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.features import OBS_SHAPE, encode_observation, evaluate_state
from orbit_wars_bot.heuristic_bot import heuristic_moves, starter_moves
from orbit_wars_bot.macro_actions import (
    NUM_MACRO_ACTIONS,
    MacroAction,
    build_tactical_context,
    execute_macro_action,
    get_legal_action_mask,
)


def test_feature_encoding_shape_and_player_symmetry():
    sim = FastOrbitSim.from_seed(2025, max_steps=200)
    t0 = encode_observation(sim, player=0)
    t1 = encode_observation(sim, player=1)

    assert t0.shape == OBS_SHAPE == (12, 16, 16)
    assert t0.dtype == np.float32
    assert np.all(np.isfinite(t0))
    assert np.all(np.isfinite(t1))

    # Symmetric initial map -> canonical tensors for P0 and P1 should have matching totals
    assert np.isclose(t0[0].sum(), t1[0].sum(), atol=1e-5)  # friendly ships
    assert np.isclose(t0[1].sum(), t1[1].sum(), atol=1e-5)  # enemy ships
    assert np.isclose(t0[3].sum(), t1[3].sum(), atol=1e-5)  # friendly prod
    assert np.isclose(t0[4].sum(), t1[4].sum(), atol=1e-5)  # enemy prod

    # Channel-last option
    t0_cl = encode_observation(sim, player=0, channel_last=True)
    assert t0_cl.shape == (16, 16, 12)


def test_state_evaluator_symmetry_and_dominance():
    sim = FastOrbitSim.from_seed(33, max_steps=200)
    v0 = evaluate_state(sim, player=0)
    v1 = evaluate_state(sim, player=1)
    assert abs(v0) < 1e-6
    assert abs(v1) < 1e-6

    # Give P0 a large ship lead
    for p in sim.planets:
        if p[1] == 0:
            p[5] += 200
    assert evaluate_state(sim, player=0) > 0.5
    assert evaluate_state(sim, player=1) < -0.5


def test_all_macro_actions_produce_valid_orders():
    sim = FastOrbitSim.from_seed(88, max_steps=200)
    # Advance past step 50 so comets exist and ships have accumulated
    for _ in range(55):
        m0 = heuristic_moves(sim, 0)
        m1 = starter_moves(sim, 1)
        sim.step([m0, m1])
        if sim.done:
            break

    if sim.done:
        sim = FastOrbitSim.from_seed(89, max_steps=200)
        for _ in range(25):
            sim.step([[], []])

    ctx = build_tactical_context(sim, 0)
    mask = get_legal_action_mask(sim, 0, ctx=ctx)
    assert mask.shape == (NUM_MACRO_ACTIONS,)
    assert mask[MacroAction.HOLD_ACCUMULATE] == 1

    planet_by_id = {p[0]: p for p in sim.planets}
    for action_id in range(NUM_MACRO_ACTIONS):
        moves = execute_macro_action(sim, 0, action_id, ctx=ctx)
        spent = {}
        for mv in moves:
            assert len(mv) == 3
            pid, angle, ships = mv
            assert pid in planet_by_id
            assert planet_by_id[pid][1] == 0
            assert isinstance(ships, int) and ships > 0
            assert np.isfinite(angle)
            spent[pid] = spent.get(pid, 0) + ships
        for pid, total in spent.items():
            assert total <= planet_by_id[pid][5]


def test_heuristic_bot_beats_starter_on_fast_sim():
    wins = 0
    for seed in (10, 20, 30, 40):
        sim = FastOrbitSim.from_seed(seed, max_steps=300)
        while not sim.done:
            m0 = heuristic_moves(sim, 0)
            m1 = starter_moves(sim, 1)
            sim.step([m0, m1])
        if sim.winner() == 0:
            wins += 1
    assert wins >= 3, f"Expected heuristic bot to beat starter on >=3/4 seeds, got {wins}/4"
