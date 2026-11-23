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


def test_d4_symmetry_augmentation_equivariance():
    import math
    from orbit_wars_bot.features import augment_obs_d4

    ang = 0.4
    s0 = {
        "step": 10,
        "planets": [[0, 0, 20.0, 30.0, 2.0, 25, 3], [1, 1, 80.0, 70.0, 2.0, 25, 3]],
        "fleets": [[0, 0, 20.0, 30.0, ang, 0, 15]],
        "comet_planet_ids": [],
    }
    t0 = encode_observation(s0, player=0)

    # Horizontal flip: x' = 100 - x, y' = y, ang' = pi - ang
    s_flip = {
        "step": 10,
        "planets": [[0, 0, 80.0, 30.0, 2.0, 25, 3], [1, 1, 20.0, 70.0, 2.0, 25, 3]],
        "fleets": [[0, 0, 80.0, 30.0, math.pi - ang, 0, 15]],
        "comet_planet_ids": [],
    }
    assert np.allclose(encode_observation(s_flip, player=0), augment_obs_d4(t0, 4), atol=1e-5)

    # 90-deg grid rotation (np.rot90 k=1): x' = y, y' = 100 - x, vx' = vy, vy' = -vx
    s_rot = {
        "step": 10,
        "planets": [[0, 0, 30.0, 80.0, 2.0, 25, 3], [1, 1, 70.0, 20.0, 2.0, 25, 3]],
        "fleets": [[0, 0, 30.0, 80.0, math.atan2(-math.cos(ang), math.sin(ang)), 0, 15]],
        "comet_planet_ids": [],
    }
    assert np.allclose(encode_observation(s_rot, player=0), augment_obs_d4(t0, 1), atol=1e-5)


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


def test_4p_ffa_state_evaluator_and_tactical_context():
    sim = FastOrbitSim.from_seed(55, num_agents=4, max_steps=200)
    # Symmetric 4P start should evaluate near 0.0 (not heavily negative!)
    for p in range(4):
        assert abs(evaluate_state(sim, player=p)) < 1e-6

    ctx = build_tactical_context(sim, player=0)
    assert ctx.is_ffa is True
    assert len(ctx.active_opponents) == 3
    assert ctx.leader_enemy_id in (1, 2, 3)
    assert ctx.weakest_enemy_id in (1, 2, 3)

    # Give P0 a clear lead over all 3 opponents -> value should be strongly positive
    for p in sim.planets:
        if p[1] == 0:
            p[5] += 250
    assert evaluate_state(sim, player=0) > 0.5
    assert evaluate_state(sim, player=1) < -0.3


def test_4p_online_mcts_handles_four_players_under_budget():
    import time
    from orbit_wars_bot.mcts_agent import OnlineMCTSAgent

    sim = FastOrbitSim.from_seed(91, num_agents=4, max_steps=120)
    for _ in range(12):
        sim.step([heuristic_moves(sim, p) for p in range(4)])

    obs = sim.to_observation(player=2)
    agent = OnlineMCTSAgent(num_simulations=10, time_budget_sec=0.20)
    t0 = time.perf_counter()
    moves = agent.act(obs)
    elapsed = time.perf_counter() - t0
    assert isinstance(moves, list)
    assert elapsed < 0.5
    pby = {p[0]: p for p in sim.planets}
    for mv in moves:
        assert pby[mv[0]][1] == 2
        assert mv[2] > 0


def test_defense_never_evacuates_saved_planet_or_double_retakes():
    from orbit_wars_bot.macro_actions import _do_defend_urgent, _do_evacuate_doomed

    # Construct a minimal deterministic state:
    # Planet 0 (mine, 5 ships, prod 3) at (20, 20) threatened by enemy fleet of 15 arriving in 1 tick
    # Planet 1 (mine, 40 ships, prod 3) at (24, 20) right next door -> can save Planet 0 in 1 tick!
    # Planet 2 (enemy, 20 ships, prod 2) at (80, 80)
    planets = [
        [0, 0, 20.0, 20.0, 2.0, 5, 3],
        [1, 0, 24.5, 20.0, 2.0, 40, 3],
        [2, 1, 80.0, 80.0, 2.0, 20, 2],
    ]
    # Enemy fleet heading straight into Planet 0 from (20.0, 17.5) upward (angle = pi/2)
    fleets = [[0, 1, 20.0, 17.0, 1.5707963, 2, 15]]
    sim = FastOrbitSim(
        planets=[p[:] for p in planets],
        initial_planets=[p[:] for p in planets],
        fleets=fleets,
        comets=[],
        comet_planet_ids=[],
        angular_velocity=0.0,
        next_fleet_id=1,
        step_count=30,
        num_agents=2,
    )
    ctx = build_tactical_context(sim, player=0)
    assert any(int(p[0]) == 0 for p, _ in ctx.threatened_my)
    avail = dict(ctx.surplus)
    raw = dict(ctx.raw_surplus)
    moves = []
    saved = _do_defend_urgent(ctx, avail, moves, raw_available=raw)
    assert 0 in saved, "Planet 1 right next door should save Planet 0 in time"
    evac = _do_evacuate_doomed(ctx, avail, moves, saved_pids=saved, raw_available=raw)
    assert evac is False, "Must NOT evacuate a planet that was just saved!"
    assert all(m[0] != 0 for m in moves), "Saved planet 0 should not evacuate its own ships"


