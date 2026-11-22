"""Tests for Orbit Wars physics, intercept solvers, and exact Kaggle simulator parity."""

from __future__ import annotations

import math
import random

import pytest
from kaggle_environments import make

from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.physics import (
    CENTER,
    SUN_RADIUS,
    build_comet_lookup,
    fleet_speed,
    is_orbiting_planet,
    predict_fleet_arrivals,
    predict_planet_position,
    segment_hits_sun,
    simulate_planet_future,
    solve_intercept,
    swept_pair_hit,
)


def test_fleet_speed_formula():
    assert fleet_speed(1, 6.0) == pytest.approx(1.0)
    assert fleet_speed(1000, 6.0) == pytest.approx(6.0)
    assert fleet_speed(5000, 6.0) == pytest.approx(6.0)
    assert 1.0 < fleet_speed(10, 6.0) < fleet_speed(50, 6.0) < fleet_speed(200, 6.0) < 6.0


def test_swept_pair_and_sun_collision():
    # Direct head-on crossing between fleet and planet
    assert swept_pair_hit((0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (0.0, 0.0), 1.0)
    # Parallel non-intersecting paths
    assert not swept_pair_hit((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0), 1.0)
    # Segment across board center hits sun
    assert segment_hits_sun((20.0, 50.0), (80.0, 50.0))
    # Segment near edge avoids sun
    assert not segment_hits_sun((10.0, 10.0), (90.0, 10.0))


def test_empirical_orbital_phase_offset_matches_kaggle():
    env = make("orbit_wars", configuration={"seed": 42, "episodeSteps": 40})
    obs = env.reset(2)
    kobs0 = obs[0].observation
    init_by_id = {p[0]: p for p in kobs0.initial_planets}
    orbiting = [p for p in kobs0.planets if is_orbiting_planet(p[2], p[3], p[4])]
    assert orbiting, "Expected at least one orbiting planet"

    # Predict 1..15 ticks ahead from step 0 and compare against kaggle_environments
    start_step = kobs0.step
    ang_vel = kobs0.angular_velocity
    for dt in range(1, 16):
        obs = env.step([[], []])
        kobs = obs[0].observation
        cur_by_id = {p[0]: p for p in kobs.planets}
        for op in orbiting:
            pred = predict_planet_position(op, dt, start_step, ang_vel, init_by_id)
            actual = (cur_by_id[op[0]][2], cur_by_id[op[0]][3])
            assert pred is not None
            assert math.hypot(pred[0] - actual[0], pred[1] - actual[1]) < 1e-9


def test_intercept_solver_hits_orbiting_and_static_targets():
    sim = FastOrbitSim.from_seed(77, max_steps=200)
    # Advance a few steps so home planet has plenty of ships and orbits are in motion
    for _ in range(12):
        sim.step([[], []])

    home = next(p for p in sim.planets if p[1] == 0)
    init_by_id = {p[0]: p for p in sim.initial_planets}
    comet_lookup = build_comet_lookup(sim.comets)

    # Test intercepting both a static neutral and an orbiting neutral
    neutrals = [p for p in sim.planets if p[1] == -1]
    orbiting_neutrals = [p for p in neutrals if is_orbiting_planet(p[2], p[3], p[4])]
    static_neutrals = [p for p in neutrals if not is_orbiting_planet(p[2], p[3], p[4])]
    assert orbiting_neutrals and static_neutrals

    hits_verified = 0
    for target in orbiting_neutrals[:3] + static_neutrals[:3]:
        ships = min(int(home[5]), int(target[5]) + 5)
        sol = solve_intercept(
            home,
            target,
            ships,
            sim.planets,
            sim.step_count,
            sim.angular_velocity,
            init_by_id,
            comet_lookup=comet_lookup,
            max_eta=40,
            max_speed=sim.ship_speed,
        )
        if sol is None:
            continue
        test_sim = sim.clone()
        tid = int(target[0])
        test_sim.step([[[int(home[0]), sol.angle, ships]], []])
        for _ in range(sol.eta - 1):
            test_sim.step([[], []])
        t_after = next(p for p in test_sim.planets if p[0] == tid)
        # Target must have received the fleet (either captured or reduced garrison)
        assert t_after[1] == 0 or t_after[5] < target[5]
        hits_verified += 1
    assert hits_verified >= 2


def test_fleet_arrival_prediction_and_planet_future():
    sim = FastOrbitSim.from_seed(99, max_steps=150)
    for _ in range(15):
        sim.step([[], []])

    home = next(p for p in sim.planets if p[1] == 0)
    target = min(
        (p for p in sim.planets if p[1] == -1),
        key=lambda p: math.hypot(p[2] - home[2], p[3] - home[3]),
    )
    init_by_id = {p[0]: p for p in sim.initial_planets}
    ships = min(int(home[5]), int(target[5]) + 3)
    sol = solve_intercept(
        home,
        target,
        ships,
        sim.planets,
        sim.step_count,
        sim.angular_velocity,
        init_by_id,
    )
    assert sol is not None

    # Launch on tick 0 so fleet is in flight (if eta > 1)
    sim.step([[[int(home[0]), sol.angle, ships]], []])
    if sol.eta > 1:
        arrivals = predict_fleet_arrivals(
            sim.fleets,
            sim.planets,
            sim.step_count,
            sim.angular_velocity,
            {p[0]: p for p in sim.initial_planets},
            comet_lookup=build_comet_lookup(sim.comets),
        )
        tid = int(target[0])
        assert any(a.owner == 0 and a.ships == ships for a in arrivals[tid])
        t_cur = next(p for p in sim.planets if p[0] == tid)
        fut = simulate_planet_future(t_cur, arrivals[tid], horizon=sol.eta + 2)
        if ships > int(t_cur[5]):
            assert fut.will_be_captured_by == 0
            assert fut.final_owner == 0


@pytest.mark.parametrize("seed", [11, 123, 404])
def test_fast_sim_exact_tick_for_tick_parity_with_kaggle(seed: int):
    """Verify 100% tick-for-tick parity including comet spawns at steps 50 and 150."""
    episode_steps = 160
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": episode_steps})
    obs = env.reset(2)
    sim = FastOrbitSim.from_seed(seed, num_agents=2, max_steps=episode_steps)

    # Initial state must match identically
    kobs0 = obs[0].observation
    assert sim.angular_velocity == pytest.approx(kobs0.angular_velocity, abs=1e-12)
    assert len(sim.planets) == len(kobs0.planets)

    rng = random.Random(seed + 1000)
    comets_seen = False

    for t in range(episode_steps):
        a0, a1 = [], []
        for p in kobs0.planets if t == 0 else obs[0].observation.planets:
            if p[1] == 0 and p[5] >= 10 and rng.random() < 0.2:
                a0.append([p[0], rng.uniform(0.0, 2.0 * math.pi), max(1, int(p[5]) // 2)])
            elif p[1] == 1 and p[5] >= 10 and rng.random() < 0.2:
                a1.append([p[0], rng.uniform(0.0, 2.0 * math.pi), max(1, int(p[5]) // 2)])

        obs = env.step([a0, a1])
        sim.step([a0, a1])
        kobs = obs[0].observation

        assert sim.step_count == kobs.step
        assert len(sim.planets) == len(kobs.planets)
        assert len(sim.fleets) == len(kobs.fleets)
        assert sim.comet_planet_ids == list(kobs.comet_planet_ids)
        if sim.comet_planet_ids:
            comets_seen = True

        for sp, kp in zip(sim.planets, kobs.planets):
            assert sp[0] == kp[0]
            assert sp[1] == kp[1]
            assert sp[5] == kp[5]
            assert sp[6] == kp[6]
            assert abs(sp[2] - kp[2]) < 1e-9
            assert abs(sp[3] - kp[3]) < 1e-9

        for sf, kf in zip(sim.fleets, kobs.fleets):
            assert sf[0] == kf[0]
            assert sf[1] == kf[1]
            assert sf[6] == kf[6]
            assert abs(sf[2] - kf[2]) < 1e-9
            assert abs(sf[3] - kf[3]) < 1e-9

        if env.done:
            assert sim.done
            assert sim.rewards == [obs[0].reward, obs[1].reward]
            break

    assert sim.done
    assert comets_seen, "Expected comets to spawn during 160-step episode"


def test_fast_sim_clone_isolation_and_serialization():
    sim = FastOrbitSim.from_seed(55, max_steps=120)
    for _ in range(55):
        sim.step([[], []])

    cloned = sim.clone()
    raw = sim.serialize_bytes()
    restored = FastOrbitSim.deserialize_bytes(raw)

    # Mutate clone and verify original + restored are unaffected
    home = next(p for p in cloned.planets if p[1] == 0)
    cloned.step([[[home[0], 0.5, 5]], []])
    assert sim.step_count == 55
    assert restored.step_count == 55
    assert len(sim.fleets) == 0
    assert len(cloned.fleets) <= 1
    assert restored.planets == sim.planets
    assert restored.comet_planet_ids == sim.comet_planet_ids
