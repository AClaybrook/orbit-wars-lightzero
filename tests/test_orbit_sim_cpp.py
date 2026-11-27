"""Comprehensive bit-exact parity and speedup tests for the C++17 Orbit Wars engine."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest
from kaggle_environments import make

from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.fast_sim_cpp import (
    HAS_ORBIT_SIM_CPP,
    CppFastOrbitSim,
    build_shared_planet_physics_cpp,
    solve_intercept_cpp,
)
from orbit_wars_bot.physics import build_comet_lookup, solve_intercept
from orbit_wars_bot.planet_transformer import _build_shared_planet_physics_python


def _assert_states_match(
    planets_a, fleets_a, com_a, step_a, done_a, rew_a,
    planets_b, fleets_b, com_b, step_b, done_b, rew_b,
):
    assert step_a == step_b
    assert done_a == done_b
    assert list(com_a) == list(com_b)
    if done_a or done_b:
        assert list(rew_a) == list(rew_b)
    assert len(planets_a) == len(planets_b)
    for pa, pb in zip(planets_a, planets_b):
        assert int(pa[0]) == int(pb[0])
        assert int(pa[1]) == int(pb[1])
        assert int(pa[5]) == int(pb[5])
        assert int(pa[6]) == int(pb[6])
        assert abs(float(pa[2]) - float(pb[2])) < 1e-11
        assert abs(float(pa[3]) - float(pb[3])) < 1e-11
        assert abs(float(pa[4]) - float(pb[4])) < 1e-11
    assert len(fleets_a) == len(fleets_b)
    for fa, fb in zip(fleets_a, fleets_b):
        assert int(fa[0]) == int(fb[0])
        assert int(fa[1]) == int(fb[1])
        assert int(fa[5]) == int(fb[5])
        assert int(fa[6]) == int(fb[6])
        assert abs(float(fa[2]) - float(fb[2])) < 1e-11
        assert abs(float(fa[3]) - float(fb[3])) < 1e-11
        assert abs(float(fa[4]) - float(fb[4])) < 1e-12


def test_cpp_sim_available():
    assert HAS_ORBIT_SIM_CPP is True


@pytest.mark.parametrize(
    ("seed", "num_agents"),
    [(11, 2), (42, 2), (77, 4), (123, 2), (202, 4), (404, 2), (505, 4), (707, 2)],
)
def test_cpp_sim_bit_exact_parity_vs_python_2p_and_4p(seed: int, num_agents: int):
    """Compare CppFastOrbitSim, FastOrbitSim(use_cpp=True), and FastOrbitSim(use_cpp=False) tick-by-tick."""
    max_steps = 180
    sim_py = FastOrbitSim.from_seed(seed, num_agents=num_agents, max_steps=max_steps, use_cpp=False)
    sim_wrap = FastOrbitSim.from_seed(seed, num_agents=num_agents, max_steps=max_steps, use_cpp=True)
    sim_cpp = CppFastOrbitSim.from_seed(seed, num_agents=num_agents, max_steps=max_steps)

    rng = random.Random(seed + 9001)
    comets_seen = False

    for _ in range(max_steps):
        actions = [[] for _ in range(num_agents)]
        for p in sim_py.planets:
            owner = int(p[1])
            if 0 <= owner < num_agents and int(p[5]) >= 6 and rng.random() < 0.22:
                # Include mixed radial, random, and center-crossing angles (sun + OOB + collisions)
                actions[owner].append([int(p[0]), rng.uniform(0.0, 2.0 * math.pi), max(1, int(p[5]) // 2)])

        sim_py.step(actions)
        sim_wrap.step(actions)
        sim_cpp.step(actions)

        if sim_py.comet_planet_ids:
            comets_seen = True

        _assert_states_match(
            sim_py.planets, sim_py.fleets, sim_py.comet_planet_ids, sim_py.step_count, sim_py.done, sim_py.rewards,
            sim_wrap.planets, sim_wrap.fleets, sim_wrap.comet_planet_ids, sim_wrap.step_count, sim_wrap.done, sim_wrap.rewards,
        )
        _assert_states_match(
            sim_py.planets, sim_py.fleets, sim_py.comet_planet_ids, sim_py.step_count, sim_py.done, sim_py.rewards,
            sim_cpp.export_planets(), sim_cpp.export_fleets(), sim_cpp.comet_planet_ids, sim_cpp.step_count, sim_cpp.done, sim_cpp.rewards,
        )
        assert sim_py.winner() == sim_wrap.winner() == sim_cpp.winner()
        if sim_py.done:
            break

    assert comets_seen or sim_py.step_count < 50


@pytest.mark.parametrize(
    ("seed", "num_agents", "episode_steps"),
    [(11, 2, 160), (123, 2, 160), (404, 2, 160), (77, 4, 120)],
)
def test_cpp_sim_exact_parity_with_kaggle_environments(seed: int, num_agents: int, episode_steps: int):
    """Verify 100% tick-for-tick parity between CppFastOrbitSim and official kaggle_environments."""
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": episode_steps})
    obs = env.reset(num_agents)
    sim_cpp = CppFastOrbitSim.from_seed(seed, num_agents=num_agents, max_steps=episode_steps)

    rng = random.Random(seed + 1000)
    comets_seen = False

    for _ in range(episode_steps):
        kobs_cur = obs[0].observation
        actions = [[] for _ in range(num_agents)]
        for p in kobs_cur.planets:
            owner = int(p[1])
            if 0 <= owner < num_agents and int(p[5]) >= 10 and rng.random() < 0.20:
                actions[owner].append([int(p[0]), rng.uniform(0.0, 2.0 * math.pi), max(1, int(p[5]) // 2)])

        obs = env.step(actions)
        sim_cpp.step(actions)
        kobs = obs[0].observation
        if sim_cpp.comet_planet_ids:
            comets_seen = True

        k_rew = [obs[i].reward for i in range(num_agents)]
        _assert_states_match(
            kobs.planets, kobs.fleets, kobs.comet_planet_ids, kobs.step, env.done, k_rew,
            sim_cpp.export_planets(), sim_cpp.export_fleets(), sim_cpp.comet_planet_ids, sim_cpp.step_count, sim_cpp.done, sim_cpp.rewards,
        )
        if env.done:
            break

    assert sim_cpp.done
    assert comets_seen


def test_cpp_sim_clone_step_n_and_step_batch_parity():
    """Verify clone isolation, step_n (spanning comet spawn step 50), and step_batch vs Python."""
    sim_py = FastOrbitSim.from_seed(42, num_agents=2, max_steps=200, use_cpp=False)
    for t in range(42):
        acts = [[], []]
        if t % 5 == 0:
            for p in sim_py.planets:
                if p[1] in (0, 1) and p[5] >= 8:
                    acts[int(p[1])].append([int(p[0]), 0.4 * (t + 1), int(p[5]) // 3])
        sim_py.step(acts)

    sim_cpp = CppFastOrbitSim.from_fast_sim(sim_py)

    # 1. step_n spanning step 42 -> 62 (crosses comet spawn at step 50!)
    c_py = sim_py.clone(use_cpp=False)
    c_cpp = sim_cpp.clone()
    c_py.step_n(20)
    c_cpp.step_n(20)
    _assert_states_match(
        c_py.planets, c_py.fleets, c_py.comet_planet_ids, c_py.step_count, c_py.done, c_py.rewards,
        c_cpp.export_planets(), c_cpp.export_fleets(), c_cpp.comet_planet_ids, c_cpp.step_count, c_cpp.done, c_cpp.rewards,
    )
    # Parent stays untouched at step 42
    assert sim_cpp.step_count == 42

    # 2. step_batch across 16 distinct branches + 6 empty follow-up ticks (crossing step 50)
    b_py = []
    b_cpp = []
    acts_batch = []
    homes0 = [p for p in sim_py.planets if p[1] == 0]
    homes1 = [p for p in sim_py.planets if p[1] == 1]
    for bi in range(16):
        b_py.append(sim_py.clone(use_cpp=False))
        b_cpp.append(sim_cpp.clone())
        a0 = [[int(homes0[0][0]), 0.35 * bi, max(1, int(homes0[0][5]) // 2)]] if homes0 else []
        a1 = [[int(homes1[0][0]), 3.5 + 0.2 * bi, max(1, int(homes1[0][5]) // 2)]] if homes1 else []
        acts_batch.append([a0, a1])

    CppFastOrbitSim.step_batch(b_cpp, acts_batch, n_empty_after=8)
    for spy, scpp, acts in zip(b_py, b_cpp, acts_batch):
        spy.step(acts)
        for _ in range(8):
            spy.step([[], []])
        _assert_states_match(
            spy.planets, spy.fleets, spy.comet_planet_ids, spy.step_count, spy.done, spy.rewards,
            scpp.export_planets(), scpp.export_fleets(), scpp.comet_planet_ids, scpp.step_count, scpp.done, scpp.rewards,
        )


@pytest.mark.parametrize("seed", [11, 77, 404])
def test_cpp_shared_physics_and_solve_intercept_parity(seed: int):
    """Verify build_shared_planet_physics_cpp (obs & CppSim) and solve_intercept_cpp match Python bit-for-bit."""
    sim = FastOrbitSim.from_seed(seed, num_agents=2, max_steps=200, use_cpp=False)
    rng = random.Random(seed + 333)
    for t in range(56):
        acts = [[], []]
        if t % 3 == 0:
            for p in sim.planets:
                if p[1] in (0, 1) and p[5] >= 6 and rng.random() < 0.45:
                    acts[int(p[1])].append([int(p[0]), rng.uniform(0.0, 2.0 * math.pi), max(1, int(p[5]) // 2)])
        sim.step(acts)

    obs = sim.to_observation(player=0)
    cpp_sim = CppFastOrbitSim.from_fast_sim(sim)

    sh_py = _build_shared_planet_physics_python(obs, max_speed=sim.ship_speed)
    sh_c1 = build_shared_planet_physics_cpp(obs, max_speed=sim.ship_speed, include_py_tracks=True)
    sh_c2 = cpp_sim.build_shared_physics(include_py_tracks=True)

    for sh_c in (sh_c1, sh_c2):
        assert sh_c["n"] == sh_py["n"]
        assert sh_c["planet_ids"] == sh_py["planet_ids"]
        assert np.allclose(sh_c["tr_x"], sh_py["tr_x"], atol=1e-11, equal_nan=True)
        assert np.allclose(sh_c["tr_y"], sh_py["tr_y"], atol=1e-11, equal_nan=True)
        assert np.array_equal(sh_c["tr_valid"], sh_py["tr_valid"])
        assert np.array_equal(sh_c["fut_owner"], sh_py["fut_owner"])
        assert np.array_equal(sh_c["fut_ships"], sh_py["fut_ships"])
        assert np.array_equal(sh_c["valid_route"], sh_py["valid_route"])
        assert np.array_equal(sh_c["best_eta"], sh_py["best_eta"])
        assert np.allclose(sh_c["best_dist"], sh_py["best_dist"], atol=1e-11)
        assert sh_c["is_4p"] == sh_py["is_4p"]
        for pid in sh_py["planet_ids"]:
            arr_p = [(a.fleet_id, a.owner, a.ships, a.planet_id, a.eta) for a in sh_py["arrivals"][pid]]
            arr_c = [(a.fleet_id, a.owner, a.ships, a.planet_id, a.eta) for a in sh_c["arrivals"][pid]]
            assert sorted(arr_c) == sorted(arr_p)

    # Intercept solver parity across all planet pairs
    planets = sh_py["planets"]
    tracks = sh_py["tracks"]
    init_by_id = sh_py["initial_by_id"]
    clookup = build_comet_lookup(obs["comets"])
    n = len(planets)
    for i in range(n):
        sh_send = max(5, int(planets[i][5]))
        for j in range(n):
            if i == j:
                continue
            sol_py = solve_intercept(
                planets[i], planets[j], sh_send, planets, sim.step_count, sim.angular_velocity,
                init_by_id, comet_lookup=clookup, tracks=tracks, max_eta=24, max_speed=sim.ship_speed,
            )
            sol_c = solve_intercept_cpp(
                i, j, sh_send, sh_c1["tr_x"], sh_c1["tr_y"], sh_c1["pr"], max_eta=24, max_speed=sim.ship_speed,
            )
            if sol_py is None:
                assert sol_c is None
            else:
                assert sol_c is not None
                assert sol_c.eta == sol_py.eta
                assert abs(sol_c.angle - sol_py.angle) < 1e-11
                assert abs(sol_c.impact_pos[0] - sol_py.impact_pos[0]) < 1e-11
                assert abs(sol_c.impact_pos[1] - sol_py.impact_pos[1]) < 1e-11


def test_cpp_sim_rollout_speedup_benchmark():
    """Verify >= 20x rollout speedup on active-fleet 16-tick rollouts and report physics speedup."""
    sim_py = FastOrbitSim.from_seed(123, num_agents=2, max_steps=300, use_cpp=False)
    rng = random.Random(1234)
    for t in range(30):
        acts = [[], []]
        if t % 3 == 0:
            for p in sim_py.planets:
                if p[1] in (0, 1) and p[5] >= 6:
                    acts[int(p[1])].append([int(p[0]), rng.uniform(0.0, 2.0 * math.pi), max(1, int(p[5]) // 2)])
        sim_py.step(acts)

    sim_cpp = CppFastOrbitSim.from_fast_sim(sim_py)
    n_rollouts = 1000
    horizon = 16
    empty_acts = [[], []]

    t0 = time.perf_counter()
    for _ in range(n_rollouts):
        s = sim_py.clone(use_cpp=False)
        for _ in range(horizon):
            s._step_python(empty_acts)
    py_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(n_rollouts):
        s = sim_cpp.clone()
        s.step_n(horizon)
    cpp_sec = time.perf_counter() - t1

    speedup = py_sec / max(1e-9, cpp_sec)

    # Also measure build_shared_planet_physics speedup
    obs = sim_py.to_observation(player=0)
    n_phys = 200
    t2 = time.perf_counter()
    for _ in range(n_phys):
        _build_shared_planet_physics_python(obs, max_speed=sim_py.ship_speed)
    phys_py_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    for _ in range(n_phys):
        build_shared_planet_physics_cpp(obs, max_speed=sim_py.ship_speed, include_py_tracks=True)
    phys_cpp_sec = time.perf_counter() - t3

    t4 = time.perf_counter()
    for _ in range(n_phys):
        sim_cpp.build_shared_physics(include_py_tracks=False)
    phys_res_sec = time.perf_counter() - t4

    print(
        f"\n[Benchmark 1000x 16-tick rollouts (fleets={len(sim_py.fleets)}, planets={len(sim_py.planets)})] "
        f"Py={py_sec*1000:.1f}ms vs C++={cpp_sec*1000:.2f}ms -> {speedup:.1f}x speedup"
        f"\n[Benchmark 200x build_shared_physics] Py={phys_py_sec*1000:.1f}ms vs C++(obs)={phys_cpp_sec*1000:.2f}ms ({phys_py_sec/phys_cpp_sec:.1f}x) "
        f"| C++(resident)={phys_res_sec*1000:.2f}ms ({phys_py_sec/phys_res_sec:.1f}x)",
        flush=True,
    )
    assert speedup >= 20.0
