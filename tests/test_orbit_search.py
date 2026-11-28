"""Unit, symmetry, and single-thread latency tests for OrbitSearchAgent."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from orbit_wars_bot.fast_sim_cpp import HAS_ORBIT_SIM_CPP, CppFastOrbitSim
from orbit_wars_bot.orbit_search import (
    OrbitSearchAgent,
    create_orbit_search_agent,
    evaluate_settled_shared_physics,
)
from orbit_wars_bot.physics import solve_intercept
from orbit_wars_bot.planet_transformer import (
    PlanetTransformerAgent,
    build_shared_planet_physics,
    extract_planet_features,
)
from scripts.evaluate import resolve_agent

CKPT_BEST = Path(__file__).resolve().parent.parent / "checkpoints" / "planet_transformer_best.pt"


def test_evaluate_settled_shared_physics_symmetry_and_in_flight_capture():
    assert HAS_ORBIT_SIM_CPP is True
    sim = CppFastOrbitSim.from_seed(200, num_agents=2, max_steps=500)
    sh = sim.build_shared_physics(include_py_tracks=False)

    v0 = evaluate_settled_shared_physics(sh, 0, settle_ticks=12)
    v1 = evaluate_settled_shared_physics(sh, 1, settle_ticks=12)
    assert abs(v0 + v1) < 1e-12

    # 4P finite output in [-1, 1]
    sim4 = CppFastOrbitSim.from_seed(77, num_agents=4, max_steps=500)
    sim4.step_n(20)
    sh4 = sim4.build_shared_physics(include_py_tracks=False)
    for pid in range(4):
        vp = evaluate_settled_shared_physics(sh4, pid, settle_ticks=12)
        assert -1.0 <= vp <= 1.0

    # Launching to a nearby capturable neutral planet arriving within settle_ticks must beat holding
    planets = sim.export_planets()
    p0_home = next(p for p in planets if int(p[1]) == 0)
    neutrals = [p for p in planets if int(p[1]) == -1]
    # Pick a neutral with affordable ships and positive production
    best_pair = None
    init_by_id = {int(p[0]): p for p in sim.initial_planets}
    for neu in neutrals:
        sol = solve_intercept(
            p0_home,
            neu,
            int(p0_home[5]),
            planets,
            sim.step_count,
            sim.angular_velocity,
            init_by_id,
            max_eta=12,
        )
        if sol is not None and int(p0_home[5]) > int(neu[5]) + 2:
            best_pair = (neu, sol)
            break

    # Fall back to boosting p0_home ships so an affordable neutral capture exists within eta <= 12
    if best_pair is None:
        closest_neu = min(neutrals, key=lambda n: (n[2] - p0_home[2]) ** 2 + (n[3] - p0_home[3]) ** 2)
        mod_planets = [list(p) for p in planets]
        for p in mod_planets:
            if int(p[0]) == int(p0_home[0]):
                p[5] = int(closest_neu[5]) + 25
            if int(p[0]) == int(closest_neu[0]):
                p[5] = 5
        sim = CppFastOrbitSim.from_parts(
            planets=mod_planets,
            initial_planets=mod_planets,
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=sim.angular_velocity,
            num_agents=2,
            max_steps=500,
        )
        planets = sim.export_planets()
        init_by_id = {int(p[0]): p for p in sim.initial_planets}
        p0_home = next(p for p in planets if int(p[1]) == 0)
        neu = next(p for p in planets if int(p[0]) == int(closest_neu[0]))
        sol = solve_intercept(
            p0_home,
            neu,
            int(p0_home[5]),
            planets,
            sim.step_count,
            sim.angular_velocity,
            init_by_id,
            max_eta=20,
        )
        assert sol is not None
        settle_h = max(12, sol.eta + 1)
    else:
        _, sol = best_pair
        settle_h = 12

    sim_hold = sim.clone()
    sim_hold.step([[], []])
    v_hold = evaluate_settled_shared_physics(
        sim_hold.build_shared_physics(include_py_tracks=False), 0, settle_ticks=settle_h
    )

    sim_attack = sim.clone()
    sim_attack.step([[[int(p0_home[0]), float(sol.angle), int(p0_home[5])]], []])
    v_attack = evaluate_settled_shared_physics(
        sim_attack.build_shared_physics(include_py_tracks=False), 0, settle_ticks=settle_h
    )
    assert v_attack > v_hold


@pytest.mark.skipif(not CKPT_BEST.is_file(), reason="Missing planet_transformer_best.pt checkpoint")
def test_orbit_search_agent_2p_and_4p_with_cpp():
    agent = OrbitSearchAgent(CKPT_BEST)
    assert agent.use_cpp_sim is True

    # 2P check with comets spawned (step >= 50)
    sim2 = CppFastOrbitSim.from_seed(200, num_agents=2, max_steps=500)
    sim2.step_n(55)
    obs2 = sim2.to_observation(player=0)
    info2 = agent.act_with_search_info(obs2)
    assert "moves" in info2
    assert "chosen_idx" in info2
    assert "scores" in info2
    assert len(info2["candidates"]) == len(info2["scores"])
    assert 0 <= int(info2["chosen_idx"]) < len(info2["candidates"])
    for mv in info2["moves"]:
        assert len(mv) == 3 and int(mv[2]) > 0

    # Large override_margin must guarantee greedy baseline selection
    agent_strict = OrbitSearchAgent(CKPT_BEST, override_margin=10.0)
    info_strict = agent_strict.act_with_search_info(obs2)
    assert info_strict["chosen_idx"] == 0

    # 4P FFA check
    sim4 = CppFastOrbitSim.from_seed(77, num_agents=4, max_steps=500)
    sim4.step_n(25)
    obs4 = sim4.to_observation(player=2)
    moves4 = agent.act(obs4)
    assert isinstance(moves4, list)


@pytest.mark.skipif(not CKPT_BEST.is_file(), reason="Missing planet_transformer_best.pt checkpoint")
def test_orbit_search_agent_latency_under_300ms_p95():
    agent = OrbitSearchAgent(CKPT_BEST, time_budget_sec=0.24)
    greedy = PlanetTransformerAgent(
        CKPT_BEST, launch_threshold=0.53, use_d4_tta=True, force_all_in=True
    )

    latencies_ms = []
    for seed in (200, 207):
        sim = CppFastOrbitSim.from_seed(seed, num_agents=2, max_steps=500)
        # Advance into active midgame
        for _ in range(15):
            o0 = sim.to_observation(player=0)
            o1 = sim.to_observation(player=1)
            sim.step([greedy.act(o0), greedy.act(o1)])
        # Warmup once on first turn
        _ = agent.act(sim.to_observation(player=0))
        for _ in range(6):
            o0 = sim.to_observation(player=0)
            o1 = sim.to_observation(player=1)
            t0 = time.perf_counter()
            m0 = agent.act(o0)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            sim.step([m0, greedy.act(o1)])
            if sim.done:
                break

    p95_ms = float(np.percentile(latencies_ms, 95))
    max_ms = float(np.max(latencies_ms))
    assert p95_ms < 300.0, f"OrbitSearchAgent p95 latency {p95_ms:.1f} ms exceeded 300 ms"
    assert max_ms < 450.0, f"OrbitSearchAgent max latency {max_ms:.1f} ms exceeded 450 ms"


@pytest.mark.skipif(not CKPT_BEST.is_file(), reason="Missing planet_transformer_best.pt checkpoint")
def test_evaluate_resolve_orbit_search_spec():
    fn = resolve_agent("orbit_search_t53:checkpoints/planet_transformer_best.pt")
    assert callable(fn)
    sim = CppFastOrbitSim.from_seed(204, num_agents=2, max_steps=500)
    moves = fn(sim.to_observation(player=0), {"shipSpeed": 2.0})
    assert isinstance(moves, list)


@pytest.mark.skipif(not CKPT_BEST.is_file(), reason="Missing planet_transformer_best.pt checkpoint")
def test_orbit_search_joint_cliff_candidates():
    sim = CppFastOrbitSim.from_seed(200, num_agents=2, max_steps=500)
    # Give player 0 two owned planets with ships > 0
    planets = sim.export_planets()
    owned_p0 = [p for p in planets if int(p[1]) == 0]
    if len(owned_p0) < 2:
        neu = next(p for p in planets if int(p[1]) == -1)
        neu[1] = 0
        neu[5] = 40
        sim = CppFastOrbitSim.from_parts(
            planets=planets,
            initial_planets=planets,
            fleets=[],
            comets=[],
            comet_planet_ids=[],
            angular_velocity=sim.angular_velocity,
            num_agents=2,
            max_steps=500,
        )
    obs = sim.to_observation(player=0)
    sh = build_shared_planet_physics(obs, max_speed=sim.ship_speed)
    enc = extract_planet_features(obs, player=0, max_speed=sim.ship_speed, feature_version=2, shared_physics=sh)
    n = len(enc.planet_ids)
    owned_idxs = [i for i in range(n) if enc.owned_mask[i] and np.any(enc.legal_edges[i]) and int(enc.planet_rows[i][5]) > 0]
    assert len(owned_idxs) >= 2

    agent = OrbitSearchAgent(
        CKPT_BEST, launch_threshold=0.53, cliff_band=0.07, joint_cliff_delta=0.08, max_candidates=4
    )
    probs = np.zeros((n, n + 1), dtype=np.float32)
    probs[:, 0] = 1.0
    for idx in owned_idxs[:2]:
        j1 = int(np.flatnonzero(enc.legal_edges[idx])[0])
        probs[idx, 0] = 0.51  # p_launch = 0.49 in [0.53 - 0.08, 0.53)
        probs[idx, 1 + j1] = 0.49
    mean_fracs = np.ones((n, n), dtype=np.float32)

    cands, _, overrides, triggered = agent._build_surgical_candidates(enc, probs, mean_fracs, sim.ship_speed, sh)
    assert triggered
    assert any(ov is not None and len(ov) >= 2 and all(v > 0 for v in ov.values()) for ov in overrides)
    assert len(cands) >= 2

    moves = agent.act(obs)
    assert isinstance(moves, list)
    for pid, ang, count in moves:
        assert isinstance(pid, int) and np.isfinite(ang) and count > 0


@pytest.mark.skipif(not CKPT_BEST.is_file(), reason="Missing planet_transformer_best.pt checkpoint")
def test_orbit_search_external_value_model_and_multi_replan(tmp_path):
    import torch
    from orbit_wars_bot.planet_transformer import PlanetTimelineTransformer

    val_model = PlanetTimelineTransformer(
        timeline_dim=14,
        d_model=32,
        n_heads=2,
        n_layers=2,
        dim_ff=64,
        feature_version=2,
        arch_version=3,
    )
    tiny_val = tmp_path / "tiny_val.pt"
    torch.save(
        {
            "model_config": val_model.config,
            "model_state_dict": val_model.state_dict(),
        },
        tiny_val,
    )

    agent = OrbitSearchAgent(
        CKPT_BEST,
        value_checkpoint_path=tiny_val,
        rollout_horizon=12,
        replan_steps=(4, 8),
    )
    assert agent.value_model is not agent.model
    assert agent.replan_steps_tuple == (4, 8)
    assert agent.replan_step == 4

    sim2 = CppFastOrbitSim.from_seed(200, num_agents=2, max_steps=500)
    sim2.step_n(55)
    info2 = agent.act_with_search_info(sim2.to_observation(player=0))
    assert "moves" in info2 and 0 <= int(info2["chosen_idx"]) < len(info2["candidates"])

    sim4 = CppFastOrbitSim.from_seed(77, num_agents=4, max_steps=500)
    sim4.step_n(25)
    moves4 = agent.act(sim4.to_observation(player=2))
    assert isinstance(moves4, list)

    fn = resolve_agent(f"orbit_search_t53:{CKPT_BEST}|{tiny_val}")
    moves_resolved = fn(sim2.to_observation(player=0), {"shipSpeed": 2.0})
    assert isinstance(moves_resolved, list)
