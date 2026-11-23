"""Unit tests for the Galcon simulator, tactical solver, PlanetTimelineTransformer, and gbots adapter."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from galcon.gbots_client import GbotsProtocolState, format_gbots_orders, run_gbots_loop
from galcon.heuristic_bot import GalconClassicBot, GalconTacticalBot
from galcon.sim import GalconSim, RedirOrder, SendOrder, evaluate_galcon_state, project_planet_future
from galcon.transformer_bot import (
    GALCON_EDGE_DIM,
    GALCON_PLANET_DIM,
    GALCON_TIMELINE_DIM,
    GALCON_TIMELINE_HORIZON,
    GalconSearchTransformerAgent,
    GalconTransformerAgent,
    apply_galcon_d4,
    build_galcon_transformer,
    extract_galcon_features,
    pad_galcon_state,
)


def test_galcon_sim_symmetry_send_and_redirect():
    sim = GalconSim.create(seed=42, num_planets=16, width=400.0, height=240.0, symmetric=True)
    assert len(sim.planets) >= 10
    p1 = sim.planets[1]
    p2 = sim.planets[2]
    assert np.isclose(p1.x + p2.x, 400.0, atol=1e-5)
    assert np.isclose(p1.y + p2.y, 240.0, atol=1e-5)

    # Launch a fleet from P1 home to planet 3, then redirect it to planet 4
    init_ships = p1.ships
    sim.step({1: [SendOrder(1, 3, 50)]})
    assert len(sim.fleets) == 1
    fl = next(iter(sim.fleets.values()))
    assert fl.owner == 1 and fl.target == 3
    assert p1.ships < init_ships

    sim.step({1: [RedirOrder(fl.id, 4)]})
    assert sim.fleets[fl.id].target == 4

    # Clone isolation
    clone = sim.clone()
    clone.step({2: [SendOrder(2, 3, 50)]})
    assert len(clone.fleets) == 2
    assert len(sim.fleets) == 1


def test_galcon_tactical_bot_beats_classic_baseline():
    tactical = GalconTacticalBot()
    classic = GalconClassicBot()
    wins = 0
    for s in range(6):
        sim = GalconSim.create(seed=100 + s, num_planets=14, width=320.0, height=200.0, max_ticks=200)
        t_pid = 1 if (s % 2 == 0) else 2
        c_pid = 2 if t_pid == 1 else 1
        while not sim.done:
            if sim.tick % 2 == 0:
                sim.step({t_pid: tactical.act(sim, t_pid), c_pid: classic.act(sim, c_pid)})
            else:
                sim.step({})
        if sim.winner() == t_pid:
            wins += 1
    assert wins >= 4, f"Expected GalconTacticalBot to dominate GalconClassicBot, got {wins}/6"


def test_galcon_transformer_d4_invariance_and_search(tmp_path: Path):
    sim = GalconSim.create(seed=77, num_planets=12, width=260.0, height=160.0)
    enc = extract_galcon_features(sim, player=1)
    n = len(enc.planet_ids)
    assert enc.planet_feats.shape == (n, GALCON_PLANET_DIM)
    assert enc.timeline_feats.shape == (n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM)
    assert enc.edge_feats.shape == (n, n, GALCON_EDGE_DIM)
    assert np.isfinite(enc.planet_feats).all()
    assert np.isfinite(enc.edge_feats).all()

    pad = pad_galcon_state(enc)
    model = build_galcon_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "galcon_test.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, ckpt)

    ag = GalconTransformerAgent(ckpt, launch_threshold=0.10, use_d4_tta=True)
    probs, fracs, val = ag.predict_probs_and_value(enc)
    assert probs.shape == (n, n + 1)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
    assert -1.0 <= val <= 1.0

    # Exact D4 reflection + rotation invariance
    enc_ref = extract_galcon_features(sim, player=1)
    enc_ref.planet_feats = apply_galcon_d4(enc_ref.planet_feats, k_rot=3, flip_x=True)
    probs_ref, _, val_ref = ag.predict_probs_and_value(enc_ref)
    assert np.allclose(probs, probs_ref, atol=1e-4)
    assert abs(val - val_ref) < 1e-4

    # Search agent runs cleanly under budget
    s_ag = GalconSearchTransformerAgent(ckpt, launch_threshold=0.10, rollout_horizon=6, time_budget_sec=0.15)
    orders = s_ag.act(sim, 1)
    assert isinstance(orders, list)
    assert -1.0 <= evaluate_galcon_state(sim, 1) <= 1.0
    my_arr, en_arr, po, ps = project_planet_future(sim, 1, 1, horizon=12)
    assert len(my_arr) == 12 and po[-1] == 1.0


def test_gbots_wire_protocol_roundtrip():
    wire_input = "\n".join(
        [
            "/RESET",
            "/SET\tYOU\t10",
            "/SET\tSTATE\tPLAY",
            "/USER\t10\tOurBot\t38bdf8\t1\t0",
            "/USER\t20\tEvanBot\tf87171\t2\t0",
            "/PLANET\t1\t10\t100.0\t50.0\t120.0\t100.0\t28.0",
            "/PLANET\t2\t20\t100.0\t350.0\t120.0\t100.0\t28.0",
            "/PLANET\t3\t0\t15.0\t140.0\t120.0\t80.0\t24.0",
            "/FLEET\t99\t20\t12.0\t300.0\t120.0\t2\t3\t6.0\t0",
            "XY\t99\t290.0\t120.0",
            "/TICK\t0.25",
            "/DESTROY\t99",
            "/TICK\t0.25",
        ]
    ) + "\n"

    # First verify state parsing directly
    proto = GbotsProtocolState()
    for line in wire_input.splitlines()[:10]:
        proto.feed_line(line)
    assert proto.you == 10
    assert len(proto.planets) == 3
    assert np.isclose(proto.fleets[99].x, 290.0)
    sim = proto.to_sim()
    assert sim.planets[1].owner == 1
    assert sim.planets[2].owner == 2
    assert sim.planets[3].owner == 0

    # End-to-end stdin/stdout loop with GalconTacticalBot
    out = io.StringIO()
    ticks = run_gbots_loop(GalconTacticalBot().act, in_stream=io.StringIO(wire_input), out_stream=out)
    assert ticks == 2
    lines = [l for l in out.getvalue().splitlines() if l]
    assert lines.count("/TOCK") == 2
    assert any(l.startswith("/SEND ") for l in lines)
    assert format_gbots_orders([SendOrder(1, 3, 50), RedirOrder(99, 2)]) == ["/SEND 50 1 3", "/REDIR 99 2"]
