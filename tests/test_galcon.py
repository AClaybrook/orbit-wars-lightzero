"""Unit tests for the Galcon simulator, sparring suite, Gen 2 PlanetTimelineTransformer, and gbots adapter."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from galcon.gbots_client import GbotsProtocolState, format_gbots_orders, run_gbots_loop
from galcon.heuristic_bot import GalconClassicBot, GalconMacroBot, GalconRusherBot, GalconTacticalBot
from galcon.sim import Fleet, GalconSim, Planet, RedirOrder, SendOrder, evaluate_galcon_state, project_planet_future
from galcon.transformer_bot import (
    GALCON_EDGE_DIM,
    GALCON_EDGE_DIM_V2,
    GALCON_PLANET_DIM,
    GALCON_PLANET_DIM_V2,
    GALCON_REDIR_DIM,
    GALCON_TIMELINE_DIM,
    GALCON_TIMELINE_HORIZON,
    GalconSearchTransformerAgent,
    GalconTransformerAgent,
    apply_galcon_d4,
    build_galcon_gen2_transformer,
    build_galcon_transformer,
    compute_smart_redirects,
    extract_galcon_features,
    extract_galcon_redir_features,
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


def test_galcon_sparring_suite_rusher_and_macro():
    sim = GalconSim.create(seed=55, num_planets=14, width=320.0, height=200.0, max_ticks=120)
    rusher_allin = GalconRusherBot(style="all_in")
    rusher_blitz = GalconRusherBot(style="blitz")
    rusher_adapt = GalconRusherBot(style="adaptive")
    macro = GalconMacroBot()

    # All-in rusher must fire a heavy tick-0 home-rush straight at enemy home (planet 2)
    o0 = rusher_allin.act(sim, 1)
    assert any(isinstance(o, SendOrder) and o.src_id == 1 and o.dst_id == 2 and o.percent >= 65 for o in o0)

    # All sparring bots run cleanly over full matches
    for opp in (rusher_blitz, rusher_adapt, macro):
        s2 = GalconSim.create(seed=56, num_planets=12, width=260.0, height=160.0, max_ticks=80)
        while not s2.done:
            if s2.tick % 2 == 0:
                s2.step({1: rusher_allin.act(s2, 1), 2: opp.act(s2, 2)})
            else:
                s2.step({})
        assert s2.done


def test_galcon_transformer_d4_invariance_and_search(tmp_path: Path):
    sim = GalconSim.create(seed=77, num_planets=12, width=260.0, height=160.0)
    enc = extract_galcon_features(sim, player=1, feature_version=1)
    n = len(enc.planet_ids)
    assert enc.planet_feats.shape == (n, GALCON_PLANET_DIM)
    assert enc.timeline_feats.shape == (n, GALCON_TIMELINE_HORIZON, GALCON_TIMELINE_DIM)
    assert enc.edge_feats.shape == (n, n, GALCON_EDGE_DIM)
    assert np.isfinite(enc.planet_feats).all()
    assert np.isfinite(enc.edge_feats).all()

    pad = pad_galcon_state(enc)
    assert pad["planet_feats"].shape[1] == GALCON_PLANET_DIM
    model = build_galcon_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "galcon_test_v1.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, ckpt)

    ag = GalconTransformerAgent(ckpt, launch_threshold=0.10, use_d4_tta=True)
    probs, fracs, val = ag.predict_probs_and_value(enc)
    assert probs.shape == (n, n + 1)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
    assert -1.0 <= val <= 1.0

    # Exact D4 reflection + rotation invariance
    enc_ref = extract_galcon_features(sim, player=1, feature_version=1)
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


def test_galcon_gen2_world_model_native_redir_and_d4(tmp_path: Path):
    sim = GalconSim.create(seed=88, num_planets=14, width=320.0, height=200.0)
    sim.step({1: [SendOrder(1, 3, 35)], 2: [SendOrder(2, 4, 35)]})
    enc = extract_galcon_features(sim, player=1, feature_version=2)
    n = len(enc.planet_ids)
    assert enc.planet_feats.shape == (n, GALCON_PLANET_DIM_V2)
    assert enc.edge_feats.shape == (n, n, GALCON_EDGE_DIM_V2)

    renc = extract_galcon_redir_features(sim, enc)
    assert len(renc.fleet_ids) == 1
    assert renc.redir_feats.shape == (1, n, GALCON_REDIR_DIM)
    assert np.isfinite(renc.redir_feats).all()

    model = build_galcon_gen2_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    logits, fracs, val, aux = model(
        torch.from_numpy(enc.planet_feats[None]),
        torch.from_numpy(enc.timeline_feats[None]),
        torch.from_numpy(enc.edge_feats[None]),
        torch.ones((1, n), dtype=torch.bool),
        return_aux=True,
    )
    assert logits.shape == (1, n, n + 1)
    assert fracs.shape == (1, n, n)
    assert val.shape == (1,)
    assert aux.shape == (1, n, 4)

    ckpt = tmp_path / "galcon_gen2_test.pt"
    torch.save(
        {"model_config": model.config, "model_state_dict": model.state_dict(), "has_trained_redir": True},
        ckpt,
    )
    ag = GalconTransformerAgent(ckpt, launch_threshold=0.10, use_d4_tta=True)
    p1, _, v1, r1 = ag.predict_probs_value_and_redir(enc, renc)
    assert r1 is not None and r1.shape == (1, n)
    assert np.allclose(r1.sum(axis=-1), 1.0, atol=1e-4)

    # Exact 8x D4 invariance on policy, value, AND native /REDIR head
    enc_ref = extract_galcon_features(sim, player=1, feature_version=2)
    enc_ref.planet_feats = apply_galcon_d4(enc_ref.planet_feats, k_rot=1, flip_x=True)
    p2, _, v2, r2 = ag.predict_probs_value_and_redir(enc_ref, renc)
    assert np.allclose(p1, p2, atol=1e-4)
    assert abs(v1 - v2) < 1e-4
    assert r2 is not None and np.allclose(r1, r2, atol=1e-4)

    # Batched search with search info
    s_ag = GalconSearchTransformerAgent(ckpt, launch_threshold=0.10, rollout_horizon=6, time_budget_sec=0.20)
    best_plan, cands, scores, best_sc = s_ag.act_with_search_info(sim, 1)
    assert isinstance(best_plan, list)
    assert len(cands) == len(scores) >= 1
    assert -1.0 <= best_sc <= 1.0

    # Smart redirect oracle rescues a doomed fleet attacking a heavily fortified enemy planet
    custom_planets = [
        Planet(id=1, owner=1, ships=40.0, x=40.0, y=100.0, production=100.0, radius=28.0),
        Planet(id=2, owner=2, ships=95.0, x=260.0, y=100.0, production=100.0, radius=28.0),
        Planet(id=3, owner=0, ships=8.0, x=150.0, y=120.0, production=90.0, radius=25.0),
    ]
    doomed_fleet = Fleet(id=500, owner=1, ships=20.0, x=140.0, y=100.0, source=1, target=2)
    sim_r = GalconSim(custom_planets, [doomed_fleet], width=300.0, height=200.0)
    enc_r = extract_galcon_features(sim_r, 1, feature_version=2)
    redirs, labels = compute_smart_redirects(sim_r, enc_r, verify_rollout=True)
    assert any(r.fleet_id == 500 and r.dst_id == 3 for r in redirs)
    assert labels[500] == 2


def test_gbots_wire_protocol_roundtrip(tmp_path: Path):
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

    # End-to-end stdin/stdout loop with Gen 2 GalconTransformerAgent
    m = build_galcon_gen2_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ck = tmp_path / "gen2_gbots.pt"
    torch.save({"model_config": m.config, "model_state_dict": m.state_dict(), "has_trained_redir": True}, ck)
    ag = GalconTransformerAgent(ck, launch_threshold=0.05, use_d4_tta=True)
    out2 = io.StringIO()
    ticks2 = run_gbots_loop(ag.act, in_stream=io.StringIO(wire_input), out_stream=out2)
    assert ticks2 == 2
    lines2 = [l for l in out2.getvalue().splitlines() if l]
    assert lines2.count("/TOCK") == 2
    assert any(l.startswith("/SEND ") for l in lines2)


def test_galcon_gen3_reactive_search_and_large_asymmetric_arena(tmp_path: Path):
    sim = GalconSim.create(
        seed=31415, num_planets=26, width=560.0, height=340.0, max_ticks=220, symmetric=False
    )
    assert len(sim.planets) >= 22
    enc = extract_galcon_features(sim, player=1, feature_version=2)
    n = len(enc.planet_ids)
    assert n == len(sim.planets)
    assert enc.planet_feats.shape == (n, GALCON_PLANET_DIM_V2)
    assert enc.edge_feats.shape == (n, n, GALCON_EDGE_DIM_V2)
    assert np.isfinite(enc.planet_feats).all()

    pad = pad_galcon_state(enc)
    assert pad["pad_mask"].sum() == n

    model = build_galcon_gen2_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "galcon_gen3_test.pt"
    torch.save(
        {"model_config": model.config, "model_state_dict": model.state_dict(), "has_trained_redir": True},
        ckpt,
    )
    s_ag = GalconSearchTransformerAgent(
        ckpt,
        launch_threshold=0.10,
        rollout_horizon=8,
        time_budget_sec=0.25,
        neural_value_weight=0.85,
        reactive_mid_ply=True,
        prior_bonus=0.025,
    )
    best_plan, cands, scores, best_sc = s_ag.act_with_search_info(sim, 1)
    assert isinstance(best_plan, list)
    assert len(cands) == len(scores) >= 2
    assert -1.0 <= best_sc <= 1.0


def test_galcon_lua_bridge_mock_bot(tmp_path: Path):
    import shutil

    if shutil.which("lua5.3") is None and shutil.which("lua") is None:
        return

    from galcon.lua_bridge import GalconLuaBotBridge

    mock_lua = tmp_path / "mock_galcon_bot.lua"
    mock_lua.write_text(
        """
local tick_count = 0
function reset_bot()
    tick_count = 0
end
function bot_act(user)
    tick_count = tick_count + 1
    local mine = g2.search("planet owner:" .. user)
    local all = g2.search("planet")
    if #mine > 0 then
        local src = mine[1]
        local best = nil
        local best_d = 1e9
        for _, p in ipairs(all) do
            if p.owner_n ~= user.n then
                local d = g2.distance(src, p)
                if d < best_d then
                    best_d = d
                    best = p
                end
            end
        end
        if best ~= nil and src.ships_value >= 10 then
            src:fleet_send(50, best)
        end
    end
    local fleets = g2.search("fleet owner:" .. user)
    if #fleets > 0 and #all >= 3 then
        for _, fl in ipairs(fleets) do
            for _, p in ipairs(all) do
                if p.n ~= fl.fleet_target and p.owner_n ~= user.n then
                    fl:fleet_redirect(p)
                    break
                end
            end
        end
    end
end
""",
        encoding="utf-8",
    )

    bridge = GalconLuaBotBridge(mock_lua)
    try:
        sim = GalconSim.create(seed=101, num_planets=10, width=280.0, height=180.0)
        o0 = bridge.act(sim, 1)
        assert any(isinstance(o, SendOrder) and o.src_id == 1 for o in o0)
        sim.step({1: o0})
        assert len(sim.fleets) >= 1
        o1 = bridge.act(sim, 1)
        assert any(isinstance(o, RedirOrder) for o in o1)
        # Verify reset triggers cleanly when starting a new game at tick 0
        sim2 = GalconSim.create(seed=102, num_planets=10, width=280.0, height=180.0)
        o2 = bridge.act(sim2, 2)
        assert any(isinstance(o, SendOrder) and o.src_id == 2 for o in o2)
    finally:
        bridge.close()


def test_galcon_gen4_multisend_canonical_d4_and_adaptive_search(tmp_path: Path):
    from galcon.transformer_bot import build_galcon_gen4_transformer, canonicalize_galcon_d4

    sim = GalconSim.create(seed=4242, num_planets=14, width=360.0, height=220.0)
    enc = extract_galcon_features(sim, player=1, feature_version=2)
    n = len(enc.planet_ids)

    # 1. Exact 8x D4 invariance in ONE single 1x forward pass via canonicalize_galcon_d4
    model = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "galcon_gen4_test.pt"
    torch.save(
        {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt,
    )
    ag = GalconTransformerAgent(ckpt, multi_send_threshold=0.05)
    assert ag.has_trained_multi_send and ag.use_d4_canonical

    p0, _, v0, _, ms0, mf0 = ag._predict_all(enc)
    assert ms0 is not None and mf0 is not None
    assert ms0.shape == (n, n) and mf0.shape == (n, n)
    for flip in (False, True):
        for k in range(4):
            enc_t = extract_galcon_features(sim, player=1, feature_version=2)
            enc_t.planet_feats = apply_galcon_d4(enc_t.planet_feats, k_rot=k, flip_x=flip)
            pt, _, vt, _, mst, _ = ag._predict_all(enc_t)
            assert np.allclose(p0, pt, atol=1e-5)
            assert abs(v0 - vt) < 1e-5
            assert mst is not None and np.allclose(ms0, mst, atol=1e-5)

    # 2. Multi-send decoder emits multiple simultaneous SendOrders from one surplus planet
    fake_ms_p = np.zeros((n, n), dtype=np.float32)
    fake_ms_f = np.full((n, n), 0.15, dtype=np.float32)
    fake_ms_p[0, 2] = 0.90
    fake_ms_p[0, 3] = 0.85
    fake_ms_p[0, 4] = 0.80
    ords = ag._decode_multi_send_orders(sim, enc, fake_ms_p, fake_ms_f, threshold=0.50)
    sends_from_home = [o for o in ords if isinstance(o, SendOrder) and o.src_id == enc.planet_ids[0]]
    assert len(sends_from_home) == 3

    # 3. Opponent-adaptive 2-ply & 3-ply search runs cleanly under budget
    s_ag = GalconSearchTransformerAgent(
        ckpt,
        multi_send_threshold=0.45,
        rollout_horizon=9,
        num_plies=3,
        time_budget_sec=0.30,
        opponent_adaptive=True,
    )
    best_plan, cands, scores, best_sc = s_ag.act_with_search_info(sim, 1)
    assert isinstance(best_plan, list)
    assert len(cands) == len(scores) >= 1
    assert -1.0 <= best_sc <= 1.0
    assert canonicalize_galcon_d4(enc.planet_feats).shape == enc.planet_feats.shape


def test_galcon_wave5_adversarial_search_and_ensemble(tmp_path: Path):
    from galcon.transformer_bot import EnsembleGalconSearchAgent, build_galcon_gen4_transformer

    sim = GalconSim.create(seed=5150, num_planets=16, width=400.0, height=240.0)
    # Give P1 multiple friendly planets with rear/front surplus to exercise tunnel-feed and focus-fire proposals
    pids = sorted(sim.planets.keys())
    sim.planets[pids[0]].owner = 1
    sim.planets[pids[0]].ships = 60.0
    sim.planets[pids[1]].owner = 1
    sim.planets[pids[1]].ships = 55.0
    sim.planets[pids[2]].owner = 2
    sim.planets[pids[2]].ships = 40.0

    ckpts = []
    for idx, sd in enumerate((11, 22)):
        torch.manual_seed(sd)
        model = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
        p = tmp_path / f"galcon_w5_m{idx}.pt"
        torch.save(
            {
                "model_config": model.config,
                "model_state_dict": model.state_dict(),
                "has_trained_redir": True,
                "has_trained_multi_send": True,
                "use_canonical_d4": True,
            },
            p,
        )
        ckpts.append(p)

    adv_ag = GalconSearchTransformerAgent(
        ckpts[0],
        multi_send_threshold=0.45,
        rollout_horizon=10,
        num_plies=2,
        time_budget_sec=0.30,
        opponent_adaptive=True,
        adversarial_search=True,
    )
    enc = extract_galcon_features(sim, player=1, feature_version=2)
    probs, _, _, _, _, _ = adv_ag._predict_all(enc)
    ff = adv_ag._build_focus_fire_plan(sim, enc, probs, max_sources=4)
    tun = adv_ag._build_tunnel_feed_plan(sim, enc)
    assert isinstance(ff, list) and isinstance(tun, list)

    best_plan, cands, scores, best_sc = adv_ag.act_with_search_info(sim, 1)
    assert isinstance(best_plan, list)
    assert len(cands) == len(scores) >= 1
    assert -1.0 <= best_sc <= 1.0

    ens = EnsembleGalconSearchAgent(
        ckpts,
        weights=[0.5, 0.5],
        multi_send_threshold=0.45,
        rollout_horizon=10,
        num_plies=2,
        time_budget_sec=0.30,
        opponent_adaptive=True,
        adversarial_search=True,
    )
    pure_orders = ens.act_pure(sim, 1)
    assert isinstance(pure_orders, list)

    p0, _, v0, _, ms0, _ = ens._predict_all(enc)
    for flip in (False, True):
        for k in range(4):
            enc_t = extract_galcon_features(sim, player=1, feature_version=2)
            enc_t.planet_feats = apply_galcon_d4(enc_t.planet_feats, k_rot=k, flip_x=flip)
            pt, _, vt, _, mst, _ = ens._predict_all(enc_t)
            assert np.allclose(p0, pt, atol=1e-5)
            assert abs(v0 - vt) < 1e-5
            assert mst is not None and np.allclose(ms0, mst, atol=1e-5)

    e_plan, e_cands, e_scores, e_sc = ens.act_with_search_info(sim, 1)
    assert isinstance(e_plan, list)
    assert len(e_cands) == len(e_scores) >= 1
    assert -1.0 <= e_sc <= 1.0


def test_dml_adamw_matches_torch_adamw_and_device_resolver():
    from galcon.train import DMLAdamW, resolve_train_device

    dev, is_dml = resolve_train_device("cpu")
    assert dev.type == "cpu" and not is_dml

    torch.manual_seed(123)
    m_ref = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    m_dml = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    m_dml.load_state_dict(m_ref.state_dict())

    opt_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_dml = DMLAdamW(m_dml.parameters(), lr=1e-3, weight_decay=1e-2)

    for step in range(5):
        torch.manual_seed(1000 + step)
        x = torch.randn(6, 8)
        y = torch.randn(6, 4)

        opt_ref.zero_grad()
        torch.nn.functional.mse_loss(m_ref(x), y).backward()
        opt_ref.step()

        opt_dml.zero_grad()
        torch.nn.functional.mse_loss(m_dml(x), y).backward()
        opt_dml.step()

    for p_ref, p_dml in zip(m_ref.parameters(), m_dml.parameters()):
        assert torch.allclose(p_ref, p_dml, atol=1e-6)

    from galcon.train import _bernoulli_log_prob_and_entropy, _stable_bce_with_logits

    torch.manual_seed(7)
    logits = torch.randn(4, 10) * 5.0
    acts = torch.randint(0, 2, (4, 10)).float()
    pw = torch.tensor(4.5)
    assert torch.allclose(
        _stable_bce_with_logits(logits, acts, pos_weight=pw),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, acts, pos_weight=pw, reduction="none"),
        atol=1e-5,
    )
    lp, ent = _bernoulli_log_prob_and_entropy(logits, acts)
    dist = torch.distributions.Bernoulli(logits=logits)
    assert torch.allclose(lp, dist.log_prob(acts), atol=1e-5)
    assert torch.allclose(ent, dist.entropy(), atol=1e-5)


def test_wave6b_stratified_schedule_and_sparky_val_gate():
    import random
    from galcon.train import (
        _build_stratified_gen4_opp_schedule,
        _compute_gen4_val_gate,
        _should_promote_gen4_val,
    )

    # 1. Stratified dual-boss schedule guarantees exact Sparky/EsparanBot allocation every iter
    sched = _build_stratified_gen4_opp_schedule(
        12, has_lua1=True, has_lua2=True, pool_size=4, rng=random.Random(123), sparky_share=0.42, esparan_share=0.17
    )
    assert len(sched) == 12
    assert sum(1 for k, _ in sched if k == "boss") == 5
    assert sum(1 for k, _ in sched if k == "boss2") == 2
    assert sum(1 for k, _ in sched if k == "snap") == 3
    assert sum(1 for k, _ in sched if k in {"tactical", "macro", "rusher", "classic"}) == 2

    # 2. Sparky-gated validation rejects checkpoints that trade Sparky wins for heuristic wins
    start = _compute_gen4_val_gate(
        {"boss": (8.0, 16), "boss2": (5.0, 8), "anc": (4.0, 8), "heur": (10.0, 12)},
        has_lua1=True,
        has_lua2=True,
    )
    regressed_sparky = _compute_gen4_val_gate(
        {"boss": (6.0, 16), "boss2": (7.0, 8), "anc": (5.0, 8), "heur": (12.0, 12)},
        has_lua1=True,
        has_lua2=True,
    )
    assert not _should_promote_gen4_val(regressed_sparky, start, start, has_lua1=True)

    # 3. Genuine dual-boss improvement promotes cleanly
    improved = _compute_gen4_val_gate(
        {"boss": (9.0, 16), "boss2": (6.0, 8), "anc": (4.0, 8), "heur": (10.0, 12)},
        has_lua1=True,
        has_lua2=True,
    )
    assert _should_promote_gen4_val(improved, start, start, has_lua1=True)


def test_wave7_ms_probs_search_and_ensemble_gbots(tmp_path):
    import numpy as np
    from galcon.gbots_client import _build_bot
    from galcon.sim import GalconSim
    from galcon.transformer_bot import (
        GalconSearchTransformerAgent,
        build_galcon_gen4_transformer,
        extract_galcon_features,
    )

    m1 = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt1 = tmp_path / "m1.pt"
    ckpt2 = tmp_path / "m2.pt"
    for p in (ckpt1, ckpt2):
        torch.save(
            {
                "model_config": m1.config,
                "model_state_dict": m1.state_dict(),
                "has_trained_redir": True,
                "has_trained_multi_send": True,
                "use_canonical_d4": True,
            },
            p,
        )

    sim = GalconSim.create(seed=14700, num_planets=18, width=400.0, height=240.0)
    ag = GalconSearchTransformerAgent(ckpt1, use_d4_canonical=True, use_d4_tta=False)
    enc = extract_galcon_features(sim, 1, feature_version=2)
    probs = np.full((len(enc.planet_ids), len(enc.planet_ids) + 1), 0.05, dtype=np.float32)
    ms_probs = np.full((len(enc.planet_ids), len(enc.planet_ids)), 0.75, dtype=np.float32)
    fan = ag._build_fanout_plan(sim, enc, probs, ms_probs=ms_probs, max_targets_per_planet=6, reserve_ratio=0.06)
    assert isinstance(fan, list)

    ens_act = _build_bot("ensemble", str(ckpt1), str(ckpt2))
    orders = ens_act(sim, 1)
    assert isinstance(orders, list)


def test_wave8_vectorized_futures_and_adaptive_large_horizon(tmp_path: Path):
    import random
    from galcon.sim import project_all_planet_futures, project_planet_future
    from galcon.train import _sample_gen3_map_config
    from galcon.transformer_bot import build_galcon_gen4_transformer

    # 1. Exact numerical parity between vectorized project_all_planet_futures and project_planet_future
    for seed, n_pl, w, h, sym in [(14001, 18, 400.0, 240.0, True), (14702, 26, 560.0, 340.0, False)]:
        sim = GalconSim.create(seed=seed, num_planets=n_pl, width=w, height=h, symmetric=sym)
        tact = GalconTacticalBot()
        for _ in range(15):
            sim.step({1: tact.act(sim, 1), 2: tact.act(sim, 2)})
            if sim.done:
                break
        planets = sorted(sim.planets.values(), key=lambda p: p.id)
        for player in (1, 2):
            my_a, en_a, po_a, ps_a = project_all_planet_futures(sim, planets, player, horizon=24)
            for idx, p in enumerate(planets):
                r_my, r_en, r_po, r_ps = project_planet_future(sim, p.id, player, horizon=24)
                assert np.allclose(my_a[idx], r_my, atol=1e-6)
                assert np.allclose(en_a[idx], r_en, atol=1e-6)
                assert np.allclose(po_a[idx], r_po, atol=1e-6)
                assert np.allclose(ps_a[idx], r_ps, atol=1e-6)

    # 2. _sample_gen3_map_config respects large_prob
    rng = random.Random(777)
    samples = [_sample_gen3_map_config(rng, large_prob=0.50) for _ in range(200)]
    large_frac = sum(1 for n_pl, _, _, _ in samples if n_pl >= 22) / len(samples)
    assert 0.35 <= large_frac <= 0.65

    # 3. Map-adaptive Large horizon runs cleanly on 24pl asymmetric arena
    m = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "w8_test.pt"
    torch.save(
        {
            "model_config": m.config,
            "model_state_dict": m.state_dict(),
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt,
    )
    ag = GalconSearchTransformerAgent(ckpt, rollout_horizon=16, num_plies=2, time_budget_sec=0.20)
    sim_lg = GalconSim.create(seed=14700, num_planets=24, width=560.0, height=340.0, symmetric=False)
    plan, cands, scores, best_sc = ag.act_with_search_info(sim_lg, 1)
    assert isinstance(plan, list)
    assert len(cands) == len(scores) >= 1
    assert -1.0 <= best_sc <= 1.0



