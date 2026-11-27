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
    num_games = 200
    for s in range(num_games):
        sim = GalconSim.create(seed=100 + s, max_ticks=200)
        t_pid = 1 if (s % 2 == 0) else 2
        c_pid = 2 if t_pid == 1 else 1
        while not sim.done:
            if sim.tick % 2 == 0:
                sim.step({t_pid: tactical.act(sim, t_pid), c_pid: classic.act(sim, c_pid)})
            else:
                sim.step({})
        if sim.winner() == t_pid:
            wins += 1
    assert wins >= 120, f"Expected GalconTacticalBot to dominate GalconClassicBot, got {wins}/{num_games}"


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

    # 2. _sample_gen3_map_config respects large_prob, samples even planet counts, and prefers symmetric maps
    rng = random.Random(777)
    samples = [_sample_gen3_map_config(rng, large_prob=0.50) for _ in range(200)]
    large_frac = sum(1 for n_pl, _, _, _ in samples if n_pl >= 22) / len(samples)
    sym_frac = sum(1 for _, _, _, sym in samples if sym) / len(samples)
    assert 0.35 <= large_frac <= 0.65
    assert sym_frac >= 0.75
    assert all(n_pl % 2 == 0 for n_pl, _, _, _ in samples)

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


def test_wave9_map_adaptive_adversarial_and_decoupled_ensemble_value_weights(tmp_path: Path):
    from galcon.transformer_bot import EnsembleGalconSearchAgent, build_galcon_gen4_transformer

    ckpts = []
    for idx, sd in enumerate((101, 202)):
        torch.manual_seed(sd)
        m = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
        p = tmp_path / f"w9_m{idx}.pt"
        torch.save(
            {
                "model_config": m.config,
                "model_state_dict": m.state_dict(),
                "has_trained_redir": True,
                "has_trained_multi_send": True,
                "use_canonical_d4": True,
            },
            p,
        )
        ckpts.append(p)

    sim = GalconSim.create(seed=14000, num_planets=18, width=400.0, height=240.0, symmetric=True)
    enc = extract_galcon_features(sim, player=1, feature_version=2)

    # 1. Map-adaptive adversarial_search is enabled by default for <22pl and runs cleanly on both Std & Large
    ag0 = GalconSearchTransformerAgent(ckpts[0], use_d4_canonical=True, use_d4_tta=False)
    assert ag0.adversarial_search is True
    plan_std, cands_std, scores_std, sc_std = ag0.act_with_search_info(sim, 1)
    assert isinstance(plan_std, list) and len(cands_std) == len(scores_std) >= 1 and -1.0 <= sc_std <= 1.0

    # 2. Decoupled policy weights [1, 0] and value weights [0, 1] route policy to m0 and value to m1
    ag1 = GalconSearchTransformerAgent(ckpts[1], use_d4_canonical=True, use_d4_tta=False)
    p0, _, v0, _, ms0, _ = ag0._predict_all(enc)
    _, _, v1, _, _, _ = ag1._predict_all(enc)
    ens = EnsembleGalconSearchAgent(
        ckpts,
        weights=[1.0, 0.0],
        value_weights=[0.0, 1.0],
        use_d4_canonical=True,
        use_d4_tta=False,
    )
    pe, _, ve, _, mse, _ = ens._predict_all(enc)
    assert np.allclose(pe, p0, atol=1e-5)
    assert mse is not None and ms0 is not None and np.allclose(mse, ms0, atol=1e-5)
    assert abs(ve - v1) < 1e-5
    leaf_v = ens._batch_evaluate_leaves([enc])
    assert abs(float(leaf_v[0]) - v1) < 1e-5


def test_wave10_botwar_lua_bridge_and_sb_stats(tmp_path: Path):
    import shutil

    if not (shutil.which("lua5.3") or shutil.which("lua")):
        return
    from galcon.lua_bridge import GalconLuaBotBridge

    mock_botwar = tmp_path / "mock_botwar.lua"
    mock_botwar.write_text(
        """
function _sandbox_init(env)
    function env.bots_mock_waffle(params)
        local items = params.items
        local user = params.user
        -- Exercise Lua 5.3 pow shim and sb_stats() budget guard inside botwar sandbox
        local _p = env.pow(2, 3)
        local guard = 0
        repeat
            guard = guard + 1
        until env.sb_stats() > 10 or guard > 100
        assert(guard <= 20, "sb_stats() budget did not increment")

        local my_fleets = {}
        local my_planets = {}
        local targets = {}
        for k, it in pairs(items) do
            if it.is_fleet and it.owner == user then
                my_fleets[#my_fleets + 1] = k
            elseif it.is_planet then
                if it.owner == user and it.ships > 10 then
                    my_planets[#my_planets + 1] = k
                elseif it.owner ~= user then
                    targets[#targets + 1] = k
                end
            end
        end
        if #my_fleets > 0 and #targets > 0 then
            local dst = targets[1]
            for _, tk in ipairs(targets) do
                if items[tk].n ~= items[my_fleets[1]].target then
                    dst = tk
                    break
                end
            end
            return { percent = 50, from = my_fleets, to = dst }
        end
        if #my_planets > 0 and #targets > 0 then
            return { percent = 50, from = my_planets, to = targets[1] }
        end
        return nil
    end
end
""",
        encoding="utf-8",
    )

    bridge = GalconLuaBotBridge(f"{mock_botwar}::bots_mock_waffle")
    try:
        assert bridge.loop_fn == "bots_mock_waffle"
        sim = GalconSim.create(seed=1010, num_planets=10, width=280.0, height=180.0)
        o0 = bridge.act(sim, 1)
        assert any(isinstance(o, SendOrder) and o.src_id == 1 for o in o0)
        sim.step({1: o0})
        assert len(sim.fleets) >= 1
        o1 = bridge.act(sim, 1)
        assert any(isinstance(o, RedirOrder) for o in o1)
        # Verify clean reset on a fresh game at tick 0
        sim2 = GalconSim.create(seed=1011, num_planets=10, width=280.0, height=180.0)
        o2 = bridge.act(sim2, 2)
        assert any(isinstance(o, SendOrder) and o.src_id == 2 for o in o2)
    finally:
        bridge.close()


def test_wave10_triple_boss_schedule_val_gate_and_lua_spec(tmp_path: Path):
    import random
    from galcon.train import (
        _build_stratified_gen4_opp_schedule,
        _compute_gen4_val_gate,
        _lua_spec_exists,
        _lua_spec_label,
        _lua_spec_resolve,
        _should_promote_gen4_val,
    )

    # 1. path::func helper parsing
    fake_lua = tmp_path / "mod_botwar_testing.lua"
    fake_lua.write_text("-- mock\n", encoding="utf-8")
    spec = f"{fake_lua}::bots_waffle3z_optimized"
    assert _lua_spec_exists(spec)
    assert not _lua_spec_exists(f"{tmp_path / 'missing.lua'}::bots_waffle3z_optimized")
    assert _lua_spec_label(spec) == "bots_waffle3z_optimized"
    assert _lua_spec_resolve(spec).endswith("::bots_waffle3z_optimized")

    # 2. Triple-boss stratified schedule allocates Sparky, Waffle3z-GA, EsparanBot, snap, and heur
    sched = _build_stratified_gen4_opp_schedule(
        20,
        has_lua1=True,
        has_lua2=True,
        has_lua3=True,
        pool_size=4,
        rng=random.Random(77),
        sparky_share=0.35,
        waffle_share=0.20,
        esparan_share=0.10,
    )
    assert len(sched) == 20
    assert sum(1 for k, _ in sched if k == "boss") == 7
    assert sum(1 for k, _ in sched if k == "boss3") == 4
    assert sum(1 for k, _ in sched if k == "boss2") == 2
    assert sum(1 for k, _ in sched if k == "snap") == 4
    assert sum(1 for k, _ in sched if k in {"tactical", "macro", "rusher", "classic"}) == 3

    # 3. Triple-boss validation gate weights (0.40 spky + 0.25 waff + 0.15 esp + 0.10 anc + 0.10 heur)
    start = _compute_gen4_val_gate(
        {"boss": (8.0, 16), "boss3": (6.0, 12), "boss2": (6.0, 8), "anc": (4.0, 8), "heur": (10.0, 12)},
        has_lua1=True,
        has_lua2=True,
        has_lua3=True,
    )
    assert abs(start["waff"] - 0.50) < 1e-9
    expected_comp = 0.40 * 0.50 + 0.25 * 0.50 + 0.15 * 0.75 + 0.10 * 0.50 + 0.10 * (10.0 / 12.0)
    assert abs(start["comp"] - expected_comp) < 1e-6

    # Reject a checkpoint that drops Waffle3z heavily while farming EsparanBot + heuristics
    regressed_waffle = _compute_gen4_val_gate(
        {"boss": (8.0, 16), "boss3": (3.0, 12), "boss2": (8.0, 8), "anc": (8.0, 8), "heur": (12.0, 12)},
        has_lua1=True,
        has_lua2=True,
        has_lua3=True,
    )
    assert not _should_promote_gen4_val(regressed_waffle, start, start, has_lua1=True, has_lua3=True)

    # Promote genuine Sparky + Waffle3z multi-boss gain
    improved = _compute_gen4_val_gate(
        {"boss": (9.0, 16), "boss3": (8.0, 12), "boss2": (6.0, 8), "anc": (4.0, 8), "heur": (10.0, 12)},
        has_lua1=True,
        has_lua2=True,
        has_lua3=True,
    )
    assert _should_promote_gen4_val(improved, start, start, has_lua1=True, has_lua3=True)


def test_wave11_soft_kd_and_batched_ppo_rollouts():
    from galcon.heuristic_bot import GalconClassicBot, GalconTacticalBot
    from galcon.train import _gen4_multisend_loss, _run_gen4_ppo_episodes_batched
    from galcon.transformer_bot import (
        GALCON_MAX_PLANETS,
        build_galcon_gen4_transformer,
        canonicalize_galcon_d4,
        pad_galcon_state,
    )

    torch.manual_seed(11)
    model = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)

    # 1. Batched lockstep PPO rollout across heterogeneous map sizes (10pl + 16pl)
    specs = [
        (7101, 10, 240.0, 150.0, True, 1, None, GalconClassicBot(), False),
        (7101, 10, 240.0, 150.0, True, 2, None, GalconClassicBot(), False),
        (7102, 16, 400.0, 240.0, True, 1, None, GalconTacticalBot(), False),
    ]
    outs = _run_gen4_ppo_episodes_batched(
        model.eval(),
        specs,
        gamma=0.985,
        gae_lam=0.95,
        frac_std=0.065,
        draw_penalty=-0.4,
        boss_draw_penalty=0.1,
        max_sends_per_planet=4,
    )
    assert len(outs) == 3
    for ep_steps, snaps, advs, rets, term_rew, won in outs:
        assert len(ep_steps) >= 1
        assert len(advs) == len(ep_steps) and len(rets) == len(ep_steps)
        assert won in (0, 1) and -1.5 <= term_rew <= 1.5

    # 2. Hybrid Search-EI + Ensemble Soft-KD loss computes finite gradients with kd_alpha=0.35
    sim = GalconSim.create(seed=7103, num_planets=12, width=320.0, height=200.0)
    enc = extract_galcon_features(sim, 1, feature_version=2)
    enc.planet_feats = canonicalize_galcon_d4(enc.planet_feats)
    pad = pad_galcon_state(enc)
    pf = torch.from_numpy(pad["planet_feats"][None])
    tf = torch.from_numpy(pad["timeline_feats"][None])
    ef = torch.from_numpy(pad["edge_feats"][None])
    le = torch.from_numpy(pad["legal_edges"][None])
    om = torch.from_numpy(pad["owned_mask"][None])
    pm = torch.from_numpy(pad["pad_mask"][None])
    ms_tar = torch.zeros((1, GALCON_MAX_PLANETS, GALCON_MAX_PLANETS), dtype=torch.float32)
    ms_fr = torch.zeros_like(ms_tar)
    t_msp = torch.full_like(ms_tar, 0.25)
    vtar = torch.tensor([0.5], dtype=torch.float32)
    aux_o = torch.zeros((1, GALCON_MAX_PLANETS), dtype=torch.long)
    aux_s = torch.zeros((1, GALCON_MAX_PLANETS), dtype=torch.float32)

    model.train()
    loss, metrics = _gen4_multisend_loss(
        model, pf, tf, ef, le, om, pm, ms_tar, ms_fr, vtar, aux_o, aux_s, t_msp=t_msp, kd_alpha=0.35
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["bce"] >= 0.0


def test_wave12_surgical_search_need_value_and_draw_filter(tmp_path):
    """Verify Wave 12 surgical rescue/snipe, direct-rush fallback, and need_value=False bit-exact parity."""
    import numpy as np
    from galcon.sim import GalconSim, SendOrder
    from galcon.transformer_bot import (
        EnsembleGalconSearchAgent,
        GalconSearchTransformerAgent,
        build_galcon_gen4_transformer,
        extract_galcon_features,
    )

    torch.manual_seed(12)
    m1 = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    m2 = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    c1, c2 = tmp_path / "w12_m1.pt", tmp_path / "w12_m2.pt"
    for mod, p in ((m1, c1), (m2, c2)):
        torch.save(
            {
                "model_config": mod.config,
                "model_state_dict": mod.state_dict(),
                "has_trained_redir": True,
                "has_trained_multi_send": True,
                "use_canonical_d4": True,
            },
            p,
        )

    # 1. Defaults: single-model keeps rescue/rush off, Ensemble turns both on
    single = GalconSearchTransformerAgent(c1, use_d4_canonical=True, use_d4_tta=False)
    assert single.enable_rescue_snipe is False and single.enable_opp_direct_rush is False
    ens = EnsembleGalconSearchAgent(
        [c1, c2], weights=[1.0, 0.0], value_weights=[0.5, 0.5], use_d4_canonical=True, use_d4_tta=False
    )
    assert ens.enable_rescue_snipe is True and ens.enable_opp_direct_rush is True

    # 2. need_value=False skips value-only ensemble member with bit-exact policy parity
    sim = GalconSim.create(seed=14006, num_planets=18, width=400.0, height=240.0, symmetric=True)
    enc = extract_galcon_features(sim, 1, feature_version=2)
    p1, f1, _, _, msp1, msf1 = ens._predict_all(enc, need_value=True)
    p2, f2, _, _, msp2, msf2 = ens._predict_all(enc, need_value=False)
    assert np.allclose(p1, p2, atol=1e-6) and np.allclose(f1, f2, atol=1e-6)
    assert msp1 is not None and msp2 is not None and np.allclose(msp1, msp2, atol=1e-6)
    bp1, bf1, _, bmsp1, bmsf1 = ens._batch_forward_policy_and_val([enc], need_value=True)
    bp2, bf2, _, bmsp2, bmsf2 = ens._batch_forward_policy_and_val([enc], need_value=False)
    assert np.allclose(bp1, bp2, atol=1e-6) and np.allclose(bf1, bf2, atol=1e-6)
    assert bmsp1 is not None and bmsp2 is not None and np.allclose(bmsp1, bmsp2, atol=1e-6)

    # 3. Direct-rush fallback and rescue/snipe helpers produce valid legal SendOrders
    opp_enc = extract_galcon_features(sim, 2, feature_version=2)
    rush = ens._build_opp_direct_rush_plan(sim, opp_enc)
    assert len(rush) == 1 and isinstance(rush[0], SendOrder) and rush[0].percent == 85
    rs = ens._build_rescue_and_snipe_plan(sim, enc, p1, ms_probs=msp1)
    assert isinstance(rs, list)
    best, cands, scores, _ = ens.act_with_search_info(sim, 1)
    assert len(cands) >= 2 and len(scores) == len(cands)


def test_wave12_vm_speedups_forward_ms_bucket_sampler_and_batched_val():
    """Verify bit-exact parity of forward_multi_send, length-bucketed batching, and batched val_56g waves."""
    import numpy as np
    from galcon.heuristic_bot import GalconClassicBot, GalconTacticalBot
    from galcon.sim import GalconSim
    from galcon.train import _LengthBucketBatchSampler, _run_gen4_val_wave_batched
    from galcon.transformer_bot import (
        build_galcon_gen4_transformer,
        canonicalize_galcon_d4,
        extract_galcon_features,
    )

    torch.manual_seed(120)
    model = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128).eval()
    sim = GalconSim.create(seed=7200, num_planets=16, width=400.0, height=240.0)
    enc = extract_galcon_features(sim, 1, feature_version=2)
    pf = torch.from_numpy(canonicalize_galcon_d4(enc.planet_feats)[None])
    tf = torch.from_numpy(enc.timeline_feats[None])
    ef = torch.from_numpy(enc.edge_feats[None])
    pm = torch.ones((1, len(enc.planet_ids)), dtype=torch.bool)

    # 1. forward_multi_send matches forward() + predict_multi_send() bit-exactly
    with torch.no_grad():
        _, _, v_ref, aux_ref, h_ref = model(pf, tf, ef, pm, return_aux=True, return_hidden=True)
        msl_ref, msf_ref = model.predict_multi_send(h_ref, ef)
        msl_fast, msf_fast, v_fast, aux_fast, h_fast = model.forward_multi_send(
            pf, tf, ef, pm, return_aux=True, return_hidden=True
        )
    assert torch.allclose(v_ref, v_fast, atol=1e-6)
    assert torch.allclose(aux_ref, aux_fast, atol=1e-6)
    assert torch.allclose(msl_ref, msl_fast, atol=1e-6)
    assert torch.allclose(msf_ref, msf_fast, atol=1e-6)
    assert torch.allclose(h_ref, h_fast, atol=1e-6)

    # 2. _LengthBucketBatchSampler keeps small (<=14), mid (15-18), and large (>=19) maps in separate batches
    lengths = [10, 24, 12, 18, 26, 14, 16, 22, 18, 10]
    sampler = _LengthBucketBatchSampler(lengths, batch_size=3, shuffle=True, seed=99)
    seen = []
    for batch in sampler:
        seen.extend(batch)
        b_lens = [lengths[i] for i in batch]
        assert (max(b_lens) <= 14) or (15 <= min(b_lens) and max(b_lens) <= 18) or (min(b_lens) >= 19)
    assert sorted(seen) == list(range(len(lengths)))

    # 3. Batched lockstep validation wave executes cleanly across heterogeneous maps
    wave = [
        ("heur", "classic", 7201, 10, 240.0, 150.0, 100, True, 1, GalconClassicBot(), False),
        ("heur", "tactical", 7202, 16, 400.0, 240.0, 100, True, 2, GalconTacticalBot(), False),
    ]
    res = _run_gen4_val_wave_batched(model, wave)
    assert len(res) == 2 and all(bk == "heur" and pts in (0.0, 0.5, 1.0) for bk, pts in res)


def test_wave13_fast_c_canonical_d4_vectorized_redir_and_ei_cache(tmp_path):
    """Verify Wave 13 C99 feature extractor parity, lazy D4 canonicalization, vectorized redir, ms_only, and EI cache."""
    import numpy as np
    from galcon.fast_c import HAS_FAST_C
    from galcon.sim import GalconSim, SendOrder
    from galcon.train import load_ei_dataset_npz, save_ei_dataset_npz
    from galcon.transformer_bot import (
        GalconSearchTransformerAgent,
        apply_galcon_d4,
        build_galcon_gen4_transformer,
        canonicalize_galcon_d4,
        extract_galcon_features,
        extract_galcon_redir_features,
    )

    # 1. C99 feature extractor matches pure-NumPy fallback bit-for-bit across maps, versions, and active fleets
    for seed, n_pl in ((1301, 10), (1302, 18), (1303, 24)):
        sim = GalconSim.create(seed=seed, num_planets=n_pl, width=400.0, height=240.0, symmetric=True)
        pids = sorted(sim.planets.keys())
        for step in range(8):
            o1, o2 = [], []
            for pl_id, olist in ((1, o1), (2, o2)):
                owned = [pid for pid in pids if sim.planets[pid].owner == pl_id and sim.planets[pid].ships >= 8]
                if owned:
                    src = owned[step % len(owned)]
                    dst = pids[(pids.index(src) + 1 + step) % len(pids)]
                    if src != dst:
                        olist.append(SendOrder(src, dst, 50))
            sim.step({1: o1, 2: o2})
        for fv in (1, 2):
            enc_c = extract_galcon_features(sim, 1, feature_version=fv, _use_fast_c=True)
            enc_py = extract_galcon_features(sim, 1, feature_version=fv, _use_fast_c=False)
            assert np.allclose(enc_c.planet_feats, enc_py.planet_feats, atol=1e-5)
            assert np.allclose(enc_c.timeline_feats, enc_py.timeline_feats, atol=1e-5)
            assert np.allclose(enc_c.edge_feats, enc_py.edge_feats, atol=1e-5)
            assert np.allclose(enc_c.safe_surplus, enc_py.safe_surplus, atol=1e-5)
            assert np.array_equal(enc_c.legal_edges, enc_py.legal_edges)
            assert np.array_equal(enc_c.owned_mask, enc_py.owned_mask)

        # 2. Vectorized extract_galcon_redir_features produces finite valid tensors with active fleets
        enc_v2 = extract_galcon_features(sim, 1, feature_version=2)
        redir_enc = extract_galcon_redir_features(sim, enc_v2)
        assert len(redir_enc.fleet_ids) > 0
        assert redir_enc.redir_feats.shape == (len(redir_enc.fleet_ids), len(enc_v2.planet_ids), 10)
        assert np.isfinite(redir_enc.redir_feats).all()

    # 3. Fast lazy canonicalize_galcon_d4 preserves exact 8-fold D4 orbit invariance (including symmetric tie case)
    pf_ref = extract_galcon_features(
        GalconSim.create(seed=1304, num_planets=14, width=400.0, height=240.0, symmetric=True), 1, feature_version=2
    ).planet_feats
    canon0 = canonicalize_galcon_d4(pf_ref)
    for flip_x in (False, True):
        for k_rot in range(4):
            cand = apply_galcon_d4(pf_ref, k_rot=k_rot, flip_x=flip_x)
            assert np.allclose(canonicalize_galcon_d4(cand), canon0, atol=1e-6)
    pf_tie = pf_ref.copy()
    pf_tie[:, 2] = 0.0
    pf_tie[:, 3] = 0.0
    pf_tie[0, 6] += 0.37  # break 180-deg map automorphism while keeping vx0==vy0==0 tie-break
    canon_tie = canonicalize_galcon_d4(pf_tie)
    for flip_x in (False, True):
        for k_rot in range(4):
            assert np.allclose(canonicalize_galcon_d4(apply_galcon_d4(pf_tie, k_rot=k_rot, flip_x=flip_x)), canon_tie, atol=1e-6)

    # 4. ms_only=True matches ms_only=False multi-send predictions bit-for-bit
    torch.manual_seed(13)
    mod = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "w13_m.pt"
    torch.save(
        {
            "model_config": mod.config,
            "model_state_dict": mod.state_dict(),
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt,
    )
    ag = GalconSearchTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False)
    sim_t = GalconSim.create(seed=1305, num_planets=16, width=400.0, height=240.0, symmetric=True)
    enc_t = extract_galcon_features(sim_t, 1, feature_version=2)
    _, _, v_full, _, msp_full, msf_full = ag._predict_all(enc_t, need_value=True, ms_only=False)
    _, _, v_fast, _, msp_fast, msf_fast = ag._predict_all(enc_t, need_value=True, ms_only=True)
    assert abs(v_full - v_fast) < 1e-6
    assert msp_full is not None and msp_fast is not None and np.allclose(msp_full, msp_fast, atol=1e-6)
    assert msf_full is not None and msf_fast is not None and np.allclose(msf_full, msf_fast, atol=1e-6)
    _, _, _, bmsp_full, bmsf_full = ag._batch_forward_policy_and_val([enc_t], need_value=False, ms_only=False)
    _, _, _, bmsp_fast, bmsf_fast = ag._batch_forward_policy_and_val([enc_t], need_value=False, ms_only=True)
    assert bmsp_full is not None and bmsp_fast is not None and np.allclose(bmsp_full, bmsp_fast, atol=1e-6)
    assert bmsf_full is not None and bmsf_fast is not None and np.allclose(bmsf_full, bmsf_fast, atol=1e-6)

    # 5. EI replay buffer .npz save/load & comma-separated pooling roundtrip
    d1 = {"pf": np.ones((4, 3), dtype=np.float32), "v": np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32)}
    d2 = {"pf": np.zeros((3, 3), dtype=np.float32), "v": np.array([0.2, 0.3, -0.4], dtype=np.float32)}
    p1, p2 = tmp_path / "ei1.npz", tmp_path / "ei2.npz"
    save_ei_dataset_npz(p1, d1)
    save_ei_dataset_npz(p2, d2)
    pooled = load_ei_dataset_npz(f"{p1},{p2}")
    assert pooled["pf"].shape == (7, 3) and pooled["v"].shape == (7,)
    assert np.allclose(pooled["pf"][:4], 1.0) and np.allclose(pooled["pf"][4:], 0.0)
    assert isinstance(HAS_FAST_C, bool)


def test_wave15_gen5_edge_gated_wdl_architecture(tmp_path):
    """Verify Gen 5 Edge-Gated Residual-Edge Transformer, WDL value head + contempt, Search agent, and WDL loss."""
    import numpy as np
    from galcon.gen5_model import GalconGen5Transformer, build_galcon_gen5_transformer
    from galcon.sim import GalconSim
    from galcon.train import _gen4_multisend_loss, _load_gen4_policy_from_ckpt
    from galcon.transformer_bot import (
        GalconSearchTransformerAgent,
        GalconTransformerAgent,
        extract_galcon_features,
    )

    torch.manual_seed(15)
    model = build_galcon_gen5_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128, edge_hidden=16)
    ckpt = tmp_path / "gen5_test.pt"
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

    # 1. Loader auto-detects GalconGen5Transformer in both transformer_bot and train
    loaded = _load_gen4_policy_from_ckpt(ckpt)
    assert isinstance(loaded, GalconGen5Transformer)

    # 2. WDL value head + contempt behavior
    sim = GalconSim.create(seed=1501, num_planets=14, width=400.0, height=240.0, symmetric=True)
    enc = extract_galcon_features(sim, 1, feature_version=2)
    pf = torch.from_numpy(enc.planet_feats[None])
    tf = torch.from_numpy(enc.timeline_feats[None])
    ef = torch.from_numpy(enc.edge_feats[None])
    pm = torch.ones((1, len(enc.planet_ids)), dtype=torch.bool)
    with torch.no_grad():
        h = loaded.encode_planets(pf, tf, ef, pm)
        wdl_logits = loaded.decode_wdl_from_hidden(h, pm)
        wdl_p = torch.softmax(wdl_logits, dim=-1)
        assert wdl_p.shape == (1, 3) and abs(float(wdl_p.sum().item()) - 1.0) < 1e-5
        v0 = float(loaded.decode_value_from_hidden(h, pm, contempt=0.0).item())
        vc = float(loaded.decode_value_from_hidden(h, pm, contempt=0.25).item())
        assert vc <= v0 + 1e-6  # positive contempt lowers value proportionally to P(draw)

    # 3. Pure and Search agents run seamlessly on Gen 5 checkpoints (default contempt=0.30, explicit override works)
    ag_pure = GalconTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False)
    ag_search_def = GalconSearchTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False)
    ag_search = GalconSearchTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False, contempt=0.10)
    assert isinstance(ag_pure.model, GalconGen5Transformer)
    assert abs(ag_search_def.model.contempt - 0.30) < 1e-6
    assert abs(ag_search.model.contempt - 0.10) < 1e-6
    _ = ag_pure.act(sim, 1)
    best, cands, scores, _ = ag_search.act_with_search_info(sim, 1)
    assert len(cands) >= 1 and len(scores) == len(cands)

    # 4. Dropout config preserves 100% state_dict compatibility with dropout=0.0 models
    m_nodrop = build_galcon_gen5_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128, edge_hidden=16, dropout=0.0)
    m_nodrop.load_state_dict(model.state_dict())
    m_nodrop.eval()
    model.eval()
    with torch.no_grad():
        assert torch.allclose(m_nodrop.encode_planets(pf, tf, ef, pm), model.encode_planets(pf, tf, ef, pm), atol=1e-6)

    # 5. Calibrated quadratic soft-WDL + Soft-KD loss backward pass works end-to-end
    n = len(enc.planet_ids)
    le = torch.from_numpy(enc.legal_edges[None])
    om = torch.from_numpy(enc.owned_mask[None])
    ms_tar = torch.zeros((1, n, n), dtype=torch.float32)
    ms_fr = torch.zeros((1, n, n), dtype=torch.float32)
    t_msp = torch.full((1, n, n), 0.1, dtype=torch.float32)
    vtar = torch.tensor([0.6], dtype=torch.float32)
    aux_o = torch.zeros((1, n), dtype=torch.long)
    aux_s = torch.zeros((1, n), dtype=torch.float32)
    model.train()
    loss, metrics = _gen4_multisend_loss(
        model, pf, tf, ef, le, om, pm, ms_tar, ms_fr, vtar, aux_o, aux_s, pos_weight=2.2, t_msp=t_msp, kd_alpha=0.40
    )
    assert torch.isfinite(loss) and metrics["bce"] >= 0.0 and metrics["v_loss"] >= 0.0
    loss.backward()


def test_wave16_def_stream_opening_guard_and_1t_ei(tmp_path):
    """Verify Wave 16 DefStream + Opening Defense Guard defaults, helper plans, and full-speed 1-tick EI collection."""
    import numpy as np
    from galcon.gen5_model import build_galcon_gen5_transformer
    from galcon.sim import GalconSim, SendOrder
    from galcon.train import collect_gen4_multisend_ei_dataset
    from galcon.transformer_bot import (
        GalconSearchTransformerAgent,
        extract_galcon_features,
    )

    torch.manual_seed(16)
    model = build_galcon_gen5_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128, edge_hidden=16)
    ckpt = tmp_path / "gen5_w16_test.pt"
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

    # 1. Defaults: enable_def_stream and enable_opening_def_guard are enabled by default in Wave 16
    ag = GalconSearchTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False)
    assert ag.enable_def_stream is True and ag.enable_opening_def_guard is True

    # 2. _build_defense_and_pressure_plan generates exact defense send for a threatened owned planet
    sim = GalconSim.create(seed=1601, num_planets=14, width=400.0, height=240.0, symmetric=True)
    # Give player 1 two owned planets: one threatened by an incoming P2 fleet and one rear helper with surplus
    my_home = next(p for p in sim.planets.values() if p.owner == 1)
    opp_home = next(p for p in sim.planets.values() if p.owner == 2)
    helper = min(
        (p for p in sim.planets.values() if p.owner == 0),
        key=lambda p: sim.planet_eta_ticks(p, my_home),
    )
    helper.owner = 1
    helper.ships = 60.0
    my_home.ships = 10.0
    opp_home.ships = 80.0
    sim.step({2: [SendOrder(opp_home.id, my_home.id, 60)]})
    enc = extract_galcon_features(sim, 1, feature_version=2)
    plan_def = ag._build_defense_and_pressure_plan(sim, enc)
    assert isinstance(plan_def, list)
    assert any(isinstance(o, SendOrder) and o.dst_id == my_home.id for o in plan_def)

    # 3. Opening deep-strike guard executes cleanly at tick <= 8 without errors
    sim_open = GalconSim.create(seed=14002, num_planets=18, width=400.0, height=240.0, symmetric=True)
    best, cands, scores, best_sc = ag.act_with_search_info(sim_open, 1)
    assert isinstance(best, list) and len(cands) == len(scores) >= 2 and -1.0 <= best_sc <= 1.0

    # 4. Full-speed 1-tick EI collection (act_every=1) records valid states across even/odd ticks
    ei_data, _ = collect_gen4_multisend_ei_dataset(
        ckpt,
        num_games=1,
        seed=1602,
        act_every=1,
        large_map_prob=0.0,
        workers=1,
    )
    assert len(ei_data.get("val_targets", [])) >= 4
    assert np.isfinite(ei_data["val_targets"]).all()


def test_galcon_sim_radius_defaults_and_settle_clearance():
    import math
    import random
    import galcon
    from galcon.sim import (
        DEFAULT_MAP_HEIGHT,
        DEFAULT_MAP_WIDTH,
        DEFAULT_NUM_PLANETS,
        DEFAULT_NUM_PLAYERS,
        MIN_PLANET_GAP,
        NEUTRAL_PROD_MAX,
        NEUTRAL_PROD_MIN,
        NEUTRAL_SHIPS_MAX,
        NEUTRAL_SHIPS_MIN,
        PLANET_SETTLE_DELTA,
        SHIP_RADIUS,
        generate_galcon_map,
        planet_radius_from_production,
        settle_planets,
        symmetric_planet_count,
    )
    from galcon.train import _sample_gen3_map_config, _sample_map_config

    # 1. Exact Galcon 2 planet radius formula + lower clamping at prod <= 15 + even 1v1 defaults
    assert SHIP_RADIUS == 6.0
    assert PLANET_SETTLE_DELTA == 0.5
    assert MIN_PLANET_GAP == 12.5
    assert DEFAULT_MAP_WIDTH == 480.0
    assert DEFAULT_MAP_HEIGHT == 320.0
    assert DEFAULT_NUM_PLAYERS == 2
    assert DEFAULT_NUM_PLANETS == 22
    assert galcon.DEFAULT_NUM_PLAYERS == DEFAULT_NUM_PLAYERS
    assert galcon.DEFAULT_NUM_PLANETS == DEFAULT_NUM_PLANETS
    assert galcon.symmetric_planet_count is symmetric_planet_count
    assert np.isclose(planet_radius_from_production(15), 12.0, atol=1e-12)
    assert np.isclose(planet_radius_from_production(100), 24.0, atol=1e-12)
    assert np.isclose(planet_radius_from_production(50), (50.0 * 12.0 / 5.0 + 168.0) / 17.0, atol=1e-12)
    for low_p in (-10.0, 0.0, 5.0, 14.99, 15.0):
        assert planet_radius_from_production(low_p) == 12.0

    def _assert_even_symmetric_no_center(plist, width: float, height: float) -> None:
        assert len(plist) % 2 == 0
        cx, cy = width * 0.5, height * 0.5
        for k in range(len(plist) // 2):
            pa, pb = plist[2 * k], plist[2 * k + 1]
            assert np.isclose(pa.x + pb.x, width, atol=1e-6)
            assert np.isclose(pa.y + pb.y, height, atol=1e-6)
            assert pa.production == pb.production and pa.ships == pb.ships and pa.radius == pb.radius
        for p in plist:
            assert math.hypot(p.x - cx, p.y - cy) >= p.radius + 0.5 * MIN_PLANET_GAP - 1e-6
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                dist = math.hypot(plist[i].x - plist[j].x, plist[i].y - plist[j].y)
                assert dist >= plist[i].radius + plist[j].radius + MIN_PLANET_GAP - 1e-6

    # 2. Default GalconSim.create() uses 480x320, 22 even planets, 12.5u clearance, exact symmetry,
    #    NO center planet at (width/2, height/2), and neutral ship (0..45) / prod (15..100) / radius invariants
    for seed in range(20):
        sim = GalconSim.create(seed=seed)
        assert sim.width == 480.0 and sim.height == 320.0
        plist = [sim.planets[pid] for pid in sorted(sim.planets.keys())]
        assert len(plist) == 22
        for p in plist:
            assert p.radius == planet_radius_from_production(p.production)
            assert p.radius + SHIP_RADIUS - 1e-6 <= p.x <= sim.width - p.radius - SHIP_RADIUS + 1e-6
            assert p.radius + SHIP_RADIUS - 1e-6 <= p.y <= sim.height - p.radius - SHIP_RADIUS + 1e-6
            if p.owner == 0:
                assert NEUTRAL_SHIPS_MIN <= p.ships <= NEUTRAL_SHIPS_MAX
                assert NEUTRAL_PROD_MIN <= p.production <= NEUTRAL_PROD_MAX
        _assert_even_symmetric_no_center(plist, sim.width, sim.height)

    # 2b. Odd num_planets in symmetric 1v1 mode drops remainder mod num_players (no center planet)
    for req_n, exp_n in ((23, 22), (19, 18), (15, 14)):
        for seed in (0, 7, 42):
            plist = generate_galcon_map(seed=seed, num_planets=req_n, width=480.0, height=320.0, symmetric=True)
            assert len(plist) == exp_n
            _assert_even_symmetric_no_center(plist, 480.0, 320.0)
            sim_odd = GalconSim.create(seed=seed, num_planets=req_n, width=480.0, height=320.0, symmetric=True)
            assert len(sim_odd.planets) == exp_n

    # 2c. General mod num_players helper & num_players=3/4 behavior
    assert symmetric_planet_count(23, 2) == 22
    assert symmetric_planet_count(22, 2) == 22
    assert symmetric_planet_count(19, 2) == 18
    assert symmetric_planet_count(15, 2) == 14
    assert symmetric_planet_count(23, 3) == 21
    assert symmetric_planet_count(20, 3) == 18
    assert symmetric_planet_count(23, 4) == 20
    assert symmetric_planet_count(24, 4) == 24
    assert symmetric_planet_count(1, 2) == 2

    pl_4p = generate_galcon_map(seed=11, num_planets=23, num_players=4, width=480.0, height=320.0, symmetric=True)
    assert len(pl_4p) == 20 and len(pl_4p) % 4 == 0
    _assert_even_symmetric_no_center(pl_4p, 480.0, 320.0)
    sim_4p = GalconSim.create(seed=11, num_planets=23, num_players=4, width=480.0, height=320.0, symmetric=True)
    assert len(sim_4p.planets) == 20

    pl_2p = generate_galcon_map(seed=11, num_planets=22, num_players=2, width=480.0, height=320.0, symmetric=True)
    assert len(pl_2p) == 22
    _assert_even_symmetric_no_center(pl_2p, 480.0, 320.0)

    for req_3p, exp_3p in ((20, 18), (23, 18)):
        pl_3p = generate_galcon_map(seed=11, num_planets=req_3p, num_players=3, width=480.0, height=320.0, symmetric=True)
        assert len(pl_3p) == exp_3p and len(pl_3p) % 3 == 0
        _assert_even_symmetric_no_center(pl_3p, 480.0, 320.0)

    # 2d. Training map samplers prefer symmetric maps and sample strictly even planet counts
    srng = random.Random(2026)
    base_samples = [_sample_map_config(srng) for _ in range(100)]
    assert all(n_pl % 2 == 0 for n_pl, _, _ in base_samples)
    gen3_samples = [_sample_gen3_map_config(srng, large_prob=0.35) for _ in range(300)]
    assert all(n_pl % 2 == 0 for n_pl, _, _, _ in gen3_samples)
    assert sum(1 for _, _, _, sym in gen3_samples if sym) / len(gen3_samples) >= 0.80

    # Asymmetric maps still honor explicit odd counts (e.g. 23), neutral bounds, radius, and 12.5u clearance
    for seed in range(5):
        asym = generate_galcon_map(seed=seed, num_planets=23, width=480.0, height=320.0, symmetric=False)
        assert len(asym) == 23
        for p in asym:
            assert p.radius == planet_radius_from_production(p.production)
            if p.owner == 0:
                assert 0.0 <= p.ships <= 45.0
                assert 15.0 <= p.production <= 100.0
        for i in range(len(asym)):
            for j in range(i + 1, len(asym)):
                dist = math.hypot(asym[i].x - asym[j].x, asym[i].y - asym[j].y)
                assert dist >= asym[i].radius + asym[j].radius + MIN_PLANET_GAP - 1e-6

    # 3. No early-break shortfall on 400x240, 18pl symmetric (including seed=204 which previously placed only 16)
    for seed in (0, 1, 2, 42, 204):
        sim18 = GalconSim.create(seed=seed, num_planets=18, width=400.0, height=240.0, symmetric=True)
        assert len(sim18.planets) == 18
        plist = [sim18.planets[pid] for pid in sorted(sim18.planets.keys())]
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                dist = math.hypot(plist[i].x - plist[j].x, plist[i].y - plist[j].y)
                assert dist >= plist[i].radius + plist[j].radius + MIN_PLANET_GAP - 1e-6

    # 4. settle_planets directly resolves synthetic overlaps while preserving 180-deg symmetry & center pin
    w, h = 480.0, 320.0
    r_home = planet_radius_from_production(100.0)
    r_mid = planet_radius_from_production(60.0)
    r_ctr = planet_radius_from_production(50.0)
    overlapping = [
        Planet(1, 1, 100.0, 230.0, 155.0, 100.0, r_home),
        Planet(2, 2, 100.0, w - 230.0, h - 155.0, 100.0, r_home),
        Planet(3, 0, 20.0, 235.0, 160.0, 60.0, r_mid),
        Planet(4, 0, 20.0, w - 235.0, h - 160.0, 60.0, r_mid),
        Planet(5, 0, 25.0, w * 0.5, h * 0.5, 50.0, r_ctr),
    ]
    settled = settle_planets(overlapping, w, h, min_gap=MIN_PLANET_GAP, symmetric=True, max_iters=30)
    assert settled is True
    assert np.isclose(overlapping[0].x + overlapping[1].x, w, atol=1e-6)
    assert np.isclose(overlapping[0].y + overlapping[1].y, h, atol=1e-6)
    assert np.isclose(overlapping[2].x + overlapping[3].x, w, atol=1e-6)
    assert np.isclose(overlapping[2].y + overlapping[3].y, h, atol=1e-6)
    assert np.isclose(overlapping[4].x, w * 0.5, atol=1e-6)
    assert np.isclose(overlapping[4].y, h * 0.5, atol=1e-6)
    for i in range(len(overlapping)):
        for j in range(i + 1, len(overlapping)):
            dist = math.hypot(overlapping[i].x - overlapping[j].x, overlapping[i].y - overlapping[j].y)
            assert dist >= overlapping[i].radius + overlapping[j].radius + MIN_PLANET_GAP - 1e-6


def test_galcon_sim_epsilon_tie_and_arrival_tolerance():
    """Verify epsilon-safe combat tie breaking, arrival boundary tolerance, and fast_c parity."""
    import math
    from galcon.sim import (
        ARRIVAL_EPS,
        SHIP_RADIUS,
        TIE_EPS,
        Fleet,
        GalconSim,
        Planet,
        project_all_planet_futures,
        project_planet_future,
    )
    from galcon.transformer_bot import extract_galcon_features, extract_galcon_redir_features

    assert TIE_EPS == 1e-9
    assert ARRIVAL_EPS == 1e-9

    step_dist = 40.0 * 0.25  # 10.0 units/tick
    p1 = Planet(id=1, owner=1, ships=50.0, x=0.0, y=50.0, production=100.0, radius=12.0)
    # Defender with IEEE-754 drift ships=4.999999999999984 and 0 production, placed at radius + SHIP_RADIUS + step_dist + 7e-15
    p2 = Planet(
        id=2,
        owner=2,
        ships=4.999999999999984,
        x=12.0 + SHIP_RADIUS + step_dist + 7e-15,
        y=0.0,
        production=0.0,
        radius=12.0,
    )
    # Pairwise planet boundary jitter: center-to-center = r1 + r3 + 2 * SHIP_RADIUS + step_dist + 7e-15
    p3 = Planet(
        id=3,
        owner=2,
        ships=50.0,
        x=12.0 + 12.0 + 2.0 * SHIP_RADIUS + step_dist + 7e-15,
        y=50.0,
        production=100.0,
        radius=12.0,
    )
    # Neutral defender with exact same IEEE-754 drift
    p4 = Planet(id=4, owner=0, ships=4.999999999999984, x=120.0, y=0.0, production=0.0, radius=12.0)

    fl_atk = Fleet(id=1000, owner=1, ships=5.0, x=0.0, y=0.0, source=1, target=2)
    fl_neu = Fleet(id=1001, owner=1, ships=5.0, x=100.0, y=0.0, source=1, target=4)

    # Confirm the 7e-15 overshoot is actually representable in float64 (strictly > exact boundary)
    assert math.hypot(p2.x - fl_atk.x, p2.y - fl_atk.y) > p2.radius + SHIP_RADIUS + step_dist
    assert math.hypot(p3.x - p1.x, p3.y - p1.y) > p1.radius + p3.radius + 2.0 * SHIP_RADIUS + step_dist

    sim = GalconSim([p1, p2, p3, p4], [fl_atk, fl_neu], fleet_speed=40.0, tick_dt=0.25)

    # 1. Arrival boundary 7e-15 jitter gives eta == 1 in both eta_ticks and planet_eta_ticks
    assert sim.eta_ticks(fl_atk.x, fl_atk.y, p2) == 1
    assert sim.planet_eta_ticks(p1, p3) == 1

    # 2. Projections keep defender (and neutral) ownership with 0.0 ships on near-tie at tick 1
    planets = [sim.planets[pid] for pid in sorted(sim.planets.keys())]
    for player, exp_p2_owner in ((1, -1.0), (2, 1.0)):
        _, _, po2, ps2 = project_planet_future(sim, 2, player, horizon=6)
        assert po2[0] == exp_p2_owner and ps2[0] == 0.0
        _, _, po4, ps4 = project_planet_future(sim, 4, player, horizon=6)
        assert po4[0] == 0.0 and ps4[0] == 0.0

        my_all, en_all, po_all, ps_all = project_all_planet_futures(sim, planets, player, horizon=6)
        assert po_all[1, 0] == exp_p2_owner and ps_all[1, 0] == 0.0
        assert po_all[3, 0] == 0.0 and ps_all[3, 0] == 0.0
        assert np.array_equal(po_all[1], po2) and np.array_equal(ps_all[1], ps2)
        assert np.array_equal(po_all[3], po4) and np.array_equal(ps_all[3], ps4)
        if player == 1:
            assert my_all[1, 0] == 5.0 and en_all[1, 0] == 0.0
        else:
            assert en_all[1, 0] == 5.0 and my_all[1, 0] == 0.0

    # 3. extract_galcon_features fast_c vs pure-NumPy parity on the exact near-tie + 7e-15 boundary state
    for player in (1, 2):
        for fv in (1, 2):
            enc_c = extract_galcon_features(sim, player, feature_version=fv, _use_fast_c=True)
            enc_py = extract_galcon_features(sim, player, feature_version=fv, _use_fast_c=False)
            assert np.allclose(enc_c.planet_feats, enc_py.planet_feats, atol=1e-5)
            assert np.allclose(enc_c.timeline_feats, enc_py.timeline_feats, atol=1e-5)
            assert np.allclose(enc_c.edge_feats, enc_py.edge_feats, atol=1e-5)
            assert np.allclose(enc_c.safe_surplus, enc_py.safe_surplus, atol=1e-5)
            assert np.array_equal(enc_c.legal_edges, enc_py.legal_edges)
            assert np.array_equal(enc_c.owned_mask, enc_py.owned_mask)
            assert enc_c.futures is not None and enc_py.futures is not None
            for fc, fp in zip(enc_c.futures, enc_py.futures):
                for ac, ap in zip(fc, fp):
                    assert np.allclose(ac, ap, atol=1e-6)
            # Pairwise p1 -> p3 edge (indices 0 -> 2) has eta == 1 (1/35 in col 0) despite 7e-15 overshoot
            assert np.isclose(enc_c.edge_feats[0, 2, 0], 1.0 / 35.0, atol=1e-6)
            assert np.isclose(enc_py.edge_feats[0, 2, 0], 1.0 / 35.0, atol=1e-6)
            # Defender keeps p2 (idx 1) with 0 ships at t=0 in both backends
            if player == 1:
                assert enc_c.timeline_feats[1, 0, 3] == 1.0 and enc_c.timeline_feats[1, 0, 5] == 0.0
                assert enc_py.timeline_feats[1, 0, 3] == 1.0 and enc_py.timeline_feats[1, 0, 5] == 0.0
            else:
                assert enc_c.timeline_feats[1, 0, 2] == 1.0 and enc_c.timeline_feats[1, 0, 5] == 0.0
                assert enc_py.timeline_feats[1, 0, 2] == 1.0 and enc_py.timeline_feats[1, 0, 5] == 0.0

    # Redir feature extractor also computes eta == 1 for fl_atk -> p2 despite 7e-15 jitter
    enc_p1 = extract_galcon_features(sim, 1, feature_version=2)
    redir_p1 = extract_galcon_redir_features(sim, enc_p1)
    idx_fl_atk = redir_p1.fleet_ids.index(1000)
    assert np.isclose(redir_p1.redir_feats[idx_fl_atk, 1, 1], 1.0 / 35.0, atol=1e-6)

    # 4. GalconSim.step() lands fl_atk on tick 1 and keeps defender/neutral ownership with 0.0 ships
    sim.step({})
    assert 1000 not in sim.fleets and 1001 not in sim.fleets
    assert sim.planets[2].owner == 2 and sim.planets[2].ships == 0.0
    assert sim.planets[4].owner == 0 and sim.planets[4].ships == 0.0

    # 5. Strictly above tolerance: arrival beyond ARRIVAL_EPS takes 2 ticks, and clear advantage captures
    p_def = Planet(id=1, owner=2, ships=5.0, x=12.0 + SHIP_RADIUS + step_dist + 1e-6, y=0.0, production=0.0, radius=12.0)
    p_home = Planet(id=2, owner=1, ships=50.0, x=200.0, y=0.0, production=100.0, radius=12.0)
    fl_far = Fleet(id=2000, owner=1, ships=5.0 + 1e-6, x=0.0, y=0.0, source=2, target=1)
    sim2 = GalconSim([p_def, p_home], [fl_far], fleet_speed=40.0, tick_dt=0.25)
    assert sim2.eta_ticks(fl_far.x, fl_far.y, p_def) == 2
    sim2.step({})
    assert 2000 in sim2.fleets  # did not arrive on tick 1
    sim2.step({})
    assert 2000 not in sim2.fleets
    assert sim2.planets[1].owner == 1 and np.isclose(sim2.planets[1].ships, 1e-6, atol=1e-12)


def test_galcon_sim_fractional_send_and_physical_ship_count():
    """Verify fractional ship packing, >= 1.0 send guard, physical_ship_count throttling, and clone parity."""
    import math
    import galcon
    from galcon.sim import (
        BASE_ITEM_USERS,
        ITEMS_MAX,
        ITEMS_RESERVED,
        Fleet,
        GalconSim,
        Planet,
        SendOrder,
        physical_ship_count,
    )

    # Re-exports & constants
    assert ITEMS_MAX == 4096
    assert ITEMS_RESERVED == 16
    assert BASE_ITEM_USERS == 3
    assert galcon.ITEMS_MAX == ITEMS_MAX
    assert galcon.ITEMS_RESERVED == ITEMS_RESERVED
    assert galcon.physical_ship_count is physical_ship_count
    assert physical_ship_count(0.99, 100.0, 10) == 0
    assert physical_ship_count(1.0 - 1e-12, 100.0, 10) == 1

    # 1. Planet with < 1.0 ships (0.99) cannot send at 100%
    p1 = Planet(id=1, owner=1, ships=0.99, x=20.0, y=100.0, production=0.0, radius=12.0)
    p2 = Planet(id=2, owner=2, ships=50.0, x=220.0, y=100.0, production=0.0, radius=12.0)
    sim = GalconSim([p1, p2], width=300.0, height=200.0)
    sim.step({1: [SendOrder(1, 2, 100)]})
    assert len(sim.fleets) == 0
    assert np.isclose(sim.planets[1].ships, 0.99, atol=1e-12)

    # 2. Planet with 1.5 ships cannot send at 50% (0.75 < 1.0), CAN send at 80% (1.2 ships)
    sim.planets[1].ships = 1.5
    sim.step({1: [SendOrder(1, 2, 50)]})
    assert len(sim.fleets) == 0
    assert np.isclose(sim.planets[1].ships, 1.5, atol=1e-12)

    sim.step({1: [SendOrder(1, 2, 80)]})
    assert len(sim.fleets) == 1
    fl = next(iter(sim.fleets.values()))
    assert np.isclose(sim.planets[1].ships, 0.3, atol=1e-12)
    assert np.isclose(fl.ships, 1.2, atol=1e-12)
    assert fl.num_units == 1
    assert np.isclose(fl.ships_per_unit, 1.2, atol=1e-12)

    # 3. Fractional ship packing evenly across physical ships: 10.6 @ 50% -> 5.3 across 5 units (1.06/unit)
    p1 = Planet(id=1, owner=1, ships=10.6, x=20.0, y=100.0, production=0.0, radius=12.0)
    p2 = Planet(id=2, owner=2, ships=50.0, x=220.0, y=100.0, production=0.0, radius=12.0)
    sim_frac = GalconSim([p1, p2], width=300.0, height=200.0)
    sim_frac.step({1: [SendOrder(1, 2, 50)]})
    assert len(sim_frac.fleets) == 1
    fl_frac = next(iter(sim_frac.fleets.values()))
    assert np.isclose(sim_frac.planets[1].ships, 5.3, atol=1e-12)
    assert np.isclose(fl_frac.ships, 5.3, atol=1e-12)
    assert fl_frac.num_units == 5
    assert np.isclose(fl_frac.ships_per_unit, 1.06, atol=1e-12)
    assert np.isclose(fl_frac.num_units * fl_frac.ships_per_unit, 5.3, atol=1e-12)

    # 4. Large send sqrt(17 * N) cap: 100 ships -> floor(sqrt(1700)) == 41 units, 100/41 each
    p1 = Planet(id=1, owner=1, ships=100.0, x=20.0, y=100.0, production=0.0, radius=24.0)
    p2 = Planet(id=2, owner=2, ships=100.0, x=320.0, y=100.0, production=0.0, radius=24.0)
    sim_lg = GalconSim([p1, p2], width=400.0, height=200.0)
    sim_lg.step({1: [SendOrder(1, 2, 100)]})
    fl_lg = next(iter(sim_lg.fleets.values()))
    assert fl_lg.num_units == int(math.sqrt(17.0 * 100.0)) == 41
    assert np.isclose(fl_lg.ships, 100.0, atol=1e-12)
    assert np.isclose(fl_lg.ships_per_unit, 100.0 / 41.0, atol=1e-12)
    assert sim_lg.item_count() == BASE_ITEM_USERS + 2 + 1 + 41

    # 5a. Table saturation & total_planet_ships scaling: item_count=3500 -> avail=290 -> int(avail * N / T)
    assert physical_ship_count(100.0, 2000.0, 3500) == int(290 * 100.0 / 2000.0) == 14
    # Build a sim with exact item_count == 3500 and total_planet_ships == 2000.0
    # item_count = BASE_ITEM_USERS(3) + planets(2) + fleets(1) + num_units(3494) = 3500
    filler_3500 = Fleet(id=900, owner=2, ships=10.0, x=20.0, y=20.0, source=2, target=1, num_units=3494)
    sim_3500 = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=1900.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [filler_3500],
        width=480.0,
        height=320.0,
    )
    assert sim_3500.item_count() == 3500
    sim_3500.step({1: [SendOrder(1, 2, 100)]})
    fl_3500 = sim_3500.fleets[1000]
    assert fl_3500.num_units == 14
    assert np.isclose(fl_3500.ships, 100.0, atol=1e-12)
    assert np.isclose(fl_3500.ships_per_unit, 100.0 / 14.0, atol=1e-12)

    # 5b. T includes source planet's ships BEFORE deduction (single planet sending 100% -> N / T = 1.0)
    # item_count = 3 + 2 + 1 + 4034 = 4040 -> avail = (4096 - 4040 - 16) // 2 = 20
    filler_4040 = Fleet(id=901, owner=2, ships=10.0, x=20.0, y=20.0, source=2, target=1, num_units=4034)
    sim_single = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=0.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [filler_4040],
        width=480.0,
        height=320.0,
    )
    assert sim_single.item_count() == 4040
    sim_single.step({1: [SendOrder(1, 2, 100)]})
    assert sim_single.fleets[1000].num_units == 20  # avail * (100 / 100) == 20

    # Also distinguish before-deduction vs after-deduction when another planet holds ships:
    # item_count = 3980 -> avail = 50; p1=100, p2=100 -> T_before = 200 -> parts = int(50 * 100 / 200) = 25
    # (post-deduction T would be 100, which would erroneously give 41)
    filler_3980 = Fleet(id=902, owner=2, ships=1.0, x=20.0, y=20.0, source=2, target=1, num_units=3974)
    sim_before = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=100.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [filler_3980],
        width=480.0,
        height=320.0,
    )
    assert sim_before.item_count() == 3980
    sim_before.step({1: [SendOrder(1, 2, 100)]})
    assert sim_before.fleets[1000].num_units == 25

    # 5c. Flying fleets do NOT inflate T (only their item_count slots matter)
    filler_huge_ships = Fleet(id=902, owner=2, ships=100000.0, x=20.0, y=20.0, source=2, target=1, num_units=3974)
    sim_huge_fl = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=100.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [filler_huge_ships],
        width=480.0,
        height=320.0,
    )
    assert sim_huge_fl.item_count() == 3980
    sim_huge_fl.step({1: [SendOrder(1, 2, 100)]})
    assert sim_huge_fl.fleets[1000].num_units == 25  # identical to sim_before despite 100,000 flying ships

    # Adding an extra 100,000-ship flying fleet only adds its 2 item_count slots (3982 -> avail=49 -> parts=24)
    sim_extra_fl = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=100.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [
            filler_3980,
            Fleet(id=903, owner=1, ships=100000.0, x=20.0, y=20.0, source=1, target=2, num_units=1),
        ],
        width=480.0,
        height=320.0,
    )
    assert sim_extra_fl.item_count() == 3982
    sim_extra_fl.step({1: [SendOrder(1, 2, 100)]})
    assert sim_extra_fl.fleets[1000].num_units == int(49 * 100.0 / 200.0) == 24

    # 5d. Soft cap: item_count >= 4080 -> avail = 0 -> parts == 1 (send is NOT refused)
    assert physical_ship_count(100.0, 100.0, 4080) == 1
    assert physical_ship_count(100.0, 100.0, 5000) == 1
    filler_full = Fleet(id=904, owner=2, ships=10.0, x=20.0, y=20.0, source=2, target=1, num_units=4074)
    sim_full = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=100.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [filler_full],
        width=480.0,
        height=320.0,
    )
    assert sim_full.item_count() == 4080
    sim_full.step({1: [SendOrder(1, 2, 100)]})
    assert 1000 in sim_full.fleets
    fl_full = sim_full.fleets[1000]
    assert np.isclose(fl_full.ships, 100.0, atol=1e-12)
    assert fl_full.num_units == 1
    assert np.isclose(fl_full.ships_per_unit, 100.0, atol=1e-12)

    # 5e. Multi-send on the same tick sees updated total_planet_ships and item_count between orders
    sim_multi = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=20.0, y=180.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=100.0, x=420.0, y=180.0, production=0.0, radius=24.0),
        ],
        [filler_3980],
        width=480.0,
        height=320.0,
    )
    assert sim_multi.item_count() == 3980
    sim_multi.step({1: [SendOrder(1, 2, 50), SendOrder(1, 2, 100)]})
    # Order 1: N=50, T=200, item_count=3980 (avail=50) -> int(50 * 50 / 200) = 12
    # Order 2: N=50, T=150, item_count=3980+1+12=3993 (avail=43) -> int(43 * 50 / 150) = 14
    assert sim_multi.fleets[1000].num_units == 12
    assert sim_multi.fleets[1001].num_units == 14
    assert np.isclose(sim_multi.planets[1].ships, 0.0, atol=1e-12)

    # 6. clone() preserves num_units and ships_per_unit
    cloned = sim_frac.clone()
    assert 1000 in cloned.fleets
    cfl = cloned.fleets[1000]
    assert cfl.num_units == 5
    assert np.isclose(cfl.ships_per_unit, 1.06, atol=1e-12)
    assert cloned.item_count() == sim_frac.item_count()
    # Mutating clone does not affect original
    cfl.num_units = 99
    assert sim_frac.fleets[1000].num_units == 5


def test_galcon_sim_physical_ships_separation_pushout_gradual_landing_and_redir():
    """Verify physical ship circle colliders: 12u disc spawn + rapid launch-tick settling, planet pushout, gradual landing, redir, timing, and clone isolation."""
    import math
    import galcon
    from galcon.sim import (
        ANTIPODE_DOT_THRESHOLD,
        ANTIPODE_ESCAPE_TANGENT,
        DEFAULT_PHYSICS_SUBSTEPS,
        SHIP_RADIUS,
        SHIP_SEP_MAX_DISP,
        SHIP_SEP_STIFFNESS,
        SPAWN_AREA_RADIUS,
        SPAWN_SETTLE_ITERS,
        SPAWN_SETTLE_TOL,
        Fleet,
        GalconSim,
        Planet,
        RedirOrder,
        SendOrder,
    )

    # Module constants & package re-exports match tuned 20 Hz C++ kernel spec
    assert DEFAULT_PHYSICS_SUBSTEPS == 5
    assert SHIP_SEP_STIFFNESS == 1.0
    assert SHIP_SEP_MAX_DISP == 6.0
    assert SPAWN_AREA_RADIUS == 12.0
    assert SPAWN_SETTLE_ITERS == 16
    assert SPAWN_SETTLE_TOL == 0.05
    assert ANTIPODE_DOT_THRESHOLD == -0.995
    assert ANTIPODE_ESCAPE_TANGENT == 0.05
    assert galcon.SHIP_SEP_MAX_DISP == SHIP_SEP_MAX_DISP
    assert galcon.SPAWN_AREA_RADIUS == SPAWN_AREA_RADIUS
    assert galcon.SPAWN_SETTLE_ITERS == SPAWN_SETTLE_ITERS
    assert galcon.SPAWN_SETTLE_TOL == SPAWN_SETTLE_TOL

    def _max_pair_overlap(units: list[list[float]]) -> float:
        min_sep = 2.0 * SHIP_RADIUS
        mov = 0.0
        for a in range(len(units)):
            ax, ay = units[a]
            for b in range(a + 1, len(units)):
                d = math.hypot(units[b][0] - ax, units[b][1] - ay)
                if d < min_sep:
                    mov = max(mov, min_sep - d)
        return mov

    # 1a. Deterministic 12u disc + launch-tick settling across small/medium flocks in open space
    p1 = Planet(id=1, owner=1, ships=100.0, x=50.0, y=120.0, production=0.0, radius=24.0)
    p2 = Planet(id=2, owner=2, ships=500.0, x=430.0, y=120.0, production=0.0, radius=24.0)

    sx = p1.x + p1.radius + SHIP_RADIUS
    sy = p1.y
    for ships_n, exp_parts in ((1.0, 1), (5.0, 5), (10.0, 10), (17.0, 17)):
        sim_sz = GalconSim(
            [
                Planet(id=1, owner=1, ships=ships_n, x=p1.x, y=p1.y, production=0.0, radius=p1.radius),
                p2,
            ],
            width=480.0,
            height=240.0,
            use_cpp=False,
        )
        if exp_parts == 1:
            sim_sz.step({1: [SendOrder(1, 2, 100)]})
            fl_s = next(iter(sim_sz.fleets.values()))
            assert fl_s.num_units == 1
            assert np.isclose(fl_s.x, sx + 10.0, atol=1e-9)
            assert np.isclose(fl_s.y, sy, atol=1e-9)
            assert fl_s.unit_headings == [[1.0, 0.0]]
        else:
            spawn_u, spawn_h, _, _, _ = sim_sz._spawn_and_settle_flock(1000, exp_parts, sx, sy, p2, 1.0, 0.0)
            assert len(spawn_u) == len(spawn_h) == exp_parts
            # Immediately on spawn: <= 0.5u overlap for parts=5, 10, 17 and zero penetration into src
            assert _max_pair_overlap(spawn_u) <= 0.5
            for u, h in zip(spawn_u, spawn_h):
                assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9
                ddx, ddy = p2.x - u[0], p2.y - u[1]
                dd = math.hypot(ddx, ddy)
                assert np.isclose(h[0] * (ddx / dd) + h[1] * (ddy / dd), 1.0, atol=1e-12)
            # After tick 1 (5 substeps): still <= 0.6u overlap and zero planet penetration
            sim_sz.step({1: [SendOrder(1, 2, 100)]})
            fl_t1 = next(iter(sim_sz.fleets.values()))
            assert fl_t1.units is not None and fl_t1.num_units == exp_parts
            assert _max_pair_overlap(fl_t1.units) <= 0.6
            for u in fl_t1.units:
                assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9

    # Also verify multi-planet non-target pushout at launch when a nearby obstacle planet borders the spawn zone
    p_near = Planet(id=3, owner=0, ships=30.0, x=95.0, y=170.0, production=0.0, radius=18.0)
    sim_near = GalconSim([p1, p2, p_near], width=480.0, height=240.0, use_cpp=False)
    near_u, _, _, _, _ = sim_near._spawn_and_settle_flock(1000, 41, sx, sy, p2, 1.0, 0.0)
    for u in near_u:
        assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9
        assert math.hypot(u[0] - p_near.x, u[1] - p_near.y) >= p_near.radius + SHIP_RADIUS - 1e-9
    sim_near.step({1: [SendOrder(1, 2, 100)]})
    fl_near = next(iter(sim_near.fleets.values()))
    assert fl_near.units is not None and fl_near.num_units == 41
    for u in fl_near.units:
        assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9
        assert math.hypot(u[0] - p_near.x, u[1] - p_near.y) >= p_near.radius + SHIP_RADIUS - 1e-9

    # 1b. Large 41-ship launch settles almost completely right on spawn tick with zero overshoot/rebound
    sim_sep = GalconSim([p1, p2], width=480.0, height=240.0, tick_dt=0.05, physics_substeps=1)
    spawn41_u, spawn41_h, spawn41_cx, _, spawn41_r = sim_sep._spawn_and_settle_flock(1000, 41, sx, sy, p2, 1.0, 0.0)
    assert len(spawn41_u) == len(spawn41_h) == 41
    assert 40.0 < spawn41_r < 70.0
    for u, h in zip(spawn41_u, spawn41_h):
        assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9
        ddx, ddy = p2.x - u[0], p2.y - u[1]
        dd = math.hypot(ddx, ddy)
        assert np.isclose(h[0] * (ddx / dd) + h[1] * (ddy / dd), 1.0, atol=1e-12)

    sim_sep.step({1: [SendOrder(1, 2, 100)]})  # substep 1 (0.05s)
    assert len(sim_sep.fleets) == 1
    fl = next(iter(sim_sep.fleets.values()))
    assert fl.units is not None
    assert len(fl.units) == fl.num_units == 41
    assert np.isclose(fl.ships, 100.0, atol=1e-12)
    assert sim_sep.item_count() == 3 + 2 + 1 + 41
    assert fl.x > spawn41_cx + 1.0
    for u in fl.units:
        assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9

    radii = [spawn41_r, fl.radius]
    overlaps = [_max_pair_overlap(spawn41_u), _max_pair_overlap(fl.units)]
    prev_units = [u[:] for u in fl.units]
    for _ in range(24):
        sim_sep.step({})
        fl = next(iter(sim_sep.fleets.values()))
        assert fl.units is not None and len(fl.units) == 41
        # Per-substep displacement bounded by forward move (2.0u) + SHIP_SEP_MAX_DISP (6.0u) + pushout margin
        max_disp = max(
            math.hypot(u[0] - pu[0], u[1] - pu[1]) for u, pu in zip(fl.units, prev_units)
        )
        assert max_disp <= 2.0 + SHIP_SEP_MAX_DISP + 1e-6
        for u in fl.units:
            assert math.hypot(u[0] - p1.x, u[1] - p1.y) >= p1.radius + SHIP_RADIUS - 1e-9
        radii.append(fl.radius)
        overlaps.append(_max_pair_overlap(fl.units))
        prev_units = [u[:] for u in fl.units]
    # By tick 1 (substep 5 = 0.25s): almost completely settled (< 2.5u max overlap); by tick 2 (substep 10): < 1.0u
    assert overlaps[5] < 2.5, f"Expected 41-ship tick-1 max_overlap < 2.5u, got {overlaps[5]:.4f}"
    assert overlaps[10] < 1.0, f"Expected 41-ship tick-2 max_overlap < 1.0u, got {overlaps[10]:.4f}"
    # Zero overshoot / launch rebound: radius stays tightly near its settled launch radius
    assert max(radii) - spawn41_r < 3.0, f"Unexpected launch overshoot: spawn_r={spawn41_r:.2f}, max_r={max(radii):.2f}"
    assert spawn41_r - min(radii) < 5.0, f"Unexpected launch rebound: spawn_r={spawn41_r:.2f}, min_r={min(radii):.2f}"

    # 2a. Cross-fleet ships sharing the SAME target DO separate (both same-owner AND opposing-owner)
    pa = Planet(id=1, owner=1, ships=50.0, x=20.0, y=100.0, production=0.0, radius=12.0)
    pb = Planet(id=2, owner=2, ships=500.0, x=380.0, y=100.0, production=0.0, radius=12.0)
    pc = Planet(id=3, owner=0, ships=500.0, x=380.0, y=150.0, production=0.0, radius=12.0)
    f_p1a = Fleet(id=10, owner=1, ships=5.0, x=100.0, y=100.0, source=1, target=2)
    f_p1b = Fleet(id=11, owner=1, ships=5.0, x=100.0, y=100.0, source=1, target=2)
    f_p2 = Fleet(id=12, owner=2, ships=5.0, x=100.0, y=100.0, source=1, target=2)
    sim_same_tgt = GalconSim([pa, pb, pc], [f_p1a, f_p1b, f_p2], width=400.0, height=200.0)
    for _ in range(5):
        sim_same_tgt.step({})
    u10 = sim_same_tgt.fleets[10].units[0]
    u11 = sim_same_tgt.fleets[11].units[0]
    u12 = sim_same_tgt.fleets[12].units[0]
    for ua, ub in ((u10, u11), (u10, u12), (u11, u12)):
        d_pair = math.sqrt((ua[0] - ub[0]) ** 2 + (ua[1] - ub[1]) ** 2)
        assert d_pair >= 2.0 * SHIP_RADIUS - 0.5, f"Same-target cross-fleet ships still overlapped: d={d_pair:.3f}"

    # 2b. Ships with DIFFERENT targets do NOT push each other apart
    f_diff_a = Fleet(id=20, owner=1, ships=5.0, x=100.0, y=100.0, source=1, target=2)
    f_diff_b = Fleet(id=21, owner=1, ships=5.0, x=100.0, y=100.0, source=1, target=3)
    sim_diff_tgt = GalconSim([pa, pb, pc], [f_diff_a, f_diff_b], width=400.0, height=200.0)
    sim_diff_ref = GalconSim([pa, pb, pc], [Fleet(id=20, owner=1, ships=5.0, x=100.0, y=100.0, source=1, target=2)], width=400.0, height=200.0)
    for _ in range(5):
        sim_diff_tgt.step({})
        sim_diff_ref.step({})
    assert sim_diff_tgt.fleets[20].units[0] == sim_diff_ref.fleets[20].units[0]
    assert np.isclose(sim_diff_tgt.fleets[20].units[0][0], 150.0, atol=1e-9)
    assert np.isclose(sim_diff_tgt.fleets[20].units[0][1], 100.0, atol=1e-9)

    # 3. Obstacle planet directly between src and dst pushes ships radially outward,
    #    slows arrival vs unobstructed flight, and 1-ship collinear fleet escapes cleanly
    def _run_corridor(ships: float, with_obstacle: bool):
        pls = [
            Planet(id=1, owner=1, ships=ships, x=50.0, y=120.0, production=0.0, radius=20.0),
            Planet(id=2, owner=2, ships=500.0, x=350.0, y=120.0, production=0.0, radius=20.0),
        ]
        if with_obstacle:
            pls.append(Planet(id=3, owner=0, ships=50.0, x=200.0, y=120.0, production=0.0, radius=24.0))
        sim_c = GalconSim(pls, width=400.0, height=240.0, tick_dt=0.05, physics_substeps=1, max_ticks=600)
        sim_c.step({1: [SendOrder(1, 2, 100)]})
        min_obs_d = float("inf")
        max_lat = 0.0
        sub = 1
        while sim_c.fleets and sub < 500:
            fl_c = next(iter(sim_c.fleets.values()))
            assert fl_c.units is not None
            for u in fl_c.units:
                if with_obstacle:
                    d_obs = math.sqrt((u[0] - 200.0) ** 2 + (u[1] - 120.0) ** 2)
                    min_obs_d = min(min_obs_d, d_obs)
                max_lat = max(max_lat, abs(u[1] - 120.0))
            sim_c.step({})
            sub += 1
        return sub, len(sim_c.fleets), min_obs_d, max_lat

    # 3a. 1-ship collinear antipode escape
    t_clear_1, rem_c1, _, _ = _run_corridor(1.0, with_obstacle=False)
    t_obs_1, rem_o1, min_obs_1, max_lat_1 = _run_corridor(1.0, with_obstacle=True)
    assert rem_c1 == 0 and rem_o1 == 0  # escaped cleanly without stalling
    assert min_obs_1 >= 24.0 + SHIP_RADIUS - 1e-9
    assert max_lat_1 > 24.0  # detoured around obstacle
    assert t_obs_1 > t_clear_1  # slowed arrival vs unobstructed

    # 3b. Multi-ship fleet around blocking planet
    t_clear_25, rem_c25, _, _ = _run_corridor(25.0, with_obstacle=False)
    t_obs_25, rem_o25, min_obs_25, max_lat_25 = _run_corridor(25.0, with_obstacle=True)
    assert rem_c25 == 0 and rem_o25 == 0
    assert min_obs_25 >= 24.0 + SHIP_RADIUS - 1e-9
    assert max_lat_25 > 24.0
    assert t_obs_25 > t_clear_25

    # 4. Gradual landing: 100 ships (41 units) lands across multiple ticks with smooth count/ship decay
    p_src = Planet(id=1, owner=1, ships=100.0, x=40.0, y=120.0, production=0.0, radius=24.0)
    p_dst = Planet(id=2, owner=2, ships=200.0, x=360.0, y=120.0, production=0.0, radius=24.0)
    sim_grad = GalconSim([p_src, p_dst], width=420.0, height=240.0)
    sim_grad.step({1: [SendOrder(1, 2, 100)]})
    spu = 100.0 / 41.0
    partial_counts: list[int] = []
    prev_units_cnt = 41
    for _ in range(60):
        if not sim_grad.fleets:
            break
        fl_g = next(iter(sim_grad.fleets.values()))
        assert fl_g.units is not None
        assert fl_g.num_units == len(fl_g.units)
        assert fl_g.num_units <= prev_units_cnt
        assert np.isclose(fl_g.ships, fl_g.num_units * spu, atol=1e-9)
        assert np.isclose(fl_g.ships + (200.0 - sim_grad.planets[2].ships), 100.0, atol=1e-9)
        if 0 < fl_g.num_units < 41:
            partial_counts.append(fl_g.num_units)
        prev_units_cnt = fl_g.num_units
        sim_grad.step({})
    assert len(sim_grad.fleets) == 0
    assert np.isclose(sim_grad.planets[2].ships, 100.0, atol=1e-9)
    assert len(partial_counts) >= 2  # spans >= 3 ticks (multiple intermediate partial-fleet ticks)
    assert partial_counts == sorted(partial_counts, reverse=True)

    # 5. RedirOrder redirects all physical ship units in the fleet together to the new target
    r_p1 = Planet(id=1, owner=1, ships=50.0, x=40.0, y=160.0, production=0.0, radius=20.0)
    r_p2 = Planet(id=2, owner=2, ships=200.0, x=360.0, y=160.0, production=0.0, radius=20.0)
    r_p3 = Planet(id=3, owner=2, ships=200.0, x=200.0, y=30.0, production=0.0, radius=20.0)
    sim_redir = GalconSim([r_p1, r_p2, r_p3], width=400.0, height=240.0)
    sim_redir.step({1: [SendOrder(1, 2, 100)]})
    fid_r = next(iter(sim_redir.fleets.keys()))
    for _ in range(5):
        sim_redir.step({})
    fl_before = sim_redir.fleets[fid_r]
    assert fl_before.units is not None and fl_before.num_units == 29
    y_before = [u[1] for u in fl_before.units]
    cy_before = fl_before.y

    sim_redir.step({1: [RedirOrder(fid_r, 3)]})
    fl_after = sim_redir.fleets[fid_r]
    assert fl_after.target == 3
    assert fl_after.y < cy_before - 5.0
    assert all(u[1] < yb for u, yb in zip(fl_after.units, y_before))

    for _ in range(40):
        if not sim_redir.fleets:
            break
        sim_redir.step({})
    assert len(sim_redir.fleets) == 0
    assert np.isclose(sim_redir.planets[2].ships, 200.0, atol=1e-9)  # old target untouched
    assert np.isclose(sim_redir.planets[3].ships, 150.0, atol=1e-9)  # all 50 ships landed on new target

    # 6. Exact 1-ship touch-to-land timing (dist <= dst.r + SHIP_RADIUS)
    t_src = Planet(id=1, owner=1, ships=1.0, x=0.0, y=100.0, production=0.0, radius=12.0)
    # Exact 3-tick travel distance (3 * 10.0 = 30.0u) between spawn edge and touch-to-land edge
    t_dst_exact = Planet(
        id=2, owner=2, ships=10.0, x=12.0 + 12.0 + 2.0 * SHIP_RADIUS + 30.0, y=100.0, production=0.0, radius=12.0
    )
    sim_t3 = GalconSim([t_src, t_dst_exact], width=200.0, height=200.0, fleet_speed=40.0, tick_dt=0.25)
    assert sim_t3.planet_eta_ticks(t_src, t_dst_exact) == 3
    sim_t3.step({1: [SendOrder(1, 2, 100)]})  # tick 1: x = 18 + 10 = 28
    assert len(sim_t3.fleets) == 1
    sim_t3.step({})  # tick 2: x = 38
    assert len(sim_t3.fleets) == 1
    sim_t3.step({})  # tick 3: x = 48, dist = 18.0 == dst.r + SHIP_RADIUS -> lands!
    assert len(sim_t3.fleets) == 0
    assert np.isclose(sim_t3.planets[2].ships, 9.0, atol=1e-9)

    t_dst_4 = Planet(
        id=2, owner=2, ships=10.0, x=12.0 + 12.0 + 2.0 * SHIP_RADIUS + 30.0 + 1e-4, y=100.0, production=0.0, radius=12.0
    )
    sim_t4 = GalconSim([t_src, t_dst_4], width=200.0, height=200.0, fleet_speed=40.0, tick_dt=0.25)
    assert sim_t4.planet_eta_ticks(t_src, t_dst_4) == 4
    sim_t4.step({1: [SendOrder(1, 2, 100)]})
    sim_t4.step({})
    sim_t4.step({})
    assert len(sim_t4.fleets) == 1  # still 1e-4 short of touch-to-land after tick 3
    sim_t4.step({})
    assert len(sim_t4.fleets) == 0
    assert np.isclose(sim_t4.planets[2].ships, 9.0, atol=1e-9)

    # 7. clone() deep-copies fl.units completely and preserves physics_substeps
    sim_orig = GalconSim(
        [
            Planet(id=1, owner=1, ships=40.0, x=40.0, y=100.0, production=0.0, radius=20.0),
            Planet(id=2, owner=2, ships=100.0, x=360.0, y=100.0, production=0.0, radius=20.0),
        ],
        width=400.0,
        height=200.0,
        physics_substeps=3,
    )
    sim_orig.step({1: [SendOrder(1, 2, 100)]})
    fid_o = next(iter(sim_orig.fleets.keys()))
    sim_copy = sim_orig.clone()
    assert sim_copy.physics_substeps == 3
    orig_units_snap = [u[:] for u in sim_orig.fleets[fid_o].units]
    assert sim_copy.fleets[fid_o].units == orig_units_snap
    assert sim_copy.fleets[fid_o].units is not sim_orig.fleets[fid_o].units
    assert sim_copy.fleets[fid_o].units[0] is not sim_orig.fleets[fid_o].units[0]

    sim_copy.fleets[fid_o].units[0][0] += 999.0
    sim_copy.fleets[fid_o].units.pop()
    sim_copy.step({})
    assert sim_orig.fleets[fid_o].units == orig_units_snap
    assert sim_orig.fleets[fid_o].num_units == len(orig_units_snap)


def test_galcon_sim_cpp_bit_exact_parity_and_fast_rollout():
    """Verify C++ drop-in physics, FastGalconCppSim parity (<= 1e-11), clone/copy_from/step_n_ticks, and >4096-unit fallback."""
    import random
    import galcon
    from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim
    from galcon.sim import Fleet, GalconSim, Planet, RedirOrder, SendOrder, generate_galcon_map

    assert isinstance(HAS_FAST_SIM_CPP, bool)
    assert galcon.HAS_FAST_SIM_CPP is HAS_FAST_SIM_CPP
    assert galcon.FastGalconCppSim is FastGalconCppSim

    def _compare_sims(ref: GalconSim, cand: GalconSim) -> float:
        assert ref.tick == cand.tick
        assert ref.done == cand.done
        assert ref.winner() == cand.winner()
        assert ref.next_fleet_id == cand.next_fleet_id
        assert sorted(ref.planets.keys()) == sorted(cand.planets.keys())

        max_diff = 0.0
        for pid in sorted(ref.planets.keys()):
            p1 = ref.planets[pid]
            p2 = cand.planets[pid]
            assert p1.owner == p2.owner, f"planet {pid} owner mismatch: {p1.owner} vs {p2.owner}"
            max_diff = max(max_diff, abs(p1.ships - p2.ships))

        assert list(ref.fleets.keys()) == list(cand.fleets.keys())
        for fid, f1 in ref.fleets.items():
            f2 = cand.fleets[fid]
            assert f1.owner == f2.owner
            assert f1.source == f2.source
            assert f1.target == f2.target
            assert f1.num_units == f2.num_units
            assert f1.units is not None and f2.units is not None
            assert len(f1.units) == len(f2.units) == f1.num_units
            assert f1.unit_headings is not None and f2.unit_headings is not None
            assert len(f1.unit_headings) == len(f2.unit_headings) == f1.num_units
            max_diff = max(
                max_diff,
                abs(f1.ships - f2.ships),
                abs(f1.x - f2.x),
                abs(f1.y - f2.y),
                abs(f1.radius - f2.radius),
            )
            for u1, u2 in zip(f1.units, f2.units):
                max_diff = max(max_diff, abs(u1[0] - u2[0]), abs(u1[1] - u2[1]))
            for h1, h2 in zip(f1.unit_headings, f2.unit_headings):
                max_diff = max(max_diff, abs(h1[0] - h2[0]), abs(h1[1] - h2[1]))
        return max_diff

    # 1. Both use_cpp=False and use_cpp=True (and default None) work cleanly on all platforms
    s_py = GalconSim.create(seed=101, num_planets=12, width=400.0, height=240.0, use_cpp=False)
    s_cpp = GalconSim.create(seed=101, num_planets=12, width=400.0, height=240.0, use_cpp=True)
    s_auto = GalconSim.create(seed=101, num_planets=12, width=400.0, height=240.0)
    assert s_py.use_cpp is False
    assert s_cpp.use_cpp is HAS_FAST_SIM_CPP
    assert s_auto.use_cpp is HAS_FAST_SIM_CPP
    assert s_py.clone().use_cpp is False
    assert s_py.clone(use_cpp=True).use_cpp is HAS_FAST_SIM_CPP
    assert s_cpp.clone(use_cpp=False).use_cpp is False

    for _ in range(10):
        ords = {1: [SendOrder(1, 3, 25)], 2: [SendOrder(2, 4, 25)]} if s_py.tick == 0 else {}
        s_py.step(ords)
        s_cpp.step(ords)
    assert _compare_sims(s_py, s_cpp) <= 1e-11

    if not HAS_FAST_SIM_CPP:
        return

    # 1b. Bit-exact 0.000e+00 parity on SendOrder 12u disc spawn + launch-tick settling (parts = 1, 5, 10, 17, 41)
    for ships_n, exp_parts in ((1.0, 1), (5.0, 5), (10.0, 10), (17.0, 17), (100.0, 41)):
        sp_planets = [
            Planet(id=1, owner=1, ships=ships_n, x=50.0, y=120.0, production=0.0, radius=24.0),
            Planet(id=2, owner=2, ships=500.0, x=430.0, y=120.0, production=0.0, radius=24.0),
            Planet(id=3, owner=0, ships=30.0, x=95.0, y=170.0, production=0.0, radius=18.0),
        ]
        sp_py = GalconSim(sp_planets, width=480.0, height=240.0, use_cpp=False)
        sp_cpp = GalconSim(sp_planets, width=480.0, height=240.0, use_cpp=True)
        sp_fast = FastGalconCppSim.from_sim(sp_py)
        sp_py.step({1: [SendOrder(1, 2, 100)]})
        sp_cpp.step({1: [SendOrder(1, 2, 100)]})
        sp_fast.step({1: [SendOrder(1, 2, 100)]})
        assert next(iter(sp_py.fleets.values())).num_units == exp_parts
        assert _compare_sims(sp_py, sp_cpp) == 0.0
        assert _compare_sims(sp_py, sp_fast.to_sim()) == 0.0

    # 2. 150-tick 22-planet randomized multi-fleet battle across pure Python, drop-in C++, and FastGalconCppSim
    planets = generate_galcon_map(seed=1337, num_planets=22, width=480.0, height=320.0, symmetric=True)
    assert len(planets) == 22
    # Reinforce homes & forward outposts so the multi-fleet battle runs all 150 ticks with heavy collisions & landings
    planets[0].ships = 350.0
    planets[1].ships = 350.0
    planets[2].owner, planets[2].ships = 1, 80.0
    planets[3].owner, planets[3].ships = 2, 80.0

    sim_py = GalconSim(planets, width=480.0, height=320.0, max_ticks=240, use_cpp=False)
    sim_cpp = GalconSim(planets, width=480.0, height=320.0, max_ticks=240, use_cpp=True)
    fast_cpp = FastGalconCppSim.from_sim(sim_py)
    assert sim_cpp.use_cpp is True

    rng = random.Random(2026)
    planet_ids = sorted(sim_py.planets.keys())
    max_diff_dropin = 0.0
    max_diff_native = 0.0
    midgame_snapshot_py: GalconSim | None = None
    peak_units = 0

    for t in range(150):
        assert not sim_py.done
        orders: dict[int, list] = {1: [], 2: []}
        for pid in (1, 2):
            owned_all = [sim_py.planets[i] for i in planet_ids if sim_py.planets[i].owner == pid]
            opp = 2 if pid == 1 else 1
            opp_ids = {i for i in planet_ids if sim_py.planets[i].owner == opp}
            sendable = [p for p in owned_all if p.ships >= 1.5]
            if sendable and (t < 12 or rng.random() < 0.70):
                n_sends = rng.randint(1, min(3, len(sendable)))
                for src in rng.sample(sendable, n_sends):
                    forbidden = {src.id}
                    if len(opp_ids) <= 1:
                        forbidden.update(opp_ids)
                    cands = [i for i in planet_ids if i not in forbidden]
                    dst_id = rng.choice(cands)
                    pct = rng.choice([20, 35, 50]) if len(owned_all) <= 2 else rng.choice([15, 33, 50, 75, 100])
                    orders[pid].append(SendOrder(src.id, dst_id, pct))
                    # Exercise same-tick multi-send from the same source planet (fractional packing + item_count update)
                    if src.ships >= 25.0 and rng.random() < 0.30:
                        dst2 = rng.choice(cands)
                        orders[pid].append(SendOrder(src.id, dst2, 25))
            # Occasionally submit a sub-1-ship edge-case send to verify guard parity
            tiny = [p for p in owned_all if 0.0 < p.ships < 1.5]
            if tiny:
                src_t = tiny[0]
                dst_t = next(i for i in planet_ids if i != src_t.id)
                orders[pid].append(SendOrder(src_t.id, dst_t, 50))
            # Mid-flight RedirOrder
            my_fleets = [fl for fl in sim_py.fleets.values() if fl.owner == pid]
            if my_fleets and rng.random() < 0.25:
                fl = rng.choice(my_fleets)
                forbidden = {fl.target}
                if len(opp_ids) <= 1:
                    forbidden.update(opp_ids)
                cands = [i for i in planet_ids if i not in forbidden]
                if cands:
                    orders[pid].append(RedirOrder(fl.id, rng.choice(cands)))

        sim_py.step(orders)
        sim_cpp.step(orders)
        fast_cpp.step(orders)

        assert fast_cpp.num_planets == len(sim_py.planets)
        assert fast_cpp.num_fleets == len(sim_py.fleets)
        cur_units = sum(fl.num_units for fl in sim_py.fleets.values())
        assert fast_cpp.num_units == cur_units
        peak_units = max(peak_units, cur_units)

        max_diff_dropin = max(max_diff_dropin, _compare_sims(sim_py, sim_cpp))
        max_diff_native = max(max_diff_native, _compare_sims(sim_py, fast_cpp.to_sim()))

        if t == 40 and len(sim_py.fleets) >= 4:
            midgame_snapshot_py = sim_py.clone(use_cpp=False)

    assert peak_units >= 50
    assert max_diff_dropin == 0.0, f"drop-in C++ parity drifted from bit-exact 0.0: {max_diff_dropin:.3e}"
    assert max_diff_native == 0.0, f"FastGalconCppSim parity drifted from bit-exact 0.0: {max_diff_native:.3e}"

    # 3. Near-tie gradual combat parity in FastGalconCppSim vs pure Python
    step_dist = 40.0 * 0.25
    tie_planets = [
        Planet(id=1, owner=1, ships=50.0, x=0.0, y=50.0, production=100.0, radius=12.0),
        Planet(
            id=2,
            owner=2,
            ships=4.999999999999984,
            x=12.0 + 6.0 + step_dist + 7e-15,
            y=0.0,
            production=0.0,
            radius=12.0,
        ),
    ]
    tie_fleets = [Fleet(id=1000, owner=1, ships=5.0, x=0.0, y=0.0, source=1, target=2)]
    tie_py = GalconSim(tie_planets, tie_fleets, use_cpp=False)
    tie_cpp = GalconSim(tie_planets, tie_fleets, use_cpp=True)
    tie_fast = FastGalconCppSim.from_sim(tie_py)
    tie_py.step({})
    tie_cpp.step({})
    tie_fast.step({})
    assert _compare_sims(tie_py, tie_cpp) <= 1e-11
    assert _compare_sims(tie_py, tie_fast.to_sim()) <= 1e-11
    assert tie_py.planets[2].owner == 2 and tie_py.planets[2].ships == 0.0

    # 4. FastGalconCppSim.clone(), copy_from(), step_n_ticks(), and to_sim() round-trip parity
    root_py = midgame_snapshot_py if midgame_snapshot_py is not None else sim_py.clone(use_cpp=False)
    fc_root = FastGalconCppSim.from_sim(root_py)
    assert _compare_sims(root_py, fc_root.to_sim()) <= 1e-11

    fc_clone = fc_root.clone()
    fc_clone.step_n_ticks(15)
    rollout_py = root_py.clone(use_cpp=False)
    for _ in range(15):
        rollout_py.step({})
    assert _compare_sims(rollout_py, fc_clone.to_sim()) <= 1e-11
    # Cloning must leave root completely untouched
    assert _compare_sims(root_py, fc_root.to_sim()) <= 1e-11

    # Zero-alloc copy_from resets fc_clone identically to fc_root and reproduces the exact same rollout
    fc_clone.copy_from(fc_root)
    assert _compare_sims(root_py, fc_clone.to_sim()) <= 1e-11
    fc_clone.step_n_ticks(15)
    assert _compare_sims(rollout_py, fc_clone.to_sim()) <= 1e-11

    # 5. Large synthetic world (> 64 planets, > 512 fleets, > 4096 physical units) exercises dynamic fallback cleanly
    big_planets = [
        Planet(
            id=i + 1,
            owner=(1 if i == 0 else (2 if i == 1 else 0)),
            ships=50.0,
            x=30.0 + (i % 10) * 65.0,
            y=30.0 + (i // 10) * 65.0,
            production=50.0,
            radius=14.0,
        )
        for i in range(70)
    ]
    big_fleets = []
    # 100 fleets x 45 units = 4,500 units + 430 single-unit fleets = 530 fleets & 4,930 total units
    for fi in range(100):
        big_fleets.append(
            Fleet(
                id=1000 + fi,
                owner=1,
                ships=45.0,
                x=60.0 + (fi % 10) * 55.0,
                y=60.0 + (fi // 10) * 40.0,
                source=1,
                target=2,
                num_units=45,
            )
        )
    for fi in range(430):
        big_fleets.append(
            Fleet(
                id=2000 + fi,
                owner=2,
                ships=1.0,
                x=80.0 + (fi % 20) * 25.0,
                y=80.0 + (fi // 20) * 15.0,
                source=2,
                target=1,
                num_units=1,
            )
        )
    big_py = GalconSim(big_planets, big_fleets, width=800.0, height=600.0, physics_substeps=2, use_cpp=False)
    big_cpp = GalconSim(big_planets, big_fleets, width=800.0, height=600.0, physics_substeps=2, use_cpp=True)
    assert len(big_cpp.planets) == 70
    assert len(big_cpp.fleets) == 530
    assert sum(fl.num_units for fl in big_cpp.fleets.values()) == 4930
    for _ in range(3):
        big_py.step({})
        big_cpp.step({})
    assert _compare_sims(big_py, big_cpp) == 0.0


def test_galcon_sim_redirect_turning_radius_and_turn_rate():
    """Verify tight redirect turning radius (~1.6-1.9u), 360 deg / 0.50s angular turn rate, straight flight, and bit-exact C++ parity."""
    import math
    import galcon
    from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim
    from galcon.sim import (
        SHIP_RADIUS,
        SHIP_TURN_MIN_SPEED_FACTOR,
        SHIP_TURN_RATE,
        GalconSim,
        Planet,
        RedirOrder,
        SendOrder,
    )

    assert np.isclose(SHIP_TURN_RATE, 4.0 * math.pi, atol=1e-12)
    assert np.isclose(SHIP_TURN_MIN_SPEED_FACTOR, 0.5, atol=1e-12)
    assert galcon.SHIP_TURN_RATE == SHIP_TURN_RATE
    assert galcon.SHIP_TURN_MIN_SPEED_FACTOR == SHIP_TURN_MIN_SPEED_FACTOR

    # Theoretical sharp-turn radius at 40 u/s with 0.5 min speed factor: R = 20 / (4*pi) ~= 1.5915u
    theo_sharp_r = (40.0 * SHIP_TURN_MIN_SPEED_FACTOR) / SHIP_TURN_RATE
    assert 1.0 <= theo_sharp_r <= 2.0

    def _fleet_max_diff(f1: Fleet, f2: Fleet) -> float:
        assert f1.target == f2.target
        assert f1.num_units == f2.num_units
        assert f1.units is not None and f2.units is not None
        assert f1.unit_headings is not None and f2.unit_headings is not None
        assert len(f1.units) == len(f2.units) == f1.num_units
        assert len(f1.unit_headings) == len(f2.unit_headings) == f1.num_units
        d = max(abs(f1.ships - f2.ships), abs(f1.x - f2.x), abs(f1.y - f2.y), abs(f1.radius - f2.radius))
        for u1, u2 in zip(f1.units, f2.units):
            d = max(d, abs(u1[0] - u2[0]), abs(u1[1] - u2[1]))
        for h1, h2 in zip(f1.unit_headings, f2.unit_headings):
            d = max(d, abs(h1[0] - h2[0]), abs(h1[1] - h2[1]))
        return d

    # 1. 1-ship 180 deg boomerang redirect: 36 deg/substep -> 5 substeps (0.25s) for 180 deg (0.50s for 360 deg)
    p1 = Planet(id=1, owner=1, ships=1.0, x=50.0, y=120.0, production=0.0, radius=20.0)
    p2 = Planet(id=2, owner=2, ships=100.0, x=350.0, y=120.0, production=0.0, radius=20.0)

    sim_py = GalconSim([p1, p2], width=400.0, height=240.0, tick_dt=0.05, physics_substeps=1, use_cpp=False)
    sim_cpp = GalconSim([p1, p2], width=400.0, height=240.0, tick_dt=0.05, physics_substeps=1, use_cpp=True)
    fast_cpp = FastGalconCppSim.from_sim(sim_py) if HAS_FAST_SIM_CPP else None

    # Fly +x for 20 substeps (40u) from P1 toward P2
    sim_py.step({1: [SendOrder(1, 2, 100)]})
    sim_cpp.step({1: [SendOrder(1, 2, 100)]})
    if fast_cpp is not None:
        fast_cpp.step({1: [SendOrder(1, 2, 100)]})
    for _ in range(19):
        sim_py.step({})
        sim_cpp.step({})
        if fast_cpp is not None:
            fast_cpp.step({})

    fid = next(iter(sim_py.fleets.keys()))
    fl0 = sim_py.fleets[fid]
    assert fl0.units is not None and fl0.unit_headings is not None
    x0, y0 = fl0.units[0]
    assert np.isclose(fl0.unit_headings[0][0], 1.0, atol=1e-12)
    assert np.isclose(fl0.unit_headings[0][1], 0.0, atol=1e-12)

    # Verify clone() deep-copies unit_headings independently
    sim_cloned = sim_py.clone()
    assert sim_cloned.fleets[fid].unit_headings == fl0.unit_headings
    assert sim_cloned.fleets[fid].unit_headings is not fl0.unit_headings
    assert sim_cloned.fleets[fid].unit_headings[0] is not fl0.unit_headings[0]
    sim_cloned.fleets[fid].unit_headings[0][0] = -0.5
    assert np.isclose(sim_py.fleets[fid].unit_headings[0][0], 1.0, atol=1e-12)

    # Issue 180 deg boomerang redirect back toward P1
    sim_py.step({1: [RedirOrder(fid, 1)]})
    sim_cpp.step({1: [RedirOrder(fid, 1)]})
    if fast_cpp is not None:
        fast_cpp.step({1: [RedirOrder(fid, 1)]})

    max_x = x0
    max_y = y0
    prev_x, prev_y = x0, y0
    max_cpp_diff = 0.0
    max_fast_diff = 0.0

    for substep in range(1, 10):
        fl_p = sim_py.fleets[fid]
        assert fl_p.units is not None and fl_p.unit_headings is not None
        x, y = fl_p.units[0]
        hx, hy = fl_p.unit_headings[0]
        step_dist = math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y
        max_x = max(max_x, x)
        max_y = max(max_y, y)

        if substep <= 5:
            expected_ang = substep * (SHIP_TURN_RATE * 0.05)  # 36, 72, 108, 144, 180 deg
            assert np.isclose(hx, math.cos(expected_ang), atol=1e-12)
            assert np.isclose(hy, math.sin(expected_ang), atol=1e-12)
        if substep == 1:
            # Does NOT instantaneously reverse: still advancing +x with cos(36 deg) ~= 0.809
            assert hx > 0.80 and np.isclose(hx, math.cos(0.2 * math.pi), atol=1e-12)
            assert x > x0
            assert np.isclose(step_dist, 1.0, atol=1e-12)  # 0.5 * 2.0u/substep during sharp turn
        elif substep == 5:
            # Full 180 deg U-turn completed in exact 5 substeps (0.25s -> 360 deg in 0.50s)
            assert np.isclose(hx, -1.0, atol=1e-12)
            assert np.isclose(hy, 0.0, atol=1e-12)
        elif substep >= 6:
            # Resumes full 40.0 u/s (2.0u/substep) aligned speed back toward P1
            assert np.isclose(step_dist, 2.0, atol=1e-9)

        if HAS_FAST_SIM_CPP:
            max_cpp_diff = max(max_cpp_diff, _fleet_max_diff(fl_p, sim_cpp.fleets[fid]))
            assert fast_cpp is not None
            max_fast_diff = max(max_fast_diff, _fleet_max_diff(fl_p, fast_cpp.to_sim().fleets[fid]))

        sim_py.step({})
        sim_cpp.step({})
        if fast_cpp is not None:
            fast_cpp.step({})

    # Tight turning radius in [1.0, 2.5]:
    # - 180 deg semicircle lateral radius R = 0.5 * (max_y - y0) ~= 1.91u
    # - forward overshoot radius (max_x - x0) ~= 1.12u
    lateral_turn_radius = 0.5 * (max_y - y0)
    forward_overshoot = max_x - x0
    assert 1.0 <= lateral_turn_radius <= 2.5, f"Unexpected U-turn lateral radius: {lateral_turn_radius:.4f}"
    assert 1.0 <= forward_overshoot <= 2.5, f"Unexpected U-turn forward overshoot: {forward_overshoot:.4f}"

    # Continue until the boomerang ship lands cleanly back on P1
    for _ in range(100):
        if not sim_py.fleets:
            break
        sim_py.step({})
        sim_cpp.step({})
        if fast_cpp is not None:
            fast_cpp.step({})
    assert len(sim_py.fleets) == 0
    assert np.isclose(sim_py.planets[1].ships, 1.0, atol=1e-12)
    if HAS_FAST_SIM_CPP:
        assert max_cpp_diff == 0.0, f"Boomerang drop-in C++ parity diff: {max_cpp_diff:.3e}"
        assert max_fast_diff == 0.0, f"Boomerang FastGalconCppSim parity diff: {max_fast_diff:.3e}"
        assert len(sim_cpp.fleets) == 0
        assert fast_cpp is not None and fast_cpp.num_fleets == 0
        assert sim_cpp.planets[1].ships == sim_py.planets[1].ships
        assert fast_cpp.to_sim().planets[1].ships == sim_py.planets[1].ships

    # 2. 90 deg redirect on a multi-ship fleet preserves headings, curves smoothly without stalling, and lands cleanly
    mp1 = Planet(id=1, owner=1, ships=40.0, x=40.0, y=180.0, production=0.0, radius=20.0)
    mp2 = Planet(id=2, owner=2, ships=200.0, x=360.0, y=180.0, production=0.0, radius=20.0)
    mp3 = Planet(id=3, owner=2, ships=200.0, x=150.0, y=40.0, production=0.0, radius=20.0)

    ms_py = GalconSim([mp1, mp2, mp3], width=400.0, height=240.0, tick_dt=0.05, physics_substeps=1, use_cpp=False)
    ms_cpp = GalconSim([mp1, mp2, mp3], width=400.0, height=240.0, tick_dt=0.05, physics_substeps=1, use_cpp=True)
    ms_fast = FastGalconCppSim.from_sim(ms_py) if HAS_FAST_SIM_CPP else None

    ms_py.step({1: [SendOrder(1, 2, 100)]})
    ms_cpp.step({1: [SendOrder(1, 2, 100)]})
    if ms_fast is not None:
        ms_fast.step({1: [SendOrder(1, 2, 100)]})
    for _ in range(41):
        ms_py.step({})
        ms_cpp.step({})
        if ms_fast is not None:
            ms_fast.step({})

    mfid = next(iter(ms_py.fleets.keys()))
    mfl_pre = ms_py.fleets[mfid]
    assert mfl_pre.num_units == 26
    assert mfl_pre.unit_headings is not None and len(mfl_pre.unit_headings) == 26
    # Fleet is roughly below mp3 (x ~= 150, y ~= 180), heading +x
    assert all(h[0] > 0.95 for h in mfl_pre.unit_headings)

    # Issue 90 deg redirect straight upward toward mp3
    ms_py.step({1: [RedirOrder(mfid, 3)]})
    ms_cpp.step({1: [RedirOrder(mfid, 3)]})
    if ms_fast is not None:
        ms_fast.step({1: [RedirOrder(mfid, 3)]})

    # On substep 1 after 90 deg redirect, units have rotated ~36 deg toward -y (still have positive hx, negative hy)
    mfl_s1 = ms_py.fleets[mfid]
    assert mfl_s1.target == 3
    assert mfl_s1.unit_headings is not None
    assert all(h[0] > 0.5 and h[1] < -0.4 for h in mfl_s1.unit_headings)

    # Track smooth convergence and exact C++ parity through arrival on mp3
    max_m_cpp = 0.0
    max_m_fast = 0.0
    for _ in range(150):
        if HAS_FAST_SIM_CPP and mfid in ms_py.fleets:
            max_m_cpp = max(max_m_cpp, _fleet_max_diff(ms_py.fleets[mfid], ms_cpp.fleets[mfid]))
            assert ms_fast is not None
            max_m_fast = max(max_m_fast, _fleet_max_diff(ms_py.fleets[mfid], ms_fast.to_sim().fleets[mfid]))
        if not ms_py.fleets:
            break
        ms_py.step({})
        ms_cpp.step({})
        if ms_fast is not None:
            ms_fast.step({})

    assert len(ms_py.fleets) == 0
    assert np.isclose(ms_py.planets[2].ships, 200.0, atol=1e-9)
    assert np.isclose(ms_py.planets[3].ships, 160.0, atol=1e-9)
    if HAS_FAST_SIM_CPP:
        assert max_m_cpp == 0.0, f"90 deg multi-ship drop-in C++ parity diff: {max_m_cpp:.3e}"
        assert max_m_fast == 0.0, f"90 deg multi-ship FastGalconCppSim parity diff: {max_m_fast:.3e}"

    # 3. Straight-line flight (no redirect) has zero turning slowdown (dot == 1.0, exact planet_eta_ticks arrival)
    s_src = Planet(id=1, owner=1, ships=1.0, x=20.0, y=100.0, production=0.0, radius=12.0)
    s_dst = Planet(
        id=2,
        owner=2,
        ships=10.0,
        x=20.0 + 12.0 + 12.0 + 2.0 * SHIP_RADIUS + 50.0,  # exact 5-tick travel distance
        y=100.0,
        production=0.0,
        radius=12.0,
    )
    sim_str = GalconSim([s_src, s_dst], width=200.0, height=200.0, fleet_speed=40.0, tick_dt=0.25)
    assert sim_str.planet_eta_ticks(s_src, s_dst) == 5
    sim_str.step({1: [SendOrder(1, 2, 100)]})
    for t_step in range(1, 5):
        assert len(sim_str.fleets) == 1
        fl_s = next(iter(sim_str.fleets.values()))
        assert fl_s.unit_headings == [[1.0, 0.0]]
        assert np.isclose(fl_s.x, 20.0 + 12.0 + SHIP_RADIUS + 10.0 * t_step, atol=1e-12)
        sim_str.step({})
    assert len(sim_str.fleets) == 0
    assert np.isclose(sim_str.planets[2].ships, 9.0, atol=1e-12)


def test_galcon_search_fast_cpp_exact_parity(tmp_path):
    """Verify FastGalconCppSim search & smart-redirect wiring has exact score/action parity with GalconSim.clone()."""
    from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim
    from galcon.gen5_model import build_galcon_gen5_transformer
    from galcon.sim import (
        Fleet,
        GalconSim,
        Planet,
        RedirOrder,
        SendOrder,
        evaluate_galcon_state,
    )
    from galcon.transformer_bot import (
        EnsembleGalconSearchAgent,
        GalconSearchTransformerAgent,
        build_galcon_gen4_transformer,
        compute_smart_redirects,
        extract_galcon_features,
    )

    # 1. FastGalconCppSim.can_represent(), context manager, idempotent close(), and to_summary_sim() parity
    sim_default = GalconSim.create(seed=1001, num_planets=18, width=480.0, height=320.0, symmetric=True)
    sim_no_cpp = GalconSim.create(seed=1001, num_planets=18, width=480.0, height=320.0, symmetric=True, use_cpp=False)
    assert FastGalconCppSim.can_represent(sim_no_cpp) is False
    assert FastGalconCppSim.can_represent(sim_default) is HAS_FAST_SIM_CPP

    # Oversized synthetic world (> 64 planets) cleanly rejected by can_represent
    oversized_planets = [
        Planet(id=i + 1, owner=(1 if i == 0 else (2 if i == 1 else 0)), ships=20.0, x=20.0 + (i % 10) * 40.0, y=20.0 + (i // 10) * 40.0, production=50.0, radius=12.0)
        for i in range(65)
    ]
    sim_oversized = GalconSim(oversized_planets, width=500.0, height=400.0)
    assert FastGalconCppSim.can_represent(sim_oversized) is False

    # Launch multi-ship flocks so to_summary_sim() is tested with active fleets
    sim_mid = sim_default.clone()
    p1_home = next(p for p in sim_mid.planets.values() if p.owner == 1)
    p2_home = next(p for p in sim_mid.planets.values() if p.owner == 2)
    neutrals = sorted((p for p in sim_mid.planets.values() if p.owner == 0), key=lambda p: p.id)
    sim_mid.step({
        1: [SendOrder(p1_home.id, neutrals[0].id, 35), SendOrder(p1_home.id, neutrals[2].id, 30)],
        2: [SendOrder(p2_home.id, neutrals[1].id, 35), SendOrder(p2_home.id, neutrals[3].id, 30)],
    })
    for _ in range(3):
        sim_mid.step({})

    if HAS_FAST_SIM_CPP:
        with FastGalconCppSim.from_sim(sim_mid) as fc:
            fc.step_n_ticks(5)
            full_s = fc.to_sim()
            sum_s = fc.to_summary_sim()
            assert sum_s.tick == full_s.tick
            assert sum_s.done == full_s.done
            assert sum_s.winner() == full_s.winner()
            assert sum_s.next_fleet_id == full_s.next_fleet_id
            assert list(sum_s.planets.keys()) == list(full_s.planets.keys())
            assert list(sum_s.fleets.keys()) == list(full_s.fleets.keys())
            for fid, fl_sum in sum_s.fleets.items():
                fl_full = full_s.fleets[fid]
                assert fl_sum.units is None and fl_sum.unit_headings is None
                assert fl_sum.ships == fl_full.ships and fl_sum.x == fl_full.x and fl_sum.y == fl_full.y
                assert fl_sum.target == fl_full.target and fl_sum.num_units == fl_full.num_units
            assert evaluate_galcon_state(sum_s, 1) == evaluate_galcon_state(full_s, 1)
            enc_sum = extract_galcon_features(sum_s, 1, feature_version=2)
            enc_full = extract_galcon_features(full_s, 1, feature_version=2)
            assert np.array_equal(enc_sum.planet_feats, enc_full.planet_feats)
            assert np.array_equal(enc_sum.timeline_feats, enc_full.timeline_feats)
            assert np.array_equal(enc_sum.edge_feats, enc_full.edge_feats)
        # Idempotent close after context manager exit
        fc.close()
        fc.close()

    # 2. compute_smart_redirects(..., verify_rollout=True): doomed-attack rescue + active midgame parity
    p_home = Planet(id=1, owner=1, ships=60.0, x=40.0, y=120.0, production=100.0, radius=24.0)
    p_fortress = Planet(id=2, owner=2, ships=120.0, x=240.0, y=120.0, production=100.0, radius=24.0)
    p_soft = Planet(id=3, owner=2, ships=8.0, x=150.0, y=45.0, production=90.0, radius=22.0)
    p_neu = Planet(id=4, owner=0, ships=10.0, x=150.0, y=195.0, production=70.0, radius=19.0)
    # Multi-unit doomed fleet of 25 ships heading at 120-ship fortress -> should redirect to soft enemy p_soft
    fl_doomed = Fleet(id=1000, owner=1, ships=25.0, x=115.0, y=110.0, source=1, target=2, num_units=16)
    sim_doomed = GalconSim([p_home, p_fortress, p_soft, p_neu], [fl_doomed], width=320.0, height=240.0)
    enc_doomed = extract_galcon_features(sim_doomed, 1, feature_version=2)
    redirs_cpp, labels_cpp = compute_smart_redirects(sim_doomed, enc_doomed, verify_rollout=True, use_cpp_search=True)
    redirs_py, labels_py = compute_smart_redirects(sim_doomed, enc_doomed, verify_rollout=True, use_cpp_search=False)
    assert redirs_cpp == redirs_py
    assert labels_cpp == labels_py
    assert redirs_cpp == [RedirOrder(1000, 3)]

    # Also test compute_smart_redirects on active midgame state with multiple flocks
    enc_mid = extract_galcon_features(sim_mid, 1, feature_version=2)
    r_mid_cpp, l_mid_cpp = compute_smart_redirects(sim_mid, enc_mid, verify_rollout=True, use_cpp_search=True)
    r_mid_py, l_mid_py = compute_smart_redirects(sim_mid, enc_mid, verify_rollout=True, use_cpp_search=False)
    assert r_mid_cpp == r_mid_py
    assert l_mid_cpp == l_mid_py

    # 3. Synthetic lightweight Gen 5 + Gen 4 checkpoints for GalconSearchTransformerAgent & EnsembleGalconSearchAgent
    torch.manual_seed(1010)
    m_gen5 = build_galcon_gen5_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128, edge_hidden=16)
    ckpt_gen5 = tmp_path / "gen5_search_parity.pt"
    torch.save(
        {
            "model_config": m_gen5.config,
            "model_state_dict": m_gen5.state_dict(),
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt_gen5,
    )

    torch.manual_seed(1011)
    m_gen4 = build_galcon_gen4_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt_gen4 = tmp_path / "gen4_search_parity.pt"
    torch.save(
        {
            "model_config": m_gen4.config,
            "model_state_dict": m_gen4.state_dict(),
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt_gen4,
    )

    # Build 3 diverse test states:
    # (a) Opening tick=0, 18pl symmetric (triggers Wave 16 opening def guard + use_adv)
    sim_open_18 = GalconSim.create(seed=14002, num_planets=18, width=480.0, height=320.0, symmetric=True)

    # (b) Midgame <22pl (tick=12, 18pl) with active multi-ship flocks & redirects (2-ply + adv soft-minimax)
    sim_mid_18 = GalconSim.create(seed=14005, num_planets=18, width=400.0, height=240.0, symmetric=True)
    h1 = next(p for p in sim_mid_18.planets.values() if p.owner == 1)
    h2 = next(p for p in sim_mid_18.planets.values() if p.owner == 2)
    n18 = sorted((p for p in sim_mid_18.planets.values() if p.owner == 0), key=lambda p: p.id)
    sim_mid_18.step({
        1: [SendOrder(h1.id, n18[0].id, 35), SendOrder(h1.id, n18[2].id, 30)],
        2: [SendOrder(h2.id, n18[1].id, 35), SendOrder(h2.id, n18[3].id, 30)],
    })
    for t_i in range(1, 12):
        if t_i == 6:
            sim_mid_18.step({
                1: [SendOrder(h1.id, n18[4].id, 40)],
                2: [SendOrder(h2.id, n18[5].id, 40)],
            })
        else:
            sim_mid_18.step({})
    assert sim_mid_18.tick == 12 and len(sim_mid_18.fleets) >= 2

    # (c) Large 22pl (tick=10, 480x320) with active multi-ship flocks (triggers 3-ply h=22 map-adaptive search)
    sim_large_22 = GalconSim.create(seed=14009, num_planets=22, width=480.0, height=320.0, symmetric=True)
    lh1 = next(p for p in sim_large_22.planets.values() if p.owner == 1)
    lh2 = next(p for p in sim_large_22.planets.values() if p.owner == 2)
    n22 = sorted((p for p in sim_large_22.planets.values() if p.owner == 0), key=lambda p: p.id)
    sim_large_22.step({
        1: [SendOrder(lh1.id, n22[0].id, 30), SendOrder(lh1.id, n22[2].id, 30)],
        2: [SendOrder(lh2.id, n22[1].id, 30), SendOrder(lh2.id, n22[3].id, 30)],
    })
    for t_i in range(1, 10):
        if t_i == 5:
            sim_large_22.step({
                1: [SendOrder(lh1.id, n22[4].id, 35)],
                2: [SendOrder(lh2.id, n22[5].id, 35)],
            })
        else:
            sim_large_22.step({})
    assert sim_large_22.tick == 10 and len(sim_large_22.planets) == 22 and len(sim_large_22.fleets) >= 2

    states = [
        ("open_18pl", sim_open_18),
        ("mid_18pl", sim_mid_18),
        ("large_22pl", sim_large_22),
    ]

    for ckpt in (ckpt_gen5, ckpt_gen4):
        ag_cpp = GalconSearchTransformerAgent(
            ckpt,
            time_budget_sec=5.0,
            use_d4_canonical=True,
            use_d4_tta=False,
            use_cpp_search=True,
        )
        ag_py = GalconSearchTransformerAgent(
            ckpt,
            time_budget_sec=5.0,
            use_d4_canonical=True,
            use_d4_tta=False,
            use_cpp_search=False,
        )
        assert ag_cpp.use_cpp_search is HAS_FAST_SIM_CPP
        assert ag_py.use_cpp_search is False

        for label, st in states:
            best_c, cands_c, scores_c, bsc_c = ag_cpp.act_with_search_info(st, 1)
            best_p, cands_p, scores_p, bsc_p = ag_py.act_with_search_info(st, 1)
            assert cands_c == cands_p, f"{ckpt.name}/{label}: candidate mismatch"
            assert len(scores_c) == len(scores_p) >= 2
            max_diff = max(abs(a - b) for a, b in zip(scores_c, scores_p))
            assert max_diff <= 1e-6, f"{ckpt.name}/{label}: score diff {max_diff:.3e} > 1e-6"
            assert abs(bsc_c - bsc_p) <= 1e-6
            assert best_c == best_p, f"{ckpt.name}/{label}: best action mismatch"

        # Clean fallback when sim.use_cpp=False
        sim_py_only = sim_mid_18.clone(use_cpp=False)
        assert FastGalconCppSim.can_represent(sim_py_only) is False
        best_fb, cands_fb, scores_fb, _ = ag_cpp.act_with_search_info(sim_py_only, 1)
        best_ref, cands_ref, scores_ref, _ = ag_py.act_with_search_info(sim_py_only, 1)
        assert cands_fb == cands_ref and best_fb == best_ref
        assert max(abs(a - b) for a, b in zip(scores_fb, scores_ref)) <= 1e-6

    # 4. EnsembleGalconSearchAgent inherits use_cpp_search via **kwargs and achieves exact parity
    ens_cpp = EnsembleGalconSearchAgent(
        [ckpt_gen5, ckpt_gen4],
        time_budget_sec=5.0,
        use_d4_canonical=True,
        use_d4_tta=False,
        use_cpp_search=True,
    )
    ens_py = EnsembleGalconSearchAgent(
        [ckpt_gen5, ckpt_gen4],
        time_budget_sec=5.0,
        use_d4_canonical=True,
        use_d4_tta=False,
        use_cpp_search=False,
    )
    assert ens_cpp.use_cpp_search is HAS_FAST_SIM_CPP
    assert ens_py.use_cpp_search is False
    for label, st in states:
        eb_c, ec_c, es_c, _ = ens_cpp.act_with_search_info(st, 1)
        eb_p, ec_p, es_p, _ = ens_py.act_with_search_info(st, 1)
        assert ec_c == ec_p, f"ensemble/{label}: candidate mismatch"
        assert max(abs(a - b) for a, b in zip(es_c, es_p)) <= 1e-6
        assert eb_c == eb_p, f"ensemble/{label}: best action mismatch"


def test_lua_bridge_same_tick_fleet_send_visibility(tmp_path):
    """Verify ItemMT:fleet_send immediately exposes launched fleets to g2.search and handles same-tick redirect."""
    import shutil
    from galcon.lua_bridge import GalconLuaBotBridge

    if not (shutil.which("lua5.3") or shutil.which("lua")):
        pytest.skip("lua5.3 / lua interpreter not installed on PATH")

    bot_src = """
function reset_bot() end
function bot_act(user)
    local p1 = g2.item(1)
    local p2 = g2.item(2)
    local p3 = g2.item(3)
    -- Send 50% from p1 -> p2; should be immediately visible in g2.search("fleet target:2")
    local fl = p1:fleet_send(50, p2)
    assert(fl ~= nil and fl.has_fleet, "fleet_send should return newly created fleet")
    local inc2 = g2.search("fleet target:" .. p2)
    assert(#inc2 == 1 and inc2[1].fleet_ships == 50, "same-tick fleet must be visible via g2.search")
    -- Redirect that same-tick fleet from p2 -> p3; should rewrite the pending SendOrder cleanly
    inc2[1]:fleet_redirect(p3)
    assert(#g2.search("fleet target:" .. p2) == 0)
    assert(#g2.search("fleet target:" .. p3) == 1)
    -- Second send from p1 -> p2, then redirect back to p1 (self-send cancels cleanly)
    local fl2 = p1:fleet_send(20, p2)
    assert(fl2 ~= nil and fl2.fleet_ships == 10)
    fl2:fleet_redirect(p1)
end
"""
    bot_file = tmp_path / "test_same_tick_bot.lua"
    bot_file.write_text(bot_src, encoding="utf-8")

    sim = GalconSim(
        [
            Planet(id=1, owner=1, ships=100.0, x=60.0, y=160.0, production=100.0, radius=24.0),
            Planet(id=2, owner=0, ships=10.0, x=240.0, y=160.0, production=50.0, radius=16.0),
            Planet(id=3, owner=2, ships=100.0, x=420.0, y=160.0, production=100.0, radius=24.0),
        ],
        width=480.0,
        height=320.0,
    )
    bridge = GalconLuaBotBridge(bot_file, loop_fn="bot_act", reset_fn="reset_bot")
    try:
        orders = bridge.act(sim, 1)
        assert orders == [SendOrder(1, 3, 50)], f"Unexpected orders: {orders}"
    finally:
        bridge.close()


def test_visualizer_always_floors_fractional_ships():
    """Visualizer snapshots and HTML/JS labels must always show floor(ships) even for fractional ships."""
    import math
    from galcon.record_replays import _floor_tenths, record_galcon_match, render_galcon_viewer_html

    # 1. Snapshot quantization must never round up across an integer boundary
    for val in (0.0, 0.04, 0.5, 0.96, 0.9999, 4.95, 9.96, math.nextafter(10.0, 0.0), 10.0, 12.34, 99.99):
        recorded = _floor_tenths(val)
        assert math.floor(recorded) == math.floor(val), f"_floor_tenths({val!r}) -> {recorded!r} crossed integer boundary"
        assert recorded <= val + 1e-12

    # 2. Recorded replay frames keep fractional ships floored to tenths (never rounded up to next int)
    def fractional_send_bot(sim: GalconSim, pid: int):
        if sim.tick == 0:
            # Force fractional ships on planet before sending 50% (leaving fractional fleet & planet ships)
            src_id = 1 if pid == 1 else 2
            dst_id = 2 if pid == 1 else 1
            sim.planets[src_id].ships = 19.96
            return [SendOrder(src_id, dst_id, 50)]
        return []

    match = record_galcon_match(
        "Fractional ship floor test",
        "P1",
        fractional_send_bot,
        "P2",
        fractional_send_bot,
        seed=42,
        num_planets=6,
        max_ticks=4,
    )
    # After tick 0 sends 50% of 19.96 (= 9.98), fleet has 9.98 ships -> must record <= 9.9 so floor == 9
    tick1_fleets = match["frames"][1]["f"]
    assert len(tick1_fleets) >= 2
    for fl in tick1_fleets:
        assert math.floor(fl[1]) == 9, f"Expected floored fleet ships == 9, got recorded {fl[1]}"

    # 3. Both generated viewer HTML and checked-in docs/galcon_viewer.html use Math.floor for fleets, planets, and HUD
    generated_html = render_galcon_viewer_html([match])
    checked_in_html = (Path(__file__).resolve().parent.parent / "docs" / "galcon_viewer.html").read_text(encoding="utf-8")
    for label, html in (("generated", generated_html), ("docs/galcon_viewer.html", checked_in_html)):
        assert "ctx.fillText(Math.floor(ships), tx(fx), ty(fy));" in html, f"{label}: fleet label must use Math.floor"
        assert "ctx.fillText(Math.floor(ships), tx(px), ty(py) - 2);" in html, f"{label}: planet label must use Math.floor"
        assert "Ships: ${Math.floor(f.s[0])}" in html, f"{label}: P1 HUD ships must use Math.floor"
        assert "Ships: ${Math.floor(f.s[1])}" in html, f"{label}: P2 HUD ships must use Math.floor"
        assert "ctx.fillText(Math.round(ships)" not in html, f"{label}: found Math.round(ships) in canvas text"
        assert "Ships: ${Math.round(f.s[" not in html, f"{label}: found Math.round(f.s[...]) in HUD"


def test_wave17_gen5_1t_ppo_and_ensemble_contempt(tmp_path):
    """Verify Wave 17 Gen 5 full-speed 1-tick PPO rollouts/val, WDL PPO update compatibility, and Ensemble contempt."""
    from galcon.gen5_model import build_galcon_gen5_transformer
    from galcon.heuristic_bot import GalconClassicBot
    from galcon.train import _run_gen4_ppo_episodes_batched, _run_gen4_val_wave_batched
    from galcon.transformer_bot import EnsembleGalconSearchAgent

    torch.manual_seed(17)
    m = build_galcon_gen5_transformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128, edge_hidden=16).eval()
    ckpt = tmp_path / "gen5_w17_test.pt"
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

    # 1. Full-speed 1-tick batched PPO rollout works natively on GalconGen5Transformer
    specs = [
        (1701, 10, 240.0, 150.0, True, 1, None, GalconClassicBot(), False),
        (1701, 10, 240.0, 150.0, True, 2, None, GalconClassicBot(), False),
    ]
    outs = _run_gen4_ppo_episodes_batched(
        m,
        specs,
        gamma=0.985,
        gae_lam=0.95,
        frac_std=0.065,
        draw_penalty=-0.4,
        boss_draw_penalty=-0.1,
        max_sends_per_planet=4,
        act_every=1,
    )
    assert len(outs) == 2 and all(len(o[0]) >= 2 for o in outs)

    # 2. Full-speed 1-tick batched validation wave runs cleanly on GalconGen5Transformer
    wave = [("heur", "classic", 1702, 10, 240.0, 150.0, 60, True, 1, GalconClassicBot(), False)]
    vres = _run_gen4_val_wave_batched(m, wave, act_every=1)
    assert len(vres) == 1 and vres[0][1] in (0.0, 0.5, 1.0)

    # 3. EnsembleGalconSearchAgent propagates Gen 5 contempt to secondary members
    ens = EnsembleGalconSearchAgent([ckpt, ckpt], use_d4_canonical=True, use_d4_tta=False)
    assert abs(ens.ensemble_agents[0].model.contempt - 0.30) < 1e-6
    assert abs(ens.ensemble_agents[1].model.contempt - 0.30) < 1e-6


def test_lua_bridge_fractional_fleet_send_matches_sim(tmp_path):
    """Verify Lua bridge ItemMT:fleet_send packs fractional ships (>= 1.0 - 1e-9) identically to GalconSim."""
    import shutil
    from galcon.lua_bridge import GalconLuaBotBridge

    if not (shutil.which("lua5.3") or shutil.which("lua")):
        pytest.skip("lua5.3 / lua interpreter not installed on PATH")

    bot_src = """
function bot_act(user)
    local p1 = g2.item(1)
    local p2 = g2.item(2)
    -- 50% of 10.8 ships should send 5.4 and leave 5.4 (matching GalconSim fractional ship packing)
    local fl = p1:fleet_send(50, p2)
    assert(fl ~= nil and math.abs(fl.fleet_ships - 5.4) < 1e-6, "expected fractional fleet_ships=5.4")
    assert(math.abs(p1.ships_value - 5.4) < 1e-6, "expected fractional remaining ships_value=5.4")
end
"""
    bot_file = tmp_path / "test_frac_send_bot.lua"
    bot_file.write_text(bot_src, encoding="utf-8")
    sim = GalconSim(
        [
            Planet(id=1, owner=1, ships=10.8, x=60.0, y=160.0, production=100.0, radius=24.0),
            Planet(id=2, owner=2, ships=50.0, x=320.0, y=160.0, production=100.0, radius=24.0),
        ]
    )
    bridge = GalconLuaBotBridge(bot_file, loop_fn="bot_act")
    try:
        orders = bridge.act(sim, 1)
        assert len(orders) == 1 and isinstance(orders[0], SendOrder) and orders[0].percent == 50
    finally:
        bridge.close()


def test_cpp_sim_multitick_and_fleet_slice_aabb_bit_exact():
    """Verify hoisted step_n_ticks_fast, fleet-slice dynamic AABB separation, and spawn broadphase match Python bit-for-bit."""
    from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim

    if not HAS_FAST_SIM_CPP:
        pytest.skip("C++ sim_core library unavailable")

    sim_py = GalconSim.create(
        seed=14000, num_planets=18, width=480.0, height=320.0, symmetric=True, max_ticks=60, use_cpp=False
    )
    pids = sorted(sim_py.planets.keys())
    for i in range(6):
        sim_py.planets[pids[i]].owner = 1
        sim_py.planets[pids[i]].ships = 60.0
        sim_py.planets[pids[-1 - i]].owner = 2
        sim_py.planets[pids[-1 - i]].ships = 60.0

    # Launch multi-fleet converging waves (same-target multi-fleet + partial landings across 16 ticks)
    o1 = [SendOrder(pids[i], pids[6 + (i % 3)], 50) for i in range(6)]
    o2 = [SendOrder(pids[-1 - i], pids[6 + (i % 3)], 50) for i in range(6)]

    root_cpp = FastGalconCppSim.from_sim(sim_py)
    try:
        sim_py.step({1: o1, 2: o2})
        root_cpp.step({1: o1, 2: o2})

        for _ in range(16):
            if sim_py.done:
                break
            sim_py.step({})
        root_cpp.step_n_ticks(16)
        out_cpp = root_cpp.to_sim()
    finally:
        root_cpp.close()

    assert sim_py.tick == out_cpp.tick
    assert sim_py.done == out_cpp.done
    assert sim_py.winner() == out_cpp.winner()
    for pid in pids:
        assert sim_py.planets[pid].owner == out_cpp.planets[pid].owner
        assert sim_py.planets[pid].ships == out_cpp.planets[pid].ships
    assert list(sim_py.fleets.keys()) == list(out_cpp.fleets.keys())
    for fid in sim_py.fleets:
        fp, fc = sim_py.fleets[fid], out_cpp.fleets[fid]
        assert fp.ships == fc.ships
        assert fp.x == fc.x and fp.y == fc.y
        assert fp.radius == fc.radius
        assert fp.units == fc.units
        assert fp.unit_headings == fc.unit_headings


def test_cpp_zero_copy_features_and_heuristic_eval_parity():
    """Verify zero-copy C++ feature extraction and C++ heuristic eval match summary-sim Python/C99 paths."""
    from galcon.fast_c import HAS_FAST_C
    from galcon.fast_sim_cpp import HAS_FAST_SIM_CPP, FastGalconCppSim
    from galcon.sim import evaluate_galcon_state
    from galcon.transformer_bot import extract_galcon_features, extract_galcon_features_from_cpp

    if not (HAS_FAST_SIM_CPP and HAS_FAST_C):
        pytest.skip("Requires compiled C++ sim_core and C99 fast_c libraries")

    sim = GalconSim.create(seed=14001, num_planets=18, width=480.0, height=320.0, symmetric=True, max_ticks=40)
    pids = sorted(sim.planets.keys())
    sim.planets[pids[0]].owner = 1
    sim.planets[pids[0]].ships = 80.0
    sim.planets[pids[-1]].owner = 2
    sim.planets[pids[-1]].ships = 80.0
    sim.step({1: [SendOrder(pids[0], pids[4], 50)], 2: [SendOrder(pids[-1], pids[5], 50)]})

    cpp_sim = FastGalconCppSim.from_sim(sim)
    try:
        cpp_sim.step_n_ticks(5)
        r_sum = cpp_sim.to_summary_sim()
        for fv in (1, 2):
            for pl in (1, 2):
                ref_enc = extract_galcon_features(r_sum, pl, feature_version=fv)
                ref_h = evaluate_galcon_state(r_sum, pl)
                p_pair = cpp_sim.read_planets_light()
                zc_enc, zc_h, zc_done = extract_galcon_features_from_cpp(
                    cpp_sim, pl, feature_version=fv, need_heuristic=True, planets_pair=p_pair
                )
                assert zc_enc is not None
                assert zc_done == r_sum.done
                assert zc_enc.planet_ids == ref_enc.planet_ids
                assert np.array_equal(zc_enc.planet_feats, ref_enc.planet_feats)
                assert np.array_equal(zc_enc.timeline_feats, ref_enc.timeline_feats)
                assert np.array_equal(zc_enc.edge_feats, ref_enc.edge_feats)
                assert np.array_equal(zc_enc.legal_edges, ref_enc.legal_edges)
                assert np.array_equal(zc_enc.owned_mask, ref_enc.owned_mask)
                assert np.array_equal(zc_enc.safe_surplus, ref_enc.safe_surplus)
                assert abs(zc_h - ref_h) < 1e-12
                assert abs(cpp_sim.evaluate_heuristic(pl) - ref_h) < 1e-12
    finally:
        cpp_sim.close()


def test_gen5_fast_infer_and_fused_root_exact_parity(tmp_path):
    """Verify FastGen5Infer dynamic-shape parity, fused B=2 root batching, and non-clobbering thread control."""
    import torch
    from galcon.gen5_model import build_fast_gen5_infer, build_galcon_gen5_transformer
    from galcon.transformer_bot import (
        EnsembleGalconSearchAgent,
        GalconSearchTransformerAgent,
        GalconTransformerAgent,
    )

    torch.manual_seed(77)
    model = build_galcon_gen5_transformer(
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_ff=128,
        edge_hidden=12,
        num_val_queries=4,
        contempt=0.25,
        dropout=0.10,
    ).eval()
    fast = build_fast_gen5_infer(model)
    assert fast is not None

    # 1. Dynamic (B, N) exact parity across eval_values, eval_ms, eval_ms_with_hidden, and live contempt
    for B, N in ((1, 10), (2, 18), (6, 22)):
        pf = torch.randn(B, N, 22)
        tf = torch.randn(B, N, 20, 8)
        ef = torch.randn(B, N, N, 12)
        pm = torch.ones(B, N, dtype=torch.bool)
        with torch.inference_mode():
            h_ref, e_ref = model.encode_planets_and_edges(pf, tf, ef, pm)
            ml_ref, mf_ref = model.predict_multi_send(h_ref, ef, edge_hidden=e_ref)
            for c_val in (0.0, 0.30):
                v_ref = model.decode_value_from_hidden(h_ref, pm, contempt=c_val)
                v_fast = fast.eval_values(pf, tf, ef, c_val)
                ml_f, mf_f, v_f2 = fast.eval_ms(pf, tf, ef, c_val, True)
                ml_h, mf_h, v_h, h_f = fast.eval_ms_with_hidden(pf, tf, ef, c_val)
                assert (v_ref - v_fast).abs().max().item() < 1e-5
                assert (v_ref - v_f2).abs().max().item() < 1e-5
                assert (v_ref - v_h).abs().max().item() < 1e-5
                assert (ml_ref - ml_f).abs().max().item() < 1e-5
                assert (mf_ref - mf_f).abs().max().item() < 1e-5
                assert (ml_ref - ml_h).abs().max().item() < 1e-5
                assert (mf_ref - mf_h).abs().max().item() < 1e-5
                assert (h_ref - h_f).abs().max().item() < 1e-5

    # 2. Save synthetic Gen 5 checkpoint and verify Search + Ensemble parity (with & without flying fleets)
    ckpt = tmp_path / "gen5_fast_parity.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "feature_version": 2,
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4_trained": True,
        },
        ckpt,
    )

    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        ag_fast = GalconSearchTransformerAgent(
            ckpt,
            use_d4_canonical=True,
            use_d4_tta=False,
            time_budget_sec=5.0,
            use_fast_infer=True,
        )
        # Default num_threads=None must NOT clobber caller's thread count
        assert torch.get_num_threads() == 2
        assert ag_fast._fast_infer is not None

        ag_ref = GalconSearchTransformerAgent(
            ckpt,
            use_d4_canonical=True,
            use_d4_tta=False,
            time_budget_sec=5.0,
            use_fast_infer=False,
            num_threads=1,
        )
        assert torch.get_num_threads() == 1
        assert ag_ref._fast_infer is None

        # Opening state (no fleets -> redir_enc is None) and midgame state (flying fleets -> redir_enc active)
        sim_open = GalconSim.create(seed=15001, num_planets=18, width=480.0, height=320.0, symmetric=True, max_ticks=60)
        sim_mid = sim_open.clone()
        pids = sorted(sim_mid.planets.keys())
        p1_home = next(pid for pid in pids if sim_mid.planets[pid].owner == 1)
        p2_home = next(pid for pid in pids if sim_mid.planets[pid].owner == 2)
        neu = next(pid for pid in pids if sim_mid.planets[pid].owner == 0)
        sim_mid.step({1: [SendOrder(p1_home, neu, 40)], 2: [SendOrder(p2_home, neu, 40)]})
        sim_mid.step({})

        for label, st in (("open", sim_open), ("mid_redir", sim_mid)):
            best_f, cands_f, scores_f, bsc_f = ag_fast.act_with_search_info(st, 1)
            best_r, cands_r, scores_r, bsc_r = ag_ref.act_with_search_info(st, 1)
            assert cands_f == cands_r, f"{label}: candidate mismatch"
            assert len(scores_f) == len(scores_r) >= 2
            max_diff = max(abs(a - b) for a, b in zip(scores_f, scores_r))
            assert max_diff < 1e-4, f"{label}: score diff {max_diff:.3e} >= 1e-4"
            assert abs(bsc_f - bsc_r) < 1e-4
            assert best_f == best_r, f"{label}: best action mismatch"

        # Ensemble parity check
        ens_fast = EnsembleGalconSearchAgent(
            [ckpt, ckpt],
            use_d4_canonical=True,
            use_d4_tta=False,
            time_budget_sec=5.0,
            use_fast_infer=True,
        )
        ens_ref = EnsembleGalconSearchAgent(
            [ckpt, ckpt],
            use_d4_canonical=True,
            use_d4_tta=False,
            time_budget_sec=5.0,
            use_fast_infer=False,
        )
        for label, st in (("ens_open", sim_open), ("ens_mid", sim_mid)):
            eb_f, ec_f, es_f, _ = ens_fast.act_with_search_info(st, 1)
            eb_r, ec_r, es_r, _ = ens_ref.act_with_search_info(st, 1)
            assert ec_f == ec_r, f"{label}: candidate mismatch"
            assert max(abs(a - b) for a, b in zip(es_f, es_r)) < 1e-4
            assert eb_f == eb_r, f"{label}: best action mismatch"

        # Pure agent parity check
        pure_f = GalconTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False, use_fast_infer=True)
        pure_r = GalconTransformerAgent(ckpt, use_d4_canonical=True, use_d4_tta=False, use_fast_infer=False)
        assert pure_f.act(sim_mid, 1) == pure_r.act(sim_mid, 1)
    finally:
        torch.set_num_threads(prev_threads)


def test_wave18_freeze_value_head_ppo(tmp_path):
    """Verify freeze_value_head=True keeps all Gen5 WDL value-head weights bit-for-bit frozen during PPO."""
    import torch
    from galcon.gen5_model import build_galcon_gen5_transformer
    from galcon.train import rl_mirror_league_ppo_gen4

    torch.manual_seed(1801)
    model = build_galcon_gen5_transformer(
        d_model=48,
        n_heads=4,
        n_layers=1,
        dim_ff=96,
        edge_hidden=8,
        num_val_queries=2,
        dropout=0.0,
    ).eval()
    ckpt_in = tmp_path / "gen5_in.pt"
    ckpt_out = tmp_path / "gen5_ppo_fv.pt"
    init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_config": model.config,
            "model_state_dict": init_sd,
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt_in,
    )

    rl_mirror_league_ppo_gen4(
        ckpt_in,
        ckpt_out,
        iters=1,
        pairs_per_iter=1,
        lr=1.0e-3,
        freeze_value_head=True,
        v_anchor_coef=0.30,
        act_every=2,
        eval_every=99,
        seed=1801,
        train_threads=1,
        workers=1,
    )
    final_ckpt = torch.load(ckpt_out.with_name(ckpt_out.stem + "_final.pt"), map_location="cpu", weights_only=False)
    fin_sd = final_ckpt["model_state_dict"]
    val_keys = [k for k in init_sd if k.startswith("val_") or "value_head" in k]
    assert len(val_keys) >= 4
    for k in val_keys:
        assert torch.equal(init_sd[k], fin_sd[k]), f"Value-head param {k} mutated despite freeze_value_head=True"
    ms_changed = any(not torch.equal(init_sd[k], fin_sd[k]) for k in init_sd if k.startswith("ms_"))
    assert ms_changed, "Expected multi-send policy weights to update during frozen-value-head PPO"


def test_wave22_freeze_trunk_wdl_recal_and_large_tunnel_cands(tmp_path):
    """Verify freeze_trunk=True updates only WDL value heads and Large 24pl search includes tunnel/focus-fire."""
    import numpy as np
    import torch
    from galcon.gen5_model import build_galcon_gen5_transformer
    from galcon.sim import GalconSim
    from galcon.train import train_gen4_multisend_supervised
    from galcon.transformer_bot import GALCON_EDGE_DIM_V2, GALCON_PLANET_DIM_V2, GALCON_TIMELINE_DIM, GalconSearchTransformerAgent

    torch.manual_seed(2201)
    model = build_galcon_gen5_transformer(
        d_model=48,
        n_heads=4,
        n_layers=1,
        dim_ff=96,
        edge_hidden=8,
        num_val_queries=2,
        dropout=0.08,
    ).eval()
    ckpt_in = tmp_path / "gen5_recal_in.pt"
    ckpt_out = tmp_path / "gen5_recal_out.pt"
    init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_config": model.config,
            "model_state_dict": init_sd,
            "has_trained_redir": True,
            "has_trained_multi_send": True,
            "use_canonical_d4": True,
        },
        ckpt_in,
    )

    b, n = 24, 18
    rng = np.random.default_rng(2202)
    le = np.ones((b, n, n), dtype=bool)
    np.fill_diagonal(le[0], False)
    for k in range(1, b):
        le[k] = le[0]
    om = np.zeros((b, n), dtype=bool)
    om[:, :5] = True
    pm = np.ones((b, n), dtype=bool)
    synth = {
        "planet_feats": rng.normal(0.0, 0.5, (b, n, GALCON_PLANET_DIM_V2)).astype(np.float32),
        "timeline_feats": rng.normal(0.0, 0.5, (b, n, 20, GALCON_TIMELINE_DIM)).astype(np.float32),
        "edge_feats": rng.normal(0.0, 0.5, (b, n, n, GALCON_EDGE_DIM_V2)).astype(np.float32),
        "legal_edges": le,
        "owned_mask": om,
        "pad_mask": pm,
        "ms_target": np.zeros((b, n, n), dtype=np.float32),
        "ms_frac": np.zeros((b, n, n), dtype=np.float32),
        "val_targets": rng.uniform(-1.0, 1.0, (b,)).astype(np.float32),
        "aux_owner": rng.integers(0, 3, (b, n), dtype=np.int64),
        "aux_ships": rng.uniform(0.0, 1.0, (b, n)).astype(np.float32),
    }

    train_gen4_multisend_supervised(
        synth,
        {},
        resume_from=ckpt_in,
        out_path=ckpt_out,
        epochs=2,
        batch_size=8,
        lr=1.0e-2,
        arch="gen5",
        freeze_trunk=True,
        seed=2203,
        train_threads=1,
        device="cpu",
    )
    out_sd = torch.load(ckpt_out, map_location="cpu", weights_only=False)["model_state_dict"]
    for k in init_sd:
        if k.startswith("val_") or "value_head" in k:
            continue
        assert torch.equal(init_sd[k], out_sd[k]), f"Trunk param {k} mutated despite freeze_trunk=True"
    val_changed = any(not torch.equal(init_sd[k], out_sd[k]) for k in init_sd if k.startswith("val_") or "value_head" in k)
    assert val_changed, "Expected WDL value-head weights to update when freeze_trunk=True"

    # Large 24pl asymmetric map check with tunnel-feed + focus-fire search enabled
    sim_lg = GalconSim.create(seed=18000, num_planets=24, width=560.0, height=340.0, symmetric=False, max_ticks=240)
    for _ in range(14):
        sim_lg.step({})
    ag = GalconSearchTransformerAgent(ckpt_out, use_d4_canonical=True, use_d4_tta=False)
    best_plan, cands, scores, _ = ag.act_with_search_info(sim_lg, 1)
    assert len(cands) >= 1 and len(scores) == len(cands)
    assert isinstance(best_plan, list)

    # Tick-0 Opening Guard75 check (exercises _eval_opening_guard_against_opp + pure-hold min)
    sim0 = GalconSim.create(seed=14003, num_planets=18, width=400.0, height=240.0, symmetric=True, max_ticks=200)
    plan0, cands0, scores0, _ = ag.act_with_search_info(sim0, 1)
    assert len(cands0) >= 1 and len(scores0) == len(cands0)
    assert isinstance(plan0, list)


def test_wave22_gen5_supervised_dml_updates_weights(tmp_path):
    """Regression test: train_gen4_multisend_supervised updates weights after .to(dml)."""
    import subprocess
    import sys
    import pytest
    try:
        import torch_directml
    except Exception:
        pytest.skip("torch_directml not installed")
    if torch_directml.device_count() <= 0:
        pytest.skip("No DirectML GPU available")

    # Run in a fresh subprocess so PyTorch's autograd engine is not pre-initialized on CPU before torch_directml loads.
    script = (
        "import numpy as np, torch\n"
        "from pathlib import Path\n"
        "from galcon.gen5_model import build_galcon_gen5_transformer\n"
        "from galcon.train import train_gen4_multisend_supervised\n"
        "from galcon.transformer_bot import GALCON_EDGE_DIM_V2, GALCON_PLANET_DIM_V2, GALCON_TIMELINE_DIM\n"
        "tmp = Path(" + repr(str(tmp_path)) + ")\n"
        "torch.manual_seed(2204)\n"
        "m = build_galcon_gen5_transformer(d_model=48, n_heads=4, n_layers=1, dim_ff=96, edge_hidden=8, num_val_queries=2, dropout=0.0).eval()\n"
        "ckpt_in, ckpt_out = tmp / 'gen5_dml_in.pt', tmp / 'gen5_dml_out.pt'\n"
        "init_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}\n"
        "torch.save({'model_config': m.config, 'model_state_dict': init_sd, 'has_trained_multi_send': True}, ckpt_in)\n"
        "b, n = 24, 18\n"
        "rng = np.random.default_rng(2205)\n"
        "le = np.ones((b, n, n), dtype=bool)\n"
        "np.fill_diagonal(le[0], False)\n"
        "for k in range(1, b): le[k] = le[0]\n"
        "om = np.zeros((b, n), dtype=bool); om[:, :5] = True\n"
        "synth = {'planet_feats': rng.normal(0.0, 0.5, (b, n, GALCON_PLANET_DIM_V2)).astype(np.float32),"
        " 'timeline_feats': rng.normal(0.0, 0.5, (b, n, 20, GALCON_TIMELINE_DIM)).astype(np.float32),"
        " 'edge_feats': rng.normal(0.0, 0.5, (b, n, n, GALCON_EDGE_DIM_V2)).astype(np.float32),"
        " 'legal_edges': le, 'owned_mask': om, 'pad_mask': np.ones((b, n), dtype=bool),"
        " 'ms_target': (rng.random((b, n, n)) < 0.1).astype(np.float32) * om[:, :, None],"
        " 'ms_frac': rng.uniform(0.2, 0.8, (b, n, n)).astype(np.float32),"
        " 'val_targets': rng.uniform(-1.0, 1.0, (b,)).astype(np.float32),"
        " 'aux_owner': rng.integers(0, 3, (b, n), dtype=np.int64),"
        " 'aux_ships': rng.uniform(0.0, 1.0, (b, n)).astype(np.float32)}\n"
        "train_gen4_multisend_supervised(synth, {}, resume_from=ckpt_in, out_path=ckpt_out, epochs=1, batch_size=8, lr=5e-3, arch='gen5', freeze_trunk=False, seed=2206, train_threads=1, device='dml')\n"
        "out_sd = torch.load(ckpt_out, map_location='cpu', weights_only=False)['model_state_dict']\n"
        "ms_diff = max(float((init_sd[k] - out_sd[k]).abs().max()) for k in init_sd if k.startswith('ms_'))\n"
        "val_diff = max(float((init_sd[k] - out_sd[k]).abs().max()) for k in init_sd if k.startswith('val_'))\n"
        "assert ms_diff > 0.0 and val_diff > 0.0, (ms_diff, val_diff)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"DirectML subprocess failed:\n{proc.stdout}\n{proc.stderr}"


def test_unified_elo_math_and_reanchor():
    """Verify exact Bradley-Terry MLE, sweep finiteness, Laplace std, and re-anchoring."""
    import math
    from galcon.elo_rating import MatchRecord, fit_bradley_terry_elo

    # Exact 75% win rate (30-10) -> +400 * log10(3) = +190.8485... Elo with zero prior
    m_75 = [MatchRecord(player_a="B", player_b="A", wins_a=30, draws=0, losses_a=10)]
    est = fit_bradley_terry_elo(["A", "B"], m_75, anchor_id="A", anchor_elo=1200.0, prior_draws=0.0, l2_reg=0.0)
    assert abs(est["A"].elo - 1200.0) < 1e-8
    expected_diff = 400.0 * math.log10(3.0)
    assert abs((est["B"].elo - est["A"].elo) - expected_diff) < 1e-6
    assert abs(round(est["B"].elo - est["A"].elo, 2) - 190.85) < 1e-6
    assert 0.0 < est["B"].elo_std < 500.0 and math.isfinite(est["B"].elo_std)

    # 10-0 sweep stays finite under default prior_draws=0.1, l2_reg=1e-4
    sweep = [MatchRecord(player_a="B", player_b="A", wins_a=10, draws=0, losses_a=0)]
    est_sw = fit_bradley_terry_elo(["A", "B"], sweep, anchor_id="A", anchor_elo=0.0, prior_draws=0.1, l2_reg=1e-4)
    assert math.isfinite(est_sw["B"].elo) and est_sw["B"].elo > 400.0
    assert math.isfinite(est_sw["B"].elo_std) and est_sw["B"].elo_std > 0.0

    # Zero-reg multi-player re-anchor shifts every Elo by an exact constant
    chain = [
        MatchRecord(player_a="B", player_b="A", wins_a=14, draws=2, losses_a=4),
        MatchRecord(player_a="C", player_b="B", wins_a=12, draws=2, losses_a=6),
    ]
    e_a = fit_bradley_terry_elo(["A", "B", "C"], chain, anchor_id="B", anchor_elo=1200.0, prior_draws=0.1, l2_reg=0.0)
    e_r = fit_bradley_terry_elo(["A", "B", "C"], chain, anchor_id="A", anchor_elo=0.0, prior_draws=0.1, l2_reg=0.0)
    assert abs((e_a["C"].elo - e_a["A"].elo) - (e_r["C"].elo - e_r["A"].elo)) < 1e-6
    assert abs((e_a["B"].elo - e_a["A"].elo) - e_r["B"].elo) < 1e-6


def test_unified_elo_ladder_selection_promotion_pruning_and_roundtrip():
    """Test EloLadder opponent selection, promotion gates, interior pruning, and JSON serialization."""
    from galcon.elo_rating import EloLadder, PlayerEntry

    ladder = EloLadder(anchor_id="bot:classic", anchor_elo=1200.0, max_ladder_size=3, opponents_per_eval=3)
    for idx, (pid, wins) in enumerate([("ckpt_1", 12), ("ckpt_2", 14), ("ckpt_3", 15), ("ckpt_4", 16)], start=1):
        ladder.register_player(PlayerEntry(player_id=pid, kind="checkpoint", train_iter=idx * 100))
        prev = "bot:classic" if idx == 1 else f"ckpt_{idx - 1}"
        ladder.record_match(pid, prev, wins_a=wins, draws=0, losses_a=20 - wins)
    ladder.fit_ratings()

    assert ladder.get_champion().player_id == "ckpt_4"
    opps = ladder.select_opponents("ckpt_5", max_opponents=3)
    opp_ids = {o.player_id for o in opps}
    assert "ckpt_4" in opp_ids and len(opps) == 3

    # Promotion thresholds: first checkpoint, elo margin (+50), and win rate (>= 55%)
    empty_ladder = EloLadder(anchor_id="bot:classic", anchor_elo=1200.0)
    ok_first, reason_first = empty_ladder.should_promote("c0", None)
    assert ok_first and reason_first == "first_checkpoint_seed"

    ladder.register_player(PlayerEntry(player_id="ckpt_new", kind="checkpoint", train_iter=500))
    ladder.record_match("ckpt_new", "ckpt_4", wins_a=12, draws=0, losses_a=8)
    ladder.fit_ratings()
    ok_prom, reason_prom = ladder.should_promote("ckpt_new", score_vs_champion=0.60)
    assert ok_prom and ("beat_champion" in reason_prom or "elo_margin" in reason_prom)

    # Interior pruning drops tightly packed middle rungs while keeping weakest, strongest, and full match history
    pruned = ladder.prune_if_needed(protected_ids={"ckpt_new"})
    assert len(pruned) >= 1
    active_ckpts = [p.player_id for p in ladder.active_checkpoint_rungs()]
    assert len(active_ckpts) <= 3
    assert "ckpt_1" in active_ckpts and "ckpt_new" in active_ckpts
    n_matches_before = len(ladder.matches)

    restored = EloLadder.from_dict(ladder.to_dict())
    assert len(restored.matches) == n_matches_before
    assert abs(restored.get_elo("ckpt_new") - ladder.get_elo("ckpt_new")) < 1e-6

    restored.reanchor("ckpt_1", 500.0)
    assert abs(restored.get_elo("ckpt_1") - 500.0) < 1e-8


def test_pass_and_random_bots():
    """Verify GalconPassBot passes cleanly and GalconRandomBot is deterministic and legal."""
    from galcon.heuristic_bot import GalconPassBot, GalconRandomBot

    sim = GalconSim.create(seed=101, num_planets=10, width=280.0, height=180.0, symmetric=True, use_cpp=True)
    pbot = GalconPassBot()
    assert pbot.act(sim, 1) == []

    r1 = GalconRandomBot(seed=7)
    r2 = GalconRandomBot(seed=7)
    o1 = r1.act(sim, 1)
    o2 = r2.act(sim, 1)
    assert len(o1) == 1 and o1 == o2
    order = o1[0]
    assert isinstance(order, SendOrder)
    assert sim.planets[order.src_id].owner == 1
    assert order.src_id != order.dst_id
    assert order.percent == 50


def test_tournament_cli_end_to_end_and_resume(tmp_path, capsys):
    """Run mini 3-bot tournament in tmp_path, --resume with 4th bot, and --report-only re-anchor."""
    import json
    from galcon.tournament import main as tournament_main

    state_path = tmp_path / "tourney.json"
    md_path = tmp_path / "tourney.md"

    tournament_main([
        "--bots",
        "bot:random=heuristic:random",
        "bot:classic=heuristic:classic",
        "bot:rusher=heuristic:rusher",
        "--state", str(state_path),
        "--mode", "ladder",
        "--arena", "compact6",
        "--games-per-pair", "2",
        "--rounds", "2",
        "--workers", "1",
        "--anchor", "bot:classic",
        "--anchor-elo", "1200",
        "--secondary-anchor", "bot:random",
        "--secondary-anchor-elo", "0",
    ])
    assert state_path.is_file()
    st1 = json.loads(state_path.read_text(encoding="utf-8"))
    matches_1 = st1["ladder"]["matches"]
    assert len(matches_1) == 3  # 2 chain pairs + 1 Swiss pair covers all 3 pairs of 3 bots
    seen_pairs_1 = {tuple(sorted((m["player_a"], m["player_b"]))) for m in matches_1}
    assert len(seen_pairs_1) == len(matches_1)
    assert all((m["wins_a"] + m["draws"] + m["losses_a"]) == 2 for m in matches_1)

    # Resume and add bot:macro; existing 3 pairs must not be re-played
    tournament_main([
        "--resume",
        "--bots",
        "bot:macro=heuristic:macro",
        "--state", str(state_path),
        "--mode", "ladder",
        "--arena", "compact6",
        "--games-per-pair", "2",
        "--rounds", "2",
        "--workers", "1",
        "--anchor", "bot:classic",
        "--anchor-elo", "1200",
    ])
    st2 = json.loads(state_path.read_text(encoding="utf-8"))
    matches_2 = st2["ladder"]["matches"]
    assert len(matches_2) > len(matches_1)
    seen_pairs_2 = {tuple(sorted((m["player_a"], m["player_b"]))) for m in matches_2}
    assert len(seen_pairs_2) == len(matches_2)
    assert any("bot:macro" in (m["player_a"], m["player_b"]) for m in matches_2[len(matches_1):])

    # Instant --report-only re-anchor to bot:random=0.0 without running games
    capsys.readouterr()
    tournament_main([
        "--report-only",
        "--state", str(state_path),
        "--anchor", "bot:random",
        "--anchor-elo", "0",
        "--secondary-anchor", "bot:classic",
        "--secondary-anchor-elo", "1200",
        "--markdown-out", str(md_path),
    ])
    out = capsys.readouterr().out
    assert "bot:random" in out and "bot:macro" in out
    md_text = md_path.read_text(encoding="utf-8")
    assert "| Rank |" in md_text and "`bot:random`" in md_text





