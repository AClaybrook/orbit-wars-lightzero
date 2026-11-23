"""Unit tests for the high-fidelity PlanetTimelineTransformer and feature extractor."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from kaggle_environments import make

from orbit_wars_bot.planet_transformer import (
    EDGE_FEAT_DIM,
    MAX_PLANETS,
    PLANET_FEAT_DIM,
    TIMELINE_FEAT_DIM,
    TIMELINE_HORIZON,
    PlanetTimelineTransformer,
    PlanetTransformerAgent,
    SearchPlanetTransformerAgent,
    apply_d4_planet_feats,
    extract_planet_features,
    match_action_to_target,
    pad_encoded_state,
)


def test_extract_planet_features_2p_and_4p_shapes():
    for n_players in (2, 4):
        env = make("orbit_wars", configuration={"seed": 42})
        obs = env.reset(n_players)[0]["observation"]
        enc = extract_planet_features(obs, player=0)
        n = len(enc.planet_ids)
        assert 4 <= n <= MAX_PLANETS
        assert enc.planet_feats.shape == (n, PLANET_FEAT_DIM)
        assert enc.timeline_feats.shape == (n, TIMELINE_HORIZON, TIMELINE_FEAT_DIM)
        assert enc.edge_feats.shape == (n, n, EDGE_FEAT_DIM)
        assert enc.legal_edges.shape == (n, n)
        assert enc.owned_mask.sum() >= 1
        assert np.isfinite(enc.planet_feats).all()
        assert np.isfinite(enc.timeline_feats).all()
        assert np.isfinite(enc.edge_feats).all()


def test_d4_planet_feats_group_closure():
    env = make("orbit_wars", configuration={"seed": 77})
    obs = env.reset(2)[0]["observation"]
    enc = extract_planet_features(obs, player=0)
    pf = enc.planet_feats

    # 4 rotations return to identity; double flip returns to identity
    r = pf.copy()
    for _ in range(4):
        r = apply_d4_planet_feats(r, k_rot=1, flip_x=False)
    assert np.allclose(r, pf, atol=1e-6)

    f = apply_d4_planet_feats(apply_d4_planet_feats(pf, flip_x=True), flip_x=True)
    assert np.allclose(f, pf, atol=1e-6)

    # Non-coordinate columns (cols 4..) are untouched
    t = apply_d4_planet_feats(pf, k_rot=3, flip_x=True)
    assert np.allclose(t[:, 4:], pf[:, 4:], atol=1e-6)


def test_model_forward_and_match_action_to_target(tmp_path: Path):
    env = make("orbit_wars", configuration={"seed": 100})
    obs = env.reset(2)[0]["observation"]
    enc = extract_planet_features(obs, player=0)
    padded = pad_encoded_state(enc)

    model = PlanetTimelineTransformer(d_model=128, n_heads=4, n_layers=4, dim_ff=256)
    n_params = sum(p.numel() for p in model.parameters())
    assert 200_000 < n_params < 1_500_000

    logits, fracs, val = model(
        torch.from_numpy(padded["planet_feats"][None]),
        torch.from_numpy(padded["timeline_feats"][None]),
        torch.from_numpy(padded["edge_feats"][None]),
        torch.from_numpy(padded["pad_mask"][None]),
    )
    assert logits.shape == (1, MAX_PLANETS, MAX_PLANETS + 1)
    assert fracs.shape == (1, MAX_PLANETS, MAX_PLANETS)
    assert val.shape == (1,)
    assert -1.0 <= float(val.item()) <= 1.0

    # Save checkpoint and verify PlanetTransformerAgent latency & action validity
    ckpt_path = tmp_path / "pt_test.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, ckpt_path)
    agent = PlanetTransformerAgent(ckpt_path, launch_threshold=0.01, use_rot_tta=True)

    t0 = time.perf_counter()
    moves = agent.act(obs)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 400.0
    assert isinstance(moves, list)
    if moves:
        rec = match_action_to_target(moves[0], enc)
        assert rec is not None
        src_i, dst_i, frac = rec
        assert enc.owned_mask[src_i]
        assert src_i != dst_i
        assert 0.0 < frac <= 1.0


def test_search_planet_transformer_agent_2p_and_4p(tmp_path: Path):
    model = PlanetTimelineTransformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt_path = tmp_path / "pt_search_test.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, ckpt_path)
    agent = SearchPlanetTransformerAgent(
        ckpt_path,
        launch_threshold=0.10,
        use_rot_tta=False,
        rollout_horizon=6,
        time_budget_sec=0.25,
    )
    for n_players in (2, 4):
        env = make("orbit_wars", configuration={"seed": 100 + n_players})
        obs = env.reset(n_players)[0]["observation"]
        enc = extract_planet_features(obs, player=0)
        t0 = time.perf_counter()
        moves = agent.act(obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert dt_ms < 600.0
        assert isinstance(moves, list)
        owned_pids = {enc.planet_ids[i] for i in range(len(enc.planet_ids)) if enc.owned_mask[i]}
        for mv in moves:
            assert len(mv) == 3
            assert int(mv[0]) in owned_pids
            assert int(mv[2]) >= 1


def test_rl_ppo_rollout_and_update():
    import copy
    import sys

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from rl_finetune_transformer import collect_episode, ppo_update

    policy = PlanetTimelineTransformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    anchor = copy.deepcopy(policy)
    anchor.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in policy.parameters()]
    batch, stats = collect_episode(
        policy, anchor, num_players=2, opponent_kind="heuristic", seed=77, max_steps=15
    )
    assert len(batch["returns"]) > 0
    assert stats["steps"] > 0
    metrics = ppo_update(policy, anchor, opt, batch, ppo_epochs=1, batch_size=16)
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["kl"])
    assert metrics["kl"] >= 0.0
    changed = any(not torch.allclose(b, a) for b, a in zip(before, policy.parameters()))
    assert changed


def test_d4_tta_and_ensemble_agent(tmp_path: Path):
    from orbit_wars_bot.planet_transformer import EnsemblePlanetTransformerAgent

    env = make("orbit_wars", configuration={"seed": 123})
    obs = env.reset(2)[0]["observation"]
    enc = extract_planet_features(obs, player=0)

    m1 = PlanetTimelineTransformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    m2 = PlanetTimelineTransformer(d_model=48, n_heads=4, n_layers=2, dim_ff=96)
    p1 = tmp_path / "m1.pt"
    p2 = tmp_path / "m2.pt"
    torch.save({"model_config": m1.config, "model_state_dict": m1.state_dict()}, p1)
    torch.save({"model_config": m2.config, "model_state_dict": m2.state_dict()}, p2)

    a_d4 = PlanetTransformerAgent(p1, launch_threshold=0.05, use_d4_tta=True)
    probs_d4, fracs_d4, val_d4 = a_d4.predict_probs_and_value(enc)
    assert probs_d4.shape == (len(enc.planet_ids), len(enc.planet_ids) + 1)
    assert np.allclose(probs_d4.sum(axis=-1), 1.0, atol=1e-4)
    assert -1.0 <= val_d4 <= 1.0

    # Verify exact D4 invariance under reflection + rotation when use_d4_tta=True
    enc_ref = extract_planet_features(obs, player=0)
    enc_ref.planet_feats = apply_d4_planet_feats(enc_ref.planet_feats, k_rot=2, flip_x=True)
    probs_ref, _, val_ref = a_d4.predict_probs_and_value(enc_ref)
    assert np.allclose(probs_d4, probs_ref, atol=1e-4)
    assert abs(val_d4 - val_ref) < 1e-4

    # Heterogeneous ensemble (different d_model sizes)
    ens = EnsemblePlanetTransformerAgent([p1, p2], weights=[0.7, 0.3], launch_threshold=0.05, use_d4_tta=True)
    p_ens, f_ens, v_ens = ens.predict_probs_and_value(enc)
    assert p_ens.shape == probs_d4.shape
    assert np.allclose(p_ens.sum(axis=-1), 1.0, atol=1e-4)
    assert -1.0 <= v_ens <= 1.0
    moves = ens.act(obs)
    assert isinstance(moves, list)


def test_decode_plan_blocked_shot_fallback(tmp_path: Path, monkeypatch):
    import orbit_wars_bot.planet_transformer as pt_mod

    env = make("orbit_wars", configuration={"seed": 55})
    obs = env.reset(2)[0]["observation"]
    enc = extract_planet_features(obs, player=0)

    model = PlanetTimelineTransformer(d_model=64, n_heads=4, n_layers=2, dim_ff=128)
    ckpt = tmp_path / "fallback.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, ckpt)
    agent = PlanetTransformerAgent(ckpt, launch_threshold=0.50, use_rot_tta=False)

    n = len(enc.planet_ids)
    owned_idx = int(np.flatnonzero(enc.owned_mask)[0])
    legal_targets = [j for j in range(n) if enc.legal_edges[owned_idx, j]]
    assert len(legal_targets) >= 2
    t1, t2 = legal_targets[0], legal_targets[1]

    probs = np.zeros((n, n + 1), dtype=np.float32)
    probs[:, 0] = 1.0
    probs[owned_idx, 0] = 0.05
    probs[owned_idx, 1 + t1] = 0.65
    probs[owned_idx, 1 + t2] = 0.30
    fracs = np.full((n, n), 0.75, dtype=np.float32)

    orig_solve = pt_mod.solve_intercept
    blocked_dst_id = int(enc.planet_rows[t1][0])
    calls = []

    def fake_solve(src_row, dst_row, *args, **kwargs):
        calls.append(int(dst_row[0]))
        if int(dst_row[0]) == blocked_dst_id:
            return None
        return orig_solve(src_row, dst_row, *args, **kwargs)

    monkeypatch.setattr(pt_mod, "solve_intercept", fake_solve)
    moves = agent._decode_plan(enc, probs, fracs, launch_threshold=0.50, force_all_in=False, max_speed=6.0)
    assert blocked_dst_id in calls
    assert len(moves) == 1
    assert int(moves[0][0]) == int(enc.planet_rows[owned_idx][0])

