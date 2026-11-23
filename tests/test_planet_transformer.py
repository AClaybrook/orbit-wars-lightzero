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
