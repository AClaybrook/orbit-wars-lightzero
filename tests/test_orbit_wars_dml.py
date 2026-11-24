"""Unit tests for shared DirectML utilities and zero-fallback Orbit Wars trainers."""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from orbit_wars_bot.dml_utils import (
    DMLAdamW,
    build_optimizer,
    cpu_state_dict,
    one_hot_no_scatter,
    resolve_train_device,
)
from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.planet_transformer import (
    MAX_PLANETS,
    PlanetTimelineTransformer,
    extract_planet_features,
    pad_encoded_state,
)


def _has_dml_gpu() -> bool:
    try:
        import torch_directml

        return torch_directml.device_count() > 0
    except Exception:
        return False


# Register DirectML backend before any test executes CPU backward() (PyTorch caches autograd device queues on first backward).
_HAS_DML_GPU = _has_dml_gpu()


def _assert_no_dml_fallback(caught_warnings) -> None:
    for w in caught_warnings:
        msg = str(w.message).lower()
        if "fallback" in msg or "not currently supported" in msg:
            pytest.fail(f"Unexpected DirectML CPU fallback warning: {w.message}")


def test_dml_utils_optimizer_and_one_hot():
    dev, is_dml = resolve_train_device("cpu")
    assert dev.type == "cpu" and not is_dml
    with pytest.raises(ValueError):
        resolve_train_device("tpu_invalid")

    torch.manual_seed(42)
    m_ref = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))
    m_dml = copy.deepcopy(m_ref)

    opt_ref = torch.optim.AdamW(
        [
            {"params": list(m_ref[0].parameters()), "lr": 1e-3},
            {"params": list(m_ref[2].parameters()), "lr": 2e-3},
        ],
        weight_decay=1e-2,
    )
    opt_dml = DMLAdamW(
        [
            {"params": list(m_dml[0].parameters()), "lr": 1e-3},
            {"params": list(m_dml[2].parameters()), "lr": 2e-3},
        ],
        weight_decay=1e-2,
    )
    sched_ref = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ref, T_max=5, eta_min=1e-4)
    sched_dml = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dml, T_max=5, eta_min=1e-4)

    for step in range(5):
        torch.manual_seed(200 + step)
        x = torch.randn(6, 8)
        y = torch.randn(6, 4)

        opt_ref.zero_grad()
        F.mse_loss(m_ref(x), y).backward()
        opt_ref.step()
        sched_ref.step()

        opt_dml.zero_grad()
        F.mse_loss(m_dml(x), y).backward()
        opt_dml.step()
        sched_dml.step()

    for p_ref, p_dml in zip(m_ref.parameters(), m_dml.parameters()):
        assert torch.allclose(p_ref, p_dml, atol=1e-6)

    idx = torch.tensor([[0, 2, 1], [3, 0, 2]], dtype=torch.long)
    assert torch.equal(one_hot_no_scatter(idx, 5, dtype=torch.float32), F.one_hot(idx, num_classes=5).float())


def test_rl_ppo_paired_rollout_aux_dual_kl_cpu_and_dml():
    from rl_finetune_transformer import collect_episode, ppo_update

    torch.manual_seed(7)
    policy = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=True)
    teacher = copy.deepcopy(policy).eval()
    fixed_anchor = copy.deepcopy(policy).eval()

    batches = []
    for slot in (0, 1):
        b, st = collect_episode(
            policy,
            teacher,
            num_players=2,
            opponent_kind="heuristic",
            seed=77,
            max_steps=20,
            force_all_in=True,
            hero_slot=slot,
            rollout_d4=True,
        )
        assert len(b["returns"]) > 0
        assert b["aux_owner"].shape == (len(b["returns"]), MAX_PLANETS)
        assert b["aux_ships"].shape == (len(b["returns"]), MAX_PLANETS)
        assert b["aux_valid"].shape == (len(b["returns"]), MAX_PLANETS)
        assert st["hero_slot"] == float(slot)
        if b["aux_valid"].any():
            valid_owners = b["aux_owner"][b["aux_valid"]]
            assert np.all((valid_owners >= 0) & (valid_owners <= 2))
            assert np.all(b["aux_ships"][b["aux_valid"]] >= 0.0)
        batches.append(b)

    rollout = {k: np.concatenate([x[k] for x in batches], axis=0) for k in batches[0].keys()}

    opt_cpu = build_optimizer(policy.parameters(), lr=1e-4, use_dml=False)
    before_cpu = [p.detach().clone() for p in policy.parameters()]
    upd_cpu = ppo_update(
        policy,
        teacher,
        opt_cpu,
        rollout,
        ppo_epochs=1,
        batch_size=16,
        clip_eps=0.15,
        kl_coef=0.06,
        fixed_anchor=fixed_anchor,
        anchor_kl_coef=0.025,
        aux_coef=0.15,
        use_dml=False,
        augment_d4=False,
    )
    assert np.isfinite(upd_cpu["loss"])
    assert upd_cpu["kl"] >= 0.0 and upd_cpu["anchor_kl"] >= 0.0
    assert any(not torch.allclose(a, b) for a, b in zip(before_cpu, policy.parameters()))

    if not _HAS_DML_GPU:
        pytest.skip("No DirectML GPU detected")

    dml_dev, is_dml = resolve_train_device("dml")
    assert is_dml
    pol_dml = copy.deepcopy(fixed_anchor).to(dml_dev)
    tea_dml = copy.deepcopy(fixed_anchor).to(dml_dev).eval()
    anc_dml = copy.deepcopy(fixed_anchor).to(dml_dev).eval()
    opt_dml = build_optimizer(pol_dml.parameters(), lr=1e-4, use_dml=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        upd_dml = ppo_update(
            pol_dml,
            tea_dml,
            opt_dml,
            rollout,
            ppo_epochs=1,
            batch_size=16,
            clip_eps=0.15,
            kl_coef=0.06,
            fixed_anchor=anc_dml,
            anchor_kl_coef=0.025,
            aux_coef=0.15,
            device=dml_dev,
            use_dml=True,
            augment_d4=False,
        )
    _assert_no_dml_fallback(caught)
    assert np.isfinite(upd_dml["loss"])
    assert upd_dml["kl"] >= 0.0 and upd_dml["anchor_kl"] >= 0.0
    cpu_sd = cpu_state_dict(pol_dml)
    assert all(v.device.type == "cpu" and torch.isfinite(v).all() for v in cpu_sd.values())


def test_soft_kd_and_supervised_loss_dml_zero_fallback():
    from distill_soft_kd import compute_soft_kd_loss
    from train_planet_transformer import _compute_batch_loss

    sim = FastOrbitSim.from_seed(seed=101, num_agents=2, max_steps=80)
    for _ in range(28):
        sim.step([[], []])
    obs = sim.to_observation(player=0)
    enc = extract_planet_features(obs, player=0, max_speed=sim.ship_speed)
    pad = pad_encoded_state(enc, MAX_PLANETS)

    B = 2
    pf = torch.from_numpy(np.stack([pad["planet_feats"]] * B))
    tf = torch.from_numpy(np.stack([pad["timeline_feats"]] * B))
    ef = torch.from_numpy(np.stack([pad["edge_feats"]] * B))
    le = torch.from_numpy(np.stack([pad["legal_edges"]] * B))
    om = torch.from_numpy(np.stack([pad["owned_mask"]] * B))
    pm = torch.from_numpy(np.stack([pad["pad_mask"]] * B))

    soft_q = torch.zeros((B, MAX_PLANETS, MAX_PLANETS + 1), dtype=torch.float32)
    soft_q[..., 0] = 0.70
    for i in range(len(enc.planet_ids)):
        legal_js = np.flatnonzero(enc.legal_edges[i])
        if len(legal_js) > 0:
            soft_q[:, i, 1 + int(legal_js[0])] = 0.30
    vtar = torch.tensor([1.0, -1.0], dtype=torch.float32)
    ao = torch.zeros((B, MAX_PLANETS), dtype=torch.long)
    ash = torch.full((B, MAX_PLANETS), 0.4, dtype=torch.float32)
    av = pm.clone()

    tcls = torch.zeros((B, MAX_PLANETS), dtype=torch.long)
    for i in range(len(enc.planet_ids)):
        legal_js = np.flatnonzero(enc.legal_edges[i])
        if enc.owned_mask[i] and len(legal_js) > 0:
            tcls[:, i] = 1 + int(legal_js[0])
    tfrac = torch.full((B, MAX_PLANETS), 0.75, dtype=torch.float32)

    # CPU checks
    m_cpu = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=True)
    kd_loss_c, kd_m_c = compute_soft_kd_loss(
        m_cpu, pf, tf, ef, le, om, pm, soft_q, vtar, ao, ash, av,
        midgame_weight=2.5, comet_weight=2.0, hold_boost=1.25, launch_boost=1.1,
    )
    bc_loss_c, bc_m_c = _compute_batch_loss(m_cpu, pf, tf, ef, le, om, pm, tcls, tfrac, vtar, launch_weight=1.2)
    assert torch.isfinite(kd_loss_c) and torch.isfinite(bc_loss_c)
    assert kd_m_c["kl"] >= 0.0 and bc_m_c["pol_loss"] >= 0.0

    if not _HAS_DML_GPU:
        pytest.skip("No DirectML GPU detected")

    dml_dev, _ = resolve_train_device("dml")
    m_dml = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=True).to(dml_dev)
    opt = build_optimizer(m_dml.parameters(), lr=2e-4, use_dml=True)

    pf_d, tf_d, ef_d, le_d, om_d, pm_d = (t.to(dml_dev) for t in (pf, tf, ef, le, om, pm))
    sq_d, vt_d, ao_d, ash_d, av_d = (t.to(dml_dev) for t in (soft_q, vtar, ao, ash, av))
    tc_d, tfr_d = tcls.to(dml_dev), tfrac.to(dml_dev)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.zero_grad(set_to_none=True)
        kd_loss_d, _ = compute_soft_kd_loss(
            m_dml, pf_d, tf_d, ef_d, le_d, om_d, pm_d, sq_d, vt_d, ao_d, ash_d, av_d,
            midgame_weight=2.5, comet_weight=2.0, hold_boost=1.25, launch_boost=1.1,
        )
        kd_loss_d.backward()
        torch.nn.utils.clip_grad_norm_(m_dml.parameters(), 1.0, foreach=False)
        opt.step()

        opt.zero_grad(set_to_none=True)
        bc_loss_d, _ = _compute_batch_loss(
            m_dml, pf_d, tf_d, ef_d, le_d, om_d, pm_d, tc_d, tfr_d, vt_d, launch_weight=1.2
        )
        bc_loss_d.backward()
        torch.nn.utils.clip_grad_norm_(m_dml.parameters(), 1.0, foreach=False)
        opt.step()
        assert torch.isfinite(kd_loss_d).item() and torch.isfinite(bc_loss_d).item()
    _assert_no_dml_fallback(caught)
