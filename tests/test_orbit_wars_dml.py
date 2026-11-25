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
            opponent_kind="pool" if slot == 1 else "heuristic",
            seed=77,
            max_steps=20,
            force_all_in=True,
            hero_slot=slot,
            rollout_d4=True,
            opp_temperature=0.55 if slot == 1 else 0.0,
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
    kd_loss_t, kd_m_t = compute_soft_kd_loss(
        m_cpu, pf, tf, ef, le, om, pm, soft_q, vtar, ao, ash, av,
        midgame_weight=2.5, comet_weight=2.0, hold_boost=1.25, launch_boost=1.1,
        return_tensor_metrics=True,
    )
    assert torch.allclose(kd_loss_c, kd_loss_t, atol=1e-6)
    assert abs(kd_m_c["kl"] - float(kd_m_t["kl"].item())) < 1e-6
    assert abs(kd_m_c["acc_all"] - float(kd_m_t["all_hits"].item()) / max(1.0, float(kd_m_t["all_tot"].item()))) < 1e-6
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
        kd_loss_d, kd_m_d = compute_soft_kd_loss(
            m_dml, pf_d, tf_d, ef_d, le_d, om_d, pm_d, sq_d, vt_d, ao_d, ash_d, av_d,
            midgame_weight=2.5, comet_weight=2.0, hold_boost=1.25, launch_boost=1.1,
            return_tensor_metrics=True,
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
        assert torch.isfinite(kd_loss_d).item() and torch.isfinite(kd_m_d["kl"]).item() and torch.isfinite(bc_loss_d).item()
    _assert_no_dml_fallback(caught)


def test_ensemble_labeler_gpu_and_cpu():
    from distill_ensemble_4m import label_boards_with_ensemble_gpu

    sim = FastOrbitSim.from_seed(seed=202, num_agents=2, max_steps=60)
    for _ in range(25):
        sim.step([[], []])
    obs = sim.to_observation(player=0)
    enc = extract_planet_features(obs, player=0, max_speed=sim.ship_speed)
    pad = pad_encoded_state(enc, MAX_PLANETS)
    n_pl = len(enc.planet_ids)

    data = {
        "planet_feats": np.stack([pad["planet_feats"].astype(np.float16)] * 2, axis=0),
        "timeline_feats": np.stack([pad["timeline_feats"].astype(np.float16)] * 2, axis=0),
        "edge_feats": np.stack([pad["edge_feats"].astype(np.float16)] * 2, axis=0),
        "legal_edges": np.stack([pad["legal_edges"]] * 2, axis=0),
        "owned_mask": np.stack([pad["owned_mask"]] * 2, axis=0),
        "pad_mask": np.stack([pad["pad_mask"]] * 2, axis=0),
        "val_targets": np.array([1.0, -1.0], dtype=np.float32),
    }

    dev = resolve_train_device("dml" if _HAS_DML_GPU else "cpu")[0]
    t1 = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=False).to(dev).eval()
    t2 = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=True).to(dev).eval()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labeled = label_boards_with_ensemble_gpu(
            data, [(t1, 0.55), (t2, 0.45)], device=dev, batch_size=2, val_blend_outcome=0.5
        )
    if _HAS_DML_GPU:
        _assert_no_dml_fallback(caught)

    sp = labeled["soft_probs"].astype(np.float32)
    assert sp.shape == (2, MAX_PLANETS, MAX_PLANETS + 1)
    assert np.all(np.isfinite(sp))
    row_sums = sp[:, :n_pl, : n_pl + 1].sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=2e-3)
    assert np.all(np.isfinite(labeled["val_targets"]))
    assert np.all((labeled["val_targets"] >= -1.05) & (labeled["val_targets"] <= 1.05))


def test_gen11_search_allin_eval_specs_and_contiguous_val(tmp_path: Path):
    from distill_soft_kd import train_soft_kd
    from evaluate import _parse_threshold_spec, resolve_agent
    from orbit_wars_bot.planet_transformer import SearchPlanetTransformerAgent

    sim = FastOrbitSim.from_seed(seed=303, num_agents=2, max_steps=60)
    for _ in range(15):
        sim.step([[], []])
    obs = sim.to_observation(player=0)
    enc = extract_planet_features(obs, player=0, max_speed=sim.ship_speed)
    pad = pad_encoded_state(enc, MAX_PLANETS)

    m = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=True)
    ckpt = tmp_path / "tiny_gen11.pt"
    torch.save({"model_config": m.config, "model_state_dict": m.state_dict()}, ckpt)

    # 1. All-in + neural_mid_replan Search agent
    s_ag = SearchPlanetTransformerAgent(
        ckpt,
        launch_threshold=0.05,
        use_d4_tta=True,
        force_all_in=True,
        neural_mid_replan=True,
        rollout_horizon=6,
        time_budget_sec=0.35,
    )
    moves = s_ag.act(obs, sim.to_config())
    assert isinstance(moves, list)
    gar_by_pid = {int(row[0]): int(row[5]) for row in obs["planets"] if int(row[1]) == 0}
    for mv in moves:
        assert int(mv[2]) == gar_by_pid[int(mv[0])]

    # 2. evaluate.py threshold suffix specs
    assert _parse_threshold_spec("pt_d4_allin", f"pt_d4_allin_t53:{ckpt}") == (True, 0.53, str(ckpt))
    assert _parse_threshold_spec("pt_search_d4_allin", "pt_search_d4_allin") == (True, 0.50, None)
    assert callable(resolve_agent(f"pt_d4_allin_t53:{ckpt}"))
    assert callable(resolve_agent(f"pt_ens_allin_t56:{ckpt},{ckpt}"))
    assert callable(resolve_agent(f"pt_search_d4_allin_t53:{ckpt}"))

    # 3. contiguous_val split on synthetic Soft-KD data
    n_samples = 12
    soft_q = np.zeros((MAX_PLANETS, MAX_PLANETS + 1), dtype=np.float16)
    soft_q[:, 0] = 1.0
    data = {
        "planet_feats": np.stack([pad["planet_feats"]] * n_samples, axis=0),
        "timeline_feats": np.stack([pad["timeline_feats"]] * n_samples, axis=0),
        "edge_feats": np.stack([pad["edge_feats"]] * n_samples, axis=0),
        "legal_edges": np.stack([pad["legal_edges"]] * n_samples, axis=0),
        "owned_mask": np.stack([pad["owned_mask"]] * n_samples, axis=0),
        "pad_mask": np.stack([pad["pad_mask"]] * n_samples, axis=0),
        "soft_probs": np.stack([soft_q] * n_samples, axis=0),
        "val_targets": np.linspace(-1.0, 1.0, n_samples, dtype=np.float32),
        "aux_owner": np.zeros((n_samples, MAX_PLANETS), dtype=np.int64),
        "aux_ships": np.zeros((n_samples, MAX_PLANETS), dtype=np.float32),
        "aux_valid": np.stack([pad["pad_mask"]] * n_samples, axis=0),
    }
    out_ckpt = tmp_path / "out_contiguous.pt"
    saved = train_soft_kd(
        data,
        resume_from=str(ckpt),
        out_path=out_ckpt,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        contiguous_val=True,
        device="cpu",
    )
    assert saved.is_file()


def test_gen12_ptv2_d4_equivariance_factorized_head_and_dml(tmp_path: Path):
    from distill_soft_kd import compute_soft_kd_loss
    from orbit_wars_bot.planet_transformer import (
        TIMELINE_FEAT_DIM_V2,
        EnsemblePlanetTransformerAgent,
        apply_d4_planet_feats,
        apply_d4_timeline_feats,
    )

    sim = FastOrbitSim.from_seed(seed=404, num_agents=2, max_steps=80)
    for _ in range(15):
        sim.step([[], []])
    obs = sim.to_observation(player=0)

    enc_v1 = extract_planet_features(obs, player=0, max_speed=sim.ship_speed, feature_version=1)
    enc_v2 = extract_planet_features(obs, player=0, max_speed=sim.ship_speed, feature_version=2)
    assert enc_v2.timeline_feats.shape == (len(enc_v2.planet_ids), 20, TIMELINE_FEAT_DIM_V2)
    assert np.allclose(enc_v2.timeline_feats[:, :, :8], enc_v1.timeline_feats, atol=1e-6)

    # Exact D4 equivariance on rotated/flipped board observation + group closure
    def _xform_xy(x: float, y: float, k_rot: int, flip_x: bool):
        dx, dy = x - 50.0, y - 50.0
        if flip_x:
            dx = -dx
        k = k_rot % 4
        if k == 1:
            dx, dy = -dy, dx
        elif k == 2:
            dx, dy = -dx, -dy
        elif k == 3:
            dx, dy = dy, -dx
        return 50.0 + dx, 50.0 + dy

    obs_rot = copy.deepcopy(obs)
    for key in ("planets", "initial_planets"):
        for row in obs_rot.get(key, []) or []:
            row[2], row[3] = _xform_xy(float(row[2]), float(row[3]), k_rot=1, flip_x=False)
    for grp in obs_rot.get("comets", []) or []:
        for pth in grp.get("paths", []) or []:
            for pt in pth:
                pt[0], pt[1] = _xform_xy(float(pt[0]), float(pt[1]), k_rot=1, flip_x=False)
    enc_rot = extract_planet_features(obs_rot, player=0, max_speed=sim.ship_speed, feature_version=2)
    assert np.allclose(
        apply_d4_planet_feats(enc_v2.planet_feats, k_rot=1, flip_x=False),
        enc_rot.planet_feats,
        atol=1e-4,
    )
    assert np.allclose(
        apply_d4_timeline_feats(enc_v2.timeline_feats, k_rot=1, flip_x=False),
        enc_rot.timeline_feats,
        atol=1e-4,
    )
    # Full D4 group closure + radial invariant check across all 8 symmetries
    for k_rot in range(4):
        for flip_x in (False, True):
            tf_t = apply_d4_timeline_feats(enc_v2.timeline_feats, k_rot=k_rot, flip_x=flip_x)
            assert np.allclose(tf_t[..., :8], enc_v2.timeline_feats[..., :8], atol=1e-6)
            assert np.allclose(tf_t[..., 10:], enc_v2.timeline_feats[..., 10:], atol=1e-6)
            valid_pos = tf_t[..., 11] > 0.5
            r_calc = np.hypot(tf_t[..., 8] * 50.0, tf_t[..., 9] * 50.0) / 70.0
            assert np.allclose(r_calc[valid_pos], tf_t[..., 10][valid_pos], atol=1e-5)
    assert np.allclose(
        apply_d4_timeline_feats(apply_d4_timeline_feats(enc_v2.timeline_feats, k_rot=1, flip_x=True), k_rot=1, flip_x=True),
        enc_v2.timeline_feats,
        atol=1e-6,
    )

    # Factorized launch invariance + zero-legal-edge safety
    pad_v2 = pad_encoded_state(enc_v2, MAX_PLANETS)
    m_v2 = PlanetTimelineTransformer(
        timeline_dim=TIMELINE_FEAT_DIM_V2,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dim_ff=64,
        use_aux_future=True,
        feature_version=2,
        arch_version=2,
    ).eval()
    pf = torch.from_numpy(pad_v2["planet_feats"][None])
    tf = torch.from_numpy(pad_v2["timeline_feats"][None])
    ef = torch.from_numpy(pad_v2["edge_feats"][None])
    le = torch.from_numpy(pad_v2["legal_edges"][None])
    pm = torch.from_numpy(pad_v2["pad_mask"][None])

    n_pl = len(enc_v2.planet_ids)
    owned_rows = [i for i in range(n_pl) if enc_v2.owned_mask[i] and np.sum(enc_v2.legal_edges[i]) >= 2]
    assert owned_rows, "Expected an owned planet with >=2 legal targets"
    row_i = owned_rows[0]

    le_sub = le.clone()
    legal_cols = [j for j in range(n_pl) if bool(le[0, row_i, j])]
    le_sub[0, row_i, legal_cols[1:]] = False
    le_zero = le.clone()
    le_zero[0, row_i, :] = False

    with torch.no_grad():
        log_full, _, _ = m_v2(pf, tf, ef, pm, legal_edges=le)
        log_sub, _, _ = m_v2(pf, tf, ef, pm, legal_edges=le_sub)
        log_zero, _, _ = m_v2(pf, tf, ef, pm, legal_edges=le_zero)

    def _masked_probs(log_t: torch.Tensor, mask_e: torch.Tensor) -> torch.Tensor:
        hold_ok = torch.ones((1, MAX_PLANETS, 1), dtype=torch.bool)
        act_m = torch.cat([hold_ok, mask_e], dim=-1)
        return F.softmax(log_t.masked_fill(~act_m, -1e9), dim=-1)

    p_full = _masked_probs(log_full, le)
    p_sub = _masked_probs(log_sub, le_sub)
    p_zero = _masked_probs(log_zero, le_zero)
    assert torch.all(torch.isfinite(p_full)) and torch.all(torch.isfinite(p_zero))
    assert torch.allclose(p_full[0, :n_pl].sum(dim=-1), torch.ones(n_pl), atol=1e-5)
    assert abs((1.0 - float(p_full[0, row_i, 0])) - (1.0 - float(p_sub[0, row_i, 0]))) < 1e-5
    assert abs(float(p_zero[0, row_i, 0]) - 1.0) < 1e-5

    # Backward compatibility: v1 model consumes 14-ch tensor & Ensemble mixes v2 + v1
    m_v1 = PlanetTimelineTransformer(d_model=32, n_heads=2, n_layers=1, dim_ff=64, use_aux_future=False).eval()
    with torch.no_grad():
        l1, f1, v1 = m_v1(pf, tf, ef, pm, legal_edges=le)
    assert torch.all(torch.isfinite(l1)) and torch.all(torch.isfinite(f1)) and torch.all(torch.isfinite(v1))

    ck2 = tmp_path / "tiny_v2.pt"
    ck1 = tmp_path / "tiny_v1.pt"
    torch.save({"model_config": m_v2.config, "model_state_dict": m_v2.state_dict()}, ck2)
    torch.save({"model_config": m_v1.config, "model_state_dict": m_v1.state_dict()}, ck1)
    ens = EnsemblePlanetTransformerAgent([ck2, ck1], use_d4_tta=True)
    assert ens.feature_version == 2
    mv_ens = ens.act(obs, sim.to_config())
    assert isinstance(mv_ens, list)

    # Soft-KD binary launch-KL loss + zero-fallback DirectML backward step
    dev, is_dml = resolve_train_device("dml" if _HAS_DML_GPU else "cpu")
    m_dev = PlanetTimelineTransformer(
        timeline_dim=TIMELINE_FEAT_DIM_V2,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dim_ff=64,
        use_aux_future=True,
        feature_version=2,
        arch_version=2,
    ).to(dev)
    opt = build_optimizer(m_dev.parameters(), lr=1e-3, use_dml=is_dml)
    soft_q = np.zeros((1, MAX_PLANETS, MAX_PLANETS + 1), dtype=np.float32)
    soft_q[:, :, 0] = 0.7
    soft_q[:, row_i, 1 + legal_cols[0]] = 0.3

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.zero_grad(set_to_none=True)
        loss, m_dict = compute_soft_kd_loss(
            m_dev,
            pf.to(dev),
            tf.to(dev),
            ef.to(dev),
            le.to(dev),
            torch.from_numpy(pad_v2["owned_mask"][None]).to(dev),
            pm.to(dev),
            torch.from_numpy(soft_q).to(dev),
            torch.tensor([1.0], dtype=torch.float32, device=dev),
            torch.zeros((1, MAX_PLANETS), dtype=torch.long, device=dev),
            torch.zeros((1, MAX_PLANETS), dtype=torch.float32, device=dev),
            pm.to(dev),
            launch_bce_weight=0.50,
            return_tensor_metrics=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_dev.parameters(), 1.0, foreach=not is_dml)
        opt.step()
        assert torch.isfinite(loss).item()
        for k in ("l_tp", "l_fp", "l_fn"):
            assert k in m_dict and torch.isfinite(m_dict[k]).item()
    if is_dml:
        _assert_no_dml_fallback(caught)

    # 4. Slot-1 missing step preservation check in Soft-KD feature extraction
    from kaggle_environments import make

    k_env = make("orbit_wars", configuration={"seed": 200}, debug=False)
    k_steps = k_env.run(["starter", "starter"])
    t_chk = min(10, len(k_steps) - 1)
    assert k_steps[t_chk][1]["observation"].get("step") is None
    obs_s1 = dict(k_steps[t_chk][1]["observation"])
    obs_s1["player"] = 1
    obs_s1["step"] = int(k_steps[t_chk][0]["observation"].get("step", t_chk))
    enc_s1 = extract_planet_features(obs_s1, player=1)
    assert enc_s1.current_step == t_chk
    assert abs(float(enc_s1.planet_feats[0, 16]) * 500.0 - t_chk) < 1e-4


def test_gen12_fast_numpy_features_batched_rollouts_and_cached_ppo():
    """Verify bit-exact parity of shared-physics NumPy features, compute_frac=False, B=4 rollouts, and cached PPO."""
    from orbit_wars_bot.heuristic_bot import heuristic_moves
    from orbit_wars_bot.physics import (
        build_comet_lookup,
        predict_all_planet_tracks,
        predict_all_planet_tracks_arrays,
        predict_fleet_arrivals,
        predict_fleet_arrivals_vectorized,
    )
    from orbit_wars_bot.planet_transformer import (
        TIMELINE_FEAT_DIM_V2,
        TIMELINE_HORIZON,
        build_shared_planet_physics,
    )
    from rl_finetune_transformer import collect_episodes_batched_2p, ppo_update

    # 1. Vectorized tracks & swept-pair fleet arrivals parity across live midgame w/ fleets + comets
    sim = FastOrbitSim.from_seed(seed=990, num_agents=2, max_steps=70)
    for _ in range(55):
        if sim.done:
            break
        sim.step([heuristic_moves(sim, 0), heuristic_moves(sim, 1)])
    obs0 = sim.to_observation(player=0)
    obs1 = sim.to_observation(player=1)
    planets = [list(p) for p in obs0["planets"]]
    fleets = obs0["fleets"]
    init_by_id = {int(p[0]): p for p in obs0["initial_planets"]}
    clookup = build_comet_lookup(obs0["comets"])
    step = int(obs0["step"])
    av = float(obs0["angular_velocity"])

    ref_tracks = predict_all_planet_tracks(planets, TIMELINE_HORIZON, step, av, init_by_id, clookup)
    tr_x, tr_y, _, vec_tracks = predict_all_planet_tracks_arrays(planets, TIMELINE_HORIZON, step, av, init_by_id, clookup)
    ref_arr = predict_fleet_arrivals(
        fleets, planets, step, av, init_by_id, comet_lookup=clookup, tracks=ref_tracks, max_horizon=TIMELINE_HORIZON
    )
    vec_arr = predict_fleet_arrivals_vectorized(fleets, planets, tr_x, tr_y, max_horizon=TIMELINE_HORIZON)
    assert vec_arr == ref_arr
    assert set(vec_tracks.keys()) == set(ref_tracks.keys())

    # 2. Shared-physics 2P extraction parity + compute_frac=False exact match + broadcast (1, N, N, E)
    sh = build_shared_planet_physics(obs0, max_speed=sim.ship_speed)
    for pid, ob in ((0, obs0), (1, obs1)):
        e_solo = extract_planet_features(ob, player=pid, max_speed=sim.ship_speed, feature_version=2)
        e_sh = extract_planet_features(ob, player=pid, max_speed=sim.ship_speed, feature_version=2, shared_physics=sh)
        assert np.allclose(e_solo.planet_feats, e_sh.planet_feats, atol=1e-6)
        assert np.allclose(e_solo.timeline_feats, e_sh.timeline_feats, atol=1e-6)
        assert np.allclose(e_solo.edge_feats, e_sh.edge_feats, atol=1e-6)
        assert np.array_equal(e_solo.legal_edges, e_sh.legal_edges)

    m = PlanetTimelineTransformer(
        timeline_dim=TIMELINE_FEAT_DIM_V2,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dim_ff=64,
        use_aux_future=True,
        feature_version=2,
        arch_version=2,
    ).eval()
    e0 = extract_planet_features(obs0, player=0, max_speed=sim.ship_speed, feature_version=2, shared_physics=sh)
    n_p = len(e0.planet_ids)
    pf = torch.from_numpy(e0.planet_feats[None]).repeat(4, 1, 1)
    tf = torch.from_numpy(e0.timeline_feats[None]).repeat(4, 1, 1, 1)
    ef_full = torch.from_numpy(e0.edge_feats[None]).repeat(4, 1, 1, 1)
    ef_bcast = torch.from_numpy(e0.edge_feats[None])
    pm = torch.ones((4, n_p), dtype=torch.bool)
    le = torch.from_numpy(e0.legal_edges[None]).repeat(4, 1, 1)
    with torch.no_grad():
        l1, f1, v1 = m(pf, tf, ef_full, pm, legal_edges=le, compute_frac=True)
        l2, f2, v2 = m(pf, tf, ef_bcast, pm, legal_edges=le, compute_frac=False)
    assert f1 is not None and f2 is None
    assert torch.allclose(l1, l2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)

    # 3. Lockstep batched 2P rollouts (float16 features) + cached ref-logp PPO update
    res = collect_episodes_batched_2p(
        m,
        m,
        specs=[(777, 0), (777, 1), (778, 0), (778, 1)],
        temperature=0.55,
        max_steps=30,
        force_all_in=True,
        opp_temperature=0.45,
    )
    assert len(res) == 4
    batches = [b for b, _ in res if len(b["returns"]) > 0]
    assert batches and batches[0]["planet_feats"].dtype == np.float16
    rollout = {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}

    dev, is_dml = resolve_train_device("auto")
    pol = copy.deepcopy(m).to(dev)
    anc = copy.deepcopy(m).to(dev).eval()
    fix = copy.deepcopy(m).to(dev).eval()
    opt = build_optimizer(pol.parameters(), lr=1e-4, use_dml=is_dml)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = ppo_update(
            pol,
            anc,
            opt,
            rollout,
            ppo_epochs=2,
            batch_size=16,
            fixed_anchor=fix,
            anchor_kl_coef=0.05,
            aux_coef=0.15,
            device=dev,
            use_dml=is_dml,
            augment_d4=False,
        )
    if is_dml:
        _assert_no_dml_fallback(caught)
    assert all(np.isfinite(v) for v in stats.values())

