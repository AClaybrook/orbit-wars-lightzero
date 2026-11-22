"""Tests for LightZero environment integration, training smoke tests, and Kaggle submission."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from ding.utils import ENV_REGISTRY
from kaggle_environments import make

import orbit_wars_bot.lightzero_env  # noqa: F401
from orbit_wars_bot.lightzero_compat import is_ctree_muzero_available, setup_lightzero
from orbit_wars_bot.lightzero_env import BOARD_BUF_SIZE, OrbitWarsLightZeroEnv
from orbit_wars_bot.mcts_agent import OnlineMCTSAgent
import submission


def test_env_registered_and_bot_mode_rollout():
    cls = ENV_REGISTRY.get("orbit_wars")
    assert cls is OrbitWarsLightZeroEnv

    env = OrbitWarsLightZeroEnv(
        dict(
            battle_mode="play_with_bot_mode",
            bot_action_type="starter",
            max_steps=60,
            decision_interval=6,
        )
    )
    env.seed(7)
    obs = env.reset()
    assert obs["observation"].shape == (12, 16, 16)
    assert obs["action_mask"].shape == (16,)
    assert obs["board"].shape == (BOARD_BUF_SIZE,)
    assert obs["to_play"] == -1

    done = False
    steps = 0
    while not done and steps < 30:
        act = env.legal_actions[0]
        ts = env.step(act)
        done = ts.done
        steps += 1
    assert done
    assert "eval_episode_return" in ts.info


def test_env_self_play_mode_and_board_state_restore():
    env = OrbitWarsLightZeroEnv(
        dict(
            battle_mode="self_play_mode",
            max_steps=50,
            decision_interval=5,
        )
    )
    env.seed(13)
    obs0 = env.reset()
    assert obs0["to_play"] == 1
    assert obs0["current_player_index"] == 0

    # Half-turn 1 (Player 1 queues action) -> switches to Player 2 without advancing sim
    ts1 = env.step(1)
    assert not ts1.done
    assert ts1.obs["to_play"] == 2
    assert ts1.obs["current_player_index"] == 1
    assert env.sim.step_count == 0

    # Clone into a fresh simulation env mid-half-turn (as AlphaZero MCTS does!)
    sim_env = OrbitWarsLightZeroEnv(
        dict(battle_mode="self_play_mode", max_steps=50, decision_interval=5)
    )
    restored = sim_env.reset(
        start_player_index=ts1.obs["current_player_index"],
        init_state=ts1.obs["board"],
    )
    assert restored["to_play"] == 2
    assert sim_env._pending_p0_action == 1

    # Half-turn 2 (Player 2 submits) -> advances sim simultaneously
    ts2 = env.step(0)
    assert ts2.obs["to_play"] == 1
    assert env.sim.step_count == 5

    # Run until done
    while not ts2.done:
        ts2 = env.step(env.legal_actions[0])
    assert ts2.done
    assert "eval_episode_return" in ts2.info


def test_alphazero_smoke_training_and_checkpoint_loading(tmp_path: Path, monkeypatch):
    """Run a tiny AlphaZero (ptree) training loop and load the saved checkpoint in OnlineMCTSAgent."""
    monkeypatch.chdir(tmp_path)
    setup_lightzero()
    from lzero.entry import train_alphazero
    from orbit_wars_bot.configs.alphazero_config import make_alphazero_config

    exp_dir = tmp_path / "az_smoke"
    main_cfg, create_cfg = make_alphazero_config(
        battle_mode="play_with_bot_mode",
        bot_action_type="starter",
        mcts_ctree=False,
        collector_env_num=1,
        evaluator_env_num=1,
        n_episode=1,
        num_simulations=4,
        update_per_collect=2,
        batch_size=4,
        max_steps=40,
        decision_interval=8,
        cuda=False,
        env_manager_type="base",
        exp_name="az_smoke",
        seed=0,
    )
    main_cfg.policy.eval_freq = 2
    main_cfg.policy.model.num_res_blocks = 1
    main_cfg.policy.model.num_channels = 16

    train_alphazero(
        [main_cfg, create_cfg],
        seed=0,
        max_train_iter=3,
        max_env_step=16,
    )

    ckpts = list(exp_dir.rglob("*.pth.tar"))
    assert ckpts, f"Expected checkpoint under {exp_dir}"

    # Verify OnlineMCTSAgent loads the trained AlphaZero checkpoint and runs inference
    agent = OnlineMCTSAgent(
        checkpoint_path=str(ckpts[0]),
        num_simulations=6,
        time_budget_sec=0.25,
    )
    assert agent.has_neural_model

    env = make("orbit_wars", configuration={"seed": 5, "episodeSteps": 40})
    obs = env.reset(2)
    t0 = time.perf_counter()
    moves = agent.act(obs[0].observation, env.configuration)
    elapsed = time.perf_counter() - t0
    assert isinstance(moves, list)
    assert elapsed < 0.5


def test_muzero_smoke_training_ptree(tmp_path: Path, monkeypatch):
    """Run a tiny MuZero (pure-Python ptree) smoke test."""
    monkeypatch.chdir(tmp_path)
    setup_lightzero()
    from lzero.entry import train_muzero
    from orbit_wars_bot.configs.muzero_config import make_muzero_config

    exp_dir = tmp_path / "mz_smoke"
    main_cfg, create_cfg = make_muzero_config(
        battle_mode="play_with_bot_mode",
        bot_action_type="starter",
        mcts_ctree=False,
        collector_env_num=1,
        evaluator_env_num=1,
        n_episode=1,
        num_simulations=4,
        update_per_collect=2,
        batch_size=4,
        max_steps=40,
        decision_interval=8,
        cuda=False,
        env_manager_type="base",
        exp_name="mz_smoke",
        seed=0,
    )
    main_cfg.policy.eval_freq = 2
    main_cfg.policy.game_segment_length = 6
    main_cfg.policy.td_steps = 3
    main_cfg.policy.num_unroll_steps = 2
    main_cfg.policy.model.num_res_blocks = 1
    main_cfg.policy.model.num_channels = 16

    train_muzero(
        [main_cfg, create_cfg],
        seed=0,
        max_train_iter=3,
        max_env_step=16,
    )
    ckpts = list(exp_dir.rglob("*.pth.tar"))
    assert ckpts, f"Expected MuZero checkpoint under {exp_dir}"

    # Verify OnlineMCTSAgent also loads MuZero checkpoints cleanly
    agent = OnlineMCTSAgent(
        checkpoint_path=str(ckpts[0]),
        num_simulations=4,
        time_budget_sec=0.25,
    )
    assert agent.has_neural_model


@pytest.mark.skipif(not is_ctree_muzero_available(), reason="C++ ctree_muzero not built")
def test_gumbel_muzero_smoke_training_ctree(tmp_path: Path, monkeypatch):
    """Run a tiny Gumbel MuZero (C++ ctree) smoke test when ctree is built."""
    monkeypatch.chdir(tmp_path)
    setup_lightzero()
    from lzero.entry import train_muzero
    from orbit_wars_bot.configs.gumbel_muzero_config import make_gumbel_muzero_config

    exp_dir = tmp_path / "gmz_smoke"
    main_cfg, create_cfg = make_gumbel_muzero_config(
        battle_mode="play_with_bot_mode",
        bot_action_type="starter",
        mcts_ctree=True,
        collector_env_num=1,
        evaluator_env_num=1,
        n_episode=1,
        num_simulations=6,
        max_num_considered_actions=4,
        update_per_collect=2,
        batch_size=4,
        max_steps=40,
        decision_interval=8,
        cuda=False,
        env_manager_type="base",
        exp_name="gmz_smoke",
        seed=0,
    )
    main_cfg.policy.eval_freq = 2
    main_cfg.policy.game_segment_length = 6
    main_cfg.policy.td_steps = 3
    main_cfg.policy.num_unroll_steps = 2
    main_cfg.policy.model.num_res_blocks = 1
    main_cfg.policy.model.num_channels = 16

    train_muzero(
        [main_cfg, create_cfg],
        seed=0,
        max_train_iter=3,
        max_env_step=16,
    )
    assert list(exp_dir.rglob("*.pth.tar"))


def test_submission_agent_kaggle_compatibility_and_budget():
    """Verify submission.agent runs cleanly in official kaggle_environments under 500ms/turn."""
    env = make("orbit_wars", configuration={"seed": 77, "episodeSteps": 40})
    obs = env.reset(2)
    worst_ms = 0.0
    for _ in range(15):
        t0 = time.perf_counter()
        a0 = submission.agent(obs[0].observation, env.configuration)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        worst_ms = max(worst_ms, dt_ms)
        obs = env.step([a0, []])
        if env.done:
            break
    assert worst_ms < 500.0, f"Turn latency {worst_ms:.1f}ms exceeded 500ms Kaggle budget"
