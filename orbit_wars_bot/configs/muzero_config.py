"""MuZero configuration for Orbit Wars (pure-Python ptree by default)."""

from __future__ import annotations

from typing import Tuple

from easydict import EasyDict


def make_muzero_config(
    *,
    battle_mode: str = "play_with_bot_mode",
    bot_action_type: str = "heuristic",
    mcts_ctree: bool = False,
    collector_env_num: int = 4,
    evaluator_env_num: int = 2,
    n_episode: int = 4,
    num_simulations: int = 25,
    update_per_collect: int = 20,
    batch_size: int = 64,
    max_steps: int = 500,
    decision_interval: int = 4,
    reanalyze_ratio: float = 0.0,
    cuda: bool = False,
    env_manager_type: str = "base",
    exp_name: str | None = None,
    seed: int = 0,
) -> Tuple[EasyDict, EasyDict]:
    """Build ``(main_config, create_config)`` for LightZero ``train_muzero``."""
    if exp_name is None:
        tree_tag = "ctree" if mcts_ctree else "ptree"
        mode_tag = "sp" if battle_mode == "self_play_mode" else "bot"
        exp_name = (
            f"data_mz_{tree_tag}/orbit_wars_mz_{mode_tag}_ns{num_simulations}_"
            f"di{decision_interval}_seed{seed}"
        )

    main_config = EasyDict(
        dict(
            exp_name=exp_name,
            env=dict(
                env_id="orbit_wars",
                board_size=16,
                action_space_size=16,
                battle_mode=battle_mode,
                battle_mode_in_simulation_env=battle_mode,
                bot_action_type=bot_action_type,
                max_steps=max_steps,
                decision_interval=decision_interval,
                channel_last=False,
                scale=True,
                stop_value=1.0,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=evaluator_env_num,
                manager=dict(shared_memory=False),
            ),
            policy=dict(
                mcts_ctree=mcts_ctree,
                model=dict(
                    observation_shape=(12, 16, 16),
                    action_space_size=16,
                    image_channel=12,
                    frame_stack_num=1,
                    num_res_blocks=2,
                    num_channels=32,
                    reward_head_channels=8,
                    value_head_channels=8,
                    policy_head_channels=8,
                    reward_head_hidden_channels=[32],
                    value_head_hidden_channels=[32],
                    policy_head_hidden_channels=[32],
                    reward_support_range=(-10.0, 11.0, 1.0),
                    value_support_range=(-10.0, 11.0, 1.0),
                    downsample=False,
                    norm_type="BN",
                ),
                model_path=None,
                cuda=cuda,
                env_type="board_games",
                action_type="varied_action_space",
                battle_mode=battle_mode,
                game_segment_length=25,
                update_per_collect=update_per_collect,
                batch_size=batch_size,
                optim_type="Adam",
                piecewise_decay_lr_scheduler=False,
                learning_rate=1e-3,
                grad_clip_value=1.0,
                num_simulations=num_simulations,
                reanalyze_ratio=reanalyze_ratio,
                td_steps=20,
                num_unroll_steps=5,
                discount_factor=0.99,
                n_episode=n_episode,
                eval_freq=100,
                replay_buffer_size=int(2e4),
                random_collect_episode_num=0,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
            ),
        )
    )

    create_config = EasyDict(
        dict(
            env=dict(
                type="orbit_wars",
                import_names=["orbit_wars_bot.lightzero_env"],
            ),
            env_manager=dict(type=env_manager_type),
            policy=dict(
                type="muzero",
                import_names=["lzero.policy.muzero"],
            ),
        )
    )
    return main_config, create_config


main_config, create_config = make_muzero_config()
