"""AlphaZero configuration for Orbit Wars (pure-Python ptree by default)."""

from __future__ import annotations

from typing import Tuple

from easydict import EasyDict


def make_alphazero_config(
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
    cuda: bool = False,
    env_manager_type: str = "base",
    exp_name: str | None = None,
    seed: int = 0,
) -> Tuple[EasyDict, EasyDict]:
    """Build ``(main_config, create_config)`` for LightZero ``train_alphazero``."""
    sim_cfg_type = "self_play" if battle_mode == "self_play_mode" else "play_with_bot"
    if exp_name is None:
        tree_tag = "ctree" if mcts_ctree else "ptree"
        mode_tag = "sp" if battle_mode == "self_play_mode" else "bot"
        exp_name = (
            f"data_az_{tree_tag}/orbit_wars_az_{mode_tag}_ns{num_simulations}_"
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
                prob_random_agent=0.0,
                prob_expert_agent=0.0,
                agent_vs_human=False,
                alphazero_mcts_ctree=mcts_ctree,
                stop_value=1.0,
                collector_env_num=collector_env_num,
                evaluator_env_num=evaluator_env_num,
                n_evaluator_episode=evaluator_env_num,
                manager=dict(shared_memory=False),
            ),
            policy=dict(
                mcts_ctree=mcts_ctree,
                simulation_env_id="orbit_wars",
                simulation_env_config_type=sim_cfg_type,
                model=dict(
                    observation_shape=(12, 16, 16),
                    action_space_size=16,
                    num_res_blocks=2,
                    num_channels=32,
                    value_head_channels=8,
                    policy_head_channels=8,
                    value_head_hidden_channels=[32],
                    policy_head_hidden_channels=[32],
                    downsample=False,
                ),
                cuda=cuda,
                board_size=16,
                update_per_collect=update_per_collect,
                batch_size=batch_size,
                optim_type="Adam",
                piecewise_decay_lr_scheduler=False,
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip_value=1.0,
                value_weight=1.0,
                entropy_weight=0.01,
                n_episode=n_episode,
                eval_freq=100,
                mcts=dict(
                    num_simulations=num_simulations,
                    max_moves=max_steps,
                    root_dirichlet_alpha=0.3,
                    root_noise_weight=0.25,
                    pb_c_base=19652,
                    pb_c_init=1.25,
                ),
                other=dict(
                    replay_buffer=dict(
                        replay_buffer_size=int(5e4),
                        save_episode=False,
                    )
                ),
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
                type="alphazero",
                import_names=["lzero.policy.alphazero"],
            ),
            collector=dict(
                type="episode_alphazero",
                import_names=["lzero.worker.alphazero_collector"],
            ),
            evaluator=dict(
                type="alphazero",
                import_names=["lzero.worker.alphazero_evaluator"],
            ),
        )
    )
    return main_config, create_config


main_config, create_config = make_alphazero_config()
