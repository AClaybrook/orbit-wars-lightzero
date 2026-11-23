"""LightZero / DI-engine environment wrapper for Orbit Wars.

Registered as ``@ENV_REGISTRY.register('orbit_wars')`` and compatible with:
- ``AlphaZeroPolicy`` (both ``mcts_ctree=False`` pure-Python ptree and C++ ctree)
- ``MuZeroPolicy`` (both ptree and ctree)
- ``GumbelMuZeroPolicy`` (ctree)

Supports both ``play_with_bot_mode`` and ``self_play_mode``. Simultaneous Orbit
Wars turns are handled cleanly in ``self_play_mode`` by alternating two
half-turns (Player 1 queues its macro-action, then Player 2 chooses its
macro-action on the same pre-tick board and both execute simultaneously).
"""

from __future__ import annotations

import copy
import pickle
import random
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.features import GRID_SIZE, NUM_CHANNELS, OBS_SHAPE, encode_observation
from orbit_wars_bot.heuristic_bot import (
    choose_heuristic_macro_action,
    heuristic_moves,
    random_moves,
    starter_moves,
)
from orbit_wars_bot.macro_actions import (
    NUM_MACRO_ACTIONS,
    build_tactical_context,
    execute_macro_action,
    get_legal_action_mask,
)

# Fixed-length buffer size for serialized board state in obs['board'] so
# DI-engine's default_collate / replay buffer can batch transitions uniformly.
BOARD_BUF_SIZE: int = 4096


@ENV_REGISTRY.register("orbit_wars")
class OrbitWarsLightZeroEnv(BaseEnv):
    """DI-engine / LightZero ``BaseEnv`` wrapping ``FastOrbitSim``."""

    config = dict(
        env_id="orbit_wars",
        board_size=GRID_SIZE,
        action_space_size=NUM_MACRO_ACTIONS,
        battle_mode="play_with_bot_mode",
        battle_mode_in_simulation_env="play_with_bot_mode",
        bot_action_type="heuristic",  # {'heuristic', 'starter', 'random'}
        num_agents=2,  # 2 for 1v1, 4 for 4P FFA (play_with_bot_mode)
        max_steps=500,
        decision_interval=4,
        ship_speed=6.0,
        comet_speed=4.0,
        channel_last=False,
        scale=True,
        prob_random_agent=0.0,
        prob_expert_agent=0.0,
        agent_vs_human=False,
        alphazero_mcts_ctree=False,
        stop_value=1.0,
        replay_path=None,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        default_cfg = self.default_config()
        if cfg is not None:
            default_cfg.update(cfg)
        self._cfg = default_cfg

        self.board_size = int(self._cfg.board_size)
        self.total_num_actions = int(self._cfg.action_space_size)
        self.battle_mode = str(self._cfg.battle_mode)
        assert self.battle_mode in ("play_with_bot_mode", "self_play_mode", "eval_mode"), (
            f"Unsupported battle_mode: {self.battle_mode!r}"
        )
        self.battle_mode_in_simulation_env = str(
            self._cfg.get("battle_mode_in_simulation_env", self.battle_mode)
        )
        if self.battle_mode_in_simulation_env == "eval_mode":
            self.battle_mode_in_simulation_env = "play_with_bot_mode"

        self.bot_action_type = str(self._cfg.bot_action_type)
        self.num_agents = int(self._cfg.get("num_agents", 2))
        if self.battle_mode == "self_play_mode":
            self.num_agents = 2
        self.max_steps = int(self._cfg.max_steps)
        self.decision_interval = max(1, int(self._cfg.decision_interval))
        self.ship_speed = float(self._cfg.ship_speed)
        self.comet_speed = float(self._cfg.comet_speed)
        self.channel_last = bool(self._cfg.channel_last)
        self.scale = bool(self._cfg.scale)
        self.prob_random_agent = float(self._cfg.prob_random_agent)
        self.prob_expert_agent = float(self._cfg.prob_expert_agent)
        self.alphazero_mcts_ctree = bool(self._cfg.alphazero_mcts_ctree)
        self.non_zero_sum = False

        self.players = [1, 2]
        self.start_player_index = 0
        self._current_player = 1
        self._pending_p0_action: Optional[int] = None
        self._ply = 0
        self._seed_val: Optional[int] = None
        self._rng = random.Random(0)
        self._episode_counter = 0

        obs_shape = (GRID_SIZE, GRID_SIZE, NUM_CHANNELS) if self.channel_last else OBS_SHAPE
        self._observation_space = gym.spaces.Box(
            low=-1.0, high=2.0, shape=obs_shape, dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(self.total_num_actions)
        self._reward_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.sim: FastOrbitSim = FastOrbitSim.from_seed(
            0,
            max_steps=self.max_steps,
            ship_speed=self.ship_speed,
            comet_speed=self.comet_speed,
        )
        self._cached_ctx = None
        self._cached_ctx_key = None

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @property
    def current_player(self) -> int:
        return self._current_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        self._current_player = int(value)

    @property
    def current_player_index(self) -> int:
        return 0 if self._current_player == 1 else 1

    @property
    def next_player(self) -> int:
        return 2 if self._current_player == 1 else 1

    def _sim_player(self) -> int:
        """Map LightZero player id {1, 2} to Orbit Wars player id {0, 1}."""
        return 0 if self._current_player == 1 else 1

    def _get_ctx(self, sim_player: int):
        key = (self.sim.step_count, self._ply, sim_player, len(self.sim.fleets))
        if self._cached_ctx_key != key:
            self._cached_ctx = build_tactical_context(self.sim, sim_player)
            self._cached_ctx_key = key
        return self._cached_ctx

    @property
    def legal_actions(self) -> List[int]:
        if self.sim.done:
            return []
        sp = self._sim_player()
        mask = get_legal_action_mask(self.sim, sp, ctx=self._get_ctx(sp))
        actions = [int(i) for i in range(self.total_num_actions) if mask[i]]
        return actions if actions else [0]

    def current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(raw_obs, scaled_obs)`` from the current player's perspective."""
        sp = self._sim_player()
        tensor = encode_observation(self.sim, player=sp, channel_last=self.channel_last)
        return tensor, tensor

    def _serialize_board(self) -> np.ndarray:
        """Pack simulator + half-turn metadata into a fixed-size ``uint8`` vector."""
        payload = (
            self.sim.planets,
            self.sim.initial_planets,
            self.sim.fleets,
            self.sim.comets,
            self.sim.comet_planet_ids,
            self.sim.angular_velocity,
            self.sim.next_fleet_id,
            self.sim.step_count,
            self.sim.num_agents,
            self.sim.max_steps,
            self.sim.ship_speed,
            self.sim.comet_speed,
            self.sim.episode_seed,
            self.sim.done,
            self.sim.rewards,
            self._pending_p0_action,
            self._current_player,
            self._ply,
        )
        compressed = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), level=1)
        n = len(compressed)
        buf = np.zeros(BOARD_BUF_SIZE, dtype=np.uint8)
        if n + 4 <= BOARD_BUF_SIZE:
            buf[:4] = np.frombuffer(struct.pack("<I", n), dtype=np.uint8)
            buf[4 : 4 + n] = np.frombuffer(compressed, dtype=np.uint8)
        else:
            # Extremely rare overflow guard: drop comet path histories beyond current window
            trimmed_comets = [
                {
                    "planet_ids": g["planet_ids"],
                    "paths": [p[max(0, g["path_index"]) :] for p in g["paths"]],
                    "path_index": 0 if g["path_index"] >= 0 else g["path_index"],
                }
                for g in self.sim.comets
            ]
            payload_small = payload[:3] + (trimmed_comets,) + payload[4:]
            compressed = zlib.compress(
                pickle.dumps(payload_small, protocol=pickle.HIGHEST_PROTOCOL), level=6
            )
            n = min(len(compressed), BOARD_BUF_SIZE - 4)
            buf[:4] = np.frombuffer(struct.pack("<I", n), dtype=np.uint8)
            buf[4 : 4 + n] = np.frombuffer(compressed[:n], dtype=np.uint8)
        return buf

    def _deserialize_board(self, init_state: Any, start_player_index: int) -> None:
        if isinstance(init_state, FastOrbitSim):
            self.sim = init_state.clone()
            self._pending_p0_action = None
            self._current_player = self.players[start_player_index]
            self._ply = self.sim.step_count
            return

        if hasattr(init_state, "detach"):
            raw_bytes = init_state.detach().cpu().numpy().astype(np.uint8).tobytes()
        elif isinstance(init_state, np.ndarray):
            raw_bytes = init_state.astype(np.uint8, copy=False).tobytes()
        elif isinstance(init_state, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(init_state)
        else:
            raise TypeError(f"Unsupported init_state type: {type(init_state)!r}")

        (n,) = struct.unpack("<I", raw_bytes[:4])
        compressed = raw_bytes[4 : 4 + n]
        (
            planets,
            initial_planets,
            fleets,
            comets,
            comet_planet_ids,
            angular_velocity,
            next_fleet_id,
            step_count,
            num_agents,
            max_steps,
            ship_speed,
            comet_speed,
            episode_seed,
            done,
            rewards,
            pending_p0_action,
            current_player,
            ply,
        ) = pickle.loads(zlib.decompress(compressed))

        self.sim = FastOrbitSim(
            planets=planets,
            initial_planets=initial_planets,
            fleets=fleets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            angular_velocity=angular_velocity,
            next_fleet_id=next_fleet_id,
            step_count=step_count,
            num_agents=num_agents,
            max_steps=max_steps,
            ship_speed=ship_speed,
            comet_speed=comet_speed,
            episode_seed=episode_seed,
            done=done,
            rewards=rewards,
        )
        self._pending_p0_action = pending_p0_action
        self._current_player = int(current_player)
        self._ply = int(ply)

    def _make_obs(self) -> Dict[str, Any]:
        action_mask = np.zeros(self.total_num_actions, dtype=np.int8)
        legal = self.legal_actions
        if legal:
            action_mask[legal] = 1
        else:
            action_mask[0] = 1
        to_play = self._current_player if self.battle_mode == "self_play_mode" else -1
        return {
            "observation": self.current_state()[1],
            "action_mask": action_mask,
            "board": self._serialize_board(),
            "current_player_index": self.current_player_index,
            "to_play": to_play,
            "timestep": np.int64(self._ply),
        }

    def reset(
        self,
        start_player_index: int = 0,
        init_state: Any = None,
        katago_policy_init: bool = False,
        katago_game_state: Any = None,
    ) -> Dict[str, Any]:
        del katago_policy_init, katago_game_state
        self.start_player_index = int(start_player_index)
        self._cached_ctx = None
        self._cached_ctx_key = None

        if init_state is not None:
            self._deserialize_board(init_state, self.start_player_index)
        else:
            self._episode_counter += 1
            ep_seed = self._rng.randint(0, 2 ** 31 - 1)
            self.sim = FastOrbitSim.from_seed(
                ep_seed,
                num_agents=self.num_agents,
                max_steps=self.max_steps,
                ship_speed=self.ship_speed,
                comet_speed=self.comet_speed,
            )
            self._current_player = self.players[self.start_player_index]
            self._pending_p0_action = None
            self._ply = 0

        return self._make_obs()

    def _bot_moves_for(self, sim_player: int) -> List[List[float]]:
        if self.bot_action_type == "heuristic":
            return heuristic_moves(self.sim, sim_player)
        if self.bot_action_type == "starter":
            return starter_moves(self.sim, sim_player)
        if self.bot_action_type == "random":
            return random_moves(self.sim, sim_player, rng=self._rng)
        return heuristic_moves(self.sim, sim_player)

    def bot_action(self) -> int:
        sp = self._sim_player()
        return choose_heuristic_macro_action(self.sim, sp, ctx=self._get_ctx(sp))

    def random_action(self) -> int:
        legal = self.legal_actions
        return int(self._rng.choice(legal)) if legal else 0

    def _advance_sim(self, actions: List[List[List[float]]]) -> None:
        self.sim.step(actions)
        empty = [[] for _ in range(self.sim.num_agents)]
        for _ in range(self.decision_interval - 1):
            if self.sim.done:
                break
            self.sim.step(empty)
        self._cached_ctx = None
        self._cached_ctx_key = None

    def get_done_winner(self) -> Tuple[bool, int]:
        """Return ``(done, winner)`` with LightZero player IDs ``{1, 2, -1}``."""
        if not self.sim.done:
            return False, -1
        w = self.sim.winner()
        if w == 0:
            return True, 1
        if w >= 1:
            return True, 2
        return True, -1

    def step(self, action: Any) -> BaseEnvTimestep:
        act = int(np.asarray(action).item())
        if self.battle_mode in ("play_with_bot_mode", "eval_mode"):
            return self._step_vs_bot(act)
        return self._step_self_play(act)

    def _step_vs_bot(self, action: int) -> BaseEnvTimestep:
        if self.sim.done:
            done, winner = self.get_done_winner()
            ret = 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)
            return BaseEnvTimestep(
                self._make_obs(),
                np.float32(0.0),
                True,
                {"eval_episode_return": ret, "score": ret},
            )

        m0 = execute_macro_action(self.sim, 0, action, ctx=self._get_ctx(0))
        all_actions = [m0] + [self._bot_moves_for(p) for p in range(1, self.sim.num_agents)]
        self._advance_sim(all_actions)
        self._ply += 1

        done, winner = self.get_done_winner()
        reward = 0.0
        info: Dict[str, Any] = {}
        if done:
            reward = 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)
            info["eval_episode_return"] = reward
            info["score"] = reward
            info["winner"] = winner
            info["scores"] = self.sim.scores()

        return BaseEnvTimestep(self._make_obs(), np.float32(reward), done, info)

    def _step_self_play(self, action: int) -> BaseEnvTimestep:
        if self.sim.done:
            done, winner = self.get_done_winner()
            ret_p1 = 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)
            return BaseEnvTimestep(
                self._make_obs(),
                np.float32(0.0),
                True,
                {"eval_episode_return": ret_p1, "score": ret_p1},
            )

        if self.prob_random_agent > 0.0 and self._rng.random() < self.prob_random_agent:
            action = self.random_action()
        elif self.prob_expert_agent > 0.0 and self._rng.random() < self.prob_expert_agent:
            action = self.bot_action()

        if self._current_player == 1:
            # Queue Player 1 (sim player 0) macro-action; wait for Player 2
            self._pending_p0_action = action
            self._current_player = 2
            self._ply += 1
            return BaseEnvTimestep(
                self._make_obs(),
                np.float32(0.0),
                False,
                {"next player to play": 2},
            )

        # Player 2 (sim player 1) submits simultaneously with queued Player 1 action
        p0_act = self._pending_p0_action if self._pending_p0_action is not None else 0
        m0 = execute_macro_action(self.sim, 0, p0_act, ctx=self._get_ctx(0))
        m1 = execute_macro_action(self.sim, 1, action, ctx=self._get_ctx(1))
        self._pending_p0_action = None
        self._advance_sim([m0, m1])
        self._current_player = 1
        self._ply += 1

        done, winner = self.get_done_winner()
        reward = 0.0
        info: Dict[str, Any] = {"next player to play": 1}
        if done:
            # Reward is from the perspective of the player who just acted (Player 2)
            if winner == 2:
                reward = 1.0
            elif winner == 1:
                reward = -1.0
            else:
                reward = 0.0
            # eval_episode_return / score are always from Player 1's perspective
            p1_ret = -reward if reward != 0.0 else 0.0
            info["eval_episode_return"] = p1_ret
            info["score"] = p1_ret
            info["winner"] = winner
            info["scores"] = self.sim.scores()

        return BaseEnvTimestep(self._make_obs(), np.float32(reward), done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed_val = int(seed)
        self._dynamic_seed = bool(dynamic_seed)
        self._rng = random.Random(self._seed_val)
        np.random.seed(self._seed_val % (2 ** 32 - 1))

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"OrbitWarsLightZeroEnv(battle_mode={self.battle_mode!r}, "
            f"bot={self.bot_action_type!r}, max_steps={self.max_steps}, "
            f"decision_interval={self.decision_interval})"
        )
