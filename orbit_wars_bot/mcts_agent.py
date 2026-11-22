"""Online PUCT MCTS Kaggle agent with LightZero checkpoint support.

Runs time-bounded MCTS over ``FastOrbitSim`` strictly within Kaggle's 500ms/turn
budget. Supports loading LightZero ``AlphaZeroModel`` and ``MuZeroModel``
checkpoints saved by ``scripts/train_lightzero.py``; when no checkpoint is
supplied, uses heuristic tactical priors and ``evaluate_state`` leaf evaluation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from orbit_wars_bot import ensure_lightzero_on_path
from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.features import OBS_SHAPE, encode_observation, evaluate_state
from orbit_wars_bot.heuristic_bot import choose_heuristic_macro_action, heuristic_moves
from orbit_wars_bot.macro_actions import (
    NUM_MACRO_ACTIONS,
    MacroAction,
    build_tactical_context,
    execute_macro_action,
    get_legal_action_mask,
)


class _SearchNode:
    __slots__ = ("parent", "prior", "visits", "value_sum", "children")

    def __init__(self, parent: Optional["_SearchNode"] = None, prior: float = 1.0) -> None:
        self.parent = parent
        self.prior = float(prior)
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, "_SearchNode"] = {}

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_leaf(self) -> bool:
        return not self.children


class _LoadedLightZeroNet:
    """Thin inference wrapper around a trained LightZero AlphaZero or MuZero model."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        from orbit_wars_bot.lightzero_compat import setup_lightzero

        setup_lightzero()
        import torch

        self._torch = torch
        self.device = torch.device(device)

        raw = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

        def _linear_weights(prefix: str) -> List[Tuple[int, Any]]:
            items: List[Tuple[int, Any]] = []
            for k, v in state_dict.items():
                if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2:
                    parts = k.split(".")
                    idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                    items.append((idx, v))
            items.sort(key=lambda kv: kv[0])
            return items

        # Infer model architecture from state_dict keys and tensor shapes
        is_muzero = any(k.startswith("dynamics_network.") for k in state_dict.keys())
        num_channels = int(state_dict["representation_network.conv.weight"].shape[0])
        res_indices = set()
        for k in state_dict.keys():
            if k.startswith("representation_network.resblocks."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    res_indices.add(int(parts[2]))
        num_res_blocks = len(res_indices) if res_indices else 1

        val_head_ch = int(state_dict["prediction_network.conv1x1_value.weight"].shape[0])
        pol_head_ch = int(state_dict["prediction_network.conv1x1_policy.weight"].shape[0])

        val_prefix = (
            "prediction_network.fc_value_head."
            if any(k.startswith("prediction_network.fc_value_head.") for k in state_dict)
            else "prediction_network.fc_value."
        )
        pol_prefix = (
            "prediction_network.fc_policy_head."
            if any(k.startswith("prediction_network.fc_policy_head.") for k in state_dict)
            else "prediction_network.fc_policy."
        )
        val_linears = _linear_weights(val_prefix)
        pol_linears = _linear_weights(pol_prefix)
        val_hid = [int(w.shape[0]) for _, w in val_linears[:-1]] or [32]
        pol_hid = [int(w.shape[0]) for _, w in pol_linears[:-1]] or [32]
        action_space_size = int(pol_linears[-1][1].shape[0])

        if is_muzero:
            from lzero.model.muzero_model import MuZeroModel

            rew_head_ch = int(state_dict["dynamics_network.conv1x1_reward.weight"].shape[0])
            rew_prefix = (
                "dynamics_network.fc_reward_head."
                if any(k.startswith("dynamics_network.fc_reward_head.") for k in state_dict)
                else "dynamics_network.fc_reward."
            )
            rew_linears = _linear_weights(rew_prefix)
            rew_hid = [int(w.shape[0]) for _, w in rew_linears[:-1]] or [32]
            val_support_size = int(val_linears[-1][1].shape[0])
            half = (val_support_size - 1) // 2
            support_range = (-float(half), float(half + 1), 1.0)

            self.model = MuZeroModel(
                observation_shape=OBS_SHAPE,
                action_space_size=action_space_size,
                num_res_blocks=num_res_blocks,
                num_channels=num_channels,
                reward_head_channels=rew_head_ch,
                value_head_channels=val_head_ch,
                policy_head_channels=pol_head_ch,
                reward_head_hidden_channels=rew_hid,
                value_head_hidden_channels=val_hid,
                policy_head_hidden_channels=pol_hid,
                reward_support_range=support_range,
                value_support_range=support_range,
                downsample=False,
            )
            self.is_muzero = True
            self._support = torch.arange(*support_range, device=self.device, dtype=torch.float32)
        else:
            from lzero.model.alphazero_model import AlphaZeroModel

            self.model = AlphaZeroModel(
                observation_shape=OBS_SHAPE,
                action_space_size=action_space_size,
                num_res_blocks=num_res_blocks,
                num_channels=num_channels,
                value_head_channels=val_head_ch,
                policy_head_channels=pol_head_ch,
                value_head_hidden_channels=val_hid,
                policy_head_hidden_channels=pol_hid,
                downsample=False,
            )
            self.is_muzero = False
            self._support = None

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, obs_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(obs_tensor).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            if self.is_muzero:
                out = self.model.initial_inference(x)
                probs = torch.softmax(out.policy_logits, dim=-1).squeeze(0).cpu().numpy()
                val_probs = torch.softmax(out.value, dim=-1).squeeze(0)
                val_scalar = float(torch.sum(val_probs * self._support).item())
                val = float(math.tanh(val_scalar))
            else:
                probs_t, val_t = self.model.compute_policy_value(x)
                probs = probs_t.squeeze(0).cpu().numpy()
                val = float(val_t.squeeze().item())
        return probs, max(-1.0, min(1.0, val))


def _heuristic_priors(state: FastOrbitSim, player: int, legal_mask: np.ndarray, ctx: Any) -> np.ndarray:
    """Fast tactical prior distribution over the 16 macro-actions."""
    priors = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
    preferred = choose_heuristic_macro_action(state, player, ctx=ctx)

    base_weights = {
        MacroAction.HOLD_ACCUMULATE: 0.45,
        MacroAction.EXPAND_BEST_ROI_SINGLE: 1.0,
        MacroAction.EXPAND_BEST_ROI_MULTI: 1.35,
        MacroAction.EXPAND_HIGH_PROD: 0.9,
        MacroAction.EXPAND_ORBITING: 0.85,
        MacroAction.EXPAND_COMET_SNIPE: 0.8,
        MacroAction.DEFEND_URGENT: 2.2,
        MacroAction.COUNTER_EVACUATE: 1.3,
        MacroAction.VULTURE_SNIPE: 1.8,
        MacroAction.ATTACK_WEAKEST_ENEMY: 1.15,
        MacroAction.ATTACK_HIGH_PROD_ENEMY: 1.1,
        MacroAction.ATTACK_CLOSEST_ENEMY: 0.95,
        MacroAction.COORDINATED_ALL_IN: 0.75,
        MacroAction.FEED_FRONTLINE: 0.8,
        MacroAction.BALANCED_MACRO: 2.0,
        MacroAction.AGGRESSIVE_SKIRMISH: 1.25,
    }
    for act, w in base_weights.items():
        if legal_mask[act]:
            priors[act] = w
    if legal_mask[preferred]:
        priors[preferred] += 1.5

    total = float(priors.sum())
    if total <= 0.0:
        priors[0] = 1.0
    else:
        priors /= total
    return priors


class OnlineMCTSAgent:
    """Time-budgeted online PUCT MCTS agent for Kaggle Orbit Wars."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        num_simulations: int = 24,
        time_budget_sec: float = 0.35,
        rollout_horizon: int = 12,
        step_stride: int = 4,
        c_puct: float = 1.4,
        neural_value_weight: float = 0.6,
        device: str = "cpu",
    ) -> None:
        self.num_simulations = max(1, int(num_simulations))
        self.time_budget_sec = float(time_budget_sec)
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.step_stride = max(1, int(step_stride))
        self.c_puct = float(c_puct)
        self.neural_value_weight = float(neural_value_weight)

        self._net: Optional[_LoadedLightZeroNet] = None
        if checkpoint_path and Path(checkpoint_path).is_file():
            try:
                self._net = _LoadedLightZeroNet(checkpoint_path, device=device)
            except Exception as exc:
                print(f"[OnlineMCTSAgent] Failed to load checkpoint {checkpoint_path!r}: {exc}")
                self._net = None

    @property
    def has_neural_model(self) -> bool:
        return self._net is not None

    def _evaluate_leaf(self, sim: FastOrbitSim, player: int) -> Tuple[np.ndarray, float]:
        ctx = build_tactical_context(sim, player, horizon=24)
        mask = get_legal_action_mask(sim, player, ctx=ctx)
        h_val = evaluate_state(sim, player)
        h_priors = _heuristic_priors(sim, player, mask, ctx)

        if self._net is not None:
            obs_t = encode_observation(sim, player=player)
            nn_probs, nn_val = self._net.predict(obs_t)
            # Mask neural priors to legal actions and softly blend with tactical prior
            masked_nn = nn_probs * mask.astype(np.float32)
            s = float(masked_nn.sum())
            if s > 1e-8:
                masked_nn /= s
                priors = 0.75 * masked_nn + 0.25 * h_priors
            else:
                priors = h_priors
            val = self.neural_value_weight * nn_val + (1.0 - self.neural_value_weight) * h_val
            return priors, val

        return h_priors, h_val

    def _select_child(self, node: _SearchNode) -> Tuple[int, _SearchNode]:
        sqrt_parent = math.sqrt(max(1, node.visits))
        best_score = -1e18
        best_act = 0
        best_child = next(iter(node.children.values()))
        for act, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            score = child.q_value + u
            if score > best_score:
                best_score = score
                best_act = act
                best_child = child
        return best_act, best_child

    def _advance_ply(self, sim: FastOrbitSim, player: int, action: int, opp_action_fn: Callable) -> None:
        opp = 1 - player
        m_me = execute_macro_action(sim, player, action)
        m_opp = opp_action_fn(sim, opp)
        actions = [m_me, m_opp] if player == 0 else [m_opp, m_me]
        sim.step(actions)
        for _ in range(self.step_stride - 1):
            if sim.done:
                break
            sim.step([[], []])

    def select_macro_action(self, obs: Any, config: Any = None) -> Tuple[int, Dict[int, int]]:
        """Run time-budgeted MCTS from ``obs`` and return ``(best_action, visit_counts)``."""
        t_start = time.perf_counter()
        deadline = t_start + self.time_budget_sec

        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        root_sim = FastOrbitSim.from_observation(obs, config)
        root_ctx = build_tactical_context(root_sim, player, horizon=24)
        root_mask = get_legal_action_mask(root_sim, player, ctx=root_ctx)
        legal_actions = [int(i) for i in range(NUM_MACRO_ACTIONS) if root_mask[i]]

        if len(legal_actions) <= 1:
            act = legal_actions[0] if legal_actions else 0
            return act, {act: 1}

        # Expand root
        root = _SearchNode()
        priors, _ = self._evaluate_leaf(root_sim, player)
        for a in legal_actions:
            root.children[a] = _SearchNode(parent=root, prior=float(priors[a]))

        max_depth = max(1, self.rollout_horizon // self.step_stride)
        opp_fn = lambda s, p: heuristic_moves(s, p)

        for sim_idx in range(self.num_simulations):
            if sim_idx > 0 and time.perf_counter() >= deadline:
                break

            sim = root_sim.clone()
            node = root
            depth = 0

            # Selection
            while not node.is_leaf() and not sim.done and depth < max_depth:
                act, node = self._select_child(node)
                self._advance_ply(sim, player, act, opp_fn)
                depth += 1

            # Evaluation / Expansion
            if sim.done:
                w = sim.winner()
                value = 1.0 if w == player else (0.0 if w == -1 else -1.0)
            else:
                leaf_priors, value = self._evaluate_leaf(sim, player)
                if depth < max_depth and node.is_leaf():
                    leaf_mask = get_legal_action_mask(sim, player)
                    for a in range(NUM_MACRO_ACTIONS):
                        if leaf_mask[a]:
                            node.children[a] = _SearchNode(parent=node, prior=float(leaf_priors[a]))

            # Backup (single-perspective vs heuristic opponent)
            cur: Optional[_SearchNode] = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += value
                cur = cur.parent

        visits = {a: ch.visits for a, ch in root.children.items()}
        best_action = max(
            root.children.items(),
            key=lambda kv: (kv[1].visits, kv[1].q_value, kv[1].prior),
        )[0]
        return int(best_action), visits

    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        """Kaggle Environments callback returning exact ``[from_id, angle, ships]`` moves."""
        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        ctx = build_tactical_context(obs, player)
        # Fast path when we have no surplus and no doomed planets (<1ms)
        if not ctx.my_planets or (sum(ctx.surplus.values()) < 2 and not ctx.doomed_my):
            return []
        best_action, _ = self.select_macro_action(obs, config)
        return execute_macro_action(obs, player, best_action, ctx=ctx)


def create_kaggle_agent(
    checkpoint_path: Optional[str] = None,
    *,
    num_simulations: int = 24,
    time_budget_sec: float = 0.35,
    rollout_horizon: int = 12,
    device: str = "cpu",
) -> Callable[[Any, Any], List[List[float]]]:
    """Factory returning a Kaggle-compatible ``agent(obs, config)`` function."""
    mcts_agent = OnlineMCTSAgent(
        checkpoint_path=checkpoint_path,
        num_simulations=num_simulations,
        time_budget_sec=time_budget_sec,
        rollout_horizon=rollout_horizon,
        device=device,
    )

    def _agent(obs: Any, config: Any = None) -> List[List[float]]:
        return mcts_agent.act(obs, config)

    return _agent
