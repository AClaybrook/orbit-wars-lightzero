"""Online PUCT MCTS Kaggle agent with LightZero checkpoint support (2P & 4P FFA).

Runs time-bounded MCTS over ``FastOrbitSim`` strictly within Kaggle's 500ms/turn
budget. Evaluates distinct, defense-guarded full-turn strategic stances (mapped
1-to-1 to ``MacroAction`` IDs) using **active-opponent** exact orbital/combat
rollouts (so opponents actually defend and counter-attack during lookahead),
and supports loading trained LightZero ``AlphaZeroModel`` and ``MuZeroModel``
checkpoints saved by ``scripts/train_lightzero.py`` or ``scripts/pretrain_alphazero.py``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from orbit_wars_bot.fast_sim import FastOrbitSim
from orbit_wars_bot.features import OBS_SHAPE, encode_observation, evaluate_state
from orbit_wars_bot.heuristic_bot import choose_heuristic_macro_action, heuristic_moves
from orbit_wars_bot.macro_actions import (
    DEFAULT_TACTICAL_HORIZON,
    NUM_MACRO_ACTIONS,
    MacroAction,
    TacticalContext,
    build_tactical_context,
    execute_macro_action,
    get_legal_action_mask,
)

# Compact, genuinely complementary full-turn strategic stances (subset of MacroAction IDs)
_CURATED_STANCES: Tuple[MacroAction, ...] = (
    MacroAction.BALANCED_MACRO,
    MacroAction.EXPAND_BEST_ROI_MULTI,
    MacroAction.FEED_FRONTLINE,
    MacroAction.ATTACK_HIGH_PROD_ENEMY,
    MacroAction.AGGRESSIVE_SKIRMISH,
    MacroAction.COORDINATED_ALL_IN,
    MacroAction.HOLD_ACCUMULATE,
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

        if device == "cpu":
            torch.set_num_threads(1)
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


def _heuristic_priors(state: FastOrbitSim, player: int, legal_mask: np.ndarray, ctx: TacticalContext) -> np.ndarray:
    """Fast tactical prior distribution over the 16 macro-actions."""
    priors = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
    preferred = choose_heuristic_macro_action(state, player, ctx=ctx)

    base_weights = {
        MacroAction.HOLD_ACCUMULATE: 0.65,
        MacroAction.EXPAND_BEST_ROI_SINGLE: 0.55,
        MacroAction.EXPAND_BEST_ROI_MULTI: 1.35,
        MacroAction.EXPAND_HIGH_PROD: 0.90,
        MacroAction.EXPAND_ORBITING: 0.50,
        MacroAction.EXPAND_COMET_SNIPE: 0.45,
        MacroAction.DEFEND_URGENT: 0.80,
        MacroAction.COUNTER_EVACUATE: 0.60,
        MacroAction.VULTURE_SNIPE: 0.80,
        MacroAction.ATTACK_WEAKEST_ENEMY: 1.10,
        MacroAction.ATTACK_HIGH_PROD_ENEMY: 1.15,
        MacroAction.ATTACK_CLOSEST_ENEMY: 0.85,
        MacroAction.COORDINATED_ALL_IN: 0.70 if not ctx.is_ffa else 0.40,
        MacroAction.FEED_FRONTLINE: 1.25 if ctx.is_ffa else 0.90,
        MacroAction.BALANCED_MACRO: 2.80,
        MacroAction.AGGRESSIVE_SKIRMISH: 1.05 if not ctx.is_ffa else 0.75,
    }
    for act, w in base_weights.items():
        if legal_mask[act]:
            priors[act] = w
    if legal_mask[preferred]:
        priors[preferred] += 1.2

    total = float(priors.sum())
    if total <= 0.0:
        priors[0] = 1.0
    else:
        priors /= total
    return priors


def _moves_signature(moves: List[List[float]]) -> Tuple[Tuple[int, int, int], ...]:
    """Canonical deduplication key for a list of launch orders."""
    if not moves:
        return ()
    return tuple(sorted((int(m[0]), int(round(float(m[1]) * 100.0)), int(m[2])) for m in moves))


class OnlineMCTSAgent:
    """Time-budgeted online PUCT MCTS agent for Kaggle Orbit Wars (2P & 4P FFA)."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        num_simulations: int = 16,
        time_budget_sec: float = 0.22,
        rollout_horizon: int = 16,
        step_stride: int = 8,
        c_puct: float = 1.10,
        neural_value_weight: float = 0.40,
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

    def _evaluate_leaf(
        self,
        sim: FastOrbitSim,
        player: int,
        ctx: Optional[TacticalContext] = None,
        need_priors: bool = True,
    ) -> Tuple[Optional[np.ndarray], float]:
        h_val = evaluate_state(sim, player)
        if not need_priors and self._net is None:
            return None, h_val

        priors: Optional[np.ndarray] = None
        if need_priors:
            if ctx is None:
                ctx = build_tactical_context(sim, player, horizon=16)
            mask = get_legal_action_mask(sim, player, ctx=ctx)
            h_priors = _heuristic_priors(sim, player, mask, ctx)
            priors = h_priors
        else:
            mask = None
            h_priors = None

        if self._net is not None:
            obs_t = encode_observation(sim, player=player)
            nn_probs, nn_val = self._net.predict(obs_t)
            if need_priors and mask is not None and h_priors is not None:
                masked_nn = nn_probs * mask.astype(np.float32)
                s = float(masked_nn.sum())
                if s > 1e-8:
                    masked_nn /= s
                    priors = 0.65 * masked_nn + 0.35 * h_priors
            val = self.neural_value_weight * nn_val + (1.0 - self.neural_value_weight) * h_val
            return priors, val

        return priors, h_val

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

    def _advance_active_ply(
        self,
        sim: FastOrbitSim,
        player: int,
        my_moves: List[List[float]],
        precomputed_opp_moves: Optional[Dict[int, List[List[float]]]] = None,
        sub_stride: int = 4,
        rollout_ctx_horizon: int = 14,
        deadline: Optional[float] = None,
    ) -> None:
        """Advance ``self.step_stride`` ticks while allowing active heuristic defense/follow-up.

        Crucially avoids the "frozen opponent" bias where reckless attacks appear strong
        only because opponents didn't respond during coast ticks.
        """
        num_agents = sim.num_agents
        actions: List[List[List[float]]] = [[] for _ in range(num_agents)]
        actions[player] = my_moves

        if precomputed_opp_moves is not None:
            for pid, mv in precomputed_opp_moves.items():
                if 0 <= pid < num_agents and pid != player:
                    actions[pid] = mv
        elif deadline is None or time.perf_counter() < deadline:
            alive_opps = {
                int(p[1]) for p in sim.planets if 0 <= int(p[1]) < num_agents and int(p[1]) != player
            }
            for pid in alive_opps:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                opp_ctx = build_tactical_context(sim, pid, horizon=rollout_ctx_horizon)
                actions[pid] = heuristic_moves(sim, pid, ctx=opp_ctx)

        sim.step(actions)
        empty = [[] for _ in range(num_agents)]

        # Step remaining ticks with a realistic mid-ply reactive turn so both sides
        # defend threatened planets and spend pooled ships!
        rem_ticks = self.step_stride - 1
        for k in range(1, rem_ticks + 1):
            if sim.done:
                break
            if (
                sub_stride > 0
                and k % sub_stride == 0
                and k < rem_ticks
                and (deadline is None or time.perf_counter() < deadline)
            ):
                alive = {int(p[1]) for p in sim.planets if 0 <= int(p[1]) < num_agents}
                mid_actions: List[List[List[float]]] = [[] for _ in range(num_agents)]
                for pid in alive:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    c = build_tactical_context(sim, pid, horizon=rollout_ctx_horizon)
                    mid_actions[pid] = heuristic_moves(sim, pid, ctx=c)
                sim.step(mid_actions)
            else:
                sim.step(empty)

    def _search_root(
        self,
        obs: Any,
        config: Any = None,
        root_ctx: Optional[TacticalContext] = None,
    ) -> Tuple[int, Dict[int, int], Dict[int, List[List[float]]]]:
        """Run time-bounded active-opponent PUCT MCTS over deduplicated full-turn stances."""
        t_start = time.perf_counter()
        deadline = t_start + self.time_budget_sec

        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        root_sim = FastOrbitSim.from_observation(obs, config)
        if root_ctx is None:
            root_ctx = build_tactical_context(root_sim, player, horizon=DEFAULT_TACTICAL_HORIZON)
        root_mask = get_legal_action_mask(root_sim, player, ctx=root_ctx)

        cand_moves: Dict[int, List[List[float]]] = {}
        seen_sigs: Dict[Tuple[Tuple[int, int, int], ...], int] = {}
        distinct_actions: List[int] = []

        for act_enum in _CURATED_STANCES:
            a = int(act_enum)
            if not root_mask[a]:
                continue
            # Never test pure HOLD_ACCUMULATE when we have threatened/doomed planets
            if act_enum == MacroAction.HOLD_ACCUMULATE and (root_ctx.threatened_my or root_ctx.doomed_my):
                continue
            mv = execute_macro_action(root_sim, player, a, ctx=root_ctx)
            sig = _moves_signature(mv)
            if sig not in seen_sigs:
                seen_sigs[sig] = a
                cand_moves[a] = mv
                distinct_actions.append(a)

        if not distinct_actions:
            fallback = int(MacroAction.BALANCED_MACRO if root_mask[MacroAction.BALANCED_MACRO] else MacroAction.HOLD_ACCUMULATE)
            cand_moves[fallback] = execute_macro_action(root_sim, player, fallback, ctx=root_ctx)
            return fallback, {fallback: 1}, cand_moves

        if len(distinct_actions) == 1:
            only = distinct_actions[0]
            return only, {only: 1}, cand_moves

        # Precompute opponent moves ONCE at the root
        root_opp_moves: Dict[int, List[List[float]]] = {}
        opp_horizon = 18 if root_sim.num_agents == 2 else 14
        for opp_id in root_ctx.active_opponents:
            if time.perf_counter() >= deadline:
                break
            opp_ctx = build_tactical_context(root_sim, opp_id, horizon=opp_horizon)
            root_opp_moves[opp_id] = heuristic_moves(root_sim, opp_id, ctx=opp_ctx)

        # Expand root over distinct stances
        root = _SearchNode()
        priors, _ = self._evaluate_leaf(root_sim, player, ctx=root_ctx, need_priors=True)
        assert priors is not None
        prior_sum = sum(float(priors[a]) for a in distinct_actions)
        if prior_sum <= 1e-8:
            for a in distinct_actions:
                root.children[a] = _SearchNode(parent=root, prior=1.0 / len(distinct_actions))
        else:
            for a in distinct_actions:
                root.children[a] = _SearchNode(parent=root, prior=float(priors[a]) / prior_sum)

        max_depth = max(1, self.rollout_horizon // self.step_stride)
        sub_stride = 4 if root_sim.num_agents == 2 else 5

        for sim_idx in range(self.num_simulations):
            if sim_idx > 0 and time.perf_counter() >= deadline:
                break

            sim = root_sim.clone()
            node = root
            depth = 0

            while not node.is_leaf() and not sim.done and depth < max_depth:
                if depth == 0 and sim_idx < len(distinct_actions):
                    act = distinct_actions[sim_idx]
                    node = root.children[act]
                else:
                    act, node = self._select_child(node)

                if depth == 0:
                    self._advance_active_ply(
                        sim,
                        player,
                        cand_moves[act],
                        precomputed_opp_moves=root_opp_moves,
                        sub_stride=sub_stride,
                        rollout_ctx_horizon=14,
                        deadline=deadline,
                    )
                else:
                    if time.perf_counter() >= deadline:
                        break
                    leaf_ctx = build_tactical_context(sim, player, horizon=14)
                    my_mv = execute_macro_action(sim, player, act, ctx=leaf_ctx)
                    self._advance_active_ply(
                        sim,
                        player,
                        my_mv,
                        precomputed_opp_moves=None,
                        sub_stride=sub_stride,
                        rollout_ctx_horizon=12,
                        deadline=deadline,
                    )
                depth += 1

            # Evaluate leaf (and expand depth-2 children only if depth < max_depth)
            if sim.done:
                w = sim.winner()
                value = 1.0 if w == player else (0.0 if w == -1 else -1.0)
            else:
                expand_here = (depth < max_depth and node.is_leaf())
                leaf_ctx = build_tactical_context(sim, player, horizon=14) if expand_here else None
                leaf_priors, value = self._evaluate_leaf(
                    sim, player, ctx=leaf_ctx, need_priors=expand_here
                )
                if expand_here and leaf_priors is not None and leaf_ctx is not None:
                    leaf_mask = get_legal_action_mask(sim, player, ctx=leaf_ctx)
                    child_acts = [
                        int(a)
                        for a in (
                            MacroAction.BALANCED_MACRO,
                            MacroAction.EXPAND_BEST_ROI_MULTI,
                            MacroAction.FEED_FRONTLINE,
                            MacroAction.AGGRESSIVE_SKIRMISH,
                        )
                        if leaf_mask[a]
                    ]
                    if not child_acts:
                        child_acts = [int(MacroAction.HOLD_ACCUMULATE)]
                    c_sum = sum(float(leaf_priors[a]) for a in child_acts)
                    for a in child_acts:
                        p_a = float(leaf_priors[a]) / c_sum if c_sum > 1e-8 else 1.0 / len(child_acts)
                        node.children[a] = _SearchNode(parent=node, prior=p_a)

            cur: Optional[_SearchNode] = node
            while cur is not None:
                cur.visits += 1
                cur.value_sum += value
                cur = cur.parent

        visits = {a: ch.visits for a, ch in root.children.items()}

        # Select the best evaluated stance, using BALANCED_MACRO as a disciplined anchor
        # so noisy near-ties don't override our strong tactical baseline.
        anchor_id = int(MacroAction.BALANCED_MACRO)
        visited_children = [(a, ch) for a, ch in root.children.items() if ch.visits > 0]
        if not visited_children:
            best_action = anchor_id if anchor_id in root.children else distinct_actions[0]
        else:
            best_action = max(
                visited_children,
                key=lambda kv: (kv[1].q_value, kv[1].visits, kv[1].prior),
            )[0]
            if anchor_id in root.children and root.children[anchor_id].visits > 0 and best_action != anchor_id:
                anchor_q = root.children[anchor_id].q_value
                best_q = root.children[best_action].q_value
                if best_q < anchor_q + 0.015:
                    best_action = anchor_id

        return int(best_action), visits, cand_moves

    def select_macro_action(self, obs: Any, config: Any = None) -> Tuple[int, Dict[int, int]]:
        """Run time-budgeted MCTS from ``obs`` and return ``(best_action, visit_counts)``."""
        best_action, visits, _ = self._search_root(obs, config)
        return best_action, visits

    def act(self, obs: Any, config: Any = None) -> List[List[float]]:
        """Kaggle Environments callback returning exact ``[from_id, angle, ships]`` moves."""
        player = int(obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0))
        ctx = build_tactical_context(obs, player, horizon=DEFAULT_TACTICAL_HORIZON)
        if not ctx.my_planets or (sum(ctx.raw_surplus.values()) < 2 and not ctx.doomed_my):
            return []
        best_action, _, cand_moves = self._search_root(obs, config, root_ctx=ctx)
        if best_action in cand_moves:
            return cand_moves[best_action]
        return execute_macro_action(obs, player, best_action, ctx=ctx)


def create_kaggle_agent(
    checkpoint_path: Optional[str] = None,
    *,
    num_simulations: int = 16,
    time_budget_sec: float = 0.22,
    rollout_horizon: int = 16,
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
