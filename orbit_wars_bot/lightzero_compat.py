"""LightZero path wiring and cross-platform compatibility helpers.

Handles:
1. Adding ``external/LightZero`` (or ``$LIGHTZERO_PATH``) onto ``sys.path``.
2. Patching ``lzero.mcts.ptree.ptree_az.MCTS`` to expose
   ``get_next_actions_batch`` when upstream LightZero's ``AlphaZeroPolicy``
   calls it with ``mcts_ctree=False`` (pure-Python MCTS on Windows/Linux).
3. Detecting whether LightZero's compiled C++ ctree extensions are available.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, List, Sequence, Tuple

from orbit_wars_bot import ensure_lightzero_on_path


def patch_ptree_az_batch_api() -> bool:
    """Ensure ``lzero.mcts.ptree.ptree_az.MCTS`` supports ``get_next_actions_batch``.

    Upstream LightZero's ``AlphaZeroPolicy._forward_collect`` and ``_forward_eval``
    call ``get_next_actions_batch(state_configs, policy_value_fn_batch, temperature,
    sample, env_list)`` unconditionally, whereas ``ptree_az.MCTS`` only implements
    ``get_next_action``. This adds a clean fallback when missing.

    Returns:
        True if patched or already present, False if ``lzero`` is not importable.
    """
    ensure_lightzero_on_path()
    try:
        from lzero.mcts.ptree.ptree_az import MCTS as PTreeAZMCTS
    except Exception:
        return False

    if hasattr(PTreeAZMCTS, "get_next_actions_batch"):
        return True

    def get_next_actions_batch(
        self: Any,
        state_configs: Sequence[dict],
        policy_value_fn_batch: Callable[[List[Any]], List[Tuple[dict, float]]],
        temperature: float = 1.0,
        sample: bool = True,
        env_list: Sequence[Any] | None = None,
    ) -> List[Tuple[int, Any]]:
        results: List[Tuple[int, Any]] = []
        orig_env = self.simulate_env

        def _single_policy_value_fn(env: Any) -> Tuple[dict, float]:
            return policy_value_fn_batch([env])[0]

        try:
            for idx, state_config in enumerate(state_configs):
                if env_list is not None and idx < len(env_list):
                    self.simulate_env = env_list[idx]
                action, action_probs = self.get_next_action(
                    state_config,
                    _single_policy_value_fn,
                    temperature=temperature,
                    sample=sample,
                )
                results.append((int(action), action_probs))
        finally:
            self.simulate_env = orig_env
        return results

    PTreeAZMCTS.get_next_actions_batch = get_next_actions_batch
    return True


def is_ctree_muzero_available() -> bool:
    """Return True if LightZero's compiled C++ MuZero ctree extension is importable."""
    ensure_lightzero_on_path()
    try:
        importlib.import_module("lzero.mcts.ctree.ctree_muzero.mz_tree")
        return True
    except Exception:
        return False


def is_ctree_alphazero_available() -> bool:
    """Return True if LightZero's compiled C++ AlphaZero ctree extension is importable."""
    ensure_lightzero_on_path()
    try:
        from lzero.mcts.ctree.ctree_alphazero.test.eval_alphazero_ctree import (
            find_and_add_to_sys_path,
        )

        find_and_add_to_sys_path("lzero/mcts/ctree/ctree_alphazero/build")
        importlib.import_module("mcts_alphazero")
        return True
    except Exception:
        return False


def install_ctree_stubs_if_missing() -> None:
    """Stub uncompiled LightZero C++ ctree extensions when running pure-Python ptree.

    On Windows without a C++ build of LightZero, ``lzero.mcts.tree_search.mcts_ctree``
    eagerly imports compiled extensions (``ez_tree``, ``mz_tree``, ``gmz_tree``, etc.)
    at module import time even when ``mcts_ctree=False``. If an extension cannot be
    imported, we install a lightweight placeholder module so ``train_alphazero`` and
    ``train_muzero`` can import cleanly in pure-Python ptree mode.
    """
    import sys
    import types

    ensure_lightzero_on_path()
    ext_specs = [
        ("lzero.mcts.ctree.ctree_efficientzero", "ez_tree"),
        ("lzero.mcts.ctree.ctree_muzero", "mz_tree"),
        ("lzero.mcts.ctree.ctree_gumbel_muzero", "gmz_tree"),
        ("lzero.mcts.ctree.ctree_stochastic_muzero", "smz_tree"),
        ("lzero.mcts.ctree.ctree_sampled_efficientzero", "ezs_tree"),
        ("lzero.mcts.ctree.ctree_sampled_muzero", "mzs_tree"),
    ]
    for pkg_name, mod_name in ext_specs:
        full_name = f"{pkg_name}.{mod_name}"
        if full_name in sys.modules:
            continue
        try:
            importlib.import_module(full_name)
        except Exception:
            try:
                pkg = importlib.import_module(pkg_name)
            except Exception:
                continue
            stub = types.ModuleType(full_name)
            stub.__dict__["__ctree_unavailable__"] = True
            sys.modules[full_name] = stub
            setattr(pkg, mod_name, stub)


def patch_muzero_prediction_network() -> bool:
    """Fix upstream LightZero ``MuZeroModel`` / ``PredictionNetwork`` signature mismatch.

    In recent ``opendilab/LightZero`` commits, ``lzero.model.common.PredictionNetwork``
    dropped ``observation_shape``, ``flatten_input_size_*``, and ``downsample`` from its
    ``__init__`` signature while ``lzero.model.muzero_model.MuZeroModel`` still passes
    them positionally. When detected, replace ``muzero_model.PredictionNetwork`` with a
    compatible implementation matching ``MuZeroModel``'s exact call site.
    """
    ensure_lightzero_on_path()
    try:
        import inspect
        import torch
        import torch.nn as nn
        from ding.torch_utils import MLP, ResBlock
        from lzero.model import muzero_model as mz_mod
    except Exception:
        return False

    orig_cls = getattr(mz_mod, "PredictionNetwork", None)
    if orig_cls is None:
        return False
    sig = inspect.signature(orig_cls.__init__)
    if "flatten_input_size_for_value_head" in sig.parameters:
        return True

    class _CompatMuZeroPredictionNetwork(nn.Module):
        def __init__(
            self,
            observation_shape,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int = 16,
            policy_head_channels: int = 16,
            value_head_hidden_channels=(32,),
            policy_head_hidden_channels=(32,),
            output_support_size: int = 601,
            flatten_input_size_for_value_head: int = 64,
            flatten_input_size_for_policy_head: int = 64,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = "BN",
        ) -> None:
            super().__init__()
            del observation_shape, downsample
            self.activation = activation
            self.resblocks = nn.ModuleList(
                [
                    ResBlock(
                        in_channels=num_channels,
                        activation=activation,
                        norm_type=norm_type,
                        res_type="basic",
                        bias=False,
                    )
                    for _ in range(num_res_blocks)
                ]
            )
            self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
            self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
            self.fc_value = MLP(
                in_channels=flatten_input_size_for_value_head,
                hidden_channels=value_head_hidden_channels[0],
                out_channels=output_support_size,
                layer_num=len(value_head_hidden_channels) + 1,
                activation=activation,
                norm_type=norm_type,
                output_activation=False,
                output_norm=False,
                last_linear_layer_init_zero=last_linear_layer_init_zero,
            )
            self.fc_policy = MLP(
                in_channels=flatten_input_size_for_policy_head,
                hidden_channels=policy_head_hidden_channels[0],
                out_channels=action_space_size,
                layer_num=len(policy_head_hidden_channels) + 1,
                activation=activation,
                norm_type=norm_type,
                output_activation=False,
                output_norm=False,
                last_linear_layer_init_zero=last_linear_layer_init_zero,
            )

        def forward(self, latent_state: torch.Tensor):
            x = latent_state
            for block in self.resblocks:
                x = block(x)
            val = self.activation(self.norm_value(self.conv1x1_value(x)))
            pol = self.activation(self.norm_policy(self.conv1x1_policy(x)))
            val = self.fc_value(val.reshape(val.size(0), -1))
            pol = self.fc_policy(pol.reshape(pol.size(0), -1))
            return pol, val

    mz_mod.PredictionNetwork = _CompatMuZeroPredictionNetwork
    return True


def setup_lightzero() -> None:
    """Wire LightZero onto ``sys.path`` and apply compatibility patches."""
    ensure_lightzero_on_path()
    install_ctree_stubs_if_missing()
    patch_ptree_az_batch_api()
    patch_muzero_prediction_network()
