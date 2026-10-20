"""Equivalence tests for the fused ``k == G`` ("dense tail") attention path.

``NESTOR_DENSE_TAIL_ATTN=1`` routes a block whose tokens select EVERY column
(``num_groups_per_token == num_groups``) through
:meth:`~nestor.backbones.nestor.RoutedTransformerLayer._attn_update_dense_tail`:
one fused q/k/v GEMM, one causal SDPA over all ``G * H`` heads, one batched
output projection. The per-group token-dispatch loop (flag off) is the
reference implementation throughout and must agree in the forward, in every
parameter gradient, in every residual topology (projection / kimi / column),
and under activation checkpointing.

Every test builds the SAME model twice under a fixed seed -- once per flag
state -- so the attention path is the only difference. Eval mode keeps the
router noise off, so routing is deterministic. CPU-runnable: the fused path is
plain ``F.linear`` / SDPA / ``bmm``.
"""

from __future__ import annotations

import os
from typing import Self

import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig

_FLAG = "NESTOR_DENSE_TAIL_ATTN"
_CKPT = "NESTOR_ACTIVATION_CHECKPOINT"


class _FlagEnv:
    """Set/restore real env flags around a block (no monkeypatching).

    Attributes:
        values (dict[str, str]): Flag -> value inside the block.
        previous (dict[str, str | None]): Restored on exit (None -> unset).
    """

    def __init__(self, **values: str) -> None:
        self.values: dict[str, str] = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> Self:
        for key, val in self.values.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = val
        return self

    def __exit__(self, *exc: object) -> None:
        for key, prev in self.previous.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _config(
    *,
    block_residual_mode: str = "projection",
    num_groups: list[int] | None = None,
    num_groups_per_token: list[int] | None = None,
    causal: bool = True,
) -> NestorConfig:
    """Tiny causal token-dispatch Nestor: a k == G block then a k < G block.

    Args:
        block_residual_mode (str): ``"projection"``, ``"attn_res_kimi"`` or
            ``"attn_res_column"``.
        num_groups (list[int] | None): Columns per block; default ``[4, 4]``.
        num_groups_per_token (list[int] | None): ``k`` per block; default
            ``[4, 2]`` (block 0 is the dense tail shape, block 1 is not).
        causal (bool): Whether the backbone is causal.

    Returns:
        NestorConfig: The config.
    """

    groups = [4, 4] if num_groups is None else num_groups
    top_k = [4, 2] if num_groups_per_token is None else num_groups_per_token
    n = len(groups)
    column = block_residual_mode == "attn_res_column"
    return NestorConfig(
        input_dim=16,
        num_blocks=n,
        feature_dims=[16] * n,
        pooling_strides=[1] * n,
        causal=causal,
        attention_token_dispatch=True,
        block_num_layers=[2] * n,
        num_groups=groups,
        num_groups_per_token=top_k,
        block_num_heads=[2] * n,
        block_head_dims=[8] * n,
        block_mlp_hidden_dims=[[16]] * n,
        router_num_experts_per_group=[1] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[1] * n,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode=block_residual_mode,
        block_column_include_aggregate=column,
        block_column_gain_init=0.5 if column else 1.0,
    )


def _build(config: NestorConfig, *, fused: bool, seed: int = 11, **env: str) -> Nestor:
    """Build ``config`` with the flag set or unset during construction.

    Args:
        config (NestorConfig): Backbone config.
        fused (bool): Whether ``NESTOR_DENSE_TAIL_ATTN=1`` is set while
            building (the flag is read in ``__init__``).
        seed (int): Parameter seed (identical for both twins).
        **env (str): Further flags to set during construction.

    Returns:
        Nestor: The backbone in eval mode.
    """

    with _FlagEnv(**{_FLAG: "1" if fused else "0"}, **env):
        torch.manual_seed(seed)
        model = Nestor(config)
    return model.eval()


def _inputs(seed: int, batch: int = 2, seq_len: int = 7) -> tuple[Tensor, Tensor, Tensor]:
    """A random input batch: features, positions, all-true base mask."""

    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, 16)
    pos = torch.arange(seq_len).expand(batch, -1)
    attn_mask = torch.ones(batch, seq_len, seq_len, dtype=torch.bool)
    return x, pos, attn_mask


def _param_grads(model: torch.nn.Module) -> dict[str, Tensor | None]:
    """Name -> gradient (None where autograd produced none)."""

    return {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in model.named_parameters()
    }


class DenseTailAttentionTest:
    """Fused all-columns attention == per-group token-dispatch loop."""

    def test_flag_is_static_and_requires_k_equals_g(self) -> None:
        """Only k == G, causal, token-dispatch, multi-group layers take it."""

        model = _build(_config(), fused=True)
        assert all(layer._dense_tail_attn for layer in model.blocks[0].layers)
        assert not any(layer._dense_tail_attn for layer in model.blocks[1].layers)
        off = _build(_config(), fused=False)
        assert not any(layer._dense_tail_attn for layer in off.blocks[0].layers)
        non_causal = _build(_config(causal=False), fused=True)
        assert not any(
            layer._dense_tail_attn for layer in non_causal.blocks[0].layers
        )

    def _forward_matches(self, block_residual_mode: str) -> None:
        config = _config(block_residual_mode=block_residual_mode)
        reference = _build(config, fused=False)
        fused = _build(config, fused=True)
        fused.load_state_dict(reference.state_dict())
        x, pos, attn_mask = _inputs(seed=3)
        with torch.no_grad():
            expected = reference(x, pos, attn_mask).pred
            actual = fused(x, pos, attn_mask).pred
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_forward_matches_projection_mode(self) -> None:
        """Plain residual path: the update is the gate-weighted column sum."""

        self._forward_matches("projection")

    def test_forward_matches_kimi_mode(self) -> None:
        """Block AttnRes reads the same merged update."""

        self._forward_matches("attn_res_kimi")

    def test_forward_matches_column_mode(self) -> None:
        """Column streams: the per-column outputs land in their slots."""

        self._forward_matches("attn_res_column")

    def _gradients_match(self, block_residual_mode: str, **env: str) -> None:
        config = _config(block_residual_mode=block_residual_mode)
        reference = _build(config, fused=False)
        fused = _build(config, fused=True, **env)
        fused.load_state_dict(reference.state_dict())
        x, pos, attn_mask = _inputs(seed=5)
        torch.manual_seed(6)
        weights = torch.randn(2, 7, 16)
        grads = []
        for model in (reference, fused):
            model.zero_grad(set_to_none=True)
            pred = model(x, pos, attn_mask).pred
            (pred * weights).sum().backward()
            grads.append(_param_grads(model))
        assert grads[0].keys() == grads[1].keys()
        for name, expected in grads[0].items():
            actual = grads[1][name]
            if expected is None:
                assert actual is None, name
                continue
            assert actual is not None, name
            torch.testing.assert_close(
                actual, expected, rtol=1e-4, atol=1e-5, msg=name
            )

    def test_gradients_match_projection_mode(self) -> None:
        """Every parameter's gradient agrees on the plain path."""

        self._gradients_match("projection")

    def test_gradients_match_column_mode(self) -> None:
        """Every parameter's gradient agrees under column streams."""

        self._gradients_match("attn_res_column")

    def test_checkpointed_backward_matches(self) -> None:
        """The static gate survives an activation-checkpoint recompute."""

        self._gradients_match("attn_res_column", **{_CKPT: "1"})

    def test_slot_layout_is_group_layout_gathered_at_topk(self) -> None:
        """Slot ``j`` of the slot layout holds column ``topk_indices[..., j]``."""

        from nestor.backbones.nestor import _build_column_stream_plan
        from nestor.constants import PADDING_TOKEN_POS

        model = _build(_config(block_residual_mode="attn_res_column"), fused=True)
        block = model.blocks[0]
        layer = block.layers[0]
        x, pos, _ = _inputs(seed=8)
        valid = pos != PADDING_TOKEN_POS
        router = block.group_router(x, valid)
        plan = _build_column_stream_plan(router.topk_indices, block.num_groups)
        norm_x = layer.attn_norm(x)
        with torch.no_grad():
            by_slot = layer._attn_update_dense_tail(norm_x, router, pos, plan)
            by_group = layer._attn_update_dense_tail(
                norm_x, router, pos, plan, group_layout=True
            )
            merged = layer._attn_update_dense_tail(norm_x, router, pos, None)
        index = router.topk_indices.unsqueeze(-1).expand(-1, -1, -1, 16)
        torch.testing.assert_close(by_slot, by_group.gather(2, index), rtol=0, atol=0)
        torch.testing.assert_close(merged, by_group.sum(2), rtol=1e-6, atol=1e-6)
