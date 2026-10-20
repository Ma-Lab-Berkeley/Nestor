"""Equivalence tests for slot-layout column sublayers.

``NESTOR_COLUMN_SLOT_UPDATES=1`` routes
:meth:`~nestor.backbones.nestor.RoutedTransformerLayer.forward_column` through
``_forward_column_slot_updates``: attention and expert updates arrive in the
streams' own ``[batch, seq, k, dim]`` slot layout (the pool via
:meth:`~nestor.backbones.expert_pools.ExpertPoolMLP.forward_per_slot`) and the
sublayer's bookkeeping collapses to one ``k``-wide add. The group-layout path
(flag off -- the scatter-fused ``forward_column``) is the reference
implementation throughout and must agree in the forward, in every parameter
gradient, with and without head-only / deviation reads, with and without
expert pools, on both pool dispatch paths, and under activation checkpointing.

Every test builds the SAME model twice under a fixed seed -- once per flag
state -- so the bookkeeping is the only difference. Eval mode keeps router
noise off. CPU-runnable: the grouped expert dispatch is forced on (its
``segment_mm`` falls back to a segment loop) so the slot-major grouped rows are
exercised, not just the sparse fallback.
"""

from __future__ import annotations

import os
from typing import Self

import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.expert_pools import ExpertPoolMLP

_FLAG = "NESTOR_COLUMN_SLOT_UPDATES"
_TAIL = "NESTOR_DENSE_TAIL_ATTN"
_FUSED = "NESTOR_GROUPED_MM_FUSED_GATE"
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
    pools: bool = True,
    public_experts: int = 2,
    head_only: bool = True,
    deviation: bool = True,
) -> NestorConfig:
    """Tiny causal token-dispatch column model: k=2/4, k=4/4, then k=1/1.

    Block 0 exercises the per-group loop with a slot scatter, block 1 the
    dense-tail shape, block 2 the single-column ``_to_single_slot`` path.

    Args:
        pools (bool): Whether every layer's MLP is an ``ExpertPoolMLP``
            (public + private pools, switch gate) or the plain expert MLP.
        public_experts (int): Public pool size (0 = private-only pools).
        head_only (bool): ``block_column_head_only``.
        deviation (bool): ``block_column_deviation_sources``.

    Returns:
        NestorConfig: The config.
    """

    groups = [4, 4, 1]
    top_k = [2, 4, 1]
    n = len(groups)
    config = NestorConfig(
        input_dim=16,
        num_blocks=n,
        feature_dims=[16] * n,
        pooling_strides=[1] * n,
        causal=True,
        attention_token_dispatch=True,
        block_num_layers=[2, 1, 1],
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
        block_residual_mode="attn_res_column",
        block_column_include_aggregate=True,
        block_column_gain_init=0.5,
        block_column_head_only=head_only,
        block_column_deviation_sources=deviation,
    )
    if pools:
        config.block_expert_pool_public_experts = [public_experts] * n
        config.block_expert_pool_public_top_k = [1] * n
        config.block_expert_pool_private_experts = [2] * n
        config.block_expert_pool_private_top_k = [1] * n
        config.block_expert_pool_private_gate = "switch"
    return config


def _build(config: NestorConfig, *, seed: int = 21, grouped: bool = True, **env: str) -> Nestor:
    """Build ``config`` under ``env`` flags (read at construction).

    Args:
        config (NestorConfig): Backbone config.
        seed (int): Parameter seed.
        grouped (bool): Force the pools' grouped-GEMM dispatch (works on CPU
            through ``segment_mm``'s fallback).
        **env (str): Flags to set while building.

    Returns:
        Nestor: The backbone in eval mode.
    """

    with _FlagEnv(**env):
        torch.manual_seed(seed)
        model = Nestor(config)
    for module in model.modules():
        if isinstance(module, ExpertPoolMLP):
            module.grouped_dispatch = grouped
    return model.eval()


def _inputs(seed: int, batch: int = 2, seq_len: int = 6) -> tuple[Tensor, Tensor, Tensor]:
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


def _assert_models_agree(
    reference: Nestor, candidate: Nestor, *, seed: int, rtol: float = 1e-5, atol: float = 1e-5
) -> None:
    """Forward and every parameter gradient agree between two twins."""

    candidate.load_state_dict(reference.state_dict())
    x, pos, attn_mask = _inputs(seed=seed)
    torch.manual_seed(seed + 1)
    weights = torch.randn_like(x)
    preds, grads = [], []
    for model in (reference, candidate):
        model.zero_grad(set_to_none=True)
        pred = model(x, pos, attn_mask).pred
        (pred * weights).sum().backward()
        preds.append(pred.detach())
        grads.append(_param_grads(model))
    assert torch.isfinite(preds[1]).all()
    torch.testing.assert_close(preds[1], preds[0], rtol=rtol, atol=atol)
    assert grads[0].keys() == grads[1].keys()
    for name, expected in grads[0].items():
        actual = grads[1][name]
        if expected is None:
            assert actual is None, name
            continue
        assert actual is not None, name
        torch.testing.assert_close(actual, expected, rtol=rtol * 10, atol=atol, msg=name)


class ColumnSlotUpdatesTest:
    """Slot-layout column sublayers == the group-layout reference."""

    def test_flag_is_read_at_construction(self) -> None:
        """The flag lands on every routed layer, and only when set."""

        on = _build(_config(), **{_FLAG: "1"})
        off = _build(_config(), **{_FLAG: "0"})
        assert all(layer._column_slot_updates for b in on.blocks for layer in b.layers)
        assert not any(layer._column_slot_updates for b in off.blocks for layer in b.layers)

    def test_head_only_deviation_pools(self) -> None:
        """The paper shape: head-only + deviation reads, hybrid pools."""

        config = _config(head_only=True, deviation=True)
        _assert_models_agree(_build(config), _build(config, **{_FLAG: "1"}), seed=30)

    def test_full_reads_pools(self) -> None:
        """Full column reads (every stream a source), hybrid pools."""

        config = _config(head_only=False, deviation=False)
        _assert_models_agree(_build(config), _build(config, **{_FLAG: "1"}), seed=31)

    def test_full_reads_with_deviation(self) -> None:
        """Full reads in the aggregate-plus-deviations basis."""

        config = _config(head_only=False, deviation=True)
        _assert_models_agree(_build(config), _build(config, **{_FLAG: "1"}), seed=32)

    def test_private_only_pools(self) -> None:
        """No public family: the private per-slot update is the whole MLP."""

        config = _config(public_experts=0)
        _assert_models_agree(_build(config), _build(config, **{_FLAG: "1"}), seed=33)

    def test_without_pools_broadcasts(self) -> None:
        """A column-agnostic MLP broadcasts one update onto every slot."""

        config = _config(pools=False)
        _assert_models_agree(_build(config), _build(config, **{_FLAG: "1"}), seed=34)

    def test_sparse_pool_dispatch(self) -> None:
        """The sparse (non-grouped) pool path gathers per-group -> per-slot."""

        config = _config()
        _assert_models_agree(
            _build(config, grouped=False), _build(config, grouped=False, **{_FLAG: "1"}), seed=35
        )

    def test_stacked_with_dense_tail_and_fused_gate(self) -> None:
        """All three exact fast paths together vs every flag off."""

        config = _config()
        _assert_models_agree(
            _build(config), _build(config, **{_FLAG: "1", _TAIL: "1", _FUSED: "1"}), seed=36
        )

    def test_checkpointed_backward_matches(self) -> None:
        """Slot updates survive an activation-checkpoint recompute."""

        config = _config()
        _assert_models_agree(
            _build(config), _build(config, **{_FLAG: "1", _CKPT: "1"}), seed=37
        )


class ExpertPoolPerSlotTest:
    """``forward_per_slot`` is ``forward_per_group`` gathered at the slots."""

    def _pool_inputs(self, grouped: bool, seed: int):
        """A pooled layer's inputs and its block's router output."""

        from nestor.constants import PADDING_TOKEN_POS

        model = _build(_config(), grouped=grouped)
        block = model.blocks[0]
        layer = block.layers[0]
        x, pos, _ = _inputs(seed=seed)
        valid = pos != PADDING_TOKEN_POS
        router = block.group_router(x, valid)
        return layer.expert_pool, layer.mlp_norm(x), valid, router

    def _check(self, grouped: bool, seed: int) -> None:
        pool, norm_x, valid, router = self._pool_inputs(grouped, seed)
        with torch.no_grad():
            public_g, private_g, _, _ = pool.forward_per_group(
                norm_x, valid, router.gates, router.active_mask
            )
            public_s, private_s, _, _ = pool.forward_per_slot(
                norm_x, valid, router.gates, router.active_mask, router.topk_indices
            )
        index = router.topk_indices.unsqueeze(-1).expand(-1, -1, -1, norm_x.shape[-1])
        torch.testing.assert_close(private_s, private_g.gather(2, index), rtol=1e-6, atol=1e-6)
        assert public_g is not None and public_s is not None
        torch.testing.assert_close(public_s, public_g, rtol=0, atol=0)
        # The slot layout sums to the merged forward's private part too.
        assert private_s.shape[2] == router.topk_indices.shape[-1]

    def test_grouped_dispatch(self) -> None:
        """Grouped-GEMM rows are slot-major before any scatter."""

        self._check(grouped=True, seed=40)

    def test_sparse_dispatch(self) -> None:
        """The sparse fallback gathers the per-group tensor at the slots."""

        self._check(grouped=False, seed=41)

    def test_explicit_slot_order_changes_rows_not_sum(self) -> None:
        """``_private_slots(selected=...)`` only permutes the dispatched rows."""

        pool, norm_x, valid, router = self._pool_inputs(True, seed=42)
        flat_x = norm_x.reshape(-1, norm_x.shape[-1])
        num_tokens = flat_x.shape[0]
        gates = router.gates.reshape(num_tokens, -1)
        _, private_outputs, _ = pool._route(norm_x, valid, router.active_mask)
        ids_a, w_a, sel_a = pool._private_slots(private_outputs, gates, num_tokens)
        ids_b, w_b, sel_b = pool._private_slots(
            private_outputs, gates, num_tokens, selected=router.topk_indices
        )
        assert set(map(tuple, sel_a.tolist())) == set(map(tuple, sel_b.tolist())) or (
            torch.sort(sel_a, dim=1).values.equal(torch.sort(sel_b, dim=1).values)
        )
        torch.testing.assert_close(
            torch.sort(ids_a, dim=1).values, torch.sort(ids_b, dim=1).values, rtol=0, atol=0
        )
        torch.testing.assert_close(w_a.sum(1), w_b.sum(1), rtol=1e-6, atol=1e-6)
