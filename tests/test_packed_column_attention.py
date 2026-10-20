"""Equivalence tests for layer-packed varlen attention in column mode.

``NESTOR_ATTN_PACKED_VARLEN=1`` now also serves ``attn_res_column`` layers:
:meth:`~nestor.backbones.nestor.RoutedTransformerLayer._attn_update_packed_varlen`
takes a block-level :class:`~nestor.backbones.nestor._PackedAttentionPlan`,
packs with one gather, and unpacks straight into the streams' ``[batch, seq,
k, dim]`` slot layout. Together with ``NESTOR_COLUMN_SLOT_UPDATES=1`` the
column sublayer never builds a group-wide tensor. The per-group token-dispatch
loop (flags off) is the reference implementation throughout -- and so is the
production per-group varlen path (``NESTOR_GATHERED_VARLEN=1``), which the
packed path replaces.

Every test builds the SAME model twice under a fixed seed, once per flag
state; eval mode keeps router noise off. The dirty-allocator test is the
regression the varlen family requires of every path that writes a subset of a
buffer: fresh CUDA allocations are zero pages, so a path that leaves rows
unwritten only NaNs in a real training loop.

Discrete routing sensitivity (learned the expensive way): the
packed path differs from the loop at the bf16 level (~4e-3, grouped GEMM and
varlen accumulation order), and a top-1 expert pool downstream flips one
token's expert on exactly that much -- a 0.2 difference on a few elements
that is NOT an attention error (the repo's pool arms show the same in eval,
with identical loss). So the end-to-end models here route every pool with
``top_k == pool size`` (continuous softmax mixing, same dispatch code, no
flip to take), and :meth:`PackedColumnAttentionTest.test_layer_update_matches_loop`
pins the attention update itself, per block, at the layer level.

CUDA only: ``varlen_attn`` and ``torch._grouped_mm`` have no CPU kernels.
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.expert_pools import ExpertPoolMLP

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="varlen/grouped_mm are CUDA-only"
)

_PACKED = "NESTOR_ATTN_PACKED_VARLEN"
_SLOT = "NESTOR_COLUMN_SLOT_UPDATES"
_TAIL = "NESTOR_DENSE_TAIL_ATTN"
_VARLEN = "NESTOR_GATHERED_VARLEN"
_FLASH = "NESTOR_GATHERED_CAUSAL_FLASH"
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
    block_residual_mode: str = "attn_res_column",
    num_heads: int = 2,
    head_dim: int = 8,
) -> NestorConfig:
    """Tiny causal token-dispatch model: k=2/4, k=4/4, then k=1/1, pooled.

    Args:
        block_residual_mode (str): ``"attn_res_column"`` (head-only reads)
            or ``"attn_res_kimi"``.
        num_heads (int): Heads per column (3 x 12 = 36 exercises the %8
            padding of the grouped GEMM operands and the varlen head pad).
        head_dim (int): Per-head width.

    Returns:
        NestorConfig: The config.
    """

    groups = [4, 4, 1]
    top_k = [2, 4, 1]
    n = len(groups)
    column = block_residual_mode == "attn_res_column"
    config = NestorConfig(
        input_dim=32,
        num_blocks=n,
        feature_dims=[32] * n,
        pooling_strides=[1] * n,
        causal=True,
        attention_token_dispatch=True,
        block_num_layers=[2, 1, 1],
        num_groups=groups,
        num_groups_per_token=top_k,
        block_num_heads=[num_heads] * n,
        block_head_dims=[head_dim] * n,
        block_mlp_hidden_dims=[[24]] * n,
        router_num_experts_per_group=[1] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[1] * n,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode=block_residual_mode,
        block_column_include_aggregate=column,
        block_column_gain_init=0.5 if column else 1.0,
        block_column_head_only=column,
        block_column_deviation_sources=column,
    )
    # top_k == pool size: continuous mixing, no discrete choice for bf16
    # noise to flip (see the module docstring).
    config.block_expert_pool_public_experts = [2] * n
    config.block_expert_pool_public_top_k = [2] * n
    config.block_expert_pool_private_experts = [2] * n
    config.block_expert_pool_private_top_k = [2] * n
    config.block_expert_pool_private_gate = "switch"
    return config


def _build(config: NestorConfig, *, seed: int = 51, **env: str) -> Nestor:
    """Build ``config`` on CUDA under ``env`` flags (read at construction)."""

    with _FlagEnv(**env):
        torch.manual_seed(seed)
        model = Nestor(config)
    for module in model.modules():
        if isinstance(module, ExpertPoolMLP):
            module.grouped_dispatch = True
    return model.cuda().eval()


def _inputs(seed: int, batch: int = 3, seq_len: int = 40) -> tuple[Tensor, Tensor, Tensor]:
    """A random CUDA batch: features, positions, all-true base mask."""

    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, 32, device="cuda")
    pos = torch.arange(seq_len, device="cuda").expand(batch, -1)
    attn_mask = torch.ones(batch, seq_len, seq_len, dtype=torch.bool, device="cuda")
    return x, pos, attn_mask


def _run(model: Nestor, x: Tensor, pos: Tensor, attn_mask: Tensor, weights: Tensor) -> tuple[Tensor, dict[str, Tensor | None]]:
    """One bf16-autocast forward/backward; returns pred and parameter grads."""

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x, pos, attn_mask).pred
    (pred.float() * weights).sum().backward()
    grads = {
        name: (None if p.grad is None else p.grad.detach().float().clone())
        for name, p in model.named_parameters()
    }
    return pred.detach().float(), grads


def _assert_agree(reference: Nestor, candidate: Nestor, *, seed: int) -> None:
    """Forward and every parameter gradient agree to bf16 tolerance.

    The forward is held to 3e-2 element-wise. Gradients are held to a
    relative Frobenius norm of 10% per tensor: through three bf16 blocks the
    element-wise noise on reduction gradients that CANCEL across tokens (a
    router score-bias sums ~500 signed O(1) terms into O(1); a norm bias sums
    120 tokens) reaches ~9% of a single entry's scale while the tensor as a
    whole agrees to a few percent -- a real backward defect (a wrong row map,
    a dropped gate) is off by O(100%). An absolute floor of 1% of the
    largest gradient norm covers the cancelling reductions (router score
    biases, measured ~0.3% of the global scale end-to-end) and tensors whose
    whole gradient sits at the noise floor. The tight, per-element guards on the
    packed path's own gradients (input, projections, norm, and the router
    gates it feeds) are in
    :meth:`PackedColumnAttentionTest.test_layer_backward_matches_loop`.
    """

    candidate.load_state_dict(reference.state_dict())
    x, pos, attn_mask = _inputs(seed=seed)
    torch.manual_seed(seed + 1)
    weights = torch.randn_like(x)
    pred_ref, grads_ref = _run(reference, x, pos, attn_mask, weights)
    pred_new, grads_new = _run(candidate, x, pos, attn_mask, weights)
    assert torch.isfinite(pred_new).all()
    torch.testing.assert_close(pred_new, pred_ref, rtol=3e-2, atol=3e-2)
    assert grads_ref.keys() == grads_new.keys()
    # Absolute floor at 1% of the model's largest gradient norm. Measured
    # in production: cancelling reductions -- router score biases
    # summing hundreds of signed O(1) terms into O(0.1) -- differ by up to
    # ~0.3% of the global scale end-to-end in bf16, and a depth-read gain at
    # its zero-gradient init has no relative accuracy at all. Everything the
    # packed path itself computes is guarded per element at the layer level.
    global_scale = max(
        (g.norm().item() for g in grads_ref.values() if g is not None), default=1.0
    )
    for name, expected in grads_ref.items():
        actual = grads_new[name]
        if expected is None:
            assert actual is None, name
            continue
        assert actual is not None, name
        assert torch.isfinite(actual).all(), name
        ref_norm = expected.norm().item()
        err_norm = (actual - expected).norm().item()
        assert err_norm <= 0.10 * ref_norm + 1e-2 * global_scale, (
            f"{name}: ||grad diff|| {err_norm:.4g} > 10% of ||grad|| {ref_norm:.4g} "
            f"+ 1% x {global_scale:.4g}"
        )


class PackedColumnAttentionTest:
    """Layer-packed column attention == per-group loop, on CUDA."""

    def test_flag_applies_to_multi_column_layers(self) -> None:
        """Every multi-column causal layer takes the packed path, no other."""

        from nestor.backbones.nestor import _PackedAttentionPlan
        from nestor.constants import PADDING_TOKEN_POS

        model = _build(_config(), **{_PACKED: "1"})
        assert all(layer._attn_packed_varlen for layer in model.blocks[0].layers)
        assert all(layer._attn_packed_varlen for layer in model.blocks[1].layers)
        assert not any(layer._attn_packed_varlen for layer in model.blocks[2].layers)
        # The block builds the packed plan (not the top-1 dispatch plan), and
        # the plan's invariants hold: every row belongs to a segment
        # (cu[-1] == rows), the grouped-GEMM offsets end at rows, and every
        # real (token, slot) pair owns one distinct packed row.
        block = model.blocks[0]
        x, pos, _ = _inputs(seed=59)
        with torch.no_grad():
            router = block.group_router(x, pos != PADDING_TOKEN_POS)
            plan = block._layer_dispatch_plan(router, pos)
        assert isinstance(plan, _PackedAttentionPlan)
        assert plan.rows == 3 * 40 * 2 + 16 * 4
        assert int(plan.cu[-1]) == plan.rows and int(plan.offs[-1]) == plan.rows
        assert plan.slot_rows.unique().numel() == 3 * 40 * 2
        assert plan.gates_packed[plan.slot_rows].gt(0).all()
        assert int(plan.row_real.sum()) == 3 * 40 * 2
        assert plan.row_real[plan.slot_rows].all()
        assert (plan.gates_packed[~plan.row_real] == 0).all()
        assert block._layer_dispatch_plan(router, pos) is not None

    def test_column_mode_matches_per_group_loop(self) -> None:
        """Packed + slot updates vs the padded per-group loop (flags off)."""

        config = _config()
        _assert_agree(
            _build(config), _build(config, **{_PACKED: "1", _SLOT: "1"}), seed=60
        )

    def test_column_mode_matches_production_varlen_loop(self) -> None:
        """Packed + slot updates vs the per-group varlen path production runs."""

        config = _config()
        _assert_agree(
            _build(config, **{_VARLEN: "1", _FLASH: "1"}),
            _build(config, **{_PACKED: "1", _SLOT: "1", _TAIL: "1"}),
            seed=61,
        )

    def test_packed_without_slot_updates_matches(self) -> None:
        """Packed attention feeds the legacy column bookkeeping too."""

        config = _config()
        _assert_agree(_build(config), _build(config, **{_PACKED: "1"}), seed=62)

    def test_layer_update_matches_loop(self) -> None:
        """Per block, the packed attention update equals the per-group loop's.

        The layer-level check that is immune to downstream routing flips:
        same layer, same router output, same normalized input, the packed
        path (with the block's plan) against the loop, in bf16 autocast.
        """

        from nestor.backbones.nestor import _build_packed_attention_plan
        from nestor.constants import PADDING_TOKEN_POS

        config = _config(block_residual_mode="attn_res_kimi")
        loop = _build(config)
        packed = _build(config, **{_PACKED: "1"})
        packed.load_state_dict(loop.state_dict())
        x, pos, _ = _inputs(seed=69)
        valid = pos != PADDING_TOKEN_POS
        base_mask = torch.tril(
            torch.ones(40, 40, dtype=torch.bool, device="cuda")
        ).expand(3, -1, -1)
        for blk in (0, 1):
            block_l, block_p = loop.blocks[blk], packed.blocks[blk]
            with torch.no_grad():
                router = block_l.group_router(x, valid)
                plan = _build_packed_attention_plan(
                    router, pos, block_p.layers[0].attention_groups[0].rope, block_p.num_groups
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    norm_x = block_l.layers[0].attn_norm(x)
                    ref = block_l.layers[0]._attn_update_general(
                        norm_x, base_mask, router, pos
                    )
                    got = block_p.layers[0]._attn_update_packed_varlen(
                        norm_x, router, pos, plan=plan
                    )
            torch.testing.assert_close(
                got.float(), ref.float(), rtol=2e-2, atol=2e-2, msg=f"block {blk}"
            )

    def test_layer_backward_matches_loop(self) -> None:
        """Per block, the packed path's gradients equal the loop's.

        Same layer, same router output, same random upstream gradient: the
        input gradient (through the gather-only pack backward), the GATE
        gradient (through the plan's packed gates -- what the group router
        trains on), and every attention-group / norm parameter gradient
        (through the grouped-GEMM and varlen backwards, with zeroed tail
        rows) match the per-group loop per element to bf16 noise (measured
        <= 1.3% of each tensor's scale on the parameters).
        """

        from nestor.backbones.nestor import _build_packed_attention_plan
        from nestor.constants import PADDING_TOKEN_POS

        config = _config(block_residual_mode="attn_res_kimi")
        loop = _build(config)
        packed = _build(config, **{_PACKED: "1"})
        packed.load_state_dict(loop.state_dict())
        x0, pos, _ = _inputs(seed=70)
        valid = pos != PADDING_TOKEN_POS
        base_mask = torch.tril(
            torch.ones(40, 40, dtype=torch.bool, device="cuda")
        ).expand(3, -1, -1)
        for blk in (0, 1):
            block_l, block_p = loop.blocks[blk], packed.blocks[blk]
            with torch.no_grad():
                router_fixed = block_l.group_router(x0, valid)
            torch.manual_seed(71 + blk)
            d_out = torch.randn(3, 40, 32, device="cuda")
            grads: dict[str, dict[str, Tensor]] = {}
            for name, layer in (("loop", block_l.layers[0]), ("packed", block_p.layers[0])):
                x = x0.clone().requires_grad_(True)
                # The gates are a differentiable input too: the packed path
                # feeds the group router through the plan's packed gates.
                gates = router_fixed.gates.detach().clone().requires_grad_(True)
                router = router_fixed._replace(gates=gates)
                plan = _build_packed_attention_plan(
                    router, pos, block_p.layers[0].attention_groups[0].rope, block_p.num_groups
                )
                layer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    norm_x = layer.attn_norm(x)
                    y = (
                        layer._attn_update_packed_varlen(norm_x, router, pos, plan=plan)
                        if name == "packed"
                        else layer._attn_update_general(norm_x, base_mask, router, pos)
                    )
                (y.float() * d_out).sum().backward()
                assert x.grad is not None and gates.grad is not None
                grads[name] = {"x": x.grad.float().clone(), "gates": gates.grad.float().clone()}
                for pn, p in layer.named_parameters():
                    if p.grad is not None and ("attention_groups" in pn or "attn_norm" in pn):
                        grads[name][pn] = p.grad.float().clone()
            assert grads["loop"].keys() == grads["packed"].keys()
            for pn, expected in grads["loop"].items():
                scale = max(expected.abs().max().item(), 1e-6)
                torch.testing.assert_close(
                    grads["packed"][pn], expected, rtol=0.0, atol=3e-2 * scale,
                    msg=lambda m, pn=pn, blk=blk: f"block {blk} {pn}: {m}",
                )

    def test_kimi_mode_matches_per_group_loop(self) -> None:
        """The merged (non-column) packed output still equals the loop."""

        config = _config(block_residual_mode="attn_res_kimi")
        _assert_agree(_build(config), _build(config, **{_PACKED: "1"}), seed=63)

    def test_odd_widths_take_the_padding_branches(self) -> None:
        """3 heads x 12: inner width 36 (grouped-GEMM pad) and head pad 12->16."""

        config = _config(num_heads=3, head_dim=12)
        _assert_agree(
            _build(config), _build(config, **{_PACKED: "1", _SLOT: "1"}), seed=64
        )

    def test_checkpointed_backward_matches(self) -> None:
        """The static gate and the block-level plan survive a recompute."""

        config = _config()
        _assert_agree(
            _build(config), _build(config, **{_PACKED: "1", _SLOT: "1", _CKPT: "1"}), seed=65
        )

    def test_block_plan_equals_on_the_fly_plan(self) -> None:
        """A layer given the block's plan computes exactly what it computes alone."""

        from nestor.backbones.nestor import (
            _build_column_stream_plan,
            _build_packed_attention_plan,
        )
        from nestor.constants import PADDING_TOKEN_POS

        model = _build(_config(), **{_PACKED: "1"})
        block = model.blocks[0]
        layer = block.layers[0]
        x, pos, _ = _inputs(seed=66)
        valid = pos != PADDING_TOKEN_POS
        with torch.no_grad():
            router = block.group_router(x, valid)
            column_plan = _build_column_stream_plan(router.topk_indices, block.num_groups)
            plan = _build_packed_attention_plan(
                router, pos, layer.attention_groups[0].rope, block.num_groups
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                norm_x = layer.attn_norm(x)
                alone = layer._attn_update_packed_varlen(norm_x, router, pos, column_plan)
                planned = layer._attn_update_packed_varlen(
                    norm_x, router, pos, column_plan, plan=plan
                )
                merged = layer._attn_update_packed_varlen(norm_x, router, pos)
        torch.testing.assert_close(planned, alone, rtol=0, atol=0)
        assert alone.shape == (3, 40, 2, 32)
        torch.testing.assert_close(
            merged.float(), alone.float().sum(2), rtol=2e-2, atol=2e-2
        )
        # Every (token, slot) pair of a real token owns exactly one packed row.
        assert plan.slot_rows.shape[0] == 3 * 40 * 2
        assert plan.slot_rows.unique().numel() == 3 * 40 * 2

    def test_backward_finite_with_dirty_allocator(self) -> None:
        """Gradients stay finite when the allocator hands back inf garbage."""

        model = _build(_config(), **{_PACKED: "1", _SLOT: "1", _TAIL: "1"})
        x, pos, attn_mask = _inputs(seed=67)
        rows = 3 * 40 * 2 + 16 * 4
        for numel in (rows * 32, rows * 64, rows * 16, 2 * rows * 32, rows * 96):
            poison = torch.full(
                (numel,), float("inf"), device="cuda", dtype=torch.bfloat16
            )
            del poison
        torch.manual_seed(68)
        weights = torch.randn_like(x)
        pred, grads = _run(model, x, pos, attn_mask, weights)
        assert torch.isfinite(pred).all()
        for name, grad in grads.items():
            if grad is not None:
                assert torch.isfinite(grad).all(), f"non-finite grad in {name}"


_POINTWISE = "NESTOR_PACKED_POINTWISE_COMPILE"


class PackedPointwiseCompileCudaTest:
    """The compiled pointwise chains inside the packed path, end to end.

    ``NESTOR_PACKED_POINTWISE_COMPILE=1`` swaps the eager RoPE / padding /
    gate chains of ``_attn_update_packed_varlen`` for ``torch.compile``'d
    helpers (unit-level equivalence in ``tests/test_packed_pointwise.py``).
    Here the whole model must agree with the per-group loop end to end, the
    layer must agree with the eager helpers per element, and helpers being
    actually COMPILED (not silently eager) is asserted through dynamo's
    counters after a cache reset.
    """

    def test_matches_eager_helpers_layer_level(self) -> None:
        """Per block, compiled == eager pointwise chains, forward and backward.

        The layer-level pin for exactly the changed code: same layer, same
        plan, same upstream gradient — the only difference is whether the
        chains run compiled. A MODEL-level pw-vs-eager comparison is
        deliberately not used: it measured its miss as 13.9% on ONE
        cancelling-reduction tensor (``final_depth_read.value_gains``) with
        the forward and every other gradient inside the criterion — the
        noise-amplification class `_assert_agree`'s docstring records, not
        an integration defect (the full-distance end-to-end check against
        the flags-off loop is :meth:`test_matches_per_group_loop`).
        """

        from nestor.backbones.nestor import _build_packed_attention_plan
        from nestor.constants import PADDING_TOKEN_POS

        config = _config(block_residual_mode="attn_res_kimi")
        eager = _build(config, **{_PACKED: "1"})
        compiled = _build(config, **{_PACKED: "1", _POINTWISE: "1"})
        compiled.load_state_dict(eager.state_dict())
        x0, pos, _ = _inputs(seed=70)
        valid = pos != PADDING_TOKEN_POS
        for blk in (0, 1):
            block_e, block_c = eager.blocks[blk], compiled.blocks[blk]
            assert block_c.layers[0]._packed_pointwise
            assert not block_e.layers[0]._packed_pointwise
            with torch.no_grad():
                router_fixed = block_e.group_router(x0, valid)
            torch.manual_seed(73 + blk)
            d_out = torch.randn(3, 40, 32, device="cuda")
            grads: dict[str, dict[str, Tensor]] = {}
            outs: dict[str, Tensor] = {}
            for name, layer in (
                ("eager", block_e.layers[0]),
                ("compiled", block_c.layers[0]),
            ):
                x = x0.clone().requires_grad_(True)
                gates = router_fixed.gates.detach().clone().requires_grad_(True)
                router = router_fixed._replace(gates=gates)
                plan = _build_packed_attention_plan(
                    router, pos, layer.attention_groups[0].rope, block_e.num_groups
                )
                layer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    norm_x = layer.attn_norm(x)
                    y = layer._attn_update_packed_varlen(
                        norm_x, router, pos, plan=plan
                    )
                outs[name] = y.detach().float().clone()
                (y.float() * d_out).sum().backward()
                assert x.grad is not None and gates.grad is not None
                grads[name] = {
                    "x": x.grad.float().clone(),
                    "gates": gates.grad.float().clone(),
                }
                for pn, p in layer.named_parameters():
                    if p.grad is not None and (
                        "attention_groups" in pn or "attn_norm" in pn
                    ):
                        grads[name][pn] = p.grad.float().clone()
            torch.testing.assert_close(
                outs["compiled"], outs["eager"], rtol=2e-2, atol=2e-2,
                msg=f"block {blk} forward",
            )
            assert grads["eager"].keys() == grads["compiled"].keys()
            for pn, expected in grads["eager"].items():
                scale = max(expected.abs().max().item(), 1e-6)
                torch.testing.assert_close(
                    grads["compiled"][pn], expected, rtol=0.0, atol=3e-2 * scale,
                    msg=lambda m, pn=pn, blk=blk: f"block {blk} {pn}: {m}",
                )

    def test_matches_per_group_loop(self) -> None:
        """Compiled pointwise + packed path vs the flags-off reference."""

        config = _config()
        _assert_agree(
            _build(config),
            _build(
                config, **{_PACKED: "1", _SLOT: "1", _TAIL: "1", _POINTWISE: "1"}
            ),
            seed=71,
        )

    def test_helpers_actually_compile_under_disable(self) -> None:
        """The compiled helpers must compile INSIDE the disabled packed method.

        ``_attn_update_packed_varlen`` is ``torch.compiler.disable``'d; an
        explicitly compiled helper called from that frame must still hit
        dynamo (torch 2.10 and 2.13 do; a torch that silently suppressed the
        nested compile would fall back to eager — same math, no speedup — and
        this test documents which regime the running torch is in). Dynamo's
        caches and counters are cleared first: the module-level helpers stay
        warm across tests in one process, and a warm cache adds no new frame
        (an earlier ``assert 12 > 12`` false negative).
        """

        import torch._dynamo as dynamo

        model = _build(
            _config(), **{_PACKED: "1", _SLOT: "1", _TAIL: "1", _POINTWISE: "1"}
        )
        x, pos, attn_mask = _inputs(seed=72)
        dynamo.reset()  # force fresh compiles; later tests just recompile
        dynamo.utils.counters.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, pos, attn_mask)
        frames = dynamo.utils.counters["frames"]["ok"]
        assert frames > 0, (
            "no dynamo frame compiled: the nested torch.compile'd pointwise "
            "helpers were suppressed inside the disabled packed method on "
            "this torch version"
        )
