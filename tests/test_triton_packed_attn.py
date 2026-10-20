"""Equivalence tests for the Triton-fused packed column attention.

``NESTOR_TRITON_PACKED_ATTN=1`` replaces every op between the packed path's
grouped GEMMs — the pack gather, the bias/RoPE/pad/scale chain, and the
gate + slot-gather combine — with fused Triton kernels over a head-padded
weight layout (``nestor/backbones/triton_packed.py``). The reference
implementation throughout is the plain packed path (flags identical minus the
Triton flag), and behind it the per-group token-dispatch loop; both bars are
the ones every exact fast path is held to (bf16 reassociation noise on the
forward, relative-Frobenius on reduced gradients).

Every model test builds the SAME model twice under a fixed seed, once per
flag state; eval mode keeps router noise off. The dirty-allocator test is the
regression the varlen family requires of every path that writes a subset of a
buffer. The function-level classes pin each fused kernel against its eager
chain on synthetic packed layouts (alignment tails included).

CUDA + triton only (one flag-reader test runs anywhere).
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.expert_pools import ExpertPoolMLP
from nestor.backbones.triton_packed import (
    triton_packed_available,
    triton_packed_requested,
)
from nestor.scaling.blocks import _rotate_half

_CUDA = pytest.mark.skipif(
    not triton_packed_available(), reason="needs CUDA + triton"
)

_PACKED = "NESTOR_ATTN_PACKED_VARLEN"
_SLOT = "NESTOR_COLUMN_SLOT_UPDATES"
_TRITON = "NESTOR_TRITON_PACKED_ATTN"

# One bf16 ulp at the test data's scale, two ulps of headroom for FMA
# contraction / one-rounding-fewer chains (same bar as test_packed_pointwise).
BF16_TOL = 0.07


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
    num_heads: int = 2,
    head_dim: int = 8,
    block_residual_mode: str = "attn_res_column",
) -> NestorConfig:
    """Tiny causal token-dispatch model: k=2/4, k=4/4, then k=1/1, pooled.

    Args:
        num_heads (int): Heads per column.
        head_dim (int): Per-head width (10 exercises the head-pad lanes).
        block_residual_mode (str): ``"attn_res_column"`` (the head-only read
            the suite above pins), ``"attn_res_kimi"`` or ``"projection"`` (the
            read-ablation modes); the column-read knobs are only
            set under the column mode.

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
    # top_k == pool size: continuous mixing, no discrete expert choice for
    # bf16 noise to flip (a documented failure mode).
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

    Same criteria as the packed-path equivalence suite: forward at 3e-2
    element-wise; gradients at 10% relative Frobenius per tensor with an
    absolute floor of 1% of the largest gradient norm (cancelling reductions
    — router score biases and the like — sit at the bf16 noise floor).
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


def _dirty_allocator() -> None:
    """Poison the CUDA caching allocator so unwritten rows read garbage."""

    junk = [
        torch.full((1 << 20,), float("nan"), device="cuda") for _ in range(16)
    ]
    del junk
    torch.cuda.synchronize()


@_CUDA
class TritonPackedAttnModelTest:
    """Triton-fused packed path == plain packed path == per-group loop."""

    def test_flag_selects_triton_path(self) -> None:
        """Multi-column causal layers take the Triton path, no other."""

        model = _build(_config(), **{_PACKED: "1", _SLOT: "1", _TRITON: "1"})
        assert all(layer._triton_packed for layer in model.blocks[0].layers)
        assert all(layer._triton_packed for layer in model.blocks[1].layers)
        assert not any(layer._triton_packed for layer in model.blocks[2].layers)
        off = _build(_config(), **{_PACKED: "1", _SLOT: "1"})
        assert not any(
            layer._triton_packed
            for block in off.blocks
            for layer in block.layers
        )

    def test_model_matches_packed_reference(self) -> None:
        """Same model, plain packed vs Triton packed, forward and grads."""

        reference = _build(_config(), **{_PACKED: "1", _SLOT: "1"})
        candidate = _build(
            _config(), **{_PACKED: "1", _SLOT: "1", _TRITON: "1"}
        )
        _assert_agree(reference, candidate, seed=11)

    def test_model_matches_pergroup_loop(self) -> None:
        """Triton packed vs the per-group dispatch loop (all flags off)."""

        reference = _build(_config())
        candidate = _build(
            _config(), **{_PACKED: "1", _SLOT: "1", _TRITON: "1"}
        )
        _assert_agree(reference, candidate, seed=13)

    def test_odd_head_dim_pad_lanes(self) -> None:
        """head_dim=10 -> hdp=16: the pad-lane stores are exercised."""

        cfg = _config(num_heads=3, head_dim=10)
        reference = _build(cfg, **{_PACKED: "1", _SLOT: "1"})
        candidate = _build(cfg, **{_PACKED: "1", _SLOT: "1", _TRITON: "1"})
        _assert_agree(reference, candidate, seed=17)

    def test_dirty_allocator(self) -> None:
        """A poisoned allocator changes nothing: every output row is written."""

        cfg = _config(num_heads=3, head_dim=10)
        reference = _build(cfg, **{_PACKED: "1", _SLOT: "1"})
        candidate = _build(cfg, **{_PACKED: "1", _SLOT: "1", _TRITON: "1"})
        _dirty_allocator()
        _assert_agree(reference, candidate, seed=19)


# The production kernel stack, minus
# NESTOR_STACKED_EXPERT_BANKS (it renames parameters, so state-dict transfer
# to the eager reference would fail; it has its own equivalence suite).
_FULL_STACK: dict[str, str] = {
    _PACKED: "1",
    _SLOT: "1",
    _TRITON: "1",
    "NESTOR_DENSE_TAIL_ATTN": "1",
    "NESTOR_GROUPED_MM_FUSED_GATE": "1",
    "NESTOR_TRITON_INDEX_BUILD": "1",
    "NESTOR_TRITON_EXPERT_DISPATCH": "1",
}


@_CUDA
class TritonPackedReadModesTest:
    """The kernel fast paths are exact under the NON-column reads too.

    The read ablation trains column arms with
    ``block_residual_mode="projection"`` (no depth read) and
    ``"attn_res_kimi"`` under the Triton kernel stack, while the suites above
    pin only the head-only (``attn_res_column``) read — under which the Triton
    packed path returns per-slot streams; under the other modes it takes
    the ``column_plan is None`` branch (slot-sum into the merged update).
    Each mode is held to the same two bars: the plain packed
    reference, and the per-group dispatch loop with every flag off — the
    latter also under the full flag stack.
    """

    def _matches_packed_reference(self, block_residual_mode: str) -> None:
        cfg = _config(block_residual_mode=block_residual_mode)
        reference = _build(cfg, **{_PACKED: "1", _SLOT: "1"})
        candidate = _build(cfg, **{_PACKED: "1", _SLOT: "1", _TRITON: "1"})
        _assert_agree(reference, candidate, seed=23)

    def _matches_pergroup_loop_full_stack(self, block_residual_mode: str) -> None:
        cfg = _config(block_residual_mode=block_residual_mode)
        reference = _build(cfg)
        candidate = _build(cfg, **_FULL_STACK)
        _assert_agree(reference, candidate, seed=29)

    def test_projection_matches_packed_reference(self) -> None:
        """No depth read: Triton packed == plain packed."""

        self._matches_packed_reference("projection")

    def test_kimi_matches_packed_reference(self) -> None:
        """Kimi depth read: Triton packed == plain packed."""

        self._matches_packed_reference("attn_res_kimi")

    def test_projection_full_stack_matches_pergroup_loop(self) -> None:
        """No depth read: the launchers' full flag stack == all flags off."""

        self._matches_pergroup_loop_full_stack("projection")

    def test_kimi_full_stack_matches_pergroup_loop(self) -> None:
        """Kimi depth read: the launchers' full flag stack == all flags off."""

        self._matches_pergroup_loop_full_stack("attn_res_kimi")


def _bijection(
    seed: int, num_tokens: int, top_k: int, num_groups: int
) -> dict[str, Tensor]:
    """A synthetic packed layout: every (token, slot) pair owns one row.

    Mirrors the plan invariants: ``rows = num_tokens * top_k + 16``
    (alignment tail), tails clamped with exactly-zero gates, group ids
    non-decreasing (group-major rows).

    Args:
        seed (int): RNG seed.
        num_tokens (int): Flat token count.
        top_k (int): Slots per token.
        num_groups (int): Groups (for ``gid`` and biases).

    Returns:
        dict[str, Tensor]: tok_packed, row_real, slot_rows, slot_packed,
        gates, gid on CUDA.
    """

    gen = torch.Generator().manual_seed(seed)
    pairs = num_tokens * top_k
    rows = pairs + 16
    perm = torch.randperm(pairs, generator=gen)
    slot_packed = torch.cat([perm, torch.full((16,), pairs - 1)])
    slot_rows = torch.empty(pairs, dtype=torch.long)
    slot_rows[perm] = torch.arange(pairs)
    tok_packed = torch.cat(
        [perm.div(top_k, rounding_mode="floor"), torch.full((16,), num_tokens - 1)]
    )
    row_real = torch.cat(
        [torch.ones(pairs, dtype=torch.bool), torch.zeros(16, dtype=torch.bool)]
    )
    gates = torch.rand(rows, generator=gen)
    gates[pairs:] = 0.0
    gid = torch.sort(torch.randint(0, num_groups, (rows,), generator=gen)).values
    return {
        name: t.cuda()
        for name, t in {
            "tok_packed": tok_packed,
            "row_real": row_real,
            "slot_rows": slot_rows,
            "slot_packed": slot_packed,
            "gates": gates,
            "gid": gid,
        }.items()
    }


@_CUDA
class TritonPackRowsFnTest:
    """Fused pack gather == cast + ``_PackedRowGather``, values and grads."""

    def test_forward_and_backward(self) -> None:
        from nestor.backbones.nestor import _PackedRowGather
        from nestor.backbones.triton_packed import pack_rows

        maps = _bijection(seed=0, num_tokens=24, top_k=3, num_groups=4)
        torch.manual_seed(1)
        x_ref = torch.randn(24, 40, device="cuda", requires_grad=True)
        x_new = x_ref.detach().clone().requires_grad_(True)
        ref = _PackedRowGather.apply(
            x_ref.to(torch.bfloat16),
            maps["tok_packed"],
            maps["row_real"],
            maps["slot_rows"],
            3,
        )
        new = pack_rows(
            x_new,
            maps["tok_packed"],
            maps["row_real"],
            maps["slot_rows"],
            3,
            torch.bfloat16,
        )
        assert torch.equal(new, ref)
        weights = torch.randn_like(ref, dtype=torch.float32)
        (ref.float() * weights).sum().backward()
        (new.float() * weights).sum().backward()
        torch.testing.assert_close(
            x_new.grad, x_ref.grad, rtol=0.0, atol=BF16_TOL
        )


@_CUDA
class TritonRopeQkvFnTest:
    """Fused RoPE-QKV == the eager head-padded chain, values and grads."""

    def _reference(self, qkv, bqkv, gid, cos, sin, num_heads, head_dim, hdp):
        """The plain path's chain, transcribed onto the head-padded layout."""

        rows = qkv.shape[0]
        dtype = qkv.dtype
        if bqkv is not None:
            qkv = qkv + bqkv[gid].to(dtype)
        qkv = qkv.view(rows, 3, num_heads, hdp)[..., :head_dim]
        q32 = qkv[:, 0].float()
        k32 = qkv[:, 1].float()
        q = (q32 * cos + _rotate_half(q32) * sin).to(dtype)
        k = (k32 * cos + _rotate_half(k32) * sin).to(dtype)
        v = qkv[:, 2]
        pad = hdp - head_dim
        if pad:
            q = F.pad(q, (0, pad))
            k = F.pad(k, (0, pad))
            v = F.pad(v, (0, pad))
        q = q * (float(hdp) ** 0.5 / head_dim)
        return q, k, v

    def _check(self, seed: int, num_heads: int, head_dim: int, bias: bool) -> None:
        from nestor.backbones.triton_packed import rope_qkv_headpad

        gen = torch.Generator().manual_seed(seed)
        rows, groups = 80, 4
        hdp = head_dim + ((-head_dim) % 8)
        width = 3 * num_heads * hdp
        qkv = torch.randn(rows, width, generator=gen).to(torch.bfloat16).cuda()
        cos = torch.randn(rows, 1, head_dim, generator=gen).cuda()
        sin = torch.randn(rows, 1, head_dim, generator=gen).cuda()
        bqkv = (
            torch.randn(groups, width, generator=gen).cuda() if bias else None
        )
        gid = (
            torch.sort(
                torch.randint(0, groups, (rows,), generator=gen)
            ).values.cuda()
            if bias
            else None
        )
        scale = float(hdp) ** 0.5 / head_dim
        qkv_ref = qkv.clone().requires_grad_(True)
        qkv_new = qkv.clone().requires_grad_(True)
        b_ref = bqkv.clone().requires_grad_(True) if bias else None
        b_new = bqkv.clone().requires_grad_(True) if bias else None
        ref = self._reference(
            qkv_ref, b_ref, gid, cos, sin, num_heads, head_dim, hdp
        )
        new = rope_qkv_headpad(
            qkv_new, b_new, gid, cos, sin, num_heads, head_dim, hdp, scale
        )
        for r, n in zip(ref, new):
            torch.testing.assert_close(
                n.float(), r.float(), rtol=0.0, atol=BF16_TOL
            )
        torch.manual_seed(seed + 1)
        weights = [torch.randn_like(t, dtype=torch.float32) for t in ref]
        sum((r.float() * w).sum() for r, w in zip(ref, weights)).backward()
        sum((n.float() * w).sum() for n, w in zip(new, weights)).backward()
        torch.testing.assert_close(
            qkv_new.grad.float(), qkv_ref.grad.float(), rtol=0.0, atol=BF16_TOL
        )
        if bias:
            diff = (b_new.grad - b_ref.grad).norm()
            assert diff <= 0.05 * b_ref.grad.norm()

    def test_aligned_head_dim(self) -> None:
        self._check(seed=0, num_heads=2, head_dim=16, bias=False)
        self._check(seed=1, num_heads=2, head_dim=16, bias=True)

    def test_odd_head_dim(self) -> None:
        self._check(seed=2, num_heads=3, head_dim=10, bias=False)
        self._check(seed=3, num_heads=3, head_dim=10, bias=True)

    def test_production_widths(self) -> None:
        """The 500m shape: 5 heads x head_dim 60 -> hdp 64."""

        self._check(seed=4, num_heads=5, head_dim=60, bias=False)


@_CUDA
class TritonGateSlotFnTest:
    """Fused combine == ``gate_combine`` + ``_PackedSlotGather``."""

    def _check(self, seed: int, *, bias: bool, out_dtype: torch.dtype) -> None:
        from nestor.backbones.nestor import _PackedSlotGather
        from nestor.backbones.packed_pointwise import gate_combine
        from nestor.backbones.triton_packed import gate_combine_slots

        maps = _bijection(seed=seed, num_tokens=32, top_k=2, num_groups=4)
        rows = maps["gates"].shape[0]
        gen = torch.Generator().manual_seed(seed + 100)
        out_c = torch.randn(rows, 24, generator=gen).to(torch.bfloat16).cuda()
        bo = torch.randn(4, 24, generator=gen).cuda() if bias else None
        oc_ref = out_c.clone().requires_grad_(True)
        oc_new = out_c.clone().requires_grad_(True)
        bo_ref = bo.clone().requires_grad_(True) if bias else None
        bo_new = bo.clone().requires_grad_(True) if bias else None
        gates_ref = maps["gates"].clone().requires_grad_(True)
        gates_new = maps["gates"].clone().requires_grad_(True)
        gated = gate_combine(
            oc_ref, bo_ref, maps["gid"] if bias else None, gates_ref, out_dtype
        )
        ref = _PackedSlotGather.apply(
            gated, maps["slot_rows"], maps["slot_packed"]
        )
        new = gate_combine_slots(
            oc_new,
            bo_new,
            maps["gid"] if bias else None,
            gates_new,
            maps["slot_rows"],
            maps["slot_packed"],
            out_dtype,
        )
        torch.testing.assert_close(
            new.float(), ref.float(), rtol=0.0, atol=BF16_TOL
        )
        torch.manual_seed(seed + 200)
        weights = torch.randn_like(ref, dtype=torch.float32)
        (ref.float() * weights).sum().backward()
        (new.float() * weights).sum().backward()
        torch.testing.assert_close(
            oc_new.grad.float(), oc_ref.grad.float(), rtol=0.0, atol=BF16_TOL
        )
        # Gate gradients REDUCE over dim; judge by relative Frobenius (the
        # packed suites' criterion for reduced gradients). Tail rows never
        # reach the plan's scatter backward, so compare real rows only.
        pairs = maps["slot_rows"].shape[0]
        real = torch.zeros(rows, dtype=torch.bool, device="cuda")
        real[: pairs] = True
        g_ref = gates_ref.grad[real]
        g_new = gates_new.grad[real]
        assert (g_new - g_ref).norm() <= 0.05 * g_ref.norm() + 1e-6
        if bias:
            diff = (bo_new.grad - bo_ref.grad).norm()
            assert diff <= 0.05 * bo_ref.grad.norm() + 1e-6

    def test_fp32_out(self) -> None:
        self._check(seed=0, bias=False, out_dtype=torch.float32)
        self._check(seed=1, bias=True, out_dtype=torch.float32)

    def test_bf16_out(self) -> None:
        self._check(seed=2, bias=False, out_dtype=torch.bfloat16)
        self._check(seed=3, bias=True, out_dtype=torch.bfloat16)


class FlagReaderTest:
    """``triton_packed_requested`` is exactly the ``=1`` test of the flag."""

    def test_flag_reader_follows_env(self) -> None:
        key = "NESTOR_TRITON_PACKED_ATTN"
        previous = os.environ.get(key)
        try:
            os.environ.pop(key, None)
            assert triton_packed_requested() is False
            os.environ[key] = "0"
            assert triton_packed_requested() is False
            os.environ[key] = "1"
            assert triton_packed_requested() is True
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
