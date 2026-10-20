"""Compiled pointwise chains of the packed attention path.

``NESTOR_PACKED_POINTWISE_COMPILE=1`` swaps the eager op chains between the
packed path's GEMMs for ``torch.compile``'d helpers. The reference here is a
LITERAL transcription of the chains ``_attn_update_packed_varlen`` originally
ran inline (slow, obviously correct); the eager helpers must match it
bitwise, and the compiled helpers to floating-point reassociation noise
(Inductor may contract mul+add into an FMA and skip an intermediate-cast
rounding — <= 1-2 bf16 ulp, the bar every exact fast path is held to).
Gradients flow through the compiled helpers via AOTAutograd and must match
the reference chain's gradients.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor

from nestor.backbones.packed_pointwise import (
    attn_out_rows,
    attn_out_rows_compiled,
    gate_combine,
    gate_combine_compiled,
    packed_pointwise_requested,
    rope_qkv,
    rope_qkv_compiled,
)
from nestor.scaling.blocks import _rotate_half

# One bf16 ulp at the test data's scale (values up to ~8): 8 * 2^-8 = 0.03125;
# two ulps of headroom for an FMA-contracted chain.
BF16_TOL = 0.07


def _rope_reference(
    qkv: Tensor,
    bqkv: Tensor | None,
    gid: Tensor | None,
    cos: Tensor,
    sin: Tensor,
    inner_dim: int,
    num_heads: int,
    head_dim: int,
    pad: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """The original inline chain of ``_attn_update_packed_varlen``, verbatim."""

    rows = qkv.shape[0]
    e8 = qkv.shape[1] // 3
    compute_dtype = qkv.dtype
    if bqkv is not None:
        qkv = qkv + bqkv[gid].to(compute_dtype)
    qkv = qkv.view(rows, 3, e8)[..., :inner_dim].reshape(
        rows, 3, num_heads, head_dim
    )
    q32 = qkv[:, 0].float()
    k32 = qkv[:, 1].float()
    q = (q32 * cos + _rotate_half(q32) * sin).to(compute_dtype)
    k = (k32 * cos + _rotate_half(k32) * sin).to(compute_dtype)
    v = qkv[:, 2]
    if pad:
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
    q = q * (float(head_dim + pad) ** 0.5 / head_dim)
    return q, k, v


def _case(
    seed: int, num_heads: int, head_dim: int, *, bias: bool
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor, Tensor, int, int, int]:
    """Random inputs for one (heads, head_dim, bias) configuration."""

    gen = torch.Generator().manual_seed(seed)
    rows, groups = 48, 4
    inner_dim = num_heads * head_dim
    e8 = (inner_dim + 7) // 8 * 8
    qkv = torch.randn(rows, 3 * e8, generator=gen).to(torch.bfloat16)
    cos = torch.randn(rows, 1, head_dim, generator=gen)
    sin = torch.randn(rows, 1, head_dim, generator=gen)
    bqkv = torch.randn(groups, 3 * e8, generator=gen) if bias else None
    gid = (
        torch.randint(0, groups, (rows,), generator=gen) if bias else None
    )
    pad = (-head_dim) % 8
    return qkv, bqkv, gid, cos, sin, inner_dim, e8, pad


class RopeQkvTest:
    """Helper == inline chain, eager bitwise and compiled to bf16 noise."""

    def _check(self, seed: int, num_heads: int, head_dim: int, bias: bool) -> None:
        qkv, bqkv, gid, cos, sin, inner_dim, _, pad = _case(
            seed, num_heads, head_dim, bias=bias
        )
        args = (qkv, bqkv, gid, cos, sin, inner_dim, num_heads, head_dim, pad)
        ref = _rope_reference(*args)
        eager = rope_qkv(*args)
        compiled = rope_qkv_compiled(*args)
        for r, e, c in zip(ref, eager, compiled):
            assert torch.equal(e, r)
            torch.testing.assert_close(
                c.float(), r.float(), rtol=0.0, atol=BF16_TOL
            )

    def test_odd_head_dim_no_bias(self) -> None:
        self._check(seed=0, num_heads=3, head_dim=10, bias=False)

    def test_odd_head_dim_with_bias(self) -> None:
        self._check(seed=1, num_heads=3, head_dim=10, bias=True)

    def test_aligned_head_dim(self) -> None:
        """head_dim % 8 == 0: the pad branch is skipped on both forms."""

        self._check(seed=2, num_heads=2, head_dim=16, bias=False)
        self._check(seed=3, num_heads=2, head_dim=16, bias=True)

    def test_gradients_flow_through_compiled(self) -> None:
        """AOTAutograd gradients match the reference chain's gradients."""

        qkv, bqkv, gid, cos, sin, inner_dim, _, pad = _case(
            4, num_heads=3, head_dim=10, bias=True
        )
        qkv_ref = qkv.clone().requires_grad_(True)
        qkv_new = qkv.clone().requires_grad_(True)
        bqkv_ref = bqkv.clone().requires_grad_(True)
        bqkv_new = bqkv.clone().requires_grad_(True)
        ref = _rope_reference(
            qkv_ref, bqkv_ref, gid, cos, sin, inner_dim, 3, 10, pad
        )
        new = rope_qkv_compiled(
            qkv_new, bqkv_new, gid, cos, sin, inner_dim, 3, 10, pad
        )
        torch.manual_seed(5)
        weights = [torch.randn_like(t, dtype=torch.float32) for t in ref]
        sum((r.float() * w).sum() for r, w in zip(ref, weights)).backward()
        sum((n.float() * w).sum() for n, w in zip(new, weights)).backward()
        torch.testing.assert_close(
            qkv_new.grad.float(), qkv_ref.grad.float(), rtol=0.0, atol=BF16_TOL
        )
        # The bias gradient REDUCES ~rows/groups bf16-rounded row grads per
        # element, so eager-vs-compiled rounding differences accumulate;
        # judge it by relative Frobenius norm (the criterion the GPU packed
        # equivalence suite settled on for reduced gradients).
        diff = (bqkv_new.grad - bqkv_ref.grad).norm()
        assert diff <= 0.05 * bqkv_ref.grad.norm()


class AttnOutRowsTest:
    """Slice + flatten + tail pad between the varlen kernel and the out GEMM."""

    def _check(self, seed: int, num_heads: int, head_dim: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        rows = 48
        inner_dim = num_heads * head_dim
        pad = (-head_dim) % 8
        pad_e = (inner_dim + 7) // 8 * 8 - inner_dim
        out = torch.randn(rows, num_heads, head_dim + pad, generator=gen).to(
            torch.bfloat16
        )
        # Reference: the original inline chain, verbatim.
        ref = out
        if pad:
            ref = ref[..., :head_dim]
        ref = ref.reshape(rows, inner_dim)
        if pad_e:
            ref = F.pad(ref, (0, pad_e))
        eager = attn_out_rows(out, inner_dim, head_dim, pad, pad_e)
        compiled = attn_out_rows_compiled(out, inner_dim, head_dim, pad, pad_e)
        assert torch.equal(eager, ref)
        assert torch.equal(compiled, ref)

    def test_odd_and_aligned_widths(self) -> None:
        self._check(seed=0, num_heads=3, head_dim=10)  # pad and pad_e both > 0
        self._check(seed=1, num_heads=2, head_dim=16)  # neither


class GateCombineTest:
    """Output bias + upcast + gate multiply after the out GEMM."""

    def _check(self, seed: int, *, bias: bool) -> None:
        gen = torch.Generator().manual_seed(seed)
        rows, groups, dim = 48, 4, 24
        out_c = torch.randn(rows, dim, generator=gen).to(torch.bfloat16)
        gates = torch.rand(rows, generator=gen)
        bo = torch.randn(groups, dim, generator=gen) if bias else None
        gid = torch.randint(0, groups, (rows,), generator=gen) if bias else None
        # Reference: the original inline chain, verbatim.
        ref_c = out_c
        if bo is not None:
            ref_c = ref_c + bo[gid].to(ref_c.dtype)
        ref = ref_c.to(torch.float32) * gates.unsqueeze(-1).to(torch.float32)
        eager = gate_combine(out_c, bo, gid, gates, torch.float32)
        compiled = gate_combine_compiled(out_c, bo, gid, gates, torch.float32)
        assert torch.equal(eager, ref)
        torch.testing.assert_close(compiled, ref, rtol=0.0, atol=BF16_TOL)

    def test_with_and_without_bias(self) -> None:
        self._check(seed=0, bias=False)
        self._check(seed=1, bias=True)


class FlagReaderTest:
    """``packed_pointwise_requested`` is exactly the ``=1`` test of the flag."""

    def test_flag_reader_follows_env(self) -> None:
        key = "NESTOR_PACKED_POINTWISE_COMPILE"
        previous = os.environ.get(key)
        try:
            os.environ.pop(key, None)
            assert packed_pointwise_requested() is False
            os.environ[key] = "0"
            assert packed_pointwise_requested() is False
            os.environ[key] = "1"
            assert packed_pointwise_requested() is True
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
