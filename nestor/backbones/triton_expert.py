"""Fused Triton kernels for the grouped-GEMM expert dispatch.

:func:`~nestor.backbones.grouped_mm.grouped_expert_rows` keeps its grouped
GEMMs (``torch._grouped_mm``, efficient) but everything around them runs as
eager gather / pointwise kernels, exactly the situation
:mod:`nestor.backbones.triton_packed` fixed for the packed attention path.
Per bank call the reference chain is, forward: ``x.to(bf16)`` +
``index_select`` + ``* pad_valid`` (dispatch), ``silu`` + ``*`` on strided
slices of the fused up|gate GEMM output, ``index_select`` (combine) and, in
the caller, ``rows * weights`` (an fp32 ``[rows, dim]`` intermediate — half a
gigabyte per private-pool call at 500m) + ``.to`` + ``.sum(slots)``. The
backward is worse: the two slice views of the up|gate output each
materialize a zero-filled ``[rows_pad, 2*hidden]`` gradient that are then
added, the weighting's backward is four kernels, the dispatch/combine gathers
three each. ~30 launches and ~8 GB of traffic per private call, twice per
column layer, 26 layers at 500m.

``NESTOR_TRITON_EXPERT_DISPATCH=1`` replaces the region with three fused
forward/backward kernel pairs, each exact against its reference chain (same
op order and dtypes up to floating-point reassociation and transcendental
ulps — the <= 1-2 bf16 ulp bar every exact fast path is held to):

1. :func:`dispatch_rows` — cast + gather + slack-zero in ONE kernel, backward
   a fused ``slots``-way gather-sum. This IS the packed path's ``pack_rows`` kernel
   (:class:`~nestor.backbones.triton_packed._TritonPackRows`): the padded
   dispatch is the same bijection contract (``pad_token`` / ``pad_valid`` /
   ``row_dest`` play ``tok_packed`` / ``row_real`` / ``slot_rows``).
2. :func:`swiglu` / :func:`swiglu_fused` — ``silu(gate) * up`` in ONE kernel
   reading the up|gate GEMM output in place; the backward writes the whole
   ``[rows_pad, 2*hidden]`` gradient directly (no zero-fill, no slice grads,
   no add) and recomputes the activation instead of saving it.
3. :func:`combine_rows` — combine gather + weighting + ``.to`` + slot-sum in
   ONE kernel; the weighted ``[rows, dim]`` intermediate never exists. The
   backward produces the padded-row gradient and the per-row weight gradient
   (a fused dot product) in one pass over the padded rows.

The flag is read at CONSTRUCTION (:func:`triton_expert_requested`) by
``ExpertPoolMLP``, so flag-on/off twins coexist in one process (the paired
profiling harness). It changes no parameter (resume-safe) and no math. Its
DEFAULT follows ``NESTOR_TRITON_INDEX_BUILD``: a launcher that asked for the
Triton dispatch-map build asked for the Triton expert-dispatch path;
``NESTOR_TRITON_EXPERT_DISPATCH=0`` opts out explicitly.

Scope: bias-free SwiGLU banks (the expert pools). Biased or non-SwiGLU banks
(the native MoE bank of ``RoutedTransformerLayer``) keep the eager chain.

Nothing here imports from ``nestor.backbones.grouped_mm`` (the caller), so
there is no import cycle. Triton is imported guardedly.
"""

from __future__ import annotations

import os

import torch
from jaxtyping import Float, Int
from torch import Tensor

from nestor.backbones.triton_packed import _TritonPackRows

try:  # pragma: no cover - trivially platform-dependent
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

# The kernels reference ONLY ``tl``/``triton`` names: the expert-pool region
# runs inside a compiled graph, so dynamo captures these kernels as
# user-defined Triton kernels and Inductor re-compiles their SOURCE in a
# worker subprocess where module-level aliases (e.g. a libdevice import) do
# not exist (``NameError`` at compile). ``tl.exp`` (ex2-based, ~2 fp32 ulp)
# vs ATen's ``expf`` is invisible after the bf16 rounding except for rare
# 1-ulp flips — within the bar every exact fast path is held to.


def triton_expert_requested() -> bool:
    """Whether the fused Triton expert-dispatch path is asked for.

    ``NESTOR_TRITON_EXPERT_DISPATCH`` when set; otherwise the value of
    ``NESTOR_TRITON_INDEX_BUILD`` (see the module docstring for why the
    default follows the index-build flag). Read at CONSTRUCTION by the
    expert-bearing modules, never inside the forward.

    Returns:
        bool: True iff the effective value is ``"1"``.
    """

    value = os.environ.get("NESTOR_TRITON_EXPERT_DISPATCH")
    if value is None:
        value = os.environ.get("NESTOR_TRITON_INDEX_BUILD", "0")
    return value == "1"


def triton_cat_cast_requested() -> bool:
    """Whether the fused cat+cast of the stacked up|gate bank is asked for.

    ``NESTOR_TRITON_EXPERT_CATCAST`` (default ``"1"``); only meaningful under
    the round-4 path. A construction-time knob so the paired profiler can
    price this kernel pair on its own (``=0`` keeps the torch cat+cast).

    Returns:
        bool: True unless set to something other than ``"1"``.
    """

    return os.environ.get("NESTOR_TRITON_EXPERT_CATCAST", "1") == "1"


def triton_expert_available() -> bool:
    """Whether the fused kernels can actually run here.

    Returns:
        bool: True iff triton imported and a CUDA device is visible.
    """

    return _HAS_TRITON and torch.cuda.is_available()


def bank_is_fusable(experts: list, fused_gate: bool) -> bool:
    """Whether a bank qualifies for the fused path: bias-free SwiGLU.

    Args:
        experts (list): The bank (``GatedScaledMLP``-style modules).
        fused_gate (bool): Unused today (both GEMM layouts are supported);
            kept so call sites can widen the rule without a signature change.

    Returns:
        bool: True for gated ``silu`` experts without up/down biases.
    """

    first = experts[0]
    return (
        bool(getattr(first, "gated", False))
        and getattr(first, "activation", None) == "silu"
        and first.layers[0].bias is None
        and first.layers[1].bias is None
    )


# --------------------------------------------------------------------------
# Kernels (defined only when triton is importable).
# --------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(
        up_ptr,  # [rows, hidden] up-projection rows (bf16), row stride STRIDE_IN
        gate_ptr,  # [rows, hidden] gate-projection rows (bf16), same stride
        out_ptr,  # [rows, hidden] silu(gate) * up (bf16), row stride STRIDE_OUT
        n_rows,
        hidden,
        stride_in,
        stride_out,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rmask = rows < n_rows
        cmask = cols < hidden
        mask = rmask[:, None] & cmask[None, :]
        rows = rows.to(tl.int64)  # offsets can exceed int32 at rows * stride
        ibase = rows[:, None] * stride_in + cols[None, :]
        u = tl.load(up_ptr + ibase, mask=mask, other=0.0)
        g = tl.load(gate_ptr + ibase, mask=mask, other=0.0)
        ety = up_ptr.dtype.element_ty
        gf = g.to(tl.float32)
        # ATen silu: x / (1 + exp(-x)) in fp32, rounded to the input dtype.
        act = (gf / (1.0 + tl.exp(-gf))).to(ety)
        out = (act.to(tl.float32) * u.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rows[:, None] * stride_out + cols[None, :], out, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        gout_ptr,  # [rows, hidden] gradient of silu(gate) * up (bf16), stride STRIDE_G
        up_ptr,  # forward inputs, row stride STRIDE_IN
        gate_ptr,
        gup_ptr,  # [rows, hidden] out: grad wrt up, row stride STRIDE_OUT
        ggate_ptr,  # [rows, hidden] out: grad wrt gate, row stride STRIDE_OUT
        n_rows,
        hidden,
        stride_in,
        stride_out,
        stride_g,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rmask = rows < n_rows
        cmask = cols < hidden
        mask = rmask[:, None] & cmask[None, :]
        rows = rows.to(tl.int64)
        ibase = rows[:, None] * stride_in + cols[None, :]
        go = tl.load(gout_ptr + rows[:, None] * stride_g + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + ibase, mask=mask, other=0.0)
        g = tl.load(gate_ptr + ibase, mask=mask, other=0.0)
        ety = up_ptr.dtype.element_ty
        gf = g.to(tl.float32)
        e = tl.exp(-gf)
        s = 1.0 / (1.0 + e)
        # The mul's backward sees the saved bf16 activation and bf16 up.
        act = (gf / (1.0 + e)).to(ety)
        grad_up = (go * act.to(tl.float32)).to(ety)
        grad_act = (go * u.to(tl.float32)).to(ety)
        # ATen silu_backward: dy * s * (1 + x * (1 - s)) in fp32.
        grad_gate = (grad_act.to(tl.float32) * s * (1.0 + gf * (1.0 - s))).to(ety)
        obase = rows[:, None] * stride_out + cols[None, :]
        tl.store(gup_ptr + obase, grad_up, mask=mask)
        tl.store(ggate_ptr + obase, grad_gate, mask=mask)

    @triton.jit
    def _cat_cast_fwd_kernel(
        a_ptr,  # [rows, hidden] fp32 (the up bank, leading dims flattened)
        b_ptr,  # [rows, hidden] fp32 (the gate bank)
        out_ptr,  # [rows, 2*hidden] out: [a | b] in the GEMM dtype
        n_rows,
        hidden,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rmask = rows < n_rows
        cmask = cols < hidden
        mask = rmask[:, None] & cmask[None, :]
        rows = rows.to(tl.int64)
        ibase = rows[:, None] * hidden + cols[None, :]
        obase = rows[:, None] * (2 * hidden) + cols[None, :]
        oty = out_ptr.dtype.element_ty
        a = tl.load(a_ptr + ibase, mask=mask, other=0.0)
        b = tl.load(b_ptr + ibase, mask=mask, other=0.0)
        tl.store(out_ptr + obase, a.to(oty), mask=mask)
        tl.store(out_ptr + obase + hidden, b.to(oty), mask=mask)

    @triton.jit
    def _cat_cast_bwd_kernel(
        g_ptr,  # [rows, 2*hidden] gradient (GEMM dtype)
        ga_ptr,  # [rows, hidden] out: fp32 grad of the up bank, contiguous
        gb_ptr,  # [rows, hidden] out: fp32 grad of the gate bank, contiguous
        n_rows,
        hidden,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rmask = rows < n_rows
        cmask = cols < hidden
        mask = rmask[:, None] & cmask[None, :]
        rows = rows.to(tl.int64)
        gbase = rows[:, None] * (2 * hidden) + cols[None, :]
        obase = rows[:, None] * hidden + cols[None, :]
        ga = tl.load(g_ptr + gbase, mask=mask, other=0.0)
        gb = tl.load(g_ptr + gbase + hidden, mask=mask, other=0.0)
        tl.store(ga_ptr + obase, ga.to(ga_ptr.dtype.element_ty), mask=mask)
        tl.store(gb_ptr + obase, gb.to(gb_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _combine_fwd_kernel(
        y_ptr,  # [rows_pad, dim] down-GEMM output (bf16)
        row_dest_ptr,  # [rows] int64 routed row -> padded slot
        w_ptr,  # [rows] combine weights
        out_ptr,  # [n_out, dim] weighted sums over each GROUP consecutive rows
        n_out,
        dim,
        GROUP: tl.constexpr,
        ROUND_TERM: tl.constexpr,  # round each weighted term to out dtype before summing
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        orow = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        omask = orow < n_out
        cmask = cols < dim
        mask = omask[:, None] & cmask[None, :]
        orow = orow.to(tl.int64)
        oty = out_ptr.dtype.element_ty
        acc = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
        for e in tl.static_range(GROUP):
            r = orow * GROUP + e  # routed row (token-major, slot-major)
            p = tl.load(row_dest_ptr + r, mask=omask, other=0)
            w = tl.load(w_ptr + r, mask=omask, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + p[:, None] * dim + cols[None, :], mask=mask, other=0.0)
            term = y.to(tl.float32) * w[:, None]
            if ROUND_TERM:
                term = term.to(oty).to(tl.float32)
            acc += term
        tl.store(out_ptr + orow[:, None] * dim + cols[None, :], acc.to(oty), mask=mask)

    @triton.jit
    def _combine_bwd_kernel(
        gout_ptr,  # [n_out, dim] gradient of the combined output
        y_ptr,  # [rows_pad, dim] forward down-GEMM output (bf16)
        pad_src_ptr,  # [rows_pad] int64 padded slot -> routed row (0 on slack)
        pad_valid_ptr,  # [rows_pad] 1 on real slots, 0 on slack
        w_ptr,  # [rows] combine weights
        gy_ptr,  # [rows_pad, dim] out: grad wrt y (bf16); slack rows exactly 0
        gw_ptr,  # [rows] out: grad wrt weights (fp32)
        n_pad,
        dim,
        GROUP: tl.constexpr,
        NEED_W: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        p = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        pmask = p < n_pad
        p = p.to(tl.int64)
        src = tl.load(pad_src_ptr + p, mask=pmask, other=0)
        valid = tl.load(pad_valid_ptr + p, mask=pmask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + src, mask=pmask, other=0.0).to(tl.float32)
        orow = src // GROUP
        gty = gy_ptr.dtype.element_ty
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
        for off in range(0, dim, BLOCK_D):
            cols = off + tl.arange(0, BLOCK_D)
            cmask = cols < dim
            mask = pmask[:, None] & cmask[None, :]
            go = tl.load(gout_ptr + orow[:, None] * dim + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            # Reference: (grad * w) in the promoted dtype, cast to y's dtype,
            # then * pad_valid (exactly 1 or 0) in y's dtype.
            gy = (go * w[:, None]).to(gty) * valid.to(gty)[:, None]
            tl.store(gy_ptr + p[:, None] * dim + cols[None, :], gy, mask=mask)
            if NEED_W:
                y = tl.load(y_ptr + p[:, None] * dim + cols[None, :], mask=mask, other=0.0).to(tl.float32)
                acc += tl.sum(go * y, axis=1)
        if NEED_W:
            # Every real routed row owns exactly one valid padded slot.
            tl.store(gw_ptr + src, acc, mask=pmask & (valid != 0))


# --------------------------------------------------------------------------
# Autograd wrappers.
# --------------------------------------------------------------------------
class _TritonSwiGLU(torch.autograd.Function):
    """``silu(gate) * up`` over two separate ``[rows, hidden]`` tensors."""

    @staticmethod
    def forward(ctx, up, gate):  # noqa: D102
        up = up.contiguous()
        gate = gate.contiguous()
        rows, hidden = up.shape
        out = torch.empty_like(up)
        grid = (triton.cdiv(rows, 32), triton.cdiv(hidden, 128))
        _swiglu_fwd_kernel[grid](
            up, gate, out, rows, hidden, hidden, hidden, BLOCK_R=32, BLOCK_D=128
        )
        ctx.save_for_backward(up, gate)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        up, gate = ctx.saved_tensors
        rows, hidden = up.shape
        grad_out = grad_out.contiguous()
        grad_up = torch.empty_like(up)
        grad_gate = torch.empty_like(gate)
        grid = (triton.cdiv(rows, 32), triton.cdiv(hidden, 128))
        _swiglu_bwd_kernel[grid](
            grad_out, up, gate, grad_up, grad_gate, rows, hidden,
            hidden, hidden, hidden, BLOCK_R=32, BLOCK_D=128,
        )
        return grad_up, grad_gate


class _TritonSwiGLUFused(torch.autograd.Function):
    """``silu(gate) * up`` over the fused ``[rows, 2*hidden]`` up|gate layout.

    Reads both halves in place (the reference slices them into strided
    views) and writes the WHOLE ``[rows, 2*hidden]`` input gradient in one
    kernel — the reference's two zero-filled slice gradients and their add
    never exist. Saves only the GEMM output; the activation is recomputed.
    """

    @staticmethod
    def forward(ctx, up_gate, hidden):  # noqa: D102
        up_gate = up_gate.contiguous()
        rows, width = up_gate.shape
        out = torch.empty(rows, hidden, device=up_gate.device, dtype=up_gate.dtype)
        grid = (triton.cdiv(rows, 32), triton.cdiv(hidden, 128))
        _swiglu_fwd_kernel[grid](
            up_gate, up_gate[:, hidden:], out, rows, hidden, width, hidden,
            BLOCK_R=32, BLOCK_D=128,
        )
        ctx.save_for_backward(up_gate)
        ctx.hidden = hidden
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        (up_gate,) = ctx.saved_tensors
        hidden = ctx.hidden
        rows, width = up_gate.shape
        grad_out = grad_out.contiguous()
        grad_ug = torch.empty_like(up_gate)
        grid = (triton.cdiv(rows, 32), triton.cdiv(hidden, 128))
        _swiglu_bwd_kernel[grid](
            grad_out, up_gate, up_gate[:, hidden:], grad_ug, grad_ug[:, hidden:],
            rows, hidden, width, width, hidden, BLOCK_R=32, BLOCK_D=128,
        )
        return grad_ug, None


class _TritonCatCast(torch.autograd.Function):
    """``torch.cat([a, b], -1).to(dtype)`` with a coalesced split backward.

    Forward reads the two fp32 banks once and writes the concatenated GEMM
    operand. Backward reads the ``[.., 2*hidden]`` gradient once and writes
    each bank's fp32 gradient CONTIGUOUSLY — the reference's backward (two
    strided halves of the gradient cast, or a contiguous cast followed by two
    strided grad accumulations) moves the same bytes through unrolled strided
    copies at ~3x the bandwidth-bound time (55 x 181 us per
    500m micro-step). Bit-exact: a cast is a cast.
    """

    @staticmethod
    def forward(ctx, a, b, dtype):  # noqa: D102
        a = a.contiguous()
        b = b.contiguous()
        hidden = a.shape[-1]
        rows = a.numel() // hidden
        out = torch.empty(*a.shape[:-1], 2 * hidden, device=a.device, dtype=dtype)
        grid = (triton.cdiv(rows, 32), triton.cdiv(hidden, 128))
        _cat_cast_fwd_kernel[grid](a, b, out, rows, hidden, BLOCK_R=32, BLOCK_D=128)
        ctx.shape = a.shape
        ctx.in_dtype = a.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        grad_out = grad_out.contiguous()
        hidden = ctx.shape[-1]
        rows = grad_out.numel() // (2 * hidden)
        ga = torch.empty(ctx.shape, device=grad_out.device, dtype=ctx.in_dtype)
        gb = torch.empty(ctx.shape, device=grad_out.device, dtype=ctx.in_dtype)
        grid = (triton.cdiv(rows, 32), triton.cdiv(hidden, 128))
        _cat_cast_bwd_kernel[grid](grad_out, ga, gb, rows, hidden, BLOCK_R=32, BLOCK_D=128)
        return ga, gb, None


class _TritonCombineRows(torch.autograd.Function):
    """Combine gather + weighting + cast + group-sum in one kernel each way."""

    @staticmethod
    def forward(ctx, y_pad, row_dest, pad_src_row, pad_valid, weights, group, out_dtype):  # noqa: D102
        y_pad = y_pad.contiguous()
        rows_pad, dim = y_pad.shape
        rows = row_dest.shape[0]
        n_out = rows // group
        weights = weights.contiguous()
        out = torch.empty(n_out, dim, device=y_pad.device, dtype=out_dtype)
        # The reference forms ``rows * weights`` in the promoted dtype, then
        # casts each term to the accumulator dtype BEFORE summing; every
        # rounding on that path is a rounding to bf16 (idempotent), so one
        # bf16 rounding per term reproduces any combination of them.
        promoted = torch.promote_types(y_pad.dtype, weights.dtype)
        round_term = promoted != torch.float32 or out_dtype != torch.float32
        grid = (triton.cdiv(n_out, 32), triton.cdiv(dim, 128))
        _combine_fwd_kernel[grid](
            y_pad, row_dest, weights, out, n_out, dim,
            GROUP=group, ROUND_TERM=round_term, BLOCK_R=32, BLOCK_D=128,
        )
        ctx.save_for_backward(y_pad, pad_src_row, pad_valid, weights)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        y_pad, pad_src_row, pad_valid, weights = ctx.saved_tensors
        rows_pad, dim = y_pad.shape
        grad_out = grad_out.contiguous()
        need_w = ctx.needs_input_grad[4]
        grad_y = torch.empty_like(y_pad)
        grad_w = torch.empty(weights.shape[0], device=y_pad.device, dtype=torch.float32)
        grid = (triton.cdiv(rows_pad, 32),)
        _combine_bwd_kernel[grid](
            grad_out, y_pad, pad_src_row, pad_valid, weights, grad_y, grad_w,
            rows_pad, dim, GROUP=ctx.group, NEED_W=need_w, BLOCK_R=32, BLOCK_D=256,
        )
        return (
            grad_y,
            None,
            None,
            None,
            grad_w.to(weights.dtype) if need_w else None,
            None,
            None,
        )


# --------------------------------------------------------------------------
# Public entry points (thin, shape-checked).
# --------------------------------------------------------------------------
def dispatch_rows(
    x: Float[Tensor, "num_tokens d_in"],
    pad_token: Int[Tensor, "rows_pad"],
    pad_valid: Tensor,
    row_dest: Int[Tensor, "rows"],
    slots: int,
    compute_dtype: torch.dtype,
) -> Float[Tensor, "rows_pad d_in"]:
    """Fused ``x.to(compute_dtype).index_select(0, pad_token) * pad_valid``.

    The packed path's ``pack_rows`` kernel under the expert dispatch's names: the
    padded dispatch is the same bijection (every real padded slot reads one
    token; every ``(token, slot)`` pair owns one padded slot at
    ``row_dest``). Backward: the ``slots``-way gather-sum of the padded
    gradient through ``row_dest``, written in ``x``'s dtype (fp32 sum; the
    reference's bf16 sum-then-upcast differs by <= 1 bf16 ulp).

    Args:
        x (Float[Tensor, "num_tokens d_in"]): Token features (any float dtype).
        pad_token (Int[Tensor, "rows_pad"]): Padded slot -> source token.
        pad_valid (Tensor): ``[rows_pad]`` 1 on real slots, 0 on slack (any
            dtype; compared against zero).
        row_dest (Int[Tensor, "rows"]): Token-major routed row -> padded slot.
        slots (int): Slots per token (``rows == num_tokens * slots``).
        compute_dtype (torch.dtype): The GEMM dtype.

    Returns:
        Float[Tensor, "rows_pad d_in"]: Dispatched rows, slack rows zero.
    """

    return _TritonPackRows.apply(x, pad_token, pad_valid, row_dest, slots, compute_dtype)


def cat_cast(
    a: Float[Tensor, "... hidden"],
    b: Float[Tensor, "... hidden"],
    dtype: torch.dtype,
) -> Float[Tensor, "... two_hidden"]:
    """Fused ``torch.cat([a, b], dim=-1).to(dtype)`` (bit-exact), coalesced backward.

    Args:
        a (Float[Tensor, "... hidden"]): First bank (e.g. ``[E, d_in, hidden]``).
        b (Float[Tensor, "... hidden"]): Second bank, same shape and dtype.
        dtype (torch.dtype): The GEMM dtype.

    Returns:
        Float[Tensor, "... two_hidden"]: ``[a | b]`` along the last dim.
    """

    assert a.shape == b.shape and a.dtype == b.dtype, "cat_cast needs twin banks"
    return _TritonCatCast.apply(a, b, dtype)


def swiglu(
    up: Float[Tensor, "rows hidden"], gate: Float[Tensor, "rows hidden"]
) -> Float[Tensor, "rows hidden"]:
    """Fused ``F.silu(gate) * up`` (separate up / gate tensors).

    Args:
        up (Float[Tensor, "rows hidden"]): Up-projection rows.
        gate (Float[Tensor, "rows hidden"]): Gate-projection rows.

    Returns:
        Float[Tensor, "rows hidden"]: The gated hidden rows, ``up``'s dtype.
    """

    return _TritonSwiGLU.apply(up, gate)


def swiglu_fused(
    up_gate: Float[Tensor, "rows two_hidden"], hidden: int
) -> Float[Tensor, "rows hidden"]:
    """Fused ``F.silu(up_gate[:, hidden:]) * up_gate[:, :hidden]``.

    Args:
        up_gate (Float[Tensor, "rows two_hidden"]): The fused up|gate grouped
            GEMM output (columns ``< hidden`` are up, the rest gate).
        hidden (int): Expert hidden width.

    Returns:
        Float[Tensor, "rows hidden"]: The gated hidden rows.
    """

    assert up_gate.shape[1] == 2 * hidden, "up_gate must be [rows, 2*hidden]"
    return _TritonSwiGLUFused.apply(up_gate, hidden)


def combine_rows(
    y_pad: Float[Tensor, "rows_pad dim"],
    row_dest: Int[Tensor, "rows"],
    pad_src_row: Int[Tensor, "rows_pad"],
    pad_valid: Tensor,
    weights: Float[Tensor, "rows"],
    group: int,
    out_dtype: torch.dtype,
) -> Float[Tensor, "rows_over_group dim"]:
    """Fused combine: ``(y_pad[row_dest] * weights[:, None]).to(out_dtype).view(-1, group, dim).sum(1)``.

    Args:
        y_pad (Float[Tensor, "rows_pad dim"]): Padded down-GEMM output.
        row_dest (Int[Tensor, "rows"]): Routed row -> padded slot.
        pad_src_row (Int[Tensor, "rows_pad"]): Padded slot -> routed row
            (0 on slack; masked by ``pad_valid``).
        pad_valid (Tensor): ``[rows_pad]`` 1 on real slots, 0 on slack.
        weights (Float[Tensor, "rows"]): Per-routed-row combine weight.
        group (int): Consecutive routed rows summed into one output row
            (``slots`` for a full combine, ``k_q`` for the per-slot layout).
        out_dtype (torch.dtype): The caller's accumulator (stream) dtype.

    Returns:
        Float[Tensor, "rows_over_group dim"]: The weighted group sums.
    """

    assert row_dest.shape[0] % group == 0, "rows must be a multiple of group"
    return _TritonCombineRows.apply(
        y_pad, row_dest, pad_src_row, pad_valid, weights, group, out_dtype
    )
