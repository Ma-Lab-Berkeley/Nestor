"""Hand-written Triton kernels for the packed column-attention fast path.

``RoutedTransformerLayer._attn_update_packed_varlen`` keeps its two grouped
GEMMs (``torch._grouped_mm``, ~550 TF/s on H200) and its flash varlen kernel
(``torch.nn.attention.varlen.varlen_attn``) — those are efficient and are NOT
replaced here. Everything BETWEEN them, however, runs as eager pointwise and
gather kernels (the method is ``torch.compiler.disable``'d), and the two
autograd gather Functions (``_PackedRowGather`` / ``_PackedSlotGather``) are
boundaries even ``NESTOR_PACKED_POINTWISE_COMPILE``'s compiled chains cannot
fuse across. Measured on a 500m column arm: ~1,000
``_to_copy`` casts + ~8,300 eager ``mul`` launches per step and a 16%
launch-bubble wall.

``NESTOR_TRITON_PACKED_ATTN=1`` replaces the whole between-GEMM region with
three fused forward/backward Triton kernel pairs, each exact against its
reference chain (same op order and dtypes up to floating-point reassociation,
the <= 1-2 bf16 ulp bar every exact fast path is held to):

1. :func:`pack_rows` — cast + ``index_select`` + alignment-tail zeroing in ONE
   kernel; backward is a fused k-way gather-sum (one kernel instead of
   ``index_select`` + ``view`` + ``sum``).
2. :func:`rope_qkv_headpad` — per-group bias add, fp32 RoPE rotation, head
   padding and the muP scale in ONE kernel over a HEAD-PADDED qkv layout (see
   below); the backward (RoPE is linear) is one kernel.
3. :func:`gate_combine_slots` — output bias, upcast, gate multiply AND the
   slot gather fused into ONE kernel, crossing the ``_PackedSlotGather``
   Function boundary: the intermediate gate-weighted ``[rows, dim]`` fp32
   tensor never exists. The backward fuses the gate/bias grads and the
   per-row ``grad_gates`` dot products into one pass.

Head-padded weight layout: the plain packed path pads the projection width to
``e8 = ceil8(H*head_dim)`` for the grouped GEMM and separately pads the head
dim to ``ceil8(head_dim)`` for flash varlen, forcing an unpad/re-pad copy
(``attn_out_rows``) between the attention kernel and the output GEMM. Here the
q/k/v and output weight stacks are laid out per head at the PADDED head width
``hdp = ceil8(head_dim)`` (``[G, dim, 3*H*hdp]`` and ``[G, H*hdp, dim]``, zero
at every pad position), so the qkv GEMM writes directly in the shape varlen
consumes and the varlen output feeds the output GEMM as a VIEW — the
``attn_out_rows`` chain disappears. Exact: pad weight columns are zero, so
q/k pad lanes are exactly zero (adding 0 to every logit) and v pad lanes are
exactly zero (so attention-output pad lanes are exactly zero against the
arbitrary — here zero — pad rows of the output stack). ``hdp % 8 == 0`` keeps
every grouped-GEMM operand stride a 16-byte multiple.

The flag is read at CONSTRUCTION (:func:`triton_packed_requested`) like every
other fast-path flag, so flag-on/off twins coexist in one process (the paired
profiling harness). It changes no parameter, so it is resume-safe and can be
flipped between runs of one checkpoint.

Nothing here imports from ``nestor.backbones.nestor`` (the caller), so there
is no import cycle. Triton is imported guardedly: on machines without it
(macOS dev boxes) the module still imports and
:func:`triton_packed_available` reports False.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

try:  # pragma: no cover - trivially platform-dependent
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def triton_packed_requested() -> bool:
    """Whether ``NESTOR_TRITON_PACKED_ATTN=1`` asks for the Triton packed path.

    Read by ``RoutedTransformerLayer`` at CONSTRUCTION (into an attribute),
    never inside the forward, so two twins of one model can coexist in a
    process under different settings (the paired profiling harness relies on
    that).

    Returns:
        bool: True iff the flag is set to ``"1"``.
    """

    return os.environ.get("NESTOR_TRITON_PACKED_ATTN") == "1"


def triton_packed_available() -> bool:
    """Whether the Triton kernels can actually run here.

    Returns:
        bool: True iff triton imported and a CUDA device is visible.
    """

    return _HAS_TRITON and torch.cuda.is_available()


def _pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (>= 1)."""

    return 1 << max(0, n - 1).bit_length()


# --------------------------------------------------------------------------
# Kernels (defined only when triton is importable).
# --------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _pack_gather_fwd_kernel(
        x_ptr,  # [num_tokens, dim] source rows (any float dtype)
        tok_ptr,  # [rows] int64 packed row -> source token
        real_ptr,  # [rows] bool, False on alignment-tail rows
        out_ptr,  # [rows, dim] packed rows in the compute dtype
        n_rows,
        dim,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rmask = rows < n_rows
        cmask = cols < dim
        mask = rmask[:, None] & cmask[None, :]
        rows = rows.to(tl.int64)  # offsets can exceed int32 at rows * dim
        tok = tl.load(tok_ptr + rows, mask=rmask, other=0)
        real = tl.load(real_ptr + rows, mask=rmask, other=0)
        vals = tl.load(x_ptr + tok[:, None] * dim + cols[None, :], mask=mask, other=0.0)
        vals = tl.where(real[:, None] != 0, vals, 0.0)
        tl.store(
            out_ptr + rows[:, None] * dim + cols[None, :],
            vals.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _pack_gather_bwd_kernel(
        g_ptr,  # [rows, dim] packed-row gradients (compute dtype)
        slot_rows_ptr,  # [num_tokens * K] int64 (token, slot) -> packed row
        out_ptr,  # [num_tokens, dim] input gradient (input dtype)
        n_tokens,
        dim,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_d = tl.program_id(1)
        toks = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        tmask = toks < n_tokens
        cmask = cols < dim
        mask = tmask[:, None] & cmask[None, :]
        toks = toks.to(tl.int64)  # offsets can exceed int32 at tokens * dim
        acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for j in tl.static_range(K):
            r = tl.load(slot_rows_ptr + toks * K + j, mask=tmask, other=0)
            g = tl.load(g_ptr + r[:, None] * dim + cols[None, :], mask=mask, other=0.0)
            acc += g.to(tl.float32)
        tl.store(
            out_ptr + toks[:, None] * dim + cols[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rope_qkv_fwd_kernel(
        qkv_ptr,  # [rows, 3*H*HDP] grouped-GEMM output (bf16), head-padded
        bq_ptr,  # [G, 3*H*HDP] fp32 stacked biases (unused if not HAS_BIAS)
        gid_ptr,  # [rows] int64 row -> group (unused if not HAS_BIAS)
        cos_ptr,  # [rows, 1, HD] fp32 RoPE cosines at packed positions
        sin_ptr,  # [rows, 1, HD] fp32 RoPE sines
        q_ptr,  # [rows, H, HDP] out (bf16)
        k_ptr,
        v_ptr,
        n_rows,
        scale,  # muP correction sqrt(HDP) / HD, folded into q
        H: tl.constexpr,
        HD: tl.constexpr,
        HDP: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_H2: tl.constexpr,  # pow2 >= HD // 2
    ):
        pid_r = tl.program_id(0)
        h = tl.program_id(1)
        HALF: tl.constexpr = HD // 2
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        c2 = tl.arange(0, BLOCK_H2)
        rmask = rows < n_rows
        hmask = c2 < HALF
        mask = rmask[:, None] & hmask[None, :]
        rows = rows.to(tl.int64)  # offsets can exceed int32 at rows * W
        W = 3 * H * HDP
        head = h * HDP
        base1 = rows[:, None] * W + head + c2[None, :]
        base2 = base1 + HALF
        q1 = tl.load(qkv_ptr + base1, mask=mask, other=0.0)
        q2 = tl.load(qkv_ptr + base2, mask=mask, other=0.0)
        k1 = tl.load(qkv_ptr + base1 + H * HDP, mask=mask, other=0.0)
        k2 = tl.load(qkv_ptr + base2 + H * HDP, mask=mask, other=0.0)
        v1 = tl.load(qkv_ptr + base1 + 2 * H * HDP, mask=mask, other=0.0)
        v2 = tl.load(qkv_ptr + base2 + 2 * H * HDP, mask=mask, other=0.0)
        if HAS_BIAS:
            # Reference adds the fp32 bias CAST TO bf16, in bf16.
            gid = tl.load(gid_ptr + rows, mask=rmask, other=0)
            bb1 = gid[:, None] * W + head + c2[None, :]
            bb2 = bb1 + HALF
            ety = qkv_ptr.dtype.element_ty
            q1 += tl.load(bq_ptr + bb1, mask=mask, other=0.0).to(ety)
            q2 += tl.load(bq_ptr + bb2, mask=mask, other=0.0).to(ety)
            k1 += tl.load(bq_ptr + bb1 + H * HDP, mask=mask, other=0.0).to(ety)
            k2 += tl.load(bq_ptr + bb2 + H * HDP, mask=mask, other=0.0).to(ety)
            v1 += tl.load(bq_ptr + bb1 + 2 * H * HDP, mask=mask, other=0.0).to(ety)
            v2 += tl.load(bq_ptr + bb2 + 2 * H * HDP, mask=mask, other=0.0).to(ety)
        # RoPE tables are cat(angles, angles): load both halves (equal by
        # construction, loaded separately for generality).
        cbase = rows[:, None] * HD + c2[None, :]
        cA = tl.load(cos_ptr + cbase, mask=mask, other=0.0)
        cB = tl.load(cos_ptr + cbase + HALF, mask=mask, other=0.0)
        sA = tl.load(sin_ptr + cbase, mask=mask, other=0.0)
        sB = tl.load(sin_ptr + cbase + HALF, mask=mask, other=0.0)
        # rotate_half([x1, x2]) = [-x2, x1]; rotation in fp32, cast back.
        q1f = q1.to(tl.float32)
        q2f = q2.to(tl.float32)
        k1f = k1.to(tl.float32)
        k2f = k2.to(tl.float32)
        oty = q_ptr.dtype.element_ty
        qo1 = ((q1f * cA - q2f * sA) * scale).to(oty)
        qo2 = ((q2f * cB + q1f * sB) * scale).to(oty)
        ko1 = (k1f * cA - k2f * sA).to(oty)
        ko2 = (k2f * cB + k1f * sB).to(oty)
        obase1 = rows[:, None] * (H * HDP) + head + c2[None, :]
        obase2 = obase1 + HALF
        tl.store(q_ptr + obase1, qo1, mask=mask)
        tl.store(q_ptr + obase2, qo2, mask=mask)
        tl.store(k_ptr + obase1, ko1, mask=mask)
        tl.store(k_ptr + obase2, ko2, mask=mask)
        tl.store(v_ptr + obase1, v1.to(oty), mask=mask)
        tl.store(v_ptr + obase2, v2.to(oty), mask=mask)
        if HDP > HD:
            # Zero the pad lanes: flash varlen reads them (q/k pad lanes add
            # 0 to every logit; v pad lanes make the output pad lanes 0).
            cp = tl.arange(0, 8)
            pmask = rmask[:, None] & (cp[None, :] < (HDP - HD))
            zero = tl.zeros((BLOCK_R, 8), dtype=tl.float32).to(oty)
            pbase = rows[:, None] * (H * HDP) + head + HD + cp[None, :]
            tl.store(q_ptr + pbase, zero, mask=pmask)
            tl.store(k_ptr + pbase, zero, mask=pmask)
            tl.store(v_ptr + pbase, zero, mask=pmask)

    @triton.jit
    def _rope_qkv_bwd_kernel(
        gq_ptr,  # [rows, H, HDP] q gradient (bf16)
        gk_ptr,
        gv_ptr,
        cos_ptr,  # [rows, 1, HD] fp32
        sin_ptr,
        out_ptr,  # [rows, 3*H*HDP] qkv gradient (bf16), pad lanes zeroed
        n_rows,
        scale,
        H: tl.constexpr,
        HD: tl.constexpr,
        HDP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_H2: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        h = tl.program_id(1)
        HALF: tl.constexpr = HD // 2
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        c2 = tl.arange(0, BLOCK_H2)
        rmask = rows < n_rows
        hmask = c2 < HALF
        mask = rmask[:, None] & hmask[None, :]
        rows = rows.to(tl.int64)  # offsets can exceed int32 at rows * W
        W = 3 * H * HDP
        head = h * HDP
        gbase1 = rows[:, None] * (H * HDP) + head + c2[None, :]
        gbase2 = gbase1 + HALF
        gq1 = tl.load(gq_ptr + gbase1, mask=mask, other=0.0).to(tl.float32)
        gq2 = tl.load(gq_ptr + gbase2, mask=mask, other=0.0).to(tl.float32)
        gk1 = tl.load(gk_ptr + gbase1, mask=mask, other=0.0).to(tl.float32)
        gk2 = tl.load(gk_ptr + gbase2, mask=mask, other=0.0).to(tl.float32)
        gv1 = tl.load(gv_ptr + gbase1, mask=mask, other=0.0)
        gv2 = tl.load(gv_ptr + gbase2, mask=mask, other=0.0)
        cbase = rows[:, None] * HD + c2[None, :]
        cA = tl.load(cos_ptr + cbase, mask=mask, other=0.0)
        cB = tl.load(cos_ptr + cbase + HALF, mask=mask, other=0.0)
        sA = tl.load(sin_ptr + cbase, mask=mask, other=0.0)
        sB = tl.load(sin_ptr + cbase + HALF, mask=mask, other=0.0)
        # Forward halves: o1 = x1*cA - x2*sA ; o2 = x2*cB + x1*sB (x scale for q).
        # Transpose:      dx1 = g1*cA + g2*sB ; dx2 = -g1*sA + g2*cB.
        oty = out_ptr.dtype.element_ty
        dq1 = ((gq1 * cA + gq2 * sB) * scale).to(oty)
        dq2 = ((gq2 * cB - gq1 * sA) * scale).to(oty)
        dk1 = (gk1 * cA + gk2 * sB).to(oty)
        dk2 = (gk2 * cB - gk1 * sA).to(oty)
        obase1 = rows[:, None] * W + head + c2[None, :]
        obase2 = obase1 + HALF
        tl.store(out_ptr + obase1, dq1, mask=mask)
        tl.store(out_ptr + obase2, dq2, mask=mask)
        tl.store(out_ptr + obase1 + H * HDP, dk1, mask=mask)
        tl.store(out_ptr + obase2 + H * HDP, dk2, mask=mask)
        tl.store(out_ptr + obase1 + 2 * H * HDP, gv1.to(oty), mask=mask)
        tl.store(out_ptr + obase2 + 2 * H * HDP, gv2.to(oty), mask=mask)
        if HDP > HD:
            # Pad lanes of the qkv gradient are exactly zero (their weight
            # columns are the zero pads of the stack; F.pad's backward slices
            # them off, but the GEMM backward still reads them).
            cp = tl.arange(0, 8)
            pmask = rmask[:, None] & (cp[None, :] < (HDP - HD))
            zero = tl.zeros((BLOCK_R, 8), dtype=tl.float32).to(oty)
            pbase = rows[:, None] * W + head + HD + cp[None, :]
            tl.store(out_ptr + pbase, zero, mask=pmask)
            tl.store(out_ptr + pbase + H * HDP, zero, mask=pmask)
            tl.store(out_ptr + pbase + 2 * H * HDP, zero, mask=pmask)

    @triton.jit
    def _gate_slot_fwd_kernel(
        outc_ptr,  # [rows, dim] output-GEMM rows (bf16)
        bo_ptr,  # [G, dim] fp32 output biases (unused if not HAS_BIAS)
        gid_ptr,  # [rows] int64 (unused if not HAS_BIAS)
        gates_ptr,  # [rows] fp32 gates, exactly 0 on tails
        slot_rows_ptr,  # [n_out] int64 (token, slot) -> packed row
        out_ptr,  # [n_out, dim] slot-layout rows (out dtype)
        n_out,
        dim,
        HAS_BIAS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_d = tl.program_id(1)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        rmask = rows < n_out
        cmask = cols < dim
        mask = rmask[:, None] & cmask[None, :]
        rows = rows.to(tl.int64)  # offsets can exceed int32 at rows * dim
        r = tl.load(slot_rows_ptr + rows, mask=rmask, other=0)
        gate = tl.load(gates_ptr + r, mask=rmask, other=0.0)
        val = tl.load(outc_ptr + r[:, None] * dim + cols[None, :], mask=mask, other=0.0)
        if HAS_BIAS:
            gid = tl.load(gid_ptr + r, mask=rmask, other=0)
            bo = tl.load(bo_ptr + gid[:, None] * dim + cols[None, :], mask=mask, other=0.0)
            val += bo.to(outc_ptr.dtype.element_ty)
        oty = out_ptr.dtype.element_ty
        res = val.to(oty) * gate.to(oty)[:, None]
        tl.store(out_ptr + rows[:, None] * dim + cols[None, :], res, mask=mask)

    @triton.jit
    def _gate_slot_bwd_kernel(
        g_ptr,  # [n_out, dim] slot-layout gradient (out dtype)
        outc_ptr,  # [rows, dim] forward output-GEMM rows (bf16)
        bo_ptr,
        gid_ptr,
        gates_ptr,  # [rows] fp32
        slot_packed_ptr,  # [rows] int64 packed row -> (token, slot) row
        goutc_ptr,  # [rows, dim] out: grad wrt out_c (bf16)
        ggates_ptr,  # [rows] out: grad wrt gates (fp32)
        n_rows,
        dim,
        HAS_BIAS: tl.constexpr,
        NEED_GATES: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < n_rows
        rows = rows.to(tl.int64)  # offsets can exceed int32 at rows * dim
        sp = tl.load(slot_packed_ptr + rows, mask=rmask, other=0)
        gate = tl.load(gates_ptr + rows, mask=rmask, other=0.0)
        if HAS_BIAS:
            gid = tl.load(gid_ptr + rows, mask=rmask, other=0)
        acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
        for off in range(0, dim, BLOCK_D):
            cols = off + tl.arange(0, BLOCK_D)
            cmask = cols < dim
            mask = rmask[:, None] & cmask[None, :]
            g = tl.load(
                g_ptr + sp[:, None] * dim + cols[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            gpre = gate[:, None] * g
            tl.store(
                goutc_ptr + rows[:, None] * dim + cols[None, :],
                gpre.to(goutc_ptr.dtype.element_ty),
                mask=mask,
            )
            if NEED_GATES:
                val = tl.load(
                    outc_ptr + rows[:, None] * dim + cols[None, :],
                    mask=mask,
                    other=0.0,
                )
                if HAS_BIAS:
                    bo = tl.load(
                        bo_ptr + gid[:, None] * dim + cols[None, :],
                        mask=mask,
                        other=0.0,
                    )
                    val += bo.to(outc_ptr.dtype.element_ty)
                acc += tl.sum(g * val.to(tl.float32), axis=1)
        if NEED_GATES:
            tl.store(ggates_ptr + rows, acc, mask=rmask)


# --------------------------------------------------------------------------
# Autograd wrappers.
# --------------------------------------------------------------------------
class _TritonPackRows(torch.autograd.Function):
    """Fused cast + row gather + tail-zeroing (``_PackedRowGather`` twin).

    Forward reads each packed row's source token once, casts to the compute
    dtype in registers, zeroes tail rows and writes ``[rows, dim]`` — one
    kernel instead of cast + ``index_select`` + mask multiply. Backward sums
    each token's ``k`` packed-row gradients through ``slot_rows`` in fp32 and
    writes the input-dtype gradient directly — one kernel instead of
    ``index_select`` + ``view`` + ``sum`` + the cast node's upcast.
    """

    @staticmethod
    def forward(ctx, x_flat, tok_packed, row_real, slot_rows, num_slots, compute_dtype):  # noqa: D102
        rows = tok_packed.shape[0]
        num_tokens, dim = x_flat.shape
        x_flat = x_flat.contiguous()
        out = torch.empty(rows, dim, device=x_flat.device, dtype=compute_dtype)
        grid = (triton.cdiv(rows, 32), triton.cdiv(dim, 128))
        _pack_gather_fwd_kernel[grid](
            x_flat, tok_packed, row_real, out, rows, dim, BLOCK_R=32, BLOCK_D=128
        )
        ctx.save_for_backward(slot_rows)
        ctx.num_slots = num_slots
        ctx.num_tokens = num_tokens
        ctx.in_dtype = x_flat.dtype
        return out

    @staticmethod
    def backward(ctx, grad_packed):  # noqa: D102
        (slot_rows,) = ctx.saved_tensors
        grad_packed = grad_packed.contiguous()
        dim = grad_packed.shape[1]
        grad_x = torch.empty(
            ctx.num_tokens, dim, device=grad_packed.device, dtype=ctx.in_dtype
        )
        grid = (triton.cdiv(ctx.num_tokens, 16), triton.cdiv(dim, 128))
        _pack_gather_bwd_kernel[grid](
            grad_packed,
            slot_rows,
            grad_x,
            ctx.num_tokens,
            dim,
            K=ctx.num_slots,
            BLOCK_T=16,
            BLOCK_D=128,
        )
        return grad_x, None, None, None, None, None


class _TritonRopeQKV(torch.autograd.Function):
    """Fused bias + fp32 RoPE + head pad + muP scale over head-padded qkv.

    Linear in ``(qkv, bqkv)``, so the backward saves only the RoPE tables.
    The per-group bias gradient is a deterministic mask-GEMM outside the
    kernel (rows are group-major, but a GEMM against the ``[G, rows]``
    membership mask avoids fp32 atomics entirely — same trick as
    ``_IndexedBias``); it only runs when the projections carry biases.
    """

    @staticmethod
    def forward(ctx, qkv, bqkv, gid, cos, sin, num_heads, head_dim, hdp, scale):  # noqa: D102
        rows = qkv.shape[0]
        qkv = qkv.contiguous()
        out_shape = (rows, num_heads, hdp)
        q = torch.empty(out_shape, device=qkv.device, dtype=qkv.dtype)
        k = torch.empty(out_shape, device=qkv.device, dtype=qkv.dtype)
        v = torch.empty(out_shape, device=qkv.device, dtype=qkv.dtype)
        has_bias = bqkv is not None
        grid = (triton.cdiv(rows, 32), num_heads)
        _rope_qkv_fwd_kernel[grid](
            qkv,
            bqkv if has_bias else qkv,
            gid if has_bias else qkv,
            cos,
            sin,
            q,
            k,
            v,
            rows,
            scale,
            H=num_heads,
            HD=head_dim,
            HDP=hdp,
            HAS_BIAS=has_bias,
            BLOCK_R=32,
            BLOCK_H2=_pow2(head_dim // 2),
        )
        ctx.save_for_backward(cos, sin, gid if has_bias else cos)
        ctx.meta = (
            num_heads,
            head_dim,
            hdp,
            scale,
            has_bias,
            qkv.dtype,
            bqkv.shape[0] if has_bias else 0,
        )
        return q, k, v

    @staticmethod
    def backward(ctx, gq, gk, gv):  # noqa: D102
        cos, sin, gid = ctx.saved_tensors
        num_heads, head_dim, hdp, scale, has_bias, dtype, num_groups = ctx.meta
        rows = gq.shape[0]
        grad_qkv = torch.empty(
            rows, 3 * num_heads * hdp, device=gq.device, dtype=dtype
        )
        grid = (triton.cdiv(rows, 32), num_heads)
        _rope_qkv_bwd_kernel[grid](
            gq.contiguous(),
            gk.contiguous(),
            gv.contiguous(),
            cos,
            sin,
            grad_qkv,
            rows,
            scale,
            H=num_heads,
            HD=head_dim,
            HDP=hdp,
            BLOCK_R=32,
            BLOCK_H2=_pow2(head_dim // 2),
        )
        grad_b = None
        if has_bias and ctx.needs_input_grad[1]:
            # grad_bqkv[g] = sum of grad_qkv over the rows of group g, in
            # fp32 (a bf16 reduction over thousands of rows fails equivalence
            # against the per-group Linear backward). Deterministic GEMM.
            member = (
                gid.unsqueeze(0)
                == torch.arange(num_groups, device=gid.device).unsqueeze(1)
            ).to(torch.float32)  # [G, rows]
            grad_b = member @ grad_qkv.float()
        return grad_qkv, grad_b, None, None, None, None, None, None, None


class _TritonGateCombineSlots(torch.autograd.Function):
    """Fused output bias + upcast + gate + slot gather (``gate_combine`` +
    ``_PackedSlotGather`` in ONE kernel each way).

    Forward reads each ``(token, slot)`` pair's packed row from the output
    GEMM, adds the bias, upcasts, applies the gate and writes the slot-layout
    row — the gate-weighted ``[rows, dim]`` intermediate of the unfused chain
    never exists. Tail safety is inherited: orphan pairs read a zero-gate
    tail row, so their output is exactly zero. Backward gathers the slot
    gradient back through ``slot_packed``, produces the bf16 ``out_c``
    gradient and the per-row gate gradient (a fused dot product) in one pass;
    tail rows carry exactly-zero gates so their ``out_c`` gradient is exactly
    zero, the invariant the packed path's weight gradients rely on.
    """

    @staticmethod
    def forward(ctx, out_c, bo, gid, gates, slot_rows, slot_packed, out_dtype):  # noqa: D102
        rows, dim = out_c.shape
        n_out = slot_rows.shape[0]
        out_c = out_c.contiguous()
        out = torch.empty(n_out, dim, device=out_c.device, dtype=out_dtype)
        has_bias = bo is not None
        grid = (triton.cdiv(n_out, 32), triton.cdiv(dim, 128))
        _gate_slot_fwd_kernel[grid](
            out_c,
            bo if has_bias else out_c,
            gid if has_bias else out_c,
            gates,
            slot_rows,
            out,
            n_out,
            dim,
            HAS_BIAS=has_bias,
            BLOCK_R=32,
            BLOCK_D=128,
        )
        ctx.save_for_backward(
            out_c,
            bo if has_bias else out_c,
            gid if has_bias else out_c,
            gates,
            slot_packed,
        )
        ctx.has_bias = has_bias
        return out

    @staticmethod
    def backward(ctx, grad_slots):  # noqa: D102
        out_c, bo, gid, gates, slot_packed = ctx.saved_tensors
        rows, dim = out_c.shape
        grad_slots = grad_slots.contiguous()
        need_gates = ctx.needs_input_grad[3]
        grad_out_c = torch.empty_like(out_c)
        grad_gates = torch.empty(
            rows, device=out_c.device, dtype=torch.float32
        )
        grid = (triton.cdiv(rows, 32),)
        _gate_slot_bwd_kernel[grid](
            grad_slots,
            out_c,
            bo,
            gid,
            gates,
            slot_packed,
            grad_out_c,
            grad_gates,
            rows,
            dim,
            HAS_BIAS=ctx.has_bias,
            NEED_GATES=need_gates,
            BLOCK_R=32,
            BLOCK_D=256,
        )
        grad_bo = None
        if ctx.has_bias and ctx.needs_input_grad[1]:
            # Deterministic fp32 mask-GEMM, mirroring the reference's fp32
            # index-accumulation of the bf16 row gradients.
            num_groups = bo.shape[0]
            member = (
                gid.unsqueeze(0)
                == torch.arange(num_groups, device=gid.device).unsqueeze(1)
            ).to(torch.float32)  # [G, rows]
            grad_bo = member @ grad_out_c.float()
        return (
            grad_out_c,
            grad_bo,
            None,
            grad_gates if need_gates else None,
            None,
            None,
            None,
        )


# --------------------------------------------------------------------------
# Public entry points (thin, shape-checked).
# --------------------------------------------------------------------------
def pack_rows(
    x_flat: Float[Tensor, "num_tokens dim"],
    tok_packed: Int[Tensor, "rows"],
    row_real: Tensor,
    slot_rows: Int[Tensor, "num_tokens_slots"],
    num_slots: int,
    compute_dtype: torch.dtype,
) -> Float[Tensor, "rows dim"]:
    """Fused pack: ``x_flat[tok_packed] -> compute_dtype``, tails zeroed.

    Args:
        x_flat (Float[Tensor, "num_tokens dim"]): Source token rows.
        tok_packed (Int[Tensor, "rows"]): Packed row -> source token.
        row_real (Tensor): Bool mask, False on alignment-tail rows.
        slot_rows (Int[Tensor, "num_tokens_slots"]): Inverse map for the
            backward's k-way gather-sum.
        num_slots (int): Slots (top-k) per token.
        compute_dtype (torch.dtype): The GEMM dtype (bf16 under autocast).

    Returns:
        Float[Tensor, "rows dim"]: The packed rows in ``compute_dtype``.
    """

    return _TritonPackRows.apply(
        x_flat, tok_packed, row_real, slot_rows, num_slots, compute_dtype
    )


def rope_qkv_headpad(
    qkv: Float[Tensor, "rows three_H_hdp"],
    bqkv: Float[Tensor, "groups three_H_hdp"] | None,
    gid: Int[Tensor, "rows"] | None,
    cos: Float[Tensor, "rows 1 head_dim"],
    sin: Float[Tensor, "rows 1 head_dim"],
    num_heads: int,
    head_dim: int,
    hdp: int,
    scale: float,
) -> tuple[
    Float[Tensor, "rows num_heads hdp"],
    Float[Tensor, "rows num_heads hdp"],
    Float[Tensor, "rows num_heads hdp"],
]:
    """Fused bias + RoPE + head pad + muP scale over the head-padded qkv.

    Args:
        qkv (Float[Tensor, "rows three_H_hdp"]): qkv grouped-GEMM output in
            the head-padded layout (section-major: q | k | v, each
            ``[H, hdp]`` per row; pad lanes exactly zero).
        bqkv (Float[Tensor, "groups three_H_hdp"] | None): fp32 stacked
            biases in the same layout (zero at pad lanes), or None.
        gid (Int[Tensor, "rows"] | None): Row -> group (required iff bqkv).
        cos (Float[Tensor, "rows 1 head_dim"]): fp32 RoPE cosine table.
        sin (Float[Tensor, "rows 1 head_dim"]): fp32 RoPE sine table.
        num_heads (int): Heads per group.
        head_dim (int): Real per-head width (must be even for RoPE halves).
        hdp (int): Padded head width, ``ceil8(head_dim)``.
        scale (float): muP correction ``sqrt(hdp) / head_dim``, folded into q.

    Returns:
        tuple: ``(q, k, v)``, each ``[rows, num_heads, hdp]`` in ``qkv``'s
        dtype, pad lanes exactly zero, ready for ``varlen_attn``.
    """

    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    return _TritonRopeQKV.apply(
        qkv, bqkv, gid, cos, sin, num_heads, head_dim, hdp, scale
    )


def gate_combine_slots(
    out_c: Float[Tensor, "rows dim"],
    bo: Float[Tensor, "groups dim"] | None,
    gid: Int[Tensor, "rows"] | None,
    gates: Float[Tensor, "rows"],
    slot_rows: Int[Tensor, "num_tokens_slots"],
    slot_packed: Int[Tensor, "rows"],
    out_dtype: torch.dtype,
) -> Float[Tensor, "num_tokens_slots dim"]:
    """Fused output bias + upcast + gate + slot gather.

    Args:
        out_c (Float[Tensor, "rows dim"]): Output grouped-GEMM rows (bf16).
        bo (Float[Tensor, "groups dim"] | None): fp32 output biases, or None.
        gid (Int[Tensor, "rows"] | None): Row -> group (required iff bo).
        gates (Float[Tensor, "rows"]): fp32 packed gates, exactly 0 on tails.
        slot_rows (Int[Tensor, "num_tokens_slots"]): (token, slot) -> row.
        slot_packed (Int[Tensor, "rows"]): Row -> (token, slot), clamped on
            tails (whose zero gates zero the backward exactly).
        out_dtype (torch.dtype): The caller's accumulator (stream) dtype.

    Returns:
        Float[Tensor, "num_tokens_slots dim"]: Gate-weighted slot-layout rows.
    """

    return _TritonGateCombineSlots.apply(
        out_c, bo, gid, gates, slot_rows, slot_packed, out_dtype
    )


# --------------------------------------------------------------------------
# Head-padded weight stacks (torch-side, autograd-tracked, once per forward).
# --------------------------------------------------------------------------
def stack_wqkv_headpad(
    groups: list[nn.Module],
    dim: int,
    num_heads: int,
    head_dim: int,
    hdp: int,
    dtype: torch.dtype,
    batched: bool = False,
) -> Float[Tensor, "groups dim three_H_hdp"]:
    """Stack q/k/v projection weights in the head-padded GEMM layout.

    Zero columns at every pad lane make the qkv GEMM output's pad lanes
    exactly zero — the invariant :func:`rope_qkv_headpad` and flash varlen
    rely on. Same per-forward cost as the plain path's ``e8`` stack.

    Args:
        groups (list[nn.Module]): Attention groups owning ``q_proj`` /
            ``k_proj`` / ``v_proj`` (``weight`` of shape ``[inner, dim]``).
        dim (int): Model width.
        num_heads (int): Heads per group.
        head_dim (int): Real per-head width.
        hdp (int): Padded head width.
        dtype (torch.dtype): GEMM dtype.
        batched (bool): Build the identical tensor as ONE stack of all
            ``3 * G`` weights, one permute copy, one pad and one cast (4
            kernels, and a 3-kernel backward before the per-weight grad
            accumulation) instead of the per-group ``cat -> t -> reshape ->
            pad`` chain (``3G + 2`` kernels forward, ~``6G`` backward).
            Round-4 lever (``NESTOR_TRITON_EXPERT_DISPATCH``); same values
            either way.

    Returns:
        Float[Tensor, "groups dim three_H_hdp"]: Contiguous stacked weights.
    """

    pad = hdp - head_dim
    if batched:
        # [G*3, H*hd, dim] -> (g, s, h, j, i) -> (g, i, s, h, j): the same
        # element order as the per-group chain, padded on j in one copy.
        w = torch.stack(
            [
                proj.weight
                for g in groups
                for proj in (g.q_proj, g.k_proj, g.v_proj)
            ]
        ).view(len(groups), 3, num_heads, head_dim, dim).permute(0, 4, 1, 2, 3)
        # Materialize the permute before padding: constant_pad_nd preserves
        # the input's (permuted) strides, and the grouped GEMM needs a dense
        # row-major stack.
        w = w.contiguous()
        if pad:
            w = F.pad(w, (0, pad))
        return w.view(len(groups), dim, 3 * num_heads * hdp).to(dtype)
    stacked = []
    for g in groups:
        w = torch.cat(
            [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight], dim=0
        ).t().reshape(dim, 3, num_heads, head_dim)
        if pad:
            w = F.pad(w, (0, pad))
        stacked.append(w.reshape(dim, 3 * num_heads * hdp))
    return torch.stack(stacked).to(dtype)


def stack_bqkv_headpad(
    groups: list[nn.Module],
    num_heads: int,
    head_dim: int,
    hdp: int,
) -> Float[Tensor, "groups three_H_hdp"]:
    """Stack q/k/v biases (fp32) in the head-padded layout, zero pad lanes.

    Args:
        groups (list[nn.Module]): Attention groups with biased projections.
        num_heads (int): Heads per group.
        head_dim (int): Real per-head width.
        hdp (int): Padded head width.

    Returns:
        Float[Tensor, "groups three_H_hdp"]: fp32 stacked biases.
    """

    pad = hdp - head_dim
    stacked = []
    for g in groups:
        b = torch.cat(
            [g.q_proj.bias, g.k_proj.bias, g.v_proj.bias]
        ).reshape(3, num_heads, head_dim)
        if pad:
            b = F.pad(b, (0, pad))
        stacked.append(b.reshape(3 * num_heads * hdp))
    return torch.stack(stacked).float()


def stack_wo_headpad(
    groups: list[nn.Module],
    dim: int,
    num_heads: int,
    head_dim: int,
    hdp: int,
    dtype: torch.dtype,
    batched: bool = False,
) -> Float[Tensor, "groups H_hdp dim"]:
    """Stack output projection weights with zero rows at every pad lane.

    The varlen output's pad lanes are exactly zero (zero v pad lanes), so the
    zero pad ROWS here contribute exactly zero to the output GEMM — the
    varlen output feeds the GEMM as a view, and the plain path's
    ``attn_out_rows`` unpad/re-pad copy disappears.

    Args:
        groups (list[nn.Module]): Attention groups owning ``out_proj``
            (``weight`` of shape ``[dim, inner]``).
        dim (int): Model width.
        num_heads (int): Heads per group.
        head_dim (int): Real per-head width.
        hdp (int): Padded head width.
        dtype (torch.dtype): GEMM dtype.
        batched (bool): One stack + one permute/pad copy + one cast instead
            of the per-group chain (see :func:`stack_wqkv_headpad`).

    Returns:
        Float[Tensor, "groups H_hdp dim"]: Contiguous stacked weights.
    """

    pad = hdp - head_dim
    if batched:
        # [G, dim, H*hd] -> (g, i, h, j) -> (g, h, j, i), padded on j.
        w = (
            torch.stack([g.out_proj.weight for g in groups])
            .view(len(groups), dim, num_heads, head_dim)
            .permute(0, 2, 3, 1)
            .contiguous()  # see stack_wqkv_headpad: pad a dense tensor
        )
        if pad:
            w = F.pad(w, (0, 0, 0, pad))
        return w.view(len(groups), num_heads * hdp, dim).to(dtype)
    stacked = []
    for g in groups:
        w = g.out_proj.weight.t().reshape(num_heads, head_dim, dim)
        if pad:
            w = F.pad(w, (0, 0, 0, pad))
        stacked.append(w.reshape(num_heads * hdp, dim))
    return torch.stack(stacked).to(dtype)
