"""Compiled pointwise chains of the packed attention path.

``RoutedTransformerLayer._attn_update_packed_varlen`` runs outside
``torch.compile`` (``_maybe_disable_compile``: under Inductor the layouts
picked for the tensors feeding ``torch._grouped_mm`` can violate the kernel's
16-byte stride requirement, and torch 2.10's dynamo has stalled tracing the
dispatch-op family). The price is that everything BETWEEN its GEMMs runs as
eager pointwise kernels: the fp32 RoPE rotation, the head padding, the muP
scale, the gate multiply and every cast — measured on a 500m column arm
as ~1,000 ``_to_copy`` casts plus ~8,300 eager ``mul`` launches
per step, alongside a 16% launch-bubble wall.

This module lifts those chains into three standalone functions and compiles
them with ``torch.compile``: a function explicitly compiled at module scope
still compiles when called from inside a ``torch.compiler.disable``-d region,
so the GEMMs and the varlen kernel keep their eager, layout-controlled calls
while the pointwise work between them fuses to a few kernels (forward AND
backward, via AOTAutograd). Each function replicates its eager chain
op-for-op — same op order, same dtypes — so the compiled and eager forms
agree to floating-point reassociation noise (<= 1-2 bf16 ulp: Inductor may
contract a mul+add into an FMA and skip an intermediate-cast rounding),
the same bar every other exact fast path is held to.

Enabled by ``NESTOR_PACKED_POINTWISE_COMPILE=1``, read at layer construction
(:func:`packed_pointwise_requested`), default off. The helpers only ever see
static shapes per block (``rows = B*S*k + 16*G``), so compilation happens once
per block shape; ``torch.compile``'s default automatic-dynamic mode keeps the
specialization count bounded when several arms share a process (the paired
profiling harness).

Nothing here imports from ``nestor.backbones.nestor`` (the caller), so there
is no import cycle.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from nestor.scaling.blocks import _rotate_half


def packed_pointwise_requested() -> bool:
    """Whether ``NESTOR_PACKED_POINTWISE_COMPILE=1`` asks for the compiled chains.

    Read by ``RoutedTransformerLayer`` at CONSTRUCTION (into an attribute),
    never inside the forward, so two twins of one model can coexist in a
    process under different settings (the paired profiling harness relies on
    that).

    Returns:
        bool: True iff the flag is set to ``"1"``.
    """

    return os.environ.get("NESTOR_PACKED_POINTWISE_COMPILE") == "1"


def rope_qkv(
    qkv: Float[Tensor, "rows 3_e8"],
    bqkv: Float[Tensor, "groups 3_e8"] | None,
    gid: Int[Tensor, "rows"] | None,
    cos: Float[Tensor, "rows 1 head_dim"],
    sin: Float[Tensor, "rows 1 head_dim"],
    inner_dim: int,
    num_heads: int,
    head_dim: int,
    pad: int,
) -> tuple[
    Float[Tensor, "rows num_heads head_dim_pad"],
    Float[Tensor, "rows num_heads head_dim_pad"],
    Float[Tensor, "rows num_heads head_dim_pad"],
]:
    """Bias add, un-pad, fp32 RoPE, head pad and muP scale of the packed q/k/v.

    Verbatim the chain ``_attn_update_packed_varlen`` runs between its q/k/v
    grouped GEMM and ``varlen_attn`` (same op order and dtypes): add the
    per-group bias (if any), drop the ``e8`` alignment columns, rotate q and k
    in fp32 with the plan's block-constant tables, cast back, zero-pad the
    head dim to a multiple of 8 for the flash kernel, and fold the muP scale
    correction ``sqrt(head_dim + pad) / head_dim`` into q.

    Args:
        qkv (Float[Tensor, "rows 3_e8"]): Raw grouped-GEMM output, laid out as
            ``3`` sections of ``e8 = ceil8(inner_dim)`` columns each, in the
            GEMM (autocast) dtype.
        bqkv (Float[Tensor, "groups 3_e8"] | None): Stacked fp32 q/k/v biases
            per group, or None for bias-free projections.
        gid (Int[Tensor, "rows"] | None): Row -> group index (required iff
            ``bqkv`` is given).
        cos (Float[Tensor, "rows 1 head_dim"]): fp32 RoPE cosine table at the
            packed positions.
        sin (Float[Tensor, "rows 1 head_dim"]): fp32 RoPE sine table.
        inner_dim (int): ``num_heads * head_dim`` (the real projection width).
        num_heads (int): Heads per group.
        head_dim (int): Per-head width.
        pad (int): Head-dim zero padding, ``(-head_dim) % 8``.

    Returns:
        tuple: ``(q, k, v)``, each ``[rows, num_heads, head_dim + pad]`` in the
        GEMM dtype, with the muP scale already folded into ``q``.
    """

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


def attn_out_rows(
    out: Float[Tensor, "rows num_heads head_dim_pad"],
    inner_dim: int,
    head_dim: int,
    pad: int,
    pad_e: int,
) -> Float[Tensor, "rows e8"]:
    """Strip the head padding and re-pad the flat rows for the out GEMM.

    Verbatim the chain between ``varlen_attn`` and the output grouped GEMM:
    drop the ``pad`` zero head columns, flatten to ``inner_dim``, and zero-pad
    the tail to ``e8 = inner_dim + pad_e`` (zero attention columns meet zero
    weight rows — exact by construction).

    Args:
        out (Float[Tensor, "rows num_heads head_dim_pad"]): ``varlen_attn``
            output at the padded head width.
        inner_dim (int): ``num_heads * head_dim``.
        head_dim (int): Per-head width (before padding).
        pad (int): Head-dim padding that was applied, ``(-head_dim) % 8``.
        pad_e (int): Tail padding to the GEMM width, ``e8 - inner_dim``.

    Returns:
        Float[Tensor, "rows e8"]: Contiguous rows for the output grouped GEMM.
    """

    rows = out.shape[0]
    if pad:
        out = out[..., :head_dim]
    out_flat = out.reshape(rows, inner_dim)
    if pad_e:
        out_flat = F.pad(out_flat, (0, pad_e))
    return out_flat


def gate_combine(
    out_c: Float[Tensor, "rows dim"],
    bo: Float[Tensor, "groups dim"] | None,
    gid: Int[Tensor, "rows"] | None,
    gates: Float[Tensor, "rows"],
    out_dtype: torch.dtype,
) -> Float[Tensor, "rows dim"]:
    """Output bias, upcast and gate multiply of the packed attention rows.

    Verbatim the chain after the output grouped GEMM: add the per-group output
    bias (if any), upcast to the accumulator dtype, and multiply by the packed
    per-row gates (tail rows carry exactly-zero gates).

    Args:
        out_c (Float[Tensor, "rows dim"]): Output grouped-GEMM rows in the
            GEMM dtype.
        bo (Float[Tensor, "groups dim"] | None): Stacked fp32 output biases per
            group, or None.
        gid (Int[Tensor, "rows"] | None): Row -> group index (required iff
            ``bo`` is given).
        gates (Float[Tensor, "rows"]): fp32 packed per-row gates.
        out_dtype (torch.dtype): Accumulator dtype of the caller (the stream /
            residual dtype).

    Returns:
        Float[Tensor, "rows dim"]: Gate-weighted rows in ``out_dtype``.
    """

    if bo is not None:
        out_c = out_c + bo[gid].to(out_c.dtype)
    return out_c.to(out_dtype) * gates.unsqueeze(-1).to(out_dtype)


# Compiled twins. Explicitly compiled functions still compile when called from
# a ``torch.compiler.disable``-d frame, which is exactly how the packed
# attention method uses them. Default (automatic-dynamic) mode: the first call
# per input signature compiles static, a second distinct shape recompiles
# dynamic — bounding specializations when many block shapes / arms share the
# process.
rope_qkv_compiled = torch.compile(rope_qkv)
attn_out_rows_compiled = torch.compile(attn_out_rows)
gate_combine_compiled = torch.compile(gate_combine)
