"""Fused dispatch-map construction for the grouped-GEMM expert paths.

:func:`~nestor.backbones.grouped_mm.grouped_expert_rows` builds, per bank
call, six index tensors from the sorted routing (``dest``, ``row_dest``,
``pad_token``, ``pad_src_row``, ``eid_pad``, ``pad_valid``) with a cascade of
~10 small arithmetic/scatter kernels — pure launch count against the 16%
launch-bubble wall measured at 500m, repeated for every bank of
every layer of every micro-step.

``NESTOR_TRITON_INDEX_BUILD=1`` replaces the cascade with ONE Triton kernel
over the sorted positions: each padded destination is written by exactly one
program lane (the dispatch is a bijection), so there are no atomics and the
resulting integer maps are EXACTLY equal to the cascade's — the argsort (an
efficient radix sort, kept), the per-bank counts and the cumsum offsets stay
in torch, everything derived per row moves into the kernel. Works for any
``(num_banks, slots)``: the native expert bank, the public pool and the
private pools all route through the same call.

The flag is read at CONSTRUCTION by the expert-bearing modules
(:func:`triton_index_requested`) and threaded into ``grouped_expert_rows`` as
the ``triton_index`` argument, so flag-on/off twins coexist in one process
(the paired profiling harness). It changes no math and no parameter:
resume-safe, exact to the last bit.

Nothing here imports from ``nestor.backbones.grouped_mm`` (the caller), so
there is no import cycle. Triton is imported guardedly so the module still
imports on machines without it.
"""

from __future__ import annotations

import os

import torch
from jaxtyping import Int
from torch import Tensor

try:  # pragma: no cover - trivially platform-dependent
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def triton_index_requested() -> bool:
    """Whether ``NESTOR_TRITON_INDEX_BUILD=1`` asks for the fused map build.

    Read by the expert-bearing modules at CONSTRUCTION (into an attribute
    they pass as ``triton_index``), never inside the forward, so two twins of
    one model can coexist in a process under different settings.

    Returns:
        bool: True iff the flag is set to ``"1"``.
    """

    return os.environ.get("NESTOR_TRITON_INDEX_BUILD") == "1"


def triton_index_available() -> bool:
    """Whether the fused map build can actually run here.

    Returns:
        bool: True iff triton imported and a CUDA device is visible.
    """

    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _dispatch_maps_kernel(
        expert_ids_ptr,  # [rows] int64 routed row -> bank
        order_ptr,  # [rows] int64 sorted position -> routed row
        padded_starts_ptr,  # [num_banks] int64 aligned segment starts
        raw_starts_ptr,  # [num_banks] int64 unaligned cumulative starts
        row_dest_ptr,  # [rows] out: routed row -> padded slot
        pad_token_ptr,  # [rows_pad] out: padded slot -> source token
        pad_src_row_ptr,  # [rows_pad] out: padded slot -> routed row
        eid_pad_ptr,  # [rows_pad] out: padded slot -> bank
        pad_valid_ptr,  # [rows_pad] out: 1 on real slots (valid dtype)
        n_rows,
        slots,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)  # sorted positions
        mask = p < n_rows
        p = p.to(tl.int64)  # dest arithmetic runs in int64 like the cascade
        orig = tl.load(order_ptr + p, mask=mask, other=0)
        eid = tl.load(expert_ids_ptr + orig, mask=mask, other=0)
        pstart = tl.load(padded_starts_ptr + eid, mask=mask, other=0)
        rstart = tl.load(raw_starts_ptr + eid, mask=mask, other=0)
        # Identical formula to the cascade: dest = padded_start - raw_start + p
        # (p is this row's global rank in the stable sort).
        dest = pstart - rstart + p
        tl.store(row_dest_ptr + orig, dest, mask=mask)
        tl.store(pad_token_ptr + dest, orig // slots, mask=mask)
        tl.store(pad_src_row_ptr + dest, orig, mask=mask)
        tl.store(eid_pad_ptr + dest, eid, mask=mask)
        one = tl.full((BLOCK,), 1.0, dtype=tl.float32)
        tl.store(pad_valid_ptr + dest, one.to(pad_valid_ptr.dtype.element_ty), mask=mask)


def build_dispatch_maps(
    expert_ids: Int[Tensor, "rows"],
    order: Int[Tensor, "rows"],
    padded_starts: Int[Tensor, "num_banks"],
    raw_starts: Int[Tensor, "num_banks"],
    slots: int,
    rows_pad: int,
    valid_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build every per-row dispatch map of ``grouped_expert_rows`` at once.

    Exactly equal (integer-exact) to the torch cascade it replaces: every
    padded destination is owned by one sorted position, slack slots keep
    their zero initialization (index 0, masked by ``pad_valid`` downstream).

    Args:
        expert_ids (Int[Tensor, "rows"]): Routed row -> bank index.
        order (Int[Tensor, "rows"]): Stable argsort of ``expert_ids``.
        padded_starts (Int[Tensor, "num_banks"]): Aligned per-bank starts.
        raw_starts (Int[Tensor, "num_banks"]): Unaligned per-bank starts.
        slots (int): Slots per token (``expert_ids`` rows are token-major).
        rows_pad (int): Static padded buffer length.
        valid_dtype (torch.dtype): dtype of ``pad_valid`` (the GEMM dtype).

    Returns:
        tuple: ``(row_dest, pad_token, pad_src_row, eid_pad, pad_valid)`` —
        the five tensors the gather Functions and the bias one-hots consume.
    """

    rows = order.shape[0]
    device = order.device
    row_dest = torch.empty(rows, dtype=torch.long, device=device)
    pad_token = torch.zeros(rows_pad, dtype=torch.long, device=device)
    pad_src_row = torch.zeros(rows_pad, dtype=torch.long, device=device)
    eid_pad = torch.zeros(rows_pad, dtype=torch.long, device=device)
    pad_valid = torch.zeros(rows_pad, dtype=valid_dtype, device=device)
    grid = (triton.cdiv(rows, 1024),)
    _dispatch_maps_kernel[grid](
        expert_ids,
        order,
        padded_starts,
        raw_starts,
        row_dest,
        pad_token,
        pad_src_row,
        eid_pad,
        pad_valid,
        rows,
        slots,
        BLOCK=1024,
    )
    return row_dest, pad_token, pad_src_row, eid_pad, pad_valid
