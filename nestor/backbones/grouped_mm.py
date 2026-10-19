"""Static-shape grouped-GEMM expert dispatch shared by the Nestor MLP paths.

Both expert-bearing paths of the backbone route every token through a *bank*
of small MLPs and would otherwise loop over the bank in Python:

* :class:`~nestor.backbones.nestor.RoutedTransformerLayer` (MoE-style shared /
  group-specific experts, ``NESTOR_MOE_GROUPED_MM``), and
* :class:`~nestor.backbones.expert_pools.ExpertPoolMLP` (public / private
  expert pools, ``NESTOR_POOL_GROUPED_MM``).

The per-expert loop is slow for the same two reasons everywhere: each iteration
needs a ``torch.nonzero`` (a device->host sync that serializes the whole step)
and each expert's GEMM sees only its own slice of the batch, so the GPU runs
dozens of tiny matmuls instead of one large one.

:func:`grouped_expert_rows` replaces the loop: every ``(token, slot)`` pair is
dispatched -- exactly ``num_tokens * slots`` rows -- so all shapes are static
(no host syncs, compile-friendly), rows are sorted by expert and padded per
expert to the 16-row alignment ``torch._grouped_mm`` requires, and the whole
bank runs as two (three for SwiGLU) grouped GEMMs. Pad rows are zeros, are
never read back, and receive zero gradient.

:func:`grouped_expert_combine` is the call-site form: it also applies the
per-row combine weights and sums each token's slots (in groups), and -- under
``NESTOR_TRITON_EXPERT_DISPATCH`` (:mod:`nestor.backbones.triton_expert`) --
runs the dispatch gather, the SwiGLU and that weighted combine as fused
Triton kernels around the same GEMMs.

Nothing here imports from ``nestor.backbones.nestor``, so both call sites can
import it at module scope without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - annotation only (import cycle otherwise)
    from nestor.backbones.expert_bank_stack import StackedExpertBank

from collections.abc import Sequence

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nestor.backbones.triton_dispatch import (
    build_dispatch_maps,
    triton_index_available,
)
from nestor.backbones.triton_expert import (
    bank_is_fusable,
    cat_cast,
    combine_rows,
    dispatch_rows,
    swiglu,
    swiglu_fused,
    triton_expert_available,
)

#: Row alignment required by the ``torch._grouped_mm`` CUDA kernel.
GROUPED_MM_ALIGN = 16


def autocast_dtype(x: Tensor) -> torch.dtype:
    """Dtype ``F.linear`` would run in for ``x`` under the active autocast.

    Args:
        x (Tensor): A GEMM input; only its device type and dtype are read.

    Returns:
        torch.dtype: The autocast dtype of ``x``'s device type if autocast is
        enabled there, else ``x.dtype``. Matching this is what keeps the
        grouped path numerically equal to the eager ``F.linear`` path (inputs,
        weights, and biases cast down; fp32 accumulation inside the GEMM).
    """

    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def segment_mm(
    a: Float[Tensor, "rows d_in"],
    b: Float[Tensor, "experts d_in d_out"],
    offs: Int[Tensor, "experts"],
) -> Float[Tensor, "rows d_out"]:
    """Per-segment matmul: rows ``[offs[i-1], offs[i])`` of ``a`` go through
    ``b[i]``. Uses ``torch._grouped_mm`` (one grouped GEMM, no host syncs) on
    CUDA; falls back to a host-side segment loop elsewhere (CPU tests, older
    torch). ``offs[-1]`` must equal ``a.shape[0]`` and every offset must be
    16-aligned for the CUDA kernel.
    """

    if a.is_cuda and hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(a, b, offs=offs)
    bounds = [0] + offs.tolist()
    return torch.cat(
        [
            a[bounds[i] : bounds[i + 1]] @ b[i]
            for i in range(b.shape[0])
        ],
        dim=0,
    )


class _DispatchGather(torch.autograd.Function):
    """Bijective row dispatch expressed as gathers in BOTH directions.

    Forward: ``x_pad[p] = x[pad_token[p]] * pad_valid[p]`` — each token row is
    replicated into its ``slots`` padded slots; slack rows are zeroed. Every
    real padded slot reads exactly one token and every (token, slot) pair owns
    exactly one padded slot, so the backward is a gather at ``row_dest``
    followed by a sum over the ``slots`` slots. Autograd's default would
    instead emit atomic ``index_add`` (``indexFuncLargeIndex``, measured ~4ms
    per call on H200 — ~10x a coalesced gather — and 31% of a MoE arm's step).
    """

    @staticmethod
    def forward(ctx, x, pad_token, pad_valid, row_dest, slots):  # noqa: D102
        ctx.save_for_backward(row_dest)
        ctx.slots = slots
        ctx.num_tokens = x.shape[0]
        return x.index_select(0, pad_token) * pad_valid.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_pad):  # noqa: D102
        (row_dest,) = ctx.saved_tensors
        grad_rows = grad_pad.index_select(0, row_dest)
        grad_x = grad_rows.view(ctx.num_tokens, ctx.slots, -1).sum(1)
        return grad_x, None, None, None, None


class _CombineGather(torch.autograd.Function):
    """Inverse of :class:`_DispatchGather` for the expert outputs.

    Forward: ``y_rows[r] = y_pad[row_dest[r]]`` (token-major). Each padded row
    is read at most once, so the backward is a gather at ``pad_src_row``
    masked by slot validity — again no atomic ``index_add``. The gradient is
    cast back to the forward dtype so the grouped GEMM's backward sees
    matching operand dtypes.
    """

    @staticmethod
    def forward(ctx, y_pad, row_dest, pad_src_row, pad_valid):  # noqa: D102
        ctx.save_for_backward(pad_src_row, pad_valid)
        ctx.pad_dtype = y_pad.dtype
        return y_pad.index_select(0, row_dest)

    @staticmethod
    def backward(ctx, grad_rows):  # noqa: D102
        pad_src_row, pad_valid = ctx.saved_tensors
        grad_pad = grad_rows.index_select(0, pad_src_row) * pad_valid.unsqueeze(
            1
        )
        return grad_pad.to(ctx.pad_dtype), None, None, None


class _IndexedBias(torch.autograd.Function):
    """Per-row expert bias whose gradient is a dense one-hot GEMM.

    Forward: ``bias[idx]`` (a coalesced gather). Autograd's backward for that
    gather is an atomic ``index_add`` of ~400k rows into E=8 — measured
    4.4ms/call on H200 from pure atomic contention (24 calls/step = 106ms).
    Instead: ``grad_bias = onehot @ grad_rows`` with the ``[E, rows]`` one-hot
    (validity-masked) — a ~10 GFLOP deterministic GEMM.
    """

    @staticmethod
    def forward(ctx, bias, idx, onehot):  # noqa: D102
        ctx.save_for_backward(onehot)
        return bias.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad_rows):  # noqa: D102
        (onehot,) = ctx.saved_tensors
        return onehot.to(grad_rows.dtype) @ grad_rows, None, None


def fused_gate_requested() -> bool:
    """Whether ``NESTOR_GROUPED_MM_FUSED_GATE=1`` asks for the fused up+gate GEMM.

    Read by the expert-bearing modules at CONSTRUCTION (into an attribute they
    pass as ``fused_gate``), never inside the forward, so two twins of one
    model can coexist in a process under different settings (the paired
    profiling harness relies on that).

    Returns:
        bool: True iff the flag is set to ``"1"``.
    """

    import os

    return os.environ.get("NESTOR_GROUPED_MM_FUSED_GATE") == "1"


class _DispatchPlan(NamedTuple):
    """The static-shape dispatch of one bank call.

    Attributes:
        rows (int): Routed rows, ``num_tokens * slots``.
        rows_pad (int): Padded buffer length (static in ``rows`` and banks).
        offs (Int[Tensor, "num_banks"]): int32 segment ends for the GEMM.
        row_dest (Int[Tensor, "rows"]): Token-major routed row -> padded slot.
        pad_token (Int[Tensor, "rows_pad"]): Padded slot -> source token.
        pad_src_row (Int[Tensor, "rows_pad"]): Padded slot -> routed row.
        eid_pad (Int[Tensor, "rows_pad"]): Padded slot -> bank (0 on slack).
        pad_valid (Tensor): ``[rows_pad]`` 1 on real slots, 0 on slack, in
            the GEMM dtype.
    """

    rows: int
    rows_pad: int
    offs: Tensor
    row_dest: Tensor
    pad_token: Tensor
    pad_src_row: Tensor
    eid_pad: Tensor
    pad_valid: Tensor


def _dispatch_plan(
    expert_ids: Int[Tensor, "rows"],
    num_banks: int,
    slots: int,
    dtype: torch.dtype,
    triton_index: bool,
) -> _DispatchPlan:
    """Sort the routed rows by bank and build every per-row dispatch map.

    Args:
        expert_ids (Int[Tensor, "rows"]): Flat token-major bank index per
            ``(token, slot)`` row.
        num_banks (int): Bank size.
        slots (int): Slots per token.
        dtype (torch.dtype): GEMM dtype (for ``pad_valid``).
        triton_index (bool): Build the maps with the fused Triton kernel
            (integer-exact) instead of the scatter cascade.

    Returns:
        _DispatchPlan: The plan.
    """

    rows = expert_ids.shape[0]
    device = expert_ids.device
    # Sort rows by expert; compute each row's destination in the padded
    # buffer (per-expert start rounded up to the 16-row alignment the grouped
    # GEMM requires). All shapes are static; counts/offsets stay on device.
    # The stable sort's result depends only on the key ORDER, so sorting the
    # bank ids as the narrowest integer type that holds them is the same
    # permutation from a radix sort with 1-2 digit passes instead of 8 (CUDA
    # ``argsort`` is cub's onesweep radix sort: one launch per 8-bit digit).
    sort_keys = (
        expert_ids.to(torch.uint8) if num_banks <= 256
        else expert_ids.to(torch.int16) if num_banks <= 32767
        else expert_ids
    )
    order = torch.argsort(sort_keys, stable=True)
    align = GROUPED_MM_ALIGN
    counts = torch.zeros(
        num_banks, dtype=torch.long, device=device
    ).scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
    padded_counts = (
        torch.div(counts + align - 1, align, rounding_mode="floor") * align
    )
    padded_starts = padded_counts.cumsum(0) - padded_counts
    raw_starts = counts.cumsum(0) - counts
    # Static buffer size; the LAST segment is extended to absorb the slack, so
    # offs[-1] == rows_pad without a host read.
    rows_pad = ((rows + align - 1) // align) * align + align * num_banks
    padded_ends = padded_counts.cumsum(0)
    offs = torch.cat(
        [
            padded_ends[:-1],
            torch.full((1,), rows_pad, dtype=padded_ends.dtype, device=device),
        ]
    ).to(torch.int32)

    # Inverse maps so every data movement is a coalesced gather (see
    # _DispatchGather/_CombineGather): the dispatch is a bijection between
    # (token, slot) rows and real padded slots, so scatters are never needed.
    if triton_index and expert_ids.is_cuda and triton_index_available():
        # ONE kernel writes all five maps (integer-exact vs the cascade).
        row_dest, pad_token, pad_src_row, eid_pad, pad_valid = (
            build_dispatch_maps(
                expert_ids,
                order,
                padded_starts,
                raw_starts,
                slots,
                rows_pad,
                dtype,
            )
        )
    else:
        # The cascade: cheap 1-D integer scatters on static shapes.
        sorted_eids = expert_ids.index_select(0, order)
        sorted_tokens = torch.div(order, slots, rounding_mode="floor")
        dest = (
            padded_starts.index_select(0, sorted_eids)
            - raw_starts.index_select(0, sorted_eids)
            + torch.arange(rows, device=device)
        )
        long_zeros = torch.zeros(rows_pad, dtype=torch.long, device=device)
        # token-major routed row -> its padded slot
        row_dest = torch.zeros(rows, dtype=torch.long, device=device).scatter_(
            0, order, dest
        )
        # padded slot -> source token (slack -> 0, masked by pad_valid)
        pad_token = long_zeros.scatter(0, dest, sorted_tokens)
        # padded slot -> token-major routed row (slack -> 0, masked)
        pad_src_row = long_zeros.scatter(0, dest, order)
        # padded slot -> bank index (slack -> 0; bias garbage on zeroed slack
        # rows is never read back)
        eid_pad = long_zeros.scatter(0, dest, sorted_eids)
        pad_valid = torch.zeros(rows_pad, dtype=dtype, device=device).scatter(
            0, dest, torch.ones(rows, dtype=dtype, device=device)
        )
    return _DispatchPlan(
        rows, rows_pad, offs, row_dest, pad_token, pad_src_row, eid_pad, pad_valid
    )


def _expert_rows_padded(
    x: Float[Tensor, "num_tokens d_in"],
    experts: Sequence[nn.Module],
    plan: _DispatchPlan,
    slots: int,
    dtype: torch.dtype,
    *,
    fused_gate: bool,
    stacked: "StackedExpertBank | None",
    triton_fused: bool,
    triton_cat_cast: bool = True,
) -> Float[Tensor, "rows_pad d_out"]:
    """Dispatch, run the bank's GEMMs and activation; return the PADDED rows.

    Args:
        x (Float[Tensor, "num_tokens d_in"]): Token features.
        experts (Sequence[nn.Module]): The bank (see :func:`grouped_expert_rows`).
        plan (_DispatchPlan): The dispatch plan of this call.
        slots (int): Slots per token.
        dtype (torch.dtype): GEMM dtype.
        fused_gate (bool): Up + gate as one grouped GEMM (SwiGLU banks).
        stacked (StackedExpertBank | None): Stacked owners, if any.
        triton_fused (bool): Use the fused Triton dispatch gather and SwiGLU
            (caller guarantees :func:`bank_is_fusable`).
        triton_cat_cast (bool): Under ``triton_fused`` with stacked owners,
            build the up|gate GEMM operand with the fused cat+cast kernel
            (coalesced split backward) instead of torch cat + cast.

    Returns:
        Float[Tensor, "rows_pad d_out"]: The padded down-GEMM output (slack
        rows are garbage-free zeros' images: never read back).
    """

    num_banks = len(experts)
    device = x.device
    gated = getattr(experts[0], "gated", False)
    w_up_gate: Tensor | None = None
    if stacked is not None and triton_fused and gated and fused_gate:
        # ONE fused cat+cast kernel whose backward writes each bank's fp32
        # gradient contiguously (bit-exact vs cat-then-cast). Either torch
        # ordering leaves a STRIDED pass in the backward — the halves of the
        # bf16 gradient cast, or the fp32 halves accumulated into .grad —
        # at ~3x the bandwidth-bound time (55 x 181 us per 500m micro-step).
        if triton_cat_cast:
            w_up_gate = cat_cast(stacked.w_up, stacked.w_gate, dtype)
        else:
            w_up_gate = torch.cat([stacked.w_up, stacked.w_gate], dim=-1).to(dtype)
        w1 = w_up_gate[:, :, : stacked.w_up.shape[-1]]  # width only
        w2 = stacked.w_down.to(dtype)
    elif stacked is not None:
        # Stacked owners already sit in the GEMM layout: one cast per bank.
        w1, w_gate, w2 = stacked.cast(dtype)
    else:
        w1 = (
            torch.stack([e.layers[0].weight for e in experts])
            .to(dtype)
            .transpose(-2, -1)
            .contiguous()
        )
        w2 = (
            torch.stack([e.layers[1].weight for e in experts])
            .to(dtype)
            .transpose(-2, -1)
            .contiguous()
        )
        if gated:
            # SwiGLU: extra gate bank, no bias (matches GatedScaledMLP.forward).
            w_gate = (
                torch.stack([e.gate.weight for e in experts])
                .to(dtype)
                .transpose(-2, -1)
                .contiguous()
            )

    if triton_fused:
        # ONE kernel: cast + gather + slack-zero (backward: fused gather-sum).
        x_pad = dispatch_rows(
            x, plan.pad_token, plan.pad_valid, plan.row_dest, slots, dtype
        )
    else:
        x_pad = _DispatchGather.apply(
            x.to(dtype), plan.pad_token, plan.pad_valid, plan.row_dest, slots
        )
    offs = plan.offs
    gate_hidden: Tensor | None = None
    if gated and fused_gate:
        # Up and gate in ONE grouped GEMM over [E, d_in, 2*hidden]: the
        # dispatched rows are read once and the GEMM runs at twice the
        # width. Column j < hidden of the result is up_j, column hidden + j is
        # gate_j -- exactly the two dot products the separate GEMMs compute.
        hidden_width = w1.shape[-1]
        if w_up_gate is None:
            w_up_gate = torch.cat([w1, w_gate], dim=-1)
        up_gate = segment_mm(x_pad, w_up_gate, offs)
        if triton_fused:
            # silu(gate) * up read in place; the backward writes the whole
            # [rows_pad, 2*hidden] gradient (no zero-filled slice grads).
            return segment_mm(swiglu_fused(up_gate, hidden_width), w2, offs)
        hidden = up_gate[:, :hidden_width]
        gate_hidden = up_gate[:, hidden_width:]
    else:
        hidden = segment_mm(x_pad, w1, offs)  # up projection
    if experts[0].layers[0].bias is not None:
        # Stacked owners hand the bias bank as ONE cast; the per-expert stack
        # would otherwise re-gather E bias views of that same owner.
        b1 = (
            stacked.b_up.to(dtype)
            if stacked is not None
            else torch.stack([e.layers[0].bias for e in experts]).to(dtype)
        )
        onehot_pad = torch.zeros(
            num_banks, plan.rows_pad, dtype=dtype, device=device
        ).scatter_(0, plan.eid_pad.unsqueeze(0), plan.pad_valid.unsqueeze(0))
        hidden = hidden + _IndexedBias.apply(b1, plan.eid_pad, onehot_pad)
    if gated:
        # down(act(gate(x)) * up(x)): gate via its own grouped GEMM (or the
        # fused half above), then gate. The product is a fresh contiguous
        # tensor either way, so the down GEMM's operand layout is unchanged.
        if gate_hidden is None:
            gate_hidden = segment_mm(x_pad, w_gate, offs)
        if triton_fused:
            hidden = swiglu(hidden, gate_hidden)
        else:
            hidden = experts[0]._activation_fn(gate_hidden) * hidden
    else:
        hidden = experts[0]._activation_fn(hidden)
    return segment_mm(hidden, w2, offs)  # down projection


def grouped_expert_rows(
    x: Float[Tensor, "num_tokens d_in"],
    experts: Sequence[nn.Module],
    expert_ids: Int[Tensor, "num_tokens slots"],
    *,
    fused_gate: bool = False,
    stacked: "StackedExpertBank | None" = None,
    triton_index: bool = False,
) -> Float[Tensor, "num_tokens*slots d_out"]:
    """Run every ``(token, slot)`` pair through its expert in grouped GEMMs.

    Row ``r = token * slots + slot`` of the result is
    ``experts[expert_ids[token, slot]](x[token])``. The caller applies the
    routing weights and sums over ``slots`` -- keeping that outside lets each
    call site choose its own combine dtype (or use
    :func:`grouped_expert_combine`, which does both and can fuse them).

    All ``experts`` must be shape-identical banks of the same kind: either
    2-layer :class:`~nestor.scaling.blocks.ScaledMLP`-style experts
    (``layers[0]`` -> activation -> ``layers[1]``) or
    :class:`~nestor.scaling.blocks.GatedScaledMLP` SwiGLU experts (attribute
    ``gated`` True, extra ``gate`` projection). Biases are optional and
    handled through :class:`_IndexedBias`.

    Every expert weight enters the autograd graph (they are stacked), so an
    expert that receives no rows still ends the step with a (zero) gradient --
    DDP's reducer sees identical collectives on every rank without a keepalive.

    Args:
        x (Float[Tensor, "num_tokens d_in"]): Token features (any dtype; cast
            to the autocast dtype internally).
        experts (Sequence[nn.Module]): The expert bank, indexed by
            ``expert_ids``.
        expert_ids (Int[Tensor, "num_tokens slots"]): Bank index of each
            ``(token, slot)`` pair. Values must lie in ``[0, len(experts))``.
        fused_gate (bool): SwiGLU banks only. Run the up and gate projections
            as ONE grouped GEMM over the concatenated ``[E, d_in, 2*hidden]``
            weights instead of two GEMMs that each re-read the dispatched
            rows. Every output column is the same dot product either way, so
            the result matches the unfused form to GEMM-tiling noise; the
            saving is one full read of ``x_pad`` per bank and a GEMM at twice
            the width (``N = 2 * hidden`` instead of ``hidden``, which fills
            tensor-core tiles better at the skinny expert widths the column
            arms use). Ignored for non-gated banks. Set from
            ``NESTOR_GROUPED_MM_FUSED_GATE=1`` by the callers
            (:func:`fused_gate_requested`).
        stacked (StackedExpertBank | None): The bank already owned as stacked
            ``[E, d_in, hidden]`` / ``[E, hidden, d_out]`` parameters (plus
            ``[E, hidden]`` / ``[E, d_out]`` bias banks when the experts carry
            biases; ``NESTOR_STACKED_EXPERT_BANKS``); when given, each bank is
            one cast instead of the per-call
            ``stack -> cast -> transpose -> contiguous`` chain, and its
            backward is one gradient per bank instead of one per expert
            weight. ``experts`` is still consulted for the activation and for
            bias presence (the views installed by the stacked owner keep
            those attributes live).
        triton_index (bool): Build every per-row dispatch map in ONE Triton
            kernel (:func:`~nestor.backbones.triton_dispatch.
            build_dispatch_maps`) instead of the ~10-kernel scatter cascade.
            Integer-exact — same maps, same math, pure launch-count saving.
            Set from ``NESTOR_TRITON_INDEX_BUILD=1`` by the callers
            (:func:`~nestor.backbones.triton_dispatch.
            triton_index_requested`); silently falls back off-CUDA.

    Returns:
        Float[Tensor, "num_tokens*slots d_out"]: Token-major expert outputs in
        the GEMM (autocast) dtype, unweighted and un-summed.
    """

    num_banks = len(experts)
    num_tokens, slots = expert_ids.shape
    rows = num_tokens * slots
    device = x.device
    expert_ids = expert_ids.reshape(rows)
    # Match F.linear-under-autocast numerics: inputs/weights/biases cast to
    # the autocast dtype, fp32 accumulation inside the GEMM.
    dtype = autocast_dtype(x)
    plan = _dispatch_plan(expert_ids, num_banks, slots, dtype, triton_index)
    out_pad = _expert_rows_padded(
        x, experts, plan, slots, dtype,
        fused_gate=fused_gate, stacked=stacked, triton_fused=False,
    )

    # Back to token-major rows; the bias and the caller's combine weights all
    # use direct (un-permuted) indices.
    routed = _CombineGather.apply(
        out_pad, plan.row_dest, plan.pad_src_row, plan.pad_valid
    )
    if experts[0].layers[1].bias is not None:
        b2 = (
            stacked.b_down.to(dtype)
            if stacked is not None
            else torch.stack([e.layers[1].bias for e in experts]).to(dtype)
        )
        onehot_rows = torch.zeros(
            num_banks, rows, dtype=dtype, device=device
        ).scatter_(
            0,
            expert_ids.unsqueeze(0),
            torch.ones(1, rows, dtype=dtype, device=device),
        )
        routed = routed + _IndexedBias.apply(b2, expert_ids, onehot_rows)
    return routed


def grouped_expert_combine(
    x: Float[Tensor, "num_tokens d_in"],
    experts: Sequence[nn.Module],
    expert_ids: Int[Tensor, "num_tokens slots"],
    weights: Float[Tensor, "num_tokens slots"],
    group: int,
    *,
    fused_gate: bool = False,
    stacked: "StackedExpertBank | None" = None,
    triton_index: bool = False,
    triton_fused: bool = False,
    triton_cat_cast: bool = True,
) -> Float[Tensor, "num_tokens slots_over_group d_out"]:
    """:func:`grouped_expert_rows`, weighted and summed over slot groups.

    Computes, in ``x``'s dtype, ``(rows * weights).to(x.dtype).view(
    num_tokens, slots // group, group, -1).sum(2)`` -- the combine every call
    site performs (``group = slots`` for a plain weighted sum, ``group = k_q``
    for the per-attention-slot layout of the private pools). Under
    ``triton_fused`` (bias-free SwiGLU banks on CUDA) the dispatch gather, the
    SwiGLU and this weighted combine run as the fused kernels of
    :mod:`nestor.backbones.triton_expert` around the same grouped GEMMs; the
    weighted ``[rows, d_out]`` intermediate never exists. Exact to bf16
    reassociation noise; other banks silently take the eager chain.

    Args:
        x (Float[Tensor, "num_tokens d_in"]): Token features.
        experts (Sequence[nn.Module]): The expert bank.
        expert_ids (Int[Tensor, "num_tokens slots"]): Bank index per pair.
        weights (Float[Tensor, "num_tokens slots"]): Combine weight per pair.
        group (int): Consecutive slots summed into one output entry; must
            divide ``slots``.
        fused_gate (bool): See :func:`grouped_expert_rows`.
        stacked (StackedExpertBank | None): See :func:`grouped_expert_rows`.
        triton_index (bool): See :func:`grouped_expert_rows`.
        triton_fused (bool): ``NESTOR_TRITON_EXPERT_DISPATCH`` -- the fused
            Triton dispatch / SwiGLU / combine kernels.
        triton_cat_cast (bool): ``NESTOR_TRITON_EXPERT_CATCAST`` -- the fused
            cat+cast of a stacked up|gate bank (under ``triton_fused``).

    Returns:
        Float[Tensor, "num_tokens slots_over_group d_out"]: The weighted slot
        sums in ``x``'s dtype.
    """

    num_tokens, slots = expert_ids.shape
    assert slots % group == 0, "group must divide the slot count"
    use_fused = (
        triton_fused
        and x.is_cuda
        and triton_expert_available()
        and bank_is_fusable(list(experts), fused_gate)
    )
    if not use_fused:
        rows = grouped_expert_rows(
            x, experts, expert_ids,
            fused_gate=fused_gate, stacked=stacked, triton_index=triton_index,
        )
        weighted = (rows * weights.reshape(-1, 1)).to(x.dtype)
        return weighted.view(num_tokens, slots // group, group, -1).sum(2)

    dtype = autocast_dtype(x)
    plan = _dispatch_plan(
        expert_ids.reshape(-1), len(experts), slots, dtype, triton_index
    )
    out_pad = _expert_rows_padded(
        x, experts, plan, slots, dtype,
        fused_gate=fused_gate, stacked=stacked, triton_fused=True,
        triton_cat_cast=triton_cat_cast,
    )
    combined = combine_rows(
        out_pad,
        plan.row_dest,
        plan.pad_src_row,
        plan.pad_valid,
        weights.reshape(-1),
        group,
        x.dtype,
    )  # [num_tokens * slots // group, d_out]
    return combined.view(num_tokens, slots // group, -1)
