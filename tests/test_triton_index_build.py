"""Equivalence tests for the fused Triton dispatch-map build.

``NESTOR_TRITON_INDEX_BUILD=1`` makes ``grouped_expert_rows`` build its five
per-row dispatch maps (``row_dest``, ``pad_token``, ``pad_src_row``,
``eid_pad``, ``pad_valid``) in ONE Triton kernel instead of the ~10-kernel
torch scatter cascade. The maps are pure integer bookkeeping over a stable
sort, so the bar is EXACT equality — same maps, bit for bit, and therefore a
bit-identical ``grouped_expert_rows`` result (same GEMMs on the same rows).

The reference implementation throughout is the torch cascade, replicated
verbatim from ``grouped_expert_rows``. CUDA-only (the kernel is Triton), with
one flag-reader test that runs anywhere.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor

from nestor.backbones.grouped_mm import (
    GROUPED_MM_ALIGN,
    grouped_expert_rows,
)
from nestor.backbones.triton_dispatch import (
    triton_index_available,
    triton_index_requested,
)
from nestor.scaling.blocks import GatedScaledMLP

_CUDA = pytest.mark.skipif(
    not triton_index_available(), reason="needs CUDA + triton"
)


def _reference_maps(
    expert_ids: Tensor, num_banks: int, slots: int, valid_dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The torch cascade of ``grouped_expert_rows``, replicated verbatim.

    Args:
        expert_ids (Tensor): Flat ``[rows]`` bank indices (int64).
        num_banks (int): Bank count.
        slots (int): Slots per token.
        valid_dtype (torch.dtype): dtype of ``pad_valid``.

    Returns:
        tuple: ``(row_dest, pad_token, pad_src_row, eid_pad, pad_valid)``.
    """

    rows = expert_ids.shape[0]
    device = expert_ids.device
    align = GROUPED_MM_ALIGN
    order = torch.argsort(expert_ids, stable=True)
    sorted_eids = expert_ids.index_select(0, order)
    sorted_tokens = torch.div(order, slots, rounding_mode="floor")
    counts = torch.zeros(
        num_banks, dtype=torch.long, device=device
    ).scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
    padded_counts = (
        torch.div(counts + align - 1, align, rounding_mode="floor") * align
    )
    padded_starts = padded_counts.cumsum(0) - padded_counts
    raw_starts = counts.cumsum(0) - counts
    dest = (
        padded_starts.index_select(0, sorted_eids)
        - raw_starts.index_select(0, sorted_eids)
        + torch.arange(rows, device=device)
    )
    rows_pad = ((rows + align - 1) // align) * align + align * num_banks
    long_zeros = torch.zeros(rows_pad, dtype=torch.long, device=device)
    row_dest = torch.zeros(rows, dtype=torch.long, device=device).scatter_(
        0, order, dest
    )
    pad_token = long_zeros.scatter(0, dest, sorted_tokens)
    pad_src_row = long_zeros.scatter(0, dest, order)
    eid_pad = long_zeros.scatter(0, dest, sorted_eids)
    pad_valid = torch.zeros(rows_pad, dtype=valid_dtype, device=device).scatter(
        0, dest, torch.ones(rows, dtype=valid_dtype, device=device)
    )
    return row_dest, pad_token, pad_src_row, eid_pad, pad_valid


@_CUDA
class BuildDispatchMapsTest:
    """The fused kernel's maps == the torch cascade's, exactly."""

    def _check(self, expert_ids_2d: Tensor, num_banks: int) -> None:
        from nestor.backbones.triton_dispatch import build_dispatch_maps

        expert_ids_2d = expert_ids_2d.cuda()
        _num_tokens, slots = expert_ids_2d.shape
        flat = expert_ids_2d.reshape(-1)
        rows = flat.shape[0]
        align = GROUPED_MM_ALIGN
        counts = torch.zeros(
            num_banks, dtype=torch.long, device=flat.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        padded_counts = (
            torch.div(counts + align - 1, align, rounding_mode="floor") * align
        )
        padded_starts = padded_counts.cumsum(0) - padded_counts
        raw_starts = counts.cumsum(0) - counts
        rows_pad = ((rows + align - 1) // align) * align + align * num_banks
        order = torch.argsort(flat, stable=True)
        got = build_dispatch_maps(
            flat, order, padded_starts, raw_starts, slots, rows_pad,
            torch.bfloat16,
        )
        want = _reference_maps(flat, num_banks, slots, torch.bfloat16)
        names = ["row_dest", "pad_token", "pad_src_row", "eid_pad", "pad_valid"]
        for name, g, w in zip(names, got, want):
            assert torch.equal(g, w), name

    def test_uniform_routing(self) -> None:
        torch.manual_seed(0)
        self._check(torch.randint(0, 8, (300, 3)), num_banks=8)

    def test_skewed_and_empty_banks(self) -> None:
        """Banks 0 and 5 empty; bank 2 takes most rows."""

        torch.manual_seed(1)
        ids = torch.randint(1, 5, (200, 2))
        ids[ids == 4] = 2
        self._check(ids, num_banks=6)

    def test_single_bank_top1(self) -> None:
        self._check(torch.zeros(65, 1, dtype=torch.long), num_banks=1)

    def test_row_count_not_block_aligned(self) -> None:
        """Row counts straddling the kernel's 1024-lane BLOCK boundary."""

        torch.manual_seed(2)
        self._check(torch.randint(0, 4, (341, 3)), num_banks=4)  # 1023 rows
        self._check(torch.randint(0, 4, (342, 3)), num_banks=4)  # 1026 rows


@_CUDA
class GroupedExpertRowsTritonIndexTest:
    """End-to-end: ``triton_index=True`` is bit-identical, fwd and grads."""

    def _bank(self, seed: int, num_experts: int = 4) -> list[GatedScaledMLP]:
        torch.manual_seed(seed)
        bank = [
            GatedScaledMLP(16, [24], 16, activation="silu", bias=True)
            for _ in range(num_experts)
        ]
        for expert in bank:
            for param in expert.parameters():
                with torch.no_grad():
                    param.add_(0.1 * torch.randn_like(param))
        return [e.cuda() for e in bank]

    def _run(self, bank, x, expert_ids, *, triton_index: bool):
        for e in bank:
            e.zero_grad(set_to_none=True)
        out = grouped_expert_rows(
            x, bank, expert_ids, fused_gate=True, triton_index=triton_index
        )
        out.float().square().sum().backward()
        grads = [
            p.grad.detach().clone()
            for e in bank
            for p in e.parameters()
            if p.grad is not None
        ]
        return out.detach(), grads

    def test_bit_identical_forward_and_grads(self) -> None:
        torch.manual_seed(3)
        x = torch.randn(120, 16, device="cuda")
        expert_ids = torch.randint(0, 4, (120, 2), device="cuda")
        bank = self._bank(seed=4)
        out_ref, grads_ref = self._run(bank, x, expert_ids, triton_index=False)
        out_tri, grads_tri = self._run(bank, x, expert_ids, triton_index=True)
        assert torch.equal(out_tri, out_ref)
        assert len(grads_ref) == len(grads_tri)
        for g_ref, g_tri in zip(grads_ref, grads_tri):
            assert torch.equal(g_tri, g_ref)

    def test_bit_identical_under_autocast(self) -> None:
        torch.manual_seed(5)
        x = torch.randn(200, 16, device="cuda")
        expert_ids = torch.randint(0, 4, (200, 3), device="cuda")
        bank = self._bank(seed=6)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_ref, grads_ref = self._run(
                bank, x, expert_ids, triton_index=False
            )
            out_tri, grads_tri = self._run(
                bank, x, expert_ids, triton_index=True
            )
        assert torch.equal(out_tri, out_ref)
        for g_ref, g_tri in zip(grads_ref, grads_tri):
            assert torch.equal(g_tri, g_ref)


class FlagReaderTest:
    """``triton_index_requested`` is exactly the ``=1`` test of the flag."""

    def test_flag_reader_follows_env(self) -> None:
        key = "NESTOR_TRITON_INDEX_BUILD"
        previous = os.environ.get(key)
        try:
            os.environ.pop(key, None)
            assert triton_index_requested() is False
            os.environ[key] = "0"
            assert triton_index_requested() is False
            os.environ[key] = "1"
            assert triton_index_requested() is True
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
