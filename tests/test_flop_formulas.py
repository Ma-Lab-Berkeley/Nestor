"""Correctness tests for the extra FLOP formulas used by MFU calibration.

:func:`nestor.train.flop_formulas.grouped_mm_flops` teaches
``FlopCounterMode`` what ``torch._grouped_mm`` costs, so MFU calibration can
measure the grouped-GEMM expert path as it actually runs instead of swapping in
the sparse fallback (which executes a different amount of work and inflated the
reported MFU ~3.7x).

The reference implementation throughout is the obvious per-group loop: a grouped
matmul is exactly a stack of ordinary matmuls, one per group, so its cost is the
sum of those matmuls' costs under the built-in ``mm`` formula (``2*M*K*N``).
"""

from __future__ import annotations

import pytest
import torch

from nestor.train.flop_formulas import grouped_mm_flops, register_extra_flop_formulas


def _reference_grouped_mm_flops(
    rows_per_group: list[int], k: int, n: int
) -> int:
    """Cost of a grouped matmul as the sum of its per-group matmuls.

    Slow and explicit on purpose: loops the groups and charges each the
    standard dense-matmul cost, which is what the grouped kernel does.

    Args:
        rows_per_group (list[int]): Rows routed to each group.
        k (int): Contraction dimension.
        n (int): Output dimension.

    Returns:
        int: Total FLOPs.
    """

    total = 0
    for rows in rows_per_group:
        total += 2 * rows * k * n
    return total


class GroupedMmFlopFormulaTest:
    """Unit tests for :func:`grouped_mm_flops`."""

    def test_matches_per_group_reference(self) -> None:
        """Formula equals the sum of the per-group dense matmuls."""

        k, n = 64, 128
        for rows_per_group in ([8, 8, 8, 8], [1, 31, 0, 96], [512], [3, 5]):
            total_rows = sum(rows_per_group)
            got = grouped_mm_flops((total_rows, k), (len(rows_per_group), k, n))
            assert got == _reference_grouped_mm_flops(rows_per_group, k, n)

    def test_independent_of_group_split(self) -> None:
        """Only the row COUNT matters, not how rows split across groups.

        ``offs`` selects which weight matrix a row uses; it does not change how
        much arithmetic that row costs. Two different splits of the same rows
        must therefore cost the same.
        """

        k, n = 32, 64
        a = grouped_mm_flops((100, k), (4, k, n))
        b = grouped_mm_flops((100, k), (8, k, n))
        assert a == b == 2 * 100 * k * n

    def test_matches_dense_mm_convention(self) -> None:
        """One group is an ordinary matmul, and must cost 2*M*K*N."""

        m, k, n = 17, 33, 65
        assert grouped_mm_flops((m, k), (1, k, n)) == 2 * m * k * n

    def test_handles_batched_row_operand(self) -> None:
        """A leading batch dim multiplies the row count."""

        k, n = 16, 32
        assert grouped_mm_flops((5, 7, k), (2, k, n)) == 2 * 5 * 7 * k * n

    def test_ignores_extra_args(self) -> None:
        """The ``offs`` tensor and kwargs must not affect the count."""

        k, n = 8, 16
        base = grouped_mm_flops((10, k), (2, k, n))
        with_extras = grouped_mm_flops(
            (10, k), (2, k, n), (2,), out_shape=(10, n), some_kwarg=object()
        )
        assert base == with_extras

    def test_registration_is_idempotent(self) -> None:
        """Calibration can run more than once per process; re-registering must
        not raise (torch raises on duplicate formula registration)."""

        register_extra_flop_formulas()
        register_extra_flop_formulas()
        register_extra_flop_formulas()


class GroupedMmCounterIntegrationTest:
    """The formula must actually fire inside ``FlopCounterMode``."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="torch._grouped_mm is CUDA-only"
    )
    def test_counter_sees_grouped_mm(self) -> None:
        """A real grouped_mm call is counted, and matches the reference.

        Without the registration this op contributes 0, which is the bug: the
        expert GEMMs -- the bulk of an MoE step -- went uncounted.
        """

        from torch.utils.flop_counter import FlopCounterMode

        register_extra_flop_formulas()

        groups, k, n = 4, 64, 128
        rows_per_group = [8, 16, 24, 16]
        total_rows = sum(rows_per_group)
        a = torch.randn(total_rows, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(groups, k, n, device="cuda", dtype=torch.bfloat16)
        offs = torch.tensor(
            [8, 24, 48, 64], device="cuda", dtype=torch.int32
        )  # cumulative group boundaries

        counter = FlopCounterMode(display=False)
        with counter:
            torch._grouped_mm(a, b, offs=offs)

        assert counter.get_total_flops() == _reference_grouped_mm_flops(
            rows_per_group, k, n
        )
