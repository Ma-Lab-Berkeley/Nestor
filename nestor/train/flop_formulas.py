"""FLOP formulas for ops :class:`torch.utils.flop_counter.FlopCounterMode` cannot count.

``FlopCounterMode`` dispatches on aten ops and carries formulas for the common
ones (``mm``, ``bmm``, ``scaled_dot_product_attention``, ...). ``torch._grouped_mm``
-- the grouped-GEMM expert dispatch behind ``NESTOR_MOE_GROUPED_MM`` -- has no
formula, so an MoE step measured with it enabled leaves the bulk of the model
uncounted.

The previous workaround was to disable the grouped path during calibration and
let the sparse fallback route the same math through counted ``aten::mm`` calls.
That measured a DIFFERENT amount of work than production executes: on
a 1b column arm it reported 1.855e14 FLOPs/microbatch against a
dense-6ND upper bound of 4.991e13 -- 3.7x more than an all-parameters dense
model, which is impossible for a sparsely-activated MoE. MFU was inflated by
roughly that factor, and it moved with the batch shape.

Registering the formula lets calibration run the REAL path, so counted FLOPs
match executed FLOPs.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

_REGISTERED = False


def grouped_mm_flops(
    a_shape: Sequence[int],
    b_shape: Sequence[int],
    *args: Any,
    out_shape: Sequence[int] | None = None,
    **kwargs: Any,
) -> int:
    """FLOPs for ``torch._grouped_mm(a, b, offs=...)``.

    Every row of ``a`` is multiplied by exactly one of ``b``'s per-group
    matrices -- the ``offs`` tensor only selects WHICH group a row uses, it does
    not change how much arithmetic that row costs. So the cost is the same as a
    dense ``[rows, K] @ [K, N]`` matmul: ``2 * rows * K * N``, counting a
    multiply-accumulate as 2 FLOPs (the convention the built-in ``mm`` formula
    uses, so the two are directly comparable).

    Args:
        a_shape (Sequence[int]): Shape of the row operand, ``[rows, K]`` or a
            batched ``[..., rows, K]``.
        b_shape (Sequence[int]): Shape of the grouped weight, ``[G, K, N]``.
        *args (Any): Remaining positional args (``offs``); ignored -- group
            assignment does not change the arithmetic per row.
        out_shape (Sequence[int] | None): Output shape; unused.
        **kwargs (Any): Remaining kwargs; ignored.

    Returns:
        int: FLOPs for the call.
    """

    rows = 1
    for dim in a_shape[:-1]:
        rows *= int(dim)
    k = int(a_shape[-1])
    n = int(b_shape[-1])
    return 2 * rows * k * n


def register_extra_flop_formulas() -> None:
    """Teach ``FlopCounterMode`` about ops it cannot count natively.

    Idempotent: registering the same op twice raises, and calibration may run
    more than once per process (resume, multi-job scripts), so the first
    successful registration latches.

    Never raises. A torch build without ``aten._grouped_mm``, or a
    ``register_flop_formula`` signature change, degrades to the previous
    behaviour (that op uncounted) rather than breaking training.
    """

    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from torch.utils.flop_counter import register_flop_formula

        op = getattr(torch.ops.aten, "_grouped_mm", None)
        if op is None:
            return
        register_flop_formula(op)(grouped_mm_flops)
        _REGISTERED = True
    except Exception:  # noqa: BLE001 -- MFU accounting must never break training
        pass
