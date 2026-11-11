"""Single-pass streamed vocabulary head + cross-entropy.

The language-modelling head is one ``[vocab, d]`` linear map read by a softmax.
Applied to a whole micro-batch it produces ``[tokens, vocab]`` logits -- 3.3 GiB
in bf16 at 4 x 4096 tokens over the 100k vocabulary, 6.6 GiB at 8 x 4096 -- and
autograd then walks that tensor many times: the ``[:, :-1]`` shift copies it,
every chunk of the row-chunked cross entropy (``NESTOR_CE_CHUNK_ROWS``) pads its
gradient back to the full ``[tokens, vocab]`` shape, and the pieces are summed.
The checkpointed in-loss head (``NESTOR_FUSED_HEAD_CE``) never stores the full
logits but pays for it by projecting every chunk twice (once to read the loss,
once more in backward).

``NESTOR_STREAMED_HEAD_CE=1`` streams the head instead: for each row chunk it
projects the hidden states to logits, reads the cross entropy, forms the logits
gradient ``softmax - onehot`` in place, and immediately contracts it against
the head weight (the hidden-state gradient) and the hidden states (the weight
gradient). Nothing of size ``[tokens, vocab]`` outlives its chunk, nothing is
recomputed, and the only tensors kept for backward are the two results,
``[tokens, d]`` and ``[vocab, d]``. Backward multiplies them by the incoming
scalar.

The math is the standard path's: the projection is per-token, so it commutes
with the loss's target shift and with row chunking; the cross entropy is
sum-reduced, so the chunk split is exact; and ``softmax - onehot`` is the
analytic gradient autograd derives. Under autocast the logits and both gradient
GEMMs run in the autocast dtype and the softmax in fp32, as they do on the
standard path. The one difference is where the loss scale is applied: autograd
scales the fp32 logits gradient by ``1 / (tokens * grad_accum)`` before
rounding it to bf16, this path rounds first and scales the GEMM results. bf16
rounding is relative, so the two agree to rounding (and bit for bit in fp32 /
fp64 up to summation order). fp16 is the exception -- its narrow exponent range
would underflow the unscaled gradient -- so under fp16 autocast the gradient
GEMMs run in fp32.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from nestor.scaling.blocks import ScaledLinear

DEFAULT_CHUNK_ROWS: int = 4096
"""Rows per chunk when ``NESTOR_CE_CHUNK_ROWS`` is unset or 0."""


def fused_head_mode() -> str | None:
    """Which in-loss head path the environment selects.

    Returns:
        str | None: ``"stream"`` under ``NESTOR_STREAMED_HEAD_CE=1`` (the
            single-pass path of this module; it takes precedence),
            ``"checkpoint"`` under ``NESTOR_FUSED_HEAD_CE=1`` (the row-chunked
            checkpointed projection), ``None`` when the head runs inside the
            model forward and the loss receives logits.
    """

    if os.environ.get("NESTOR_STREAMED_HEAD_CE") == "1":
        return "stream"
    if os.environ.get("NESTOR_FUSED_HEAD_CE") == "1":
        return "checkpoint"
    return None


def streamable_projection(head: nn.Module) -> ScaledLinear | None:
    """Return the head's linear map when the head is exactly one linear map.

    The streamed path differentiates the head by hand, which is only valid for
    a single affine readout. A head with hidden layers is not streamable; the
    loss then falls back to the checkpointed projection.

    Args:
        head (nn.Module): The model head (a ``LanguageModelingHead`` whose
            ``projection`` is a ``ScaledMLP``).

    Returns:
        ScaledLinear | None: The single readout layer, or ``None`` when the
            head has hidden layers or is not built from ``ScaledLinear``.
    """

    layers = getattr(getattr(head, "projection", None), "layers", None)
    if layers is None or len(layers) != 1:
        return None
    layer = layers[0]
    return layer if isinstance(layer, ScaledLinear) else None


class _StreamedHeadCE(torch.autograd.Function):
    """Sum-reduced cross entropy of a linear readout, differentiated in forward.

    ``forward`` returns the summed loss and stores the gradients of that sum
    with respect to the hidden states, the weight and the bias; ``backward``
    scales them by the incoming gradient. See the module docstring.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        targets: Tensor,
        logit_multiplier: float,
        chunk_rows: int,
        ignore_index: int,
        label_smoothing: float,
    ) -> Tensor:
        """Stream the head over row chunks; return the summed cross entropy.

        Args:
            ctx (Any): Autograd context.
            hidden (Tensor): Hidden states. Shape: ``(rows, d)``.
            weight (Tensor): Readout weight. Shape: ``(vocab, d)``.
            bias (Tensor | None): Readout bias. Shape: ``(vocab,)``.
            targets (Tensor): Target token ids. Shape: ``(rows,)``, int64.
            logit_multiplier (float): Scalar applied to the logits.
            chunk_rows (int): Rows projected per chunk (positive).
            ignore_index (int): Target value excluded from the loss.
            label_smoothing (float): Label smoothing in ``[0, 1)``.

        Returns:
            Tensor: Summed loss over the non-ignored rows. Shape: ``()``,
                fp32 (fp64 when the inputs are fp64).
        """

        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
        need_bias = need_bias and bias is not None
        rows, vocab = hidden.shape[0], weight.shape[0]
        device_type = hidden.device.type
        # fp64 stays fp64 (the reference tests); everything else reads the
        # softmax in fp32, as cross entropy does under autocast.
        stat_dtype = (
            torch.float64 if weight.dtype == torch.float64 else torch.float32
        )

        # The ambient autocast picks the logits dtype, exactly as it does for
        # the head inside the model forward; without autocast the logits take
        # the readout's dtype.
        logit_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type)
            else weight.dtype
        )
        # Gradient GEMMs run in the logits dtype; fp16 would underflow the
        # unscaled gradient (see the module docstring), so it widens.
        gemm_dtype = torch.float32 if logit_dtype == torch.float16 else logit_dtype
        # The readout is cast ONCE per call. Left to autocast, ``F.linear``
        # would re-cast the fp32 [vocab, d] weight for every chunk.
        weight_logit = weight.to(logit_dtype)  # [vocab, d]
        bias_logit = None if bias is None else bias.to(logit_dtype)  # [vocab]
        weight_gemm = (
            weight_logit if gemm_dtype == logit_dtype else weight.to(gemm_dtype)
        )  # [vocab, d]

        loss_sum = torch.zeros((), dtype=stat_dtype, device=hidden.device)
        grad_hidden: Tensor | None = (
            torch.empty(
                (rows, hidden.shape[1]), dtype=gemm_dtype, device=hidden.device
            )
            if need_hidden
            else None
        )  # [rows, d]
        grad_weight: Tensor | None = None  # [vocab, d], stat dtype
        grad_bias: Tensor | None = None  # [vocab], stat dtype

        for start in range(0, rows, chunk_rows):
            h = hidden[start : start + chunk_rows]  # [c, d]
            t = targets[start : start + chunk_rows]  # [c]
            logits = F.linear(h.to(logit_dtype), weight_logit, bias_logit)  # [c, vocab]
            if logit_multiplier != 1.0:
                logits = logits.mul_(logit_multiplier)
            log_probs = F.log_softmax(logits.to(stat_dtype), dim=-1)  # [c, vocab]
            del logits

            valid = t != ignore_index  # [c]
            index = torch.where(valid, t, torch.zeros_like(t)).unsqueeze(1)  # [c, 1]
            row_loss = -log_probs.gather(1, index).squeeze(1)  # [c]
            if label_smoothing > 0.0:
                row_loss = (1.0 - label_smoothing) * row_loss - (
                    label_smoothing / vocab
                ) * log_probs.sum(dim=-1)
            loss_sum = loss_sum + torch.where(
                valid, row_loss, torch.zeros_like(row_loss)
            ).sum()

            if not (need_hidden or need_weight or need_bias):
                continue

            # d(loss sum)/d(scaled logits) = softmax - (1 - eps) onehot
            # - eps / vocab on the valid rows, zero on the ignored ones. Built
            # in place; the ignored-row mask and the logit multiplier are
            # applied to the SMALL operand of each product below (row i of
            # every product depends on row i of this matrix only), which
            # spares two more passes over [c, vocab].
            grad_logits = log_probs.exp_()  # [c, vocab]
            del log_probs
            grad_logits.scatter_add_(
                1,
                index,
                torch.full_like(index, -(1.0 - label_smoothing), dtype=stat_dtype),
            )
            if label_smoothing > 0.0:
                grad_logits.sub_(label_smoothing / vocab)

            # Explicit dtypes from here on: these products are the backward of
            # the logits GEMM and must not be re-cast by the ambient autocast.
            with torch.autocast(device_type=device_type, enabled=False):
                if need_bias:
                    # sum over the valid rows, in the stat dtype: [vocab]
                    column_sum = (
                        valid.to(stat_dtype) * logit_multiplier
                    ) @ grad_logits
                    grad_bias = (
                        column_sum if grad_bias is None else grad_bias + column_sum
                    )
                grad_logits = grad_logits.to(gemm_dtype)
                row_scale = (
                    valid.to(gemm_dtype) * logit_multiplier
                ).unsqueeze(1)  # [c, 1]
                if grad_hidden is not None:
                    grad_hidden[start : start + chunk_rows] = (
                        grad_logits @ weight_gemm
                    ) * row_scale
                if need_weight:
                    outer = (
                        grad_logits.t() @ (h.to(gemm_dtype) * row_scale)
                    ).to(stat_dtype)  # [vocab, d]
                    grad_weight = (
                        outer if grad_weight is None else grad_weight.add_(outer)
                    )
            del grad_logits

        ctx.grad_hidden = grad_hidden
        ctx.grad_weight = grad_weight
        ctx.grad_bias = grad_bias
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return loss_sum

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor | None, ...]:
        """Scale the gradients stored by ``forward`` by the incoming scalar.

        Args:
            ctx (Any): Autograd context filled by ``forward``.
            grad_output (Tensor): Gradient of the objective with respect to
                the summed loss. Shape: ``()``.

        Returns:
            tuple[Tensor | None, ...]: Gradients for ``hidden``
                ``(rows, d)``, ``weight`` ``(vocab, d)`` and ``bias``
                ``(vocab,)`` in their own dtypes (``None`` where not
                required), then ``None`` for the five non-differentiable
                arguments.
        """

        def _scaled(grad: Tensor | None, dtype: torch.dtype | None) -> Tensor | None:
            """``grad_output * grad`` in ``dtype``; ``None`` passes through.

            The product is taken in fp32 or wider. A 0-dim scalar does not
            promote a bf16 tensor, so multiplying them directly would round
            the loss scale itself to bf16 (8 mantissa bits).
            """

            if grad is None or dtype is None:
                return None
            wide = torch.promote_types(
                torch.promote_types(grad.dtype, grad_output.dtype), torch.float32
            )
            return (grad.to(wide) * grad_output.to(wide)).to(dtype)

        return (
            _scaled(ctx.grad_hidden, ctx.hidden_dtype),
            _scaled(ctx.grad_weight, ctx.weight_dtype),
            _scaled(ctx.grad_bias, ctx.bias_dtype),
            None,
            None,
            None,
            None,
            None,
        )


def streamed_head_ce(
    hidden: Tensor,
    targets: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    logit_multiplier: float = 1.0,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Sum-reduced cross entropy of ``(hidden @ weight.T + bias) * multiplier``.

    Equal to ``F.cross_entropy(F.linear(hidden, weight, bias) * multiplier,
    targets, ignore_index=..., label_smoothing=..., reduction="sum")`` in
    value and in the gradients it sends to ``hidden``, ``weight`` and ``bias``,
    without materializing the ``(rows, vocab)`` logits (see the module
    docstring).

    Assumes ``targets`` holds either ``ignore_index`` or a valid class id in
    ``[0, vocab)`` (not checked: the check would cost a device sync).

    Args:
        hidden (Tensor): Hidden states, already shifted to the predicting
            positions. Shape: ``(rows, d)``.
        targets (Tensor): Target token ids. Shape: ``(rows,)``, int64.
        weight (Tensor): Readout weight. Shape: ``(vocab, d)``.
        bias (Tensor | None): Readout bias. Shape: ``(vocab,)``.
        logit_multiplier (float): Scalar applied to the logits.
        chunk_rows (int): Rows projected per chunk; non-positive values fall
            back to ``DEFAULT_CHUNK_ROWS``.
        ignore_index (int): Target value excluded from the loss.
        label_smoothing (float): Label smoothing in ``[0, 1)``.

    Returns:
        Tensor: Summed loss over the non-ignored rows. Shape: ``()``.

    Raises:
        ValueError: If the shapes of ``hidden``, ``targets`` and ``weight`` do
            not agree.
    """

    if hidden.dim() != 2 or targets.shape != hidden.shape[:1]:
        raise ValueError(
            "streamed_head_ce expects hidden (rows, d) and targets (rows,), got "
            f"{tuple(hidden.shape)} and {tuple(targets.shape)}."
        )
    if weight.dim() != 2 or weight.shape[1] != hidden.shape[1]:
        raise ValueError(
            f"weight {tuple(weight.shape)} does not read hidden {tuple(hidden.shape)}."
        )
    return _StreamedHeadCE.apply(
        hidden,
        weight,
        bias,
        targets,
        float(logit_multiplier),
        int(chunk_rows) if chunk_rows > 0 else DEFAULT_CHUNK_ROWS,
        int(ignore_index),
        float(label_smoothing),
    )
