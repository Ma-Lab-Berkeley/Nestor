"""Equivalence tests for chunked cross-entropy (``NESTOR_CE_CHUNK_ROWS``).

One fused ``F.cross_entropy(reduction="sum")`` retains the full fp32
log-softmax for backward -- ~13 GiB at the 1b arm's per-rank 8, the allocation
that made that batch shape OOM by ~20 MiB. Chunking the rows, each chunk under
activation checkpointing, retains one chunk's softmax at a time and recomputes
in backward.

Sum reduction makes the split EXACT up to floating-point association: these
tests pin the loss value AND the gradient of the logits against the unchunked
reference, including ignore_index rows, a chunk size that does not divide the
row count, and a chunk larger than the input.
"""

from __future__ import annotations

import os
from typing import Self

import torch

from nestor.constants import PADDING_TOKEN_VALUE
from nestor.train.losses import CausalLMLoss


class _ChunkEnv:
    """Set/restore ``NESTOR_CE_CHUNK_ROWS`` (no monkeypatching).

    Attributes:
        value (str): Value inside the block.
        previous (str | None): Restored on exit (None -> unset).
    """

    def __init__(self, value: int) -> None:
        self.value: str = str(value)
        self.previous: str | None = None

    def __enter__(self) -> Self:
        self.previous = os.environ.get("NESTOR_CE_CHUNK_ROWS")
        os.environ["NESTOR_CE_CHUNK_ROWS"] = self.value
        return self

    def __exit__(self, *exc: object) -> None:
        if self.previous is None:
            os.environ.pop("NESTOR_CE_CHUNK_ROWS", None)
        else:
            os.environ["NESTOR_CE_CHUNK_ROWS"] = self.previous


def _loss_and_grad(
    chunk_rows: int, *, label_smoothing: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Loss, logits gradient, and token count at a given chunk setting.

    Fixed seed; every call sees identical logits/targets, including some
    ignore_index targets, so the chunk setting is the only difference.

    Args:
        chunk_rows (int): ``NESTOR_CE_CHUNK_ROWS`` value (0 = fused reference).
        label_smoothing (float): Passed through to the loss.

    Returns:
        tuple[Tensor, Tensor, int]: Sum loss, d(loss)/d(pred), token count.
    """

    torch.manual_seed(0)
    batch, seq, vocab = 3, 33, 97
    pred = torch.randn(batch, seq, vocab, requires_grad=True)
    target = torch.randint(0, vocab, (batch, seq))
    target[0, 5:9] = PADDING_TOKEN_VALUE  # ignored rows inside a chunk
    with _ChunkEnv(chunk_rows):
        loss, n = CausalLMLoss.sum_loss_and_tokens(
            pred, target, label_smoothing=label_smoothing
        )
    loss.backward()
    return loss.detach(), pred.grad.detach().clone(), n


class ChunkedCrossEntropyTest:
    """Chunked CE must match the fused call exactly (fp32 association)."""

    def test_matches_fused_loss_and_grad(self) -> None:
        """Chunk sizes that split, do not divide, and exceed the rows."""

        ref_loss, ref_grad, ref_n = _loss_and_grad(0)
        for chunk in (16, 25, 10_000):
            loss, grad, n = _loss_and_grad(chunk)
            assert n == ref_n
            torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(grad, ref_grad, rtol=1e-6, atol=1e-6)

    def test_label_smoothing_passes_through(self) -> None:
        """The chunk closure must forward label_smoothing, not drop it."""

        ref_loss, ref_grad, _ = _loss_and_grad(0, label_smoothing=0.1)
        loss, grad, _ = _loss_and_grad(25, label_smoothing=0.1)
        torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(grad, ref_grad, rtol=1e-6, atol=1e-6)

    def test_no_grad_path_unchanged(self) -> None:
        """Eval-time calls (no grad) must bypass chunking entirely."""

        torch.manual_seed(0)
        pred = torch.randn(2, 17, 31)  # no requires_grad
        target = torch.randint(0, 31, (2, 17))
        with _ChunkEnv(8):
            loss_chunk, n1 = CausalLMLoss.sum_loss_and_tokens(pred, target)
        with _ChunkEnv(0):
            loss_ref, n2 = CausalLMLoss.sum_loss_and_tokens(pred, target)
        assert n1 == n2
        torch.testing.assert_close(loss_chunk, loss_ref)
