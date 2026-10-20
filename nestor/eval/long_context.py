"""Per-token loss by position for long-context evaluation.

The held-out read of the long-context experiments: for every target position
``i`` of a fixed-length held-out sequence, the mean next-token negative
log-likelihood over sequences, optionally smoothed with a centred moving
average and binned, next to the token-weighted mean over all positions (the
loss at that context length).

The loss is computed from the backbone's hidden states with the unembedding
applied only to a slice of targets at a time, so a 32K-token sequence never
materialises a ``(32K, vocab)`` logits tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from nestor.backbones.base import ModelOutput
from nestor.models import Model


def token_nll(
    model: Model,
    tokens: Tensor,
    *,
    target_start: int,
    target_end: int,
    autocast_bf16: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    """Next-token negative log-likelihood of one token slice of a sequence.

    Runs the adapter + backbone on the whole ``tokens`` sequence (causal
    attention, so every prediction sees exactly its prefix) and the head only
    on the hidden states that predict ``tokens[target_start:target_end]``.

    Args:
        model (Model): A ``SequenceModel`` with ``encode`` and ``head``.
        tokens (Tensor): Token ids. Shape: (1, seq_len).
        target_start (int): First target position (``>= 1``: position 0 has no
            prediction).
        target_end (int): One past the last target position (``<= seq_len``).
        autocast_bf16 (bool): Run the forward under bf16 autocast (CUDA only).

    Returns:
        tuple[Tensor, dict[str, Any]]: ``float32`` NLL per target, shape
        (target_end - target_start,), on the model's device; and the backbone's
        auxiliary outputs.
    """

    if tokens.dim() != 2 or tokens.shape[0] != 1:
        raise ValueError(f"tokens must have shape (1, seq_len), got {tuple(tokens.shape)}")
    seq_len = tokens.shape[1]
    if not 1 <= target_start < target_end <= seq_len:
        raise ValueError(
            f"need 1 <= target_start < target_end <= seq_len, got "
            f"{target_start}, {target_end}, {seq_len}"
        )
    use_autocast = autocast_bf16 and tokens.device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        backbone_output = model.encode(tokens)  # type: ignore[attr-defined]
        hidden = backbone_output.pred  # (1, seq_len, dim)
        logits = model.head(  # type: ignore[attr-defined]
            ModelOutput(pred=hidden[:, target_start - 1 : target_end - 1], aux=backbone_output.aux)
        )  # (1, num_targets, vocab)
    targets = tokens[0, target_start:target_end]
    nll = F.cross_entropy(logits[0].float(), targets, reduction="none")  # (num_targets,)
    return nll, backbone_output.aux


def sequence_nll(
    model: Model,
    tokens: Tensor,
    *,
    head_chunk: int = 4096,
    autocast_bf16: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    """Next-token NLL of every target position of a sequence, one backbone pass.

    Equivalent to concatenating :func:`token_nll` over consecutive slices of
    ``head_chunk`` targets, but the adapter + backbone run ONCE per sequence
    and only the unembedding is chunked, so a ``(seq_len, vocab)`` logits
    tensor is never materialised.

    Args:
        model (Model): A ``SequenceModel`` with ``encode`` and ``head``.
        tokens (Tensor): Token ids. Shape: (1, seq_len), ``seq_len >= 2``.
        head_chunk (int): Targets per unembedding call (bounds logits memory).
        autocast_bf16 (bool): Run the forward under bf16 autocast (CUDA only).

    Returns:
        tuple[Tensor, dict[str, Any]]: ``float32`` NLL per target position
        ``1 .. seq_len - 1`` (shape ``(seq_len - 1,)``, on the model's device)
        and the backbone's auxiliary outputs.
    """

    if tokens.dim() != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 2:
        raise ValueError(f"tokens must have shape (1, seq_len >= 2), got {tuple(tokens.shape)}")
    if head_chunk < 1:
        raise ValueError(f"head_chunk must be >= 1, got {head_chunk}")
    seq_len = tokens.shape[1]
    use_autocast = autocast_bf16 and tokens.device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        backbone_output = model.encode(tokens)  # type: ignore[attr-defined]
        hidden = backbone_output.pred  # (1, seq_len, dim)
        pieces: list[Tensor] = []
        for start in range(1, seq_len, head_chunk):
            end = min(start + head_chunk, seq_len)
            logits = model.head(  # type: ignore[attr-defined]
                ModelOutput(pred=hidden[:, start - 1 : end - 1], aux=backbone_output.aux)
            )  # (1, end - start, vocab)
            pieces.append(F.cross_entropy(logits[0].float(), tokens[0, start:end], reduction="none"))
    return torch.cat(pieces), backbone_output.aux


def moving_average(values: np.ndarray, window: int = 101) -> np.ndarray:
    """Centred moving average that ignores NaN entries.

    Args:
        values (np.ndarray): Series with NaN where no data exists. Shape: (n,).
        window (int): Odd window width; entries within ``window // 2`` of each
            side are averaged (edges use the available part of the window).

    Returns:
        np.ndarray: Smoothed series, NaN where the window holds no data.
            Shape: (n,).
    """

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    # 'same' convolution == centred window; the zero padding is harmless because
    # the count is convolved the same way.
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts
    out[counts == 0] = np.nan
    return out


def binned_mean(sums: np.ndarray, counts: np.ndarray, bin_size: int) -> list[dict[str, float | int]]:
    """Token-weighted mean loss per contiguous position bin.

    Args:
        sums (np.ndarray): Summed NLL per position. Shape: (n,).
        counts (np.ndarray): Token count per position. Shape: (n,).
        bin_size (int): Positions per bin.

    Returns:
        list[dict[str, float | int]]: One entry per bin with ``start``, ``end``,
        ``tokens`` and ``cross_entropy`` (NaN when the bin is empty).
    """

    out: list[dict[str, float | int]] = []
    for start in range(0, len(sums), bin_size):
        end = min(start + bin_size, len(sums))
        tokens = int(counts[start:end].sum())
        ce = float(sums[start:end].sum() / tokens) if tokens else float("nan")
        out.append({"start": start, "end": end, "tokens": tokens, "cross_entropy": ce})
    return out


class PerPositionLoss:
    """Accumulator of summed NLL and token counts per absolute target position.

    Attributes:
        sums (np.ndarray): Summed NLL per position. Shape: (max_positions,).
        counts (np.ndarray): Tokens seen per position. Shape: (max_positions,).
    """

    def __init__(self, max_positions: int) -> None:
        """Create an empty accumulator.

        Args:
            max_positions (int): Largest position index + 1 that will be added.
        """

        self.sums = np.zeros(max_positions, dtype=np.float64)
        self.counts = np.zeros(max_positions, dtype=np.int64)

    def add(self, nll: Tensor | np.ndarray, start: int) -> None:
        """Add the NLLs of consecutive positions ``start, start + 1, ...``.

        Args:
            nll (Tensor | np.ndarray): Per-token NLL. Shape: (n,).
            start (int): Absolute position of ``nll[0]``.
        """

        values = (
            nll.detach().float().cpu().numpy()
            if isinstance(nll, Tensor)
            else np.asarray(nll, dtype=np.float64)
        )
        end = start + len(values)
        if end > len(self.sums):
            raise ValueError(f"positions up to {end} exceed max_positions {len(self.sums)}")
        self.sums[start:end] += values
        self.counts[start:end] += 1

    @property
    def tokens(self) -> int:
        """Total tokens accumulated."""

        return int(self.counts.sum())

    def cross_entropy(self) -> float:
        """Token-weighted mean NLL over every position."""

        return float(self.sums.sum() / max(self.tokens, 1))

    def mean_by_position(self) -> np.ndarray:
        """Mean NLL per position, NaN where nothing was accumulated. Shape: (max_positions,)."""

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.sums / self.counts
        mean[self.counts == 0] = np.nan
        return mean

    def to_dict(self, *, ma_window: int = 101, bin_size: int = 1024) -> dict[str, Any]:
        """Summarise as JSON-serialisable curves.

        Args:
            ma_window (int): Moving-average window.
            bin_size (int): Bin width for the binned curve.

        Returns:
            dict[str, Any]: ``cross_entropy``, ``tokens``, ``mean_by_position``
            (list, NaN -> ``None``), ``moving_average`` (same), ``binned``.
        """

        mean = self.mean_by_position()
        smoothed = moving_average(mean, window=ma_window)

        def to_list(values: np.ndarray) -> list[float | None]:
            return [None if not np.isfinite(v) else float(v) for v in values]

        return {
            "cross_entropy": self.cross_entropy(),
            "tokens": self.tokens,
            "max_positions": len(self.sums),
            "ma_window": ma_window,
            "mean_by_position": to_list(mean),
            "moving_average": to_list(smoothed),
            "binned": binned_mean(self.sums, self.counts, bin_size),
        }
