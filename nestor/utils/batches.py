"""Helpers for unpacking supervised batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from torch import Tensor


def split_supervised_batch(batch: object) -> tuple[object, Tensor]:
    """Extract model inputs and targets from a supervised batch.

    Supported shapes:
    - ``(inputs, targets)`` tuples/lists
    - mappings with ``inputs``/``targets``
    - mappings with ``inputs``/``labels``
    - mappings with ``x``/``y``

    Args:
        batch (object): Batch object returned by a dataset or dataloader.

    Returns:
        tuple[object, Tensor]: Raw model inputs and supervision targets.

    Raises:
        TypeError: If the batch shape is unsupported.
    """

    if isinstance(batch, tuple) and len(batch) == 2:
        inputs, targets = batch
        if not isinstance(targets, Tensor):
            raise TypeError(
                "Expected tuple/list supervised targets to be a Tensor, got "
                f"{type(targets).__name__}."
            )
        return inputs, targets

    if isinstance(batch, list) and len(batch) == 2:
        inputs, targets = batch
        if not isinstance(targets, Tensor):
            raise TypeError(
                "Expected tuple/list supervised targets to be a Tensor, got "
                f"{type(targets).__name__}."
            )
        return inputs, targets

    if isinstance(batch, Mapping):
        mapping = cast(Mapping[str, object], batch)
        for input_key, target_key in (
            ("inputs", "targets"),
            ("inputs", "labels"),
            ("x", "y"),
        ):
            if input_key in mapping and target_key in mapping:
                targets = mapping[target_key]
                if not isinstance(targets, Tensor):
                    raise TypeError(
                        f"Expected mapping key '{target_key}' to be a Tensor, got "
                        f"{type(targets).__name__}."
                    )
                return mapping[input_key], targets

    raise TypeError(
        "Unsupported supervised batch format. Expected a length-2 tuple/list or "
        "a mapping with inputs/targets, inputs/labels, or x/y keys."
    )
