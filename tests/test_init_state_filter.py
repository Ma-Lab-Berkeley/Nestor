"""Pins for ``nestor.train.checkpoint.filter_init_state`` -- the non-strict init load.

``DCLM_INIT_CHECKPOINT`` loads pretrained weights into a freshly built model
with a fresh optimizer (the length-extension path). The filter keeps
name-and-shape matches and drops the rest, which is right for
length-dependent buffers and wrong for parameters: a stacked-bank checkpoint
(``NESTOR_STACKED_EXPERT_BANKS=1``) fed to an unstacked model names its
expert weights differently, and the old filter silently trained random
experts. These tests pin the refusal and the opt-out.
"""

from __future__ import annotations

import pytest
import torch

from nestor.train.checkpoint import filter_init_state


def _state(**shapes: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """A state dict of zeros with the given shapes."""
    return {name: torch.zeros(*shape) for name, shape in shapes.items()}


class FilterInitStateTest:
    """Exact-layout files load whole; buffers may drop; uncovered parameters refuse."""

    def test_matching_layout_loads_every_key(self) -> None:
        ckpt = _state(**{"a.weight": (4, 4), "a.bias": (4,), "rope.cache": (16, 4)})
        current = _state(**{"a.weight": (4, 4), "a.bias": (4,), "rope.cache": (16, 4)})
        filtered, skipped = filter_init_state(ckpt, current, {"a.weight", "a.bias"})
        assert set(filtered) == set(ckpt)
        assert skipped == set()

    def test_length_dependent_buffer_is_dropped_but_parameters_load(self) -> None:
        # The 4k checkpoint's RoPE cache vs a model built at 32k: a buffer
        # shape mismatch, not a parameter -- the length-extension case that must keep working.
        ckpt = _state(**{"a.weight": (4, 4), "rope.cache": (4096, 4)})
        current = _state(**{"a.weight": (4, 4), "rope.cache": (32768, 4)})
        filtered, skipped = filter_init_state(ckpt, current, {"a.weight"})
        assert set(filtered) == {"a.weight"}
        assert skipped == {"rope.cache"}

    def test_stacked_file_into_unstacked_model_is_refused(self) -> None:
        ckpt = _state(**{"a.weight": (4, 4), "bank.stacked_moe_bank.w_up": (3, 4, 8)})
        current = _state(
            **{"a.weight": (4, 4), "bank.experts.0.w_up.weight": (8, 4), "bank.experts.1.w_up.weight": (8, 4)}
        )
        params = {"a.weight", "bank.experts.0.w_up.weight", "bank.experts.1.w_up.weight"}
        with pytest.raises(RuntimeError, match="uninitialised"):
            filter_init_state(ckpt, current, params)

    def test_allow_partial_opts_out_and_reports_the_skips(self) -> None:
        ckpt = _state(**{"a.weight": (4, 4), "bank.stacked_moe_bank.w_up": (3, 4, 8)})
        current = _state(**{"a.weight": (4, 4), "bank.experts.0.w_up.weight": (8, 4)})
        filtered, skipped = filter_init_state(
            ckpt, current, {"a.weight", "bank.experts.0.w_up.weight"}, allow_partial=True
        )
        assert set(filtered) == {"a.weight"}
        assert skipped == {"bank.stacked_moe_bank.w_up"}

    def test_parameter_shape_mismatch_counts_as_uncovered(self) -> None:
        ckpt = _state(**{"a.weight": (4, 4)})
        current = _state(**{"a.weight": (8, 4)})
        with pytest.raises(RuntimeError, match="a.weight"):
            filter_init_state(ckpt, current, {"a.weight"})
