"""Equivalence tests for the fused up+gate grouped GEMM.

``grouped_expert_rows(..., fused_gate=True)`` runs a SwiGLU bank's up and gate
projections as ONE grouped GEMM over the concatenated ``[E, d_in, 2*hidden]``
weights instead of two GEMMs that each re-read the dispatched rows. Every
output column is the same dot product either way, so the unfused call (the
reference implementation throughout) must agree to accumulation noise, in the
forward and in every parameter gradient. ``NESTOR_GROUPED_MM_FUSED_GATE=1``
turns it on; the expert-bearing modules read the flag at construction.

CPU-runnable: ``segment_mm`` falls back to a per-segment loop without
``torch._grouped_mm``, which exercises exactly the same concatenate/split
bookkeeping.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from nestor.backbones.grouped_mm import fused_gate_requested, grouped_expert_rows
from nestor.scaling.blocks import GatedScaledMLP, ScaledMLP


def _swiglu_bank(
    seed: int, num_experts: int = 3, dim: int = 8, hidden: int = 12, bias: bool = True
) -> list[GatedScaledMLP]:
    """Build a small SwiGLU expert bank under a fixed seed.

    Args:
        seed (int): Seed for the parameter draw.
        num_experts (int): Bank size.
        dim (int): Input/output width.
        hidden (int): Expert hidden width.
        bias (bool): Whether up/down carry biases (the gate never does).

    Returns:
        list[GatedScaledMLP]: The bank, in fp32.
    """

    torch.manual_seed(seed)
    bank = [
        GatedScaledMLP(dim, [hidden], dim, activation="silu", bias=bias)
        for _ in range(num_experts)
    ]
    for expert in bank:
        for param in expert.parameters():
            with torch.no_grad():
                param.add_(0.1 * torch.randn_like(param))
    return bank


def _clone_bank(bank: list[GatedScaledMLP]) -> list[GatedScaledMLP]:
    """Deep-copy a bank so two runs own independent parameters/gradients."""

    import copy

    return copy.deepcopy(bank)


class GroupedExpertRowsFusedGateTest:
    """Fused up+gate grouped GEMM == two grouped GEMMs."""

    def _inputs(self, seed: int, num_experts: int = 3) -> tuple[Tensor, Tensor]:
        """Random token features and (token, slot) expert assignments."""

        torch.manual_seed(seed)
        x = torch.randn(10, 8)
        expert_ids = torch.randint(0, num_experts, (10, 2))
        return x, expert_ids

    def test_forward_matches_unfused(self) -> None:
        """Same rows out of the fused and the unfused dispatch."""

        bank = _swiglu_bank(seed=1)
        x, expert_ids = self._inputs(seed=2)
        with torch.no_grad():
            unfused = grouped_expert_rows(x, bank, expert_ids, fused_gate=False)
            fused = grouped_expert_rows(x, bank, expert_ids, fused_gate=True)
        torch.testing.assert_close(fused, unfused, rtol=1e-6, atol=1e-6)

    def test_forward_matches_without_bias(self) -> None:
        """Bias-free banks take the no-bias branch on both paths."""

        bank = _swiglu_bank(seed=3, bias=False)
        x, expert_ids = self._inputs(seed=4)
        with torch.no_grad():
            unfused = grouped_expert_rows(x, bank, expert_ids, fused_gate=False)
            fused = grouped_expert_rows(x, bank, expert_ids, fused_gate=True)
        torch.testing.assert_close(fused, unfused, rtol=1e-6, atol=1e-6)

    def test_gradients_match_unfused(self) -> None:
        """Every expert parameter and the input get the same gradient."""

        bank_ref = _swiglu_bank(seed=5)
        bank_fused = _clone_bank(bank_ref)
        x, expert_ids = self._inputs(seed=6)
        x_ref = x.clone().requires_grad_(True)
        x_fused = x.clone().requires_grad_(True)

        torch.manual_seed(7)
        weights = torch.randn(10 * 2, 8)
        (grouped_expert_rows(x_ref, bank_ref, expert_ids, fused_gate=False)
         * weights).sum().backward()
        (grouped_expert_rows(x_fused, bank_fused, expert_ids, fused_gate=True)
         * weights).sum().backward()

        assert x_ref.grad is not None and x_fused.grad is not None
        torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
        for expert_ref, expert_fused in zip(bank_ref, bank_fused):
            for (name, p_ref), (_, p_fused) in zip(
                expert_ref.named_parameters(), expert_fused.named_parameters()
            ):
                assert p_ref.grad is not None, name
                assert p_fused.grad is not None, name
                torch.testing.assert_close(
                    p_fused.grad, p_ref.grad, rtol=1e-5, atol=1e-6, msg=name
                )

    def test_non_gated_bank_ignores_flag(self) -> None:
        """A 2-layer bank has no gate to fuse: the flag must be a no-op."""

        torch.manual_seed(8)
        bank = [ScaledMLP(8, [12], 8, activation="gelu") for _ in range(3)]
        x, expert_ids = self._inputs(seed=9)
        with torch.no_grad():
            unfused = grouped_expert_rows(x, bank, expert_ids, fused_gate=False)
            fused = grouped_expert_rows(x, bank, expert_ids, fused_gate=True)
        torch.testing.assert_close(fused, unfused, rtol=0.0, atol=0.0)

    def test_flag_reader_follows_env(self) -> None:
        """``fused_gate_requested`` is exactly the ``=1`` test of the env flag."""

        key = "NESTOR_GROUPED_MM_FUSED_GATE"
        previous = os.environ.get(key)
        try:
            os.environ.pop(key, None)
            assert fused_gate_requested() is False
            os.environ[key] = "0"
            assert fused_gate_requested() is False
            os.environ[key] = "1"
            assert fused_gate_requested() is True
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
