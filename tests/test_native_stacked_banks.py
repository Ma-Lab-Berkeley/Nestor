"""Stacked NATIVE expert banks: biased/non-gated support + layer wiring.

``NESTOR_STACKED_EXPERT_BANKS`` originally covered only the expert pools
(bias-free SwiGLU). The native shared / group-specific bank of
``RoutedTransformerLayer`` (MoE baselines, shared-bank column arms) re-stacked
its experts per micro-step the same way but carries up/down BIASES, so
:class:`~nestor.backbones.expert_bank_stack.StackedExpertBank` now supports
biased and non-gated banks, and the layer declares its combined bank as
``stacked_moe_bank``. Reference = the identical bank/layer without stacking
(separate per-expert parameters); the CPU ``segment_mm`` fallback exercises
the same dispatch bookkeeping as the CUDA grouped GEMM.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor

from nestor.backbones.expert_bank_stack import (
    StackedExpertBank,
    consolidate_stacked_expert_banks,
)
from nestor.backbones.grouped_mm import grouped_expert_rows
from nestor.backbones.nestor import RoutedTransformerLayer, RouterOutput
from nestor.scaling.blocks import GatedScaledMLP, ScaledMLP, get_scaling


def _swiglu_bank(
    seed: int,
    num_experts: int = 4,
    dim: int = 8,
    hidden: int = 12,
    bias: bool = True,
) -> list[GatedScaledMLP]:
    """A small SwiGLU expert bank with non-trivial biases under a fixed seed.

    Args:
        seed (int): Seed for the parameter draw.
        num_experts (int): Bank size.
        dim (int): Input/output width.
        hidden (int): Expert hidden width.
        bias (bool): Whether up/down carry biases (the gate never does).

    Returns:
        list[GatedScaledMLP]: The bank, in fp32, biases perturbed away from 0.
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


class StackedBiasedBankTest:
    """Owner-level checks: biased SwiGLU and non-gated banks."""

    def test_biased_bank_layout_views_and_tags(self) -> None:
        experts = _swiglu_bank(seed=0, num_experts=3, dim=8, hidden=6)
        reference = [
            (e.layers[0].bias.clone(), e.layers[1].bias.clone())
            for e in experts
        ]
        bank = StackedExpertBank(experts)
        assert tuple(bank.b_up.shape) == (3, 6)
        assert tuple(bank.b_down.shape) == (3, 8)
        for i, (b1, b2) in enumerate(reference):
            assert torch.equal(bank.b_up[i], b1)
            assert torch.equal(bank.b_down[i], b2)
            # Bias attributes are now views, not parameters.
            assert "bias" not in experts[i].layers[0]._parameters
            assert torch.equal(experts[i].layers[0].bias, b1)
        # A [E, hidden] bias bank must carry the per-expert bias tag (AdamW),
        # never a matrix (Muon) tag.
        tag = get_scaling(bank.b_up)
        assert tag is not None and tag.optimizer == "adamw"
        # Gradients through an expert's forward land on the bias owners.
        experts[2](torch.randn(4, 8)).sum().backward()
        assert bank.b_down.grad is not None
        assert bank.b_down.grad[2].abs().sum() > 0
        assert bank.b_down.grad[:2].abs().sum() == 0

    def test_non_gated_bank(self) -> None:
        torch.manual_seed(1)
        experts = [ScaledMLP(8, [6], 8, activation="gelu") for _ in range(3)]
        bank = StackedExpertBank(experts)
        assert bank.w_gate is None and not bank.gated
        w_up, w_gate, w_down = bank.cast(torch.float32)
        assert w_gate is None
        assert tuple(w_up.shape) == (3, 8, 6)
        assert tuple(w_down.shape) == (3, 6, 8)

    def test_grouped_rows_stacked_matches_reference_with_biases(self) -> None:
        bank_ref = _swiglu_bank(seed=2)
        bank_new = copy.deepcopy(bank_ref)
        stacked = StackedExpertBank(bank_new)
        torch.manual_seed(3)
        x = torch.randn(10, 8)
        expert_ids = torch.randint(0, len(bank_ref), (10, 2))
        x_ref = x.clone().requires_grad_(True)
        x_new = x.clone().requires_grad_(True)
        out_ref = grouped_expert_rows(x_ref, bank_ref, expert_ids)
        out_new = grouped_expert_rows(
            x_new, bank_new, expert_ids, stacked=stacked
        )
        torch.testing.assert_close(out_new, out_ref, rtol=1e-6, atol=1e-6)
        torch.manual_seed(4)
        weights = torch.randn_like(out_ref)
        (out_ref * weights).sum().backward()
        (out_new * weights).sum().backward()
        torch.testing.assert_close(x_new.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
        up_ref = torch.stack([e.layers[0].weight.grad.t() for e in bank_ref])
        b_up_ref = torch.stack([e.layers[0].bias.grad for e in bank_ref])
        b_down_ref = torch.stack([e.layers[1].bias.grad for e in bank_ref])
        torch.testing.assert_close(stacked.w_up.grad, up_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(stacked.b_up.grad, b_up_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            stacked.b_down.grad, b_down_ref, rtol=1e-5, atol=1e-6
        )


def _build_layer(
    seed: int, *, num_shared: int = 8, num_specific: int = 0, num_groups: int = 1
) -> RoutedTransformerLayer:
    """A small native-bank layer (no expert pool) under a fixed seed.

    Args:
        seed (int): Seed for the parameter draw.
        num_shared (int): Shared experts in the bank.
        num_specific (int): Group-specific experts per group.
        num_groups (int): Attention groups.

    Returns:
        RoutedTransformerLayer: The layer, unstacked (``stacked_moe_bank`` is
        declared but None).
    """

    torch.manual_seed(seed)
    return RoutedTransformerLayer(
        dim=16,
        num_experts_per_group=num_shared + num_specific,
        num_shared_experts=num_shared,
        num_experts_per_group_per_token=2,
        num_groups=num_groups,
        num_heads=2,
        head_dim=8,
        # 16 (not 12): the CUDA grouped-GEMM kernel requires every operand
        # stride to be a 16-BYTE multiple, i.e. widths % 8 == 0 in bf16 --
        # the first run failed on hidden 12 (24-byte rows) in the REFERENCE call
        # already. Production native banks are all 8-aligned.
        mlp_hidden_dims=[16],
        mlp_activation="swiglu",
        router_score_mlp_hidden_dims=[],
        router_score_mlp_activation="gelu",
        router_noise_mlp_hidden_dims=[],
        router_noise_mlp_activation="gelu",
        depth_scale=0.05,
    )


def _router_outputs(
    layer: RoutedTransformerLayer, seed: int, num_groups: int
) -> tuple[Tensor, RouterOutput, RouterOutput]:
    """Input plus fabricated group (top-1) and real expert router outputs."""

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 5, 16, generator=gen)
    valid = torch.ones(2, 5, dtype=torch.bool)
    scores = torch.rand(2, 5, num_groups, generator=gen)
    top1 = scores.argmax(dim=-1, keepdim=True)  # [2, 5, 1]
    gates = torch.zeros(2, 5, num_groups).scatter(-1, top1, 1.0)
    active = gates.bool()
    group_out = RouterOutput(
        scores=scores,
        softmax_probs=gates,
        gates=gates,
        topk_indices=top1,
        active_mask=active,
        valid_token_mask=valid,
    )
    layer.eval()  # expert router: no noise
    # Detached: the same routing feeds BOTH the reference and the stacked
    # layer, and each runs its own backward.
    expert_out = RouterOutput(
        *(t.detach() for t in layer.expert_router(x, valid))
    )
    return x, group_out, expert_out


class RoutedLayerStackedBankTest:
    """The layer's declared native bank: consolidation, dispatch, detection."""

    def test_declared_then_consolidated(self) -> None:
        layer = _build_layer(seed=0)
        assert layer.stacked_moe_bank is None  # declared, not stacked
        n_ref = sum(p.numel() for p in layer.parameters())
        assert consolidate_stacked_expert_banks(layer, force=True) == 1
        assert isinstance(layer.stacked_moe_bank, StackedExpertBank)
        assert sum(p.numel() for p in layer.parameters()) == n_ref
        keys = set(layer.state_dict().keys())
        assert "stacked_moe_bank.w_up" in keys
        assert "stacked_moe_bank.b_up" in keys
        assert not any(k.startswith("shared_experts.") for k in keys)
        assert consolidate_stacked_expert_banks(layer, force=True) == 0

    def test_shared_bank_grouped_residual_matches(self) -> None:
        reference = _build_layer(seed=1)
        stacked = copy.deepcopy(reference)
        consolidate_stacked_expert_banks(stacked, force=True)
        x, group_out, expert_out = _router_outputs(reference, seed=2, num_groups=1)
        stacked.eval()
        out_ref = reference._grouped_expert_residual(x, group_out, expert_out)
        out_new = stacked._grouped_expert_residual(x, group_out, expert_out)
        torch.testing.assert_close(out_new, out_ref, rtol=1e-6, atol=1e-6)
        out_ref.pow(2).sum().backward()
        out_new.pow(2).sum().backward()
        up_ref = torch.stack(
            [e.layers[0].weight.grad.t() for e in reference.shared_experts]
        )
        b_up_ref = torch.stack(
            [e.layers[0].bias.grad for e in reference.shared_experts]
        )
        torch.testing.assert_close(
            stacked.stacked_moe_bank.w_up.grad, up_ref, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            stacked.stacked_moe_bank.b_up.grad, b_up_ref, rtol=1e-5, atol=1e-6
        )

    def test_group_specific_bank_order_matches(self) -> None:
        """The stacked owner's expert order == the grouped path's bank index."""

        reference = _build_layer(seed=3, num_shared=2, num_specific=3, num_groups=2)
        stacked = copy.deepcopy(reference)
        consolidate_stacked_expert_banks(stacked, force=True)
        # Bank order [shared | group 0 | group 1]:
        want = [
            *reference.shared_experts,
            *(e for grp in reference.group_specific_experts for e in grp),
        ]
        for i, expert in enumerate(want):
            assert torch.equal(
                stacked.stacked_moe_bank.w_up[i].t(), expert.layers[0].weight
            )
        x, group_out, expert_out = _router_outputs(reference, seed=4, num_groups=2)
        stacked.eval()
        out_ref = reference._grouped_expert_residual(x, group_out, expert_out)
        out_new = stacked._grouped_expert_residual(x, group_out, expert_out)
        torch.testing.assert_close(out_new, out_ref, rtol=1e-6, atol=1e-6)
        # The stacked path must also agree with the sparse per-expert loop.
        out_sparse = reference._compute_expert_residual(x, group_out, expert_out)
        torch.testing.assert_close(out_new, out_sparse, rtol=1e-5, atol=1e-6)

    def test_checkpoint_marker_and_cross_load(self) -> None:
        from nestor.eval.checkpoint import _STACKED_BANK_MARKERS

        stacked = copy.deepcopy(_build_layer(seed=5))
        consolidate_stacked_expert_banks(stacked, force=True)
        state = stacked.state_dict()
        assert any(m in k for k in state for m in _STACKED_BANK_MARKERS)
        # An eval-side rebuild consolidates on detection, then loads cleanly.
        fresh = _build_layer(seed=6)
        consolidate_stacked_expert_banks(fresh, force=True)
        missing, unexpected = fresh.load_state_dict(state, strict=False)
        assert not missing and not unexpected

    def test_views_survive_apply(self) -> None:
        stacked = copy.deepcopy(_build_layer(seed=8))
        consolidate_stacked_expert_banks(stacked, force=True)
        moved = stacked.to(torch.float64)
        expert = moved.shared_experts[3]
        assert expert.layers[0].weight.dtype == torch.float64
        assert torch.equal(
            expert.layers[0].weight, moved.stacked_moe_bank.w_up[3].t()
        )
        assert torch.equal(expert.layers[0].bias, moved.stacked_moe_bank.b_up[3])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM path is CUDA-only"
)
class NativeStackedBankCudaTest:
    """On CUDA the grouped-GEMM path with the stacked native bank matches."""

    def test_grouped_residual_matches_on_cuda(self) -> None:
        # Router outputs are computed on CPU (before the move: the router
        # weights and the input must share a device), then everything moves.
        reference = _build_layer(seed=11)
        stacked = copy.deepcopy(reference)
        consolidate_stacked_expert_banks(stacked, force=True)
        x, group_out, expert_out = _router_outputs(reference, seed=12, num_groups=1)
        reference = reference.cuda()
        stacked = stacked.cuda()
        x = x.cuda()
        group_out = RouterOutput(*(t.cuda() for t in group_out))
        expert_out = RouterOutput(*(t.cuda() for t in expert_out))
        stacked.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_ref = reference._grouped_expert_residual(x, group_out, expert_out)
            out_new = stacked._grouped_expert_residual(x, group_out, expert_out)
        torch.testing.assert_close(
            out_new.float(), out_ref.float(), rtol=2e-2, atol=2e-2
        )
        out_ref.float().pow(2).sum().backward()
        out_new.float().pow(2).sum().backward()
        up_ref = torch.stack(
            [e.layers[0].weight.grad.t() for e in reference.shared_experts]
        )
        b_down_ref = torch.stack(
            [e.layers[1].bias.grad for e in reference.shared_experts]
        )
        torch.testing.assert_close(
            stacked.stacked_moe_bank.w_up.grad, up_ref, rtol=5e-2, atol=5e-2
        )
        torch.testing.assert_close(
            stacked.stacked_moe_bank.b_down.grad, b_down_ref, rtol=5e-2, atol=5e-2
        )
