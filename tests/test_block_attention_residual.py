"""Reference and integration tests for writeup-style block AttnRes."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from nestor.backbones import BlockAttentionResidual, Nestor, NestorConfig


def _reference_block_attention_residual(
    module: BlockAttentionResidual,
    sources: list[Tensor],
) -> Tensor:
    """Direct transcription of the writeup's per-token depth attention."""

    keys = [
        F.linear(source, projection.weight)
        for source, projection in zip(sources, module.key_projections)
    ]
    values = [
        F.linear(source, projection.weight)
        for source, projection in zip(sources, module.value_projections)
    ]
    logits = torch.stack(
        [F.linear(key, module.query.weight).squeeze(-1) for key in keys],
        dim=0,
    )
    weights = logits.float().softmax(dim=0).to(dtype=values[0].dtype)
    return (torch.stack(values, dim=0) * weights.unsqueeze(-1)).sum(dim=0)


def _create_block_attn_res_config(*, causal: bool = False) -> NestorConfig:
    """Create a small heterogeneous-width Nestor block-AttnRes config."""

    return NestorConfig(
        input_dim=16,
        num_blocks=3,
        feature_dims=[24, 32, 40],
        pooling_strides=[1, 1, 1],
        causal=causal,
        block_num_layers=[1, 1, 1],
        num_groups=[1, 1, 1],
        num_groups_per_token=[1, 1, 1],
        block_num_heads=[1, 1, 1],
        block_head_dims=[8, 8, 8],
        block_mlp_hidden_dims=[[24], [32], [40]],
        router_num_experts_per_group=[1, 1, 1],
        router_num_shared_experts=[1, 1, 1],
        router_num_experts_per_group_per_token=[1, 1, 1],
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="attn_res",
    )


class BlockAttentionResidualReferenceTest:
    """Numerical checks against the writeup equation."""

    def test_forward_and_gradients_match_reference(self) -> None:
        """Production forward and backward should match a direct transcription."""

        torch.manual_seed(0)
        module = BlockAttentionResidual([5, 7, 9], target_width=6)
        with torch.no_grad():
            module.query.weight.normal_()
        reference_module = deepcopy(module)
        sources = [
            torch.randn(2, 4, width, requires_grad=True)
            for width in module.source_widths
        ]
        reference_sources = [
            source.detach().clone().requires_grad_() for source in sources
        ]

        output = module(sources)
        reference = _reference_block_attention_residual(
            reference_module, reference_sources
        )
        torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

        parameters = [*module.parameters(), *sources]
        reference_parameters = [*reference_module.parameters(), *reference_sources]
        gradients = torch.autograd.grad(output.square().sum(), parameters)
        reference_gradients = torch.autograd.grad(
            reference.square().sum(), reference_parameters
        )
        for gradient, reference_gradient in zip(gradients, reference_gradients):
            torch.testing.assert_close(
                gradient, reference_gradient, rtol=1e-5, atol=1e-6
            )

    def test_zero_query_is_uniform_projected_value_mix(self) -> None:
        """The zero-initialized query should assign every source equal weight."""

        torch.manual_seed(1)
        module = BlockAttentionResidual([3, 5], target_width=4)
        sources = [torch.randn(2, 6, 3), torch.randn(2, 6, 5)]
        expected = torch.stack(
            [
                projection(source)
                for source, projection in zip(sources, module.value_projections)
            ],
            dim=0,
        ).mean(dim=0)

        torch.testing.assert_close(module(sources), expected)

    def test_rejects_mismatched_source_shape(self) -> None:
        """Source widths and leading token shapes are part of the API contract."""

        module = BlockAttentionResidual([3, 5], target_width=4)
        with pytest.raises(ValueError, match="Source 1 must have shape"):
            module([torch.randn(2, 6, 3), torch.randn(2, 7, 5)])


class NestorBlockAttentionResidualTest:
    """End-to-end checks for Nestor block-boundary AttnRes wiring."""

    def test_heterogeneous_blocks_forward_backward(self) -> None:
        """Every retained value path should connect directly to the final loss."""

        torch.manual_seed(2)
        config = _create_block_attn_res_config()
        model = Nestor(config)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        output = model(x, pos, attn_mask)
        assert output.pred.shape == (2, 5, config.output_dim)
        output.pred.square().mean().backward()

        assert [module.source_widths for module in model.block_attention_residuals] == [
            (16, 16),
            (16, 16, 24),
            (16, 16, 24, 32),
        ]
        for module in model.block_attention_residuals:
            assert module.query.weight.grad is not None
            for value_projection in module.value_projections:
                assert value_projection.weight.grad is not None
                assert torch.count_nonzero(value_projection.weight.grad) > 0

    def test_causal_cached_forward_matches_full_forward(self) -> None:
        """Pointwise depth mixing must not change causal KV-cache semantics."""

        torch.manual_seed(3)
        config = _create_block_attn_res_config(causal=True)
        model = Nestor(config).eval()
        with torch.no_grad():
            for module in model.block_attention_residuals:
                module.query.weight.normal_()

        x = torch.randn(2, 4, config.input_dim)
        pos = torch.arange(4).expand(2, -1)
        attn_mask = torch.ones(2, 4, 4, dtype=torch.bool)
        with torch.no_grad():
            full = model(x, pos, attn_mask).pred
            cache = model.init_kv_cache(batch_size=2)
            steps = []
            for token_idx in range(x.shape[1]):
                step, cache = model.forward_step(
                    x[:, token_idx], pos[:, token_idx], cache
                )
                steps.append(step.pred)
            cached = torch.cat(steps, dim=1)

        torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-5)

    def test_rejects_token_pooling(self) -> None:
        """Per-token cross-depth mixing requires aligned token positions."""

        config = _create_block_attn_res_config()
        config.pooling_strides = [2, 1, 1]
        with pytest.raises(ValueError, match="requires constant token count"):
            Nestor(config)
