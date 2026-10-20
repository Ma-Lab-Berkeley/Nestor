"""Sanity check tests for Nestor model forward pass."""

import torch

from nestor.backbones import (
    ModelOutput,
    Nestor,
    NestorBlock,
    NestorConfig,
)
from nestor.backbones.nestor import RoutedTransformerLayer, RouterOutput
from nestor.constants import PADDING_TOKEN_POS


def create_default_transformer_config() -> NestorConfig:
    """
    Create a default transformer configuration (single-expert dense special case).

    This exercises the dense-transformer special case of hierarchical Nestor.

    Returns:
        NestorConfig: A minimal config with 1 Nestor layer, no routing.
    """
    feature_dim = 32
    return NestorConfig(
        input_dim=feature_dim,
        num_blocks=1,
        feature_dims=[feature_dim],
        router_num_experts_per_group=[1],
        router_num_shared_experts=[1],
        pooling_strides=[1],
        num_groups=[1],
        num_groups_per_token=[1],
        block_num_layers=2,
        block_num_heads=2,
        block_head_dims=8,
        block_mlp_hidden_dims=[32],
        block_mlp_activations="gelu",
        router_num_experts_per_group_per_token=1,
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activations="gelu",
    )


def create_minimal_nestor_config() -> NestorConfig:
    """
    Create a minimal Nestor configuration for testing.

    Uses small dimensions to allow fast execution while exercising all
    model components: routing and pooled transformer stacks.

    Returns:
        NestorConfig: A minimal config with 2 Nestor layers.
    """
    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[48, 64],
        router_num_experts_per_group=[4, 1],
        router_num_shared_experts=[1, 1],
        pooling_strides=[2, 1],
        num_groups=[2, 1],
        num_groups_per_token=[2, 1],
        block_num_layers=[2, 1],
        block_num_heads=[2, 4],
        block_head_dims=[8, 8],
        block_mlp_hidden_dims=[[64], [128]],
        block_mlp_activations=["gelu", "gelu"],
        router_num_experts_per_group_per_token=[2, 1],
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activations="gelu",
    )


def create_causal_cached_nestor_config() -> NestorConfig:
    """Create a small causal Nestor config that exercises hierarchical caching."""
    return NestorConfig(
        input_dim=24,
        num_blocks=2,
        feature_dims=[32, 48],
        router_num_experts_per_group=[2, 1],
        router_num_shared_experts=[1, 1],
        pooling_strides=[2, 1],
        num_groups=[2, 1],
        num_groups_per_token=[1, 1],
        causal=True,
        block_num_layers=[2, 1],
        block_num_heads=[2, 3],
        block_head_dims=[8, 8],
        block_mlp_hidden_dims=[[48], [96]],
        block_mlp_activations=["gelu", "gelu"],
        router_num_experts_per_group_per_token=[1, 1],
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activations="gelu",
    )


class NestorForwardPassTest:
    """Test class for Nestor model forward pass sanity checks."""

    def test_forward_pass_runs_without_error(self) -> None:
        """
        Test that a forward pass through Nestor completes without errors.

        Creates a minimal Nestor model and verifies that:
        1. The model can be instantiated from config
        2. A forward pass completes without raising exceptions
        3. The output has the expected type (ModelOutput)
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        batch_size = 2
        seq_len = 4
        input_dim = config.input_dim

        # Create input tensors
        x = torch.randn(batch_size, seq_len, input_dim)
        pos = (
            torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        )  # (batch_size, seq_len)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        # Forward pass should complete without error
        output = model(x, pos, attn_mask)

        assert isinstance(output, ModelOutput)

    def test_forward_pass_output_shape(self) -> None:
        """
        Test that the forward pass produces outputs with correct shapes.

        Verifies:
        1. pred tensor has shape (batch_size, output_dim)
        2. aux dict contains 'blocks' key with per-block outputs
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        batch_size = 4
        seq_len = 8
        input_dim = config.input_dim
        x = torch.randn(batch_size, seq_len, input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)

        # Check pred shape
        assert output.pred.shape == (batch_size, 4, config.output_dim), (
            f"Expected pred shape ({batch_size}, 4, {config.output_dim}), got {output.pred.shape}"
        )

        # Check aux structure
        assert "blocks" in output.aux
        assert len(output.aux["blocks"]) == config.num_blocks
        assert "group_router" in output.aux["blocks"][0]
        assert "layers" in output.aux["blocks"][0]

    def test_forward_pass_with_varying_batch_sizes(self) -> None:
        """
        Test forward pass with different batch sizes.

        Ensures the model handles varying batch sizes correctly.
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        seq_len = 4
        input_dim = config.input_dim

        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, seq_len, input_dim)
            pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
            attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

            output = model(x, pos, attn_mask)

            assert output.pred.shape[0] == batch_size, (
                f"Expected batch size {batch_size}, got {output.pred.shape[0]}"
            )

    def test_forward_pass_with_varying_seq_lengths(self) -> None:
        """
        Test forward pass with different sequence lengths.

        Ensures the model handles varying sequence lengths correctly.
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        batch_size = 2
        input_dim = config.input_dim

        for seq_len in [2, 4, 16]:
            x = torch.randn(batch_size, seq_len, input_dim)
            pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
            attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

            output = model(x, pos, attn_mask)

            expected_seq_len = (seq_len + 1) // 2
            assert output.pred.shape == (
                batch_size,
                expected_seq_len,
                config.output_dim,
            )

    def test_forward_pass_eval_mode(self) -> None:
        """
        Test that forward pass works in eval mode.

        Ensures model behaves correctly with dropout/batchnorm disabled.
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)
        model.eval()

        batch_size = 2
        seq_len = 4
        input_dim = config.input_dim

        x = torch.randn(batch_size, seq_len, input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        with torch.no_grad():
            output = model(x, pos, attn_mask)

        assert isinstance(output, ModelOutput)
        assert output.pred.shape == (batch_size, 2, config.output_dim)

    def test_local_pooling_uses_fixed_stride(self) -> None:
        """Local pooling should reduce sequence length by the configured stride."""
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, config.input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)

        assert output.pred.shape == (batch_size, 3, config.output_dim)
        assert torch.equal(output.aux["positions"][0], torch.tensor([0, 2, 4]))

    def test_forward_pass_gradient_flow(self) -> None:
        """
        Test that gradients flow through the model correctly.

        Verifies that backpropagation works and gradients are non-zero.
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        batch_size = 2
        seq_len = 4
        input_dim = config.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)

        # Compute simple loss and backpropagate
        loss = output.pred.sum()
        loss.backward()

        # Check that input received gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that model parameters received gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        assert has_gradients, "No gradients found in model parameters"

    def test_config_roundtrip(self) -> None:
        """
        Test that model config can be retrieved after instantiation.

        Verifies to_config() returns a valid config.
        """
        config = create_minimal_nestor_config()
        model = Nestor.from_config(config)

        retrieved_config = model.to_config()

        assert isinstance(retrieved_config, NestorConfig)
        assert retrieved_config.input_dim == config.input_dim
        assert retrieved_config.num_blocks == config.num_blocks

    def test_detached_router_parameters_update(self) -> None:
        """Detached router state should update after a simulated optimizer step."""
        config = create_minimal_nestor_config()
        config.router_loss_free_balancing = [True, False]
        config.router_bias_update_rate = [0.5, 0.0]
        model = Nestor.from_config(config)
        first_block: NestorBlock = model.blocks[0]
        first_layer = first_block.layers[0]

        group_bias_before = first_block.group_router.choice_bias.clone()
        expert_bias_before = first_layer.expert_router.choice_bias.clone()
        model.update_detached_parameters(
            {
                "blocks": [
                    {
                        "group_router": {
                            "topk_indices": torch.zeros((2, 4, 2), dtype=torch.long),
                            "valid_token_mask": torch.ones((2, 4), dtype=torch.bool),
                        },
                        "layers": [
                            {
                                "expert_router": {
                                    "topk_indices": torch.zeros(
                                        (2, 4, 2), dtype=torch.long
                                    ),
                                    "valid_token_mask": torch.ones(
                                        (2, 4), dtype=torch.bool
                                    ),
                                }
                            }
                        ],
                    }
                ]
            }
        )
        group_bias_after = first_block.group_router.choice_bias
        expert_bias_after = first_layer.expert_router.choice_bias

        assert not torch.equal(group_bias_before, group_bias_after)
        assert not torch.equal(expert_bias_before, expert_bias_after)


class NestorKVCacheTest:
    """Tests for causal incremental inference with hierarchical KV caches."""

    def test_cached_forward_matches_full_forward_after_finalize(self) -> None:
        """Incremental cached decoding should match full causal forward output."""
        torch.manual_seed(0)
        config = create_causal_cached_nestor_config()
        model = Nestor.from_config(config)
        model.eval()

        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, config.input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        with torch.no_grad():
            full_output = model(x, pos, attn_mask)
            cache = model.init_kv_cache(
                batch_size=batch_size, device=x.device, dtype=x.dtype
            )
            emitted_tokens: list[torch.Tensor] = []
            emitted_positions: list[torch.Tensor] = []
            for token_idx in range(seq_len):
                step_output, cache = model.forward_step(
                    x[:, token_idx, :], pos[:, token_idx], cache
                )
                emitted_tokens.append(step_output.pred)
                emitted_positions.append(step_output.aux["positions"])
            cached_output, _ = model.finalize_kv_cache(cache)

        streamed_tokens_with_padding = torch.cat(
            emitted_tokens + [cached_output.pred], dim=1
        )
        streamed_positions_with_padding = torch.cat(
            emitted_positions + [cached_output.aux["positions"]],
            dim=1,
        )
        valid_stream_mask = streamed_positions_with_padding[0] >= 0
        streamed_tokens = streamed_tokens_with_padding[:, valid_stream_mask, :]
        streamed_positions = streamed_positions_with_padding[:, valid_stream_mask]

        assert torch.allclose(full_output.pred, streamed_tokens, atol=1e-5)
        assert torch.equal(full_output.aux["positions"], streamed_positions)

    def test_cached_forward_reports_when_final_output_updates(self) -> None:
        """Local-pooling caches should only emit final tokens when a pool closes."""
        config = create_causal_cached_nestor_config()
        model = Nestor.from_config(config)
        model.eval()

        batch_size = 1
        seq_len = 4
        x = torch.randn(batch_size, seq_len, config.input_dim)
        pos = torch.arange(seq_len).unsqueeze(0)

        with torch.no_grad():
            cache = model.init_kv_cache(
                batch_size=batch_size, device=x.device, dtype=x.dtype
            )
            update_steps: list[bool] = []
            for token_idx in range(seq_len):
                step_output, cache = model.forward_step(
                    x[:, token_idx, :],
                    pos[:, token_idx],
                    cache,
                )
                update_steps.append(bool(step_output.aux["output_valid"].item()))

        assert update_steps == [False, True, False, True]


class NestorOptimizationTest:
    """Regression tests for sparse dispatch and vectorized pooling."""

    def test_sparse_group_specific_experts_match_dense_reference(self) -> None:
        """Sparse expert dispatch should match the dense weighted sum exactly."""

        torch.manual_seed(0)
        layer = RoutedTransformerLayer(
            dim=8,
            num_experts_per_group=3,
            num_shared_experts=1,
            num_experts_per_group_per_token=2,
            num_groups=2,
            num_heads=1,
            head_dim=4,
            mlp_hidden_dims=[12],
            mlp_activation="gelu",
            router_score_mlp_hidden_dims=[],
            router_score_mlp_activation="gelu",
            router_noise_mlp_hidden_dims=[],
            router_noise_mlp_activation="gelu",
            depth_scale=0.25,
        )

        x = torch.randn(2, 3, 8)
        group_gates = torch.tensor(
            [
                [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]],
                [[0.6, 0.4], [0.0, 0.0], [1.0, 0.0]],
            ]
        )
        expert_gates = torch.tensor(
            [
                [[0.5, 0.5, 0.0], [0.0, 0.3, 0.7], [1.0, 0.0, 0.0]],
                [[0.2, 0.0, 0.8], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        valid_token_mask = torch.tensor(
            [[True, True, True], [True, False, True]],
            dtype=torch.bool,
        )

        group_router_output = RouterOutput(
            scores=torch.zeros_like(group_gates),
            softmax_probs=torch.zeros_like(group_gates),
            gates=group_gates,
            topk_indices=torch.zeros((2, 3, 1), dtype=torch.long),
            active_mask=group_gates > 0,
            valid_token_mask=valid_token_mask,
        )
        expert_router_output = RouterOutput(
            scores=torch.zeros_like(expert_gates),
            softmax_probs=torch.zeros_like(expert_gates),
            gates=expert_gates,
            topk_indices=torch.zeros((2, 3, 1), dtype=torch.long),
            active_mask=expert_gates > 0,
            valid_token_mask=valid_token_mask,
        )

        sparse_residual = layer._compute_expert_residual(
            x,
            group_router_output,
            expert_router_output,
        )

        dense_residual = torch.zeros_like(x)
        for expert_idx, expert in enumerate(layer.shared_experts):
            dense_residual = dense_residual + expert(x) * expert_gates[
                ...,
                expert_idx,
            ].unsqueeze(-1)
        specific_gates = expert_gates[..., layer.num_shared_experts :]
        for group_idx, group_experts in enumerate(layer.group_specific_experts):
            for expert_idx, expert in enumerate(group_experts):
                dense_residual = dense_residual + expert(x) * (
                    group_gates[..., group_idx] * specific_gates[..., expert_idx]
                ).unsqueeze(-1)

        assert torch.allclose(sparse_residual, dense_residual, atol=1e-6)

    def test_vectorized_pooling_matches_reference(self) -> None:
        """Vectorized pooling should preserve the old windowed semantics."""

        config = create_minimal_nestor_config()
        block = Nestor.from_config(config).blocks[0]

        x = torch.randn(2, 5, config.input_dim)
        pos = torch.tensor(
            [
                [0, 1, 2, 3, PADDING_TOKEN_POS],
                [0, PADDING_TOKEN_POS, 2, 3, 4],
            ]
        )
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        pooled_x, pooled_pos, pooled_attn_mask = block._pool_tokens(x, pos, attn_mask)

        valid_mask = pos != PADDING_TOKEN_POS
        output_seq_len = (pos.shape[1] + block.pooling_stride - 1) // block.pooling_stride
        ref_x = torch.zeros_like(pooled_x)
        ref_pos = torch.full_like(pooled_pos, fill_value=PADDING_TOKEN_POS)
        ref_valid = torch.zeros_like(pooled_pos, dtype=torch.bool)
        representative_indices = torch.zeros_like(pooled_pos, dtype=torch.long)

        for pooled_idx in range(output_seq_len):
            start = pooled_idx * block.pooling_stride
            end = min(start + block.pooling_stride, pos.shape[1])
            window_x = x[:, start:end, :]
            window_pos = pos[:, start:end]
            window_valid = valid_mask[:, start:end]
            weights = window_valid.unsqueeze(-1).to(dtype=x.dtype)
            # RMS pooling divides the masked window sum by sqrt(window size).
            denom = weights.sum(dim=1).clamp_min(1.0).sqrt()
            ref_x[:, pooled_idx] = (window_x * weights).sum(dim=1) / denom
            ref_valid[:, pooled_idx] = window_valid.any(dim=1)

            ranking = window_valid.to(dtype=torch.long) * torch.arange(
                1,
                end - start + 1,
                dtype=torch.long,
            )
            representative_indices[:, pooled_idx] = start + ranking.argmax(dim=1)

            mean_pos = torch.where(
                window_valid,
                window_pos,
                torch.zeros_like(window_pos),
            ).to(dtype=torch.float32).sum(dim=1) / window_valid.to(
                dtype=torch.float32
            ).sum(dim=1).clamp_min(1.0)
            ref_pos[:, pooled_idx] = torch.where(
                ref_valid[:, pooled_idx],
                torch.round(mean_pos).to(dtype=pos.dtype),
                torch.full(
                    (pos.shape[0],),
                    fill_value=PADDING_TOKEN_POS,
                    dtype=pos.dtype,
                ),
            )

        ref_attn_mask = torch.zeros_like(pooled_attn_mask)
        for batch_idx in range(pos.shape[0]):
            batch_representatives = representative_indices[batch_idx]
            ref_attn_mask[batch_idx] = (
                attn_mask[batch_idx]
                .index_select(0, batch_representatives)
                .index_select(1, batch_representatives)
            )
        ref_attn_mask = block._compose_attn_mask(ref_valid, ref_attn_mask)

        assert torch.allclose(pooled_x, ref_x, atol=1e-6)
        assert torch.equal(pooled_pos, ref_pos)
        assert torch.equal(pooled_attn_mask, ref_attn_mask)


class DefaultTransformerForwardPassTest:
    """Test class for the dense single-expert special case."""

    def test_forward_pass_runs_without_error(self) -> None:
        """
        Test that a forward pass through default transformer completes without errors.

        Creates a dense-transformer special case of hierarchical Nestor and verifies that:
        1. The model can be instantiated from config
        2. A forward pass completes without raising exceptions
        3. The output has the expected type (ModelOutput)
        """
        config = create_default_transformer_config()
        model = Nestor.from_config(config)

        batch_size = 2
        seq_len = 4
        input_dim = config.input_dim

        x = torch.randn(batch_size, seq_len, input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)

        assert isinstance(output, ModelOutput)

    def test_forward_pass_output_shape(self) -> None:
        """
        Test that the forward pass produces outputs with correct shapes.

        Verifies:
        1. pred tensor has shape (batch_size, output_dim)
        2. aux dict contains 'blocks' key with per-block outputs
        3. Block aux still contains router metadata in the single-expert special case
        """
        config = create_default_transformer_config()
        model = Nestor.from_config(config)

        batch_size = 4
        seq_len = 8
        input_dim = config.input_dim
        x = torch.randn(batch_size, seq_len, input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)

        # Check pred shape
        assert output.pred.shape == (batch_size, seq_len, config.output_dim), (
            f"Expected pred shape ({batch_size}, {seq_len}, {config.output_dim}), got {output.pred.shape}"
        )

        # Check aux structure
        assert "blocks" in output.aux
        assert len(output.aux["blocks"]) == config.num_blocks

        for block_aux in output.aux["blocks"]:
            assert "group_router" in block_aux
            assert "layers" in block_aux
            assert "scores" in block_aux["group_router"]
            assert "topk_indices" in block_aux["group_router"]

    def test_forward_pass_gradient_flow(self) -> None:
        """
        Test that gradients flow through the default transformer correctly.

        Verifies that backpropagation works and gradients are non-zero.
        """
        config = create_default_transformer_config()
        model = Nestor.from_config(config)

        batch_size = 2
        seq_len = 4
        input_dim = config.input_dim

        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)

        loss = output.pred.sum()
        loss.backward()

        # Check that input received gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that model parameters received gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        assert has_gradients, "No gradients found in model parameters"

    def test_forward_pass_with_varying_seq_lengths(self) -> None:
        """
        Test forward pass with different sequence lengths.

        Ensures the default transformer handles varying sequence lengths correctly.
        """
        config = create_default_transformer_config()
        model = Nestor.from_config(config)

        batch_size = 2
        input_dim = config.input_dim

        for seq_len in [2, 4, 16]:
            x = torch.randn(batch_size, seq_len, input_dim)
            pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
            attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

            output = model(x, pos, attn_mask)

            assert output.pred.shape == (batch_size, seq_len, config.output_dim)

    def test_forward_pass_eval_mode(self) -> None:
        """
        Test that forward pass works in eval mode for default transformer.

        Ensures model behaves correctly with dropout/batchnorm disabled.
        """
        config = create_default_transformer_config()
        model = Nestor.from_config(config)
        model.eval()

        batch_size = 2
        seq_len = 4
        input_dim = config.input_dim

        x = torch.randn(batch_size, seq_len, input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        with torch.no_grad():
            output = model(x, pos, attn_mask)

        assert isinstance(output, ModelOutput)
        assert output.pred.shape == (batch_size, seq_len, config.output_dim)
