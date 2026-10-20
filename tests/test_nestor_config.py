"""Tests for Nestor configuration expansion helpers."""

from __future__ import annotations

from nestor.backbones import NestorConfig


def _create_two_layer_config() -> NestorConfig:
    """Create a compact two-layer config used for promotion tests."""
    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[32, 32],
        router_num_experts_per_group=[2, 1],
        router_num_shared_experts=[1, 1],
        pooling_strides=[4, 1],
        num_groups=[2, 1],
        num_groups_per_token=[2, 1],
        block_num_layers=1,
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


class NestorConfigPromotionTest:
    """Tests for per-layer config promotion logic."""

    def test_block_num_layers_promotes_per_block(self) -> None:
        """Layerwise `block_num_layers` should stay integer per Nestor block."""
        config = _create_two_layer_config()
        config.block_num_layers = [5, 7]

        config_dict = config.to_dict()
        num_layer_values = [
            block_config["block_num_layers"]
            for block_config in config_dict["block_configs"]
        ]

        assert num_layer_values == [5, 7]
        assert all(isinstance(v, int) for v in num_layer_values)

    def test_pooling_schedule_is_preserved_in_block_configs(self) -> None:
        """Hierarchical pooling schedules should survive config expansion."""
        config = _create_two_layer_config()
        config_dict = config.to_dict()

        pooling_strides = [
            block_config["pooling_stride"]
            for block_config in config_dict["block_configs"]
        ]

        assert pooling_strides == [4, 1]

    def test_group_and_expert_routing_schedule_is_preserved(self) -> None:
        """Config expansion should keep the per-block routing schedule intact."""
        config = _create_two_layer_config()
        config_dict = config.to_dict()

        assert [
            block_config["num_groups_per_token"]
            for block_config in config_dict["block_configs"]
        ] == [2, 1]
        assert [
            block_config["router_num_shared_experts"]
            for block_config in config_dict["block_configs"]
        ] == [1, 1]
