"""Tests for top-level model definitions."""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch

from nestor.adapters import list_adapter_types
from nestor.backbones import (
    Backbone,
    BackboneConfig,
    ModelOutput,
    NestorConfig,
    list_backbone_types,
    register_backbone,
)
from nestor.configs.adapters import TextAdapterConfig
from nestor.heads import LanguageModelingHeadConfig, list_head_types
from nestor.models import (
    Model,
    SequenceModel,
    SequenceModelConfig,
)
from nestor.models.base import list_model_types

VOCAB = 32


@dataclasses.dataclass(kw_only=True)
class ToyBackboneConfig(BackboneConfig):
    """Configuration for a tiny test-only backbone."""

    backbone_type: Literal["toy_backbone"] = "toy_backbone"
    input_dim: int = 16
    hidden_dim: int = 12


@register_backbone("toy_backbone")
class ToyBackbone(Backbone):
    """Minimal backbone used to verify registry-based swapping."""

    config: ToyBackboneConfig

    def __init__(self, config: ToyBackboneConfig) -> None:
        """Initialize the test backbone.

        Args:
            config (ToyBackboneConfig): Backbone configuration.
        """

        super().__init__()
        self.config = config
        self.proj = torch.nn.Linear(config.input_dim, config.hidden_dim)

    @property
    def output_dim(self) -> int:
        """Return the output width."""

        return self.config.hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> ModelOutput:
        """Project the token sequence and forward positions through aux.

        Args:
            x (torch.Tensor): Input tokens of shape ``(batch_size, seq_len, input_dim)``.
            pos (torch.Tensor): Token positions of shape ``(batch_size, seq_len)``.
            attn_mask (torch.Tensor | None): Attention mask of shape
                ``(batch_size, seq_len, seq_len)`` or ``None``.

        Returns:
            ModelOutput: Projected token sequence and aux metadata.
        """

        del attn_mask
        pred = self.proj(x)
        return ModelOutput(pred=pred, aux={"positions": pos, "hidden_states": pred})

    def to_config(self) -> ToyBackboneConfig:
        """Serialize the backbone."""

        return self.config

    @classmethod
    def from_config(cls, config: BackboneConfig) -> ToyBackbone:
        """Construct the backbone from a generic config.

        Args:
            config (BackboneConfig): Backbone config.

        Returns:
            ToyBackbone: Constructed backbone.
        """

        assert isinstance(config, ToyBackboneConfig)
        return cls(config=config)


def _create_backbone_config() -> NestorConfig:
    """Create a tiny dense Nestor backbone config for model tests."""

    return NestorConfig(
        input_dim=16,
        num_blocks=1,
        feature_dims=[16],
        causal=True,
        router_num_experts_per_group=[1],
        pooling_strides=[1],
        num_groups=[1],
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


def _create_adapter_config() -> TextAdapterConfig:
    """Create a tiny text adapter config for model tests."""

    return TextAdapterConfig(embed_dim=16, vocab_size=VOCAB, max_length=8)


class SequenceModelTest:
    """Tests for generic sequence top-level models."""

    def test_sequence_model_forward_and_registry(self) -> None:
        """Sequence models should be registered and produce per-token logits."""

        config = SequenceModelConfig(
            adapter_config=_create_adapter_config(),
            backbone_config=_create_backbone_config(),
            head_config=LanguageModelingHeadConfig(vocab_size=VOCAB, hidden_dims=[12]),
        )
        assert "sequence" in set(list_model_types())
        assert "text" in set(list_adapter_types())
        assert "language_modeling" in set(list_head_types())

        model = Model.from_config(config)
        assert isinstance(model, SequenceModel)

        token_ids = torch.randint(0, VOCAB, (3, 5))
        encoded_output = model.encode(token_ids)
        prediction_output = model(token_ids)

        assert encoded_output.pred.shape == (3, 5, 16)
        assert prediction_output.pred.shape == (3, 5, VOCAB)
        assert model.to_config() == config


class BackboneRegistryTest:
    """Tests for the generic backbone registry."""

    def test_model_accepts_non_nestor_backbone_configs(self) -> None:
        """`SequenceModel` should construct from any registered backbone."""

        config = SequenceModelConfig(
            adapter_config=_create_adapter_config(),
            backbone_config=ToyBackboneConfig(input_dim=16, hidden_dim=12),
            head_config=LanguageModelingHeadConfig(vocab_size=VOCAB),
        )

        assert {"nestor", "toy_backbone"}.issubset(set(list_backbone_types()))

        model = Model.from_config(config)
        token_ids = torch.randint(0, VOCAB, (2, 4))
        output = model(token_ids)

        assert output.pred.shape == (2, 4, VOCAB)
        assert isinstance(model, SequenceModel)
        assert model.output_dim == 12
        assert model.to_config() == config
