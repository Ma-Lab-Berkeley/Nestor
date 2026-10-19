"""Generic adapter-backbone-head model with a seq-to-seq backbone."""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch
from torch import nn

from nestor.adapters import Adapter
from nestor.backbones import Backbone, BackboneConfig, ModelOutput
from nestor.configs.adapters import AdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.heads import Head, HeadConfig
from nestor.models.base import Model, ModelConfig, register_model


@dataclasses.dataclass(kw_only=True)
class SequenceModelConfig(ModelConfig):
    """Configuration for a generic adapter-backbone-head model.

    Attributes:
        model_type (Literal["sequence"]): Registry key for this model.
        adapter_config (AdapterConfig): Raw-input adapter configuration.
        backbone_config (BackboneConfig): Token-domain backbone configuration.
        head_config (HeadConfig): Prediction head configuration.
        reference_mup (ReferenceMupConfig | None): Optional post-construction
            transform matching the variable-width-transformers paper recipe.
    """

    model_type: Literal["sequence"] = "sequence"
    adapter_config: AdapterConfig
    backbone_config: BackboneConfig
    head_config: HeadConfig
    reference_mup: ReferenceMupConfig | None = None


@register_model("sequence")
class SequenceModel(Model):
    """Generic model composed of adapter, seq-to-seq backbone, and head."""

    adapter: Adapter
    backbone: Backbone
    head: Head
    reference_mup: ReferenceMupConfig | None

    def __init__(
        self,
        *,
        adapter: Adapter,
        backbone: Backbone,
        head: Head,
        reference_mup: ReferenceMupConfig | None = None,
    ) -> None:
        """Initialize the sequence model.

        Args:
            adapter (Adapter): Raw-input adapter.
            backbone (Backbone): Token-domain seq-to-seq backbone.
            head (Head): Task-specific prediction head.
            reference_mup (ReferenceMupConfig | None): Applied reference muP
                configuration, retained for serialization.
        """

        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.head = head
        self.reference_mup = reference_mup

    def encode(self, batch: object) -> ModelOutput:
        """Encode a raw batch through the adapter and backbone.

        Args:
            batch (object): Raw model inputs.

        Returns:
            ModelOutput: Backbone output sequence and auxiliary metadata.
        """

        x, pos, attn_mask = self.adapter(batch)
        return self.backbone(x, pos, attn_mask)

    def forward(self, batch: object) -> ModelOutput:
        """Predict task outputs for a raw batch.

        Args:
            batch (object): Raw model inputs.

        Returns:
            ModelOutput: Head predictions in ``pred`` and backbone aux in
                ``aux``.
        """

        backbone_output = self.encode(batch)

        # NESTOR_FUSED_HEAD_CE=1 (train mode, grad on): skip the unembedding
        # and hand the LOSS the hidden states plus the head module via aux.
        # The loss projects hidden -> logits in row chunks under activation
        # checkpointing, so the full [tokens, vocab] logits tensor -- 6.12 GiB
        # in bf16 at the 1b arm's per-rank 8, the forward-phase allocation
        # that OOMs that batch shape -- is never materialized. Identical math:
        # the projection is per-token, so chunking and the loss's target shift
        # commute with it. Eval and no-grad calls keep the real logits path,
        # so held-out metrics and calibration probes are unchanged.
        import os

        if (
            self.training
            and torch.is_grad_enabled()
            and os.environ.get("NESTOR_FUSED_HEAD_CE") == "1"
        ):
            aux = dict(backbone_output.aux)
            aux["fused_head_module"] = self.head
            return ModelOutput(pred=backbone_output.pred, aux=aux)

        predictions = self.head(backbone_output)
        return ModelOutput(pred=predictions, aux=backbone_output.aux)

    @property
    def output_dim(self) -> int:
        """Return the backbone token width."""

        return self.backbone.output_dim

    def to_config(self) -> SequenceModelConfig:
        """Serialize the model configuration."""

        return SequenceModelConfig(
            adapter_config=self.adapter.config,
            backbone_config=self.backbone.to_config(),
            head_config=self.head.to_config(),
            reference_mup=self.reference_mup,
        )

    def fsdp2_wrap_order(self) -> list[nn.Module]:
        """Return a bottom-up FSDP2 wrap order."""

        return [
            self.adapter,
            *self.backbone.fsdp2_wrap_order(),
            *self.head.fsdp2_wrap_order(),
            self,
        ]

    @classmethod
    def from_config(cls, config: ModelConfig) -> "SequenceModel":
        """Construct the model from generic config."""

        if not isinstance(config, SequenceModelConfig):
            raise TypeError(
                f"{cls.__name__}.from_config expected SequenceModelConfig, got "
                f"{type(config).__name__}."
            )

        backbone = Backbone.from_config(config.backbone_config)
        adapter = Adapter.from_config(config.adapter_config)
        head = Head.from_config(config.head_config, input_dim=backbone.output_dim)
        if config.reference_mup is not None:
            from nestor.scaling.reference_mup import configure_reference_mup_model

            configure_reference_mup_model(
                adapter,
                backbone,
                head,
                config.reference_mup,
            )
        # NESTOR_STACKED_EXPERT_BANKS=1: stack the expert banks LAST, after
        # every init/tagging pass wrote the per-expert parameters.
        from nestor.backbones.expert_bank_stack import consolidate_stacked_expert_banks

        consolidate_stacked_expert_banks(backbone)
        return cls(
            adapter=adapter,
            backbone=backbone,
            head=head,
            reference_mup=config.reference_mup,
        )
