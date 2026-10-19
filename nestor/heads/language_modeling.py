"""Language-modeling head that predicts a vocabulary distribution per token."""

from __future__ import annotations

import dataclasses
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from nestor.backbones import ModelOutput
from nestor.heads.base import Head, HeadConfig, register_head
from nestor.scaling.blocks import ScaledMLP


@dataclasses.dataclass(kw_only=True)
class LanguageModelingHeadConfig(HeadConfig):
    """Configuration for a language-modeling (next-token) head.

    Attributes:
        head_type (Literal["language_modeling"]): Registry key for this head.
        vocab_size (int): Output vocabulary size. Must match the tokenizer /
            embedding vocabulary used by the text adapter.
        hidden_dims (list[int]): Hidden widths between the backbone features and
            the vocabulary logits. Usually empty (a single linear readout).
        activation (str): Activation used between hidden layers.
    """

    head_type: Literal["language_modeling"] = "language_modeling"
    vocab_size: int
    hidden_dims: list[int] = dataclasses.field(default_factory=list)
    activation: str = "gelu"


@register_head("language_modeling")
class LanguageModelingHead(Head):
    """Head for autoregressive language modeling.

    This head does **not** pool the sequence: it
    maps every token's hidden state to a distribution over the vocabulary,
    producing logits of shape ``(batch_size, seq_len, vocab_size)``. Paired with
    a causal backbone and a shifted cross-entropy loss, this is next-token
    prediction -- the standard LLM training objective.
    """

    config: LanguageModelingHeadConfig
    input_dim: int
    projection: ScaledMLP
    logit_multiplier: float

    def __init__(
        self,
        config: LanguageModelingHeadConfig,
        *,
        input_dim: int,
    ) -> None:
        """Initialize the language-modeling head.

        Args:
            config (LanguageModelingHeadConfig): Head configuration.
            input_dim (int): Backbone token width consumed by the head.
        """

        super().__init__()
        self.config = config
        self.input_dim = input_dim
        # The unembedding is a score head: its logits are read by the softmax,
        # so it is initialized with variance 1/d_in^2 and trained with AdamW at
        # rate eta_A / d_in (see nestor.scaling.blocks).
        self.projection = ScaledMLP(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.vocab_size,
            activation=config.activation,
            final_role="score",
        )
        self.logit_multiplier = 1.0

    @property
    def output_dim(self) -> int:
        """Return the vocabulary size."""

        return self.config.vocab_size

    def forward(
        self,
        backbone_output: ModelOutput,
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """Map each backbone token to vocabulary logits.

        Args:
            backbone_output (ModelOutput): Backbone output sequence.

        Returns:
            Float[Tensor, "batch_size seq_len vocab_size"]: Per-token logits.
        """

        return self.projection(backbone_output.pred) * self.logit_multiplier

    def to_config(self) -> LanguageModelingHeadConfig:
        """Serialize the head configuration."""

        return self.config

    @classmethod
    def from_config(
        cls,
        config: HeadConfig,
        *,
        input_dim: int,
    ) -> "LanguageModelingHead":
        """Construct the head from generic head config.

        Args:
            config (HeadConfig): Generic head config.
            input_dim (int): Backbone token width consumed by the head.

        Returns:
            LanguageModelingHead: Constructed head.
        """

        if not isinstance(config, LanguageModelingHeadConfig):
            raise TypeError(
                f"{cls.__name__}.from_config expected LanguageModelingHeadConfig, "
                f"got {type(config).__name__}."
            )
        return cls(config=config, input_dim=input_dim)
