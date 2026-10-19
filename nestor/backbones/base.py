"""Token-domain backbone interface, shared abstractions, and registry.

A backbone turns a token sequence ``(x, pos, attn_mask)`` into a
:class:`ModelOutput`. Everything a concrete backbone needs to interoperate with
the rest of the codebase lives in this file:

- :class:`ModelOutput`: the common output container returned by backbones and
  (ultimately) by top-level models.
- :class:`BackboneConfig`: the base configuration class subclasses extend.
- :class:`Backbone`: the modality-agnostic interface implemented by backbones.
- :func:`register_backbone` / :func:`list_backbone_types`: registry helpers
  that let the config layer instantiate the right backbone by name.

Concrete backbones live in sibling files (one self-contained file per family)
and import these abstractions.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class ModelOutput(NamedTuple):
    """Prediction container shared by backbones and top-level models.

    Attributes:
        pred (Tensor): Prediction tensor. For backbones this is the output
            token sequence; for top-level models this is the head output.
        aux (dict[str, Any]): Auxiliary metadata (per-block router outputs,
            positions, validity masks, etc.).
    """

    pred: Tensor
    aux: dict[str, Any]


@dataclasses.dataclass(kw_only=True)
class BackboneConfig:
    """Base configuration for token-domain backbones.

    Attributes:
        backbone_type (str): Registry key identifying the concrete backbone
            implementation.
    """

    backbone_type: str = "base"


BackboneT = TypeVar("BackboneT", bound="Backbone")

_BACKBONE_REGISTRY: dict[str, type["Backbone"]] = {}


def register_backbone(
    backbone_type: str,
) -> Callable[[type[BackboneT]], type[BackboneT]]:
    """Register a concrete backbone implementation.

    Args:
        backbone_type (str): String key stored in :class:`BackboneConfig`.

    Returns:
        Callable[[type[BackboneT]], type[BackboneT]]: Class decorator.

    Raises:
        ValueError: If the registry key is already in use.
    """

    def decorator(cls: type[BackboneT]) -> type[BackboneT]:
        if backbone_type in _BACKBONE_REGISTRY:
            raise ValueError(f"Backbone type '{backbone_type}' is already registered.")
        _BACKBONE_REGISTRY[backbone_type] = cls
        return cls

    return decorator


def list_backbone_types() -> list[str]:
    """List registered backbone types.

    Returns:
        list[str]: Sorted registry keys.
    """

    return sorted(_BACKBONE_REGISTRY)


class Backbone(nn.Module, ABC):
    """Base class for modality-agnostic token backbones.

    A backbone consumes token embeddings, token positions, and an attention
    mask, then returns a :class:`ModelOutput` whose ``pred`` field contains the
    transformed token sequence or final feature representation. Adapters and
    heads live outside this interface.
    """

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len input_dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
    ) -> ModelOutput:
        """Run the backbone on a token sequence.

        Args:
            x (Float[Tensor, "batch_size seq_len input_dim"]): Input token
                embeddings.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions, with any
                padded entries encoded by the caller.
            attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None):
                Attention mask before backbone-specific validity masking, or
                ``None`` under the trivial-mask contract (every token valid,
                no pair structure beyond the backbone's own causality).

        Returns:
            ModelOutput: Backbone predictions and auxiliary routing/state data.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the feature dimension produced by the backbone."""

        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> BackboneConfig:
        """Serialize the backbone into a config."""

        raise NotImplementedError

    def fsdp2_wrap_order(self) -> list[nn.Module]:
        """Return a bottom-up FSDP2 wrap order for this backbone.

        Returns:
            list[nn.Module]: Modules to pass to ``fully_shard(...)``.
        """

        return [self]

    def update_detached_parameters(self, aux: dict[str, object]) -> None:
        """Update any detached backbone state after the optimizer step.

        Args:
            aux (dict[str, object]): Merged auxiliary outputs for the current
                optimizer step.
        """

        del aux

    @staticmethod
    def from_config(config: BackboneConfig) -> "Backbone":
        """Instantiate a registered backbone from configuration.

        Args:
            config (BackboneConfig): Backbone configuration.

        Returns:
            Backbone: Constructed backbone.

        Raises:
            ValueError: If ``config.backbone_type`` is unknown.
        """

        backbone_cls = _BACKBONE_REGISTRY.get(config.backbone_type)
        if backbone_cls is None:
            available = ", ".join(list_backbone_types()) or "<none>"
            raise ValueError(
                f"Unknown backbone type '{config.backbone_type}'. "
                f"Registered backbones: {available}."
            )
        return backbone_cls.from_config(config)
