"""Prediction-head interface and registry."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from nestor.backbones import ModelOutput


@dataclasses.dataclass(kw_only=True)
class HeadConfig:
    """Base configuration for prediction heads.

    Attributes:
        head_type (str): Registry key identifying the concrete head.
    """

    head_type: str = "base"


HeadT = TypeVar("HeadT", bound="Head")

_HEAD_REGISTRY: dict[str, type["Head"]] = {}


def register_head(head_type: str) -> Callable[[type[HeadT]], type[HeadT]]:
    """Register a concrete prediction head implementation.

    Args:
        head_type (str): Registry key used by :class:`HeadConfig`.

    Returns:
        Callable[[type[HeadT]], type[HeadT]]: Class decorator.
    """

    def decorator(cls: type[HeadT]) -> type[HeadT]:
        if head_type in _HEAD_REGISTRY:
            raise ValueError(f"Head type '{head_type}' is already registered.")
        _HEAD_REGISTRY[head_type] = cls
        return cls

    return decorator


def list_head_types() -> list[str]:
    """List registered head types.

    Returns:
        list[str]: Sorted registry keys.
    """

    return sorted(_HEAD_REGISTRY)


class Head(nn.Module, ABC):
    """Base class for heads mapping backbone outputs to task predictions."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the prediction width produced by the head."""

        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        backbone_output: ModelOutput,
    ) -> Float[Tensor, "batch_size output_dim"]:
        """Map a backbone output to task predictions."""

        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> HeadConfig:
        """Serialize the head configuration."""

        raise NotImplementedError

    def fsdp2_wrap_order(self) -> list[nn.Module]:
        """Return a bottom-up FSDP2 wrap order for the head."""

        return [self]

    @staticmethod
    def from_config(config: HeadConfig, *, input_dim: int) -> "Head":
        """Instantiate a registered head from configuration.

        Args:
            config (HeadConfig): Head configuration.
            input_dim (int): Embedding dimension consumed by the head.

        Returns:
            Head: Constructed prediction head.
        """

        head_cls = _HEAD_REGISTRY.get(config.head_type)
        if head_cls is None:
            available = ", ".join(list_head_types()) or "<none>"
            raise ValueError(
                f"Unknown head type '{config.head_type}'. "
                f"Registered heads: {available}."
            )
        return head_cls.from_config(config, input_dim=input_dim)
