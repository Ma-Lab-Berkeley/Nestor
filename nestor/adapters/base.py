"""Base adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from torch import nn, Tensor

from nestor.configs.adapters import AdapterConfig


AdapterT = TypeVar("AdapterT", bound="Adapter")

_ADAPTER_REGISTRY: dict[str, type["Adapter"]] = {}


def register_adapter(
    adapter_type: str,
) -> Callable[[type[AdapterT]], type[AdapterT]]:
    """Register a concrete adapter implementation.

    Args:
        adapter_type (str): Registry key stored in :class:`AdapterConfig`.

    Returns:
        Callable[[type[AdapterT]], type[AdapterT]]: Class decorator.
    """

    def decorator(cls: type[AdapterT]) -> type[AdapterT]:
        if adapter_type in _ADAPTER_REGISTRY:
            raise ValueError(f"Adapter type '{adapter_type}' is already registered.")
        _ADAPTER_REGISTRY[adapter_type] = cls
        return cls

    return decorator


def list_adapter_types() -> list[str]:
    """List registered adapter types.

    Returns:
        list[str]: Sorted registry keys.
    """

    return sorted(_ADAPTER_REGISTRY)


class Adapter(nn.Module, ABC):
    """
    Base adapter class for converting raw data to model inputs.

    Adapters transform raw inputs (e.g., images, audio, text, or already
    tokenized batches) into the token-domain format expected by backbones:
    ``(x, pos, attn_mask)``.

    Attributes:
        config (AdapterConfig): Adapter configuration.
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize the adapter.

        Args:
            config (AdapterConfig): Adapter configuration.
        """
        super().__init__()
        self.config: AdapterConfig = config

    @abstractmethod
    def forward(self, batch: Any) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Transform a batch of data into model inputs.

        Args:
            batch: Input batch (format depends on adapter type).

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: (x, pos, attn_mask)
                x: Input features. Shape: (batch_size, seq_len, input_dim).
                pos: Position indices. Shape: (batch_size, seq_len).
                attn_mask: Attention mask, shape (batch_size, seq_len,
                    seq_len), or ``None`` under the trivial-mask contract
                    (every token valid, no pair structure beyond the
                    backbone's own causality -- see TextAdapter).
        """
        pass

    @staticmethod
    def from_config(config: AdapterConfig) -> "Adapter":
        """
        Create an adapter from configuration.

        Args:
            config (AdapterConfig): Adapter configuration.

        Returns:
            Adapter: The constructed adapter.
        """
        adapter_cls = _ADAPTER_REGISTRY.get(config.adapter_type)
        if adapter_cls is None:
            available = ", ".join(list_adapter_types()) or "<none>"
            raise ValueError(
                f"Unknown adapter type '{config.adapter_type}'. "
                f"Registered adapters: {available}."
            )
        return adapter_cls(config)
