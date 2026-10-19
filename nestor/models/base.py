"""General model interface and registry."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from torch import nn

from nestor.backbones import ModelOutput


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
    """Base configuration for top-level models.

    Attributes:
        model_type (str): Registry key identifying the concrete model.
    """

    model_type: str = "base"


ModelT = TypeVar("ModelT", bound="Model")

_MODEL_REGISTRY: dict[str, type["Model"]] = {}


def register_model(model_type: str) -> Callable[[type[ModelT]], type[ModelT]]:
    """Register a concrete model implementation.

    Args:
        model_type (str): String key used in :class:`ModelConfig`.

    Returns:
        Callable[[type[ModelT]], type[ModelT]]: Class decorator.

    Raises:
        ValueError: If the model type is already registered.
    """

    def decorator(cls: type[ModelT]) -> type[ModelT]:
        if model_type in _MODEL_REGISTRY:
            raise ValueError(f"Model type '{model_type}' is already registered.")
        _MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def list_model_types() -> list[str]:
    """List all registered model types.

    Returns:
        list[str]: Sorted registry keys.
    """

    return sorted(_MODEL_REGISTRY)


class Model(nn.Module, ABC):
    """Top-level trainable model interface.

    A :class:`Model` owns the full learned mapping used by the pipelines:
    input adapter, seq-to-seq backbone, and optional prediction head. This
    keeps training, evaluation, checkpointing, and distributed wrapping
    independent from the details of any specific model file.
    """

    @abstractmethod
    def encode(self, batch: Any) -> ModelOutput:
        """Run the adapter and backbone on a raw batch.

        Args:
            batch (Any): Raw model input batch. The concrete type depends on the
                selected adapter.

        Returns:
            ModelOutput: Backbone output sequence in ``pred`` and auxiliary
                metadata in ``aux``.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the token width produced by the backbone.

        Returns:
            int: Backbone token feature dimension.
        """

        raise NotImplementedError

    @abstractmethod
    def to_config(self) -> ModelConfig:
        """Return the model configuration.

        Returns:
            ModelConfig: Serializable configuration that reconstructs the model.
        """

        raise NotImplementedError

    def fsdp2_wrap_order(self) -> list[nn.Module]:
        """Return modules to fully shard in bottom-up order.

        Concrete model files override this when they want to control FSDP2
        group boundaries. The default is to shard only the model root.

        Returns:
            list[nn.Module]: Modules to pass to ``fully_shard(...)``.
        """

        return [self]

    @staticmethod
    def from_config(config: ModelConfig) -> "Model":
        """Instantiate a registered model from configuration.

        Args:
            config (ModelConfig): Model configuration.

        Returns:
            Model: Constructed top-level model.

        Raises:
            ValueError: If ``config.model_type`` is unknown.
        """

        model_cls = _MODEL_REGISTRY.get(config.model_type)
        if model_cls is None:
            available = ", ".join(list_model_types()) or "<none>"
            raise ValueError(
                f"Unknown model type '{config.model_type}'. "
                f"Registered model types: {available}."
            )
        return model_cls.from_config(config)


def iter_unique_modules(modules: Iterable[nn.Module]) -> list[nn.Module]:
    """Deduplicate modules while preserving order.

    Args:
        modules (Iterable[nn.Module]): Candidate modules.

    Returns:
        list[nn.Module]: Ordered unique modules.
    """

    unique_modules: list[nn.Module] = []
    seen_ids: set[int] = set()
    for module in modules:
        module_id = id(module)
        if module_id in seen_ids:
            continue
        unique_modules.append(module)
        seen_ids.add(module_id)
    return unique_modules


def unwrap_model(model: nn.Module) -> Model:
    """Cast an ``nn.Module`` to a :class:`Model` with a type check.

    FSDP2 shards parameters in place and does not add a ``.module`` wrapper,
    so this helper is purely a typed access point to the top-level model
    interface.

    Args:
        model (nn.Module): Model expected to implement the top-level model
            interface.

    Returns:
        Model: The same module, typed as :class:`Model`.

    Raises:
        TypeError: If ``model`` is not a :class:`Model`.
    """

    if not isinstance(model, Model):
        raise TypeError(
            f"Expected Model, got {type(model).__name__}. "
            "The module must implement the top-level model interface."
        )
    return model
