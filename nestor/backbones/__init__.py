"""Backbone interface, shared abstractions, and the Nestor backbone.

The base abstractions (:class:`Backbone`, :class:`BackboneConfig`,
:class:`ModelOutput`, and the registry helpers) live in
:mod:`nestor.backbones.base`; the Nestor model in :mod:`nestor.backbones.nestor`.
"""

from nestor.backbones.base import (
    Backbone,
    BackboneConfig,
    ModelOutput,
    list_backbone_types,
    register_backbone,
)
from nestor.backbones.attention_residual import BlockAttentionResidual
from nestor.backbones.nestor import (
    Nestor,
    NestorBlock,
    NestorBlockConfig,
    NestorConfig,
    RoutedTransformerLayer,
    RouterOutput,
)

__all__ = [
    "Backbone",
    "BackboneConfig",
    "BlockAttentionResidual",
    "ModelOutput",
    "Nestor",
    "NestorBlock",
    "NestorBlockConfig",
    "NestorConfig",
    "RoutedTransformerLayer",
    "RouterOutput",
    "list_backbone_types",
    "register_backbone",
]
