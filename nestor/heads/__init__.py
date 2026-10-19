"""Prediction heads and head configs."""

from nestor.heads.base import Head, HeadConfig, list_head_types, register_head
from nestor.heads.language_modeling import (
    LanguageModelingHead,
    LanguageModelingHeadConfig,
)

__all__ = [
    "Head",
    "HeadConfig",
    "LanguageModelingHead",
    "LanguageModelingHeadConfig",
    "list_head_types",
    "register_head",
]
