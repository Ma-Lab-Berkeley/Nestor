"""Adapters for converting raw data to model inputs."""

from nestor.adapters.base import Adapter, list_adapter_types, register_adapter
from nestor.adapters.text import TextAdapter

__all__ = ["Adapter", "TextAdapter", "list_adapter_types", "register_adapter"]
