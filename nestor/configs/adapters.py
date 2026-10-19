"""Adapter configuration for converting raw data to model inputs."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class AdapterConfig:
    """
    Base configuration for adapters.

    Attributes:
        adapter_type (str): Adapter type identifier.
    """

    adapter_type: str = "base"


@dataclasses.dataclass(kw_only=True)
class TextAdapterConfig(AdapterConfig):
    """
    Configuration for the text adapter that tokenizes and embeds raw text.

    The adapter tokenizes strings (with tiktoken) and embeds the integer
    token ids. Position is encoded by RoPE inside the
    backbone, so the adapter emits only integer position indices and marks any
    padded slot with ``PADDING_TOKEN_POS``.

    Attributes:
        adapter_type (Literal["text"]): Adapter type identifier.
        tokenizer (str): Name of the tiktoken encoding (e.g. ``"gpt2"`` or
            ``"cl100k_base"``). Only loaded when raw text is tokenized.
        embed_dim (int): Embedding dimension for tokens. Must match the backbone
            ``input_dim``.
        max_length (int): Maximum number of tokens kept per example; longer
            sequences are truncated.
        vocab_size (int | None): Embedding table size. When ``None`` it is
            derived from the tokenizer's ``n_vocab``; set it explicitly to embed
            pre-tokenized integer inputs without loading a tokenizer.
    """

    adapter_type: Literal["text"] = "text"
    tokenizer: str = "gpt2"
    embed_dim: int = 768
    max_length: int = 256
    vocab_size: int | None = None

