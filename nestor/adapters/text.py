"""Text adapter that tokenizes raw text and embeds the tokens."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

from nestor.adapters.base import Adapter, register_adapter
from nestor.configs.adapters import TextAdapterConfig
from nestor.constants import PADDING_TOKEN_POS
from nestor.scaling.blocks import RMSNorm, ScaledEmbedding


@register_adapter("text")
class TextAdapter(Adapter):
    """
    Adapter for converting text to token-embedding sequences for Nestor.

    The adapter *is* the tokenizer: it encodes raw strings with a tiktoken BPE
    encoding, looks up a learned embedding per token, normalizes the embeddings
    to unit RMS, and produces the ``(x, pos, attn_mask)`` token-domain inputs the
    backbone expects. Position is carried only by the integer ``pos`` indices --
    Nestor encodes position with RoPE inside attention -- and padded slots are
    marked with ``PADDING_TOKEN_POS`` so the backbone masks them out.

    For convenience the adapter also accepts an already-tokenized integer tensor
    of shape ``(batch_size, seq_len)``, in which case every position is treated
    as a valid token and no tokenizer is loaded.

    Attributes:
        config (TextAdapterConfig): Adapter configuration.
        embed (ScaledEmbedding): Token embedding table.
        norm (RMSNorm): RMS normalization applied to the token embeddings.

    Input:
        Either a sequence of ``batch_size`` strings, or an integer tensor of
        token ids with shape ``(batch_size, seq_len)``.

    Output:
        x: Token embeddings. Shape: (batch_size, seq_len, embed_dim).
        pos: Position indices (``PADDING_TOKEN_POS`` on pads). Shape: (batch_size, seq_len).
        attn_mask: Full attention mask. Shape: (batch_size, seq_len, seq_len).
    """

    config: TextAdapterConfig
    embedding_multiplier: float

    def __init__(self, config: TextAdapterConfig):
        """
        Initialize the text adapter.

        Args:
            config (TextAdapterConfig): Adapter configuration.
        """
        super().__init__(config)

        # The tokenizer is loaded lazily: only string inputs need it, and an
        # explicit ``vocab_size`` lets pre-tokenized inputs skip it entirely.
        self._tokenizer: Any | None = None
        if config.vocab_size is not None:
            vocab_size = config.vocab_size
        else:
            vocab_size = self._get_tokenizer().n_vocab
        self.vocab_size = vocab_size

        # Embeddings contract over a finite vocabulary axis, so they are
        # base-rate AdamW with no width factor (see ScaledEmbedding).
        self.embed = ScaledEmbedding(vocab_size, config.embed_dim)
        # Normalize token embeddings to unit RMS before the first block.
        self.norm = RMSNorm(config.embed_dim)
        self.embedding_multiplier = 1.0

    def _get_tokenizer(self) -> Any:
        """Return the tiktoken encoding, loading and caching it on first use."""

        if self._tokenizer is None:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding(self.config.tokenizer)
        return self._tokenizer

    def _encode_texts(
        self, texts: list[str]
    ) -> tuple[Int[Tensor, "batch_size seq_len"], Bool[Tensor, "batch_size seq_len"]]:
        """Tokenize a batch of strings into a padded id tensor and validity mask.

        Args:
            texts (list[str]): Raw input strings.

        Returns:
            tuple[Tensor, Tensor]: ``(token_ids, valid_mask)`` on the CPU, where
                ``token_ids`` has shape ``(batch_size, seq_len)`` and
                ``valid_mask`` marks the real (non-padding) tokens.
        """

        tokenizer = self._get_tokenizer()
        max_length = self.config.max_length

        # ``encode_ordinary`` treats any special-token markers as ordinary text,
        # so arbitrary user strings never raise.
        encoded: list[list[int]] = []
        for text in texts:
            ids = tokenizer.encode_ordinary(text)[:max_length]
            if not ids:
                # Keep at least one token so the example contributes a position.
                ids = [0]
            encoded.append(ids)

        seq_len = max(len(ids) for ids in encoded)
        token_ids = torch.zeros((len(encoded), seq_len), dtype=torch.long)
        valid_mask = torch.zeros((len(encoded), seq_len), dtype=torch.bool)
        for i, ids in enumerate(encoded):
            token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            valid_mask[i, : len(ids)] = True
        return token_ids, valid_mask

    def forward(
        self,
        batch: object,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len embed_dim"],
        Int[Tensor, "batch_size seq_len"],
        Bool[Tensor, "batch_size seq_len seq_len"] | None,
    ]:
        """
        Convert text (or token ids) into a token-embedding sequence.

        Args:
            batch (object): Either a sequence of strings or an integer tensor of
                token ids with shape ``(batch_size, seq_len)``.

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: (x, pos, attn_mask)
                x: Token embeddings. Shape: (batch_size, seq_len, embed_dim).
                pos: Position indices. Shape: (batch_size, seq_len).
                attn_mask: Attention mask, shape (batch_size, seq_len, seq_len)
                    for padded string batches; ``None`` (the trivial-mask
                    contract) for pre-tokenized tensors, where every position
                    is valid and the mask would carry no information. The
                    backbone applies causality analytically on the ``None``
                    path -- materializing the dense mask here is O(S^2) memory
                    (16 GiB bool at S=131072) for pure redundancy.
        """

        device = self.embed.weight.device

        if isinstance(batch, Tensor):
            token_ids = batch.to(device=device, dtype=torch.long)
            valid_mask = None  # every position is a valid token
        else:
            token_ids, valid_mask = self._encode_texts(list(batch))
            token_ids = token_ids.to(device)
            valid_mask = valid_mask.to(device)

        # Embed and normalize to unit RMS before the first block.
        x = self.norm(self.embed(token_ids)) * self.embedding_multiplier

        batch_size, seq_len = token_ids.shape

        # Real tokens carry positions 0, 1, ...; padded slots are marked with
        # PADDING_TOKEN_POS so the backbone masks them out of attention/routing.
        pos = torch.arange(seq_len, device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch_size, -1)
        if valid_mask is None:
            # Trivial-mask contract: no padding, no pair structure -- emit no
            # dense mask at all (the backbone's None paths handle causality).
            return x, pos.clone(), None
        pos = pos.clone().masked_fill(~valid_mask, PADDING_TOKEN_POS)

        # Full attention mask; the backbone composes it with token validity.
        attn_mask = torch.ones(
            (batch_size, seq_len, seq_len),
            device=device,
            dtype=torch.bool,
        )

        return x, pos, attn_mask
