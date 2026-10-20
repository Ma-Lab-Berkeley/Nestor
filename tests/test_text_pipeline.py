"""Tests for the text (LLM) adapter and end-to-end text classification model."""

from __future__ import annotations

import torch

from nestor.adapters import TextAdapter, list_adapter_types
from nestor.backbones import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.constants import PADDING_TOKEN_POS
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig


def _create_backbone_config(input_dim: int) -> NestorConfig:
    """Create a tiny dense Nestor backbone config for text tests."""

    return NestorConfig(
        input_dim=input_dim,
        num_blocks=1,
        feature_dims=[input_dim],
        router_num_experts_per_group=[1],
        pooling_strides=[1],
        num_groups=[1],
        block_num_layers=2,
        block_num_heads=2,
        block_head_dims=8,
        block_mlp_hidden_dims=[32],
        block_mlp_activations="gelu",
        router_num_experts_per_group_per_token=1,
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activations="gelu",
    )


class TextAdapterTest:
    """Tests for the tiktoken-based text adapter."""

    def test_registered(self) -> None:
        """The text adapter should be discoverable through the registry."""

        assert "text" in set(list_adapter_types())

    def test_token_id_tensor_input(self) -> None:
        """A pre-tokenized id tensor should embed to (B, L, embed_dim) inputs."""

        embed_dim = 16
        vocab_size = 64
        adapter = TextAdapter(
            TextAdapterConfig(
                embed_dim=embed_dim, vocab_size=vocab_size, max_length=8
            )
        )

        token_ids = torch.randint(0, vocab_size, (3, 5))
        x, pos, attn_mask = adapter(token_ids)

        assert x.shape == (3, 5, embed_dim)
        assert pos.shape == (3, 5)
        # Trivial-mask contract: tensor input has no padding and no pair
        # structure, so the adapter emits NO dense mask (None) and the
        # backbone applies causality analytically. See
        # tests/test_trivial_mask_contract.py.
        assert attn_mask is None
        # Every supplied id is a real token -> no padding sentinels.
        assert bool((pos != PADDING_TOKEN_POS).all())
        # Positions are 0..L-1 per row.
        assert torch.equal(pos[0], torch.arange(5))

    def test_string_input(self) -> None:
        """Raw strings should tokenize, pad, and mark padding via positions.

        Skipped offline when the tiktoken BPE vocabulary cannot be loaded.
        """

        try:
            import tiktoken

            tiktoken.get_encoding("gpt2")
        except Exception:
            return  # tokenizer/network unavailable; skip.

        embed_dim = 16
        adapter = TextAdapter(
            TextAdapterConfig(
                tokenizer="gpt2", embed_dim=embed_dim, max_length=32
            )
        )

        texts = ["hello world", "a much longer piece of example text here"]
        x, pos, attn_mask = adapter(texts)

        batch_size, seq_len, dim = x.shape
        assert batch_size == 2
        assert dim == embed_dim
        assert pos.shape == (2, seq_len)
        assert attn_mask.shape == (2, seq_len, seq_len)
        # The shorter sentence must contain at least one padded (masked) slot.
        assert bool((pos[0] == PADDING_TOKEN_POS).any())
        # Valid tokens carry ascending positions starting at 0.
        assert int(pos[1, 0]) == 0


class TextSequenceModelTest:
    """End-to-end tests for the adapter + Nestor + language-modeling head model."""

    def test_forward_produces_token_logits(self) -> None:
        """A token-id batch should flow end-to-end to per-token vocabulary logits."""

        embed_dim = 16
        vocab_size = 64
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(
                embed_dim=embed_dim, vocab_size=vocab_size, max_length=16
            ),
            backbone_config=_create_backbone_config(embed_dim),
            head_config=LanguageModelingHeadConfig(vocab_size=vocab_size),
        )

        model = Model.from_config(config)
        assert isinstance(model, SequenceModel)

        token_ids = torch.randint(0, vocab_size, (3, 7))
        output = model(token_ids)

        assert output.pred.shape == (3, 7, vocab_size)
        assert model.to_config() == config

    def test_backward_pass(self) -> None:
        """Gradients should reach the token embedding through the full model."""

        embed_dim = 16
        vocab_size = 64
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(
                embed_dim=embed_dim, vocab_size=vocab_size, max_length=16
            ),
            backbone_config=_create_backbone_config(embed_dim),
            head_config=LanguageModelingHeadConfig(vocab_size=vocab_size),
        )
        model = Model.from_config(config)

        token_ids = torch.randint(0, vocab_size, (2, 6))
        logits = model(token_ids).pred  # [B, S, V]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size), token_ids[:, 1:].reshape(-1)
        )
        loss.backward()

        assert model.adapter.embed.weight.grad is not None
        assert torch.isfinite(loss)
