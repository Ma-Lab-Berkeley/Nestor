"""Tests for the causal language-modeling (next-token prediction) pipeline."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nestor.backbones import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.losses import CausalLMLossConfig
from nestor.constants import PADDING_TOKEN_VALUE
from nestor.heads import LanguageModelingHeadConfig, list_head_types
from nestor.models import Model, SequenceModel, SequenceModelConfig
from nestor.train.losses import CausalLMLoss, Loss


def _causal_backbone_config(input_dim: int) -> NestorConfig:
    """Create a tiny causal Nestor decoder config (no token pooling)."""

    return NestorConfig(
        input_dim=input_dim,
        num_blocks=1,
        feature_dims=[input_dim],
        causal=True,
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


def _lm_model(
    *, embed_dim: int = 16, vocab_size: int = 64, max_length: int = 16
) -> SequenceModel:
    """Build a tiny end-to-end causal LM (adapter + causal backbone + LM head)."""

    config = SequenceModelConfig(
        adapter_config=TextAdapterConfig(
            embed_dim=embed_dim, vocab_size=vocab_size, max_length=max_length
        ),
        backbone_config=_causal_backbone_config(embed_dim),
        head_config=LanguageModelingHeadConfig(vocab_size=vocab_size),
    )
    model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model


class LanguageModelingHeadTest:
    """Tests for the per-token vocabulary head."""

    def test_registered(self) -> None:
        """The language-modeling head should be discoverable in the registry."""

        assert "language_modeling" in set(list_head_types())

    def test_per_token_vocab_logits(self) -> None:
        """The model should emit (batch, seq_len, vocab) logits, not pooled."""

        vocab_size = 64
        model = _lm_model(vocab_size=vocab_size)
        token_ids = torch.randint(0, vocab_size, (3, 7))
        output = model(token_ids)

        assert output.pred.shape == (3, 7, vocab_size)


class CausalLMLossTest:
    """Tests for the shifted next-token cross-entropy loss."""

    def test_matches_manual_shift(self) -> None:
        """The loss should equal a manual shifted cross-entropy."""

        loss_fn = Loss.from_config(CausalLMLossConfig())
        assert isinstance(loss_fn, CausalLMLoss)

        batch_size, seq_len, vocab_size = 2, 5, 11
        torch.manual_seed(0)
        pred = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        out = loss_fn(pred, targets)
        expected = F.cross_entropy(
            pred[:, :-1, :].reshape(-1, vocab_size),
            targets[:, 1:].reshape(-1),
            ignore_index=PADDING_TOKEN_VALUE,
        )
        assert torch.allclose(out.total, expected)
        assert torch.allclose(out.components["perplexity"], expected.exp())

    def test_ignores_padding(self) -> None:
        """Padded target positions must not contribute to the loss."""

        loss_fn = Loss.from_config(CausalLMLossConfig())
        vocab_size = 11
        pred = torch.randn(1, 4, vocab_size)
        targets = torch.tensor([[3, 5, PADDING_TOKEN_VALUE, 2]])

        # Only positions whose *target* (shifted) is valid count: predict 5 then
        # skip the padded token, then predict 2.
        out = loss_fn(pred, targets)
        kept = torch.tensor([True, False, True])
        expected = F.cross_entropy(
            pred[:, :-1, :].reshape(-1, vocab_size)[kept],
            targets[:, 1:].reshape(-1)[kept],
        )
        assert torch.allclose(out.total, expected)

    def test_rejects_batches_with_no_valid_shifted_targets(self) -> None:
        """The training loss should not silently accept zero-token LM batches."""

        loss_fn = Loss.from_config(CausalLMLossConfig())
        pred = torch.randn(1, 1, 7)
        targets = torch.tensor([[3]])

        with pytest.raises(ValueError, match="zero valid target tokens"):
            loss_fn(pred, targets)


class CausalLMModelTest:
    """End-to-end tests for the causal language model."""

    def test_backward_reaches_embedding_and_head(self) -> None:
        """Next-token loss gradients should reach the embedding and the head."""

        vocab_size = 64
        model = _lm_model(vocab_size=vocab_size)
        loss_fn = Loss.from_config(CausalLMLossConfig())

        token_ids = torch.randint(0, vocab_size, (2, 8))
        logits = model(token_ids).pred
        loss = loss_fn(logits, token_ids).total
        loss.backward()

        assert torch.isfinite(loss)
        assert model.adapter.embed.weight.grad is not None
        assert model.head.projection.layers[-1].weight.grad is not None

    def test_is_causal(self) -> None:
        """A token's logits must not depend on any later input token."""

        vocab_size = 64
        model = _lm_model(vocab_size=vocab_size).eval()
        token_ids = torch.randint(0, vocab_size, (2, 9))

        with torch.no_grad():
            logits = model(token_ids).pred
            perturbed = token_ids.clone()
            perturbed[:, -1] = (perturbed[:, -1] + 1) % vocab_size
            logits_perturbed = model(perturbed).pred

        # Changing only the last token must leave every earlier position's
        # logits unchanged.
        assert torch.allclose(logits[:, :-1], logits_perturbed[:, :-1], atol=1e-5)
        # ... while the last position's logits do change.
        assert not torch.allclose(logits[:, -1], logits_perturbed[:, -1])
