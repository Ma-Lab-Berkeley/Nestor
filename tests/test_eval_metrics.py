"""Tests for evaluation metric helpers."""

from __future__ import annotations

import pytest
import torch

from nestor.eval.metrics import (
    accumulate_routing_stats,
    routing_stats_to_metrics,
)


def _router_aux(topk_indices, num_choices, valid=None, probs=None):
    """Build a minimal router aux entry as produced by the Nestor backbone."""
    topk = torch.as_tensor(topk_indices, dtype=torch.long)
    batch_size, seq_len = topk.shape[0], topk.shape[1]
    if valid is None:
        valid = torch.ones((batch_size, seq_len), dtype=torch.bool)
    aux = {
        # Only the last dim (num_choices) of `scores` is read.
        "scores": torch.zeros((batch_size, seq_len, num_choices)),
        "topk_indices": topk,
        "valid_token_mask": torch.as_tensor(valid, dtype=torch.bool),
    }
    if probs is not None:
        aux["probs"] = torch.as_tensor(probs, dtype=torch.float32)
    return aux


class RoutingStatsTest:
    """Tests for eval-time MoE / attention-group routing statistics."""

    def test_top1_group_fractions(self) -> None:
        """Per-choice fractions count valid tokens routed to each group."""
        # 4 tokens, top-1 over 2 groups: two to group 0, two to group 1.
        aux = {
            "blocks": [
                {
                    "group_router": _router_aux([[[0], [0], [1], [1]]], num_choices=2),
                    "layers": [],
                }
            ]
        }
        acc: dict = {}
        accumulate_routing_stats(aux, acc)
        metrics = routing_stats_to_metrics(acc)

        assert metrics["routing/block0/group/frac/c0"] == pytest.approx(0.5)
        assert metrics["routing/block0/group/frac/c1"] == pytest.approx(0.5)
        assert metrics["routing/block0/group/max_frac"] == pytest.approx(0.5)
        # Perfectly uniform -> balance 1.0.
        assert metrics["routing/block0/group/balance"] == pytest.approx(1.0)
        assert metrics["routing/group/mean_balance"] == pytest.approx(1.0)

    def test_topk_expert_fractions_sum_to_k(self) -> None:
        """Top-k expert fractions sum to k (each token uses k experts)."""
        # 2 tokens, top-2 over 3 experts.
        aux = {
            "blocks": [
                {
                    "group_router": _router_aux([[[0], [0]]], num_choices=1),
                    "layers": [
                        {
                            "expert_router": _router_aux(
                                [[[0, 1], [0, 2]]], num_choices=3
                            )
                        }
                    ],
                }
            ]
        }
        acc: dict = {}
        accumulate_routing_stats(aux, acc)
        metrics = routing_stats_to_metrics(acc)

        key = "routing/block0/layer0/expert"
        total = (
            metrics[f"{key}/frac/c0"]
            + metrics[f"{key}/frac/c1"]
            + metrics[f"{key}/frac/c2"]
        )
        assert total == pytest.approx(2.0)  # top-k == 2
        assert metrics[f"{key}/frac/c0"] == pytest.approx(1.0)  # both tokens
        # The single-choice group router is skipped (no information).
        assert "routing/block0/group/frac/c0" not in metrics

    def test_padding_tokens_excluded(self) -> None:
        """Padding (invalid) tokens do not contribute to routing fractions."""
        aux = {
            "blocks": [
                {
                    "group_router": _router_aux(
                        [[[0], [1], [1]]],
                        num_choices=2,
                        valid=[[True, True, False]],
                    ),
                    "layers": [],
                }
            ]
        }
        acc: dict = {}
        accumulate_routing_stats(aux, acc)
        metrics = routing_stats_to_metrics(acc)

        # Only the 2 valid tokens count: one to c0, one to c1.
        assert metrics["routing/block0/group/frac/c0"] == pytest.approx(0.5)
        assert metrics["routing/block0/group/frac/c1"] == pytest.approx(0.5)

    def test_accumulates_across_batches(self) -> None:
        """Counts accumulate across forwards before being normalized."""
        acc: dict = {}
        # Batch 1: both tokens -> group 0.
        accumulate_routing_stats(
            {"blocks": [{"group_router": _router_aux([[[0], [0]]], 2), "layers": []}]},
            acc,
        )
        # Batch 2: both tokens -> group 1.
        accumulate_routing_stats(
            {"blocks": [{"group_router": _router_aux([[[1], [1]]], 2), "layers": []}]},
            acc,
        )
        metrics = routing_stats_to_metrics(acc)
        assert metrics["routing/block0/group/frac/c0"] == pytest.approx(0.5)
        assert metrics["routing/block0/group/frac/c1"] == pytest.approx(0.5)

    def test_no_routing_is_noop(self) -> None:
        """A model without multi-choice routing yields no routing metrics."""
        acc: dict = {}
        accumulate_routing_stats({"hidden_states": torch.zeros(1)}, acc)
        accumulate_routing_stats(None, acc)
        assert routing_stats_to_metrics(acc) == {}

    def test_soft_routing_probability_logged(self) -> None:
        """Mean softmax routing probability per expert is logged and sums to 1."""
        # 2 tokens, 3 experts, top-1; expert router with explicit soft probs.
        probs = [[[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]]]
        aux = {
            "blocks": [
                {
                    "group_router": _router_aux([[[0], [0]]], num_choices=1),
                    "layers": [
                        {
                            "expert_router": _router_aux(
                                [[[0], [0]]], num_choices=3, probs=probs
                            )
                        }
                    ],
                }
            ]
        }
        acc: dict = {}
        accumulate_routing_stats(aux, acc)
        metrics = routing_stats_to_metrics(acc)

        key = "routing/block0/layer0/expert"
        # Mean over the 2 tokens.
        assert metrics[f"{key}/prob/c0"] == pytest.approx(0.6)
        assert metrics[f"{key}/prob/c1"] == pytest.approx(0.25)
        assert metrics[f"{key}/prob/c2"] == pytest.approx(0.15)
        assert metrics[f"{key}/max_prob"] == pytest.approx(0.6)
        # Soft probabilities form a distribution.
        prob_sum = (
            metrics[f"{key}/prob/c0"]
            + metrics[f"{key}/prob/c1"]
            + metrics[f"{key}/prob/c2"]
        )
        assert prob_sum == pytest.approx(1.0)
        assert metrics["routing/expert/mean_max_prob"] == pytest.approx(0.6)

    def test_prob_defaults_to_uniform_without_stored_probs(self) -> None:
        """With no stored `probs`, P_i falls back to softmax(scores)=uniform."""
        aux = {
            "blocks": [
                {
                    "group_router": _router_aux([[[0], [1]]], num_choices=2),
                    "layers": [],
                }
            ]
        }
        acc: dict = {}
        accumulate_routing_stats(aux, acc)
        metrics = routing_stats_to_metrics(acc)
        # scores are all zeros -> softmax is uniform 1/2 per choice.
        assert metrics["routing/block0/group/prob/c0"] == pytest.approx(0.5)
        assert metrics["routing/block0/group/prob/c1"] == pytest.approx(0.5)

    def test_distributed_padding_examples_are_excluded(self) -> None:
        """An example mask should exclude padded distributed samples."""
        aux = {
            "blocks": [
                {
                    "group_router": _router_aux(
                        [[[0], [0]], [[1], [1]]],
                        num_choices=2,
                    ),
                    "layers": [],
                }
            ]
        }
        acc: dict = {}

        accumulate_routing_stats(
            aux,
            acc,
            valid_example_mask=torch.tensor([True, False]),
        )
        metrics = routing_stats_to_metrics(acc)

        assert metrics["routing/block0/group/frac/c0"] == pytest.approx(1.0)
        assert metrics["routing/block0/group/frac/c1"] == pytest.approx(0.0)
