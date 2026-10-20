"""Tests for trainer accumulation helpers."""

from __future__ import annotations

import torch

from nestor.train.trainer import Trainer


class TrainerAccumulationHelpersTest:
    """Tests for gradient-accumulation helper logic in ``Trainer``."""

    def test_merge_model_aux_batches_concatenates_router_updates(self) -> None:
        """Merged aux should include all microbatches for router bias updates."""
        aux_batch_1 = {
            "blocks": [
                {
                    "group_router": {
                        "topk_indices": torch.tensor([[[0], [1]]], dtype=torch.long),
                        "valid_token_mask": torch.tensor([[True, True]]),
                        "scores": torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
                    },
                    "layers": [
                        {
                            "expert_router": {
                                "topk_indices": torch.tensor(
                                    [[[0], [1]]], dtype=torch.long
                                ),
                                "valid_token_mask": torch.tensor([[True, True]]),
                                "scores": torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
                            }
                        }
                    ],
                }
            ]
        }
        aux_batch_2 = {
            "blocks": [
                {
                    "group_router": {
                        "topk_indices": torch.tensor([[[1], [1]]], dtype=torch.long),
                        "valid_token_mask": torch.tensor([[True, False]]),
                        "scores": torch.tensor([[[0.0, 1.0], [0.2, 0.8]]]),
                    },
                    "layers": [
                        {
                            "expert_router": {
                                "topk_indices": torch.tensor(
                                    [[[1], [1]]], dtype=torch.long
                                ),
                                "valid_token_mask": torch.tensor([[True, False]]),
                                "scores": torch.tensor([[[0.0, 1.0], [0.2, 0.8]]]),
                            }
                        }
                    ],
                }
            ]
        }

        merged = Trainer._merge_model_aux_batches([aux_batch_1, aux_batch_2])

        expected_topk = torch.cat(
            [
                aux_batch_1["blocks"][0]["group_router"]["topk_indices"],
                aux_batch_2["blocks"][0]["group_router"]["topk_indices"],
            ],
            dim=0,
        )
        expected_mask = torch.cat(
            [
                aux_batch_1["blocks"][0]["group_router"]["valid_token_mask"],
                aux_batch_2["blocks"][0]["group_router"]["valid_token_mask"],
            ],
            dim=0,
        )

        assert torch.equal(
            merged["blocks"][0]["group_router"]["topk_indices"],
            expected_topk,
        )
        assert torch.equal(
            merged["blocks"][0]["group_router"]["valid_token_mask"],
            expected_mask,
        )
        assert torch.equal(
            merged["blocks"][0]["layers"][0]["expert_router"]["topk_indices"],
            expected_topk,
        )

    def test_build_train_metrics_averages_component_sums(self) -> None:
        """Training metrics should be averaged over accumulation steps."""
        metrics = Trainer._build_train_metrics(
            accum_loss=6.0,
            accum_steps=2,
            accum_component_sums={
                "cross_entropy": 4.0,
                "router_balance": 2.0,
            },
        )

        assert metrics["loss"] == 3.0
        assert metrics["cross_entropy"] == 2.0
        assert metrics["router_balance"] == 1.0
