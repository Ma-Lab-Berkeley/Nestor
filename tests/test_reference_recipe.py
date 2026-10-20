"""Equation-level tests for the paper scheduler, MoE loss, and token remap."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from nestor.configs.data import DclmDatasetConfig
from nestor.configs.losses import ReferenceMoEAuxLossConfig
from nestor.configs.sched import PowerLawConfig
from nestor.data.datasets import DclmTokenDataset
from nestor.train.losses import Loss, ReferenceMoEAuxLoss
from nestor.train.optim import build_scheduler


class PowerLawSchedulerTest:
    """Checks against the reference PowerScheduler formula."""

    def test_warmup_and_decay_values(self) -> None:
        """LR multipliers should equal the reference closed-form equations."""

        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        config = PowerLawConfig()
        scheduler = build_scheduler(config, optimizer)
        assert scheduler is not None

        warmup_peak = min(
            1.0,
            (config.a / config.base_lr)
            * ((config.warmup_steps * config.c) ** config.b),
        )
        assert math.isclose(scheduler.lr_lambdas[0](1000), warmup_peak / 2)
        expected_final = min(
            1.0,
            (config.a / config.base_lr) * ((25_000 * config.c) ** config.b),
        )
        assert math.isclose(scheduler.lr_lambdas[0](25_000), expected_final)


class ReferenceMoEAuxLossTest:
    """Checks the summed switch plus z auxiliary objective."""

    def test_matches_manual_two_layer_equation(self) -> None:
        """The implementation should sum layer terms before coefficient scaling."""

        logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        topk = torch.tensor([[[0], [1]]])
        router_aux = {
            "scores": logits,
            "topk_indices": topk,
            "valid_token_mask": torch.ones((1, 2), dtype=torch.bool),
        }
        aux = {
            "blocks": [
                {
                    "layers": [
                        {"expert_router": router_aux},
                        {"expert_router": router_aux},
                    ]
                }
            ]
        }
        loss = Loss.from_config(ReferenceMoEAuxLossConfig())
        assert isinstance(loss, ReferenceMoEAuxLoss)
        output = loss(torch.zeros(()), torch.zeros(()), aux)

        switch = torch.tensor(1.0)
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        expected = 0.01 * 2 * (switch + 0.1 * z_loss)
        assert torch.allclose(output.total, expected)


class DclmTokenRemapTest:
    """Checks vectorized alignment of local and reference special-token ids."""

    def test_window_remaps_eod_without_mutating_other_ids(
        self, tmp_path: Path
    ) -> None:
        """Only configured ids should change in the returned training window."""

        data_path = tmp_path / "tokens.bin"
        tokens = np.array([1, 100_257, 2, 100_257], dtype=np.uint32)
        tokens.tofile(data_path)
        meta_path = data_path.with_suffix(".bin.meta.json")
        meta_path.write_text(
            json.dumps({"token_dtype": "uint32", "num_tokens": 4})
        )
        dataset = DclmTokenDataset(
            DclmDatasetConfig(
                data_path=data_path,
                block_size=4,
                token_remap={100_257: 100_261},
            )
        )
        inputs, targets = dataset[0]
        expected = torch.tensor([1, 100_261, 2, 100_261])
        assert torch.equal(inputs, expected)
        assert torch.equal(targets, expected)
