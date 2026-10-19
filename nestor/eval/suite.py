"""Evaluation suite for running multiple evaluations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from nestor.backbones import ModelOutput
from nestor.configs.evals import (
    EvalConfig,
    EvalSuiteConfig,
    LanguageModelingEvalConfig,
)
from nestor.constants import PADDING_TOKEN_VALUE
from nestor.data.datasets import Dataset
from nestor.eval.metrics import (
    accumulate_routing_stats,
    routing_stats_to_metrics,
)
from nestor.train.losses import CausalLMLoss
from nestor.utils.batches import split_supervised_batch
from nestor.utils.distributed import (
    get_padded_rank_indices,
    get_per_rank_batch_size,
    sum_tensor_across_ranks,
)

if TYPE_CHECKING:
    from nestor.models import Model


def _sum_routing_stats_across_ranks(
    accumulator: dict[str, dict[str, Any]],
) -> None:
    """Sum routing-stat tensors across distributed ranks in place.

    Args:
        accumulator (dict): Routing accumulator produced by
            :func:`accumulate_routing_stats`.
    """
    for entry in accumulator.values():
        for key in ("counts", "prob_sum", "total"):
            entry[key] = sum_tensor_across_ranks(entry[key])


class EvalSuite:
    """Suite of evaluations to run on a model."""

    def __init__(
        self,
        config: EvalSuiteConfig,
        model: "Model | nn.Module",
        device: torch.device,
        logger: logging.Logger,
    ):
        """Initialize the evaluation suite.

        Args:
            config (EvalSuiteConfig): Suite configuration.
            model (Model | nn.Module): Possibly wrapped model to evaluate.
            device (torch.device): Evaluation device.
            logger (logging.Logger): Logger instance.
        """

        self.config = config
        self.model = model
        self.device = device
        self.logger = logger
        self.evals = [
            BaseEval.from_config(
                config=eval_config,
                model=model,
                device=device,
                logger=logger,
            )
            for eval_config in config.evals
        ]

    def run(self, step: int | None = None) -> dict[str, dict[str, Any]]:
        """Run all evaluations in the suite.

        Args:
            step (int | None): Optional current training step.

        Returns:
            dict[str, dict[str, Any]]: Results keyed by evaluation name.
        """

        results: dict[str, dict[str, Any]] = {}
        step_info = f" at step {step}" if step is not None else ""
        for eval_runner in self.evals:
            self.logger.info(
                f"Running evaluation: {eval_runner.config.name}{step_info}"
            )
            eval_results = eval_runner.run()
            results[eval_runner.config.name] = eval_results
            for metric_name, value in eval_results.items():
                if isinstance(value, float):
                    self.logger.info(f"  {metric_name}: {value:.4f}")
                else:
                    self.logger.info(f"  {metric_name}: {value}")
        return results

    @classmethod
    def from_config(
        cls,
        config: EvalSuiteConfig,
        model: "Model | nn.Module",
        device: torch.device,
        logger: logging.Logger,
    ) -> "EvalSuite":
        """Create an evaluation suite from configuration.

        Args:
            config (EvalSuiteConfig): Suite configuration.
            model (Model | nn.Module): Possibly wrapped model to evaluate.
            device (torch.device): Evaluation device.
            logger (logging.Logger): Logger instance.

        Returns:
            EvalSuite: Configured evaluation suite.
        """

        return cls(config=config, model=model, device=device, logger=logger)


class BaseEval:
    """Base class for evaluation runners."""

    def __init__(
        self,
        config: EvalConfig,
        model: "Model | nn.Module",
        device: torch.device,
        logger: logging.Logger,
    ):
        """Initialize the evaluation runner.

        Args:
            config (EvalConfig): Evaluation configuration.
            model (Model | nn.Module): Possibly wrapped model to evaluate.
            device (torch.device): Evaluation device.
            logger (logging.Logger): Logger instance.
        """

        self.config = config
        self.model = model
        self.device = device
        self.logger = logger

    def run(self) -> dict[str, Any]:
        """Run the evaluation.

        Returns:
            dict[str, Any]: Evaluation results.
        """

        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: EvalConfig,
        model: "Model | nn.Module",
        device: torch.device,
        logger: logging.Logger,
    ) -> "BaseEval":
        """Build an evaluation runner from configuration.

        Args:
            config (EvalConfig): Evaluation configuration.
            model (Model | nn.Module): Possibly wrapped model to evaluate.
            device (torch.device): Evaluation device.
            logger (logging.Logger): Logger instance.

        Returns:
            BaseEval: Configured evaluation runner.
        """

        if isinstance(config, LanguageModelingEvalConfig):
            return LanguageModelingEval(
                config=config,
                model=model,
                device=device,
                logger=logger,
            )
        raise ValueError(f"Unknown eval type: {config.eval_type}")


class LanguageModelingEval(BaseEval):
    """Next-token prediction evaluation reporting cross-entropy and perplexity."""

    config: LanguageModelingEvalConfig

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        """Run language-modeling evaluation.

        Returns:
            dict[str, Any]: Mean next-token ``cross_entropy`` and ``perplexity``.
        """

        self.model.eval()
        dataset = Dataset.from_config(self.config.dataset_config)
        indices, remaining_valid_examples = get_padded_rank_indices(len(dataset))
        if indices:
            dataset = Subset(dataset, indices)
        dataloader = DataLoader(
            dataset,
            batch_size=get_per_rank_batch_size(
                self.config.dataloader_config.batch_size
            ),
            shuffle=False,
            num_workers=self.config.dataloader_config.num_workers,
            pin_memory=self.config.dataloader_config.pin_memory,
        )

        total_loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        total_tokens = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        # Per-choice token routing distribution (MoE experts + attention groups),
        # accumulated over the whole eval set. No-op for non-routed models.
        routing_acc: dict[str, dict[str, Any]] = {}

        for batch in dataloader:
            inputs, targets = split_supervised_batch(batch)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            output: ModelOutput = self.model(inputs)
            batch_size = targets.shape[0]
            num_valid_examples = min(batch_size, remaining_valid_examples)
            valid_example_mask = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=targets.device,
            )
            if num_valid_examples > 0:
                valid_example_mask[:num_valid_examples] = True
            remaining_valid_examples -= num_valid_examples

            target_tokens = targets[:, 1:]
            valid_token_mask = (
                target_tokens != PADDING_TOKEN_VALUE
            ) & valid_example_mask.unsqueeze(-1)
            batch_loss, num_tokens = CausalLMLoss.sum_loss_and_tokens(
                output.pred,
                targets,
                valid_token_mask=valid_token_mask,
            )
            accumulate_routing_stats(
                output.aux,
                routing_acc,
                valid_example_mask=valid_example_mask,
            )
            if num_tokens == 0:
                continue

            total_loss += batch_loss
            total_tokens += num_tokens

        total_loss = sum_tensor_across_ranks(total_loss)
        total_tokens = sum_tensor_across_ranks(total_tokens)
        if total_tokens.item() == 0:
            raise ValueError("Language-modeling evaluation received zero valid tokens.")

        cross_entropy = total_loss / total_tokens
        results = {
            "cross_entropy": float(cross_entropy.item()),
            "perplexity": float(cross_entropy.exp().item()),
        }
        _sum_routing_stats_across_ranks(routing_acc)
        results.update(routing_stats_to_metrics(routing_acc))
        return results
