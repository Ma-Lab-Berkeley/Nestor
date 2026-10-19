"""Evaluation configuration."""

from __future__ import annotations

import dataclasses
from typing import Literal

from nestor.configs.data import DatasetConfig, DataLoaderConfig


@dataclasses.dataclass(kw_only=True)
class EvalConfig:
    """
    Base configuration for evaluations.

    Attributes:
        eval_type (str): Eval type identifier.
        name (str): Display name for the evaluation.
        dataset_config (DatasetConfig): Dataset configuration for evaluation.
        dataloader_config (DataLoaderConfig): DataLoader configuration.
    """

    eval_type: str = "base"
    name: str = "eval"
    dataset_config: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    dataloader_config: DataLoaderConfig = dataclasses.field(
        default_factory=DataLoaderConfig
    )


@dataclasses.dataclass(kw_only=True)
class LanguageModelingEvalConfig(EvalConfig):
    """
    Configuration for language-modeling evaluation (next-token prediction).

    Reports the mean next-token cross-entropy and its exponential, perplexity,
    over a held-out corpus.

    Attributes:
        eval_type (Literal["language_modeling"]): Eval type identifier.
        name (str): Display name for the evaluation.
        dataset_config (DatasetConfig): Dataset configuration for evaluation.
        dataloader_config (DataLoaderConfig): DataLoader configuration.
    """

    eval_type: Literal["language_modeling"] = "language_modeling"
    name: str = "language_modeling"


@dataclasses.dataclass(kw_only=True)
class EvalSuiteConfig:
    """
    Configuration for a suite of evaluations.

    Attributes:
        evals (list[EvalConfig]): List of evaluation configurations.
    """

    evals: list[EvalConfig] = dataclasses.field(default_factory=list)
