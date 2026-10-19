"""Job configuration for training and evaluation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

from nestor.configs.data import DatasetConfig, DataLoaderConfig
from nestor.configs.distributed import DistributedConfig
from nestor.configs.evals import EvalSuiteConfig
from nestor.configs.losses import LossConfig
from nestor.configs.optim import OptimizerConfig
from nestor.configs.sched import SchedulerConfig
from nestor.models import ModelConfig


@dataclasses.dataclass(kw_only=True)
class JobConfig:
    """
    Base configuration for jobs.

    Attributes:
        job_type (str): Job type identifier.
        name (str): Job name for logging and output directories.
        seed (int | None): Random seed for reproducibility.
    """

    job_type: str = "base"
    name: str = "job"
    seed: int | None = None


@dataclasses.dataclass(kw_only=True)
class TrainJobConfig(JobConfig):
    """
    Configuration for a training job.

    Attributes:
        job_type (Literal["train"]): Job type identifier.
        name (str): Job name for logging and output directories.

        # Model
        model (ModelConfig): Top-level model configuration.

        # Data
        train_dataset (DatasetConfig): Training dataset configuration.
        train_dataloader (DataLoaderConfig): Training dataloader configuration.

        # Optimization
        optimizer (OptimizerConfig): Optimizer configuration.
        scheduler (SchedulerConfig | None): LR scheduler configuration.
        max_steps (int): Maximum number of training steps.
        grad_accum_steps (int): Gradient accumulation steps.
        max_grad_norm (float | None): Max gradient norm for clipping (None = no clipping).
        use_amp (bool): Whether to use automatic mixed precision.

        # Loss
        loss (LossConfig): Loss function configuration.

        # Evaluation
        eval_suite (EvalSuiteConfig | None): Evaluation suite configuration.
        eval_frequency (int): Run eval suite every N steps.

        # Checkpointing and logging
        checkpoint_dir (Path | None): Directory for checkpoints (relative to run dir).
        checkpoint_frequency (int): Save checkpoint every N steps.
        log_frequency (int): Log metrics every N steps.

        # Distributed
        distributed (DistributedConfig): Distributed training configuration.

        # WandB
        wandb_project (str | None): WandB project name (None = disabled).
        wandb_entity (str | None): WandB entity/team name.
        wandb_run_name (str | None): Explicit WandB run name (None = let WandB choose).
        wandb_tags (list[str]): Tags for the WandB run.

        # Resume
        resume_model_checkpoint (Path | None): Path to model checkpoint to resume from.

        # Reproducibility
        seed (int): Random seed.
    """

    job_type: Literal["train"] = "train"
    name: str = "train"

    # Model
    model: ModelConfig

    # Data
    train_dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    train_dataloader: DataLoaderConfig = dataclasses.field(
        default_factory=DataLoaderConfig
    )

    # Optimization
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    max_steps: int = 100000
    grad_accum_steps: int = 1
    max_grad_norm: float | None = 1.0
    use_amp: bool = False

    # Model EMA: exponential moving average of weights for eval/deployment.
    # None disables it; e.g. 0.9999 enables. Updated after
    # every optimizer step; eval runs against the EMA weights when set.
    ema_decay: float | None = None

    # Loss
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)

    # Evaluation
    eval_suite: EvalSuiteConfig | None = None
    eval_frequency: int = 1000

    # Checkpointing and logging
    checkpoint_dir: Path | None = Path("checkpoints")
    checkpoint_frequency: int = 5000
    log_frequency: int = 100

    # Distributed
    distributed: DistributedConfig = dataclasses.field(
        default_factory=DistributedConfig
    )

    # WandB
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = dataclasses.field(default_factory=list)

    # Resume
    resume_model_checkpoint: Path | None = None
    resume_strict: bool = True

    # Eval job submission: list of (slurm_script_path, step_frequency) pairs.
    # At each checkpoint, any script whose frequency divides the current step
    # is submitted via ``sbatch script checkpoint_path``.
    eval_submit_scripts: list[tuple[Path, int]] | None = None

    # Reproducibility
    seed: int = 42


@dataclasses.dataclass(kw_only=True)
class EvalJobConfig(JobConfig):
    """
    Configuration for an evaluation job.

    Attributes:
        job_type (Literal["eval"]): Job type identifier.
        name (str): Job name for logging and output directories.

        # Model
        model (ModelConfig): Top-level model configuration.
        model_checkpoint_path (Path): Path to model checkpoint.

        # Evaluation
        eval_suite (EvalSuiteConfig): Evaluation suite configuration.

        # Distributed
        distributed (DistributedConfig): Distributed training configuration.

        # Output
        output_format (Literal["json", "markdown", "both"]): Format for eval results.

        # Reproducibility
        seed (int): Random seed.
    """

    job_type: Literal["eval"] = "eval"
    name: str = "eval"

    # Model
    model: ModelConfig
    model_checkpoint_path: Path

    # Evaluation
    eval_suite: EvalSuiteConfig = dataclasses.field(default_factory=EvalSuiteConfig)

    # Distributed
    distributed: DistributedConfig = dataclasses.field(
        default_factory=DistributedConfig
    )

    # Output
    output_format: Literal["json", "markdown", "both"] = "both"

    # Reproducibility
    seed: int = 42

