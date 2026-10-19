"""Base pipeline class with common setup functionality."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from nestor.configs.distributed import DistributedConfig
from nestor.configs.experiment import ExperimentConfig
from nestor.configs.jobs import JobConfig
from nestor.utils.distributed import init_distributed, is_main_process
from nestor.utils.logging import get_logger
from nestor.utils.seed import set_seed


class BasePipeline(ABC):
    """
    Base class for training and evaluation pipelines.

    Handles common setup: output directory, config saving, WandB, source snapshot.

    Attributes:
        job_config (JobConfig): Job configuration.
        experiment_config (ExperimentConfig): Experiment configuration.
        device (torch.device): Device to run on.
        run_dir (Path): Directory for this run's outputs.
        logger (logging.Logger): Logger instance.
    """

    def __init__(
        self,
        job_config: JobConfig,
        experiment_config: ExperimentConfig,
        device: torch.device,
    ):
        """
        Initialize the pipeline.

        Args:
            job_config (JobConfig): Job configuration.
            experiment_config (ExperimentConfig): Experiment configuration.
            device (torch.device): Device to run on.
        """
        self.job_config: JobConfig = job_config
        self.experiment_config: ExperimentConfig = experiment_config
        self.device: torch.device = device
        self.logger: logging.Logger = get_logger(f"nestor.{job_config.name}")

        # Initialize distributed if enabled
        if hasattr(job_config, "distributed"):
            assert isinstance(job_config.distributed, DistributedConfig)
            init_distributed(job_config.distributed)

        # Set seed
        if job_config.seed is not None:
            set_seed(job_config.seed)

        # Create run directory
        self.run_dir = self._create_run_dir()

    def _create_run_dir(self) -> Path:
        """
        Create the run directory for this job.

        Returns:
            Path: Path to the run directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}__{self.job_config.name}"
        run_dir = (
            self.experiment_config.output_root
            / self.experiment_config.name
            / run_name
        )

        if is_main_process():
            run_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Run directory: {run_dir}")

        return run_dir

    def _save_config(self) -> None:
        """Save the configuration to the run directory."""
        if not is_main_process():
            return

        config_path = self.run_dir / "config.json"

        # Convert dataclasses to dicts for JSON serialization
        config_dict = {
            "experiment": _dataclass_to_dict(self.experiment_config),
            "job": _dataclass_to_dict(self.job_config),
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        self.logger.info(f"Config saved to {config_path}")

    def _save_git_sha(self) -> None:
        """Save the current git commit hash to the run directory."""
        if not is_main_process():
            return

        import subprocess

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent.parent.parent,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return

        sha_path = self.run_dir / "git_sha.txt"
        sha_path.write_text(sha + "\n")
        self.logger.info(f"Git SHA: {sha}")

    def _init_wandb(
        self,
        project: str | None,
        entity: str | None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Initialize WandB logging.

        Args:
            project (str | None): WandB project name. None to disable.
            entity (str | None): WandB entity/team name.
            run_name (str | None): Explicit WandB run name.
            tags (list[str] | None): Tags for the run.
        """
        if not is_main_process():
            return

        if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "off"):
            self.logger.info("WandB disabled via WANDB_MODE env var")
            return

        if project is None:
            return

        try:
            import wandb

            init_kwargs: dict[str, Any] = {
                "project": project,
                "entity": entity,
                "config": _dataclass_to_dict(self.job_config),
                "tags": tags or [],
                "dir": str(self.run_dir),
            }
            if run_name is not None:
                init_kwargs["name"] = run_name

            # Continue the originating wandb run when resuming a checkpoint, so
            # metrics append to one curve instead of starting a disconnected run.
            resume_id = getattr(self, "_resume_wandb_id", None)
            if resume_id:
                init_kwargs["id"] = resume_id
                init_kwargs["resume"] = "allow"

            wandb.init(
                **init_kwargs,
            )
            assert wandb.run is not None
            self.logger.info(
                f"WandB initialized: {wandb.run.url}"
                + (f" (resumed run {resume_id})" if resume_id else "")
            )

            # Persist this run's id so a future resume continues the same run
            # (the id stays constant across a resume chain A->B->C).
            try:
                (self.run_dir / "wandb_run_id.txt").write_text(str(wandb.run.id))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Failed to persist wandb run id: {exc}")
        except ImportError:
            self.logger.warning(
                "wandb not installed, skipping WandB initialization"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")

    def _finish_wandb(self) -> None:
        """Finish WandB run."""
        if not is_main_process():
            return

        if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "off"):
            return

        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    def setup(self) -> None:
        """
        Perform common setup tasks.

        Called at the start of run(). Saves config, source snapshot, inits WandB.
        """
        self._save_config()
        self._save_git_sha()

    @abstractmethod
    def run(self) -> None:
        """Run the pipeline. Implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        job_config: JobConfig,
        experiment_config: ExperimentConfig,
        device: str | None = None,
    ) -> "BasePipeline":
        """
        Create a pipeline from configuration.

        Args:
            job_config (JobConfig): Job configuration.
            experiment_config (ExperimentConfig): Experiment configuration.
            device (str | None): Device string (e.g., "cuda", "cpu"). Auto-detected if None.

        Returns:
            BasePipeline: The initialized pipeline.
        """
        pass


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively convert dataclass objects to dictionaries.

    Args:
        obj: Object to convert.

    Returns:
        Converted object (dict for dataclasses, list for lists, etc.).
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()
        }
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj
