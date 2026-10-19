"""Evaluation pipeline."""

from __future__ import annotations

import json
from typing import cast

import torch

from nestor.configs.experiment import ExperimentConfig
from nestor.configs.jobs import EvalJobConfig, JobConfig
from nestor.models import Model
from nestor.pipelines.base import BasePipeline
from nestor.utils.distributed import is_main_process


class EvalPipeline(BasePipeline):
    """Pipeline for evaluation jobs."""

    job_config: EvalJobConfig

    def __init__(
        self,
        job_config: EvalJobConfig,
        experiment_config: ExperimentConfig,
        device: torch.device,
    ):
        """Initialize the evaluation pipeline.

        Args:
            job_config (EvalJobConfig): Evaluation job configuration.
            experiment_config (ExperimentConfig): Experiment configuration.
            device (torch.device): Device for this rank.
        """

        super().__init__(job_config, experiment_config, device)
        self.model: Model | None = None

    def setup(self) -> None:
        """Set up the evaluation pipeline."""

        super().setup()
        self._build_model()
        self._load_checkpoint()

        self.logger.info("Evaluation pipeline setup complete")
        assert self.model is not None
        self.logger.info(
            f"  Model parameters: {sum(parameter.numel() for parameter in self.model.parameters()):,}"
        )

    def _build_model(self) -> None:
        """Build the top-level model."""

        self.model = Model.from_config(self.job_config.model).to(self.device)

    def _load_checkpoint(self) -> None:
        """Load model weights from checkpoint."""

        from nestor.train.checkpoint import load_checkpoint

        assert self.model is not None
        model_path = self.job_config.model_checkpoint_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.logger.info(f"Loading model checkpoint from {model_path}")
        load_checkpoint(model_path, model=self.model, device=self.device)

    def run(self) -> None:
        """Run the evaluation pipeline."""

        from nestor.eval.suite import EvalSuite

        self.setup()
        assert self.model is not None
        self.model.eval()

        eval_suite = EvalSuite.from_config(
            config=self.job_config.eval_suite,
            model=self.model,
            device=self.device,
            logger=self.logger,
        )
        results = eval_suite.run()
        self._save_results(cast(dict[str, object], results))
        self.logger.info("Evaluation complete!")

    def _save_results(self, results: dict[str, object]) -> None:
        """Save evaluation results to disk.

        Args:
            results (dict[str, object]): Evaluation results.
        """

        if not is_main_process():
            return

        if self.job_config.output_format in ("json", "both"):
            json_path = self.run_dir / "eval_results.json"
            with open(json_path, "w") as file:
                json.dump(results, file, indent=2)
            self.logger.info(f"Results saved to {json_path}")

        if self.job_config.output_format in ("markdown", "both"):
            md_path = self.run_dir / "eval_results.md"
            with open(md_path, "w") as file:
                file.write("# Evaluation Results\n\n")
                for eval_name, eval_results in results.items():
                    file.write(f"## {eval_name}\n\n")
                    if isinstance(eval_results, dict):
                        for metric_name, value in eval_results.items():
                            if isinstance(value, float):
                                file.write(
                                    f"- **{metric_name}**: {value:.4f}\n"
                                )
                            else:
                                file.write(f"- **{metric_name}**: {value}\n")
                    else:
                        file.write(f"{eval_results}\n")
                    file.write("\n")
            self.logger.info(f"Results saved to {md_path}")

    @classmethod
    def from_config(
        cls,
        job_config: JobConfig,
        experiment_config: ExperimentConfig,
        device: str | None = None,
    ) -> "EvalPipeline":
        """Create an evaluation pipeline from configuration.

        Args:
            job_config (JobConfig): Evaluation job configuration.
            experiment_config (ExperimentConfig): Experiment configuration.
            device (str | None): Optional device string.

        Returns:
            EvalPipeline: Initialized pipeline.
        """

        assert isinstance(job_config, EvalJobConfig)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            job_config=job_config,
            experiment_config=experiment_config,
            device=torch.device(device),
        )
