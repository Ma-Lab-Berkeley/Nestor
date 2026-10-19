"""Experiment configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from nestor.configs.jobs import JobConfig


@dataclasses.dataclass(kw_only=True)
class ExperimentConfig:
    """
    Configuration for an experiment consisting of one or more jobs.

    Attributes:
        name (str): Experiment name (typically set from the function name).
        jobs (list[JobConfig]): List of job configurations to run.
        output_root (Path): Root directory for all experiment outputs.
        save_source_snapshot (bool): Whether to save a zip of the source code.
    """

    name: str = "experiment"
    jobs: list[JobConfig] = dataclasses.field(default_factory=list)
    output_root: Path = dataclasses.field(default_factory=lambda: Path("./outputs"))
    save_source_snapshot: bool = True
