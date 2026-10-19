"""Logging utilities."""

from __future__ import annotations

import logging
import sys
from typing import Any

from nestor.utils.distributed import is_main_process


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with console output.

    Only the main process (rank 0) will emit logs in distributed mode.

    Args:
        name (str): Logger name.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: dict[str, Any],
    step: int,
    prefix: str = "",
    wandb_log: bool = True,
) -> None:
    """
    Log metrics to console and optionally to WandB.

    Only logs on the main process in distributed mode.

    Args:
        logger (logging.Logger): Logger instance.
        metrics (dict[str, Any]): Dictionary of metric names and values.
        step (int): Current step number.
        prefix (str): Prefix for metric names (e.g., "train/", "val/").
        wandb_log (bool): Whether to log to WandB.
    """
    if not is_main_process():
        return

    # Format metrics for console
    metric_strs = [
        f"{prefix}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix}{k}: {v}"
        for k, v in metrics.items()
    ]
    logger.info(f"Step {step} | " + " | ".join(metric_strs))

    # Log to WandB if available and enabled
    if wandb_log:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
        except ImportError:
            pass
