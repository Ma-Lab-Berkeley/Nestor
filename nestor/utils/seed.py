"""Seed utilities for reproducibility."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Performance-oriented backend defaults for training. NESTOR_DETERMINISTIC=1
    # restores the slow bit-reproducible path. Otherwise: cudnn autotuning on,
    # and TF32 ("high") for the fp32 matmuls outside the bf16 autocast region
    # (router/muP/loss) -- large speedup, negligible accuracy cost.
    deterministic = os.environ.get("NESTOR_DETERMINISTIC") == "1"
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.set_float32_matmul_precision("highest" if deterministic else "high")
