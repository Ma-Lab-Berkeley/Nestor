"""Learning rate scheduler configuration."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class SchedulerConfig:
    """
    Base configuration for learning rate schedulers.

    Attributes:
        scheduler_type (str): Scheduler type identifier.
    """

    scheduler_type: str = "base"


@dataclasses.dataclass(kw_only=True)
class CosineAnnealingConfig(SchedulerConfig):
    """
    Configuration for cosine annealing LR scheduler.

    Attributes:
        scheduler_type (Literal["cosine"]): Scheduler type identifier.
        T_max (int): Maximum number of iterations (typically total training steps).
        eta_min (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps with linear LR increase.
    """

    scheduler_type: Literal["cosine"] = "cosine"
    T_max: int = 100000
    eta_min: float = 1e-6
    warmup_steps: int = 1000


@dataclasses.dataclass(kw_only=True)
class WSDConfig(SchedulerConfig):
    """Warmup-Stable-Decay LR schedule (a.k.a. WSD / trapezoid).

    Linear warmup to the base LR, a long constant (stable) phase, then a short
    decay to ``eta_min`` over the final ``decay_steps``. The decay shape is
    ``1 - sqrt(progress)`` (MiniCPM/WSD), which empirically matches or beats
    cosine while decaying harder at the very end -- the behaviour of the
    variable-width-transformers reference's ``power`` decay. All optimizer
    param-groups follow the same multiplier, so muP per-group base LRs are
    preserved.

    Attributes:
        scheduler_type (Literal["wsd"]): Scheduler type identifier.
        total_steps (int): Total optimizer steps (overridden by the trainer's
            ``num_training_steps`` when available).
        warmup_steps (int): Linear warmup steps.
        decay_steps (int): Length of the final decay phase.
        eta_min (float): Final LR floor (absolute, like CosineAnnealing).
        decay_shape (Literal["one_minus_sqrt", "linear", "cosine"]): Decay curve.
    """

    scheduler_type: Literal["wsd"] = "wsd"
    total_steps: int = 100000
    warmup_steps: int = 2000
    decay_steps: int = 20000
    eta_min: float = 1e-5
    decay_shape: Literal["one_minus_sqrt", "linear", "cosine"] = "one_minus_sqrt"


@dataclasses.dataclass(kw_only=True)
class PowerLawConfig(SchedulerConfig):
    """Power-law schedule used by the variable-width-transformers reference.

    The post-warmup absolute learning rate is ``a * (step * c) ** b``, capped
    at the optimizer base rate. Warmup is linear to the power curve's value at
    the boundary, matching the reference PowerScheduler.

    Attributes:
        scheduler_type (Literal["power"]): Scheduler type identifier.
        warmup_steps (int): Number of linear warmup optimizer steps.
        base_lr (float): Reference optimizer base learning rate.
        a (float): Power-law amplitude.
        b (float): Power-law exponent.
        c (float): Tokens represented by one optimizer step.
    """

    scheduler_type: Literal["power"] = "power"
    warmup_steps: int = 2000
    base_lr: float = 0.01
    a: float = 4096.0
    b: float = -0.566
    c: float = 4_194_304.0


@dataclasses.dataclass(kw_only=True)
class StepLRConfig(SchedulerConfig):
    """
    Configuration for step LR scheduler.

    Attributes:
        scheduler_type (Literal["step"]): Scheduler type identifier.
        step_size (int): Period of learning rate decay (in epochs or steps).
        gamma (float): Multiplicative factor for LR decay.
        warmup_steps (int): Number of warmup steps with linear LR increase.
    """

    scheduler_type: Literal["step"] = "step"
    step_size: int = 30
    gamma: float = 0.1
    warmup_steps: int = 0
