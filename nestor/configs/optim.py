"""Optimizer configuration."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class OptimizerConfig:
    """
    Base configuration for optimizers.

    Attributes:
        optimizer_type (str): Optimizer type identifier.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
    """

    optimizer_type: str = "base"
    lr: float = 1e-4
    weight_decay: float = 0.0


@dataclasses.dataclass(kw_only=True)
class AdamConfig(OptimizerConfig):
    """Configuration for Adam optimizer (no decoupled weight decay)."""

    optimizer_type: Literal["adam"] = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclasses.dataclass(kw_only=True)
class AdamWConfig(OptimizerConfig):
    """
    Configuration for AdamW optimizer.

    Attributes:
        optimizer_type (Literal["adamw"]): Optimizer type identifier.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        betas (tuple[float, float]): Coefficients for computing running averages.
        eps (float): Term added to denominator for numerical stability.
    """

    optimizer_type: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclasses.dataclass(kw_only=True)
class MuonAdamWConfig(OptimizerConfig):
    """
    Configuration for the combined Muon + AdamW optimizer.

    Muon trains the in-stream and between-block matrices (tagged by the scaling
    blocks); AdamW trains embeddings, score heads, norm gains, and biases. The
    two base rates are shape-independent: all per-parameter shape dependence is
    applied from the scaling tags when the optimizer is built.

    Attributes:
        optimizer_type (Literal["muon_adamw"]): Optimizer type identifier.
        lr (float): Mirror of ``lr_adamw`` kept for the generic interface.
        lr_muon (float): Base Muon learning rate ``eta_M``.
        lr_adamw (float): Base AdamW learning rate ``eta_A``.
        weight_decay (float): Decoupled weight decay (skipped for norms/biases).
        betas (tuple[float, float]): AdamW moment decay rates.
        eps (float): AdamW denominator epsilon.
        muon_momentum (float): Muon momentum.
        muon_nesterov (bool): Whether Muon uses Nesterov momentum.
        muon_ns_steps (int): Newton--Schulz iterations per Muon step.
    """

    optimizer_type: Literal["muon_adamw"] = "muon_adamw"
    lr: float = 1e-3
    lr_muon: float = 0.02
    lr_adamw: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5


@dataclasses.dataclass(kw_only=True)
class SGDConfig(OptimizerConfig):
    """
    Configuration for SGD optimizer.

    Attributes:
        optimizer_type (Literal["sgd"]): Optimizer type identifier.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        momentum (float): Momentum factor.
        nesterov (bool): Whether to use Nesterov momentum.
    """

    optimizer_type: Literal["sgd"] = "sgd"
    lr: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = True
