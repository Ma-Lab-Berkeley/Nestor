"""Loss function configuration."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class LossConfig:
    """
    Base configuration for loss functions.

    Attributes:
        loss_type (str): Loss type identifier.
    """

    loss_type: str = "base"


@dataclasses.dataclass(kw_only=True)
class CausalLMLossConfig(LossConfig):
    """
    Configuration for the causal language-modeling (next-token) loss.

    The loss shifts the per-token logits and targets by one position so that
    position ``t`` is trained to predict the token at ``t + 1``.

    Attributes:
        loss_type (Literal["causal_lm"]): Loss type identifier.
        label_smoothing (float): Label smoothing factor (0.0 = no smoothing).
    """

    loss_type: Literal["causal_lm"] = "causal_lm"
    label_smoothing: float = 0.0


@dataclasses.dataclass(kw_only=True)
class RouterBalanceLossConfig(LossConfig):
    """
    Configuration for router load balancing loss (auxiliary loss for MoE).

    Attributes:
        loss_type (Literal["router_balance"]): Loss type identifier.
        balance_weight (float): Weight for the load balancing term.
        diversity_weight (float): Weight for the diversity term.
    """

    loss_type: Literal["router_balance"] = "router_balance"
    balance_weight: float = 0.01
    diversity_weight: float = 0.01


@dataclasses.dataclass(kw_only=True)
class ReferenceMoEAuxLossConfig(LossConfig):
    """Switch plus router z-loss, as in the reference implementation.

    Attributes:
        loss_type (Literal["reference_moe_aux"]): Loss type identifier.
        coefficient (float): Coefficient applied after summing all layer terms.
        z_loss_weight (float): Router z-loss weight inside each layer term.
    """

    loss_type: Literal["reference_moe_aux"] = "reference_moe_aux"
    coefficient: float = 0.01
    z_loss_weight: float = 0.1


@dataclasses.dataclass(kw_only=True)
class CompositeLossConfig(LossConfig):
    """
    Configuration for composite loss combining multiple loss functions.

    Attributes:
        loss_type (Literal["composite"]): Loss type identifier.
        losses (list[LossConfig]): List of loss configurations.
        weights (list[float]): Weight for each loss component.
    """

    loss_type: Literal["composite"] = "composite"
    losses: list[LossConfig] = dataclasses.field(default_factory=list)
    weights: list[float] = dataclasses.field(default_factory=list)
