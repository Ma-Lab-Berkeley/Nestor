"""Forward FLOPs per token of the registered models at any context length.

The accounting behind the equal-FLOP settings of the length ladder and the
compute reference of the domain-adaptation experiments. Per layer and token:

* attention projections of a column layer: ``8 d w k`` (q / k / v / out over
  the ``k`` selected columns, each of width ``w = heads x head_dim``);
* the causal attention core: each of the ``G`` columns runs over its
  ``S k / G`` member tokens, ``2 (S k / G)^2 w`` per column per sequence
  (``4 ...`` without the causal halving);
* the MLP: ``6 d x active_width`` for a gated (SwiGLU) MLP, ``4 d x active_width``
  for a plain one, where the active width is ``top_k x expert_width`` for a
  bank (the dense MLP is the ``E = 1`` bank) and, for Nestor's hybrid pool,
  ``k_p w_public + k k_q w_private`` under the width contract.

Routers, the depth reads, the norms and the LM head are excluded. Balanced
routing is assumed: every column sees exactly ``S k / G`` tokens.

With one attention group the formulas reduce to the dense transformer's, so the
two baselines of a rung cost the same to within the rounding of the MoE's
expert widths (``round8(4d / 3) x 3`` against ``4d``: exact at 100M and 500M,
0.3 % at 200M, 0.2 % at 1B), and Nestor's share of attention FLOPs falls with
the context length (``k / G`` of the dense core per block).
"""

from __future__ import annotations

import dataclasses

from nestor.backbones import NestorConfig
from nestor.backbones.expert_pool_dispatch import (
    private_expert_hidden_dim,
    public_expert_hidden_dim,
)
from nestor.experiments.pretrain import MOE_EXPERTS, SCALES, baseline_backbone, nestor_backbone

FAMILIES: tuple[str, ...] = ("dense", "moe", "nestor")
"""tuple[str, ...]: The three model families of the scaling figure."""

GATED_ACTIVATIONS: tuple[str, ...] = ("swiglu", "geglu", "glu", "reglu")
"""tuple[str, ...]: MLP activations with a third (gate) matrix."""


@dataclasses.dataclass(frozen=True)
class BodyFlops:
    """Forward FLOPs of one sequence, split by component.

    Attributes:
        projections (float): Attention q / k / v / out projections.
        core (float): Attention score / value core.
        mlp (float): Expert / MLP matrices.
    """

    projections: float
    core: float
    mlp: float

    @property
    def total(self) -> float:
        """float: Sum of the three components."""

        return self.projections + self.core + self.mlp


def split_experiment_name(experiment: str) -> tuple[str, str]:
    """``"<family>_<scale>"`` -> ``(family, scale)``.

    Args:
        experiment (str): A registered experiment name (``nestor_200m``, ``dense_1b``, ...).

    Returns:
        tuple[str, str]: The family and the rung.

    Raises:
        ValueError: When the name is not one of the twelve registered cells.
    """

    family, _, scale = experiment.rpartition("_")
    if family not in FAMILIES or scale not in SCALES:
        raise ValueError(
            f"{experiment!r} is not a registered cell: expected <family>_<scale> with "
            f"family in {FAMILIES} and scale in {tuple(SCALES)}"
        )
    return family, scale


def experiment_backbone(experiment: str) -> NestorConfig:
    """The backbone configuration of a registered cell, built from its family and rung.

    Built directly from the registry's builders, so neither the ``DCLM_*``
    environment nor the data path matter.

    Args:
        experiment (str): A registered experiment name.

    Returns:
        NestorConfig: The cell's backbone.
    """

    family, scale = split_experiment_name(experiment)
    if family == "nestor":
        return nestor_backbone(SCALES[scale])
    return baseline_backbone(SCALES[scale], experts=1 if family == "dense" else MOE_EXPERTS)


def attention_flops(
    *, seq_len: int, dim: int, groups: int, top_k: int, width: int, causal: bool = True
) -> tuple[float, float]:
    """Projection and core FLOPs of one column-attention layer over one sequence.

    Args:
        seq_len (int): Sequence length ``S``.
        dim (int): Model width ``d``.
        groups (int): Columns ``G`` (1 for dense attention).
        top_k (int): Columns per token ``k``.
        width (int): Per-column width ``heads x head_dim``.
        causal (bool): Halve the core (the models are causal).

    Returns:
        tuple[float, float]: ``(projections, core)``.
    """

    projections = 8.0 * seq_len * dim * width * top_k
    members = seq_len * top_k / groups
    core = groups * (2.0 if causal else 4.0) * members**2 * width
    return projections, core


def mlp_flops(*, seq_len: int, dim: int, active_width: float, gated: bool) -> float:
    """FLOPs of one MLP / expert layer at ``active_width`` per token over one sequence.

    Args:
        seq_len (int): Sequence length.
        dim (int): Model width.
        active_width (float): Sum of the intermediate widths a token runs through.
        gated (bool): Three matrices (SwiGLU) instead of two.

    Returns:
        float: Forward FLOPs of the layer.
    """

    return 2.0 * seq_len * dim * active_width * (3 if gated else 2)


def block_active_mlp_width(block: dict, dim: int) -> tuple[float, bool]:
    """Active MLP width per token of one block, and whether the MLP is gated.

    Args:
        block (dict): One entry of ``NestorConfig.to_dict()["block_configs"]``.
        dim (int): Model width.

    Returns:
        tuple[float, bool]: ``(active_width, gated)``.
    """

    gated = str(block.get("block_mlp_activation", "swiglu")).lower() in GATED_ACTIVATIONS
    private_experts = int(block.get("expert_pool_private_experts") or 0)
    public_experts = int(block.get("expert_pool_public_experts") or 0)
    if private_experts or public_experts:
        # The hybrid public + private pool under the width contract.
        top_k = int(block["num_groups_per_token"])
        public_top_k = int(block.get("expert_pool_public_top_k") or 0)
        private_top_k = int(block.get("expert_pool_private_top_k") or 0)
        width = 0.0
        if public_experts:
            width += public_top_k * public_expert_hidden_dim(
                dim, public_top_k, active_budget=block.get("expert_pool_public_budget")
            )
        if private_experts:
            width += (
                top_k
                * private_top_k
                * private_expert_hidden_dim(
                    dim,
                    top_k,
                    private_top_k,
                    with_public_pool=public_experts > 0,
                    active_budget=block.get("expert_pool_active_budget"),
                )
            )
        return width, True
    active_experts = int(block.get("router_num_experts_per_group_per_token") or 1)
    return float(active_experts * int(block["block_mlp_hidden_dims"][0])), gated


def backbone_body_flops(backbone: NestorConfig, *, seq_len: int, causal: bool = True) -> BodyFlops:
    """Forward body FLOPs of one sequence through every layer of a backbone.

    Args:
        backbone (NestorConfig): The backbone configuration.
        seq_len (int): Sequence length.
        causal (bool): Causal attention core.

    Returns:
        BodyFlops: Summed over every layer of every block.
    """

    dim = int(backbone.feature_dims[0])  # type: ignore[index]
    projections = core = mlp = 0.0
    for block in backbone.to_dict()["block_configs"]:
        layers = int(block["block_num_layers"])
        width = int(block["block_num_heads"]) * int(block["block_head_dim"])
        p, c = attention_flops(
            seq_len=seq_len,
            dim=dim,
            groups=int(block["num_groups"]),
            top_k=int(block["num_groups_per_token"]),
            width=width,
            causal=causal,
        )
        active_width, gated = block_active_mlp_width(block, dim)
        m = mlp_flops(seq_len=seq_len, dim=dim, active_width=active_width, gated=gated)
        projections += layers * p
        core += layers * c
        mlp += layers * m
    return BodyFlops(projections, core, mlp)


def body_flops(experiment: str, *, seq_len: int, causal: bool = True) -> BodyFlops:
    """Forward body FLOPs of one sequence through a registered cell.

    Args:
        experiment (str): A registered experiment name.
        seq_len (int): Sequence length.
        causal (bool): Causal attention core.

    Returns:
        BodyFlops: Summed over every layer.
    """

    return backbone_body_flops(experiment_backbone(experiment), seq_len=seq_len, causal=causal)


def flops_per_token(experiment: str, *, seq_len: int, causal: bool = True) -> float:
    """Forward body FLOPs per token of a registered cell at a context length.

    Args:
        experiment (str): A registered experiment name.
        seq_len (int): Sequence length.
        causal (bool): Causal attention core.

    Returns:
        float: ``body_flops(...).total / seq_len``.
    """

    return body_flops(experiment, seq_len=seq_len, causal=causal).total / seq_len


def flop_ratio(experiment: str, reference: str, *, seq_len: int, causal: bool = True) -> float:
    """Forward body FLOPs of ``experiment`` relative to ``reference`` at a context length.

    Args:
        experiment (str): A registered experiment name.
        reference (str): The experiment to divide by (``moe_<scale>`` for the
            ladder's ``r(S)``).
        seq_len (int): Sequence length.
        causal (bool): Causal attention core.

    Returns:
        float: The ratio; below 1 is a rebate, and ``1 / ratio`` the equal-FLOP
            token multiplier.
    """

    return (
        body_flops(experiment, seq_len=seq_len, causal=causal).total
        / body_flops(reference, seq_len=seq_len, causal=causal).total
    )
