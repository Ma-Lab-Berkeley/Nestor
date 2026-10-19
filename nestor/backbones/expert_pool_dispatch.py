"""Width rules and expert-bank dispatch helpers for the expert pools.

Module-level companions of :class:`~nestor.backbones.expert_pools.ExpertPoolMLP`
(split out to keep that file under the repo's size limit; ``expert_pools``
re-exports every public name here):

* the width contract that sizes public and private experts from the active
  budget (:func:`public_expert_hidden_dim`, :func:`private_expert_hidden_dim`);
* the two dispatch forms a bank can run in -- the static-shape grouped-GEMM
  combine (:func:`_grouped_expert_bank`, over
  :func:`~nestor.backbones.grouped_mm.grouped_expert_rows`) and the per-expert
  sparse loop it replaces (:func:`_accumulate_sparse_expert_bank`, the CPU /
  ``NESTOR_POOL_GROUPED_MM=0`` fallback).
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from nestor.backbones.grouped_mm import grouped_expert_combine


def round_to_multiple_of_8(value: float) -> int:
    """Round a positive width to the nearest multiple of 8 (min 8).

    Args:
        value (float): Ideal (fractional) width.

    Returns:
        int: ``max(8, round(value / 8) * 8)``.
    """

    return max(8, int(round(value / 8.0)) * 8)


def public_expert_hidden_dim(
    dim: int, public_top_k: int, *, active_budget: float | None = None
) -> int:
    """Public-expert SwiGLU intermediate width ``round_to_8(budget * d / k_p)``.

    The public family owns an aggregate ``2d`` of active width by default;
    dividing by the expert top-k keeps that budget constant as ``k_p`` grows
    (the doc's table: d=640 -> 1280, d=960 -> 1920, d=1280 -> 2560 at
    ``k_p=1``).

    Args:
        dim (int): Token feature width ``d``.
        public_top_k (int): Public experts active per token ``k_p``.
        active_budget (float | None): Override the public family's active
            budget in multiples of ``dim`` (``None`` = the ``2d`` contract).
            Paired with :func:`private_expert_hidden_dim`'s ``active_budget``
            this rescales the whole hybrid MLP -- e.g. 1.25 + 1.25 is the
            "2.5d" MLP whose FLOP share is handed to attention.

    Returns:
        int: Intermediate width of each public expert.
    """

    budget = 2.0 if active_budget is None else float(active_budget)
    return round_to_multiple_of_8(budget * dim / public_top_k)


def private_expert_hidden_dim(
    dim: int,
    num_groups_per_token: int,
    private_top_k: int,
    *,
    with_public_pool: bool = True,
    active_budget: float | None = None,
) -> int:
    """Private-expert SwiGLU intermediate width, split over ``k_a * k_q``.

    The private family's active budget is split across the ``k_a`` selected
    attention groups times ``k_q`` experts per group. Alongside a public pool
    that budget is ``2d`` (the doc's table: d=640 -> 424, d=960 -> 640,
    d=1280 -> 856 at ``k_a=3, k_q=1``); with no public family to share with it
    is the whole ``4d`` (d=640 -> 856 at ``k_a=3``, 512 at ``k_a=5``, 368 at
    ``k_a=7``), so the active total stays ~``4d`` in both modes.

    Args:
        dim (int): Token feature width ``d``.
        num_groups_per_token (int): Active attention groups per token ``k_a``.
        private_top_k (int): Private experts active per selected group ``k_q``.
        with_public_pool (bool): Whether a public pool shares the active
            budget (``2d`` each) or the private family owns all of it
            (``4d``).
        active_budget (float | None): Override the active budget in multiples
            of ``dim``, replacing the ``2d``/``4d`` default. Used to widen the
            MLP when an attention schedule spends less than the dense FLOP
            budget, so the model can hand the refund to the experts instead of
            leaving it unspent.

    Returns:
        int: Intermediate width of each private expert.
    """

    if active_budget is not None:
        budget = float(active_budget)
    else:
        budget = 2.0 if with_public_pool else 4.0
    return round_to_multiple_of_8(
        budget * dim / (num_groups_per_token * private_top_k)
    )


def _grouped_expert_bank(
    x: Float[Tensor, "num_tokens dim"],
    experts: list[nn.Module],
    expert_ids: Int[Tensor, "num_tokens slots"],
    slot_weights: Float[Tensor, "num_tokens slots"],
    *,
    fused_gate: bool = False,
    stacked: object | None = None,
    triton_index: bool = False,
    triton_fused: bool = False,
    triton_cat_cast: bool = True,
) -> Float[Tensor, "num_tokens dim"]:
    """Combine an expert bank over static ``(token, slot)`` rows in one GEMM.

    The grouped-GEMM counterpart of :func:`_accumulate_sparse_expert_bank`:
    instead of looping over experts and selecting each one's tokens with a
    host-syncing ``nonzero``, every ``(token, slot)`` pair is dispatched
    (``num_tokens * slots`` rows, static shape) and the bank runs as three
    grouped GEMMs. Slots whose weight is zero -- padding tokens, non-member
    attention groups -- contribute exactly zero, so they need no masking; they
    only cost (already-paid) capacity in the batched GEMM.

    The per-slot product is formed in the expert's (possibly autocast-reduced)
    dtype and the sum over slots is taken in ``x``'s dtype, matching the sparse
    path's ``weighted_out.to(residual.dtype)`` accumulation exactly.

    Args:
        x (Float[Tensor, "num_tokens dim"]): Pre-normalized token features.
        experts (list[nn.Module]): Expert MLPs (``dim -> dim``), all the same
            shape; indexed by ``expert_ids``.
        expert_ids (Int[Tensor, "num_tokens slots"]): Bank index of each
            dispatched ``(token, slot)`` pair.
        slot_weights (Float[Tensor, "num_tokens slots"]): Combine weight of
            each dispatched pair (the product of every gate on its path).
        fused_gate (bool): Forwarded to
            :func:`~nestor.backbones.grouped_mm.grouped_expert_combine` (one
            grouped GEMM for up + gate on SwiGLU banks).
        stacked (StackedExpertBank | None): Forwarded; the bank's stacked
            owners when ``NESTOR_STACKED_EXPERT_BANKS`` is on.
        triton_index (bool): Forwarded; ``NESTOR_TRITON_INDEX_BUILD``'s fused
            dispatch-map kernel (integer-exact, launch count only).
        triton_fused (bool): Forwarded; ``NESTOR_TRITON_EXPERT_DISPATCH``'s
            fused dispatch / SwiGLU / weighted-combine kernels.
        triton_cat_cast (bool): Forwarded; the fused cat+cast of a stacked
            up|gate bank under ``triton_fused``.

    Returns:
        Float[Tensor, "num_tokens dim"]: ``sum_slot weight * expert(x)``.
    """

    num_tokens, slots = expert_ids.shape
    return grouped_expert_combine(
        x, experts, expert_ids, slot_weights, slots,
        fused_gate=fused_gate, stacked=stacked, triton_index=triton_index,
        triton_fused=triton_fused, triton_cat_cast=triton_cat_cast,
    ).reshape(num_tokens, -1)


def _accumulate_sparse_expert_bank(
    residual: Float[Tensor, "num_tokens dim"],
    x: Float[Tensor, "num_tokens dim"],
    experts: list[nn.Module],
    expert_weights: Float[Tensor, "num_tokens num_experts"],
) -> None:
    """Apply an expert bank only on tokens with non-zero routing weight.

    The CPU / ``NESTOR_POOL_GROUPED_MM=0`` fallback for
    :func:`_grouped_expert_bank`.

    Mirrors ``RoutedTransformerLayer._accumulate_sparse_expert_bank`` with one
    addition: the empty-bank keepalive multiplies by an ``expert_weights``
    entry (times 0) so the ROUTING weights stay in the autograd graph too --
    an attention group with zero member tokens would otherwise drop its whole
    private router from the graph and break DDP's reducer. (The grouped path
    needs no keepalive: it stacks every expert weight and every private
    router's gates, so all of them join the graph unconditionally.)

    Args:
        residual (Float[Tensor, "num_tokens dim"]): Accumulator, updated in
            place via ``index_add_``.
        x (Float[Tensor, "num_tokens dim"]): Pre-normalized token features.
        experts (list[nn.Module]): Expert MLPs (``dim -> dim``).
        expert_weights (Float[Tensor, "num_tokens num_experts"]): Combined
            gate weight of each (token, expert) pair; 0 skips the pair.
    """

    for expert_idx, expert in enumerate(experts):
        token_indices = torch.nonzero(
            expert_weights[:, expert_idx],
            as_tuple=False,
        ).squeeze(-1)
        if token_indices.numel() == 0:
            # Keepalive: keep the expert AND its routing weights in the graph
            # (zero gradients) so every rank issues identical collectives.
            keepalive = expert(x[:1]) * expert_weights[
                :1, expert_idx
            ].unsqueeze(-1)
            residual[:1].add_((keepalive * 0).to(residual.dtype))
            continue
        expert_out = expert(x.index_select(0, token_indices))
        weighted_out = expert_out * expert_weights.index_select(
            0,
            token_indices,
        )[:, expert_idx].unsqueeze(-1)
        # Under bf16 autocast the experts emit bf16 while the accumulator
        # follows the (fp32) residual stream; accumulate in the accumulator's
        # dtype, which also keeps the sum in the wider type.
        residual.index_add_(0, token_indices, weighted_out.to(residual.dtype))
