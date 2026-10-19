"""One batched pass over a layer's ``G`` private pool routers.

A profiler trace of the 1b Nestor arm at per-rank batch 1 showed the forward
64% idle: ~2,500 launches per
micro-step with the eager (graph-broken) router region issuing 608 ``topk`` +
469 ``sort`` + 855 ``copy_`` host calls. Most of those are the expert pool's
private routers: :meth:`ExpertPoolMLP._route` runs ``G`` separate
:class:`~nestor.backbones.nestor.Router` modules per layer -- each a score MLP
(2 GEMMs), a noise MLP (2 GEMMs, softplus, ``randn_like``), two softmaxes, a
``topk``, a gather, ``zeros_like`` + ``scatter_add_`` and a mask multiply --
and :meth:`ExpertPoolMLP._private_slots` then re-stacks their outputs (16 more
copies per layer).

Every private router of a layer sees the SAME input ``x`` (masked to its
group's members only at the very end), has the same shapes and the same gate
mode, so the whole family evaluates as ONE batched router: the ``G`` score
heads' first layers concatenate into a single ``d -> G*h`` GEMM, their second
layers into one ``[G]``-batched matmul, and every subsequent op (noise,
softmax, bias, top-k, switch/renorm gate, scatter, mask) runs once on
``[B, S, G, Q]``. The result is returned both as the stacked tensors the
grouped dispatch wants and as ``G`` :class:`RouterOutput` VIEWS, so the
aux / loss-free-balancing plumbing (one aux per private router) is untouched.

Exactness: the parameters are the routers' own (stacked at forward time, so
checkpoints and resumes are unaffected); every value equals the sequential
path up to GEMM accumulation order (the concatenated first layer and the
batched second layer tile differently). Routing decisions therefore agree
except on near-ties at the bf16 ulp. The exploration noise draws ONE
``randn_like`` of ``[B, S, G, Q]`` instead of ``G`` draws of ``[B, S, Q]`` --
the same distribution, a different consumption of the RNG stream. The
per-router noise-anneal counters advance exactly as before.

Enabled per module at construction by ``NESTOR_BATCHED_ROUTERS=1`` (default
off), see :func:`batched_routers_enabled`.
"""

from __future__ import annotations

import os

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional as F

from nestor.backbones.nestor import Router, RouterOutput, _is_recomputing
from nestor.scaling.blocks import ScaledMLP


def batched_routers_enabled() -> bool:
    """Whether ``NESTOR_BATCHED_ROUTERS=1`` is set (read at module construction).

    Returns:
        bool: True iff the environment enables the batched private routers.
    """

    return os.environ.get("NESTOR_BATCHED_ROUTERS", "0") == "1"


def check_batchable(routers: list[Router]) -> None:
    """Raise unless the routers can be evaluated as one batched router.

    Args:
        routers (list[Router]): The layer's private routers, in group order.

    Raises:
        ValueError: On mismatched choice counts, top-k, gate modes, MLP
            depths/widths or bias presence between any two routers.
    """

    if not routers:
        raise ValueError("Need at least one router to batch.")
    first = routers[0]

    def _shape(mlp: ScaledMLP) -> tuple:
        return tuple(
            (layer.in_features, layer.out_features, layer.bias is not None)
            for layer in mlp.layers
        )

    for router in routers[1:]:
        same = (
            router.num_choices == first.num_choices
            and router.num_choices_per_token == first.num_choices_per_token
            and router.gate_mode == first.gate_mode
            and _shape(router.score_mlp) == _shape(first.score_mlp)
            and _shape(router.noise_mlp) == _shape(first.noise_mlp)
            and router.score_mlp.activation == first.score_mlp.activation
            and router.noise_mlp.activation == first.noise_mlp.activation
        )
        if not same:
            raise ValueError(
                "Private routers differ in shape/gate mode; cannot batch them."
            )


def batched_mlp(
    x: Float[Tensor, "num_tokens d_in"], mlps: list[ScaledMLP]
) -> Float[Tensor, "num_tokens G d_out"]:
    """Evaluate ``G`` same-shaped :class:`ScaledMLP` heads on a shared input.

    Layer 0 of every head reads the same ``x``, so the ``G`` weight matrices
    concatenate along their output axis into ONE GEMM (``d_in -> G * h_0``);
    each later layer is a ``[G]``-batched matmul on the per-head hidden states
    (``einsum("ngi,goi->ngo")``). Activations are applied between layers
    exactly as :meth:`ScaledMLP.forward` does (none after the last).

    Args:
        x (Float[Tensor, "num_tokens d_in"]): Shared input rows.
        mlps (list[ScaledMLP]): ``G`` heads with identical layer shapes.

    Returns:
        Float[Tensor, "num_tokens G d_out"]: Head ``g``'s output in ``[:, g]``
        (equal to ``mlps[g](x)`` up to GEMM accumulation order).
    """

    depth = len(mlps[0].layers)
    activation = mlps[0]._activation_fn
    num_heads = len(mlps)
    # Layer 0: one GEMM with the G weights stacked along the output axis.
    w0 = torch.stack([mlp.layers[0].weight for mlp in mlps])  # [G, h0, d_in]
    b0 = (
        torch.stack([mlp.layers[0].bias for mlp in mlps]).reshape(-1)  # [G*h0]
        if mlps[0].layers[0].bias is not None
        else None
    )
    hidden = F.linear(x, w0.reshape(num_heads * w0.shape[1], w0.shape[2]), b0)
    hidden = hidden.view(x.shape[0], num_heads, w0.shape[1])  # [N, G, h0]
    for index in range(1, depth):
        hidden = activation(hidden)
        w = torch.stack([mlp.layers[index].weight for mlp in mlps])  # [G, o, i]
        hidden = torch.einsum("ngi,goi->ngo", hidden, w)
        if mlps[0].layers[index].bias is not None:
            hidden = hidden + torch.stack(
                [mlp.layers[index].bias for mlp in mlps]
            )  # [G, o] broadcast over N
    return hidden


def route_private_batched(
    routers: list[Router],
    x: Float[Tensor, "batch_size seq_len dim"],
    valid_token_mask: Bool[Tensor, "batch_size seq_len"],
    group_active_mask: Bool[Tensor, "batch_size seq_len num_groups"],
    training: bool,
    noise: Float[Tensor, "batch_size seq_len num_groups num_choices"] | None = None,
) -> tuple[
    list[RouterOutput],
    Float[Tensor, "batch_size seq_len num_groups num_choices"],
    Int[Tensor, "batch_size seq_len num_groups top_k"],
]:
    """Run every private router of a layer in one batched pass.

    Reproduces :meth:`Router.forward` op for op on ``[B, S, G, Q]`` tensors
    (score MLP, training noise with the per-router anneal schedule, softmax,
    detached choice bias, top-k, ``"switch"`` or ``"renorm"`` gate, scatter,
    ``valid & member`` mask) and hands back per-group views.

    Args:
        routers (list[Router]): The ``G`` private routers, in group order
            (validated by :func:`check_batchable` at construction).
        x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized tokens.
        valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real tokens.
        group_active_mask (Bool[Tensor, "batch_size seq_len num_groups"]):
            Attention-group membership per token.
        training (bool): Draw the exploration noise (``Router.training``).
        noise (Float[Tensor, "batch_size seq_len num_groups num_choices"] | None):
            Pre-drawn ``N(0, 1)`` factor for every group (see
            :meth:`ExpertPoolMLP.draw_router_noise`); ``None`` draws it here.

    Returns:
        tuple[list[RouterOutput], Float[Tensor, "batch_size seq_len
        num_groups num_choices"], Int[Tensor, "batch_size seq_len num_groups
        top_k"]]: One output per group (views into the stacked tensors, each
        masked to ``valid & member``), the stacked gates and the stacked
        top-k choices -- what :meth:`ExpertPoolMLP._private_slots` otherwise
        rebuilds with ``torch.stack``.
    """

    first = routers[0]
    num_groups = len(routers)
    batch_size, seq_len, dim = x.shape
    flat_x = x.reshape(batch_size * seq_len, dim)

    scores = batched_mlp(flat_x, [r.score_mlp for r in routers]).view(
        batch_size, seq_len, num_groups, first.num_choices
    )
    if training and first.noise_mlp is not None:
        std = F.softplus(
            batched_mlp(flat_x, [r.noise_mlp for r in routers]).view_as(scores)
        )
        eps = torch.randn_like(scores) if noise is None else noise.to(scores.dtype)
        noise = eps * std
        if any(r._noise_anneal_steps > 0 for r in routers):
            frac = torch.stack(
                [
                    (r._noise_steps.float() / r._noise_anneal_steps).clamp(max=1.0)
                    if r._noise_anneal_steps > 0
                    else torch.zeros((), device=x.device)
                    for r in routers
                ]
            )  # [G]
            noise = noise * (1.0 - frac).to(noise.dtype)[None, None, :, None]
            if not _is_recomputing():
                for r in routers:
                    if r._noise_anneal_steps > 0:
                        r._noise_steps += 1
        scores = scores + noise

    softmax_probs = F.softmax(scores, dim=-1)
    choice_bias = torch.stack([r.choice_bias for r in routers])  # [G, Q]
    topk_indices = torch.topk(
        scores + choice_bias, k=first.num_choices_per_token, dim=-1
    ).indices  # [B, S, G, k]
    if first.gate_mode == "switch":
        topk_gates = (
            F.softmax(scores.float(), dim=-1)
            .gather(dim=-1, index=topk_indices)
            .to(scores.dtype)
        )
    else:
        topk_scores = scores.gather(dim=-1, index=topk_indices)
        topk_gates = F.softmax(topk_scores.float(), dim=-1).to(scores.dtype)
    gates = torch.zeros_like(scores).scatter_add_(-1, topk_indices, topk_gates)
    member = valid_token_mask.unsqueeze(-1) & group_active_mask  # [B, S, G]
    gates = gates * member.unsqueeze(-1).to(dtype=gates.dtype)
    active = gates > 0

    outputs = [
        RouterOutput(
            scores=scores[:, :, g],
            softmax_probs=softmax_probs[:, :, g],
            gates=gates[:, :, g],
            topk_indices=topk_indices[:, :, g],
            active_mask=active[:, :, g],
            valid_token_mask=member[:, :, g],
        )
        for g in range(num_groups)
    ]
    return outputs, gates, topk_indices
