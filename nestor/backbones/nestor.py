"""The Nestor backbone: column attention, hybrid expert pools, depth reads.

A Nestor model is a stack of constant-width *blocks*. Inside a block every
layer runs ``G`` attention *columns*, each owning a few heads with their own
projections; a per-block router sends every token to ``k`` of the ``G``
columns, and token dispatch keeps each column's attention core over its member
tokens only, so the attention cost scales with ``k / G``. The MLP of a layer is
either one expert bank (``E`` SwiGLU experts routed top-``k_b``; ``E = 1`` is
the plain dense MLP) or the hybrid pool: a public bank shared by every token
plus a private pool owned by each column, mixed by a learned path gate. Blocks
are joined by a depth read that attends over the completed block
representations (the Kimi block read, or the column read that also exposes
each column's deviation from its block aggregate). The dense transformer and
the MoE of the paper are the special case ``G = 1``.

Outline (top to bottom):

1. Small helpers (``_promote_to_blockwise``, router-output utilities, the
   column stream and dispatch plans, activation-checkpoint helpers).
2. Internal primitives: ``MLP`` and the token-choice top-k ``Router``.
3. Auxiliary output / cache containers.
4. Per-block config and the top-level ``NestorConfig``.
5. ``GroupRoutedAttention``, the depth-read helpers, ``RoutedTransformerLayer``
   and ``NestorBlock``.
6. The top-level ``Nestor`` backbone.

The public contract is intentionally small:

- :class:`NestorConfig` describes the backbone.
- :class:`Nestor` consumes token-domain inputs ``(x, pos, attn_mask)``.
- :meth:`Nestor.to_config` round-trips back into a config.

Adapters and task heads live outside this file; the depth reads live in
``attention_residual.py`` and the hybrid pool in ``expert_pools.py``.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, cast

import torch
import torch.distributed as dist
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

try:
    _flex_attention_module = importlib.import_module(
        "torch.nn.attention.flex_attention"
    )
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    _flex_attention_module = None
    _FLEX_ATTENTION_AVAILABLE = False

from nestor.backbones.base import (
    Backbone,
    BackboneConfig,
    ModelOutput,
    register_backbone,
)
from nestor.backbones.attention_residual import (
    BlockAttentionResidual,
    DepthAttentionRead,
    DepthKeyCache,
)
from nestor.backbones.expert_bank_stack import StackedBanksMixin
from nestor.backbones.grouped_mm import fused_gate_requested, grouped_expert_rows
from nestor.backbones.packed_pointwise import (
    attn_out_rows,
    attn_out_rows_compiled,
    gate_combine,
    gate_combine_compiled,
    packed_pointwise_requested,
    rope_qkv,
    rope_qkv_compiled,
)
from nestor.backbones.triton_dispatch import (
    triton_index_available,
    triton_index_requested,
)
from nestor.backbones.triton_expert import triton_expert_requested
from nestor.backbones.triton_packed import (
    triton_packed_available,
    triton_packed_requested,
)
from nestor.constants import PADDING_TOKEN_POS
from nestor.scaling.blocks import (
    GatedScaledMLP,
    RMSNorm,
    RoPE,
    ScaledLinear,
    ScaledMLP,
    _rotate_half,
    convex_residual,
)

_COMPILED_FLEX_ATTENTION: Callable[..., Tensor] | None = None
_FLEX_ATTENTION_COMPILE_FAILED = False
create_block_mask: Callable[..., Any] | None = (
    getattr(_flex_attention_module, "create_block_mask", None)
    if _flex_attention_module is not None
    else None
)
flex_attention: Callable[..., Any] | None = (
    getattr(_flex_attention_module, "flex_attention", None)
    if _flex_attention_module is not None
    else None
)


def _get_flex_attention_impl() -> Callable[..., Tensor] | None:
    """Return a compiled flex-attention kernel when available."""

    global _COMPILED_FLEX_ATTENTION, _FLEX_ATTENTION_COMPILE_FAILED

    if not _FLEX_ATTENTION_AVAILABLE or flex_attention is None:
        return None
    if _COMPILED_FLEX_ATTENTION is not None:
        return _COMPILED_FLEX_ATTENTION
    if _FLEX_ATTENTION_COMPILE_FAILED or not hasattr(torch, "compile"):
        # No fused kernel available: report None so callers use SDPA rather
        # than eager flex (which materializes the full score matrix).
        return None

    try:
        _COMPILED_FLEX_ATTENTION = cast(
            Callable[..., Tensor],
            torch.compile(flex_attention),
        )
    except Exception:
        _FLEX_ATTENTION_COMPILE_FAILED = True
        return cast(Callable[..., Tensor], flex_attention)
    return _COMPILED_FLEX_ATTENTION


# ---------------------------------------------------------------------------
# Block-wise broadcasting helpers.
# ---------------------------------------------------------------------------


def _promote_to_blockwise(
    value: Any | list[Any],
    num_blocks: int,
    *,
    nested_list: bool = False,
) -> list[Any]:
    """Broadcast a scalar or list to one value per block.

    Args:
        value (Any | list[Any]): Scalar, flat list, or blockwise nested list.
        num_blocks (int): Number of blocks to produce values for.
        nested_list (bool): Whether a flat list should be copied for each block
            instead of interpreted as one value per block.

    Returns:
        list[Any]: One value per block.
    """

    if isinstance(value, list):
        if nested_list:
            if value and isinstance(value[0], list):
                if len(value) != num_blocks:
                    raise ValueError(
                        "Nested blockwise values must match num_blocks."
                    )
                return cast(list[Any], value)
            return [deepcopy(value) for _ in range(num_blocks)]
        if len(value) == num_blocks:
            return value
        return [deepcopy(value) for _ in range(num_blocks)]
    return [deepcopy(value) for _ in range(num_blocks)]


# ---------------------------------------------------------------------------
# Internal primitives: MLP and token-choice top-k router.
# ---------------------------------------------------------------------------


_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "tanh": F.tanh,
}


def causal_full_query_chunk() -> int:
    """Query-block size of the chunked fp32 causal fallback (``NESTOR_CAUSAL_FULL_QCHUNK``).

    Read at call time. ``0`` disables the fallback; the default 1024 bounds
    the score matrix of one block to ``[B, H, 1024, S]``.

    Returns:
        int: Queries per block (``>= 0``).
    """

    return max(0, int(os.environ.get("NESTOR_CAUSAL_FULL_QCHUNK", "1024")))


def causal_attention_query_chunked(
    q: Float[Tensor, "batch_size num_heads seq_len head_dim"],
    k: Float[Tensor, "batch_size num_heads seq_len head_dim"],
    v: Float[Tensor, "batch_size num_heads seq_len head_dim"],
    scale: float,
    chunk: int,
) -> Float[Tensor, "batch_size num_heads seq_len head_dim"]:
    """Causal SDPA computed in query blocks (exact, bounded memory).

    Block ``[lo, hi)`` attends to keys ``[0, hi)`` under ``tril(diagonal=lo)``
    -- query ``lo + j`` sees keys ``<= lo + j`` -- which is row-for-row the
    ``is_causal=True`` kernel's output, up to accumulation order. Peak
    memory is one ``[B, H, chunk, hi]`` score block instead of ``[B, H, S, S]``.

    Args:
        q (Float[Tensor, "batch_size num_heads seq_len head_dim"]): Queries.
        k (Float[Tensor, "batch_size num_heads seq_len head_dim"]): Keys.
        v (Float[Tensor, "batch_size num_heads seq_len head_dim"]): Values.
        scale (float): Softmax scale (``1 / head_dim`` in this module).
        chunk (int): Queries per block (``> 0``).

    Returns:
        Float[Tensor, "batch_size num_heads seq_len head_dim"]: Attention output.
    """

    seq_len = q.shape[2]
    outs: list[Tensor] = []  # each [B, H, hi - lo, head_dim]
    for lo in range(0, seq_len, chunk):
        hi = min(lo + chunk, seq_len)
        mask = torch.ones(hi - lo, hi, dtype=torch.bool, device=q.device).tril(diagonal=lo)  # [c, hi]
        outs.append(
            F.scaled_dot_product_attention(
                q[:, :, lo:hi], k[:, :, :hi], v[:, :, :hi], attn_mask=mask, dropout_p=0.0, scale=scale
            )
        )
    return torch.cat(outs, dim=2)


class MLP(nn.Module):
    """Simple multi-layer perceptron used inside the Nestor backbone.

    Attributes:
        input_dim (int): Input feature width.
        hidden_dims (list[int]): Widths of any hidden layers.
        output_dim (int): Output feature width.
        activation (str): Named activation applied between linear layers.
    """

    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    activation: str

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): Input feature width.
            hidden_dims (list[int]): Hidden layer widths (may be empty).
            output_dim (int): Output feature width.
            activation (str): Named activation. Must be one of the keys in
                ``_ACTIVATIONS``.
        """

        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Invalid activation '{activation}'. "
                f"Allowed: {sorted(_ACTIVATIONS)}."
            )

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation

        if len(hidden_dims) == 0:
            layers: list[nn.Linear] = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dims[0])]
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = cast(list[nn.Linear], nn.ModuleList(layers))
        self._activation_fn: Callable[[Tensor], Tensor] = _ACTIVATIONS[
            activation
        ]

    def forward(
        self,
        x: Float[Tensor, "... input_dim"],
    ) -> Float[Tensor, "... output_dim"]:
        """Apply the MLP to the last-axis features.

        Args:
            x (Float[Tensor, "... input_dim"]): Input features.

        Returns:
            Float[Tensor, "... output_dim"]: Output features.
        """

        for layer in self.layers[:-1]:
            x = self._activation_fn(layer(x))
        return self.layers[-1](x)


class RouterOutput(NamedTuple):
    """Output from the Nestor token-choice top-k router.

    Attributes:
        scores (Float[Tensor, "batch_size seq_len num_choices"]): Raw router
            scores before softmax/top-k selection.
        softmax_probs (Float[Tensor, "batch_size seq_len num_choices"]): Dense
            softmax probabilities over the full choice set.
        gates (Float[Tensor, "batch_size seq_len num_choices"]): Gating
            coefficients for each choice (non-zero only on top-k choices).
        topk_indices (Int[Tensor, "batch_size seq_len top_k"]): Indices of the
            top-k selected choices per token.
        active_mask (Bool[Tensor, "batch_size seq_len num_choices"]): Whether a
            given choice is active for each token.
        valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Whether a token
            is a real (non-padding) token.
    """

    scores: Float[Tensor, "batch_size seq_len num_choices"]
    softmax_probs: Float[Tensor, "batch_size seq_len num_choices"]
    gates: Float[Tensor, "batch_size seq_len num_choices"]
    topk_indices: Int[Tensor, "batch_size seq_len top_k"]
    active_mask: Bool[Tensor, "batch_size seq_len num_choices"]
    valid_token_mask: Bool[Tensor, "batch_size seq_len"]


def _lossfree_global_counts() -> bool:
    """Whether loss-free balancing sums its choice counts across DDP ranks.

    Returns:
        bool: True iff ``NESTOR_LOSSFREE_GLOBAL_COUNTS=1`` is set.
    """

    return os.environ.get("NESTOR_LOSSFREE_GLOBAL_COUNTS", "0") == "1"


class Router(nn.Module):
    """Token-choice top-k router with optional loss-free balancing.

    The router produces a score MLP and a noise MLP. During training it adds
    a softplus-noise perturbation to the scores, then selects top-k choices
    per token using a bias-adjusted score. Optionally, a detached choice bias
    is nudged after each optimizer step to keep the active-choice distribution
    balanced (see DeepSeek's loss-free balancing).
    """

    feature_dim: int
    num_choices: int
    num_choices_per_token: int
    loss_free_balancing: bool
    bias_update_rate: float
    gate_mode: str
    choice_bias: Tensor

    def __init__(
        self,
        feature_dim: int,
        num_choices: int,
        num_choices_per_token: int,
        score_mlp_hidden_dims: list[int],
        score_mlp_activation: str,
        noise_mlp_hidden_dims: list[int],
        noise_mlp_activation: str,
        *,
        score_lr_mul: float = 1.0,
        loss_free_balancing: bool = False,
        bias_update_rate: float = 0.0,
        gate_mode: str = "renorm",
    ) -> None:
        """Initialize the router.

        Args:
            feature_dim (int): Input feature width.
            num_choices (int): Number of routing choices.
            num_choices_per_token (int): Top-k per token (1 to ``num_choices``).
            score_mlp_hidden_dims (list[int]): Hidden widths of the score MLP.
            score_mlp_activation (str): Score MLP activation.
            noise_mlp_hidden_dims (list[int]): Hidden widths of the noise MLP.
            noise_mlp_activation (str): Noise MLP activation.
            score_lr_mul (float): Extra learning-rate multiplier for the score
                and noise heads (e.g. ``1/sqrt(M_l)`` for a group router shared
                across ``M_l`` layers).
            loss_free_balancing (bool): Whether detached bias updates are used.
            bias_update_rate (float): Step size for detached bias updates.
            gate_mode (str): How the selected choices are weighted.
                ``"renorm"`` (default) softmaxes the k SELECTED scores so the
                gates sum to 1 -- for top-1 that is the constant 1.0 and the
                router receives exactly ZERO gradient (a top-1 renorm router
                without an aux loss is an untrained random hash).
                ``"switch"`` weights each selected choice by its FULL-softmax
                probability (Switch Transformer), which keeps a top-1 router
                trainable; gates then sum to < 1.
        """

        super().__init__()
        if num_choices < 1:
            raise ValueError("num_choices must be positive.")
        if not 1 <= num_choices_per_token <= num_choices:
            raise ValueError(
                "num_choices_per_token must be in [1, num_choices]."
            )
        if gate_mode not in ("renorm", "switch"):
            raise ValueError(
                f"gate_mode must be 'renorm' or 'switch', got {gate_mode!r}."
            )
        self.gate_mode = gate_mode
        if num_choices > 1 and num_choices_per_token == 1 and gate_mode == "renorm":
            # The trap every early pooled arm fell into: a renormalized gate
            # over ONE selected score is the constant 1.0 with exactly zero
            # gradient, so unless an aux balance loss reads softmax_probs this
            # router never trains -- it routes as a frozen random hash. That
            # cost the private-pool arms ~0.086 nats. gate_mode="switch" makes
            # top-1 routing differentiable. renorm stays the default only so
            # existing registered arms/checkpoints rebuild byte-identically.
            import warnings

            warnings.warn(
                f"Router(num_choices={num_choices}, top-1, gate_mode='renorm'): "
                "the gate is the constant 1.0 with ZERO gradient; without an "
                "aux balance loss this router will NOT train (it routes as a "
                "random hash). Pass gate_mode='switch' for a trainable top-1 "
                "router.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.feature_dim = feature_dim
        self.num_choices = num_choices
        self.num_choices_per_token = num_choices_per_token
        self.loss_free_balancing = loss_free_balancing
        self.bias_update_rate = bias_update_rate
        self.score_mlp = ScaledMLP(
            input_dim=feature_dim,
            hidden_dims=score_mlp_hidden_dims,
            output_dim=num_choices,
            activation=score_mlp_activation,
            final_role="score",
            final_lr_mul=score_lr_mul,
        )
        self.noise_mlp = ScaledMLP(
            input_dim=feature_dim,
            hidden_dims=noise_mlp_hidden_dims,
            output_dim=num_choices,
            activation=noise_mlp_activation,
            final_role="score",
            final_lr_mul=score_lr_mul,
        )
        self.register_buffer("choice_bias", torch.zeros(num_choices))

        # NESTOR_ROUTER_NOISE_ANNEAL_STEPS=N linearly decays the exploration
        # noise to 0 over the first N optimizer steps, then routes
        # deterministically -- so the model trains with the same routing it
        # evals with for the back half (probes the train/eval routing
        # mismatch). 0 (default) keeps the original constant noise. The step
        # counter advances once per training forward (== one optimizer step at
        # grad_accum=1); persistent so it survives resume.
        import os as _os

        self._noise_anneal_steps = int(
            _os.environ.get("NESTOR_ROUTER_NOISE_ANNEAL_STEPS", "0")
        )
        if self._noise_anneal_steps > 0:
            self.register_buffer(
                "_noise_steps", torch.zeros((), dtype=torch.long)
            )

    def draw_noise(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch_size seq_len num_choices"]:
        """Draw the standard-normal factor of this router's exploration noise.

        :meth:`forward` multiplies it by ``softplus(noise_mlp(x))`` exactly as
        it multiplies its own ``randn_like`` draw. Drawing it here -- OUTSIDE
        an activation checkpoint -- and passing it in as ``noise`` makes the
        checkpoint's backward recompute route identically to the forward: no
        RNG runs inside the recomputed region at all (torch 2.10's compiled
        checkpoint HOP does not reproduce in-region ``randn`` draws; see
        ``_maybe_checkpoint``).

        Args:
            batch_size (int): Batch size.
            seq_len (int): Sequence length.
            device (torch.device): Device of the scores.
            dtype (torch.dtype): Dtype of the scores (the autocast dtype under
                autocast).

        Returns:
            Float[Tensor, "batch_size seq_len num_choices"]: ``N(0, 1)`` draws.
        """

        return torch.randn(
            batch_size, seq_len, self.num_choices, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len feature_dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        noise: Float[Tensor, "batch_size seq_len num_choices"] | None = None,
    ) -> RouterOutput:
        """Route each token to its top-k choices.

        Args:
            x (Float[Tensor, "batch_size seq_len feature_dim"]): Router inputs.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of
                real tokens.
            noise (Float[Tensor, "batch_size seq_len num_choices"] | None):
                Pre-drawn ``N(0, 1)`` factor of the training noise (see
                :meth:`draw_noise`); ``None`` draws it here.

        Returns:
            RouterOutput: Scores, soft probabilities, gates, selected indices,
            active mask, and the original validity mask.
        """

        scores = self.score_mlp(x)
        if self.training and self.noise_mlp is not None:
            eps = torch.randn_like(scores) if noise is None else noise.to(scores.dtype)
            noise = eps * F.softplus(self.noise_mlp(x))
            if self._noise_anneal_steps > 0:
                frac = (
                    self._noise_steps.float() / self._noise_anneal_steps
                ).clamp(max=1.0)
                noise = noise * (1.0 - frac)
                # Advance the schedule only on a real forward. Anything that
                # re-enters this method for the same step -- an activation
                # checkpoint's backward recompute, or NESTOR_FASTPATH_CHECK's
                # second pass -- would otherwise anneal the exploration noise
                # faster than the optimizer actually steps, silently and
                # invisibly except as a routing-quality regression.
                if not _is_recomputing():
                    self._noise_steps += 1
            scores = scores + noise

        softmax_probs = F.softmax(scores, dim=-1)
        biased_scores = scores + self.choice_bias
        topk_indices = torch.topk(
            biased_scores,
            k=self.num_choices_per_token,
            dim=-1,
        ).indices
        if self.gate_mode == "switch":
            # Switch-style gate: each selected choice's FULL-softmax
            # probability. Unlike the renormalized gate this is differentiable
            # for top-1 (a softmax over one score is the constant 1.0).
            topk_gates = (
                F.softmax(scores.float(), dim=-1)
                .gather(dim=-1, index=topk_indices)
                .to(scores.dtype)
            )
        else:
            topk_scores = scores.gather(dim=-1, index=topk_indices)
            # Autocast may run softmax in fp32 even when router scores are
            # bf16. scatter_add_ requires source and destination dtypes to
            # match; the reference likewise casts normalized top-k weights back
            # to x.dtype.
            topk_gates = F.softmax(topk_scores.float(), dim=-1).to(scores.dtype)

        gates = torch.zeros_like(scores)
        gates.scatter_add_(dim=-1, index=topk_indices, src=topk_gates)
        gates = gates * valid_token_mask.unsqueeze(-1).to(dtype=gates.dtype)

        return RouterOutput(
            scores=scores,
            softmax_probs=softmax_probs,
            gates=gates,
            topk_indices=topk_indices,
            active_mask=gates > 0,
            valid_token_mask=valid_token_mask,
        )

    @torch.no_grad()
    def update_detached_parameters(
        self,
        topk_indices: Int[Tensor, "batch_size seq_len top_k"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
    ) -> None:
        """Update the detached choice bias for loss-free balancing.

        Args:
            topk_indices (Int[Tensor, "batch_size seq_len top_k"]): Indices
                selected for each token during the most recent forward.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Validity
                mask matching ``topk_indices``.
        """

        if not self.loss_free_balancing:
            return

        one_hot = F.one_hot(topk_indices, num_classes=self.num_choices).float()
        assigned = one_hot.sum(dim=-2) * valid_token_mask.unsqueeze(-1).float()
        counts = assigned.sum(dim=(0, 1))
        # NESTOR_LOSSFREE_GLOBAL_COUNTS=1: balance on the GLOBAL batch. By
        # default every rank nudges its bias from its own micro-batches, and
        # with DDP broadcast_buffers=True rank 0's copy then overwrites the
        # rest at the next forward -- so the balancer effectively sees 1/world
        # of the data and the other ranks' updates are discarded. Summing the
        # counts across ranks makes every rank compute the identical update
        # (the broadcast becomes a no-op). Opt-in so in-flight comparisons
        # keep their statistic; every rank calls this method the same number
        # of times per step, so the collective is safe.
        if (
            _lossfree_global_counts()
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        avg_count = counts.mean()
        error = avg_count - counts
        self.choice_bias += self.bias_update_rate * torch.sign(error)


def _router_output_to_aux(router_output: RouterOutput) -> dict[str, Tensor]:
    """Convert a router output into the aux structure used by losses and logs.

    Args:
        router_output (RouterOutput): Router output tensors.

    Returns:
        dict[str, Tensor]: Tensor-only aux dictionary.
    """

    return {
        "scores": router_output.scores,
        "probs": router_output.softmax_probs,
        "gates": router_output.gates,
        "topk_indices": router_output.topk_indices,
        "active_mask": router_output.active_mask,
        "valid_token_mask": router_output.valid_token_mask,
    }


def _update_router_from_aux(router: Router, router_aux: Any) -> None:
    """Feed one router's aux back into its detached loss-free-balancing bias.

    A no-op unless the aux is a dictionary carrying both tensors the bias
    update reads, so a layer whose router did not run (or did not report)
    simply skips its update.

    Args:
        router (Router): Router whose ``choice_bias`` is nudged.
        router_aux (Any): Candidate aux entry; expected to be a dict with
            ``"topk_indices"`` (Int[Tensor, "batch_size seq_len top_k"]) and
            ``"valid_token_mask"`` (Bool[Tensor, "batch_size seq_len"]).
    """

    if not isinstance(router_aux, dict):
        return
    topk_indices = router_aux.get("topk_indices")
    valid_token_mask = router_aux.get("valid_token_mask")
    if topk_indices is None or valid_token_mask is None:
        return
    router.update_detached_parameters(topk_indices, valid_token_mask)


def _constant_router_output(
    x: Float[Tensor, "batch_size seq_len dim"],
    valid_token_mask: Bool[Tensor, "batch_size seq_len"],
    num_choices: int,
) -> RouterOutput:
    """Fabricate the output of a single-choice router without running it.

    With one choice the top-k softmax gate is exactly ``1.0`` for every valid
    token and carries exactly zero gradient, so the router MLPs never influence
    the model function nor receive non-zero gradients. The dense no-router fast
    path (``NESTOR_DENSE_NO_ROUTER=1``) therefore skips the router (forward and
    its dead backward) and substitutes this constant output. All tensors are
    broadcast views, so nothing materializes at ``[B, S, num_choices]``.
    """

    batch_size, seq_len = x.shape[0], x.shape[1]
    valid_choices = valid_token_mask.unsqueeze(-1).expand(
        batch_size, seq_len, num_choices
    )
    return RouterOutput(
        scores=torch.zeros(
            (1, 1, 1), device=x.device, dtype=x.dtype
        ).expand(batch_size, seq_len, num_choices),
        softmax_probs=torch.ones(
            (1, 1, 1), device=x.device, dtype=x.dtype
        ).expand(batch_size, seq_len, num_choices),
        gates=valid_choices.to(dtype=x.dtype),
        topk_indices=torch.zeros(
            (1, 1, 1), device=x.device, dtype=torch.long
        ).expand(batch_size, seq_len, 1),
        active_mask=valid_choices,
        valid_token_mask=valid_token_mask,
    )


def _force_global_tokens_all_groups(
    router_output: RouterOutput,
    num_global: int,
    num_groups: int,
) -> RouterOutput:
    """Make the first ``num_global`` tokens members of every attention group.

    Global/register tokens (prepended in ``Nestor.forward``) must reach every
    group so they attend to all patches and every patch attends to them. We
    overwrite their group router rows: ``active_mask`` True for all groups and
    ``gates`` a uniform ``1/num_groups`` (so the per-group attention outputs are
    averaged across groups -- the global token's effective attention is over the
    union of all groups' members = the whole sequence). Patch rows are
    unchanged. No-op for single-group blocks (the lone group already holds
    everyone). The per-group gathered loop honors these directly; the top-1
    batched dispatch plan (which assumes one row per token) is disabled upstream
    when global tokens are present.
    """

    if num_global <= 0 or num_groups <= 1:
        return router_output
    active_mask = router_output.active_mask.clone()
    active_mask[:, :num_global, :] = True
    gates = router_output.gates.clone()
    gates[:, :num_global, :] = 1.0 / num_groups
    return router_output._replace(active_mask=active_mask, gates=gates)


def _squeeze_router_output(router_output: RouterOutput) -> RouterOutput:
    """Drop the singleton sequence axis from a single-token router output.

    Args:
        router_output (RouterOutput): Router output with ``seq_len == 1``.

    Returns:
        RouterOutput: Equivalent per-token output without the sequence axis.
    """

    return RouterOutput(
        scores=router_output.scores[:, 0, :],
        softmax_probs=router_output.softmax_probs[:, 0, :],
        gates=router_output.gates[:, 0, :],
        topk_indices=router_output.topk_indices[:, 0, :],
        active_mask=router_output.active_mask[:, 0, :],
        valid_token_mask=router_output.valid_token_mask[:, 0],
    )


class _ColumnStreamPlan(NamedTuple):
    """Block-level plan for column-wise residual streams (``attn_res_column``).

    Decided once per block (the group router runs once per block) and reused
    by every layer, exactly like :class:`_AttentionDispatchPlan`.

    Attributes:
        slot_of (Int[Tensor, "batch_size seq_len num_groups"]): Column ->
            stream-slot map; ``num_slots`` marks a column the token did not
            select (see :func:`_build_column_stream_plan`).
        num_slots (int): Streams per token ``k = num_groups_per_token``.
    """

    slot_of: Tensor
    num_slots: int


def _build_column_stream_plan(
    topk_indices: Int[Tensor, "batch_size seq_len top_k"],
    num_groups: int,
) -> _ColumnStreamPlan:
    """Invert the group router's top-k selection into a per-column slot map.

    Under ``block_residual_mode='attn_res_column'`` a token routed to ``k``
    attention groups ("columns") carries ``k`` parallel residual streams, one
    per SLOT ``j in [0, k)``. Slot ``j`` of token ``(b, t)`` belongs to column
    ``topk_indices[b, t, j]``. Attention, however, is computed column-major
    (one pass per group), so the scatter back into the stream axis needs the
    inverse map: given a column, which slot does this token hold it in?

    Non-selected columns map to the SENTINEL slot ``k``, so a caller can
    accumulate into a ``[batch, seq, k + 1, dim]`` buffer and drop the last
    slice -- no masking, no ``where``, and every index stays in range (columns
    a token never selected contribute exactly-zero gates anyway).

    Built ONCE per block (the group router runs once per block and every layer
    reuses its plan), so this costs one ``scatter_`` per block.

    Args:
        topk_indices (Int[Tensor, "batch_size seq_len top_k"]): Selected
            column indices per token, in slot order (may be a broadcast view).
        num_groups (int): Number of attention groups ``G`` in the block.

    Returns:
        _ColumnStreamPlan: ``slot_of[b, t, c]`` is the slot holding column
        ``c`` for token ``(b, t)``, or ``num_slots`` (the sentinel) when the
        token did not select column ``c``, together with ``num_slots == k``.
    """

    batch_size, seq_len, top_k = topk_indices.shape
    slot_of = torch.full(
        (batch_size, seq_len, num_groups),
        top_k,
        dtype=torch.long,
        device=topk_indices.device,
    )
    slots = torch.arange(
        top_k, dtype=torch.long, device=topk_indices.device
    ).expand(batch_size, seq_len, top_k)
    slot_of.scatter_(2, topk_indices.long(), slots)
    return _ColumnStreamPlan(slot_of=slot_of, num_slots=top_k)


def _to_single_slot(
    update: Float[Tensor, "batch_size seq_len dim"],
    column_plan: _ColumnStreamPlan | None,
) -> (
    Float[Tensor, "batch_size seq_len dim"]
    | Float[Tensor, "batch_size seq_len slots dim"]
):
    """Lift a single-column attention update into the stream-slot layout.

    The attention fast paths that assume ONE column per token (single-group
    layers and the top-1 batched dispatch plan) already produce that token's
    only stream, so column mode just needs the slot axis added.

    Args:
        update (Float[Tensor, "batch_size seq_len dim"]): Gated attention
            update of the token's single column.
        column_plan (_ColumnStreamPlan | None): Column plan; None passes the
            update through unchanged.

    Returns:
        Float[Tensor, "batch_size seq_len dim"] | Float[Tensor, "batch_size
        seq_len slots dim"]: The update, with a length-1 slot axis under
        column mode.
    """

    if column_plan is None:
        return update
    assert column_plan.num_slots == 1, (
        "single-column attention path reached with "
        f"num_groups_per_token={column_plan.num_slots}."
    )
    return update.unsqueeze(2)


def _column_read_sources(
    block_sources: list[Tensor],
    streams: Float[Tensor, "batch_size seq_len slots dim"],
    include_aggregate: bool,
    *,
    head_only: bool = False,
    deviation_sources: bool = False,
) -> list[Tensor]:
    """Assemble one column-mode depth read's source list.

    The completed sources (the embedding plus every earlier block's exports),
    then this block's ``k`` live column streams, then -- under
    ``block_column_include_aggregate`` -- their aggregate, the same merged
    direction ``attn_res_kimi`` would have carried as the block's only stream.
    The aggregate is the MEAN rather than the sum ``attn_res_kimi``
    accumulates. The read's key map divides every source by its own RMS, so a
    positive rescaling cancels out of both the depth logit and the mixture
    weight -- but only down to the RMSNorm epsilon floor, which stops
    mattering once a source's RMS is comfortably above ``sqrt(eps)``. The mean
    keeps the aggregate on the same scale as the streams it summarizes, so
    that floor treats every source in the read alike; the sum would sit ``k``
    times higher. It is also the tensor the block boundary already computes
    for the next group router.

    Args:
        block_sources (list[Tensor]): Completed depth sources, each
            ``[batch_size, seq_len, dim]``, in depth order.
        streams (Float[Tensor, "batch_size seq_len slots dim"]): This block's
            live per-column partial sums.
        include_aggregate (bool): Whether to append the streams' aggregate as
            one further source.

    Returns:
        list[Tensor]: ``len(block_sources) + k (+ 1)`` sources of shape
        ``[batch_size, seq_len, dim]``, in depth order.
    """

    if head_only:
        # Narrow (kimi-shaped) read: the completed aggregates plus the live
        # block's running aggregate. The mean of a single-column block's
        # streams is that column itself, so k == 1 needs no special case.
        return [*block_sources, streams.mean(dim=2)]
    if deviation_sources:
        # Aggregate-plus-differences basis for the LIVE block too, mirroring
        # the export re-basing (see block_column_deviation_sources).
        aggregate = streams.mean(dim=2)
        deviations = list(
            (streams - aggregate.unsqueeze(2)).unbind(dim=2)
        )
        return [*block_sources, *deviations, aggregate]
    columns = list(streams.unbind(dim=2))
    if not include_aggregate:
        return [*block_sources, *columns]
    return [*block_sources, *columns, streams.mean(dim=2)]


def _accumulate_group_update(
    accumulator: Float[Tensor, "batch_size seq_len *slots dim"],
    contribution: Float[Tensor, "batch_size seq_len dim"],
    group_idx: int,
    column_plan: _ColumnStreamPlan | None,
) -> Float[Tensor, "batch_size seq_len *slots dim"]:
    """Fold one attention group's gated output into the layer's accumulator.

    Without a column plan this is the usual sum over groups. With one, the
    contribution is scattered into the slot this token holds group
    ``group_idx`` in -- ``accumulator`` then has a ``num_slots + 1`` slot axis
    whose last entry is the sentinel bin for tokens that did not select the
    group (their gate is exactly zero, so the bin only ever absorbs zeros).
    Out-of-place ``scatter_add`` keeps the path functional (no buffer mutation
    for dynamo, and its backward is a plain gather).

    Args:
        accumulator (Float[Tensor, "batch_size seq_len *slots dim"]): Running
            sum ``[batch, seq, dim]``, or ``[batch, seq, num_slots + 1, dim]``
            under a column plan.
        contribution (Float[Tensor, "batch_size seq_len dim"]): This group's
            gate-weighted attention output.
        group_idx (int): Index of the attention group being folded in.
        column_plan (_ColumnStreamPlan | None): Column plan, or None for the
            plain sum.

    Returns:
        Float[Tensor, "batch_size seq_len *slots dim"]: Updated accumulator.
    """

    if column_plan is None:
        return accumulator + contribution
    # Match the dtype promotion a plain ``accumulator + contribution`` would
    # have done (autocast can hand back a lower-precision group output).
    mix_dtype = torch.promote_types(accumulator.dtype, contribution.dtype)
    accumulator = accumulator.to(mix_dtype)
    contribution = contribution.to(mix_dtype)
    batch_size, seq_len, dim = contribution.shape
    index = column_plan.slot_of[:, :, group_idx].view(
        batch_size, seq_len, 1, 1
    ).expand(batch_size, seq_len, 1, dim)
    return accumulator.scatter_add(2, index, contribution.unsqueeze(2))


def _scatter_columns_to_slots(
    per_column: Float[Tensor, "batch_size seq_len num_groups dim"],
    column_plan: _ColumnStreamPlan,
) -> Float[Tensor, "batch_size seq_len slots dim"]:
    """Route every column's update into the slot its token holds it in.

    The all-columns-at-once form of :func:`_accumulate_group_update`: same
    ``slot_of`` map, same sentinel bin for columns a token did not select, but
    one vectorized ``scatter_add`` over the whole group axis instead of ``G``
    out-of-place ones. Used where all ``G`` per-column tensors already exist
    side by side -- the expert pool's per-group private updates
    (:meth:`~nestor.backbones.expert_pools.ExpertPoolMLP.forward_per_group`),
    whose non-selected columns are exactly zero, so the dropped sentinel bin
    only ever absorbs zeros.

    Args:
        per_column (Float[Tensor, "batch_size seq_len num_groups dim"]):
            Column-indexed updates (entry ``c`` belongs to attention group
            ``c``).
        column_plan (_ColumnStreamPlan): Block-level column -> slot map.

    Returns:
        Float[Tensor, "batch_size seq_len slots dim"]: Slot-indexed updates,
        sentinel bin dropped.
    """

    batch_size, seq_len, num_groups, dim = per_column.shape
    index = column_plan.slot_of.view(
        batch_size, seq_len, num_groups, 1
    ).expand(batch_size, seq_len, num_groups, dim)
    slots = per_column.new_zeros(
        batch_size, seq_len, column_plan.num_slots + 1, dim
    ).scatter_add(2, index, per_column)
    return slots[:, :, : column_plan.num_slots]


class _AttentionDispatchPlan(NamedTuple):
    """Block-level compaction plan for batched token-dispatch attention.

    Group membership, gates, and positions are decided once per block by the
    group router, so every layer in the block shares one plan: ``cap`` slots
    per (group, image), members packed to the front in original token order.
    Rows are GROUP-major (``r = (g * B + b) * cap + c``) so the projections
    run as G large GEMMs. Requires top-1 group routing (each token owns
    exactly one real slot — the dispatch is a bijection, so both autograd
    directions are coalesced gathers; see ``_AttnDispatchGather``).

    Attributes:
        src_token (Int[Tensor, "rows"]): Flat source token (``b*S + t``) per
            row (slack rows point at arbitrary non-members; masked).
        inv_slot (Int[Tensor, "batch_seq"]): Each token's unique real row.
        valid_rows (Bool[Tensor, "rows"]): Real (non-slack) rows.
        gates_rows (Float[Tensor, "rows"]): Group gate per row (exactly 0 on
            slack rows — non-members carry zero gate).
        cos (Float[Tensor, "groups_batch 1 cap head_dim"]): RoPE cosines at
            the gathered positions (block-constant, shared by all layers).
        sin (Float[Tensor, "groups_batch 1 cap head_dim"]): RoPE sines.
        valid (Bool[Tensor, "groups_batch 1 1 cap"]): SDPA key mask.
        cap (int): Compact capacity = max members per (image, group).
    """

    src_token: Tensor
    inv_slot: Tensor
    valid_rows: Tensor
    gates_rows: Tensor
    cos: Tensor
    sin: Tensor
    valid: Tensor
    cap: int


class _AttnDispatchGather(torch.autograd.Function):
    """Token -> compact-row dispatch with gather-only autograd.

    Forward reads each row's source token (``index_select``). At top-1 group
    routing every token owns exactly ONE real row, and slack rows receive
    exactly-zero gradient (their queries/values are masked out of SDPA and
    their outputs are where-discarded), so the backward is a single gather at
    ``inv_slot`` — autograd's default ``index_add`` backward is the atomic
    ``indexFuncLargeIndex`` kernel (~4ms/call on H200 at these shapes).
    """

    @staticmethod
    def forward(ctx, x_flat, src_token, inv_slot):  # noqa: D102
        ctx.save_for_backward(inv_slot)
        return x_flat.index_select(0, src_token)

    @staticmethod
    def backward(ctx, grad_rows):  # noqa: D102
        (inv_slot,) = ctx.saved_tensors
        return grad_rows.index_select(0, inv_slot), None, None


class _AttnCombineGather(torch.autograd.Function):
    """Compact rows -> token outputs, gather-only in both directions.

    Forward reads each token's unique real row (slack rows are never read,
    so any masked-SDPA garbage there is dropped without a ``scatter_add``).
    Backward gathers the token gradient back to every row and zeroes slack
    rows.
    """

    @staticmethod
    def forward(ctx, rows, inv_slot, src_token, valid_rows):  # noqa: D102
        ctx.save_for_backward(src_token, valid_rows)
        return rows.index_select(0, inv_slot)

    @staticmethod
    def backward(ctx, grad_y):  # noqa: D102
        src_token, valid_rows = ctx.saved_tensors
        grad_rows = grad_y.index_select(0, src_token) * valid_rows.unsqueeze(
            1
        ).to(grad_y.dtype)
        return grad_rows, None, None, None



def _is_recomputing() -> bool:
    """Whether this forward is an activation checkpoint's backward recompute.

    A normal forward runs with no autograd node in flight; a checkpoint's
    recompute runs from inside the backward, under the node it is recomputing
    for. Anything with a side effect on a per-step schedule (a counter, a
    probe budget) must skip it, or the schedule advances twice per step.

    Returns:
        bool: True when called from within a backward pass.
    """

    return torch._C._current_autograd_node() is not None


# Fuse attention's and the private pool's per-sublayer scatters into one (they
# share the column -> slot map). NESTOR_COLUMN_FUSE_SCATTER=0 forces the
# separate-scatter path, which is what the equivalence test A/Bs against.
_COLUMN_FUSE_SCATTER = (
    __import__("os").environ.get("NESTOR_COLUMN_FUSE_SCATTER", "1") == "1"
)


def _activation_checkpoint_enabled() -> bool:
    """Whether sublayers should recompute their activations in the backward.

    Read per call rather than cached at import so a test can flip
    ``NESTOR_ACTIVATION_CHECKPOINT`` and compare the two paths in one process;
    the cost is one dict lookup per layer, against a whole sublayer's compute.

    Returns:
        bool: True when ``NESTOR_ACTIVATION_CHECKPOINT=1``.
    """

    import os

    return os.environ.get("NESTOR_ACTIVATION_CHECKPOINT", "0") == "1"


def _activation_checkpoint_split_enabled() -> bool:
    """Whether column layers checkpoint their two halves as separate regions.

    Under ``NESTOR_ACTIVATION_CHECKPOINT=1`` a column layer is one recompute
    region, so the backward holds one whole layer's activations (depth read +
    column attention + depth read + expert pools) at once; at 64k tokens per
    rank on the 1b column arm that transient alone is ~55 GiB.
    ``NESTOR_ACTIVATION_CHECKPOINT_SPLIT=1``
    checkpoints the attention half and the expert half separately
    (:meth:`RoutedTransformerLayer.column_attn_half` /
    :meth:`~RoutedTransformerLayer.column_expert_half`): the same ops in the
    same order, so the result is identical, at one extra boundary per layer
    (the attention update, ``[batch, seq, k, dim]``) and roughly half the
    recompute transient. Only the slot-updates column path
    (``NESTOR_COLUMN_SLOT_UPDATES=1``) is split; other layer paths keep the
    one-region checkpoint. Read per call, like the checkpoint flag.

    Returns:
        bool: True when ``NESTOR_ACTIVATION_CHECKPOINT_SPLIT=1``.
    """

    import os

    return os.environ.get("NESTOR_ACTIVATION_CHECKPOINT_SPLIT", "0") == "1"


def _activation_checkpoint_block_enabled() -> bool:
    """Whether a block's column layers are wrapped in one OUTER recompute region.

    Per-layer (or per-half, :func:`_activation_checkpoint_split_enabled`)
    checkpointing keeps one boundary per region alive across the whole
    backward -- at 64k tokens per rank on the 1b column arm, ~150-190 GiB of
    ``[batch, seq, k, dim]`` fp32 streams per rank, more than a GPU or (once
    spilled to pinned host memory) a node's share of RAM.
    ``NESTOR_ACTIVATION_CHECKPOINT_BLOCK=1`` nests
    the per-layer regions inside one region per block
    (:meth:`NestorBlock._column_layers_region`): across the backward only the
    block's entry state (its running aggregate; the streams are None at block
    entry) is retained, and the inner boundaries exist only while that block
    is being recomputed. Costs one more forward pass per layer. Exact: the
    same ops in the same order, with every layer's router noise drawn before
    the region (so the RNG stream is consumed as without it; drop-path must be
    0, which the DCLM arms satisfy). Read per call, like the checkpoint flag.

    Returns:
        bool: True when ``NESTOR_ACTIVATION_CHECKPOINT_BLOCK=1``.
    """

    import os

    return os.environ.get("NESTOR_ACTIVATION_CHECKPOINT_BLOCK", "0") == "1"


def _expert_half_chunks() -> int:
    """Sequence chunks per column expert half under split checkpointing.

    ``NESTOR_EXPERT_HALF_CHUNKS=n`` (default 1 = off) runs each column layer's
    expert half (:meth:`RoutedTransformerLayer.column_expert_half`) as ``n``
    recompute regions over ``n`` contiguous sequence slices instead of one, so
    the backward recomputes and differentiates one slice's expert dispatch at
    a time: the expert half's backward transient (its grouped-GEMM buffers and
    the ``[rows, hidden]`` activations of the private experts, ``rows = S *
    k``) shrinks by ``n`` (the 1b column arm at 64k was ~5 GiB over the card
    with every other lever on).

    Exact: the expert half is per-token end to end (depth read, norm, routers,
    expert pools, path gate, combine), the router noise is pre-drawn and
    sliced with the tokens, and the per-slice aux dicts are concatenated back
    along the sequence, so the layer's outputs, gradients and aux are those of
    the unchunked layer up to the accumulation order of the grouped GEMMs. A
    mode of the split path: ignored unless ``NESTOR_ACTIVATION_CHECKPOINT=1``
    and ``NESTOR_ACTIVATION_CHECKPOINT_SPLIT=1``; requires drop-path 0 (a
    per-sample mask drawn per slice would differ between slices).

    Returns:
        int: The chunk count (``>= 1``).
    """

    try:
        return max(1, int(os.environ.get("NESTOR_EXPERT_HALF_CHUNKS", "1")))
    except ValueError:
        return 1


def _activation_checkpoint_lean_enabled() -> bool:
    """Whether split column layers run their stream adds OUTSIDE the recompute regions.

    Under ``NESTOR_ACTIVATION_CHECKPOINT_SPLIT=1`` each half is a recompute
    region whose INPUTS are saved for the backward: the attention half saves
    the layer's entry streams and the expert half saves those streams plus
    the attention update -- two ``[batch, seq, k, dim]`` fp32 tensors per
    layer, all of a block's alive at once while the block is recomputed
    (``NESTOR_ACTIVATION_CHECKPOINT_BLOCK=1``). At 128k on the 500m column arm
    that is ~20 GiB per block on top of a peak already at the card
    (every large live tensor at that OOM was one such stream).

    A head-only column read (the paper arms' read) never reads the
    streams inside a layer: the attention half reads their MEAN and the
    expert half reads the threaded running aggregate plus the attention
    update's mean -- both ``[batch, seq, dim]``. ``NESTOR_ACTIVATION_CHECKPOINT_LEAN=1``
    therefore hands the regions those means instead of the streams
    (:meth:`RoutedTransformerLayer.column_attn_half_lean` /
    :meth:`~RoutedTransformerLayer.column_expert_half_lean`, which return the
    two updates) and performs the stream adds and the aggregate update
    outside them, as plain autograd ops whose backward saves nothing. The
    regions then retain only ``[batch, seq, dim]`` inputs, the updates are
    freed right after the adds, and a block's recompute holds one running
    stream instead of two per layer. Same ops in the same order as
    :meth:`~RoutedTransformerLayer.column_expert_half` (the means are the
    same reductions, the adds the same expressions), so the outputs and
    gradients are identical (``tests/test_checkpoint_split.py``). A mode of
    the split path: ignored unless ``NESTOR_ACTIVATION_CHECKPOINT=1`` and
    ``..._SPLIT=1``; layers with a full (Kimi) column read, which do read the
    streams inside, keep the plain split. Read per call, like the other flags.

    Returns:
        bool: True when ``NESTOR_ACTIVATION_CHECKPOINT_LEAN=1``.
    """

    return os.environ.get("NESTOR_ACTIVATION_CHECKPOINT_LEAN", "0") == "1"


def _tail_query_chunks() -> int:
    """Query slices per dense-tail attention half under the lean split.

    ``NESTOR_DENSE_TAIL_QUERY_CHUNKS=n`` (default 1 = off) runs a ``k == G``
    (dense-tail, :meth:`RoutedTransformerLayer._attn_update_dense_tail`)
    layer's attention half as one recompute region for the depth read, the
    norm and the K / V projection of the whole sequence
    (:meth:`RoutedTransformerLayer.column_tail_kv_lean`) followed by ``n``
    regions that each project the queries of one contiguous sequence slice,
    attend them causally against the K / V prefix they may see, and apply the
    output projection, the gates and the slot gather to that slice
    (:meth:`~RoutedTransformerLayer.column_tail_attn_slice_lean`). The
    backward then recomputes and differentiates one slice's attention at a
    time: the tail's transient -- its fused-QKV rows, the SDPA workspace and
    the ``[S, G, dim]`` fp32 gated / slot tensors, the item the 1b column arm
    at 128k was ~5 GiB short by with every other lever on
    -- shrinks by ``n``, at the cost
    of holding the tail's K / V (bf16) across the backward. Per-token ops
    (projections, gates, gather) are sliced exactly; causal attention of a
    query slice against its key prefix is the full attention's rows; the
    fused q/k/v GEMM becomes a q GEMM and a k/v GEMM (the same dot products,
    GEMM tiling aside). A mode of the lean split: ignored unless
    ``NESTOR_ACTIVATION_CHECKPOINT=1``, ``..._SPLIT=1`` and ``..._LEAN=1``,
    and only on dense-tail layers with a head-only read; drop-path must be 0.

    Returns:
        int: The slice count (``>= 1``).
    """

    try:
        return max(1, int(os.environ.get("NESTOR_DENSE_TAIL_QUERY_CHUNKS", "1")))
    except ValueError:
        return 1


class _SeqAssembler:
    """Assemble per-token sequence slices into one tensor without a ``cat`` transient.

    ``torch.cat(pieces, dim=1)`` holds every piece AND the result at once --
    one whole ``[batch, seq, k, dim]`` fp32 stream's worth of extra memory at
    the moment the last slice lands (5 GiB at 1b / 128k). Writing each slice
    into a preallocated buffer as it is produced frees it immediately; the
    copy is exact and autograd routes the slice's gradient back through the
    copy (``CopySlices``), saving nothing large.

    Attributes:
        seq_len (int): Full sequence length (the buffer's dim 1).
        buffer (Tensor | None): Allocated at the first slice, from its shape,
            dtype and device.
    """

    def __init__(self, seq_len: int) -> None:
        """Prepare an assembler for ``seq_len`` tokens.

        Args:
            seq_len (int): Full sequence length.
        """

        self.seq_len = seq_len
        self.buffer: Tensor | None = None

    def put(self, piece: Tensor, lo: int, hi: int) -> None:
        """Write the slice ``lo:hi`` (along dim 1) into the buffer.

        Args:
            piece (Tensor): ``[batch, hi - lo, ...]``.
            lo (int): First token of the slice.
            hi (int): One past the last token of the slice.
        """

        if self.buffer is None:
            shape = (piece.shape[0], self.seq_len, *piece.shape[2:])
            self.buffer = torch.empty(shape, dtype=piece.dtype, device=piece.device)
        self.buffer[:, lo:hi] = piece

    def result(self) -> Tensor:
        """Return the assembled tensor.

        Returns:
            Tensor: ``[batch, seq_len, ...]``.
        """

        assert self.buffer is not None, "no slice was assembled"
        return self.buffer


def _cat_aux_along_seq(parts: list[Any]) -> Any:
    """Concatenate per-sequence-slice aux structures back into one.

    Mirrors the aux layout the layers return: dicts (per key), lists / tuples
    (elementwise), tensors (``cat`` along the sequence axis 1), ``None``.
    Anything else (a scalar, a string) is taken from the first slice.

    Args:
        parts (list[Any]): One aux structure per slice, in sequence order,
            all of the same shape.

    Returns:
        Any: The merged structure.
    """

    first = parts[0]
    if isinstance(first, Tensor):
        return torch.cat(parts, dim=1)
    if isinstance(first, dict):
        return {key: _cat_aux_along_seq([part[key] for part in parts]) for key in first}
    if isinstance(first, (list, tuple)):
        merged = [_cat_aux_along_seq([part[i] for part in parts]) for i in range(len(first))]
        return type(first)(merged) if isinstance(first, tuple) else merged
    return first


def _maybe_checkpoint(sublayer: Callable[..., Any], /, **kwargs: Any) -> Any:
    """Run one sublayer, optionally trading its activations for a recompute.

    Nestor's activation memory is dominated by what the forward RETAINS, not by
    any transient: measured on a 200m column arm at batch 4, the
    forward hands 54.8 GiB to the backward and the backward's own workspace adds
    only ~12 GiB on top. Recomputing sublayers is therefore the lever that moves
    the ceiling, and offloading a transient is not.

    Applied per LAYER rather than per block: the schedule arms are 4 blocks of 4
    layers, so block granularity would give only four segments.

    Recompute is only correct here because the callers hand this a source list
    that is stable across the backward -- see :meth:`NestorBlock.forward`, which
    snapshots the backbone's growing depth-source list at block entry. The depth
    key cache is disabled alongside this flag for the same reason (its contents
    grow between blocks, so a recomputed early block would see a later block's
    sources); :class:`Nestor` enforces that.

    Args:
        sublayer (Callable[..., Any]): Bound layer method to run, e.g.
            ``layer.forward_column``.
        **kwargs (Any): Keyword arguments forwarded to ``sublayer``.

    Returns:
        Any: Whatever ``sublayer`` returns.
    """

    if not _activation_checkpoint_enabled() or not torch.is_grad_enabled():
        return sublayer(**kwargs)
    # use_reentrant=False handles non-tensor arguments and the aux dicts these
    # layers return; the reentrant implementation handles none of them.
    # preserve_rng_state stays on (the default) so drop-path and router noise
    # draw the same values on the recompute as on the forward -- BUT torch
    # only captures the CUDA generator for devices it finds among the
    # POSITIONAL arguments (``get_device_states(*args)``; kwargs are never
    # inspected). Passing the layer's tensors as keywords therefore saved no
    # CUDA RNG state at all, and every GPU recompute redrew the router noise:
    # the backward routed tokens to different experts than the forward had
    # (top-3-of-24 on the MoE control), which silently degraded every
    # checkpointed routed arm (long-context probes: +0.15 nats vs the
    # checkpoint-off twin by step 300) while the CPU path -- whose generator
    # is captured unconditionally -- stayed exact. Hand the arguments over
    # positionally so the device sweep sees them.
    keys = tuple(kwargs)

    def _call(*values: Any) -> Any:
        return sublayer(**dict(zip(keys, values)))

    return torch.utils.checkpoint.checkpoint(
        _call, *(kwargs[key] for key in keys), use_reentrant=False
    )


def _maybe_disable_compile(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Keep a function out of torch.compile unless explicitly opted in.

    torch 2.10's dynamo stalls >25 min tracing the dispatch-attention op
    family (argsort/gather/SDPA/where). The regions are
    a handful of large kernels, so eager costs little; set
    NESTOR_BATCHED_ATTN_COMPILE=1 to let them compile (e.g. on torch 2.11+).
    """

    import os

    if os.environ.get("NESTOR_BATCHED_ATTN_COMPILE") == "1":
        return fn
    return torch.compiler.disable(fn)


def _static_capacity(seq_len: int) -> int | None:
    """Capacity from shape alone, so no host sync is needed.

    ``forward_gathered`` normally derives its capacity from live routing counts
    via ``counts.max().item()``. That ``.item()`` is a GRAPH BREAK, not merely a
    sync: dynamo cannot trace it, so the model compiles as 37 fragments instead
    of a few, and Inductor cannot fuse across the boundaries. Measured on
    a 500m column arm, 14 of 18 graph breaks come from this call and
    roughly a third of GPU time is spent in elementwise kernels
    (``mul``/``copy_``/``add_``) that a contiguous graph would have folded away.

    ``NESTOR_STATIC_CAPACITY_FRAC`` sets the capacity as a fraction of the
    sequence length -- for ``G`` groups with top-1 routing and a safety factor
    ``f``, that is ``f / G`` (e.g. 8 groups at 1.5x -> 0.1875). The value is
    derived from ``seq_len``, a static shape under compile, so the break
    disappears.

    **This can drop tokens.** A group whose routed count exceeds the capacity
    loses its overflow, exactly like a capacity-factor MoE. That is a MODELLING
    choice, not a free optimisation: set the fraction high enough for the
    router's imbalance, and validate loss against an exact-capacity run before
    trusting a result produced with it.

    Args:
        seq_len (int): Sequence length; the capacity is a fraction of this.

    Returns:
        int | None: Static capacity, or ``None`` when the feature is off
        (``NESTOR_STATIC_CAPACITY_FRAC`` unset or <= 0), in which case callers
        keep the exact data-dependent capacity.
    """

    import math
    import os

    try:
        frac = float(os.environ.get("NESTOR_STATIC_CAPACITY_FRAC", "0"))
    except ValueError:
        return None
    if frac <= 0.0:
        return None
    return max(1, min(seq_len, int(math.ceil(frac * seq_len))))


def _bucket_capacity(cap: int, seq_len: int) -> int:
    """Round a data-dependent dispatch capacity up to a fixed bucket.

    ``cap`` is derived from live routing counts, so it moves whenever the
    router's load shifts. Every distinct value is a distinct tensor shape and
    therefore a fresh ``torch.compile`` specialization -- and because routing
    drifts continuously during training, the recompiles never stop. (A profiler
    that replays one fixed batch holds ``cap`` constant and so never sees this.)

    Rounding up to a multiple of ``NESTOR_GATHERED_CAP_BUCKET`` bounds the
    distinct shapes to ``seq_len / bucket`` -- 32 at the default 128 with
    ``seq_len`` 4096 -- so the variants compile once during warmup and are
    reused afterwards. Keep that count comfortably under the dynamo recompile
    limit (64 in the training pipeline), or exhausting it silently drops the
    model back to eager.

    No token is dropped: capacity only ever grows, and every caller already
    masks the padded slots through its ``valid`` mask. The cost is the padding
    carried in those slots, so the bucket trades a few percent of wasted
    attention width against unbounded recompilation.

    Args:
        cap (int): Exact capacity required (max routed tokens per image/group).
        seq_len (int): Sequence length; capacity never usefully exceeds it.

    Returns:
        int: Bucketed capacity, in ``[cap, seq_len]``. Returns ``cap``
        unchanged when the bucket is disabled (``NESTOR_GATHERED_CAP_BUCKET``
        <= 1) or when ``cap`` is 0, which callers treat as "group is empty".
    """

    import os

    bucket = int(os.environ.get("NESTOR_GATHERED_CAP_BUCKET", "128"))
    if bucket <= 1 or cap <= 0:
        return cap
    return min(((cap + bucket - 1) // bucket) * bucket, seq_len)


class _PackedAttentionPlan(NamedTuple):
    """Block-level packing plan for layer-packed varlen attention.

    The layer-packed path (:meth:`RoutedTransformerLayer.
    _attn_update_packed_varlen`) lays every group's routed tokens out in ONE
    static ``[rows, dim]`` buffer (group-major, image-major, original token
    order, each group followed by a <=16-row alignment tail; the last tail
    absorbs the slack so ``rows`` is static). Which token sits in which packed
    row is decided by the block's group router, so all of it is computed once
    per block here and reused by every layer -- the same economy as
    :class:`_AttentionDispatchPlan`, and it turns the per-layer pack into a
    single gather. Every index below is static-shaped; nothing here syncs
    with the host.

    Attributes:
        cu (Int[Tensor, "segments_plus_one"]): int32 varlen segment
            boundaries: per group, ``B`` member segments then its alignment
            tail as a segment of its own, so ``cu[-1] == rows`` and EVERY
            packed row belongs to some segment (no row is ever left as
            uninitialized kernel output).
        offs (Int[Tensor, "num_groups"]): int32 grouped-GEMM offsets (the
            aligned end of each group's rows).
        gid (Int[Tensor, "rows"]): Group of every packed row (tails included),
            for per-group bias adds.
        tok_packed (Int[Tensor, "rows"]): Flat source token ``b*S + t`` of
            every packed row (tail rows clamped onto a real token; see
            ``row_real``).
        row_real (Bool[Tensor, "rows"]): Real (non-tail) rows. The pack
            zeroes tail rows with it, which is load-bearing: a tail row's
            attention output is gated to exactly zero, but whatever the
            varlen backward leaves in its dq/dk/dv (zeros if it writes them,
            allocator garbage if it does not) multiplies ``x_packed`` into
            the projection WEIGHT gradients -- with zero tail rows that
            product is exactly zero whatever the kernel does.
        slot_rows (Int[Tensor, "batch_seq_slots"]): Inverse map: the packed
            row of every ``(token, slot)`` pair, ``slot`` being the column's
            position in the token's top-k (== the column-stream slot). Pairs
            without a row (padding tokens) point at ``rows - 1``, which is
            always a tail row of the last group (zero gate -> exactly zero).
        slot_packed (Int[Tensor, "rows"]): The ``(token, slot)`` row of every
            packed row, clamped for tail rows (whose gradient is zeroed by
            the gate multiply before it reaches anything).
        gates_packed (Float[Tensor, "rows"]): fp32 group gate per row
            (exactly zero on tails).
        cos (Float[Tensor, "rows 1 head_dim"]): RoPE cosines at the packed
            positions (fp32, mirrors ``RoPE.forward``).
        sin (Float[Tensor, "rows 1 head_dim"]): RoPE sines.
        rows (int): Static packed row count ``B*S*k + 16*G``.
        num_slots (int): Slots per token ``k`` (the router's top-k).
    """

    cu: Tensor
    offs: Tensor
    gid: Tensor
    tok_packed: Tensor
    row_real: Tensor
    slot_rows: Tensor
    slot_packed: Tensor
    gates_packed: Tensor
    cos: Tensor
    sin: Tensor
    rows: int
    num_slots: int


class _PackedRowGather(torch.autograd.Function):
    """Token rows -> packed rows, gather-only in both directions.

    Forward replicates each token into its (at most ``k``) packed rows with
    one ``index_select`` and zeroes the alignment-tail rows (see
    :attr:`_PackedAttentionPlan.row_real` for why that is load-bearing).
    Backward reads each token's ``k`` packed-row gradients through the
    inverse map and sums them -- a coalesced gather instead of autograd's
    atomic ``index_add`` over ``B*S*k`` rows. Pairs without a row read a
    tail row: its attention output is gated to exactly zero, so its
    gradient contributes nothing.
    """

    @staticmethod
    def forward(ctx, x_flat, tok_packed, row_real, slot_rows, num_slots):  # noqa: D102
        ctx.save_for_backward(slot_rows)
        ctx.num_slots = num_slots
        ctx.num_tokens = x_flat.shape[0]
        return x_flat.index_select(0, tok_packed) * row_real.unsqueeze(1).to(
            x_flat.dtype
        )

    @staticmethod
    def backward(ctx, grad_packed):  # noqa: D102
        (slot_rows,) = ctx.saved_tensors
        grad_x = grad_packed.index_select(0, slot_rows).view(
            ctx.num_tokens, ctx.num_slots, -1
        ).sum(1)
        return grad_x, None, None, None, None


class _PackedSlotGather(torch.autograd.Function):
    """Gated packed rows -> ``(token, slot)`` rows, gather-only both ways.

    Forward reads each ``(token, slot)`` pair's packed row, so the result is
    ALREADY the column streams' slot layout (or, summed over slots, the
    gate-weighted group sum). Backward gathers the pair gradient back to
    each packed row through :attr:`_PackedAttentionPlan.slot_packed`; tail
    rows read a clamped entry, which the gate multiply upstream (exactly
    zero on tails) turns into an exact zero before it reaches a weight.
    RELIES ON that gate: the gate multiply must stay between the projection
    and this gather.
    """

    @staticmethod
    def forward(ctx, gated, slot_rows, slot_packed):  # noqa: D102
        ctx.save_for_backward(slot_packed)
        return gated.index_select(0, slot_rows)

    @staticmethod
    def backward(ctx, grad_slots):  # noqa: D102
        (slot_packed,) = ctx.saved_tensors
        return grad_slots.index_select(0, slot_packed), None, None


@_maybe_disable_compile
def _build_packed_attention_plan(
    group_router_output: RouterOutput,
    pos: Int[Tensor, "batch_size seq_len"],
    rope: "RoPE",
    num_groups: int,
) -> _PackedAttentionPlan:
    """Build the per-block :class:`_PackedAttentionPlan`.

    Static shapes throughout (``rows = B*S*k + 16*G`` is the worst case, which
    top-k routing can never exceed); segment boundaries and row maps live in
    tensors, so nothing here needs a host sync. Tail rows read a clamped real
    token index but are zeroed by the pack (``row_real``). Runs outside torch.compile
    like :func:`_build_attention_dispatch_plan` (same dispatch-op family).

    Args:
        group_router_output (RouterOutput): The block's group routing
            (``active_mask``, ``gates`` and ``topk_indices`` are read).
        pos (Int[Tensor, "batch_size seq_len"]): Token positions (RoPE).
        rope (RoPE): Any group's rotary module (uniform groups share one
            ``inv_freq`` table).
        num_groups (int): Attention groups ``G`` in the block.

    Returns:
        _PackedAttentionPlan: The plan every layer of the block reuses.
    """

    active = group_router_output.active_mask.permute(2, 0, 1)  # [G, B, S]
    gates = group_router_output.gates.permute(2, 0, 1)  # [G, B, S]
    topk_indices = group_router_output.topk_indices  # [B, S, k]
    batch_size, seq_len, top_k = topk_indices.shape
    num_tokens = batch_size * seq_len
    device = pos.device
    align = 16  # grouped-GEMM offset alignment
    rows = num_tokens * top_k + align * num_groups

    counts = active.sum(dim=-1)  # [G, B] members per (group, image)
    group_rows = counts.sum(dim=-1)  # [G]
    group_pad = (-group_rows) % align  # [G] alignment tail per group
    # Per group: B member segments then the alignment tail as its own
    # segment; the LAST group's tail absorbs the residual up to ``rows`` so
    # ``cu[-1] == rows`` always (segment_mm's contract, and no orphan rows).
    seg_counts = torch.cat(
        [counts, group_pad.unsqueeze(-1)], dim=-1
    ).clone()  # [G, B+1]
    seg_counts[-1, -1] += rows - seg_counts.sum()
    cu = torch.zeros(
        num_groups * (batch_size + 1) + 1, device=device, dtype=torch.int32
    )
    torch.cumsum(seg_counts.reshape(-1).to(torch.int32), dim=0, out=cu[1:])
    offs = cu[1:].reshape(num_groups, batch_size + 1)[:, -1].contiguous()
    seg_start = cu[:-1].reshape(num_groups, batch_size + 1)[
        :, :batch_size
    ]  # [G, B]

    # Members-first stable order keeps original token order per image, so
    # every packed segment is causally ordered.
    order = torch.argsort(
        active.to(torch.int8), dim=-1, descending=True, stable=True
    )  # [G, B, S]
    slot = torch.arange(seq_len, device=device)[None, None, :]
    real = slot < counts.unsqueeze(-1)  # [G, B, S]
    dest = torch.where(
        real,
        seg_start.unsqueeze(-1).long() + slot,
        torch.full_like(order, rows),  # scratch entry, sliced off
    )  # [G, B, S] packed destination of every members-first slot
    tok_ord = (
        torch.arange(batch_size, device=device)[None, :, None] * seq_len
        + order
    )  # [G, B, S] flat token of every ordered slot
    # Which stream slot the token holds this group in (sentinel k for
    # non-members, which are never real rows).
    slot_of = _build_column_stream_plan(topk_indices, num_groups).slot_of
    slot_ord = torch.gather(slot_of.permute(2, 0, 1), 2, order)  # [G, B, S]
    slotrow_ord = tok_ord * top_k + slot_ord
    pos_ord = torch.gather(pos.unsqueeze(0).expand(num_groups, -1, -1), 2, order)
    gates_ord = torch.gather(gates, 2, order)

    def pack(src: Tensor, fill: float | int) -> Tensor:
        """Scatter members-first slots into a ``rows``-long buffer."""

        buf = torch.full((rows + 1,), fill, dtype=src.dtype, device=device)
        buf.scatter_(0, dest.reshape(-1), src.reshape(-1))
        return buf[:rows]

    tok_packed = pack(tok_ord, num_tokens)
    row_real = tok_packed < num_tokens  # tail rows keep the fill value
    slot_packed = pack(slotrow_ord, num_tokens * top_k)
    gates_packed = pack(gates_ord.to(torch.float32), 0.0)
    pos_packed = pack(pos_ord, 0)

    # Inverse map (token, slot) -> packed row. Pairs without a row keep the
    # scratch target ``rows - 1``: the last group's tail has >= 16 rows by
    # construction, so that row is always a zero-gate tail row.
    slot_rows = torch.full(
        (num_tokens * top_k + 1,), rows - 1, dtype=torch.long, device=device
    )
    slot_rows.scatter_(0, slot_packed, torch.arange(rows, device=device))
    slot_rows = slot_rows[: num_tokens * top_k]

    # Row -> group from the aligned group ends (rows are group-major).
    gid = torch.searchsorted(
        offs.long(), torch.arange(rows, device=device), right=True
    )

    # RoPE tables at the packed positions, mirroring RoPE.forward exactly.
    angles = (
        pos_packed.clamp_min(0).to(torch.float32).unsqueeze(-1) * rope.inv_freq
    )
    emb = torch.cat((angles, angles), dim=-1)  # [rows, head_dim]
    return _PackedAttentionPlan(
        cu=cu,
        offs=offs,
        gid=gid,
        tok_packed=tok_packed.clamp(max=num_tokens - 1),
        row_real=row_real,
        slot_rows=slot_rows,
        slot_packed=slot_packed.clamp(max=num_tokens * top_k - 1),
        gates_packed=gates_packed,
        cos=emb.cos().unsqueeze(1),
        sin=emb.sin().unsqueeze(1),
        rows=rows,
        num_slots=top_k,
    )


@_maybe_disable_compile
def _build_attention_dispatch_plan(
    group_router_output: RouterOutput,
    pos: Int[Tensor, "batch_size seq_len"],
    rope: "RoPE",
) -> _AttentionDispatchPlan:
    """Build the per-block compaction plan for batched dispatch attention.

    ONE ``.item()`` host sync per block (the capacity), instead of one per
    (layer, group) in the per-group gathered path — for the g4 mtx model
    that's 3 syncs per forward instead of 24. RoPE cos/sin are precomputed
    here once (positions are block-constant) instead of per layer per q/k.
    Runs outside torch.compile (same torch-2.10 dynamo stall family as
    ``forward_gathered_full``; the ``.item()`` would graph-break anyway).
    """

    members = group_router_output.active_mask.permute(0, 2, 1)  # [B, G, S]
    gates = group_router_output.gates.permute(0, 2, 1)  # [B, G, S]
    batch_size, num_groups, seq_len = members.shape
    device = pos.device
    counts = members.sum(dim=-1)  # [B, G]
    cap = max(int(counts.max().item()), 1)  # >=1 keeps shapes non-empty
    cap = _bucket_capacity(cap, seq_len)  # bound distinct shapes; see helper
    # Stable members-first order preserves original token order (causal-safe).
    order = torch.argsort(
        members.to(torch.int8), dim=-1, descending=True, stable=True
    )
    gather_idx = order[..., :cap]  # [B, G, cap]
    valid = (
        torch.arange(cap, device=device)[None, None, :]
        < counts.unsqueeze(-1)
    )  # [B, G, cap]

    # Group-major row space: r = (g*B + b)*cap + c.
    flat_token = (
        torch.arange(batch_size, device=device).view(-1, 1, 1) * seq_len
        + gather_idx
    )  # [B, G, cap], values in [0, B*S)
    src_token = flat_token.permute(1, 0, 2).reshape(-1)  # [R]
    valid_rows = valid.permute(1, 0, 2).reshape(-1)  # [R]
    gates_rows = (
        torch.gather(gates, 2, gather_idx).permute(1, 0, 2).reshape(-1)
    )  # [R]

    # Each token's unique real row (top-1 routing). Slack rows scatter into a
    # trash slot so they can never clobber a real token's entry.
    rows = src_token.shape[0]
    row_ids = torch.arange(rows, device=device)
    target = torch.where(
        valid_rows, src_token, torch.full_like(src_token, batch_size * seq_len)
    )
    inv_slot = torch.zeros(
        batch_size * seq_len + 1, dtype=torch.long, device=device
    ).scatter_(0, target, row_ids)[: batch_size * seq_len]

    # Block-constant RoPE tables at the gathered positions, in the SDPA
    # layout [G*B, 1, cap, head_dim] (broadcast over heads). Mirrors
    # RoPE.forward exactly: angles from clamped positions, fp32 tables.
    pos_c = torch.gather(
        pos.unsqueeze(1).expand(batch_size, num_groups, -1), 2, gather_idx
    )  # [B, G, cap]
    pos_g = pos_c.permute(1, 0, 2).reshape(num_groups * batch_size, cap)
    angles = (
        pos_g.clamp_min(0).to(torch.float32).unsqueeze(-1) * rope.inv_freq
    )
    emb = torch.cat((angles, angles), dim=-1)  # [G*B, cap, head_dim]
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)

    return _AttentionDispatchPlan(
        src_token=src_token,
        inv_slot=inv_slot,
        valid_rows=valid_rows,
        gates_rows=gates_rows,
        cos=cos,
        sin=sin,
        valid=valid.permute(1, 0, 2).reshape(
            num_groups * batch_size, 1, 1, cap
        ),
        cap=cap,
    )


# ---------------------------------------------------------------------------
# Auxiliary output / KV cache containers.
# ---------------------------------------------------------------------------


class NestorBlockOutput(NamedTuple):
    """Output produced by one :class:`NestorBlock` forward pass.

    Attributes:
        x (Float[Tensor, "batch_size seq_len output_dim"]): Block output
            tokens after pooling and projection. When the top-level block
            AttnRes owns the boundary transition, this is instead the raw
            position-aligned block output at ``input_dim``.
        pos (Int[Tensor, "batch_size seq_len"]): Pooled token positions.
        attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Attention
            mask for the next block.
        aux (dict[str, Any]): Block-level routing/validity metadata.
        summary (Tensor | None): Under ``residual_mode='attn_res_xblock'``, this
            block's global summary vector ``[batch, summary_dim]`` fed to later
            blocks' cross-block depth attention; ``None`` otherwise.
        streams (Tensor | None): Under
            ``block_residual_mode='attn_res_column'``, this block's ``k``
            column-wise residual streams ``[batch, seq_len, k, output_dim]``,
            exported as separate depth sources; ``None`` otherwise. ``x`` then
            carries their mean, which only feeds the next block's group router.
            Under ``block_column_include_aggregate`` that mean is appended as a
            further slot, making the exported count ``k + 1``.
    """

    x: Float[Tensor, "batch_size seq_len output_dim"]
    pos: Int[Tensor, "batch_size seq_len"]
    attn_mask: Bool[Tensor, "batch_size seq_len seq_len"]
    aux: dict[str, Any]
    summary: Tensor | None = None
    streams: Tensor | None = None


@dataclass
class AttentionGroupKVCache:
    """Cached keys, values, and group-activity mask for one attention group.

    Attributes:
        keys (Tensor): Cached key projections.
        values (Tensor): Cached value projections.
        active_mask (Tensor): Per-position mask of whether the group was active
            when each cached token arrived.
        length (int): Number of valid cached time steps currently stored.
    """

    keys: Tensor
    values: Tensor
    active_mask: Tensor
    length: int = 0


@dataclass
class RoutedTransformerLayerKVCache:
    """Cached attention state for one routed transformer layer.

    Attributes:
        group_caches (list[AttentionGroupKVCache]): One cache per attention
            group in the layer.
    """

    group_caches: list[AttentionGroupKVCache]


@dataclass
class LocalPoolingKVCache:
    """Running state for fixed-stride local pooling between blocks.

    Attributes:
        token_sum (Tensor): Running sum of token embeddings in the current
            pooling window.
        token_count (Tensor): Number of valid tokens accumulated in the window.
        position_sum (Tensor): Running sum of token positions in the window.
    """

    token_sum: Tensor
    token_count: Tensor
    position_sum: Tensor


@dataclass
class NestorBlockKVCache:
    """Cached state for one :class:`NestorBlock`.

    Attributes:
        layer_caches (list[RoutedTransformerLayerKVCache]): One cache per
            routed transformer layer in the block.
        pooling_cache (LocalPoolingKVCache | None): Local-pooling window state
            between this block and the next.
    """

    layer_caches: list[RoutedTransformerLayerKVCache]
    pooling_cache: LocalPoolingKVCache | None = None


@dataclass
class NestorKVCache:
    """Full hierarchical KV cache for incremental causal decoding.

    Attributes:
        block_caches (list[NestorBlockKVCache]): One cache per block.
    """

    block_caches: list[NestorBlockKVCache]


# ---------------------------------------------------------------------------
# Per-block config and top-level config.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(kw_only=True)
class NestorBlockConfig:
    """Concrete configuration for one hierarchical Nestor block.

    Attributes:
        input_dim (int): Token feature dimension entering the block.
        output_dim (int): Token feature dimension after projection.
        pooling_stride (int): Local pooling stride between this block and the
            next one.
        block_num_layers (int): Number of routed transformer layers.
        num_groups (int): Number of attention groups in the group router.
        num_groups_per_token (int): Top-k attention groups per token.
        block_num_heads (int): Attention heads per group.
        block_head_dim (int): Width of each attention head.
        block_mlp_hidden_dims (list[int]): Hidden widths for expert MLPs.
        block_mlp_activation (str): Activation for expert MLPs.
        router_num_experts_per_group (int): Total experts seen by each group.
        router_num_shared_experts (int): Experts shared across groups.
        router_num_experts_per_group_per_token (int): Top-k experts per token.
        router_score_mlp_hidden_dims (list[int]): Hidden widths for the router
            score MLP.
        router_score_mlp_activation (str): Activation for the score MLP.
        router_noise_mlp_hidden_dims (list[int]): Hidden widths for the router
            noise MLP.
        router_noise_mlp_activation (str): Activation for the noise MLP.
        loss_free_balancing (bool): Whether to use detached bias updates.
        bias_update_rate (float): Step size for detached router bias updates.
    """

    input_dim: int
    output_dim: int
    pooling_stride: int
    block_num_layers: int
    num_groups: int
    num_groups_per_token: int
    block_num_heads: int
    block_head_dim: int
    block_mlp_hidden_dims: list[int]
    block_mlp_activation: str
    router_num_experts_per_group: int
    router_num_shared_experts: int
    router_num_experts_per_group_per_token: int
    router_score_mlp_hidden_dims: list[int]
    router_score_mlp_activation: str
    router_noise_mlp_hidden_dims: list[int]
    router_noise_mlp_activation: str
    block_rope_base: float = 10000.0
    loss_free_balancing: bool = False
    bias_update_rate: float = 0.0
    pooling_keep_ratio: float | None = None
    attention_token_dispatch: bool = False
    # DeepSeekMoE-style always-on experts: applied to EVERY token with unit
    # gate, outside the router entirely (unlike router_num_shared_experts,
    # whose "shared" experts are routed choices shared across groups).
    router_num_dense_shared_experts: int = 0
    # Per-layer routed-expert counts (length == block_num_layers). None falls
    # back to a uniform router_num_experts_per_group for every layer; set to a
    # list to build a depth-varying MoE sparsity schedule within the block.
    layer_num_experts: list[int] | None = None
    # Public/private expert pools: when the private expert count is > 0
    # every layer in this block replaces its routed expert banks with an
    # ExpertPoolMLP (public pool shared by all tokens + one private pool per
    # attention group + a 2-logit path gate). A 0 PUBLIC count is the
    # private-only mode: private pools only, no path gate, private experts
    # take the whole 4d active budget. 0/0 (default) disables.
    expert_pool_public_experts: int = 0
    expert_pool_public_top_k: int = 1
    expert_pool_private_experts: int = 0
    expert_pool_private_top_k: int = 1
    expert_pool_private_gate: str = "renorm"
    expert_pool_active_budget: float | None = None
    expert_pool_public_budget: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the block config.

        Returns:
            dict[str, Any]: Dictionary form of the config.
        """

        return dataclasses.asdict(self)


@dataclasses.dataclass(kw_only=True)
class NestorConfig(BackboneConfig):
    """Top-level configuration for the hierarchical Nestor backbone.

    Attributes:
        backbone_type (Literal["nestor"]): Registry key for this backbone.
        input_dim (int): Adapter output dimension consumed by the first block.
        num_blocks (int): Number of hierarchical blocks.
        feature_dims (list[int]): Output feature dimension for each block.
        pooling_strides (int | list[int]): Pooling stride for each block.
        causal (bool): Whether to enable prefix-stable causal execution.
        block_num_layers (int | list[int]): Routed layers per block.
        num_groups (int | list[int]): Attention groups per block.
        num_groups_per_token (int | list[int]): Active groups per token.
        block_num_heads (int | list[int]): Attention heads per group.
        block_head_dims (int | list[int]): Attention head width per block.
        block_mlp_hidden_dims (list[int] | list[list[int]]): Expert MLP hidden
            widths.
        block_mlp_activations (str | list[str]): Expert MLP activations.
        router_num_experts_per_group (int | list[int]): Experts per group.
        router_num_shared_experts (int | list[int]): Shared experts per block.
        router_num_experts_per_group_per_token (int | list[int]): Top-k experts
            per token.
        router_score_mlp_hidden_dims (list[int] | list[list[int]]): Score-router
            MLP hidden widths.
        router_score_mlp_activations (str | list[str]): Score-router activations.
        router_noise_mlp_hidden_dims (list[int] | list[list[int]]): Noise-router
            MLP hidden widths.
        router_noise_mlp_activations (str | list[str]): Noise-router activations.
        router_loss_free_balancing (bool | list[bool]): Whether detached bias
            updates are enabled.
        router_bias_update_rate (float | list[float]): Detached bias update
            step size.
        block_expert_pool_public_experts (int | list[int]): Public expert-pool
            size per block; 0 (default) drops the public path -- with a
            private pool that is the private-only mode, without one it
            disables pooling.
        block_expert_pool_public_top_k (int | list[int]): Public experts
            active per token.
        block_expert_pool_private_experts (int | list[int]): Private
            expert-pool size per attention group; 0 disables pooling for the
            block (default), and is illegal alongside a public pool.
        block_expert_pool_active_budget (float | None): Active expert
            intermediate width in multiples of ``input_dim``, overriding the
            pool's ``2d``/``4d`` contract. Lets a model that spends less than
            the dense attention FLOP budget hand the refund to wider experts
            instead of leaving it unspent.
        block_expert_pool_public_budget (float | None): The PUBLIC family's
            active width in multiples of ``input_dim`` (``None`` = the ``2d``
            contract). With ``block_expert_pool_active_budget`` this rescales
            the whole hybrid MLP (1.25 + 1.25 = the "2.5d" MLP).
        block_expert_pool_private_top_k (int | list[int]): Private experts
            active per selected attention group.
        block_expert_pool_private_gate (str): Gate rule of the column-private
            pool routers: ``"renorm"`` (softmax over the selected ``k_q``
            scores; for ``k_q = 1`` the gate is the constant 1.0 and the
            private routers receive NO gradient) or ``"switch"`` (the selected
            expert's full-softmax probability, which keeps a top-1 private
            router trainable).
        residual_mode (str): Within-block residual topology.
        block_residual_mode (str): Between-block transition topology,
            ``"projection"``, writeup-faithful ``"attn_res"``, or Kimi-paper-
            faithful ``"attn_res_kimi"`` (arXiv:2603.15031 Eq. 3-6).
        block_attn_res_include_embedding (bool): Whether block AttnRes retains
            the input embedding as an additional depth source.
        block_column_include_aggregate (bool): Under ``"attn_res_column"``,
            whether every multi-column block also exports the aggregate of its
            ``k`` streams, so it contributes ``k + 1`` depth sources.
        block_column_gain_init (float): Initial value gain of the COLUMN depth
            sources (embedding, aggregates, and single-column blocks keep
            1.0). With the aggregate flag and 0.0 here, every read starts at
            exactly the ``attn_res_kimi`` mixture and learns the columns from
            silence. Default 1.0 (plain uniform init).
        block_column_mass_lambda (float | None): Mass-conserving alternative to
            the constant init: each multi-column block's init mass is fixed at
            1 and split as ``1 - lambda`` on its aggregate and ``lambda / k``
            on each column, making the init invariant to the block's column
            granularity. ``0.0`` degenerates to ``block_column_gain_init=0.0``.
            None (default) disables.
    """

    backbone_type: Literal["nestor"] = "nestor"
    input_dim: int
    num_blocks: int
    feature_dims: list[int]
    pooling_strides: int | list[int] = 1
    # Optional fractional token pooling: keep this fraction of tokens at each
    # block's exit (adaptive average pool). None/1.0 = no fractional pooling.
    pooling_keep_ratios: float | list[float] | None = None
    causal: bool = False
    block_num_layers: int | list[int] = 1
    num_groups: int | list[int] = 1
    num_groups_per_token: int | list[int] = 1
    block_num_heads: int | list[int] = 1
    block_head_dims: int | list[int] = 64
    block_rope_bases: float | list[float] = 10000.0
    block_mlp_hidden_dims: list[int] | list[list[int]] = dataclasses.field(
        default_factory=list
    )
    block_mlp_activations: str | list[str] = "gelu"
    router_num_experts_per_group: int | list[int] = 1
    # Per-layer expert-count schedule WITHIN each block. None (default) keeps
    # every layer uniform at router_num_experts_per_group. When set, the outer
    # list is per block and the inner list is per layer (length ==
    # block_num_layers[block]); layer i of that block is built with
    # block_layer_num_experts[block][i] routed experts. This is how a
    # depth-varying MoE sparsity schedule is expressed: top-k and the per-expert
    # width are held constant, so active params/FLOPs are unchanged, while the
    # total expert count (hence total params) is sum(schedule). Keep
    # sum(schedule) equal across configs to param-match. A single-block decoder
    # passes a one-element outer list, e.g. [[12, 13, ..., 32]] for L layers.
    block_layer_num_experts: list[list[int]] | None = None
    router_num_shared_experts: int | list[int] = 0
    # DeepSeekMoE-style always-on experts per layer (unit gate, no routing).
    router_num_dense_shared_experts: int | list[int] = 0
    router_num_experts_per_group_per_token: int | list[int] = 1
    router_score_mlp_hidden_dims: list[int] | list[list[int]] = (
        dataclasses.field(default_factory=list)
    )
    router_score_mlp_activations: str | list[str] = "gelu"
    router_noise_mlp_hidden_dims: list[int] | list[list[int]] = (
        dataclasses.field(default_factory=list)
    )
    router_noise_mlp_activations: str | list[str] = "gelu"
    router_loss_free_balancing: bool | list[bool] = False
    router_bias_update_rate: float | list[float] = 0.0
    # If True, attention gathers each group's member tokens before projecting
    # (compute scales with groups-per-token, not total groups) instead of
    # projecting the full sequence and masking. Numerically equivalent.
    attention_token_dispatch: bool | list[bool] = False
    # Public/private expert pools per block: every layer of
    # a pool-enabled block replaces its expert banks with an ExpertPoolMLP --
    # a public pool of `public_experts` (top-`public_top_k`) shared by all
    # tokens, one private pool of `private_experts` (top-`private_top_k` inside
    # each of the token's active attention groups), and a learned 2-logit path
    # gate mixing the two always-on paths. Expert widths keep the SP-50 active
    # contract (public 2d/k_p, private 2d/(k_a*k_q), each rounded to a multiple
    # of 8), so active MLP FLOPs stay ~ a dense 4d SwiGLU while stored capacity
    # multiplies with the pool sizes. Setting public_experts to 0 alongside a
    # private pool is the PRIVATE-ONLY mode: no public path and no path gate,
    # and the private experts widen to 4d/(k_a*k_q) so the active budget is
    # unchanged. 0 private experts (default) disables pooling, and is illegal
    # alongside a public pool (public-only is not implemented); a pool-enabled
    # block must be "nomoe" (1 routed expert, no dense shared experts) so no
    # configured expert is silently discarded. Every pool router family uses
    # loss-free balancing when router_loss_free_balancing is set.
    block_expert_pool_public_experts: int | list[int] = 0
    block_expert_pool_public_top_k: int | list[int] = 1
    block_expert_pool_private_experts: int | list[int] = 0
    block_expert_pool_private_top_k: int | list[int] = 1
    block_expert_pool_private_gate: str = "renorm"
    block_expert_pool_active_budget: float | None = None
    block_expert_pool_public_budget: float | None = None
    # When set, each block's expert MLP hidden width is auto-derived as
    # round(mlp_auto_dense_ratio * feature_dims[block] / top_k[block]) and
    # overrides block_mlp_hidden_dims (top_k = num_experts_per_group_per_token).
    # This holds active expert FLOPs equal to a dense (ratio x) FFN at any top_k
    # -- the DeepSeek/Kimi fine-grained-expert recipe. None = use the explicit
    # block_mlp_hidden_dims. (e.g. ratio=4.0 reproduces the nestorv42* configs.)
    mlp_auto_dense_ratio: float | None = None
    # When True, each block's per-group head count is auto-derived as
    # round((feature_dims[block] / head_dim) / num_groups_per_token[block]) and
    # overrides block_num_heads. This splits a block's full head budget across
    # the active groups so active attention compute matches a dense block at any
    # top_k (= num_groups_per_token) -- the attention analogue of
    # mlp_auto_dense_ratio. At top_k=1 it is a no-op (full heads per group).
    attention_auto_head_split: bool = False
    # Residual topology. "convex" (default) uses the convex residual
    # (1-1/D) h + (1/D) f(h); "attn_res" replaces it with Kimi's Attention
    # Residuals (arXiv:2603.15031) -- learned softmax attention over depth within
    # each block; "attn_res_xblock" additionally attends over preceding blocks'
    # global summaries (cross-block depth attention; non-causal only). Model-
    # global for now.
    residual_mode: str = "convex"
    # Between-block transition. "projection" preserves the original single
    # projection from the most recent block. "attn_res" implements Eq. battn
    # from the latest writeup: per-token attention over every earlier block
    # output, with per-source key/value maps into the target width.
    # "attn_res_kimi" implements the Kimi paper's Block AttnRes
    # (arXiv:2603.15031 Eq. 3-6) with this repo's muP adaptation: EVERY
    # sub-layer reads a depth-attention mixture of the completed block
    # representations (gain-scaled unit-RMS values, weight-free RMSNorm keys,
    # 1/sqrt(d) depth logits, one zero-init pseudo-query per sub-layer), the
    # running within-block partial SUM replaces the residual stream, and a
    # final read over all block representations feeds the head. See
    # DepthAttentionRead for why the paper-verbatim read loses here. Constant
    # width / stride-1 / residual_mode="convex" only; training forward only.
    # "attn_res_column" is the Nestor-specific generalization of
    # "attn_res_kimi": a block routed to k = num_groups_per_token attention
    # groups ("columns") carries k PARALLEL partial sums instead of one, each
    # attention column writing only into its own stream, and exports all k of
    # them as separate depth-read sources at the block boundary (see
    # RoutedTransformerLayer.forward_column). Same structural constraints as
    # "attn_res_kimi", plus at least one block with k > 1 (at k == 1
    # everywhere it IS "attn_res_kimi"). Pair it with
    # block_column_include_aggregate to keep the merged block-wise residual
    # beside the k column-wise ones.
    block_residual_mode: str = "projection"
    # The writeup makes the embedding O^(0) optional. When True, retain it as
    # an additional source at every block boundary (including the head).
    block_attn_res_include_embedding: bool = True
    # Under "attn_res_column", also keep the AGGREGATED block-wise residual
    # beside the k column-wise ones, so a block routed to k columns contributes
    # k + 1 depth sources instead of k: its k streams plus their mean. Column
    # mode alone is a re-parameterization of "attn_res_kimi" rather than a
    # superset of it -- splitting a block into k separately RMS-normalized
    # sources means no setting of the read's parameters recovers the merged
    # block direction Kimi reads, because the read normalizes each source on its
    # own. Exporting the aggregate too puts that direction back in the source
    # list, so every read (within-block sublayer reads and the head read alike)
    # can address the merged block AND its individual columns. Blocks with
    # k == 1 are left alone: their aggregate IS their single stream, so the
    # extra source would be an exact duplicate. Costs one extra source per
    # multi-column block per read; the aggregate is a mean over the slot axis,
    # free at the block boundary (it is the tensor the next group router already
    # consumes) and one reduction per read within a block.
    block_column_include_aggregate: bool = False
    # Initial value gain of the COLUMN sources in every depth read (embedding,
    # aggregate, and single-column-block sources keep 1.0). The default 1.0 is
    # the plain uniform init. Small values counter the simplex dilution that
    # extra sources cost at initialization: with
    # ``block_column_include_aggregate=True`` and this at 0.0, every read's
    # mixture starts EXACTLY at what ``attn_res_kimi`` would compute (a zeroed
    # source contributes nothing, and softmax mass stolen by zero-gain sources
    # only rescales the mixture, which the pre-norm consumers discard) -- while
    # the zeroed gains still receive gradient, so the columns are learned from
    # silence instead of imposed at 1/n from step 0.
    block_column_gain_init: float = 1.0
    # Mass-conserving gain init (supersedes the constant form above when set).
    # The exact identity behind it: under private-only pools a block's merged
    # residual IS the sum of its K_b column streams, so the columns are a
    # DECOMPOSITION of the block residual, and K_b is an architectural choice
    # the read's initialization should be invariant to. A constant column gain
    # g is not: block b starts with total value mass 1 + g*K_b, so a k=8 block
    # outweighs a k=3 block 3.0-to-1.75 at g=0.25 -- depth skew from nothing
    # but granularity. This knob instead fixes every multi-column block's init
    # mass at exactly 1 and splits it: the aggregate gets ``1 - lambda`` and
    # each column ``lambda / K_b``. Every block then contributes one
    # block-direction of mass regardless of K_b, the column ensemble's
    # idiosyncratic deviation from the merged direction is averaged down by
    # ~1/sqrt(K_b) (measured 0.52 at K=3, 0.34 at K=8, tracking the
    # prediction), and lambda = 0 degenerates to block_column_gain_init = 0.0
    # (every read starts at exactly the kimi mixture). Requires
    # ``block_column_include_aggregate=True`` (the ``1 - lambda`` share needs
    # its slot) and is mutually exclusive with a non-default constant init.
    # None disables.
    block_column_mass_lambda: float | None = None
    # Head-only columns: every SUBLAYER read consumes the narrow kimi-style
    # sources (embedding + completed blocks' aggregates + the live block's
    # running aggregate) and only the FINAL head read sees the full
    # column-resolved list. Motivated by the measured g0 gain trajectory --
    # sublayer reads keep their column gains near zero (mean |gain| 0.068,
    # several exactly 0) while the head read uses them (0.146) -- and by the
    # speed decomposition: the wide read's ~1,150 per-slot streaming ops are
    # ~22-28% of a matched-shape step, and all but one of the 33 reads shrink
    # from up to 40 sources to at most num_blocks + 1. Requires
    # ``block_column_include_aggregate=True`` (the narrow list is MADE of the
    # aggregates).
    block_column_head_only: bool = False
    # One query vector PER SOURCE on the FINAL head read (the sublayer reads
    # keep the paper's shared pseudo-query). Both measured widenings of the
    # read lost to merged-only under the shared query until their extra
    # sources were silenced at init -- the shared query makes source
    # PROMOTION zero-sum (aligning it with one source's direction mis-scores
    # every other). Per-source queries remove that coupling at zero runtime
    # cost (one GEMV per source either way; ~25k extra parameters at d=640,
    # n=40) and zero-init keeps the step-0 mixture exactly uniform.
    block_column_head_query_per_source: bool = False
    # Deviation sources: export each multi-column block as its aggregate plus
    # the per-column DEVIATIONS ``s_j - a`` instead of the raw columns. The raw
    # set is linearly redundant (``sum_j s_j = k * a`` under private-only
    # pools), so after per-source RMS normalization every column key is highly
    # correlated with the aggregate's -- mass spent on columns re-buys the
    # block direction the read already has, which is the dilution that made
    # every non-silent column init lose. Deviations carry ONLY the
    # column-specific signal (``sum_j delta_j = 0``): the aggregate's share
    # directly controls the merged direction, a deviation's share buys nothing
    # but its column's difference, and gains at zero still recover kimi
    # exactly. Same source count, same cost, same slot layout (k deviation
    # slots then the aggregate), so every gain-init knob composes unchanged.
    block_column_deviation_sources: bool = False
    # Which blocks actually keep per-column streams. None (default) = all of
    # them, the historical behaviour. Any other value lists the block indices
    # that stay column-wise; every other block runs the ordinary merged
    # (kimi-style) recurrence with ONE stream and exports ONE source.
    #
    # Evidence-driven: the learned head-read gains show the model already
    # ignores some blocks' columns. On a k = 3..8 ramp the per-block deviation
    # gain relative to that block's aggregate came out 0.41 / 0.14 / **0.014** /
    # **0.008** / 0.08 / 0.15 -- blocks 3 and 4 pay the full k-slot stream cost
    # for deviations weighted at ~1% of their own aggregate. Dropping a block
    # from this list removes its per-layer [batch, seq, k, dim] state entirely,
    # which is both the residual speed gap (stream maintenance) and the +15 GiB
    # that makes column mode the first arm to OOM at 500m.
    block_column_blocks: list[int] | None = None
    # How many TRAILING layers of a column block carry per-column streams. None
    # (default) = all of them. The measured cost of column mode is
    # ``sum_b (layers_b * k_b)`` -- slot-layers -- so a wide block is affordable
    # if only its last layer or two are column-wise, which is otherwise
    # impossible: `block_column_blocks` can only take a block whole.
    #
    # Semantics stay coherent because the two exported quantities have
    # different scopes by construction. The block's AGGREGATE still accumulates
    # every layer (the merged prefix runs the ordinary recurrence and its
    # partial carries forward), so nothing is dropped from the residual path.
    # The DEVIATIONS are formed only over the trailing window, so they carry
    # "how the columns differed while they were tracked" -- and they still sum
    # to zero, because they are re-based on the mean of the same window.
    block_column_trailing_layers: int | None = None
    # Stochastic depth (DeiT drop-path). 0.0 disables. When >0, each routed
    # sublayer's update (attention and feed-forward) is dropped per-sample at
    # train time with a probability that increases linearly with depth within a
    # block (timm rule: dpr_i = drop_path_rate * i / (num_layers - 1)), surviving
    # updates rescaled by 1/(1-p) so the expectation is unchanged. Exact DeiT
    # schedule for single-block models (e.g. the dense ViT); per-block otherwise.
    drop_path_rate: float = 0.0
    # Global / register tokens: a small set of learned tokens prepended to the
    # sequence that participate in EVERY attention group (active in all groups,
    # uniform 1/G gate), so they attend to all patches and every patch attends
    # to them -- a cheap O(num_global * L) all-to-all channel bolted onto the
    # cheap local grouped attention (Longformer/BigBird global+local). They
    # bypass token pooling (carried through unchanged) and are stripped before
    # the classification head (pure registers). 0 disables. Forcing globals into
    # all groups makes membership multi-group, so the top-1 batched dispatch
    # plan is disabled when this is >0 (the per-group gathered loop is used).
    num_global_tokens: int = 0
    # Token-pool operator at each block exit. "mean" (default) RMS-pools each
    # stride-S window into one token (lossy averaging); "concat" stacks the S
    # tokens into one S*d vector and lets the block's output projection
    # (widened to S*d -> d') learn how to combine them -- no averaging loss.
    # LCLM (arXiv:2606.09659) finds concat beats mean at LOW compression ratios
    # (Nestor strides are 2-4x), which is exactly where pooling's detail loss
    # bites. Concat needs integer strides (no pooling_keep_ratios) and is
    # incompatible with num_global_tokens>0 (the d-wide globals can't share the
    # S*d->d' projection). Only blocks with stride>1 are affected; stride-1
    # blocks are an identity pool either way.
    pooling_mode: str = "mean"

    @property
    def output_dim(self) -> int:
        """Return the final block width.

        Returns:
            int: Final backbone feature dimension.
        """

        return self.feature_dims[-1]

    def to_dict(self) -> dict[str, Any]:
        """Expand the config into concrete per-block dictionaries.

        Returns:
            dict[str, Any]: Expanded config with one block dictionary per block.
        """

        if len(self.feature_dims) != self.num_blocks:
            raise ValueError("feature_dims must contain one entry per block.")

        pooling_strides = cast(
            list[int],
            _promote_to_blockwise(self.pooling_strides, self.num_blocks),
        )
        if self.pooling_keep_ratios is None:
            pooling_keep_ratios: list[float | None] = [None] * self.num_blocks
        else:
            pooling_keep_ratios = cast(
                list[float | None],
                _promote_to_blockwise(self.pooling_keep_ratios, self.num_blocks),
            )
        if self.block_residual_mode not in (
            "projection",
            "attn_res",
            "attn_res_kimi",
            "attn_res_column",
        ):
            raise ValueError(
                "block_residual_mode must be 'projection', 'attn_res', "
                "'attn_res_kimi', or 'attn_res_column'."
            )
        if self.block_residual_mode in (
            "attn_res",
            "attn_res_kimi",
            "attn_res_column",
        ):
            if self.residual_mode == "attn_res_xblock":
                raise ValueError(
                    f"block_residual_mode='{self.block_residual_mode}' cannot "
                    "be combined with residual_mode='attn_res_xblock'; both "
                    "define a cross-block residual, but attn_res_xblock uses "
                    "global summaries."
                )
            if any(stride != 1 for stride in pooling_strides) or any(
                ratio not in (None, 1.0) for ratio in pooling_keep_ratios
            ):
                raise ValueError(
                    f"block_residual_mode='{self.block_residual_mode}' "
                    "requires constant token "
                    "count: use pooling_strides=1 and pooling_keep_ratios=None "
                    "(or 1.0)."
                )
        if self.block_residual_mode in ("attn_res_kimi", "attn_res_column"):
            # The Kimi reads mix RAW block outputs (no per-source projections),
            # so every depth source must share one width; the partial-sum
            # recurrence (RoutedTransformerLayer.forward_kimi /
            # .forward_column) replaces the within-block residual stream, so
            # the layers must run the plain convex path.
            mode = self.block_residual_mode
            if any(dim != self.input_dim for dim in self.feature_dims):
                raise ValueError(
                    f"block_residual_mode='{mode}' requires constant "
                    "width: every feature_dims entry must equal input_dim "
                    "(depth reads mix raw block outputs without projections)."
                )
            if self.residual_mode != "convex":
                raise ValueError(
                    f"block_residual_mode='{mode}' requires "
                    "residual_mode='convex'; the per-sublayer depth reads "
                    "replace the within-block residual stream."
                )
            if not self.block_attn_res_include_embedding:
                raise ValueError(
                    f"block_residual_mode='{mode}' requires "
                    "block_attn_res_include_embedding=True (the embedding "
                    "b_0 is the first block's only depth source)."
                )
            if self.num_global_tokens > 0:
                raise ValueError(
                    f"block_residual_mode='{mode}' does not support "
                    "num_global_tokens > 0."
                )
        num_layers = cast(
            list[int],
            _promote_to_blockwise(self.block_num_layers, self.num_blocks),
        )
        num_groups = cast(
            list[int],
            _promote_to_blockwise(self.num_groups, self.num_blocks),
        )
        num_groups_per_token = cast(
            list[int],
            _promote_to_blockwise(self.num_groups_per_token, self.num_blocks),
        )
        if (
            self.block_column_include_aggregate
            and self.block_residual_mode != "attn_res_column"
        ):
            raise ValueError(
                "block_column_include_aggregate only applies to "
                "block_residual_mode='attn_res_column' (it adds the aggregated "
                "block-wise residual beside the column-wise ones); got "
                f"block_residual_mode='{self.block_residual_mode}'."
            )
        if (
            self.block_column_gain_init != 1.0
            and self.block_residual_mode != "attn_res_column"
        ):
            raise ValueError(
                "block_column_gain_init scales COLUMN sources, which only "
                "exist under block_residual_mode='attn_res_column'; got "
                f"block_residual_mode='{self.block_residual_mode}'."
            )
        if self.block_column_gain_init < 0.0:
            raise ValueError(
                "block_column_gain_init must be non-negative (0.0 silences "
                f"the columns at init); got {self.block_column_gain_init}."
            )
        if self.block_column_mass_lambda is not None:
            if not 0.0 <= self.block_column_mass_lambda <= 1.0:
                raise ValueError(
                    "block_column_mass_lambda splits one unit of per-block "
                    "init mass, so it must lie in [0, 1]; got "
                    f"{self.block_column_mass_lambda}."
                )
            if not self.block_column_include_aggregate:
                raise ValueError(
                    "block_column_mass_lambda gives the aggregate the "
                    "1 - lambda share, so it requires "
                    "block_column_include_aggregate=True."
                )
            if self.block_column_gain_init != 1.0:
                raise ValueError(
                    "block_column_mass_lambda and a non-default "
                    "block_column_gain_init both prescribe the column gains; "
                    "set one."
                )
        if self.block_column_head_only:
            if self.block_residual_mode != "attn_res_column":
                raise ValueError(
                    "block_column_head_only narrows the column-mode sublayer "
                    "reads, so it requires "
                    "block_residual_mode='attn_res_column'; got "
                    f"'{self.block_residual_mode}'."
                )
            if not self.block_column_include_aggregate:
                raise ValueError(
                    "block_column_head_only feeds sublayer reads the per-block "
                    "aggregates, so it requires "
                    "block_column_include_aggregate=True."
                )
        if (
            self.block_column_head_query_per_source
            and self.block_residual_mode != "attn_res_column"
        ):
            raise ValueError(
                "block_column_head_query_per_source configures the column-mode "
                "head read; got "
                f"block_residual_mode='{self.block_residual_mode}'."
            )
        if self.block_column_trailing_layers is not None:
            if self.block_residual_mode != "attn_res_column":
                raise ValueError(
                    "block_column_trailing_layers narrows column-mode blocks, "
                    "so it requires block_residual_mode='attn_res_column'; got "
                    f"'{self.block_residual_mode}'."
                )
            if self.block_column_trailing_layers < 1:
                raise ValueError(
                    "block_column_trailing_layers must be at least 1 (0 would "
                    "mean the block keeps no columns -- drop it from "
                    f"block_column_blocks instead); got "
                    f"{self.block_column_trailing_layers}."
                )
        if self.block_column_blocks is not None:
            if self.block_residual_mode != "attn_res_column":
                raise ValueError(
                    "block_column_blocks selects which blocks stay "
                    "column-wise, so it requires "
                    "block_residual_mode='attn_res_column'; got "
                    f"'{self.block_residual_mode}'."
                )
            if not self.block_column_blocks:
                raise ValueError(
                    "block_column_blocks is empty: no block would keep "
                    "columns, which is plain attn_res_kimi -- set that mode "
                    "instead."
                )
            if any(
                not 0 <= b < self.num_blocks
                for b in self.block_column_blocks
            ):
                raise ValueError(
                    f"block_column_blocks {self.block_column_blocks} has an "
                    f"index outside [0, {self.num_blocks})."
                )
        if self.block_column_deviation_sources:
            if self.block_residual_mode != "attn_res_column":
                raise ValueError(
                    "block_column_deviation_sources re-bases the column-mode "
                    "exports; got "
                    f"block_residual_mode='{self.block_residual_mode}'."
                )
            if not self.block_column_include_aggregate:
                raise ValueError(
                    "block_column_deviation_sources needs the aggregate "
                    "exported (deviations are relative to it), so it requires "
                    "block_column_include_aggregate=True."
                )
        if self.block_residual_mode == "attn_res_column" and all(
            top_k <= 1 for top_k in num_groups_per_token
        ):
            # Column-wise residual streams ARE the routed attention columns; a
            # model whose every block routes each token to a single column has
            # exactly one stream per block, i.e. it IS 'attn_res_kimi'.
            raise ValueError(
                "block_residual_mode='attn_res_column' is Nestor-specific: it "
                "needs at least one block with num_groups_per_token > 1 (the "
                "columns each token carries its own residual stream through). "
                "A dense / top-1-routed model should use "
                "block_residual_mode='attn_res_kimi', which this mode "
                "degenerates to when every block has k == 1."
            )
        num_heads = cast(
            list[int],
            _promote_to_blockwise(self.block_num_heads, self.num_blocks),
        )
        head_dims = cast(
            list[int],
            _promote_to_blockwise(self.block_head_dims, self.num_blocks),
        )
        if self.attention_auto_head_split:
            # Split a block's full head count across its active groups so active
            # attention compute matches a dense block at any top_k:
            # heads/group = round(full_heads / num_groups_per_token), with
            # full_heads = feature_dims[block] / head_dim.
            num_heads = [
                max(
                    1,
                    round(
                        (self.feature_dims[block_idx] // head_dims[block_idx])
                        / num_groups_per_token[block_idx]
                    ),
                )
                for block_idx in range(self.num_blocks)
            ]
        rope_bases = cast(
            list[float],
            _promote_to_blockwise(self.block_rope_bases, self.num_blocks),
        )
        if self.mlp_auto_dense_ratio is None:
            mlp_hidden_dims = cast(
                list[list[int]],
                _promote_to_blockwise(
                    self.block_mlp_hidden_dims,
                    self.num_blocks,
                    nested_list=True,
                ),
            )
        else:
            # Auto-derived below, once per-block top_k is known.
            mlp_hidden_dims = []
        mlp_activations = cast(
            list[str],
            _promote_to_blockwise(self.block_mlp_activations, self.num_blocks),
        )
        router_num_experts_per_group = cast(
            list[int],
            _promote_to_blockwise(
                self.router_num_experts_per_group,
                self.num_blocks,
            ),
        )
        router_num_shared_experts = cast(
            list[int],
            _promote_to_blockwise(
                self.router_num_shared_experts,
                self.num_blocks,
            ),
        )
        router_num_dense_shared_experts = cast(
            list[int],
            _promote_to_blockwise(
                self.router_num_dense_shared_experts,
                self.num_blocks,
            ),
        )
        router_num_experts_per_group_per_token = cast(
            list[int],
            _promote_to_blockwise(
                self.router_num_experts_per_group_per_token,
                self.num_blocks,
            ),
        )
        if self.mlp_auto_dense_ratio is not None:
            if self.mlp_auto_dense_ratio <= 0:
                raise ValueError("mlp_auto_dense_ratio must be positive.")
            # Fine-grained experts: hidden = ratio * d / top_k per block, so the
            # active expert FLOPs equal a dense (ratio x) FFN at any top_k. ``d``
            # is the block's feature width (feature_dims[block]).
            mlp_hidden_dims = [
                [
                    max(
                        1,
                        round(
                            self.mlp_auto_dense_ratio
                            * self.feature_dims[block_idx]
                            / router_num_experts_per_group_per_token[block_idx]
                        ),
                    )
                ]
                for block_idx in range(self.num_blocks)
            ]
        score_hidden_dims = cast(
            list[list[int]],
            _promote_to_blockwise(
                self.router_score_mlp_hidden_dims,
                self.num_blocks,
                nested_list=True,
            ),
        )
        score_activations = cast(
            list[str],
            _promote_to_blockwise(
                self.router_score_mlp_activations,
                self.num_blocks,
            ),
        )
        noise_hidden_dims = cast(
            list[list[int]],
            _promote_to_blockwise(
                self.router_noise_mlp_hidden_dims,
                self.num_blocks,
                nested_list=True,
            ),
        )
        noise_activations = cast(
            list[str],
            _promote_to_blockwise(
                self.router_noise_mlp_activations,
                self.num_blocks,
            ),
        )
        loss_free_balancing = cast(
            list[bool],
            _promote_to_blockwise(
                self.router_loss_free_balancing,
                self.num_blocks,
            ),
        )
        bias_update_rates = cast(
            list[float],
            _promote_to_blockwise(
                self.router_bias_update_rate,
                self.num_blocks,
            ),
        )
        attention_token_dispatch = cast(
            list[bool],
            _promote_to_blockwise(
                self.attention_token_dispatch,
                self.num_blocks,
            ),
        )
        expert_pool_public_experts = cast(
            list[int],
            _promote_to_blockwise(
                self.block_expert_pool_public_experts,
                self.num_blocks,
            ),
        )
        expert_pool_public_top_k = cast(
            list[int],
            _promote_to_blockwise(
                self.block_expert_pool_public_top_k,
                self.num_blocks,
            ),
        )
        expert_pool_private_experts = cast(
            list[int],
            _promote_to_blockwise(
                self.block_expert_pool_private_experts,
                self.num_blocks,
            ),
        )
        expert_pool_private_top_k = cast(
            list[int],
            _promote_to_blockwise(
                self.block_expert_pool_private_top_k,
                self.num_blocks,
            ),
        )
        # Speed A/B knob (numerically identical): force the masked attention path
        # instead of capacity-padded token dispatch. Token dispatch shrinks the
        # attention seq to per-image members but pays a cap=int(max()) device
        # sync + gather/scatter (graph breaks under compile); the masked path
        # runs the full seq per group (more attention FLOPs) but no sync/gather.
        import os as _os

        if _os.environ.get("NESTOR_FORCE_NO_TOKEN_DISPATCH") == "1":
            attention_token_dispatch = [False] * self.num_blocks

        # Per-layer expert schedule. Uniform (one value repeated) unless
        # block_layer_num_experts overrides it for depth-varying sparsity. When
        # overridden, the per-block scalar is set to the schedule's max so any
        # block-level use sees a value consistent with the layers it builds; the
        # top-k / shared-expert bounds below are checked against the schedule MIN.
        if self.block_layer_num_experts is None:
            layer_num_experts_per_block = [
                [router_num_experts_per_group[block_idx]] * num_layers[block_idx]
                for block_idx in range(self.num_blocks)
            ]
        else:
            if len(self.block_layer_num_experts) != self.num_blocks:
                raise ValueError(
                    "block_layer_num_experts must have one entry per block."
                )
            layer_num_experts_per_block = []
            for block_idx in range(self.num_blocks):
                schedule = [int(e) for e in self.block_layer_num_experts[block_idx]]
                if len(schedule) != num_layers[block_idx]:
                    raise ValueError(
                        f"block_layer_num_experts[{block_idx}] must have "
                        f"block_num_layers ({num_layers[block_idx]}) entries, "
                        f"got {len(schedule)}."
                    )
                if any(e < 1 for e in schedule):
                    raise ValueError(
                        "block_layer_num_experts entries must be >= 1."
                    )
                layer_num_experts_per_block.append(schedule)
            router_num_experts_per_group = [
                max(layer_num_experts_per_block[block_idx])
                for block_idx in range(self.num_blocks)
            ]

        block_configs: list[dict[str, Any]] = []
        input_dim = self.input_dim
        for block_idx in range(self.num_blocks):
            if pooling_strides[block_idx] < 1:
                raise ValueError(
                    "pooling_strides entries must be positive integers."
                )
            if num_groups[block_idx] < 1:
                raise ValueError("num_groups must be positive.")
            if (
                not 1
                <= num_groups_per_token[block_idx]
                <= num_groups[block_idx]
            ):
                raise ValueError(
                    "num_groups_per_token must be in [1, num_groups]."
                )
            if router_num_experts_per_group[block_idx] < 1:
                raise ValueError(
                    "router_num_experts_per_group must be positive."
                )
            # Bound the routed top-k / shared count by the SMALLEST per-layer
            # expert count in this block (a layer cannot route to more experts
            # than it owns); with a uniform schedule this is the scalar.
            min_experts = min(layer_num_experts_per_block[block_idx])
            if not (0 <= router_num_shared_experts[block_idx] <= min_experts):
                raise ValueError(
                    "router_num_shared_experts must be in "
                    "[0, min(block_layer_num_experts)]."
                )
            if not (
                1
                <= router_num_experts_per_group_per_token[block_idx]
                <= min_experts
            ):
                raise ValueError(
                    "router_num_experts_per_group_per_token must be in "
                    "[1, min(block_layer_num_experts)]."
                )
            # Expert pools: the PRIVATE pool carries the layer, so it is
            # required whenever pooling is on (public-only is not
            # implemented); a 0 public count is the legal private-only mode
            # (no public path, no path gate). Pool top-k stays within its
            # pool, and pools only sit on "nomoe" blocks so no configured
            # routed/dense-shared expert is silently replaced.
            pool_public = expert_pool_public_experts[block_idx]
            pool_private = expert_pool_private_experts[block_idx]
            if pool_public > 0 and pool_private == 0:
                raise ValueError(
                    "block_expert_pool_public_experts requires "
                    "block_expert_pool_private_experts > 0 (public-only pools "
                    "are not implemented); set both to 0 to disable pooling."
                )
            if pool_private > 0:
                public_top_k = expert_pool_public_top_k[block_idx]
                private_top_k = expert_pool_private_top_k[block_idx]
                if pool_public > 0 and not 1 <= public_top_k <= pool_public:
                    raise ValueError(
                        "block_expert_pool_public_top_k must be in "
                        "[1, block_expert_pool_public_experts]."
                    )
                if not 1 <= private_top_k <= pool_private:
                    raise ValueError(
                        "block_expert_pool_private_top_k must be in "
                        "[1, block_expert_pool_private_experts]."
                    )
                if self.block_expert_pool_private_gate not in (
                    "renorm",
                    "switch",
                ):
                    raise ValueError(
                        "block_expert_pool_private_gate must be 'renorm' or "
                        "'switch'."
                    )
                if (
                    max(layer_num_experts_per_block[block_idx]) != 1
                    or router_num_dense_shared_experts[block_idx] != 0
                ):
                    raise ValueError(
                        "Expert pools replace the layer's expert banks: a "
                        "pool-enabled block requires 1 routed expert per "
                        "layer and no dense shared experts."
                    )

            block_configs.append(
                NestorBlockConfig(
                    input_dim=input_dim,
                    output_dim=self.feature_dims[block_idx],
                    pooling_stride=pooling_strides[block_idx],
                    pooling_keep_ratio=pooling_keep_ratios[block_idx],
                    block_num_layers=num_layers[block_idx],
                    num_groups=num_groups[block_idx],
                    num_groups_per_token=num_groups_per_token[block_idx],
                    block_num_heads=num_heads[block_idx],
                    block_head_dim=head_dims[block_idx],
                    block_rope_base=rope_bases[block_idx],
                    block_mlp_hidden_dims=mlp_hidden_dims[block_idx],
                    block_mlp_activation=mlp_activations[block_idx],
                    router_num_experts_per_group=(
                        router_num_experts_per_group[block_idx]
                    ),
                    layer_num_experts=layer_num_experts_per_block[block_idx],
                    router_num_shared_experts=router_num_shared_experts[
                        block_idx
                    ],
                    router_num_dense_shared_experts=(
                        router_num_dense_shared_experts[block_idx]
                    ),
                    router_num_experts_per_group_per_token=(
                        router_num_experts_per_group_per_token[block_idx]
                    ),
                    router_score_mlp_hidden_dims=score_hidden_dims[block_idx],
                    router_score_mlp_activation=score_activations[block_idx],
                    router_noise_mlp_hidden_dims=noise_hidden_dims[block_idx],
                    router_noise_mlp_activation=noise_activations[block_idx],
                    loss_free_balancing=loss_free_balancing[block_idx],
                    bias_update_rate=bias_update_rates[block_idx],
                    attention_token_dispatch=attention_token_dispatch[block_idx],
                    expert_pool_public_experts=(
                        expert_pool_public_experts[block_idx]
                    ),
                    expert_pool_public_top_k=(
                        expert_pool_public_top_k[block_idx]
                    ),
                    expert_pool_private_experts=(
                        expert_pool_private_experts[block_idx]
                    ),
                    expert_pool_active_budget=self.block_expert_pool_active_budget,
                    expert_pool_public_budget=self.block_expert_pool_public_budget,
                    expert_pool_private_top_k=(
                        expert_pool_private_top_k[block_idx]
                    ),
                    expert_pool_private_gate=self.block_expert_pool_private_gate,
                ).to_dict()
            )
            input_dim = self.feature_dims[block_idx]

        return {
            "causal": self.causal,
            "residual_mode": self.residual_mode,
            "block_residual_mode": self.block_residual_mode,
            "block_attn_res_include_embedding": (
                self.block_attn_res_include_embedding
            ),
            "block_column_head_only": self.block_column_head_only,
            "block_column_deviation_sources": (
                self.block_column_deviation_sources
            ),
            "block_column_blocks": self.block_column_blocks,
            "block_column_trailing_layers": self.block_column_trailing_layers,
            "block_column_head_query_per_source": (
                self.block_column_head_query_per_source
            ),
            "block_column_include_aggregate": (
                self.block_column_include_aggregate
            ),
            "block_configs": block_configs,
        }


# ---------------------------------------------------------------------------
# Routed attention, routed transformer layer, and Nestor block.
# ---------------------------------------------------------------------------


class GroupRoutedAttention(nn.Module):
    """Multi-head self-attention owned by one attention group.

    Each attention group has its own parameters. The block-level group router
    decides which groups are active for each token, and the token mixes the
    resulting group outputs with the router gates.
    """

    dim: int
    num_heads: int
    head_dim: int
    causal: bool

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        *,
        rope_base: float = 10000.0,
        causal: bool = False,
    ) -> None:
        """Initialize one routed attention group.

        Args:
            dim (int): Input and output feature width.
            num_heads (int): Number of attention heads.
            head_dim (int): Per-head width.
            rope_base (float): RoPE frequency base ``theta_l^rope``.
            causal (bool): Whether the owning model is causal. Consulted ONLY
                by the trivial-mask (``attn_mask=None``) paths, where the mask
                the caller would otherwise have built is analytically known to
                be exactly ``tril`` (causal) or all-true (non-causal); the
                dense-mask paths keep deriving structure from the mask itself.
        """

        super().__init__()
        self.causal = causal
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = ScaledLinear(dim, inner_dim, role="in_stream")
        self.k_proj = ScaledLinear(dim, inner_dim, role="in_stream")
        self.v_proj = ScaledLinear(dim, inner_dim, role="in_stream")
        self.out_proj = ScaledLinear(inner_dim, dim, role="in_stream")
        self.rope = RoPE(head_dim, base=rope_base)

        # NESTOR_GATHERED_CAUSAL_FLASH=1: compact-causal fast path for the
        # token-dispatch (gathered) attention core -- see
        # ``forward_gathered``. Static (env) flag read once so the branch stays
        # a compile-time constant; the per-module verdict below is the only
        # mutable part and flips at most once (first call).
        import os as _os

        self._gathered_causal_flash: bool = (
            _os.environ.get("NESTOR_GATHERED_CAUSAL_FLASH") == "1"
        )
        # None = not yet verified, True = verified compact-causal (fast path
        # active), False = mismatch on the first call (permanent fallback).
        self._gathered_causal_flash_ok: bool | None = None
        # NESTOR_GATHERED_VARLEN=1: packed variable-length attention instead of
        # capacity padding (see forward_gathered_varlen). Gated on the SAME
        # verified compact-causal contract as the maskless flash path, because
        # the packed path hardcodes within-segment causality; until the one-time
        # verification passes, calls take the padded path unchanged.
        self._gathered_varlen = (
            _os.environ.get("NESTOR_GATHERED_VARLEN") == "1"
        )

    def _build_block_mask(
        self,
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"],
    ) -> Any:
        """Convert a dense boolean mask into a flex-attention block mask."""

        if create_block_mask is None:
            raise RuntimeError(
                "Flex attention is not available in this PyTorch build."
            )

        batch_size, seq_len, _ = attn_mask.shape
        if seq_len >= 49152:
            # Fail loud instead of the two silent long-context failure modes:
            # eager create_block_mask evaluates mask_mod over the FULL S x S
            # grid, materializing int64 index tensors (exactly 128 GiB at
            # S=131072), and the fused flex kernel hits an illegal memory
            # access at S=65536 (S^2 = 2^32).
            raise RuntimeError(
                f"_build_block_mask at seq_len={seq_len}: create_block_mask "
                f"materializes O(S^2) index tensors "
                f"({seq_len ** 2 * 8 / 2 ** 30:.0f} GiB) and the flex kernel "
                "faults at S>=64k. Long-sequence causal runs must use the "
                "trivial-mask contract (attn_mask=None from the adapter), "
                "which takes the maskless is_causal flash paths instead."
            )

        def mask_mod(
            batch: Tensor,
            head: Tensor,
            q_idx: Tensor,
            kv_idx: Tensor,
        ) -> Tensor:
            del head
            return attn_mask[batch, q_idx, kv_idx]

        return create_block_mask(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=attn_mask.device,
        )

    @property
    def gathered_causal_flash_ok(self) -> bool:
        """Whether the compact-causal gathered fast path is active and verified.

        ``True`` only once ``forward_gathered`` has checked the real compact
        pair mask against pure compact causality (see there). Callers use this
        to skip building the dense ``[B, S, S]`` group mask altogether and pass
        ``attn_mask=None``.

        Returns:
            bool: ``True`` if the fast path is enabled and verified.
        """

        return self._gathered_causal_flash and self._gathered_causal_flash_ok is True

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        pos: Int[Tensor, "batch_size seq_len"],
        members: Bool[Tensor, "batch_size seq_len"] | None = None,
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """Compute masked self-attention for one routed group.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Group input sequence.
            attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None): Group
                mask. ``None`` means the trivial-mask contract: every token is
                valid and the mask would be exactly ``tril`` (causal) or
                all-true (non-causal), so the maskless paths run instead. With
                ``members`` it is additionally allowed once
                :attr:`gathered_causal_flash_ok` is ``True``.
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions for
                rotary embeddings.
            members (Bool[Tensor, "batch_size seq_len"] | None): If given, project
                only these member tokens via ``forward_gathered`` (token-dispatch
                path); otherwise project the full sequence and rely on the mask.
                Routing through ``forward`` keeps the call under ``__call__`` so
                profilers attribute the gathered ops to this module.

        Returns:
            Float[Tensor, "batch_size seq_len dim"]: Group output sequence.
        """

        if members is not None:
            return self.forward_gathered(x, attn_mask, pos, members)
        if attn_mask is None:
            # Trivial-mask contract: no padding, no structure beyond
            # causality -- run the maskless flash paths (never builds an
            # [S, S] tensor of any kind).
            if self.causal:
                return self.forward_causal_full(x, pos)
            return self.forward_full(x, pos)

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        q = self.rope(q.transpose(1, 2), pos)
        k = self.rope(k.transpose(1, 2), pos)
        v = v.transpose(1, 2)

        valid_queries = attn_mask.any(dim=-1)
        out = self._attend(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        return out * valid_queries.unsqueeze(-1).to(dtype=out.dtype)

    def forward_full(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        pos: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """Unmasked self-attention (every token attends to every token).

        For the dense single-group, no-padding case (e.g. image classification)
        the group mask is all-ones, so we skip the per-layer ``[B,S,S]`` mask
        construction and ``create_block_mask`` entirely and take the fastest
        flash / full flex-attention path. Numerically identical to ``forward``
        with an all-true mask.
        """

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        q = self.rope(q.transpose(1, 2), pos)
        k = self.rope(k.transpose(1, 2), pos)
        v = v.transpose(1, 2)

        # Unmasked full attention -> plain SDPA, which dispatches to the FLASH
        # backend (fastest; needs fp16/bf16 q/k/v -- so this only truly flashes
        # under bf16 params / autocast). Faster than flex_attention for the
        # simple no-mask case and avoids flex's block-mask machinery.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=1.0 / self.head_dim
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)

    def forward_causal_full(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        pos: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """Maskless CAUSAL self-attention (flash ``is_causal`` kernel).

        The causal twin of :meth:`forward_full`, for the trivial-mask contract
        (``attn_mask=None``): every token is valid and the dense mask would be
        exactly ``tril``, so SDPA's ``is_causal`` flash path replaces the
        entire ``[B, S, S]`` mask build + flex block-mask machinery. Those are
        O(S^2) memory (128 GiB of ``create_block_mask`` index tensors at
        S=131072) and the fused flex kernel faults at S>=64k, so this path is
        what makes long-context causal runs feasible at all. Numerically
        identical to ``forward`` with a pure causal mask.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Group input sequence.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions for RoPE.

        Returns:
            Float[Tensor, "batch_size seq_len dim"]: Group output sequence.
        """

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        q = self.rope(q.transpose(1, 2), pos)
        k = self.rope(k.transpose(1, 2), pos)
        v = v.transpose(1, 2)

        # A head_dim that is not a multiple of 8 has no fused fp32 kernel
        # (flash / mem-efficient both require it), so SDPA falls back to the
        # math kernel and materialises [B, H, S, S] -- 16 GiB per allocation
        # at S=32k for the 100m arms (head_dim 30; their fp32 held-out evals
        # OOMed at >= 32k). The query-chunked form below is the same math in
        # bounded memory; bf16 training never takes it (cuDNN handles it).
        chunk = causal_full_query_chunk()
        if chunk > 0 and seq_len > chunk and self.head_dim % 8 != 0 and q.dtype == torch.float32:
            out = causal_attention_query_chunked(q, k, v, scale=1.0 / self.head_dim, chunk=chunk)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0, scale=1.0 / self.head_dim
            )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)

    def _attend(
        self,
        q: Float[Tensor, "batch_size num_heads seq_len head_dim"],
        k: Float[Tensor, "batch_size num_heads seq_len head_dim"],
        v: Float[Tensor, "batch_size num_heads seq_len head_dim"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"],
    ) -> Float[Tensor, "batch_size num_heads seq_len head_dim"]:
        """Run the masked attention core, preferring the flex kernel on CUDA.

        Shared by ``forward`` (dense projections) and ``forward_gathered``
        (compact projections); both feed the same ``[B, H, L, head_dim]`` tensors
        and the same dense boolean ``attn_mask`` here.
        """

        import os as _os
        if _os.environ.get("NESTOR_ATTN_CAUSAL_SDPA") == "1":
            # Pure-causal fast path: flash backend via is_causal (works for any
            # head_dim, unlike the fused flex kernel, which e.g. rejects
            # head_dim 60). Verified against the real mask once per module;
            # modules with non-causal masks (e.g. token-dispatch groups with
            # gathered membership masks) fall through to the normal path.
            _verdict = getattr(self, "_causal_sdpa_verdict", None)
            if _verdict is None:
                _S = attn_mask.shape[-1]
                _tril = torch.tril(
                    torch.ones(_S, _S, dtype=torch.bool, device=attn_mask.device)
                )
                _verdict = bool((attn_mask == _tril.unsqueeze(0)).all())
                self._causal_sdpa_verdict = _verdict
            if _verdict:
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=0.0,
                    scale=1.0 / self.head_dim,
                )
        if q.is_cuda and _FLEX_ATTENTION_AVAILABLE:
            global _COMPILED_FLEX_ATTENTION, _FLEX_ATTENTION_COMPILE_FAILED
            flex_impl = _get_flex_attention_impl()
            if flex_impl is not None:
                block_mask = self._build_block_mask(attn_mask)
                try:
                    return cast(
                        Tensor,
                        flex_impl(
                            q, k, v, block_mask=block_mask, scale=1.0 / self.head_dim
                        ),
                    )
                except Exception:
                    # The fused flex kernel rejects some shapes (e.g. head_dim
                    # not a multiple of 8). Eager flex materializes the full
                    # [B, H, S, S] score matrix (~10x slower), so fall through
                    # to masked SDPA instead.
                    _COMPILED_FLEX_ATTENTION = None
                    _FLEX_ATTENTION_COMPILE_FAILED = True
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask.unsqueeze(1),
            dropout_p=0.0,
            scale=1.0 / self.head_dim,
        )

    def forward_gathered_varlen(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        members: Bool[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """``forward_gathered`` with packed variable-length attention.

        The padded path attends over ``[B, cap, cap]`` where ``cap`` is the max
        members over the batch, so ONE saturated image forces every image to the
        full sequence. Measured on the trained 500m router (median 544 members,
        max 4096) that is **4.67x the attention the routed tokens require**.

        Here every image's members are concatenated into one packed sequence and
        attended with ``cu_seqlens`` boundaries, so the attention core costs the
        REAL member count -- no padding, no dropped tokens, identical math.

        Static shapes, so nothing here graph-breaks: the packed buffer is a fixed
        ``[B*seq_len, ...]`` (the worst case, which the padded path already
        reaches whenever one image saturates), the segment boundaries live in a
        tensor rather than a Python int, and ``max_q``/``max_k`` are only kernel
        launch hints so ``seq_len`` is a valid static upper bound. Rows past
        ``cu_seq[-1]`` are ignored by the kernel (verified: exact agreement with
        an exactly-sized buffer).

        NOTE the projections still run over the full ``[B*seq_len, dim]`` buffer,
        so this recovers the ATTENTION saving, not the projection saving. Since
        the measured ``cap`` is frequently already ``seq_len``, that is not a
        regression in practice, but a fully packed path (dynamic shapes) would
        save more.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Input sequence.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions for RoPE.
            members (Bool[Tensor, "batch_size seq_len"]): Per-token membership in
                this group.

        Returns:
            Float[Tensor, "batch_size seq_len dim"]: Group output, zero on
            non-member positions, same dtype as ``forward_gathered``.
        """

        from torch.nn.attention.varlen import varlen_attn

        batch_size, seq_len, _ = x.shape
        device = x.device
        counts = members.sum(dim=1)  # [B] members per image
        rows = batch_size * seq_len  # static packed capacity

        # Segment boundaries. cumsum on-device: no host sync, no graph break.
        cu = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        torch.cumsum(counts.to(torch.int32), dim=0, out=cu[1:])

        # Rows past ``cu[-1]`` belong to NO segment: the kernel never writes
        # their forward output NOR their backward dq/dk/dv, which stay as
        # reused-allocator garbage -- and both weight-gradient GEMMs turn that
        # into ``0 * inf = NaN`` (``x_packed^T @ dq_garbage`` on the input
        # side, ``d_out_c^T @ out_garbage`` on the output side). Unit tests
        # missed it because fresh CUDA allocations are zero pages; a production
        # run trained NaN from step 20. Fixed below by ``torch.where``
        # masks around the kernel: where's backward routes the unselected
        # branch to zero, so the garbage never reaches a weight gradient.
        # (Extending ``cu`` with tail segments also fixes it but inflates the
        # kernel grid -- segments x max_q blocks -- and measured 12-15% slower
        # in production at any tail chunk size.)
        row_real = (
            torch.arange(rows, device=device)[:, None, None] < cu[-1]
        )  # [rows, 1, 1] rows the kernel actually owns

        # Stable members-first order preserves original token order, so causal
        # order survives compaction (same contract as forward_gathered).
        order = torch.argsort(
            members.to(torch.int8), dim=1, descending=True, stable=True
        )  # [B, seq_len]
        slot = torch.arange(seq_len, device=device)[None, :]  # [1, seq_len]
        real = slot < counts[:, None]  # [B, seq_len] members-first slots

        # Destination row for every slot; non-members go to a scratch row that is
        # sliced off, which keeps the scatter static instead of boolean-indexed.
        dest = cu[:-1, None].long() + slot  # [B, seq_len]
        dest = torch.where(real, dest, torch.full_like(dest, rows))

        x_ord = torch.gather(
            x, 1, order.unsqueeze(-1).expand(-1, -1, self.dim)
        )  # [B, seq_len, dim] members first
        pos_ord = torch.gather(pos, 1, order)

        x_packed = x_ord.new_zeros(rows + 1, self.dim)
        x_packed.scatter_(
            0, dest.reshape(-1, 1).expand(-1, self.dim), x_ord.reshape(-1, self.dim)
        )
        pos_packed = pos_ord.new_zeros(rows + 1)
        pos_packed.scatter_(0, dest.reshape(-1), pos_ord.reshape(-1))
        x_packed, pos_packed = x_packed[:rows], pos_packed[:rows]

        # RoPE wants [B, H, T, hd]; the packed sequence is one "batch" of length
        # rows, and positions are the ORIGINAL token positions, so rotation is
        # unchanged by compaction.
        shape = (1, rows, self.num_heads, self.head_dim)
        q = self.rope(
            self.q_proj(x_packed).view(shape).transpose(1, 2), pos_packed[None, :]
        )
        k = self.rope(
            self.k_proj(x_packed).view(shape).transpose(1, 2), pos_packed[None, :]
        )
        v = self.v_proj(x_packed).view(shape).transpose(1, 2)
        # varlen wants [T, H, hd].
        q = q.transpose(1, 2).reshape(rows, self.num_heads, self.head_dim)
        k = k.transpose(1, 2).reshape(rows, self.num_heads, self.head_dim)
        v = v.transpose(1, 2).reshape(rows, self.num_heads, self.head_dim)

        # The flash varlen kernel requires head_size % 8 == 0 (the same
        # constraint that made flex-attention reject these arms; the 500m
        # head_dim is 60). Zero-pad the head dim: padded q/k dims contribute 0
        # to every logit and padded v dims produce output columns sliced away
        # below, so this is exact, not approximate.
        pad = (-self.head_dim) % 8
        if pad:
            q = F.pad(q, (0, pad))
            k = F.pad(k, (0, pad))
            v = F.pad(v, (0, pad))

        # forward_gathered attends with scale = 1/head_dim (the muP attention
        # scale), but varlen_attn exposes no scale argument and applies the
        # standard 1/sqrt(head_size) -- where head_size is the PADDED dim it
        # actually sees. Fold the ratio into q so the kernel's 1/sqrt(d_pad)
        # becomes the required 1/head_dim. (Getting either the fold or the
        # padded-vs-true dim wrong shifts the softmax temperature and fails the
        # equivalence tests -- both variants were caught there.)
        q = q * (float(self.head_dim + pad) ** 0.5 / self.head_dim)

        # Gradient firewall (see the NaN note above): values on rows past
        # ``cu[-1]`` are unchanged in effect (the kernel ignores them), but
        # where's backward replaces the kernel's uninitialized dq/dk/dv there
        # with exact zeros before they can reach the projection weight grads.
        zero = q.new_zeros(())
        q = torch.where(row_real, q, zero)
        k = torch.where(row_real, k, zero)
        v = torch.where(row_real, v, zero)

        out = varlen_attn(q, k, v, cu, cu, seq_len, seq_len, is_causal=True)
        # Same firewall on the output: rows past ``cu[-1]`` hold garbage the
        # kernel never wrote; zero them so ``out_proj``'s weight gradient
        # (``d^T @ out``) cannot see inf/NaN even though ``d`` is zero there.
        out = torch.where(row_real, out, zero)
        if pad:
            out = out[..., : self.head_dim]
        out_c = self.out_proj(out.reshape(rows, self.num_heads * self.head_dim))

        # Unpack: read each slot's packed row back, zero the non-member slots,
        # then scatter to the slots' ORIGINAL token positions.
        out_padded = torch.cat([out_c, out_c.new_zeros(1, self.dim)], dim=0)
        out_slots = out_padded[dest.reshape(-1)].view(batch_size, seq_len, self.dim)
        out_slots = torch.where(real.unsqueeze(-1), out_slots, out_slots.new_zeros(()))

        y = out_slots.new_zeros(batch_size, seq_len, self.dim)
        y.scatter_(1, order.unsqueeze(-1).expand(-1, -1, self.dim), out_slots)
        return y

    def forward_gathered(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        pos: Int[Tensor, "batch_size seq_len"],
        members: Bool[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """Numerically equivalent to ``forward`` but project and attend only over
        each image's member tokens (capacity-padded token dispatch).

        Per image we pack this group's member tokens to the front of a compact
        ``[B, cap, dim]`` tensor (``cap`` = max members per image, **no token
        drop**) and run the projections, attention core, and output projection on
        that compact sequence. Both costs then scale with groups-*per-token*
        rather than total groups ``G``: with a balanced partition
        (``num_groups_per_token == 1``) the projections sum to ``~8 L d^2``
        (``G``-independent) and the core to ``~4 L^2 d / G`` (decreasing in ``G``),
        so adding groups adds parameters without adding attention compute -- the
        MoE property, applied to attention. (Caveat: more groups at ``k=1`` means
        smaller per-token attention neighborhoods, since attention is pairwise and
        only same-group tokens share a Q/K projection space.)

        Same-image confinement is automatic (the batch dim); the per-pair mask is
        gathered from ``attn_mask`` so causal / padding / membership masking is
        preserved. Member outputs match ``forward``; non-members are zero.

        **Compact-causal fast path** (opt-in, ``NESTOR_GATHERED_CAUSAL_FLASH=1``):
        for a causal LM the incoming mask is exactly ``causal & valid (x) valid``
        and the router only ever activates valid tokens, so the gathered
        ``[B, cap, cap]`` mask is *provably* plain causality restricted to the
        real compact slots. The stable argsort keeps members in increasing
        original position, so compact index order equals original order and
        ``causal(orig_i, orig_j) == (compact_j <= compact_i)``; padding sits at
        the END of every compact row, so a real query (``i < counts``) never
        reaches a padded key (``j >= counts > i``), and padded *query* rows are
        discarded by the ``valid`` zeroing below either way (they produce NaN on
        the masked path and finite junk here). Both ``O(B*cap*L)`` mask gathers
        therefore disappear and the core takes the flash ``is_causal`` kernel
        instead of masked SDPA. The identity is verified ONCE per module against
        the materialized mask; a mismatch (non-causal model, document masks,
        ...) permanently reverts this module to the general path.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Group input sequence.
            attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None): Group
                mask (already ``base & members & members``); gathered to the
                compact layout, so non-member keys and disallowed pairs stay
                masked. ``None`` is allowed only after the compact-causal fast
                path has been verified (:attr:`gathered_causal_flash_ok`), where
                the mask carries no information.
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions.
            members (Bool[Tensor, "batch_size seq_len"]): Per-token membership in
                this group (``group_router_output.active_mask[..., group_idx]``).

        Returns:
            Float[Tensor, "batch_size seq_len dim"]: Group output sequence.

        Raises:
            ValueError: If ``attn_mask`` is ``None`` while the compact-causal
                fast path is not (yet) verified for this module.
        """

        # Trivial-mask contract: ``None`` from a causal caller means the base
        # mask is exactly ``causal & valid (x) valid`` with every token valid,
        # so the compact-causal identity in the docstring above holds BY
        # CONSTRUCTION -- no one-time mask materialization/verification needed
        # (at 128k the verification itself is O(S^2) and cannot run). Kept as
        # a local so the branch stays argument-pure for checkpoint recompute.
        causal_ok = bool(self.gathered_causal_flash_ok) or (
            attn_mask is None and self.causal
        )
        if attn_mask is None and not causal_ok:
            raise ValueError(
                "forward_gathered requires attn_mask on non-causal modules "
                "unless the compact-causal fast path has been verified."
            )

        # Packed varlen path (opt-in). Dispatch on the ARGUMENTS ONLY:
        # ``attn_mask is None`` already implies the compact-causal contract is
        # verified (the guard above raises otherwise). Keying this branch on
        # module state (``gathered_causal_flash_ok``) broke activation
        # checkpointing: the flag flips inside the very forward that verifies
        # it, so the checkpoint RECOMPUTE took the packed branch where the
        # forward had taken the padded one -- "recomputed values have different
        # metadata". Arguments are replayed identically by use_reentrant=False
        # checkpointing, so an argument-pure branch cannot diverge between a
        # forward and its recompute.
        if self._gathered_varlen and attn_mask is None:
            return self.forward_gathered_varlen(x, pos, members)

        batch_size, seq_len, _ = x.shape
        inner_dim = self.num_heads * self.head_dim
        counts = members.sum(dim=1)  # members per image
        # Shapes below depend on cap, so this host sync cannot be removed
        # without padding to seq_len (which would undo the whole point).
        # Bucketing keeps the sync but bounds how many distinct shapes -- and
        # therefore how many torch.compile specializations -- it can produce.
        # A static capacity skips the .item() entirely -- that call is a graph
        # break, so removing it is what lets Inductor fuse across this region.
        # It can drop a group's overflow; see _static_capacity.
        cap = _static_capacity(seq_len)
        if cap is None:
            cap = int(counts.max().item())  # capacity; no token is dropped
            cap = _bucket_capacity(cap, seq_len)

        if cap == 0:
            # Keep every projection submodule in the autograd graph even when a
            # group receives no tokens, so FSDP2's per-module reduce-scatter fires
            # in the same order on every rank (mirrors the expert-bank guard).
            probe = self.out_proj(
                self.q_proj(x[:, :1])
                + self.k_proj(x[:, :1])
                + self.v_proj(x[:, :1])
            )
            # Return the out_proj (probe) dtype, matching the main path below and
            # the non-gathered forward() -- not x's (fp32-upcast) dtype.
            return probe.new_zeros(batch_size, seq_len, self.dim) + 0.0 * probe.sum()

        # Per-image compaction: stable argsort sends members (in original token
        # order, so causal order is preserved) to the front; keep the first cap.
        order = torch.argsort(
            members.to(torch.int8), dim=1, descending=True, stable=True
        )
        gather_idx = order[:, :cap]  # [B, cap] token positions per image
        valid = (
            torch.arange(cap, device=x.device)[None, :] < counts[:, None]
        )  # [B, cap] real (non-pad) compact slots

        x_c = torch.gather(
            x, 1, gather_idx.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        pos_c = torch.gather(pos, 1, gather_idx)

        shape = (batch_size, cap, self.num_heads, self.head_dim)
        q = self.rope(self.q_proj(x_c).view(shape).transpose(1, 2), pos_c)
        k = self.rope(self.k_proj(x_c).view(shape).transpose(1, 2), pos_c)
        v = self.v_proj(x_c).view(shape).transpose(1, 2)

        # Gather the [B, L, L] pair mask down to [B, cap, cap]; non-member keys
        # map to masked entries, so padded compact slots are never attended.
        # Skipped entirely once the compact-causal identity is verified.
        compact_mask: Tensor | None = None
        if attn_mask is not None and not self.gathered_causal_flash_ok:
            mask_rows = torch.gather(
                attn_mask, 1, gather_idx.unsqueeze(-1).expand(-1, -1, seq_len)
            )
            compact_mask = torch.gather(
                mask_rows, 2, gather_idx.unsqueeze(1).expand(-1, cap, -1)
            )

        if (
            compact_mask is not None
            and self._gathered_causal_flash
            and self._gathered_causal_flash_ok is None
        ):
            # One-time verification: the compact mask must be exactly compact
            # causality restricted to the real slots. Costs one host sync per
            # module, ever; a mismatch is a permanent fallback (never raises).
            assert compact_mask is not None  # unverified => mask was gathered
            causal_ref = (
                torch.tril(
                    torch.ones(cap, cap, dtype=torch.bool, device=x.device)
                ).unsqueeze(0)
                & valid[:, :, None]
                & valid[:, None, :]
            )  # [B, cap, cap]
            self._gathered_causal_flash_ok = bool(
                torch.equal(compact_mask, causal_ref)
            )

        chunk = causal_full_query_chunk()
        if causal_ok or self.gathered_causal_flash_ok:
            if chunk > 0 and cap > chunk and self.head_dim % 8 != 0 and q.dtype == torch.float32:
                # No fused fp32 kernel at this head_dim: the math kernel would
                # materialise [B, H, cap, cap] (60 GiB at cap = 64k for the
                # 100m arms); same math in query blocks instead.
                out = causal_attention_query_chunked(q, k, v, scale=1.0 / self.head_dim, chunk=chunk)
            else:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    dropout_p=0.0,
                    scale=1.0 / self.head_dim,
                )
        else:
            assert compact_mask is not None  # guaranteed by the guard above
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=compact_mask.unsqueeze(1),
                dropout_p=0.0,
                scale=1.0 / self.head_dim,
            )
        out = out.transpose(1, 2).reshape(batch_size, cap, inner_dim)
        out_c = self.out_proj(out)
        # Zero padded slots (where() discards any NaN from fully-masked rows).
        out_c = torch.where(valid.unsqueeze(-1), out_c, out_c.new_zeros(()))

        # Build the output buffer from out_c (the out_proj result), NOT x: under
        # autocast out_c is fp16 while x is the RMSNorm input (fp32-upcast), and
        # scatter_ requires self.dtype == src.dtype. This also makes the gathered
        # path return the same dtype as the non-gathered forward() so the fp16
        # residual stream / attn_update add stays consistent. (Token-dispatch
        # path only.)
        y = out_c.new_zeros(batch_size, seq_len, self.dim)
        y.scatter_(
            1, gather_idx.unsqueeze(-1).expand(-1, -1, self.dim), out_c
        )
        return y

    @torch.compiler.disable
    def forward_gathered_full(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        members: Bool[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """``forward_gathered`` under the no-padding contract (opt-in via
        ``NESTOR_DENSE_FULL_ATTN``, like the dense ``forward_full``).

        Runs OUTSIDE torch.compile (``@torch.compiler.disable``): on torch
        2.10 dynamo spins >25 min pre-step-1 tracing this method inside the
        compiled model (one rank pegged
        ~90% CPU holding the GIL while the others wait at the first
        collective — not reproducible on 2.11 CPU). The ``.item()`` capacity
        sync graph-breaks here anyway, so the fusion loss is minimal: the
        method is gathers + one SDPA + a scatter.

        With an all-true ``base_attn_mask`` the gathered ``[B, cap, cap]``
        pair mask is exactly ``valid & valid`` (same image, same group), so we
        skip the caller's per-group ``[B, S, S]`` mask build and this method's
        two ``[B, cap, S]`` / ``[B, cap, cap]`` mask gathers, and mask only
        the padded compact KEY slots with a broadcast ``[B, 1, 1, cap]`` mask.
        Padded QUERY rows compute garbage on both paths (fully-masked rows
        give NaN there, unmasked pad rows give finite junk here) and are
        zeroed by the same ``where`` below, so member outputs are numerically
        identical to ``forward_gathered`` with an all-true base mask.
        """

        batch_size, seq_len, _ = x.shape
        inner_dim = self.num_heads * self.head_dim
        counts = members.sum(dim=1)  # members per image
        # A static capacity skips the .item() entirely -- that call is a graph
        # break, so removing it is what lets Inductor fuse across this region.
        # It can drop a group's overflow; see _static_capacity.
        cap = _static_capacity(seq_len)
        if cap is None:
            cap = int(counts.max().item())  # capacity; no token is dropped
            cap = _bucket_capacity(cap, seq_len)

        if cap == 0:
            # Same empty-group guard as forward_gathered (keeps every
            # projection in the autograd graph for FSDP2/DDP ordering).
            probe = self.out_proj(
                self.q_proj(x[:, :1])
                + self.k_proj(x[:, :1])
                + self.v_proj(x[:, :1])
            )
            return probe.new_zeros(batch_size, seq_len, self.dim) + 0.0 * probe.sum()

        order = torch.argsort(
            members.to(torch.int8), dim=1, descending=True, stable=True
        )
        gather_idx = order[:, :cap]  # [B, cap] token positions per image
        valid = (
            torch.arange(cap, device=x.device)[None, :] < counts[:, None]
        )  # [B, cap] real (non-pad) compact slots

        x_c = torch.gather(
            x, 1, gather_idx.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        pos_c = torch.gather(pos, 1, gather_idx)

        shape = (batch_size, cap, self.num_heads, self.head_dim)
        q = self.rope(self.q_proj(x_c).view(shape).transpose(1, 2), pos_c)
        k = self.rope(self.k_proj(x_c).view(shape).transpose(1, 2), pos_c)
        v = self.v_proj(x_c).view(shape).transpose(1, 2)

        # Key-only validity mask: padded keys are never attended; padded
        # queries are zeroed after out_proj (below), as in forward_gathered.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid[:, None, None, :],
            dropout_p=0.0,
            scale=1.0 / self.head_dim,
        )
        out = out.transpose(1, 2).reshape(batch_size, cap, inner_dim)
        out_c = self.out_proj(out)
        out_c = torch.where(valid.unsqueeze(-1), out_c, out_c.new_zeros(()))

        y = out_c.new_zeros(batch_size, seq_len, self.dim)
        y.scatter_(
            1, gather_idx.unsqueeze(-1).expand(-1, -1, self.dim), out_c
        )
        return y

    def init_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AttentionGroupKVCache:
        """Initialize the per-group causal KV cache.

        Args:
            batch_size (int): Batch size for cached decoding.
            device (torch.device): Cache device.
            dtype (torch.dtype): Cache dtype.

        Returns:
            AttentionGroupKVCache: Empty cache state.
        """

        empty = torch.empty(
            (batch_size, 0, self.num_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        return AttentionGroupKVCache(
            keys=empty,
            values=empty.clone(),
            active_mask=torch.empty(
                (batch_size, 0), dtype=torch.bool, device=device
            ),
            length=0,
        )

    def forward_step(
        self,
        x: Float[Tensor, "batch_size dim"],
        cache: AttentionGroupKVCache,
        active_mask: Bool[Tensor, "batch_size"],
        pos: Int[Tensor, "batch_size"],
    ) -> tuple[Float[Tensor, "batch_size dim"], AttentionGroupKVCache]:
        """Attend one new token against the cached history.

        Args:
            x (Float[Tensor, "batch_size dim"]): New token embedding.
            cache (AttentionGroupKVCache): Existing group cache.
            active_mask (Bool[Tensor, "batch_size"]): Whether the token is
                assigned to this group on each batch element.
            pos (Int[Tensor, "batch_size"]): Integer position of the new token.

        Returns:
            tuple[Float[Tensor, "batch_size dim"], AttentionGroupKVCache]:
                Attention output and updated cache.
        """

        batch_size = x.shape[0]
        q = self.q_proj(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, self.num_heads, self.head_dim)
        # Rotate the new query/key at their absolute position; cached keys were
        # rotated when they were the current token, so the cache stays valid.
        pos_step = pos.unsqueeze(1)
        q = self.rope(q.unsqueeze(2), pos_step).squeeze(2)
        k = self.rope(k.unsqueeze(2), pos_step).squeeze(2)

        next_length = cache.length + 1
        keys = cache.keys
        values = cache.values
        cached_active_mask = cache.active_mask
        if keys.shape[1] < next_length:
            new_capacity = max(1, keys.shape[1])
            while new_capacity < next_length:
                new_capacity *= 2
            expanded_keys = torch.empty(
                (batch_size, new_capacity, self.num_heads, self.head_dim),
                dtype=k.dtype,
                device=k.device,
            )
            expanded_values = torch.empty_like(expanded_keys)
            expanded_active_mask = torch.zeros(
                (batch_size, new_capacity),
                dtype=torch.bool,
                device=active_mask.device,
            )
            if cache.length > 0:
                expanded_keys[:, : cache.length].copy_(keys[:, : cache.length])
                expanded_values[:, : cache.length].copy_(
                    values[:, : cache.length]
                )
                expanded_active_mask[:, : cache.length].copy_(
                    cached_active_mask[:, : cache.length]
                )
            keys = expanded_keys
            values = expanded_values
            cached_active_mask = expanded_active_mask

        keys[:, cache.length].copy_(k)
        values[:, cache.length].copy_(v)
        cached_active_mask[:, cache.length] = active_mask

        active_prefix = cached_active_mask[:, :next_length]
        out = F.scaled_dot_product_attention(
            q.unsqueeze(2),
            keys[:, :next_length].transpose(1, 2),
            values[:, :next_length].transpose(1, 2),
            attn_mask=active_prefix[:, None, None, :],
            dropout_p=0.0,
            scale=1.0 / self.head_dim,
        ).squeeze(2)
        out = out.reshape(batch_size, -1)
        out = self.out_proj(out)
        out = out * active_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, AttentionGroupKVCache(
            keys=keys,
            values=values,
            active_mask=cached_active_mask,
            length=next_length,
        )


def _depth_inv_rms(
    value: Float[Tensor, "batch_size seq_len dim"],
    eps: float = 1e-6,
) -> Float[Tensor, "batch_size seq_len"]:
    """Inverse channel RMS of one depth source, cached once at append time.

    The AttnRes key RMSNorm factors as ``RMSNorm_g(k) = (g * k) * inv_rms(k)``
    with ``inv_rms`` layer-independent, so each source's inverse RMS is computed
    once here and every consuming sublayer folds its own gain ``g`` into the
    pseudo-query inside :func:`depth_attention`. ``eps`` must match the key
    RMSNorm's (both use the :class:`RMSNorm` default).
    """

    mean_sq = value.pow(2).mean(dim=-1, dtype=torch.float32)
    return torch.rsqrt(mean_sq + eps)


class DepthHistory:
    """AttnRes depth sources: raw sublayer outputs plus cached inverse RMS.

    The values are the *same tensors* already retained for backprop (paper
    Sec. 3.1: Full AttnRes adds no activation memory in vanilla training);
    :func:`depth_attention` must read them in place. ``inv_rms`` caches the
    layer-independent part of the key RMSNorm so reads never re-stack or
    re-normalize the whole history (the previous implementation did, retaining
    O(M^2) fresh ``(H, batch, seq, dim)`` copies per block in autograd -- the
    AttnRes OOM).
    """

    __slots__ = ("values", "inv_rms")

    def __init__(self) -> None:
        self.values: list[Float[Tensor, "batch_size seq_len dim"]] = []
        self.inv_rms: list[Float[Tensor, "batch_size seq_len"]] = []

    def append(self, value: Float[Tensor, "batch_size seq_len dim"]) -> None:
        """Add one depth source and cache its inverse RMS."""

        self.values.append(value)
        self.inv_rms.append(_depth_inv_rms(value))

    def __len__(self) -> int:
        return len(self.values)


# Escape hatch, shared with the column reader's ``_depth_softmax`` (see
# ``attention_residual``): NESTOR_DEPTH_READ_EAGER=1 puts BOTH depth reads back
# outside torch.compile. This whole function used to be unconditionally
# ``@torch.compiler.disable``d (40f0b99), because with the depth axis stacked
# FIRST Inductor fused every source's inverse-RMS reduction and the stack into
# one persistent reduction that overflowed shared memory past ~20 sources. The
# depth axis is now LAST (see the layout note in the read itself), which is the
# layout the column reader compiles at 31 and 35 sources on H200, so it compiles by
# default again -- worth roughly 2 * num_layers graph breaks per block plus the
# per-source kernel launches Inductor now fuses away. Flip the flag if a read
# ever defeats even that layout; the measured cost of the break when the column
# arms needed it was ~2.2x step time, so it is a last resort, not a fix.
_DEPTH_READ_EAGER = os.environ.get("NESTOR_DEPTH_READ_EAGER", "0") == "1"


def _maybe_disable_depth_read_compile(
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Graph-break the depth read iff ``NESTOR_DEPTH_READ_EAGER=1``.

    Args:
        fn (Callable[..., Any]): The depth read to wrap.

    Returns:
        Callable[..., Any]: ``fn``, or a ``torch.compiler.disable``d copy.
    """

    return torch.compiler.disable(fn) if _DEPTH_READ_EAGER else fn


@_maybe_disable_depth_read_compile
def depth_attention(
    history: DepthHistory,
    query: ScaledLinear,
    key_norm: RMSNorm,
) -> Float[Tensor, "batch_size seq_len dim"]:
    """Attention Residuals: softmax attention over the depth axis (AttnRes).

    Implements Eq 2-4 of Kimi's *Attention Residuals* (arXiv:2603.15031). Each
    source in ``history`` is a prior sublayer output ``f_i(h_i)`` (or the block
    input ``h_1`` at index 0); they act as both keys and values. The depth-
    attention weights are ``softmax_i( w . RMSNorm(k_i) )`` with the per-sublayer
    pseudo-query ``w`` held as the (zero-init) weight of ``query`` and the key
    RMSNorm ``key_norm``. The returned input to the sublayer is the weighted sum
    of the *raw* (un-normed) values. Mixing is purely over depth -- each token
    position is handled independently, so this never mixes tokens.

    Memory: the logit ``w . RMSNorm_g(k_i)`` is computed as
    ``(w * g) . k_i * inv_rms(k_i)`` -- one mat-vec per source against the
    gain-folded pseudo-query, using the inverse RMS cached on ``history`` --
    and the mix is a running weighted sum. Nothing here materializes (or makes
    autograd retain) an ``(H, batch, seq, dim)`` copy of the history; the only
    per-call activations are ``(H, batch, seq)`` weights and the mixed output.
    Every stack below is over ``(batch, seq)`` SCALARS, ``dim`` times smaller
    than the sources, so that promise holds unchanged.

    Launch count: the depth-dependent work is ``H`` mat-vecs plus a fixed
    handful of ops, not ``O(H)`` elementwise kernels. The inverse-RMS scaling
    is applied once to the stacked logits instead of once per source; the
    softmax weights are ``unbind``-ed (``H`` views, no kernel) instead of
    sliced per source; and the mixture accumulates in place with ``addcmul_``,
    which drops the per-source product temporary and the separate add. Under
    torch.compile Inductor fuses the whole tail -- scaling, softmax and the
    entire accumulation chain -- into a single pointwise pass over the sources.

    Args:
        history (DepthHistory): Prior sublayer outputs (with cached inverse
            RMS), each ``[batch, seq, dim]``; index 0 is the block input ``h_1``.
        query (ScaledLinear): ``dim -> 1`` score head whose weight is the
            pseudo-query ``w`` (zero-init => uniform weights at step 0).
        key_norm (RMSNorm): RMSNorm applied to the keys before the dot product.
            Must be gain-only (``bias=False``) so it folds into the query.

    Returns:
        Tensor: The depth-attended sublayer input ``[batch, seq, dim]``.
    """

    values = history.values
    assert key_norm.bias is None, "AttnRes key norms must be gain-only."
    w_eff = (query.weight.squeeze(0) * key_norm.weight).to(values[0].dtype)
    # Depth axis LAST. Stacking to (H, batch, seq) instead makes Inductor fuse
    # all H inverse-RMS reductions and this stack into ONE persistent reduction
    # whose shared memory scales with H, and past roughly 20 sources there is no
    # valid Triton config at all -- compilation dies with
    #   triton_per_fused_add_div_expand_mul_pow_squeeze_stack_sum_285
    #   Required: 311432   Hardware limit: 232448
    # deterministically, so the retry wrapper cannot rescue it. That killed the
    # 31- and 35-source column arms of the sparsity-schedule series at step 0.
    # With H trailing, each (batch, seq) position writes H contiguous values and
    # H is a short unrolled loop rather than a tile multiplier. The stacked
    # tensor is tiny either way ((batch, seq, H) fp32 against (batch, seq, dim)
    # values). Do not "simplify" this back to dim=0.
    #
    # Scaling by the cached inverse RMS happens AFTER the stack, as one
    # (batch, seq, H) multiply rather than H (batch, seq) ones. Exact: the
    # product is elementwise and ``torch.stack`` promotes, so stacking the
    # mat-vecs and then multiplying by the stacked fp32 scales is the same
    # arithmetic in the same fp32 precision as scaling each source first.
    logits = torch.stack(
        [torch.matmul(value, w_eff) for value in values], dim=-1
    ).float() * torch.stack(
        history.inv_rms, dim=-1
    )  # (batch, seq, H), fp32
    alpha = logits.softmax(dim=-1).to(values[0].dtype)
    # ``unbind`` is one call returning H views; slicing ``alpha[..., i]`` per
    # source was H selects plus H unsqueezes.
    weights = alpha.unsqueeze(-1).unbind(dim=-2)  # H x (batch, seq, 1)
    # The old chain of out-of-place adds promoted the accumulator lazily, so a
    # history whose LATER sources are wider than its first (an fp32 block input
    # appended after bf16 summary projections, under ``attn_res_xblock`` and
    # autocast) still worked. An in-place accumulate cannot widen, so promote
    # once up front. Guarded on a pure-Python dtype comparison so the uniform
    # history -- every other path -- pays no dispatch at all, and ``.to`` is
    # then a no-op that returns the product itself.
    mix_dtype = values[0].dtype
    if any(value.dtype is not mix_dtype for value in values):
        for value in values[1:]:
            mix_dtype = torch.promote_types(mix_dtype, value.dtype)
    mixed = (values[0] * weights[0]).to(mix_dtype)
    for value, weight in zip(values[1:], weights[1:]):
        # In-place accumulate into the freshly allocated ``mixed``: no
        # per-source weighted-value temporary and no separate add. Safe for
        # autograd -- ``addcmul_``'s backward needs the two operands, not the
        # accumulator's pre-update value. ``addcmul`` forms ``value * weight``
        # at the promoted compute dtype and rounds once on the store, so in
        # bf16 this rounds the per-source product one time fewer than the old
        # ``mixed + value * weight`` did -- a reassociation-grade difference
        # (strictly more accurate), exact in fp32.
        mixed.addcmul_(value, weight)
    return mixed


_GROUPED_PROJ_BMM_FALLBACK_LOGGED = False


def _attention_groups_batchable(
    groups: Sequence["GroupRoutedAttention"],
) -> bool:
    """Whether a layer's attention groups can share batched projection GEMMs.

    The batched path (:meth:`RoutedTransformerLayer.
    _gathered_proj_bmm_attention`) stacks the groups' ``q/k/v/out_proj``
    weights into ``[G, out, in]`` tensors and fuses q/k/v along the output
    axis, so every group must agree on ``dim``, ``num_heads``, ``head_dim``
    and on which projections carry a bias; q, k and v must additionally agree
    with each other because their biases are concatenated. The RoPE base must
    match too, since one shared ``inv_freq`` table rotates every group.

    :class:`RoutedTransformerLayer` always builds its groups from one
    ``num_heads``/``head_dim`` pair -- ``attention_auto_head_split`` splits
    heads per BLOCK, not per group, so the g8t3 arms' 8 groups are uniform
    (5 heads x 40/60) -- so this only rejects groups replaced after
    construction.

    Args:
        groups (Sequence[GroupRoutedAttention]): The layer's attention groups.

    Returns:
        bool: ``True`` if a single batched GEMM can serve every group.
    """

    if not groups:
        return False
    signatures = {
        (
            group.dim,
            group.num_heads,
            group.head_dim,
            group.rope.base,
            group.q_proj.bias is not None,
            group.k_proj.bias is not None,
            group.v_proj.bias is not None,
            group.out_proj.bias is not None,
        )
        for group in groups
    }
    if len(signatures) != 1:
        return False
    *_, q_bias, k_bias, v_bias, _ = next(iter(signatures))
    return q_bias == k_bias == v_bias


class RoutedTransformerLayer(StackedBanksMixin, nn.Module):
    """Pre-norm routed transformer layer used inside a Nestor block.

    Declares its native shared / group-specific expert bank for
    ``NESTOR_STACKED_EXPERT_BANKS`` (attribute ``stacked_moe_bank``, stacked
    by the model builder AFTER the reference-muP pass, exactly like the
    expert pools); when stacked, the grouped-GEMM expert dispatch pays one
    cast per bank per micro-step instead of the per-expert
    ``stack -> cast -> transpose`` chain and per-parameter backward copies.
    """

    dim: int
    num_experts: int
    num_shared_experts: int
    num_group_specific_experts: int
    num_attention_groups: int
    expert_router: Router | None

    def __init__(
        self,
        dim: int,
        num_experts_per_group: int,
        num_shared_experts: int,
        num_experts_per_group_per_token: int,
        num_groups: int,
        num_heads: int,
        head_dim: int,
        mlp_hidden_dims: list[int],
        mlp_activation: str,
        router_score_mlp_hidden_dims: list[int],
        router_score_mlp_activation: str,
        router_noise_mlp_hidden_dims: list[int],
        router_noise_mlp_activation: str,
        *,
        depth_scale: float,
        rope_base: float = 10000.0,
        loss_free_balancing: bool = False,
        bias_update_rate: float = 0.0,
        attention_token_dispatch: bool = False,
        num_dense_shared_experts: int = 0,
        residual_mode: str = "convex",
        block_residual_mode: str = "projection",
        kimi_max_sources: int | None = None,
        column_include_aggregate: bool = False,
        column_head_only: bool = False,
        column_deviation_sources: bool = False,
        keeps_columns: bool = True,
        column_trailing_layers: int | None = None,
        drop_path_rate: float = 0.0,
        causal: bool = False,
        num_groups_per_token: int = 1,
        expert_pool_public_experts: int = 0,
        expert_pool_public_top_k: int = 1,
        expert_pool_private_experts: int = 0,
        expert_pool_private_top_k: int = 1,
        expert_pool_active_budget: float | None = None,
        expert_pool_public_budget: float | None = None,
        expert_pool_private_gate: str = "renorm",
    ) -> None:
        """Initialize one routed transformer layer.

        Args:
            dim (int): Token feature width.
            num_experts_per_group (int): Experts visible to each group.
            num_shared_experts (int): Experts shared across groups.
            num_experts_per_group_per_token (int): Top-k experts per token.
            num_groups (int): Attention groups.
            num_heads (int): Heads per group.
            head_dim (int): Width of each head.
            mlp_hidden_dims (list[int]): Expert MLP hidden widths.
            mlp_activation (str): Expert MLP activation.
            router_score_mlp_hidden_dims (list[int]): Hidden widths for the
                expert-router score MLP.
            router_score_mlp_activation (str): Score MLP activation.
            router_noise_mlp_hidden_dims (list[int]): Hidden widths for the
                expert-router noise MLP.
            router_noise_mlp_activation (str): Noise MLP activation.
            depth_scale (float): Convex residual weight ``1/D`` with
                ``D = 2 sum_l M_l`` the total number of branches in the model.
            rope_base (float): RoPE frequency base ``theta_l^rope`` passed to each
                attention group.
            loss_free_balancing (bool): Whether detached bias updates are used.
            bias_update_rate (float): Detached bias update step size.
            attention_token_dispatch (bool): If True, gather each group's member
                tokens before the attention projections (compute scales with
                groups-per-token, not total groups) instead of projecting the
                full sequence and masking. Numerically equivalent.
            block_residual_mode (str): Between-block transition mode of the
                owning backbone. ``"attn_res_kimi"`` builds the per-sublayer
                Kimi depth reads consumed by :meth:`forward_kimi`;
                ``"attn_res_column"`` builds the same two reads for
                :meth:`forward_column`; other values change nothing.
            kimi_max_sources (int | None): Largest depth-read source count of
                the owning backbone (``num_blocks + 1``, or ``1 + sum_b k_b``
                under ``"attn_res_column"``); required under
                ``block_residual_mode="attn_res_kimi"`` /
                ``"attn_res_column"`` to size the reads' per-source value
                gains, ignored otherwise.
            column_include_aggregate (bool): Under ``"attn_res_column"``,
                whether each sublayer read also sees the aggregate of the
                block's live column streams (:func:`_column_read_sources`).
                Ignored when the block routes to a single column, whose
                aggregate is that column.
            num_groups_per_token (int): Top-k of the block's attention-group
                router (``k_a``); consumed by the expert pool's private width
                formula and by ``column_include_aggregate``.
            expert_pool_public_experts (int): Public expert-pool size; 0 with
                a private pool builds the private-only
                :class:`~nestor.backbones.expert_pools.ExpertPoolMLP` (no
                public path, no path gate).
            expert_pool_public_top_k (int): Public experts per token.
            expert_pool_private_experts (int): Private expert-pool size per
                attention group; > 0 enables pooling (0 disables it, and is
                illegal alongside a public pool).
            expert_pool_active_budget (float | None): Active expert width in
                multiples of ``dim``, overriding the pool's 2d/4d contract.
            expert_pool_public_budget (float | None): Public family's active
                width in multiples of ``dim`` (``None`` = the 2d contract).
            expert_pool_private_top_k (int): Private experts per selected
                attention group.
            expert_pool_private_gate (str): Private-router gate rule,
                ``"renorm"`` or ``"switch"`` (see ``Router.gate_mode``).
        """

        super().__init__()
        if not 0 <= num_shared_experts <= num_experts_per_group:
            raise ValueError(
                "num_shared_experts must be in [0, num_experts_per_group]."
            )

        self.dim = dim
        self.num_experts = num_experts_per_group
        self.num_shared_experts = num_shared_experts
        self.num_group_specific_experts = (
            num_experts_per_group - num_shared_experts
        )
        self.num_attention_groups = num_groups
        self.attention_token_dispatch = attention_token_dispatch
        self.depth_scale = depth_scale
        self.reference_residual_multiplier: float | None = None
        self.residual_mode = residual_mode
        # NESTOR_DROP_PATH_RATE overrides the configured rate at launch (read
        # once here, not in the compiled hot path). Set to 0 to disable
        # stochastic depth -- needed for residual_mode="attn_res" under compile,
        # where the drop-path mask ops fuse into the (inherently persistent)
        # depth-attention softmax reduction and overflow H200 shared memory.
        import os as _os

        _dp_env = _os.environ.get("NESTOR_DROP_PATH_RATE")
        self.drop_path_rate = (
            float(_dp_env) if _dp_env is not None else drop_path_rate
        )
        self.causal = causal
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        if residual_mode == "attn_res":
            # Attention Residuals (arXiv:2603.15031): one zero-init pseudo-query
            # and one key RMSNorm per sublayer (attention + MLP). Zero-init holds
            # the depth-attention uniform at step 0. The pseudo-query is a
            # ``dim -> 1`` score head (AdamW @ base/d_in), so it self-tags for
            # the optimizer like any router head.
            self.attn_query = ScaledLinear(
                dim, 1, role="score", bias=False, zero_init=True
            )
            self.attn_key_norm = RMSNorm(dim, bias=False)
            self.mlp_query = ScaledLinear(
                dim, 1, role="score", bias=False, zero_init=True
            )
            self.mlp_key_norm = RMSNorm(dim, bias=False)
        self.block_residual_mode = block_residual_mode
        if block_residual_mode in ("attn_res_kimi", "attn_res_column"):
            # Kimi Block AttnRes (arXiv:2603.15031 Eq. 3-6): one depth read per
            # sublayer, consumed by :meth:`forward_kimi` (the attention read
            # produces the sublayer input; the MLP read runs after the
            # attention update joins the running partial sum).
            # ``attn_res_column`` reuses the SAME two reads (and therefore the
            # same checkpoint keys) -- one shared read per sublayer across all
            # columns; only the sources it is handed change.
            if kimi_max_sources is None:
                raise ValueError(
                    f"block_residual_mode='{block_residual_mode}' requires "
                    "kimi_max_sources."
                )
            self.kimi_attn_read = DepthAttentionRead(
                dim, max_sources=kimi_max_sources
            )
            self.kimi_mlp_read = DepthAttentionRead(
                dim, max_sources=kimi_max_sources
            )
        # Aggregated block-wise residual beside the column-wise ones. A block
        # routed to a single column has nothing to aggregate (its mean IS that
        # column), so the flag only ever fires where it adds a source.
        self._column_aggregate = (
            column_include_aggregate and num_groups_per_token > 1
        )
        # Head-only columns: this layer's reads take the narrow kimi-style
        # source list; only the backbone's final read sees the columns.
        self._column_head_only = column_head_only
        # Deviation sources: live reads see (s_j - a) beside a, mirroring the
        # block-export re-basing (see NestorConfig.block_column_deviation_sources).
        # Gated like the aggregate: a single-column block's deviation is
        # identically zero, so k == 1 keeps its raw stream.
        self._column_deviation = (
            column_deviation_sources and num_groups_per_token > 1
        )
        self.attention_groups = cast(
            list[GroupRoutedAttention],
            nn.ModuleList(
                [
                    GroupRoutedAttention(
                        dim=dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        rope_base=rope_base,
                        causal=causal,
                    )
                    for _ in range(num_groups)
                ]
            ),
        )
        # Public/private expert pools (see nestor/backbones/expert_pools.py):
        # when enabled the pool REPLACES the routed expert machinery entirely
        # -- the expert router and every expert bank are left empty so no
        # unused trainable parameter exists (unused params break DDP).
        pool_enabled = (
            expert_pool_public_experts > 0 or expert_pool_private_experts > 0
        )
        if pool_enabled and (
            num_experts_per_group != 1 or num_dense_shared_experts != 0
        ):
            raise ValueError(
                "Expert pools replace the layer's expert banks: they require "
                "num_experts_per_group == 1 and num_dense_shared_experts == 0 "
                "so no configured expert is silently discarded."
            )
        if pool_enabled:
            # Lazy import: expert_pools.py imports Router from this module.
            from nestor.backbones.expert_pools import ExpertPoolMLP

            self.expert_pool: ExpertPoolMLP | None = ExpertPoolMLP(
                dim=dim,
                num_groups=num_groups,
                num_groups_per_token=num_groups_per_token,
                num_public_experts=expert_pool_public_experts,
                public_top_k=expert_pool_public_top_k,
                num_private_experts=expert_pool_private_experts,
                private_top_k=expert_pool_private_top_k,
                active_budget=expert_pool_active_budget,
                public_budget=expert_pool_public_budget,
                private_gate_mode=expert_pool_private_gate,
                router_score_mlp_hidden_dims=router_score_mlp_hidden_dims,
                router_score_mlp_activation=router_score_mlp_activation,
                router_noise_mlp_hidden_dims=router_noise_mlp_hidden_dims,
                router_noise_mlp_activation=router_noise_mlp_activation,
                loss_free_balancing=loss_free_balancing,
                bias_update_rate=bias_update_rate,
            )
            self.expert_router = None
        else:
            self.expert_pool = None
            self.expert_router = Router(
                feature_dim=dim,
                num_choices=num_experts_per_group,
                num_choices_per_token=num_experts_per_group_per_token,
                score_mlp_hidden_dims=router_score_mlp_hidden_dims,
                score_mlp_activation=router_score_mlp_activation,
                noise_mlp_hidden_dims=router_noise_mlp_hidden_dims,
                noise_mlp_activation=router_noise_mlp_activation,
                loss_free_balancing=loss_free_balancing,
                bias_update_rate=bias_update_rate,
            )

        # Expert FFN factory: ``block_mlp_activation == "swiglu"`` builds a gated
        # SwiGLU expert (down(silu(gate(x))*up(x))); any other activation builds
        # the pointwise ScaledMLP (down(act(up(x)))). The gated module exposes
        # up/down as .layers[0]/[1], so the dense and sparse paths (which call
        # ``expert(x)``) and the grouped-GEMM path (which stacks .layers) both
        # work; the grouped path adds the gate bank when ``expert.gated``.
        def _make_expert() -> nn.Module:
            if mlp_activation == "swiglu":
                return GatedScaledMLP(
                    input_dim=dim,
                    hidden_dims=mlp_hidden_dims,
                    output_dim=dim,
                    activation="silu",
                )
            return ScaledMLP(
                input_dim=dim,
                hidden_dims=mlp_hidden_dims,
                output_dim=dim,
                activation=mlp_activation,
            )

        num_bank_shared = 0 if pool_enabled else num_shared_experts
        num_bank_specific = (
            0 if pool_enabled else self.num_group_specific_experts
        )
        self.shared_experts = cast(
            list[ScaledMLP],
            nn.ModuleList([_make_expert() for _ in range(num_bank_shared)]),
        )
        self.group_specific_experts = cast(
            list[list[ScaledMLP]],
            nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            _make_expert()
                            for _ in range(num_bank_specific)
                        ]
                    )
                    for _ in range(num_groups)
                ]
            ),
        )
        # DeepSeekMoE-style always-on experts: run on EVERY token with unit
        # gate, outside the router (vs ``shared_experts``, which are routed
        # choices shared across groups). Plain dense MLPs — no dispatch.
        self.num_dense_shared_experts = num_dense_shared_experts
        self.dense_shared_experts = cast(
            list[ScaledMLP],
            nn.ModuleList(
                [_make_expert() for _ in range(num_dense_shared_experts)]
            ),
        )
        # NESTOR_STACKED_EXPERT_BANKS: declare the native bank in the order
        # _grouped_expert_residual dispatches it -- [shared | group 0 | group 1
        # | ...] -- so the stacked owner's expert index IS the grouped path's
        # bank index. The model builder stacks it LAST
        # (consolidate_stacked_expert_banks, after the reference-muP pass).
        # Only banks the grouped path can dispatch are declared (>1 expert,
        # one hidden layer); ``dense_shared_experts`` run as standalone dense
        # GEMMs and gain nothing from stacking.
        moe_bank_stackable = (
            self.num_experts > 1 and len(mlp_hidden_dims) == 1
        )
        self._declare_expert_banks(
            [
                (
                    "stacked_moe_bank",
                    (
                        [
                            *self.shared_experts,
                            *(
                                e
                                for group_experts in self.group_specific_experts
                                for e in group_experts
                            ),
                        ]
                        if moe_bank_stackable
                        else []
                    ),
                )
            ]
        )

        # Dense fast-path eligibility: a single attention group and a single
        # expert collapse the routing machinery to identity (group/expert gates
        # are softmax over one logit == 1.0). The degenerate ViT / width-schedule
        # baselines hit this; real MoE layers (>1 expert) never do and keep the
        # general path untouched. The fast path skips the per-layer router
        # ``nonzero`` dispatch + ``group_members.any()`` syncs and the dynamic
        # shapes that block torch.compile/cudagraphs. Numerically identical;
        # set NESTOR_NO_DENSE_FASTPATH=1 to force the general path.
        import os as _os

        self._dense_fastpath = (
            self.num_attention_groups == 1
            and self.num_experts == 1
            and not self.attention_token_dispatch
            and _os.environ.get("NESTOR_NO_DENSE_FASTPATH") != "1"
        )
        # When the input has no padding (e.g. fixed-size image classification),
        # the all-ones mask makes attention fully dense, so the dense path can
        # skip per-layer mask + ``create_block_mask`` and take the maskless flash
        # path. Static (env) flag → stays compile-friendly; only valid when the
        # data truly has no padding, so it is opt-in. Never on causal layers
        # (the maskless path would drop the causal mask).
        self._dense_full_attn = (
            self._dense_fastpath
            and not causal
            and _os.environ.get("NESTOR_DENSE_FULL_ATTN") == "1"
        )
        # NESTOR_DENSE_NO_ROUTER=1: with one expert the expert gates are softmax
        # over a single logit == 1.0 exactly and carry exactly zero gradient, so
        # the router (score MLP + noise MLP + randn + topk machinery) never
        # affects the model and only burns compute — forward AND the
        # numerically-zero backward chain. Skip it and freeze its parameters so
        # DDP and the optimizer drop them. Opt-in env, dense fast path only.
        self._dense_noroute = (
            self._dense_fastpath
            and _os.environ.get("NESTOR_DENSE_NO_ROUTER") == "1"
        )
        if self._dense_noroute and self.expert_router is not None:
            self.expert_router.requires_grad_(False)
        # 1-group shortcuts for the GENERAL path (V-MoE-style layers: a single
        # attention group but >1 expert, so the dense fast path never applies).
        # _group_noroute: the group gates are softmax over one logit == 1.0
        # exactly — skip the gate multiplies (the block fabricates the group
        # router output under the same env flag).
        self._group_noroute = (
            self.num_attention_groups == 1
            and _os.environ.get("NESTOR_DENSE_NO_ROUTER") == "1"
        )
        # _general_full_attn: no padding + one group → attention is fully
        # dense; skip the per-layer [B,S,S] mask build, create_block_mask, and
        # the group_members.any() host sync. Same no-padding contract as
        # NESTOR_DENSE_FULL_ATTN on the dense fast path.
        self._general_full_attn = (
            self.num_attention_groups == 1
            and not self.attention_token_dispatch
            and not causal
            and _os.environ.get("NESTOR_DENSE_FULL_ATTN") == "1"
        )
        # _moe_grouped_mm: sort-based static-shape expert dispatch through one
        # grouped GEMM per MLP layer instead of per-expert nonzero/index_add
        # (host-syncing, graph-breaking dynamic-shape ops — 8/layer for the
        # V-MoE baselines, (1 + G*E_s)*4 for the grouped-attention MoE). The
        # bank is [shared experts | group 0 specific | group 1 specific | ...];
        # shared choices map 1:1 onto it, group-specific choices map onto the
        # token's group slice (top-1 group routing only — the caller falls
        # back to the sparse path when a token can sit in several groups).
        self._moe_grouped_mm = (
            self.num_experts > 1
            and len(mlp_hidden_dims) == 1
            and _os.environ.get("NESTOR_MOE_GROUPED_MM") == "1"
        )
        # _expert_noroute: ONE expert that is shared across groups is an
        # always-on dense MLP — its gate is softmax over a single logit
        # (== 1.0 exactly, carrying exactly zero gradient), so skip the expert
        # router and the nonzero/index_select/index_add dispatch and run the
        # MLP densely (general-path analogue of _dense_noroute; the dropped
        # gate multiplies only zero dead padding lanes, which the block exit
        # re-zeroes). Freeze the dead router so DDP and the optimizer drop it.
        self._expert_noroute = (
            self.num_experts == 1
            and self.num_shared_experts == 1
            and _os.environ.get("NESTOR_DENSE_NO_ROUTER") == "1"
        )
        if (
            self._expert_noroute
            and not self._dense_noroute
            and self.expert_router is not None
        ):
            self.expert_router.requires_grad_(False)
        # _dispatch_full_attn: token-dispatch attention under the same
        # no-padding contract as NESTOR_DENSE_FULL_ATTN: the gathered pair
        # mask reduces to valid&valid, so each group skips the [B,S,S] mask
        # build + the two mask gathers (forward_gathered_full); with a single
        # group the gather is the identity and the maskless full path runs.
        self._dispatch_full_attn = (
            self.attention_token_dispatch
            and not causal
            and _os.environ.get("NESTOR_DENSE_FULL_ATTN") == "1"
        )
        # NESTOR_GROUPED_ATTN_BMM=1: batch the G groups' q/k/v/out projections
        # into 2 grouped GEMMs per layer (see
        # :meth:`_gathered_proj_bmm_attention`). Token dispatch capacity-pads
        # every group to the same width, so the per-group ``ScaledLinear``
        # calls -- 4G tiny GEMMs whose output width (``heads_per_group *
        # head_dim``) is far below the ~128 tensor cores want -- become one
        # ``bmm`` over ``[G, B*cap, d] x [G, d, out]``. Unlike
        # ``_batched_dispatch_attention`` (which also fuses the data movement
        # and needs a top-1 bijection) this works for ANY groups-per-token and
        # for causal models. The per-group modules stay the parameter owners,
        # so muP tagging / reference_mup / checkpoints are untouched.
        # ``NESTOR_GROUPED_PROJ_BMM`` is the original spelling, kept as an alias.
        self._grouped_proj_bmm_requested = (
            self.attention_token_dispatch
            and num_groups > 1
            and (
                _os.environ.get("NESTOR_GROUPED_ATTN_BMM") == "1"
                or _os.environ.get("NESTOR_GROUPED_PROJ_BMM") == "1"
            )
        )
        # Batching also needs structurally uniform groups; decided once here.
        self._grouped_proj_bmm = False
        self.refresh_grouped_proj_bmm()
        # NESTOR_ATTN_PACKED_VARLEN=1: fully packed token-dispatch attention
        # (no capacity, no padding; see _attn_update_packed_varlen). STATIC
        # gate only -- causal by config plus structurally uniform groups --
        # never runtime-verified state, which is checkpoint-recompute-unsafe.
        self._attn_packed_varlen = (
            self.attention_token_dispatch
            and num_groups > 1
            and causal
            and _attention_groups_batchable(self.attention_groups)
            and _os.environ.get("NESTOR_ATTN_PACKED_VARLEN") == "1"
        )
        # Layer-level twin of GroupRoutedAttention._gathered_causal_flash: the
        # batched path verifies compact causality ONCE for all groups at once
        # (they share the base mask and the router, so their verdicts agree).
        self._grouped_proj_bmm_causal_flash = (
            _os.environ.get("NESTOR_GATHERED_CAUSAL_FLASH") == "1"
        )
        self._grouped_proj_bmm_causal_ok: bool | None = None
        # Verify equivalence on the first few forwards (NESTOR_FASTPATH_CHECK=1).
        self._fastpath_check_remaining = (
            3 if _os.environ.get("NESTOR_FASTPATH_CHECK") == "1" else 0
        )
        # NESTOR_STACKED_ATTN_WEIGHTS=1: own the attention projections as the
        # STACKED tensors the packed path consumes ([G, dim, 3*e8] etc.),
        # instead of re-stacking/padding/casting every group's weights on every
        # forward. The per-forward stack chain is pure launch count (~40
        # kernels/layer) and, worse, keeps 4G+ parameter tensors per layer that
        # autocast re-casts individually each microstep. Per-group
        # ``q_proj.weight`` etc. become autograd views of the stacked
        # parameters (reinstalled after every ``_apply``), so any residual
        # reader still works and gradients land on the stacked owner. FRESH
        # RUNS ONLY: the checkpoint parameter set changes, exactly like
        # NESTOR_DENSE_NO_ROUTER.
        #
        # MEASURED NET NEGATIVE at 200m (same-job A/B, bs8 fresh):
        # 0.592 vs 0.561 s/microstep -- the backward regressed +45 ms (DDP
        # bucket-overlap granularity over fewer, larger tensors is the leading
        # suspect), while the removed stack/cast launches bought nothing
        # measurable. Keep OFF unless a future DDP/bucketing change re-opens
        # the question; equivalence tests stay green either way.
        self._stacked_attn = (
            self._attn_packed_varlen
            and _os.environ.get("NESTOR_STACKED_ATTN_WEIGHTS") == "1"
        )
        if self._stacked_attn:
            self._consolidate_attention_weights()
        # NESTOR_DENSE_TAIL_ATTN=1: a block whose tokens select EVERY column
        # (num_groups_per_token == num_groups -- the "dense tail" of the
        # ramp arms) has nothing to dispatch, so its G columns run as ONE
        # fused multi-head attention (see _attn_update_dense_tail). Static
        # gate, config + env only, exactly like the packed path.
        self._dense_tail_attn = (
            self.attention_token_dispatch
            and num_groups > 1
            and num_groups_per_token == num_groups
            and causal
            and _attention_groups_batchable(self.attention_groups)
            and _os.environ.get("NESTOR_DENSE_TAIL_ATTN") == "1"
        )
        # NESTOR_COLUMN_SLOT_UPDATES=1: column-mode sublayers accumulate both
        # their updates in the streams' own [batch, seq, k, dim] slot layout
        # instead of group-wide [batch, seq, G, dim] intermediates (see
        # _forward_column_slot_updates).
        self._column_slot_updates = (
            _os.environ.get("NESTOR_COLUMN_SLOT_UPDATES") == "1"
        )
        # NESTOR_GROUPED_MM_FUSED_GATE=1 for the native expert bank's grouped
        # dispatch; ExpertPoolMLP reads the same flag for the pools.
        self._moe_fused_gate = fused_gate_requested()
        # NESTOR_PACKED_POINTWISE_COMPILE=1: the pointwise chains of the
        # packed attention path (bias add, fp32 RoPE, head padding, muP scale,
        # gate multiply) run as torch.compile'd helpers instead of ~25 eager
        # kernels per layer per micro-step. The grouped GEMMs and the varlen
        # kernel stay eager (their operand layouts are why the method is
        # compile-disabled); only the chains BETWEEN them fuse. Exact: both
        # settings call the same helper functions, compiled or not (see
        # nestor/backbones/packed_pointwise.py).
        self._packed_pointwise = (
            self._attn_packed_varlen and packed_pointwise_requested()
        )
        # NESTOR_TRITON_PACKED_ATTN=1: hand-written Triton kernels replace
        # EVERY op between the packed path's GEMMs — the pack gather, the
        # bias/RoPE/pad/scale chain, and the gate+slot-gather combine (which
        # fuses across the _PackedSlotGather autograd boundary that the
        # compiled chains cannot cross) — plus a head-padded weight-stack
        # layout that eliminates the attn_out_rows unpad/re-pad entirely.
        # Exact (<= 1-2 bf16 ulp, same bar as every fast path), resume-safe
        # (no parameter change), read at construction like every flag. Takes
        # precedence over NESTOR_PACKED_POINTWISE_COMPILE, which it subsumes.
        # See nestor/backbones/triton_packed.py.
        self._triton_packed = (
            self._attn_packed_varlen
            and not self._stacked_attn
            and self.attention_groups[0].head_dim % 2 == 0
            and triton_packed_requested()
            and triton_packed_available()
        )
        # NESTOR_TRITON_INDEX_BUILD=1: the native bank's grouped dispatch
        # builds its per-row index maps in ONE Triton kernel instead of the
        # ~10-kernel scatter cascade (integer-exact; pure launch count). The
        # pools read the same flag themselves (ExpertPoolMLP).
        self._triton_index_build = (
            triton_index_requested() and triton_index_available()
        )
        # NESTOR_TRITON_EXPERT_DISPATCH (default follows INDEX_BUILD):
        # the Triton packed path builds its head-padded weight stacks as ONE
        # stack + one permute/pad copy + one cast per projection set instead
        # of the per-group chain (identical tensor, ~1,700 fewer launches per
        # micro-step at 500m). The pools read the same flag for their fused
        # dispatch kernels (ExpertPoolMLP.triton_fused).
        self._batched_attn_stacks = triton_expert_requested()

    def refresh_grouped_proj_bmm(self) -> None:
        """(Re)decide whether the batched projection path may run.

        ``self._grouped_proj_bmm`` is the AND of "requested" (env flag, token
        dispatch, more than one group -- fixed at construction) and "batchable"
        (:func:`_attention_groups_batchable`, a property of the current
        :attr:`attention_groups`). Called from ``__init__``; call it again after
        replacing any group module so the layer re-checks the shapes.

        A requested-but-not-batchable layer logs once per process and keeps the
        per-group loop -- it never raises, so mixed-shape groups degrade to
        today's behaviour instead of crashing a run. The verdict depends only on
        module structure, so every DDP rank reaches the same one and the
        collectives stay identical.
        """

        global _GROUPED_PROJ_BMM_FALLBACK_LOGGED

        batchable = _attention_groups_batchable(self.attention_groups)
        self._grouped_proj_bmm = self._grouped_proj_bmm_requested and batchable
        if self._grouped_proj_bmm_requested and not batchable:
            if not _GROUPED_PROJ_BMM_FALLBACK_LOGGED:
                _GROUPED_PROJ_BMM_FALLBACK_LOGGED = True
                logging.getLogger(__name__).warning(
                    "NESTOR_GROUPED_ATTN_BMM=1 but this layer's attention "
                    "groups are not structurally uniform (mixed dim / heads / "
                    "head_dim / bias); keeping the per-group projection loop."
                )

    def _apply_residual(
        self,
        x: Float[Tensor, "... dim"],
        update: Float[Tensor, "... dim"],
    ) -> Float[Tensor, "... dim"]:
        """Combine a residual stream and branch under the configured rule.

        Args:
            x (Float[Tensor, "... dim"]): Residual stream.
            update (Float[Tensor, "... dim"]): Branch output.

        Returns:
            Float[Tensor, "... dim"]: Updated residual stream.
        """

        if self.reference_residual_multiplier is not None:
            return x + self.reference_residual_multiplier * update
        return convex_residual(x, update, self.depth_scale)

    def _accumulate_sparse_expert_bank(
        self,
        residual: Float[Tensor, "num_tokens dim"],
        x: Float[Tensor, "num_tokens dim"],
        experts: list[ScaledMLP],
        expert_weights: Float[Tensor, "num_tokens num_experts"],
    ) -> None:
        """Apply an expert bank only on tokens with non-zero routing weight."""

        for expert_idx, expert in enumerate(experts):
            token_indices = torch.nonzero(
                expert_weights[:, expert_idx],
                as_tuple=False,
            ).squeeze(-1)
            if token_indices.numel() == 0:
                # Run a single-token forward to keep the autograd graph identical
                # across ranks. FSDP2's reduce-scatter fires per-module during
                # backward; skipping an expert removes it from the graph, causing
                # ranks to issue collectives in different order → deadlock.
                expert_out = expert(x[:1])
                residual[:1].add_((expert_out * 0).to(residual.dtype))
                continue
            expert_out = expert(x.index_select(0, token_indices))
            weighted_out = expert_out * expert_weights.index_select(
                0,
                token_indices,
            )[:, expert_idx].unsqueeze(-1)
            # Under bf16 autocast the experts emit bf16 while the accumulator
            # follows the (fp32) residual stream; accumulate in the wider dtype.
            residual.index_add_(
                0, token_indices, weighted_out.to(residual.dtype)
            )

    def _compute_expert_residual(
        self,
        x: Float[Tensor, "... dim"],
        group_router_output: RouterOutput,
        expert_router_output: RouterOutput,
    ) -> Float[Tensor, "... dim"]:
        """Route shared and group-specific experts sparsely over the active tokens."""

        flat_x = x.reshape(-1, self.dim)
        flat_group_gates = group_router_output.gates.reshape(
            -1,
            self.num_attention_groups,
        )
        flat_expert_gates = expert_router_output.gates.reshape(
            -1, self.num_experts
        )
        residual = torch.zeros_like(flat_x)

        if self.num_shared_experts > 0:
            self._accumulate_sparse_expert_bank(
                residual,
                flat_x,
                self.shared_experts,
                flat_expert_gates[:, : self.num_shared_experts],
            )

        if self.num_group_specific_experts > 0:
            specific_gates = flat_expert_gates[:, self.num_shared_experts :]
            for group_idx, group_experts in enumerate(
                self.group_specific_experts
            ):
                self._accumulate_sparse_expert_bank(
                    residual,
                    flat_x,
                    group_experts,
                    specific_gates
                    * flat_group_gates[:, group_idx : group_idx + 1],
                )

        return residual.reshape_as(x)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
        """Apply group-routed attention then expert-routed MLPs.

        Dispatches to the dense fast path when the layer is structurally
        degenerate (1 group, 1 expert); otherwise the general sparse path.
        ``dispatch_plan`` (block-level, multi-group token dispatch under the
        no-padding contract) selects the batched dispatch attention.
        """

        if not self._dense_fastpath:
            return self._forward_general(
                x,
                base_attn_mask,
                valid_token_mask,
                group_router_output,
                pos,
                dispatch_plan=dispatch_plan,
                router_noise=router_noise,
            )

        if self._fastpath_check_remaining > 0:
            self._fastpath_check_remaining -= 1
            out_fast, aux_fast = self._forward_dense(
                x, base_attn_mask, valid_token_mask, group_router_output, pos,
                router_noise=router_noise,
            )
            with torch.no_grad():
                out_ref, _ = self._forward_general(
                    x, base_attn_mask, valid_token_mask, group_router_output, pos,
                    router_noise=router_noise,
                )
                diff = (out_fast.detach() - out_ref).abs().max().item()
            print(
                f"[FASTPATH-CHECK] max|dense - general| = {diff:.3e}",
                flush=True,
            )
            return out_fast, aux_fast

        return self._forward_dense(
            x, base_attn_mask, valid_token_mask, group_router_output, pos,
            router_noise=router_noise,
        )

    def _drop_path(
        self, update: Float[Tensor, "batch_size seq_len dim"]
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """Stochastic depth on a sublayer update (timm-style, per-sample).

        Drops the whole update for a fraction ``drop_path_rate`` of batch
        elements at train time and rescales survivors by ``1/(1-p)`` so the
        expected update is unchanged. No-op at eval or when the rate is 0. The
        mask is drawn in fp32 (bf16's 8-bit mantissa would bias the keep prob).
        """

        p = self.drop_path_rate
        if p <= 0.0 or not self.training:
            return update
        keep = 1.0 - p
        shape = (update.shape[0],) + (1,) * (update.dim() - 1)
        mask = (torch.rand(shape, device=update.device) < keep).to(update.dtype)
        return update * mask / keep

    def _forward_dense(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
        """Sync-free path for 1-group / 1-expert layers (numerically identical).

        With a single attention group and a single expert the group and expert
        gates are ``softmax`` over one logit (== 1.0), so routing reduces to:
        run the one attention group masked by token validity, then apply the one
        expert densely. No ``any()`` / ``nonzero`` syncs, no dynamic shapes — so
        the layer compiles to a clean fused graph.
        """

        norm_x = self.attn_norm(x)
        attn_update = self._attn_update_dense(
            norm_x, base_attn_mask, group_router_output, pos
        )
        x = self._apply_residual(x, self._drop_path(attn_update))

        norm_x = self.mlp_norm(x)
        expert_update, expert_aux = self._expert_update(
            norm_x, valid_token_mask, group_router_output, dense=True,
            router_noise=router_noise,
        )
        x = self._apply_residual(x, self._drop_path(expert_update))
        return x, expert_aux

    def _attn_update_dense(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """Single-group attention update from a pre-normalized input.

        Extracted from the dense fast path so the convex residual
        (:meth:`_forward_dense`) and the Attention-Residuals path
        (:meth:`forward_attn_res`) share one implementation.
        """

        # The 1-group gates are exactly 1.0 for valid tokens; under
        # NESTOR_DENSE_NO_ROUTER the multiplies are dropped entirely (they only
        # zero dead padding lanes, which the block exit re-zeroes anyway).
        gate = (
            None
            if self._dense_noroute
            else group_router_output.gates[..., 0].unsqueeze(-1)
        )
        if self._dense_full_attn:
            # No padding + single group → fully dense attention; skip the mask.
            attn_update = self.attention_groups[0].forward_full(norm_x, pos)
        elif base_attn_mask is None:
            # Trivial-mask contract: every token is valid, and 1-group
            # membership is all-true (softmax over one logit), so the group
            # mask would be exactly ``tril`` (causal) or all-true. Maskless
            # flash path -- never builds an [S, S] tensor of any kind.
            group = self.attention_groups[0]
            attn_update = (
                group.forward_causal_full(norm_x, pos)
                if self.causal
                else group.forward_full(norm_x, pos)
            )
        else:
            group_members = group_router_output.active_mask[:, :, 0]
            group_mask = (
                base_attn_mask
                & group_members[:, :, None]
                & group_members[:, None, :]
            )
            attn_update = self.attention_groups[0](norm_x, group_mask, pos)
        if gate is not None:
            attn_update = attn_update * gate
        return attn_update

    def draw_router_noise(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor] | None:
        """Pre-draw this layer's expert-router noise factors (train mode only).

        Called by the block OUTSIDE the layer's activation checkpoint and
        handed back in as ``router_noise`` so the checkpoint's recompute
        routes exactly as the forward did -- no RNG runs inside the
        recomputed region (see :meth:`Router.draw_noise`). Drawn in train mode
        whether or not checkpointing is on, so the RNG consumption is the
        same either way.

        Args:
            batch_size (int): Batch size.
            seq_len (int): Sequence length.
            device (torch.device): Device of the router scores.
            dtype (torch.dtype): Dtype of the router scores (the autocast
                dtype under autocast).

        Returns:
            dict[str, Tensor] | None: ``None`` in eval mode or when the layer
            routes deterministically; otherwise the pool's
            ``{"private", "public"}`` slabs or the bank router's ``"bank"``.
        """

        if not self.training:
            return None
        if self.expert_pool is not None:
            return self.expert_pool.draw_router_noise(
                batch_size, seq_len, device, dtype
            )
        if getattr(self, "_dense_noroute", False):
            return None
        return {
            "bank": self.expert_router.draw_noise(batch_size, seq_len, device, dtype)
        }

    def _expert_update(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        *,
        dense: bool,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
        """Expert MLP update plus its aux dictionary, on any residual path.

        Single entry point used by every forward variant (convex dense /
        general, :meth:`forward_attn_res`, :meth:`forward_kimi`, and -- via
        :meth:`_expert_update_column` -- :meth:`forward_column`). Dispatches
        to the expert pool when one is configured -- the pool replaces the
        routed expert banks on ALL paths -- otherwise to the standard dense
        or general expert update selected by ``dense``.

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                sublayer input (the ``mlp_norm`` output).
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of
                real tokens.
            group_router_output (RouterOutput): Block-level group router
                output.
            dense (bool): Whether the caller is on the dense fast path.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
                The expert update and the layer aux dictionary. With a pool,
                ``"expert_router"`` carries the public router output (OMITTED
                in private-only mode, where no public router exists) and
                ``"expert_pool_private"`` one aux per private pool (consumed
                by :meth:`update_detached_parameters` for loss-free
                balancing); otherwise ``"expert_router"`` carries the expert
                router output as before.
        """

        if self.expert_pool is not None:
            expert_update, public_output, private_outputs = self.expert_pool(
                norm_x,
                valid_token_mask,
                group_router_output.gates,
                group_router_output.active_mask,
                router_noise=router_noise,
            )
            return expert_update, self._pool_aux(public_output, private_outputs)
        if dense:
            expert_update, expert_router_output = self._expert_update_dense(
                norm_x, valid_token_mask, group_router_output,
                bank_noise=None if router_noise is None else router_noise.get("bank"),
            )
        else:
            expert_update, expert_router_output = self._expert_update_general(
                norm_x, valid_token_mask, group_router_output,
                bank_noise=None if router_noise is None else router_noise.get("bank"),
            )
        return expert_update, {
            "expert_router": _router_output_to_aux(expert_router_output)
        }

    def _pool_aux(
        self,
        public_output: RouterOutput | None,
        private_outputs: list[RouterOutput],
    ) -> dict[str, Any]:
        """Build the layer aux dictionary of an expert-pool MLP.

        Args:
            public_output (RouterOutput | None): Public router output; ``None``
                in private-only mode, where the key is omitted entirely.
            private_outputs (list[RouterOutput]): One output per private pool.

        Returns:
            dict[str, Any]: ``"expert_pool_private"`` (one aux per private
            pool, consumed by :meth:`update_detached_parameters` for loss-free
            balancing) plus ``"expert_router"`` when a public pool exists.
        """

        pool_aux: dict[str, Any] = {
            "expert_pool_private": [
                _router_output_to_aux(private_output)
                for private_output in private_outputs
            ],
        }
        if public_output is not None:
            pool_aux["expert_router"] = _router_output_to_aux(public_output)
        return pool_aux

    def _expert_update_column(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        column_plan: _ColumnStreamPlan,
        *,
        dense: bool,
        defer_scatter: bool = False,
        slot_layout: bool = False,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Any, ...]:
        """Expert MLP update laid out over the column-wise residual streams.

        With ``defer_scatter`` the per-group private tensor and the broadcast
        public part are returned SEPARATELY (as ``(private_per_group | None,
        public | None, aux)``) instead of a single slot-space update. Attention
        and the private pool index by the same column -> slot map, so a caller
        holding both can add them per group and scatter the sum ONCE per
        sublayer instead of twice. Costs nothing when it does not apply: a
        column-agnostic feed-forward returns its update as the broadcast part.

        With ``slot_layout`` the split is the same but the private part comes
        back ALREADY in slot layout, ``[batch, seq, k, dim]`` (as
        ``(private_per_slot | None, public | None, aux)``), from
        :meth:`~nestor.backbones.expert_pools.ExpertPoolMLP.forward_per_slot`:
        the grouped dispatch produces its rows slot-major anyway, so no
        group-wide tensor and no scatter are needed at all. Mutually
        exclusive with ``defer_scatter``.

        The MLP counterpart of ``column_plan`` in :meth:`_attn_update_general`:
        the same experts on the same tokens, only the reduction changes.

        - **No expert pool** (plain dense MLP or ordinary MoE): the feed-forward
          is column-agnostic, so it stays ONE update, returned with a
          length-1 slot axis that broadcasts onto every stream.
        - **Expert pool**: the pool's PRIVATE experts are already per attention
          group -- a token's ``k_a`` selected groups each run their own private
          pool, weighted by that group's attention gate, and
          :meth:`~nestor.backbones.expert_pools.ExpertPoolMLP.forward`
          only sums them at the very end. Those per-group tensors ARE the
          column-specific signal the streams exist to keep apart, so each is
          scattered into the slot holding its column
          (:func:`_scatter_columns_to_slots`, same ``slot_of`` map attention
          uses). The PUBLIC pool is shared by construction and broadcasts to
          all slots. The learned 2-logit path gate is applied inside the pool,
          BEFORE the split, exactly as on every other path.

        Costs exactly the same FLOPs as the broadcast form: the per-group
        private tensors already exist transiently inside the pool (they are
        summed on its last line), so only their accumulation target changes --
        no expert is evaluated twice and none is added.

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                sublayer input (the ``mlp_norm`` output).
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of real
                tokens.
            group_router_output (RouterOutput): Block-level group router output.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            dense (bool): Whether the caller is on the dense fast path.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len bcast_slots dim"],
            dict[str, Any]]: The update with a slot axis of length
            ``column_plan.num_slots`` (pooled) or 1 (broadcast), and the layer
            aux dictionary -- identical in either case to
            :meth:`_expert_update`'s.
        """

        assert not (defer_scatter and slot_layout)
        if self.expert_pool is None:
            expert_update, expert_aux = self._expert_update(
                norm_x, valid_token_mask, group_router_output, dense=dense,
                router_noise=router_noise,
            )
            if defer_scatter or slot_layout:
                # Column-agnostic: nothing to scatter, so it is purely the
                # broadcast part.
                return None, expert_update, expert_aux
            return expert_update.unsqueeze(2), expert_aux
        if slot_layout:
            (
                public_update,
                private_per_slot,
                public_output,
                private_outputs,
            ) = self.expert_pool.forward_per_slot(
                norm_x,
                valid_token_mask,
                group_router_output.gates,
                group_router_output.active_mask,
                group_router_output.topk_indices,
                router_noise=router_noise,
            )
            return (
                private_per_slot,
                public_update,
                self._pool_aux(public_output, private_outputs),
            )
        (
            public_update,
            private_per_group,
            public_output,
            private_outputs,
        ) = self.expert_pool.forward_per_group(
            norm_x,
            valid_token_mask,
            group_router_output.gates,
            group_router_output.active_mask,
            router_noise=router_noise,
        )
        if defer_scatter:
            # Hand the caller the per-group tensor and the broadcast public
            # part separately, so it can add attention's per-group update and
            # scatter the SUM once. Attention and the private pool index by the
            # same column -> slot map, so one scatter serves both.
            return (
                private_per_group,
                public_update,
                self._pool_aux(public_output, private_outputs),
            )
        expert_update = _scatter_columns_to_slots(
            private_per_group, column_plan
        )
        if public_update is not None:
            expert_update = expert_update + public_update.unsqueeze(2)
        return expert_update, self._pool_aux(public_output, private_outputs)

    def _expert_update_dense(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        bank_noise: Tensor | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], RouterOutput]:
        """Single-expert MLP update from a pre-normalized input.

        Shared by the dense convex residual and Attention-Residuals paths.
        """

        if self._dense_noroute:
            expert_router_output = _constant_router_output(
                norm_x, valid_token_mask, self.num_experts
            )
        else:
            expert_router_output = self.expert_router(
                norm_x, valid_token_mask, noise=bank_noise
            )
        flat_x = norm_x.reshape(-1, self.dim)
        if self.num_shared_experts == 1:
            expert = self.shared_experts[0]
        else:
            expert = self.group_specific_experts[0][0]
        weight = None
        if not self._dense_noroute:
            expert_gate = expert_router_output.gates.reshape(
                -1, self.num_experts
            )[:, 0:1]
            if self.num_shared_experts == 1:
                weight = expert_gate
            else:
                group_gate = group_router_output.gates.reshape(
                    -1, self.num_attention_groups
                )[:, 0:1]
                weight = expert_gate * group_gate
        expert_out = expert(flat_x)
        if weight is not None:
            expert_out = expert_out * weight
        expert_update = expert_out.reshape_as(norm_x)
        for dense_shared in self.dense_shared_experts:
            expert_update = expert_update + dense_shared(norm_x)
        return expert_update, expert_router_output

    def _forward_general(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
        """General sparse routing path (>1 group or >1 expert).

        Both branches use the convex residual ``(1 - 1/D) h + (1/D) f(h)`` so the
        network's sensitivity stays ``Theta(1)`` at any depth.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Input sequence.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Mask
                shared across attention groups before group membership masking.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Token-validity
                mask derived from positions.
            group_router_output (RouterOutput): Block-level group router output.
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
                Updated token sequence and aux metadata.
        """

        norm_x = self.attn_norm(x)
        attn_update = self._attn_update_general(
            norm_x,
            base_attn_mask,
            group_router_output,
            pos,
            dispatch_plan=dispatch_plan,
        )
        x = self._apply_residual(x, self._drop_path(attn_update))

        norm_x = self.mlp_norm(x)
        expert_update, expert_aux = self._expert_update(
            norm_x, valid_token_mask, group_router_output, dense=False,
            router_noise=router_noise,
        )
        x = self._apply_residual(x, self._drop_path(expert_update))
        return x, expert_aux

    def _attn_update_general(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        column_plan: _ColumnStreamPlan | None = None,
        defer_column_scatter: bool = False,
    ) -> (
        Float[Tensor, "batch_size seq_len dim"]
        | Float[Tensor, "batch_size seq_len slots dim"]
    ):
        """Group-routed attention update from a pre-normalized input.

        Extracted from the general path so the convex residual
        (:meth:`_forward_general`) and the Attention-Residuals path
        (:meth:`forward_attn_res`) share one implementation.

        ``column_plan`` switches the REDUCTION over attention groups without
        touching the attention math (same kernels, same dispatch, same gates):
        instead of summing every group's gated output into one ``[batch, seq,
        dim]`` update, each group's output is scattered into the stream slot
        that token holds it in (:func:`_build_column_stream_plan`), producing a
        ``[batch, seq, num_slots, dim]`` update for
        ``block_residual_mode='attn_res_column'``. Groups a token did not
        select land in the sentinel slot, which is dropped -- they carry
        exactly-zero gates, so the two forms differ only in layout.

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                sublayer input.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Mask
                shared across groups before membership masking.
            group_router_output (RouterOutput): Block-level group router output.
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions.
            dispatch_plan (_AttentionDispatchPlan | None): Block-level token
                dispatch plan (top-1 routing only).
            column_plan (_ColumnStreamPlan | None): Column-wise residual-stream
                plan; None reduces over groups as usual.

        Returns:
            Float[Tensor, "batch_size seq_len dim"] | Float[Tensor,
            "batch_size seq_len slots dim"]: The gate-weighted sum of the
            groups' attention updates, or -- with ``column_plan`` -- their
            per-slot scatter.
        """

        if (
            self._general_full_attn
            or (self._dispatch_full_attn and self.num_attention_groups == 1)
            or (base_attn_mask is None and self.num_attention_groups == 1)
        ):
            # Single group, no padding: every token is a member and the mask
            # is all-true (with token dispatch the gather is the identity),
            # so run the maskless flash path directly. The gate is exactly
            # 1.0; multiply only when the group router is real. Under the
            # trivial-mask contract (base None) the same holds with causality
            # applied analytically by the is_causal kernel.
            group = self.attention_groups[0]
            attn_update = (
                group.forward_causal_full(norm_x, pos)
                if (base_attn_mask is None and self.causal)
                else group.forward_full(norm_x, pos)
            )
            if not self._group_noroute:
                attn_update = attn_update * group_router_output.gates[
                    ..., 0
                ].unsqueeze(-1)
            # One group => k == 1 => the lone slot is slot 0 (a layer cannot
            # have more slots than groups).
            return _to_single_slot(attn_update, column_plan)

        if (
            isinstance(dispatch_plan, _AttentionDispatchPlan)
            and self._dispatch_full_attn
        ):
            # Multi-group token dispatch, no padding: one batched gather/
            # GEMM/SDPA/scatter for all groups instead of the per-group loop.
            # The plan is built for top-1 routing only (NestorBlock.
            # _attn_dispatch_plan), so again every token owns exactly one slot.
            return _to_single_slot(
                self._batched_dispatch_attention(norm_x, dispatch_plan),
                column_plan,
            )

        if self._dense_tail_attn:
            # Every token is in every column (k == G): nothing to dispatch,
            # so the G columns run as one fused multi-head attention. Static
            # config gate -- see _attn_update_dense_tail.
            return self._attn_update_dense_tail(
                norm_x,
                group_router_output,
                pos,
                column_plan,
                group_layout=defer_column_scatter,
            )

        if self._attn_packed_varlen and (
            column_plan is None or not defer_column_scatter
        ):
            # Fully packed: projections AND attention cost exactly the routed
            # tokens (grouped GEMMs + varlen segments; no capacity anywhere).
            # Static config gate -- see _attn_update_packed_varlen. Under a
            # column plan the packed rows already are (token, slot) pairs, so
            # the update comes back in slot layout directly; only the legacy
            # fused-scatter hand-off (a group-layout tensor) cannot use it.
            return self._attn_update_packed_varlen(
                norm_x,
                group_router_output,
                pos,
                column_plan=column_plan,
                plan=(
                    dispatch_plan
                    if isinstance(dispatch_plan, _PackedAttentionPlan)
                    else None
                ),
            )

        if self._grouped_proj_bmm:
            # Multi-group token dispatch (any groups-per-token, causal or
            # not): keep the per-group compaction but run the 4G projections
            # as 2 grouped GEMMs.
            return self._gathered_proj_bmm_attention(
                norm_x,
                base_attn_mask,
                group_router_output,
                pos,
                column_plan=column_plan,
            )

        # Under a column plan on the token-dispatch path, the per-group
        # contributions are COLLECTED and routed to their slots by ONE
        # vectorized scatter after the loop (_scatter_columns_to_slots, the
        # same op the expert pools use). The previous per-group accumulate did
        # an out-of-place scatter_add per group, each copying the whole
        # [batch, seq, k+1, dim] accumulator: at G=8 that is ~60 ms/microstep
        # of pure copy+scatter on the fast stack -- over half of the column
        # read's remaining speed gap to the Kimi read. The non-dispatch
        # branch keeps the per-group accumulate: it skips empty groups, so its
        # contribution list would have holes.
        batched_columns = (
            column_plan is not None and self.attention_token_dispatch
        )
        per_column: list[Tensor] = []
        attn_update = (
            norm_x.new_zeros(0)
            if batched_columns
            else (
                torch.zeros_like(norm_x)
                if column_plan is None
                else norm_x.new_zeros(
                    (
                        *norm_x.shape[:2],
                        column_plan.num_slots + 1,
                        norm_x.shape[2],
                    )
                )
            )
        )
        for group_idx, group_attn in enumerate(self.attention_groups):
            group_members = group_router_output.active_mask[:, :, group_idx]
            gate = group_router_output.gates[..., group_idx].unsqueeze(-1)
            if self.attention_token_dispatch:
                if self._dispatch_full_attn:
                    # No padding: skip the [B,S,S] mask build and the mask
                    # gathers; only padded compact key slots need masking.
                    contribution = group_attn.forward_gathered_full(
                        norm_x,
                        pos,
                        group_members,
                    ) * gate
                    if batched_columns:
                        per_column.append(contribution)
                    else:
                        attn_update = _accumulate_group_update(
                            attn_update, contribution, group_idx, column_plan
                        )
                    continue
                # Gather member tokens before projection (projection FLOPs no
                # longer scale with the number of groups). Always run -- the
                # gathered path keeps empty groups in the autograd graph.
                #
                # Once the group has verified the compact-causal identity
                # (NESTOR_GATHERED_CAUSAL_FLASH=1) the dense [B, S, S] group
                # mask carries no information, so skip building it too -- it is
                # the largest allocation on this path (16 GiB PER GROUP at
                # S=131072; the 500m arm's OOM at 128k). Under the
                # trivial-mask contract (base None) the identity holds by
                # construction and no mask is ever built.
                group_mask = (
                    None
                    if base_attn_mask is None
                    or group_attn.gathered_causal_flash_ok
                    else base_attn_mask
                    & group_members[:, :, None]
                    & group_members[:, None, :]
                )
                contribution = group_attn(
                    norm_x,
                    group_mask,
                    pos,
                    group_members,
                ) * gate
                if batched_columns:
                    per_column.append(contribution)
                else:
                    attn_update = _accumulate_group_update(
                        attn_update, contribution, group_idx, column_plan
                    )
                continue
            if not group_members.any():
                continue
            if base_attn_mask is None:
                # Trivial-mask contract on the NON-dispatch grouped path: the
                # base mask is exactly ``tril`` (causal) or all-true, but
                # membership masking still needs the dense [B, S, S] form, so
                # materialize it locally. Only short-S arms take this branch
                # (the long-context arms are token-dispatch), so the O(S^2)
                # build is acceptable here -- and _build_block_mask guards the
                # pathological lengths.
                group_mask = (
                    group_members[:, :, None] & group_members[:, None, :]
                )
                if self.causal:
                    seq_len = norm_x.shape[1]
                    group_mask = group_mask & torch.tril(
                        torch.ones(
                            (seq_len, seq_len),
                            dtype=torch.bool,
                            device=norm_x.device,
                        )
                    )
            else:
                group_mask = (
                    base_attn_mask
                    & group_members[:, :, None]
                    & group_members[:, None, :]
                )
            attn_update = _accumulate_group_update(
                attn_update,
                group_attn(
                    norm_x,
                    group_mask,
                    pos,
                ) * gate,
                group_idx,
                column_plan,
            )
        if column_plan is not None:
            if batched_columns:
                stacked = torch.stack(per_column, dim=2)
                if defer_column_scatter:
                    # Hand back the per-group tensor: the caller adds the
                    # private expert update (same column -> slot map) and
                    # scatters the SUM once, halving the per-sublayer scatters.
                    return stacked
                # One scatter for all G groups; sentinel dropped inside.
                return _scatter_columns_to_slots(stacked, column_plan)
            # Drop the sentinel slot (columns no token selected there).
            return attn_update[:, :, : column_plan.num_slots]
        return attn_update

    @_maybe_disable_compile
    def _batched_dispatch_attention(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        plan: _AttentionDispatchPlan,
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        """All groups' token-dispatch attention as batched GEMMs + one SDPA.

        Numerically equivalent to the per-group ``forward_gathered_full``
        loop (modulo GEMM accumulation order). Group-major rows
        (``[G, B*cap, d]``) so the fused Q/K/V projection and the out
        projection run as G LARGE GEMMs; data movement uses the gather-only
        autograd pair (:class:`_AttnDispatchGather` /
        :class:`_AttnCombineGather` — no atomic ``index_add`` in either
        direction); RoPE tables come precomputed from the block-level plan.
        Zero host syncs per call; every group's parameters join the graph
        every step, so the empty-group probe is unnecessary. Slack rows are
        where-masked, carry exactly-zero gates, and are never read by the
        combine. Runs outside torch.compile by default (torch 2.10 dynamo
        stalls on this op family; NESTOR_BATCHED_ATTN_COMPILE=1 opts in).
        """

        batch_size, seq_len, _ = norm_x.shape
        groups = list(self.attention_groups)
        num_groups = len(groups)
        cap = plan.cap
        num_heads = groups[0].num_heads
        head_dim = groups[0].head_dim
        inner_dim = num_heads * head_dim

        rows = _AttnDispatchGather.apply(
            norm_x.reshape(-1, self.dim), plan.src_token, plan.inv_slot
        )
        x_g = rows.view(num_groups, batch_size * cap, self.dim)

        wqkv = torch.stack(
            [
                torch.cat(
                    [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight], dim=0
                )
                for g in groups
            ]
        )  # [G, 3e, d]
        qkv = torch.bmm(x_g, wqkv.transpose(1, 2))  # [G, B*cap, 3e]
        if groups[0].q_proj.bias is not None:
            bqkv = torch.stack(
                [
                    torch.cat(
                        [g.q_proj.bias, g.k_proj.bias, g.v_proj.bias]
                    )
                    for g in groups
                ]
            )  # [G, 3e]
            qkv = qkv + bqkv.to(qkv.dtype).unsqueeze(1)
        qkv = qkv.reshape(
            num_groups, batch_size, cap, 3, num_heads, head_dim
        ).permute(3, 0, 1, 4, 2, 5)  # [3, G, B, H, cap, hd]
        qkv = qkv.reshape(
            3, num_groups * batch_size, num_heads, cap, head_dim
        )

        # RoPE with the plan's block-constant fp32 tables (RoPE.forward
        # semantics: rotate in fp32, cast back).
        q32 = qkv[0].float()
        k32 = qkv[1].float()
        q = (q32 * plan.cos + _rotate_half(q32) * plan.sin).to(qkv.dtype)
        k = (k32 * plan.cos + _rotate_half(k32) * plan.sin).to(qkv.dtype)
        v = qkv[2]

        # Key-only validity mask; slack QUERY rows compute garbage (NaN when
        # a group is empty for an image) and are never read by the combine.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=plan.valid,
            dropout_p=0.0,
            scale=1.0 / head_dim,
        )
        # [G*B, H, cap, hd] -> group-major [G, B*cap, e] for the out bmm.
        out = (
            out.permute(0, 2, 1, 3)
            .reshape(num_groups, batch_size * cap, inner_dim)
        )
        wo = torch.stack([g.out_proj.weight for g in groups])  # [G, d, e]
        out_c = torch.bmm(out, wo.transpose(1, 2))  # [G, B*cap, d]
        bo0 = groups[0].out_proj.bias
        if bo0 is not None:
            bo = torch.stack([g.out_proj.bias for g in groups])
            out_c = out_c + bo.to(out_c.dtype).unsqueeze(1)
        out_rows = out_c.reshape(-1, self.dim)
        out_rows = torch.where(
            plan.valid_rows.unsqueeze(-1), out_rows, out_rows.new_zeros(())
        )
        # Upcast then gate, matching the loop's ``y_g (low prec) * gate
        # (accumulator prec)`` ordering exactly.
        gated = out_rows.to(norm_x.dtype) * plan.gates_rows.unsqueeze(-1).to(
            norm_x.dtype
        )
        y = _AttnCombineGather.apply(
            gated, plan.inv_slot, plan.src_token, plan.valid_rows
        )
        return y.view(batch_size, seq_len, self.dim)

    def _attn_update_dense_tail(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        column_plan: _ColumnStreamPlan | None,
        *,
        group_layout: bool = False,
    ) -> (
        Float[Tensor, "batch_size seq_len dim"]
        | Float[Tensor, "batch_size seq_len slots dim"]
    ):
        """All-columns attention (``k == G``) as one fused multi-head pass.

        In a block whose router selects every column for every token (the
        dense ``k = G`` tail of the ramp arms) the token-dispatch machinery is
        the identity: each column attends causally over the whole sequence
        with its own heads and projections, and the dispatch, packing and
        unpacking move every token to every group and back. Mathematically
        that is one multi-head attention with ``G * H`` heads whose heads are
        partitioned into ``G`` columns with separate output projections and
        per-(token, column) gates, which is what runs here:

        * one fused q/k/v GEMM over the sequence (``[B*S, dim] x [dim,
          3*G*e]``) instead of ``4G`` skinny per-column GEMMs at width ``e``;
        * one causal SDPA call with ``G * H`` heads instead of ``G`` calls
          with ``H`` heads each (the 2-head tail columns starve the GPU);
        * one batched output projection (``[G, B*S, e] x [G, e, dim]``), so
          each column's contribution stays separate for the gates and, in
          column mode, for its stream slot.

        Numerically the per-group loop up to GEMM accumulation order. Under a
        column plan the per-column outputs are gathered straight into slot
        order (slot ``j`` holds column ``topk_indices[..., j]``); with
        ``group_layout`` they are returned column-indexed instead, for the
        legacy fused-scatter path. Static config gate
        (``NESTOR_DENSE_TAIL_ATTN`` + token dispatch + causal + ``k == G`` +
        uniform groups); same pure-causality assumption on the base mask as
        the packed path (padding tokens carry zero gates, so their rows are
        zero as on the per-group path, but they are not excluded as keys --
        the DCLM regime has none).

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                input sequence.
            group_router_output (RouterOutput): Group routing (gates and
                ``topk_indices`` are read; every column is active).
            pos (Int[Tensor, "batch_size seq_len"]): Token positions for RoPE.
            column_plan (_ColumnStreamPlan | None): Column-stream plan; None
                returns the gate-weighted sum over columns.
            group_layout (bool): Under a column plan, return the per-column
                tensor ``[batch, seq, G, dim]`` (column-indexed) instead of
                the slot layout.

        Returns:
            Float[Tensor, "batch_size seq_len dim"] | Float[Tensor,
            "batch_size seq_len slots dim"]: The gated update, summed over
            columns, or laid out per slot / per column.
        """

        groups = list(self.attention_groups)
        num_groups = len(groups)
        batch_size, seq_len, dim = norm_x.shape
        num_tokens = batch_size * seq_len
        num_heads, head_dim = groups[0].num_heads, groups[0].head_dim
        inner_dim = num_heads * head_dim

        x = norm_x.reshape(num_tokens, dim)
        wqkv = torch.stack(
            [
                torch.cat(
                    [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight], dim=0
                )
                for g in groups
            ]
        )  # [G, 3e, dim]
        # F.linear picks the autocast dtype exactly like ScaledLinear does.
        qkv = F.linear(x, wqkv.reshape(num_groups * 3 * inner_dim, dim))
        if groups[0].q_proj.bias is not None:
            bqkv = torch.cat(
                [
                    torch.cat([g.q_proj.bias, g.k_proj.bias, g.v_proj.bias])
                    for g in groups
                ]
            )  # [G*3e]
            qkv = qkv + bqkv.to(qkv.dtype)
        qkv = qkv.view(
            batch_size, seq_len, num_groups, 3, num_heads, head_dim
        ).permute(3, 0, 2, 4, 1, 5)  # [3, B, G, H, S, hd]
        qkv = qkv.reshape(3, batch_size, num_groups * num_heads, seq_len, head_dim)
        rope = groups[0].rope
        q = rope(qkv[0], pos)
        k = rope(qkv[1], pos)
        v = qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0, scale=1.0 / head_dim
        )  # [B, G*H, S, hd]
        # reshape, not view: the flash backend may hand back [B, H, S, hd] as
        # a transposed view of [B, S, H, hd] storage.
        out = (
            out.reshape(batch_size, num_groups, num_heads, seq_len, head_dim)
            .permute(1, 0, 3, 2, 4)
            .reshape(num_groups, num_tokens, inner_dim)
        )
        wo = torch.stack([g.out_proj.weight for g in groups])  # [G, dim, e]
        out_c = torch.bmm(out, wo.transpose(1, 2).to(out.dtype))  # [G, B*S, dim]
        if groups[0].out_proj.bias is not None:
            bo = torch.stack([g.out_proj.bias for g in groups])  # [G, dim]
            out_c = out_c + bo.to(out_c.dtype).unsqueeze(1)
        # Upcast then gate, matching the loop's ``y_g (low prec) * gate
        # (accumulator prec)`` ordering exactly.
        gates = group_router_output.gates.permute(2, 0, 1).reshape(
            num_groups, num_tokens, 1
        )
        gated = out_c.to(norm_x.dtype) * gates.to(norm_x.dtype)
        if column_plan is None:
            return gated.sum(0).view(batch_size, seq_len, dim)
        per_token = gated.permute(1, 0, 2)  # [B*S, G, dim]
        if group_layout:
            return per_token.reshape(batch_size, seq_len, num_groups, dim)
        # Slot j holds column topk_indices[..., j] (every column is selected,
        # so this is a per-token permutation of the columns).
        index = group_router_output.topk_indices.reshape(
            num_tokens, -1, 1
        ).expand(-1, -1, dim)
        return per_token.gather(1, index).view(batch_size, seq_len, -1, dim)

    def _gathered_compact_pair_mask(
        self,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"],
        gather_idx: Int[Tensor, "groups batch_size cap"],
        valid: Bool[Tensor, "groups batch_size cap"],
    ) -> Bool[Tensor, "groups_batch cap cap"]:
        """Compact per-group pair mask for the batched gathered path.

        Equal to the per-group loop's ``gather(gather(base & members (x)
        members))``: a compact slot holds a member exactly when it is
        ``valid``, so the membership factor is the ``valid`` outer product and
        only ``base_attn_mask`` has to be gathered. Gathered one group at a
        time so the transient ``[B, cap, seq_len]`` row buffer stays the same
        size as in the per-group loop (a batched gather would hold ``G`` of
        them at once).

        Args:
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Mask
                shared by all groups (causality / padding), before membership.
            gather_idx (Int[Tensor, "groups batch_size cap"]): Compact slot ->
                original token index.
            valid (Bool[Tensor, "groups batch_size cap"]): Real (non-pad)
                compact slots.

        Returns:
            Bool[Tensor, "groups_batch cap cap"]: Pair mask in the SDPA batch
            layout ``[G*B, cap, cap]``.
        """

        num_groups, batch_size, cap = gather_idx.shape
        seq_len = base_attn_mask.shape[-1]
        per_group: list[Tensor] = []
        for group_idx in range(num_groups):
            idx = gather_idx[group_idx]  # [B, cap]
            rows = torch.gather(
                base_attn_mask,
                1,
                idx.unsqueeze(-1).expand(batch_size, cap, seq_len),
            )  # [B, cap, seq_len]
            per_group.append(
                torch.gather(
                    rows, 2, idx.unsqueeze(1).expand(batch_size, cap, cap)
                )
            )
        compact = torch.stack(per_group)  # [G, B, cap, cap]
        compact = compact & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        return compact.reshape(num_groups * batch_size, cap, cap)

    def _consolidate_attention_weights(self) -> None:
        """Move ownership of the group projections to stacked parameters.

        Builds the exact tensors :meth:`_attn_update_packed_varlen` stacks per
        forward -- ``attn_wqkv_stacked`` ``[G, dim, 3*e8]`` and
        ``attn_wo_stacked`` ``[G, e8, dim]`` (plus fp32 ``attn_bqkv_stacked``
        ``[G, 3*e8]`` / ``attn_bo_stacked`` ``[G, dim]`` when the groups carry
        biases) -- ONCE, from the already-initialized group modules, so the
        parameter values and the RNG stream are identical to the unstacked
        model. muP scaling tags are copied from group 0's weights (all groups
        are structurally uniform here, so their scalings agree). The zero pad
        columns provably receive zero gradients (their forward contributions
        are sliced away), so they stay exactly zero under AdamW.

        The per-group ``ScaledLinear`` weights/biases stop being parameters and
        become views of the stacked owners (:meth:`_install_group_weight_views`),
        which keeps every other read path functional and routes any gradient
        through the views back to the stacked parameters.
        """

        from nestor.scaling.blocks import get_scaling, set_scaling

        groups = list(self.attention_groups)
        dim = groups[0].dim
        inner = groups[0].num_heads * groups[0].head_dim
        e8 = (inner + 7) // 8 * 8
        pad_e = e8 - inner
        with torch.no_grad():
            wqkv = torch.stack(
                [
                    F.pad(
                        torch.cat(
                            [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight],
                            dim=0,
                        ).t().reshape(dim, 3, inner),
                        (0, pad_e),
                    ).reshape(dim, 3 * e8)
                    for g in groups
                ]
            ).contiguous()  # [G, dim, 3*e8]
            wo = torch.stack(
                [
                    F.pad(g.out_proj.weight.t(), (0, 0, 0, pad_e))
                    for g in groups
                ]
            ).contiguous()  # [G, e8, dim]
        self.attn_wqkv_stacked = nn.Parameter(wqkv)
        self.attn_wo_stacked = nn.Parameter(wo)
        scaling = get_scaling(groups[0].q_proj.weight)
        if scaling is not None:
            set_scaling(self.attn_wqkv_stacked, scaling)
            set_scaling(self.attn_wo_stacked, scaling)
        if groups[0].q_proj.bias is not None:
            with torch.no_grad():
                bqkv = torch.stack(
                    [
                        F.pad(
                            torch.cat(
                                [g.q_proj.bias, g.k_proj.bias, g.v_proj.bias]
                            ).reshape(3, inner),
                            (0, pad_e),
                        ).reshape(3 * e8)
                        for g in groups
                    ]
                ).contiguous()
            self.attn_bqkv_stacked = nn.Parameter(bqkv)  # [G, 3*e8]
            bias_scaling = get_scaling(groups[0].q_proj.bias)
            if bias_scaling is not None:
                set_scaling(self.attn_bqkv_stacked, bias_scaling)
        if groups[0].out_proj.bias is not None:
            with torch.no_grad():
                bo = torch.stack(
                    [g.out_proj.bias for g in groups]
                ).contiguous()
            self.attn_bo_stacked = nn.Parameter(bo)  # [G, dim]
            bias_scaling = get_scaling(groups[0].out_proj.bias)
            if bias_scaling is not None:
                set_scaling(self.attn_bo_stacked, bias_scaling)
        for g in groups:
            for lin in (g.q_proj, g.k_proj, g.v_proj, g.out_proj):
                if "weight" in lin._parameters:
                    del lin._parameters["weight"]
                if "bias" in lin._parameters and lin._parameters["bias"] is not None:
                    del lin._parameters["bias"]
        self._install_group_weight_views()

    def _install_group_weight_views(self) -> None:
        """Point each group's ``weight``/``bias`` at views of the stacked owners.

        Views follow the base parameter through autograd, so residual readers
        (fallback paths, probes) behave exactly as before. Must be re-run after
        any ``_apply`` (``.to()``/``.cuda()``): those rebuild the stacked
        parameters, which orphans previously derived views.
        """

        groups = list(self.attention_groups)
        dim = groups[0].dim
        inner = groups[0].num_heads * groups[0].head_dim
        e8 = (inner + 7) // 8 * 8
        wqkv, wo = self.attn_wqkv_stacked, self.attn_wo_stacked
        bqkv = getattr(self, "attn_bqkv_stacked", None)
        bo = getattr(self, "attn_bo_stacked", None)
        for gi, g in enumerate(groups):
            per = wqkv[gi].view(dim, 3, e8)
            g.q_proj.weight = per[:, 0, :inner].t()
            g.k_proj.weight = per[:, 1, :inner].t()
            g.v_proj.weight = per[:, 2, :inner].t()
            g.out_proj.weight = wo[gi][:inner, :].t()
            if bqkv is not None:
                per_b = bqkv[gi].view(3, e8)
                g.q_proj.bias = per_b[0, :inner]
                g.k_proj.bias = per_b[1, :inner]
                g.v_proj.bias = per_b[2, :inner]
            if bo is not None:
                g.out_proj.bias = bo[gi]

    def _apply(self, fn, recurse=True):  # noqa: ANN001, ANN202
        """Re-derive the group weight views after device/dtype moves."""

        module = super()._apply(fn, recurse)
        if getattr(self, "_stacked_attn", False) and hasattr(
            self, "attn_wqkv_stacked"
        ):
            self._install_group_weight_views()
        return module

    @_maybe_disable_compile
    def _attn_update_packed_varlen(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        column_plan: _ColumnStreamPlan | None = None,
        plan: _PackedAttentionPlan | None = None,
    ) -> (
        Float[Tensor, "batch_size seq_len dim"]
        | Float[Tensor, "batch_size seq_len slots dim"]
    ):
        """Fully packed token-dispatch attention: no capacity, no padding.

        Every other dispatch path pads: the per-group loop and the BMM path
        compact each group to ``cap = max`` members over the batch, so one
        saturated (group, image) pair forces EVERY pair to that width -- on the
        trained 500m router that is ~4.7x the attention and (whenever cap is
        large) the projections the routed tokens require. The per-group varlen
        path (:meth:`GroupRoutedAttention.forward_gathered_varlen`) removed the
        attention padding but pinned its packed buffer to the worst case
        ``B*S`` PER GROUP, so its projections run ``G / k`` times the routed
        rows (measured net negative on the ramped arms, 2.7x over-work on
        k=3-of-8 blocks).

        Here the packing is at the LAYER level, where there is no worst case:
        top-k routing sends every token to at most ``k`` groups, so ALL groups'
        routed tokens together are at most ``B*S*k`` -- a static shape equal
        to the real work. Layout (group-major, image-major, original token
        order; see :class:`_PackedAttentionPlan`):

        * projections: per-group q/k/v and out weights stacked once per forward
          (parameter owners unchanged, exactly like the BMM path) and applied
          with ONE grouped GEMM over the packed rows (:func:`segment_mm`, the
          same primitive as the MoE expert dispatch);
        * attention: :func:`varlen_attn` over per-(group, image) segments;
        * every row is a real routed token or an alignment-tail row: zero
          input, its own segment (finite, never NaN), zero gate, never read by
          the combine -- so the ``valid`` masks and ``where``-firewalls of the
          padded paths have nothing to do, and no tail gradient can reach a
          weight (see :attr:`_PackedAttentionPlan.row_real`).

        The pack indices are built ONCE PER BLOCK
        (:func:`_build_packed_attention_plan`, handed in as ``plan`` by
        :meth:`NestorBlock.forward`; built here only for callers without one):
        group membership is block-constant, so per-layer index building was
        pure repetition. Both data movements are single coalesced gathers with
        gather-only backwards (:class:`_PackedRowGather`,
        :class:`_PackedSlotGather`) -- no ``[G, B, S, dim]`` expanded copy of
        the input, no ``index_add_``. And because the packed rows ARE
        ``(token, slot)`` pairs, the unpack yields the column streams'
        ``[batch, seq, k, dim]`` slot layout directly when ``column_plan`` is
        given: the per-group loop's ``[batch, seq, G, dim]`` stack and its
        scatter into slots never exist.

        Dispatch safety: this path is selected by STATIC configuration only
        (``NESTOR_ATTN_PACKED_VARLEN`` + ``self.causal`` + uniform groups),
        never by runtime-verified state, so an activation-checkpoint recompute
        takes the same branch as the forward it replays. Causality holds by
        construction (the model is declared causal and every segment keeps
        original token order).

        Runs outside ``torch.compile`` (``_maybe_disable_compile``, like the
        rest of the dispatch-op family): under Inductor the layouts it picks
        for the tensors feeding ``_grouped_mm`` can violate the kernel's
        16-byte stride requirement, while eager controls its own layouts.
        ``NESTOR_BATCHED_ATTN_COMPILE=1`` opts back in. Independently,
        ``NESTOR_PACKED_POINTWISE_COMPILE=1`` compiles just the pointwise
        chains BETWEEN the GEMMs (bias add / RoPE / padding / scale / gate;
        see :mod:`nestor.backbones.packed_pointwise`) while the GEMMs and the
        varlen kernel stay eager -- both settings run the same helper
        functions, so the math is identical either way.

        Assumes ``base_attn_mask`` semantics are pure causality (no
        padding-token masking) -- the DCLM regime, same assumption as
        ``NESTOR_ATTN_CAUSAL_SDPA``. Requires structurally uniform groups
        (:func:`_attention_groups_batchable`).

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                input sequence.
            group_router_output (RouterOutput): Group routing for this layer.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions for RoPE.
            column_plan (_ColumnStreamPlan | None): Column-stream plan; when
                given the update is returned per slot.
            plan (_PackedAttentionPlan | None): The block's packing plan;
                None builds one for this call.

        Returns:
            Float[Tensor, "batch_size seq_len dim"] | Float[Tensor,
            "batch_size seq_len slots dim"]: Gate-weighted sum of every
            group's attention output (zero where no group is active), or --
            under ``column_plan`` -- each slot's gated column output.
        """

        from torch.nn.attention.varlen import varlen_attn

        from nestor.backbones.grouped_mm import segment_mm

        groups = list(self.attention_groups)
        num_groups = len(groups)
        batch_size, seq_len, dim = norm_x.shape
        num_tokens = batch_size * seq_len
        num_heads, head_dim = groups[0].num_heads, groups[0].head_dim
        inner_dim = num_heads * head_dim
        if plan is None:
            plan = _build_packed_attention_plan(
                group_router_output, pos, groups[0].rope, num_groups
            )
        if column_plan is not None:
            assert column_plan.num_slots == plan.num_slots, (
                "packed plan / column plan slot mismatch: "
                f"{plan.num_slots} vs {column_plan.num_slots}"
            )
        if self._triton_packed:
            return self._attn_update_packed_triton(
                norm_x, plan, column_plan, groups, num_heads, head_dim
            )
        compute_dtype = torch.bfloat16

        # --- Pack: ONE gather of the routed rows (k per token, no G-wide
        # intermediate), in the GEMM dtype. ---
        x_packed = _PackedRowGather.apply(
            norm_x.reshape(num_tokens, dim).to(compute_dtype),
            plan.tok_packed,
            plan.row_real,
            plan.slot_rows,
            plan.num_slots,
        )  # [rows, dim]

        # --- Grouped GEMM 1: q/k/v of every group over the packed rows. ---
        # Stack directly in [G, dim, 3e] layout: torch.stack COPIES, so the
        # result is contiguous. A transpose(1, 2) view instead has a 2-byte
        # middle stride at production widths, which the grouped-GEMM kernel
        # rejects ("strides should be multiple of 16 bytes").
        #
        # The kernel also needs every operand stride to be a 16-BYTE multiple,
        # i.e. widths % 8 == 0 in bf16 -- and the ramped arms use ODD heads
        # per group (3/5/7 x head_dim 60 -> widths like 180). Zero-pad each
        # q/k/v section to e8 = ceil8(e): zero weight columns produce zero
        # outputs that the slicing below discards, so this is exact.
        e8 = (inner_dim + 7) // 8 * 8
        pad_e = e8 - inner_dim
        if self._stacked_attn:
            # Stacked ownership: the parameters ARE the packed layout; one
            # cast replaces the whole stack/pad/copy chain.
            wqkv = self.attn_wqkv_stacked.to(compute_dtype)
        else:
            wqkv = torch.stack(
                [
                    F.pad(
                        torch.cat(
                            [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight],
                            dim=0,
                        ).t().reshape(dim, 3, inner_dim),
                        (0, pad_e),
                    ).reshape(dim, 3 * e8)
                    for g in groups
                ]
            ).to(compute_dtype)  # [G, dim, 3*e8], contiguous
        qkv = segment_mm(x_packed, wqkv, plan.offs)  # [rows, 3*e8]
        bqkv: Tensor | None = None
        if groups[0].q_proj.bias is not None:
            # fp32 stack: the gather's backward then index-accumulates the
            # bias gradient in fp32 (a bf16 scatter-sum over thousands of rows
            # loses enough precision to fail equivalence against the per-group
            # Linear backward).
            if self._stacked_attn:
                bqkv = self.attn_bqkv_stacked.float()  # [G, 3*e8]
            else:
                bqkv = torch.stack(
                    [
                        F.pad(
                            torch.cat(
                                [g.q_proj.bias, g.k_proj.bias, g.v_proj.bias]
                            ).reshape(3, inner_dim),
                            (0, pad_e),
                        ).reshape(3 * e8)
                        for g in groups
                    ]
                ).float()  # [G, 3*e8]

        # Bias add, e8 un-pad, RoPE with the plan's block-constant fp32
        # tables (RoPE.forward semantics: rotate in fp32, cast back), head
        # pad to a multiple of 8 for flash varlen (padded q/k dims add 0 to
        # every logit; padded v columns are sliced away) and the muP scale
        # correction folded into q (the kernel has no scale argument and
        # applies 1/sqrt(head_size as it sees it); the layer's contract is
        # 1/head_dim). One pointwise chain, fused under
        # NESTOR_PACKED_POINTWISE_COMPILE; both forms run the same ops.
        pad = (-head_dim) % 8
        rope_fn = rope_qkv_compiled if self._packed_pointwise else rope_qkv
        q, k, v = rope_fn(
            qkv,
            bqkv,
            plan.gid if bqkv is not None else None,
            plan.cos,
            plan.sin,
            inner_dim,
            num_heads,
            head_dim,
            pad,
        )

        out = varlen_attn(
            q, k, v, plan.cu, plan.cu, seq_len, seq_len, is_causal=True
        )
        # Strip the head padding and re-pad the flat rows to e8 for the out
        # GEMM (zero attention columns meet zero weight rows -- exact).
        rows_fn = (
            attn_out_rows_compiled if self._packed_pointwise else attn_out_rows
        )
        out_flat = rows_fn(out, inner_dim, head_dim, pad, pad_e)

        # --- Grouped GEMM 2: every group's output projection. ---
        # Same width-8 rule for the output GEMM: pad the reduction dim with
        # zero attention columns / zero weight rows -- exact by construction.
        if self._stacked_attn:
            wo = self.attn_wo_stacked.to(compute_dtype)  # [G, e8, dim]
        else:
            wo = torch.stack(
                [
                    F.pad(g.out_proj.weight.t(), (0, 0, 0, pad_e))
                    for g in groups
                ]
            ).to(compute_dtype)  # [G, e8, dim], contiguous
        out_c = segment_mm(out_flat, wo, plan.offs)  # [rows, dim]
        bo: Tensor | None = None
        if groups[0].out_proj.bias is not None:
            bo = (
                self.attn_bo_stacked.float()
                if self._stacked_attn
                else torch.stack([g.out_proj.bias for g in groups]).float()
            )

        # Output bias, then gate (tail rows have exactly-zero gates -- the
        # invariant both gather Functions rely on), then read every
        # (token, slot) pair's row: that IS the slot layout, and summed over
        # slots it is the gate-weighted group sum the other paths return.
        combine_fn = (
            gate_combine_compiled if self._packed_pointwise else gate_combine
        )
        gated = combine_fn(
            out_c,
            bo,
            plan.gid if bo is not None else None,
            plan.gates_packed,
            norm_x.dtype,
        )
        slots = _PackedSlotGather.apply(
            gated, plan.slot_rows, plan.slot_packed
        )  # [B*S*k, dim]
        if column_plan is not None:
            return slots.view(batch_size, seq_len, plan.num_slots, dim)
        return (
            slots.view(num_tokens, plan.num_slots, dim)
            .sum(1)
            .view(batch_size, seq_len, dim)
        )

    def _attn_update_packed_triton(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        plan: _PackedAttentionPlan,
        column_plan: _ColumnStreamPlan | None,
        groups: list[nn.Module],
        num_heads: int,
        head_dim: int,
    ) -> (
        Float[Tensor, "batch_size seq_len dim"]
        | Float[Tensor, "batch_size seq_len slots dim"]
    ):
        """The packed path with fused Triton kernels between the GEMMs.

        Same math as the tail of :meth:`_attn_update_packed_varlen` (the
        caller, which handles path selection and plan building), with every
        pointwise/gather op between the two grouped GEMMs and the varlen
        kernel replaced by the three fused forward/backward kernel pairs of
        :mod:`nestor.backbones.triton_packed`, and the weight stacks laid out
        HEAD-PADDED (``hdp = ceil8(head_dim)`` lanes per head, zero pads) so
        the qkv GEMM writes exactly what ``varlen_attn`` consumes and the
        varlen output feeds the out GEMM as a view — no ``attn_out_rows``
        copy, no separate ``e8`` padding. Exact against the plain packed
        path to bf16 reassociation noise (``NESTOR_TRITON_PACKED_ATTN``;
        equivalence suite ``tests/test_triton_packed_attn.py``).

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                input sequence.
            plan (_PackedAttentionPlan): The block's packing plan.
            column_plan (_ColumnStreamPlan | None): When given, the update is
                returned per slot.
            groups (list[nn.Module]): The layer's attention groups.
            num_heads (int): Heads per group.
            head_dim (int): Per-head width.

        Returns:
            Float[Tensor, "batch_size seq_len dim"] | Float[Tensor,
            "batch_size seq_len slots dim"]: Gate-weighted group sum, or —
            under ``column_plan`` — each slot's gated column output.
        """

        from torch.nn.attention.varlen import varlen_attn

        from nestor.backbones import triton_packed as tp
        from nestor.backbones.grouped_mm import segment_mm

        batch_size, seq_len, dim = norm_x.shape
        num_tokens = batch_size * seq_len
        hdp = head_dim + ((-head_dim) % 8)
        compute_dtype = torch.bfloat16

        # Pack: ONE fused gather+cast+tail-zero kernel.
        x_packed = tp.pack_rows(
            norm_x.reshape(num_tokens, dim),
            plan.tok_packed,
            plan.row_real,
            plan.slot_rows,
            plan.num_slots,
            compute_dtype,
        )  # [rows, dim]

        # Grouped GEMM 1 over the head-padded stack: the output IS the
        # [rows, 3, H, hdp] layout the varlen kernel wants (pad lanes zero).
        wqkv = tp.stack_wqkv_headpad(
            groups, dim, num_heads, head_dim, hdp, compute_dtype,
            batched=self._batched_attn_stacks,
        )  # [G, dim, 3*H*hdp]
        qkv = segment_mm(x_packed, wqkv, plan.offs)  # [rows, 3*H*hdp]
        bqkv = (
            tp.stack_bqkv_headpad(groups, num_heads, head_dim, hdp)
            if groups[0].q_proj.bias is not None
            else None
        )

        # Bias + fp32 RoPE + muP scale, ONE kernel (pad lanes stay zero).
        scale = float(hdp) ** 0.5 / head_dim
        q, k, v = tp.rope_qkv_headpad(
            qkv,
            bqkv,
            plan.gid if bqkv is not None else None,
            plan.cos,
            plan.sin,
            num_heads,
            head_dim,
            hdp,
            scale,
        )
        out = varlen_attn(
            q, k, v, plan.cu, plan.cu, seq_len, seq_len, is_causal=True
        )  # [rows, H, hdp]; pad lanes exactly zero (zero v pad lanes)

        # Grouped GEMM 2: the varlen output feeds it as a VIEW; the stack's
        # zero pad rows meet the output's zero pad lanes — exact.
        wo = tp.stack_wo_headpad(
            groups, dim, num_heads, head_dim, hdp, compute_dtype,
            batched=self._batched_attn_stacks,
        )  # [G, H*hdp, dim]
        out_c = segment_mm(
            out.reshape(plan.rows, num_heads * hdp), wo, plan.offs
        )  # [rows, dim]
        bo = (
            torch.stack([g.out_proj.bias for g in groups]).float()
            if groups[0].out_proj.bias is not None
            else None
        )

        # Output bias + upcast + gate + slot gather, ONE kernel (the gated
        # [rows, dim] intermediate of the unfused chain never exists).
        slots = tp.gate_combine_slots(
            out_c,
            bo,
            plan.gid if bo is not None else None,
            plan.gates_packed,
            plan.slot_rows,
            plan.slot_packed,
            norm_x.dtype,
        )  # [B*S*k, dim]
        if column_plan is not None:
            return slots.view(batch_size, seq_len, plan.num_slots, dim)
        return (
            slots.view(num_tokens, plan.num_slots, dim)
            .sum(1)
            .view(batch_size, seq_len, dim)
        )

    def _gathered_proj_bmm_attention(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        column_plan: _ColumnStreamPlan | None = None,
    ) -> (
        Float[Tensor, "batch_size seq_len dim"]
        | Float[Tensor, "batch_size seq_len slots dim"]
    ):
        """Token-dispatch attention with the per-group projections batched.

        Numerically equivalent to the per-group ``forward_gathered`` /
        ``forward_gathered_full`` loop in :meth:`_attn_update_general` (modulo
        GEMM accumulation order), but every group is compacted to ONE shared
        capacity ``cap = max_(image, group) members``, which makes the four
        projections perfectly batchable:

        * ``[G, B*cap, dim] x [G, dim, 3 * heads * head_dim]`` -- one ``bmm``
          for q, k and v of every group (the per-group path issues ``3G``
          GEMMs whose output width is ``heads_per_group * head_dim``, e.g. 120
          at the 200m g8t3 shape -- far below the ~128 tensor cores need).
        * ``[G, B*cap, heads * head_dim] x [G, heads * head_dim, dim]`` -- one
          ``bmm`` for every group's output projection.

        The per-group :class:`ScaledLinear` modules remain the parameter
        owners; the batched weights are ``torch.stack``/``torch.cat`` views
        built per forward, so autograd routes gradients back to each group's
        weight and muP tagging, ``reference_mup``, DDP bucketing and
        checkpoints are all unchanged.

        Only reached when the groups are structurally uniform -- see
        :func:`_attention_groups_batchable` and
        :meth:`refresh_grouped_proj_bmm`, which fall back to the per-group loop
        otherwise -- so every group's ``dim``/``num_heads``/``head_dim`` and
        bias-ness may be read off group 0 here.

        Padding to a shared ``cap`` costs nothing semantically: extra slots are
        masked out of the attention core and ``where``-zeroed after the output
        projection, exactly as the per-group path treats its own padding. A
        group that is empty on every image contributes only zeros (and gets
        exactly-zero gradients), matching the per-group empty-group probe;
        because all G groups always take part in the two ``bmm`` calls, every
        projection stays in the autograd graph on every rank without a probe.

        Attention core selection mirrors the per-group path: the
        ``NESTOR_DENSE_FULL_ATTN`` no-padding contract uses a key-validity mask
        only, a verified compact-causal layer
        (``NESTOR_GATHERED_CAUSAL_FLASH``) takes flash ``is_causal``, and
        everything else uses the gathered ``[G*B, cap, cap]`` pair mask.
        Runs outside torch.compile by default (same torch-2.10 dynamo stall
        family as ``forward_gathered_full``; the capacity ``.item()`` sync
        graph-breaks anyway) -- ``NESTOR_BATCHED_ATTN_COMPILE=1`` opts in.

        Args:
            norm_x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized
                sublayer input.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Mask
                shared across groups before membership masking.
            group_router_output (RouterOutput): Block-level group router
                output (membership and gates).
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions.
            column_plan (_ColumnStreamPlan | None): Column-wise residual-stream
                plan. When given, the final scatter routes each compact row to
                its ``(token, slot)`` cell instead of summing groups onto the
                token -- the attention math above is untouched, only the
                ``index_add`` target widens by the slot axis.

        Returns:
            Float[Tensor, "batch_size seq_len dim"] | Float[Tensor,
            "batch_size seq_len slots dim"]: Gate-weighted sum of the groups'
            attention updates, or their per-slot scatter under a column plan.
        """

        groups = list(self.attention_groups)
        num_groups = len(groups)
        batch_size, seq_len, dim = norm_x.shape
        num_heads = groups[0].num_heads
        head_dim = groups[0].head_dim
        inner_dim = num_heads * head_dim
        device = norm_x.device

        # Group-major membership / gates: [G, B, S].
        members = group_router_output.active_mask.permute(2, 0, 1)
        gates = group_router_output.gates.permute(2, 0, 1)
        counts = members.sum(dim=-1)  # [G, B] members per (group, image)
        # ONE host sync per layer for ALL groups (the per-group loop pays one
        # per group). >=1 keeps every shape non-empty when no group has members.
        max_count = int(counts.max().item())
        cap = _bucket_capacity(max(max_count, 1), seq_len)

        # Stable members-first order keeps original token order (causal-safe).
        order = torch.argsort(
            members.to(torch.int8), dim=-1, descending=True, stable=True
        )
        gather_idx = order[..., :cap]  # [G, B, cap]
        valid = (
            torch.arange(cap, device=device)[None, None, :]
            < counts.unsqueeze(-1)
        )  # [G, B, cap] real (non-pad) compact slots

        x_g = torch.gather(
            norm_x.unsqueeze(0).expand(num_groups, batch_size, seq_len, dim),
            2,
            gather_idx.unsqueeze(-1).expand(num_groups, batch_size, cap, dim),
        ).reshape(num_groups, batch_size * cap, dim)
        pos_c = torch.gather(
            pos.unsqueeze(0).expand(num_groups, batch_size, seq_len),
            2,
            gather_idx,
        )  # [G, B, cap]

        # --- Grouped GEMM 1: q/k/v of every group in one bmm. ---
        wqkv = torch.stack(
            [
                torch.cat(
                    [g.q_proj.weight, g.k_proj.weight, g.v_proj.weight], dim=0
                )
                for g in groups
            ]
        )  # [G, 3e, dim]
        qkv = torch.bmm(x_g, wqkv.transpose(1, 2))  # [G, B*cap, 3e]
        if groups[0].q_proj.bias is not None:
            bqkv = torch.stack(
                [
                    torch.cat([g.q_proj.bias, g.k_proj.bias, g.v_proj.bias])
                    for g in groups
                ]
            )  # [G, 3e]
            qkv = qkv + bqkv.to(qkv.dtype).unsqueeze(1)
        qkv = qkv.view(
            num_groups, batch_size, cap, 3, num_heads, head_dim
        ).permute(3, 0, 1, 4, 2, 5)  # [3, G, B, H, cap, head_dim]
        qkv = qkv.reshape(
            3, num_groups * batch_size, num_heads, cap, head_dim
        )

        # RoPE at the gathered positions, mirroring RoPE.forward exactly
        # (fp32 tables, rotate in fp32, cast back). All groups in a layer share
        # head_dim and rope_base, so one table serves them all.
        inv_freq = groups[0].rope.inv_freq
        angles = pos_c.reshape(num_groups * batch_size, cap).clamp_min(0).to(
            torch.float32
        ).unsqueeze(-1) * inv_freq
        emb = torch.cat((angles, angles), dim=-1)  # [G*B, cap, head_dim]
        cos = emb.cos().unsqueeze(1)  # [G*B, 1, cap, head_dim]
        sin = emb.sin().unsqueeze(1)
        q32 = qkv[0].float()
        k32 = qkv[1].float()
        q = (q32 * cos + _rotate_half(q32) * sin).to(qkv.dtype)
        k = (k32 * cos + _rotate_half(k32) * sin).to(qkv.dtype)
        v = qkv[2]

        valid_rows = valid.reshape(num_groups * batch_size, cap)
        compact_mask: Tensor | None = None
        if base_attn_mask is None and not (
            self.causal or self._grouped_proj_bmm_causal_ok
        ):
            raise ValueError(
                "_gathered_proj_bmm_attention requires base_attn_mask on "
                "non-causal layers unless compact causality is verified."
            )
        if not self._dispatch_full_attn:
            # Trivial-mask contract (base None + causal): compact causality
            # holds by construction (see forward_gathered), so no pair mask is
            # ever gathered and the is_causal branch below runs.
            if (
                base_attn_mask is not None
                and self._grouped_proj_bmm_causal_ok is not True
            ):
                compact_mask = self._gathered_compact_pair_mask(
                    base_attn_mask, gather_idx, valid
                )
            if (
                compact_mask is not None
                and self._grouped_proj_bmm_causal_flash
                and self._grouped_proj_bmm_causal_ok is None
                and max_count > 0
            ):
                # One-time verification for the whole layer: the compact pair
                # mask must be exactly compact causality restricted to the real
                # slots (see GroupRoutedAttention.forward_gathered). A mismatch
                # is a permanent fallback; the verdict is mirrored onto the
                # group modules so both paths agree if the caller switches.
                assert compact_mask is not None
                causal_ref = (
                    torch.tril(
                        torch.ones(cap, cap, dtype=torch.bool, device=device)
                    ).unsqueeze(0)
                    & valid_rows[:, :, None]
                    & valid_rows[:, None, :]
                )
                verdict = bool(torch.equal(compact_mask, causal_ref))
                self._grouped_proj_bmm_causal_ok = verdict
                for group_attn in groups:
                    group_attn._gathered_causal_flash_ok = verdict
                if verdict:
                    compact_mask = None
            if compact_mask is not None:
                # Let a padded QUERY row attend to its own (padded) slot. Those
                # rows are where-zeroed after the output projection, so this
                # cannot change any member output -- but it keeps a fully
                # masked row (a group with no members on some image) out of the
                # softmax, whose NaN would otherwise reach the grouped output
                # GEMM's weight gradient (all G groups share that GEMM here,
                # whereas the per-group loop isolates each group's out_proj).
                compact_mask = compact_mask | (
                    ~valid_rows[:, :, None]
                    & torch.eye(cap, dtype=torch.bool, device=device)
                )

        if self._dispatch_full_attn:
            # No-padding contract: the pair mask reduces to valid (x) valid, so
            # masking the padded KEY slots suffices (padded query rows are
            # discarded below), matching forward_gathered_full.
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=valid_rows[:, None, None, :],
                dropout_p=0.0,
                scale=1.0 / head_dim,
            )
            # A group with no members on an image has an all-masked KEY set, so
            # its rows come back NaN on backends that softmax an empty row.
            # ``where`` below discards them from the output, but they would
            # still poison the SHARED output GEMM's weight gradient
            # (``grad_W = out^T @ grad_out_c`` and ``0 * NaN = NaN``), which
            # the per-group loop confines to that one group's out_proj. The
            # masked branch gets the same protection from its eye term above,
            # and the causal branch cannot produce an empty row at all.
            out = torch.where(
                valid_rows[:, None, :, None], out, out.new_zeros(())
            )
        elif compact_mask is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=0.0,
                scale=1.0 / head_dim,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=compact_mask.unsqueeze(1),
                dropout_p=0.0,
                scale=1.0 / head_dim,
            )

        # --- Grouped GEMM 2: every group's output projection in one bmm. ---
        out = out.permute(0, 2, 1, 3).reshape(
            num_groups, batch_size * cap, inner_dim
        )
        wo = torch.stack([g.out_proj.weight for g in groups])  # [G, dim, e]
        out_c = torch.bmm(out, wo.transpose(1, 2))  # [G, B*cap, dim]
        if groups[0].out_proj.bias is not None:
            bo = torch.stack([g.out_proj.bias for g in groups])  # [G, dim]
            out_c = out_c + bo.to(out_c.dtype).unsqueeze(1)
        out_c = out_c.view(num_groups, batch_size, cap, dim)
        # Zero padded slots (where() discards any NaN from fully-masked rows).
        out_c = torch.where(valid.unsqueeze(-1), out_c, out_c.new_zeros(()))
        # Upcast then gate, matching the loop's ``y (low prec) * gate
        # (accumulator prec)`` ordering. Non-member positions are already zero,
        # so gathering the gates is the same as gating the scattered output.
        gates_c = torch.gather(gates, 2, gather_idx)  # [G, B, cap]
        out_c = out_c.to(norm_x.dtype) * gates_c.unsqueeze(-1).to(norm_x.dtype)

        # Scatter every group's compact rows back onto their tokens and sum
        # (a token may belong to several groups). Padded slots point at
        # arbitrary non-members but carry exactly zero, so they are inert.
        token_idx = (
            torch.arange(batch_size, device=device).view(1, -1, 1) * seq_len
            + gather_idx
        )  # [G, B, cap] flat token index b*S + t
        if column_plan is None:
            attn_update = torch.zeros(
                batch_size * seq_len, dim, dtype=out_c.dtype, device=device
            ).index_add(0, token_idx.reshape(-1), out_c.reshape(-1, dim))
            return attn_update.view(batch_size, seq_len, dim)

        # Column mode: widen the scatter target by the slot axis. A compact row
        # of group g belonging to token t lands in cell (t, slot_of[t, g]);
        # padded rows are non-members, so they hit the sentinel slot AND carry
        # exactly zero -- inert twice over.
        num_cells = column_plan.num_slots + 1
        slot_c = torch.gather(
            column_plan.slot_of.permute(2, 0, 1), 2, gather_idx
        )  # [G, B, cap]
        flat_idx = (token_idx * num_cells + slot_c).reshape(-1)
        cells = torch.zeros(
            batch_size * seq_len * num_cells,
            dim,
            dtype=out_c.dtype,
            device=device,
        ).index_add(0, flat_idx, out_c.reshape(-1, dim))
        return cells.view(batch_size, seq_len, num_cells, dim)[
            :, :, : column_plan.num_slots
        ]

    def _expert_update_general(
        self,
        norm_x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        bank_noise: Tensor | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], RouterOutput]:
        """Expert-routed MLP update from a pre-normalized input.

        Shared by the convex residual and Attention-Residuals paths.
        """

        if self._expert_noroute:
            # Single always-on shared expert (gate exactly 1.0): dense MLP,
            # no router, no dispatch. Same contract as the dense fast path.
            expert_router_output = _constant_router_output(
                norm_x, valid_token_mask, self.num_experts
            )
            expert_update = self.shared_experts[0](norm_x)
        else:
            expert_router_output = self.expert_router(
                norm_x, valid_token_mask, noise=bank_noise
            )
            if self._moe_grouped_mm and (
                self.num_attention_groups == 1
                or self.num_group_specific_experts == 0
                or group_router_output.topk_indices.shape[-1] == 1
            ):
                # Grouped GEMM needs each token to own exactly one slice of the
                # bank. That means top-1 group routing whenever the bank HAS
                # group-specific slices; a shared-only bank has none, so the
                # group gate never enters the expert update and any group top-k
                # is fine (the shared-bank column arms rely on this). Otherwise
                # k>1 group routing falls back to the sparse loop.
                expert_update = self._grouped_expert_residual(
                    norm_x, group_router_output, expert_router_output
                )
            else:
                expert_update = self._compute_expert_residual(
                    norm_x,
                    group_router_output,
                    expert_router_output,
                )
        # DeepSeekMoE-style always-on experts: unit gate, every token, on top
        # of the routed update. Dense GEMMs — fully compile/speed-stack
        # friendly.
        for dense_shared in self.dense_shared_experts:
            expert_update = expert_update + dense_shared(norm_x)
        return expert_update, expert_router_output

    def _grouped_expert_residual(
        self,
        x: Float[Tensor, "... dim"],
        group_router_output: RouterOutput,
        expert_router_output: RouterOutput,
    ) -> Float[Tensor, "... dim"]:
        """Static-shape top-k expert dispatch via grouped GEMM.

        Numerically equivalent to :meth:`_compute_expert_residual` (modulo
        GEMM accumulation order): every (token, slot) pair is dispatched —
        exactly ``N * top_k`` rows — so all shapes are static, there are no
        ``nonzero`` host syncs, and the whole MLP runs as two grouped GEMMs
        (see :func:`~nestor.backbones.grouped_mm.grouped_expert_rows`, which
        this method shares with the expert-pool MLP). This method only builds
        the bank index and combine weight of each row.

        The bank is ``[shared | group 0 specific | group 1 specific | ...]``.
        A token's shared choices (columns ``< num_shared``) map 1:1 onto the
        bank; its group-specific choices map onto its OWN group's slice and
        carry the group gate, exactly as the sparse loop applies
        ``specific_gates * group_gate`` per group. Requires top-1 group
        routing for ``num_attention_groups > 1`` **when the bank owns
        group-specific experts** (caller-enforced): the group gate of every
        other group is exactly 0, so each token only ever contributes rows to
        its own group's experts. A shared-only bank
        (``num_group_specific_experts == 0``) carries no group gate at all and
        is therefore valid at any group top-k.
        """

        experts = list(self.shared_experts)
        for group_experts in self.group_specific_experts:
            experts.extend(group_experts)
        num_shared = self.num_shared_experts
        num_specific = self.num_group_specific_experts
        top_k = self.expert_router.num_choices_per_token
        flat_x = x.reshape(-1, self.dim)
        num_tokens = flat_x.shape[0]

        gates = expert_router_output.gates.reshape(
            num_tokens, self.num_experts
        )
        topk_indices = expert_router_output.topk_indices.reshape(
            num_tokens, top_k
        )
        gathered_gates = gates.gather(1, topk_indices)
        if self._group_noroute or num_specific == 0:
            # Single fabricated group, or a shared-only bank: the bank index
            # IS the choice column (no specific slices to offset into) and the
            # combine weight is the expert gate alone. A shared-only bank never
            # multiplies in a group gate, so this branch is exact at any group
            # top-k -- which is what lets the shared-bank column arms keep the
            # grouped GEMM instead of falling back to the sparse loop.
            row_weights = gathered_gates
            expert_ids = topk_indices
        else:
            token_group = group_router_output.topk_indices.reshape(
                num_tokens, -1
            )[:, 0]
            is_specific = topk_indices >= num_shared
            expert_ids = torch.where(
                is_specific,
                topk_indices + (token_group * num_specific).unsqueeze(1),
                topk_indices,
            )
            group_gate = group_router_output.gates.reshape(
                num_tokens, -1
            ).gather(1, token_group.unsqueeze(1))
            row_weights = torch.where(
                is_specific, gathered_gates * group_gate, gathered_gates
            )

        # One grouped GEMM per projection over all N * top_k dispatched rows.
        # ``stacked_moe_bank`` (NESTOR_STACKED_EXPERT_BANKS) owns exactly this
        # ``experts`` list in the same order; None keeps the per-expert stacks.
        routed = grouped_expert_rows(
            flat_x,
            experts,
            expert_ids,
            fused_gate=self._moe_fused_gate,
            stacked=self.stacked_moe_bank,
            triton_index=self._triton_index_build,
        )
        routed = routed * row_weights.reshape(-1, 1)
        out = routed.view(num_tokens, top_k, -1).sum(1)
        return out.to(flat_x.dtype).reshape_as(x)

    def forward_attn_res(
        self,
        history: DepthHistory,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[DepthHistory, dict[str, Any]]:
        """Attention-Residuals layer: read a softmax-over-depth mix of history.

        Replaces the convex residual with Kimi's Attention Residuals
        (arXiv:2603.15031): each sublayer's input is a learned, content-dependent
        convex combination of all prior sublayer outputs (plus the block input
        ``history.values[0]``), instead of ``(1 - 1/D) h + (1/D) f(h)``. The raw
        sublayer outputs are appended to ``history`` (the depth values); the
        per-sublayer pseudo-query and key RMSNorm produce the depth weights.

        Structurally degenerate layers (1 group, 1 expert) take the same dense
        fast path as the convex residual (maskless flash attention under
        NESTOR_DENSE_FULL_ATTN, router skip under NESTOR_DENSE_NO_ROUTER,
        numerically identical); otherwise the general sparse path runs.

        Returns:
            tuple[DepthHistory, dict[str, Any]]: ``history`` extended (in place)
            by this layer's attention and expert outputs, and aux metadata.
        """

        if self._dense_fastpath and self._fastpath_check_remaining > 0:
            self._fastpath_check_remaining -= 1
            ref_history = DepthHistory()
            ref_history.values = list(history.values)
            ref_history.inv_rms = list(history.inv_rms)
            with torch.no_grad():
                ref_history, _ = self._forward_attn_res_impl(
                    ref_history,
                    base_attn_mask,
                    valid_token_mask,
                    group_router_output,
                    pos,
                    dense=False,
                    dispatch_plan=dispatch_plan,
                    router_noise=router_noise,
                )
            history, aux = self._forward_attn_res_impl(
                history,
                base_attn_mask,
                valid_token_mask,
                group_router_output,
                pos,
                dense=True,
                dispatch_plan=dispatch_plan,
                router_noise=router_noise,
            )
            diff = max(
                (fast.detach() - ref).abs().max().item()
                for fast, ref in zip(
                    history.values[-2:], ref_history.values[-2:]
                )
            )
            print(
                f"[ATTNRES-FASTPATH-CHECK] max|dense - general| = {diff:.3e}",
                flush=True,
            )
            return history, aux

        return self._forward_attn_res_impl(
            history,
            base_attn_mask,
            valid_token_mask,
            group_router_output,
            pos,
            dense=self._dense_fastpath,
            dispatch_plan=dispatch_plan,
            router_noise=router_noise,
        )

    def _forward_attn_res_impl(
        self,
        history: DepthHistory,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        *,
        dense: bool,
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[DepthHistory, dict[str, Any]]:
        """AttnRes layer body; ``dense`` picks the sublayer update path."""

        h = depth_attention(history, self.attn_query, self.attn_key_norm)
        norm_h = self.attn_norm(h)
        if dense:
            attn_update = self._attn_update_dense(
                norm_h, base_attn_mask, group_router_output, pos
            )
        else:
            attn_update = self._attn_update_general(
                norm_h,
                base_attn_mask,
                group_router_output,
                pos,
                dispatch_plan=dispatch_plan,
            )
        history.append(self._drop_path(attn_update))

        h = depth_attention(history, self.mlp_query, self.mlp_key_norm)
        norm_h = self.mlp_norm(h)
        expert_update, expert_aux = self._expert_update(
            norm_h, valid_token_mask, group_router_output, dense=dense,
            router_noise=router_noise,
        )
        history.append(self._drop_path(expert_update))
        return history, expert_aux

    def forward_kimi(
        self,
        kimi_blocks: list[Tensor],
        partial: Float[Tensor, "batch_size seq_len dim"] | None,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        depth_key_cache: DepthKeyCache | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
        """Kimi block AttnRes layer with functional partial-sum threading.

        Implements the within-block recurrence of the Kimi paper
        (arXiv:2603.15031 Eq. 3-6): each sublayer reads a depth-attention
        mixture of the completed block representations plus the running
        partial sum of this block's sublayer updates, and its (drop-path-
        wrapped) update is summed into that partial sum. All state is threaded
        through arguments and return values -- no module attributes are
        mutated mid-forward -- so torch.compile can trace the block loop as
        one graph (it specializes on the ``kimi_blocks`` length, a bounded
        number of variants).

        Structurally degenerate layers (1 group, 1 expert) use the same dense
        sublayer-update helpers as the convex residual path; otherwise the
        general sparse helpers run.

        Args:
            kimi_blocks (list[Tensor]): Completed block representations, each
                of shape ``[batch_size, seq_len, dim]``, in depth order (index
                0 is the embedding ``b_0``).
            partial (Float[Tensor, "batch_size seq_len dim"] | None): Running
                sum of the current block's earlier sublayer updates; None for
                the block's first layer.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Mask
                shared across attention groups before group membership masking.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Token
                validity mask derived from positions.
            group_router_output (RouterOutput): Block-level group router
                output.
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions.
            dispatch_plan (_AttentionDispatchPlan | None): Block-level token
                dispatch plan (general path only).
            depth_key_cache (DepthKeyCache | None): Backbone-level cache of the
                unit-RMS keys of ``kimi_blocks`` (fixed for this whole block, so
                both reads share them); must cover exactly ``kimi_blocks``.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len dim"], dict[str, Any]]:
                The partial sum extended by this layer's attention and expert
                updates, and aux metadata.
        """

        dense = self._dense_fastpath
        sources = kimi_blocks if partial is None else [*kimi_blocks, partial]
        h = self.kimi_attn_read(sources, key_cache=depth_key_cache)
        norm_h = self.attn_norm(h)
        if dense:
            attn_update = self._attn_update_dense(
                norm_h, base_attn_mask, group_router_output, pos
            )
        else:
            attn_update = self._attn_update_general(
                norm_h,
                base_attn_mask,
                group_router_output,
                pos,
                dispatch_plan=dispatch_plan,
            )
        attn_update = self._drop_path(attn_update)
        partial = attn_update if partial is None else partial + attn_update

        h = self.kimi_mlp_read(
            [*kimi_blocks, partial], key_cache=depth_key_cache
        )
        norm_h = self.mlp_norm(h)
        expert_update, expert_aux = self._expert_update(
            norm_h, valid_token_mask, group_router_output, dense=dense,
            router_noise=router_noise,
        )
        partial = partial + self._drop_path(expert_update)
        return partial, expert_aux

    def forward_column(
        self,
        block_sources: list[Tensor],
        streams: Float[Tensor, "batch_size seq_len slots dim"] | None,
        column_plan: _ColumnStreamPlan,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        depth_key_cache: DepthKeyCache | None = None,
        agg: Float[Tensor, "batch_size seq_len dim"] | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len slots dim"],
        Float[Tensor, "batch_size seq_len dim"] | None,
        dict[str, Any],
    ]:
        """Column-wise residual streams: one partial sum per routed column.

        The Nestor generalization of :meth:`forward_kimi`. A Nestor block runs
        its group router ONCE and every layer reuses that plan, so within a
        block each token sits in a FIXED set of ``k = num_groups_per_token``
        attention groups -- "columns" that behave like parallel sub-networks
        but that today are collapsed into one tensor by ``sum_c gate_c *
        Attn_c(h)`` at every sublayer. This mode keeps them apart: the token
        carries ``k`` partial sums (``streams``, slot ``j`` owning column
        ``topk_indices[..., j]``), each column's gated attention output updates
        only ITS slot, and all ``k`` are exported at the block boundary as
        separate depth-read sources, so later blocks can attend to an
        individual earlier column instead of to a pre-merged average.

        Per sublayer:

        - input ``h`` = ONE shared depth read over ``block_sources`` (the
          embedding plus every completed block's exported streams) plus this
          block's ``k`` partial streams -- as in :meth:`forward_kimi`, the
          streams are omitted on the block's first layer, where they are still
          all-zero (a zero source is not free: it would dilute the depth
          softmax). Under ``block_column_include_aggregate`` the streams'
          aggregate joins them as a ``k + 1``-th source, which is the merged
          direction :meth:`forward_kimi` would have read (see
          :func:`_column_read_sources`);
        - attention: the per-column outputs are computed by the SAME helpers as
          every other path (:meth:`_attn_update_general`, same kernels, same
          dispatch, same gates) and scattered into their slots via
          ``column_plan``;
        - MLP (:meth:`_expert_update_column`): column-agnostic feed-forwards
          (the plain dense MLP, ordinary MoE) are computed once on the shared
          read and broadcast to ALL ``k`` slots. With an ``expert_pool`` the
          update SPLITS: the pool's private experts are already per attention
          group, so each group's contribution lands in the slot holding that
          column, while the shared public pool broadcasts. No extra FLOPs --
          the per-group private tensors already exist inside the pool, only
          their accumulation target changes.

        Follow-up deliberately left out of this change: per-column depth-read
        queries (each column reading the depth axis with its own query rather
        than sharing one).

        With ``k == 1`` (a dense or top-1-routed block, e.g. g8t3's dense
        suffix) the slot axis is a singleton and this reduces exactly to
        :meth:`forward_kimi`.

        All state is threaded through arguments and return values -- no module
        attributes are mutated mid-forward -- so torch.compile can trace the
        block loop as one graph (it specializes on the source-list length, a
        bounded number of variants).

        Args:
            block_sources (list[Tensor]): Completed depth sources, each of
                shape ``[batch_size, seq_len, dim]``, in depth order (index 0
                is the embedding ``b_0``; every completed block contributed its
                ``k_b`` streams).
            streams (Float[Tensor, "batch_size seq_len slots dim"] | None):
                Running per-column partial sums of this block's earlier
                sublayer updates; None for the block's first layer.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Mask
                shared across attention groups before group membership masking.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Token
                validity mask derived from positions.
            group_router_output (RouterOutput): Block-level group router
                output.
            pos (Int[Tensor, "batch_size seq_len"]): Integer token positions.
            dispatch_plan (_AttentionDispatchPlan | None): Block-level token
                dispatch plan (general path only).
            depth_key_cache (DepthKeyCache | None): Backbone-level cache of the
                unit-RMS keys of ``block_sources``. Those sources are fixed for
                the whole block while the ``k`` stream keys change every
                sublayer, so this is where the column mode's read cost lives:
                without it each of the block's ``2 * num_layers`` reads
                re-normalizes (and retains, for the backward) every completed
                source. Must cover exactly ``block_sources``.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len slots dim"], dict[str,
            Any]]: The ``k`` partial streams extended by this layer's
            attention and expert updates, and aux metadata.
        """

        if self._column_slot_updates:
            return self._forward_column_slot_updates(
                block_sources,
                streams,
                column_plan,
                base_attn_mask,
                valid_token_mask,
                group_router_output,
                pos,
                dispatch_plan=dispatch_plan,
                depth_key_cache=depth_key_cache,
                agg=agg,
                router_noise=router_noise,
            )
        dense = self._dense_fastpath
        # Attention and the private expert pool scatter through the SAME
        # column -> slot map, so when both are available per group their sum
        # can be scattered ONCE instead of twice per sublayer. Only the general
        # (non-dense) path produces per-group attention, and only head-only
        # reads can consume the pre-scatter mean, so the fusion is gated on
        # both.
        # RELIES ON: a non-selected group's gate is exactly 0, so its
        # contribution is exactly 0. That is what makes both identities below
        # exact -- the same invariant _scatter_columns_to_slots already depends
        # on (its dropped sentinel bin "only ever absorbs zeros"). Without it
        # the pre-scatter mean would be wrong, so if gating ever becomes soft,
        # this fusion must go.
        fuse_scatter = (
            _COLUMN_FUSE_SCATTER
            and self._column_head_only
            and not dense
            and self.attention_token_dispatch
            and self.expert_pool is not None
        )
        sources = (
            block_sources
            if streams is None
            else _column_read_sources(
                block_sources,
                streams,
                self._column_aggregate,
                head_only=self._column_head_only,
                deviation_sources=self._column_deviation,
            )
        )
        h = self.kimi_attn_read(sources, key_cache=depth_key_cache)
        norm_h = self.attn_norm(h)
        if dense:
            # One group => one column => one slot (see _to_single_slot).
            attn_update = _to_single_slot(
                self._attn_update_dense(
                    norm_h, base_attn_mask, group_router_output, pos
                ),
                column_plan,
            )
        else:
            attn_update = self._attn_update_general(
                norm_h,
                base_attn_mask,
                group_router_output,
                pos,
                dispatch_plan=dispatch_plan,
                column_plan=column_plan,
                defer_column_scatter=fuse_scatter,
            )
        attn_update = self._drop_path(attn_update)

        if self._column_head_only:
            # Head-only reads consume the block's running AGGREGATE, never the
            # individual streams, so the post-attention state is needed only in
            # mean form here: ``mean(streams + attn_update) == mean(streams) +
            # mean(attn_update)``. Taking the two cheap [batch, seq, dim] means
            # instead skips materializing a whole [batch, seq, k, dim]
            # intermediate per sublayer -- one of the two full-width stream
            # tensors every layer used to allocate, write, and read back
            # (measured as the elementwise/copy_ block that is most of
            # head-only column mode's remaining gap to kimi). The streams
            # themselves are folded once, below.
            # mean over SLOTS of scatter(per-group) equals (1/k) * sum over
            # GROUPS: a non-selected group's gate is exactly zero and each
            # selected group lands in exactly one slot. So the deferred form
            # gets the same aggregate from a plain sum, with no scatter.
            attn_mean = (
                attn_update.sum(dim=2) / column_plan.num_slots
                if fuse_scatter
                else attn_update.mean(dim=2)
            )
            # The running aggregate is THREADED as [batch, seq, dim] rather
            # than re-reduced from the streams every sublayer: mean is linear,
            # so accumulating the per-sublayer means is the same quantity
            # without touching the [batch, seq, k, dim] state at all. The block
            # exit still takes a real ``streams.mean`` for the deviation base,
            # so the exported deviations sum to exactly zero rather than to
            # accumulated drift.
            running = (
                agg
                if agg is not None
                else (None if streams is None else streams.mean(dim=2))
            )
            mlp_sources = [
                *block_sources,
                attn_mean if running is None else running + attn_mean,
            ]
        else:
            streams = (
                attn_update if streams is None else streams + attn_update
            )
            mlp_sources = _column_read_sources(
                block_sources,
                streams,
                self._column_aggregate,
                head_only=False,
                deviation_sources=self._column_deviation,
            )

        h = self.kimi_mlp_read(mlp_sources, key_cache=depth_key_cache)
        norm_h = self.mlp_norm(h)
        if fuse_scatter:
            private_per_group, public_update, expert_aux = (
                self._expert_update_column(
                    norm_h,
                    valid_token_mask,
                    group_router_output,
                    column_plan,
                    dense=dense,
                    defer_scatter=True,
                    router_noise=router_noise,
                )
            )
            # ONE scatter for both sublayer updates. ``attn_update`` is
            # already drop-pathed above; the expert part is dropped here, so
            # each update keeps its own independent mask exactly as the
            # unfused path applies them.
            combined = attn_update
            if private_per_group is not None:
                combined = combined + self._drop_path(private_per_group)
            # Scatter DIRECTLY into the running streams: one scatter_add
            # replaces _scatter_columns_to_slots' zero-fill + scatter + slice
            # AND the separate accumulate, so the sublayer's slot bookkeeping
            # becomes a single full-width pass.
            #
            # The sentinel bin disappears the same way: a non-selected group's
            # slot index is ``num_slots``, and clamping it onto a real slot is
            # harmless because its contribution is exactly zero (the same gate
            # invariant the fused mean above relies on). So the streams stay
            # ``num_slots`` wide instead of carrying a sentinel column.
            if streams is None:
                streams = combined.new_zeros(
                    *combined.shape[:2], column_plan.num_slots, combined.shape[3]
                )
            index = column_plan.slot_of.clamp(
                max=column_plan.num_slots - 1
            ).unsqueeze(-1).expand(-1, -1, -1, combined.shape[3])
            streams = streams.scatter_add(2, index, combined)
            # ``combined`` already carries BOTH sublayer updates, so its
            # group-sum is the whole aggregate delta -- the attention mean
            # cancels rather than being added and subtracted back.
            new_agg = combined.sum(dim=2) / column_plan.num_slots
            if running is not None:
                new_agg = running + new_agg
            if public_update is not None:
                public = self._drop_path(public_update)
                streams = streams + public.unsqueeze(2)
                new_agg = new_agg + public
            return streams, new_agg, expert_aux
        expert_update, expert_aux = self._expert_update_column(
            norm_h,
            valid_token_mask,
            group_router_output,
            column_plan,
            dense=dense,
            router_noise=router_noise,
        )
        # Either [batch, seq, k, dim] (pooled: private per column) or
        # [batch, seq, 1, dim] (column-agnostic), which broadcasts over slots.
        expert_update = self._drop_path(expert_update)
        if self._column_head_only:
            # Fold BOTH sublayer updates in one pass now that the intermediate
            # was skipped above. Summing the two updates first keeps the single
            # full-width accumulate onto ``streams``; when the expert update is
            # column-agnostic ([batch, seq, 1, dim]) that sum broadcasts and
            # stays cheap.
            update = attn_update + expert_update
            streams = update if streams is None else streams + update
        else:
            streams = streams + expert_update
        # Unfused paths do not thread an aggregate; the caller falls back to
        # reducing the streams.
        return streams, None, expert_aux

    def _forward_column_slot_updates(
        self,
        block_sources: list[Tensor],
        streams: Float[Tensor, "batch_size seq_len slots dim"] | None,
        column_plan: _ColumnStreamPlan,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        depth_key_cache: DepthKeyCache | None = None,
        agg: Float[Tensor, "batch_size seq_len dim"] | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len slots dim"],
        Float[Tensor, "batch_size seq_len dim"] | None,
        dict[str, Any],
    ]:
        """:meth:`forward_column` with every update in SLOT layout.

        Same reads, same attention, same experts, same gates and the same
        streams as :meth:`forward_column`; only the bookkeeping between them
        changes. The per-group paths hand back ``[batch, seq, G, dim]``
        tensors (attention's stack of column outputs, the pool's per-group
        private update) that are ``G / k`` times wider than the streams they
        feed and mostly exactly zero, and then scatter them into the ``k``
        slots. Here both updates arrive in ``[batch, seq, k, dim]`` already:

        * attention: the packed path (:meth:`_attn_update_packed_varlen`) and
          the dense-tail path (:meth:`_attn_update_dense_tail`) index their
          outputs by ``(token, slot)`` natively; the per-group loop still
          scatters, once, as its unfused form always did;
        * experts: :meth:`~nestor.backbones.expert_pools.ExpertPoolMLP.
          forward_per_slot` -- the grouped dispatch's rows are slot-major
          before they are ever scattered onto groups;

        so the sublayer's whole slot bookkeeping is one ``k``-wide add. The
        threaded aggregate is the mean over slots of that same update (a
        non-selected column contributes exactly zero, the invariant every
        column path relies on), so head-only reads stay exact. Drop-path is
        applied per update (attention, private, public), as on the fused
        path. Enabled by ``NESTOR_COLUMN_SLOT_UPDATES=1``.

        Args:
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            streams (Float[Tensor, "batch_size seq_len slots dim"] | None):
                Running per-column partial sums; None on the block's first
                layer.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]):
                Shared attention mask before group membership.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real-token
                mask.
            group_router_output (RouterOutput): Block-level group routing.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            dispatch_plan (_AttentionDispatchPlan | _PackedAttentionPlan |
                None): Block-level attention plan, if any.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.
            agg (Float[Tensor, "batch_size seq_len dim"] | None): Threaded
                running aggregate (head-only reads).

        Returns:
            tuple[Float[Tensor, "batch_size seq_len slots dim"],
            Float[Tensor, "batch_size seq_len dim"] | None, dict[str, Any]]:
            The extended streams, the new running aggregate (head-only reads;
            None otherwise), and the layer aux.
        """

        attn_update = self.column_attn_half(
            block_sources=block_sources,
            streams=streams,
            column_plan=column_plan,
            base_attn_mask=base_attn_mask,
            group_router_output=group_router_output,
            pos=pos,
            dispatch_plan=dispatch_plan,
            depth_key_cache=depth_key_cache,
        )
        return self.column_expert_half(
            block_sources=block_sources,
            streams=streams,
            attn_update=attn_update,
            column_plan=column_plan,
            valid_token_mask=valid_token_mask,
            group_router_output=group_router_output,
            depth_key_cache=depth_key_cache,
            agg=agg,
            router_noise=router_noise,
        )

    def column_attn_half(
        self,
        *,
        block_sources: list[Tensor],
        streams: Float[Tensor, "batch_size seq_len slots dim"] | None,
        column_plan: _ColumnStreamPlan,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        depth_key_cache: DepthKeyCache | None = None,
    ) -> Float[Tensor, "batch_size seq_len k dim"]:
        """First half of :meth:`_forward_column_slot_updates`: the attention update.

        The depth read over the block sources (and the running streams), the
        attention norm, the column attention in slot layout and drop-path --
        everything up to and including ``attn_update``. Split out so
        ``NESTOR_ACTIVATION_CHECKPOINT_SPLIT=1`` can checkpoint it as its own
        region (:func:`_activation_checkpoint_split_enabled`); calling it and
        :meth:`column_expert_half` back to back is exactly the unsplit layer.
        Keyword-only, so the checkpoint wrapper can pass every argument
        positionally under its own names.

        Args:
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            streams (Float[Tensor, "batch_size seq_len slots dim"] | None):
                Running per-column partial sums; None on the block's first
                layer.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None):
                Attention mask.
            group_router_output (RouterOutput): Block-level group routing.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            dispatch_plan (_AttentionDispatchPlan | _PackedAttentionPlan |
                None): Block-level attention plan, if any.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.

        Returns:
            Float[Tensor, "batch_size seq_len k dim"]: The drop-pathed
            attention update in slot layout.
        """

        sources = (
            block_sources
            if streams is None
            else _column_read_sources(
                block_sources,
                streams,
                self._column_aggregate,
                head_only=self._column_head_only,
                deviation_sources=self._column_deviation,
            )
        )
        return self._column_attn_from_sources(
            sources,
            column_plan=column_plan,
            base_attn_mask=base_attn_mask,
            group_router_output=group_router_output,
            pos=pos,
            dispatch_plan=dispatch_plan,
            depth_key_cache=depth_key_cache,
        )

    def _column_attn_from_sources(
        self,
        sources: list[Tensor],
        *,
        column_plan: _ColumnStreamPlan,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None,
        depth_key_cache: DepthKeyCache | None,
    ) -> Float[Tensor, "batch_size seq_len k dim"]:
        """The attention half from an assembled source list (see :meth:`column_attn_half`).

        Args:
            sources (list[Tensor]): The depth read's sources, each
                ``[batch, seq, dim]``, in depth order.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None):
                Attention mask.
            group_router_output (RouterOutput): Block-level group routing.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            dispatch_plan (_AttentionDispatchPlan | _PackedAttentionPlan |
                None): Block-level attention plan, if any.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.

        Returns:
            Float[Tensor, "batch_size seq_len k dim"]: The drop-pathed
            attention update in slot layout.
        """

        h = self.kimi_attn_read(sources, key_cache=depth_key_cache)
        norm_h = self.attn_norm(h)
        if self._dense_fastpath:
            attn_update = _to_single_slot(
                self._attn_update_dense(
                    norm_h, base_attn_mask, group_router_output, pos
                ),
                column_plan,
            )
        else:
            attn_update = self._attn_update_general(
                norm_h,
                base_attn_mask,
                group_router_output,
                pos,
                dispatch_plan=dispatch_plan,
                column_plan=column_plan,
            )  # [batch, seq, k, dim]
        return self._drop_path(attn_update)

    def column_attn_half_lean(
        self,
        *,
        block_sources: list[Tensor],
        stream_mean: Float[Tensor, "batch_size seq_len dim"] | None,
        column_plan: _ColumnStreamPlan,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None = None,
        depth_key_cache: DepthKeyCache | None = None,
    ) -> Float[Tensor, "batch_size seq_len k dim"]:
        """:meth:`column_attn_half` for a head-only read, fed the streams' mean instead of the streams.

        The head-only read's sources are ``[*block_sources, streams.mean(dim=2)]``
        (:func:`_column_read_sources`); handing in that mean lets the recompute
        region save a ``[batch, seq, dim]`` input instead of the
        ``[batch, seq, k, dim]`` streams (:func:`_activation_checkpoint_lean_enabled`).
        Identical ops from the sources on.

        Args:
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            stream_mean (Float[Tensor, "batch_size seq_len dim"] | None): The
                entry streams' mean over slots; None on the block's first layer.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None):
                Attention mask.
            group_router_output (RouterOutput): Block-level group routing.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            dispatch_plan (_AttentionDispatchPlan | _PackedAttentionPlan |
                None): Block-level attention plan, if any.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.

        Returns:
            Float[Tensor, "batch_size seq_len k dim"]: The drop-pathed
            attention update in slot layout.
        """

        assert self._column_head_only, "the lean halves need a head-only column read"
        sources = block_sources if stream_mean is None else [*block_sources, stream_mean]
        return self._column_attn_from_sources(
            sources,
            column_plan=column_plan,
            base_attn_mask=base_attn_mask,
            group_router_output=group_router_output,
            pos=pos,
            dispatch_plan=dispatch_plan,
            depth_key_cache=depth_key_cache,
        )

    def column_expert_half_lean(
        self,
        *,
        block_sources: list[Tensor],
        running: Float[Tensor, "batch_size seq_len dim"] | None,
        attn_mean: Float[Tensor, "batch_size seq_len dim"],
        column_plan: _ColumnStreamPlan,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        depth_key_cache: DepthKeyCache | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len *k dim"], dict[str, Any]]:
        """The expert update of :meth:`column_expert_half` (head-only read), without the stream adds.

        Takes the running aggregate and the attention update's mean -- the
        only two things the head-only expert half reads -- and returns the
        expert update; the caller folds it and the attention update into the
        streams and the aggregate with the very expressions
        :meth:`column_expert_half` uses (:meth:`NestorBlock._column_layers_region`),
        so that a recompute region saves neither the streams nor the attention
        update (:func:`_activation_checkpoint_lean_enabled`).

        Args:
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            running (Float[Tensor, "batch_size seq_len dim"] | None): The
                running aggregate at layer entry (the threaded ``agg``, else
                the entry streams' mean); None on the block's first layer.
            attn_mean (Float[Tensor, "batch_size seq_len dim"]): The attention
                update's mean over slots.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real-token
                mask.
            group_router_output (RouterOutput): Block-level group routing.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.
            router_noise (dict[str, Tensor] | None): Pre-drawn router noise
                factors for the expert pool.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len *k dim"], dict[str, Any]]:
            The drop-pathed expert update in slot layout (``k`` slots, or one
            broadcast slot when only the public pool is present) and the
            layer aux.
        """

        assert self._column_head_only, "the lean halves need a head-only column read"
        mlp_sources = [
            *block_sources,
            attn_mean if running is None else running + attn_mean,
        ]
        h = self.kimi_mlp_read(mlp_sources, key_cache=depth_key_cache)
        norm_h = self.mlp_norm(h)
        private_slots, public_update, expert_aux = self._expert_update_column(
            norm_h,
            valid_token_mask,
            group_router_output,
            column_plan,
            dense=self._dense_fastpath,
            slot_layout=True,
            router_noise=router_noise,
        )
        expert_update: Tensor | None = (
            None if private_slots is None else self._drop_path(private_slots)
        )
        if public_update is not None:
            public = self._drop_path(public_update).unsqueeze(2)
            expert_update = (
                public if expert_update is None else expert_update + public
            )
        assert expert_update is not None
        return expert_update, expert_aux

    def column_tail_kv_lean(
        self,
        *,
        block_sources: list[Tensor],
        stream_mean: Float[Tensor, "batch_size seq_len dim"] | None,
        pos: Int[Tensor, "batch_size seq_len"],
        depth_key_cache: DepthKeyCache | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len dim"],
        Float[Tensor, "batch_size heads seq_len head_dim"],
        Float[Tensor, "batch_size heads seq_len head_dim"],
    ]:
        """The dense tail's attention half up to K / V (see :func:`_tail_query_chunks`).

        The head-only depth read and the attention norm over the whole
        sequence, then the K and V projections of every column stacked as
        ``G * H`` heads (RoPE on K), exactly as
        :meth:`_attn_update_dense_tail` computes them inside its fused q/k/v
        GEMM.

        Args:
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            stream_mean (Float[Tensor, "batch_size seq_len dim"] | None): The
                entry streams' mean over slots; None on the block's first layer.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.

        Returns:
            tuple[Tensor, Tensor, Tensor]: ``norm_h`` ``[batch, seq, dim]``
            (the normed read, the queries' input), ``k`` and ``v``
            ``[batch, G * H, seq, head_dim]`` in the projection dtype.
        """

        assert self._column_head_only and self._dense_tail_attn
        sources = block_sources if stream_mean is None else [*block_sources, stream_mean]
        h = self.kimi_attn_read(sources, key_cache=depth_key_cache)
        norm_h = self.attn_norm(h)
        groups = list(self.attention_groups)
        num_groups = len(groups)
        batch_size, seq_len, dim = norm_h.shape
        num_heads, head_dim = groups[0].num_heads, groups[0].head_dim
        inner_dim = num_heads * head_dim
        x = norm_h.reshape(batch_size * seq_len, dim)
        wkv = torch.stack(
            [torch.cat([g.k_proj.weight, g.v_proj.weight], dim=0) for g in groups]
        )  # [G, 2e, dim]
        kv = F.linear(x, wkv.reshape(num_groups * 2 * inner_dim, dim))
        if groups[0].k_proj.bias is not None:
            bkv = torch.cat(
                [torch.cat([g.k_proj.bias, g.v_proj.bias]) for g in groups]
            )  # [G*2e]
            kv = kv + bkv.to(kv.dtype)
        kv = kv.view(
            batch_size, seq_len, num_groups, 2, num_heads, head_dim
        ).permute(3, 0, 2, 4, 1, 5)  # [2, B, G, H, S, hd]
        kv = kv.reshape(2, batch_size, num_groups * num_heads, seq_len, head_dim)
        k = groups[0].rope(kv[0], pos)
        return norm_h, k, kv[1]

    def column_tail_attn_slice_lean(
        self,
        *,
        norm_h: Float[Tensor, "batch_size seq_len dim"],
        k: Float[Tensor, "batch_size heads seq_len head_dim"],
        v: Float[Tensor, "batch_size heads seq_len head_dim"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        lo: int,
        hi: int,
    ) -> Float[Tensor, "batch_size slice k dim"]:
        """One query slice of the dense tail's attention (see :func:`_tail_query_chunks`).

        Projects the queries of tokens ``lo:hi`` (RoPE at their positions),
        attends them causally against keys / values ``:hi`` (a query at
        ``lo + j`` sees keys ``<= lo + j``: the full attention's rows), then
        the batched output projection, the per-(token, column) gates and the
        slot gather of :meth:`_attn_update_dense_tail`, restricted to the
        slice. Drop-path is the caller's (applied once to the whole update).

        Args:
            norm_h (Float[Tensor, "batch_size seq_len dim"]): The normed read
                (:meth:`column_tail_kv_lean`).
            k (Float[Tensor, "batch_size heads seq_len head_dim"]): Roped keys
                of the whole sequence.
            v (Float[Tensor, "batch_size heads seq_len head_dim"]): Values of
                the whole sequence.
            group_router_output (RouterOutput): Group routing (gates and
                ``topk_indices`` are read; every column is active).
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            lo (int): First token of the slice.
            hi (int): One past the last token of the slice.

        Returns:
            Float[Tensor, "batch_size slice k dim"]: The gated attention update
            of the slice in slot layout (``slice = hi - lo``).
        """

        groups = list(self.attention_groups)
        num_groups = len(groups)
        batch_size, _, dim = norm_h.shape
        length = hi - lo
        rows = batch_size * length
        num_heads, head_dim = groups[0].num_heads, groups[0].head_dim
        inner_dim = num_heads * head_dim
        x = norm_h[:, lo:hi].reshape(rows, dim)
        wq = torch.stack([g.q_proj.weight for g in groups])  # [G, e, dim]
        q = F.linear(x, wq.reshape(num_groups * inner_dim, dim))
        if groups[0].q_proj.bias is not None:
            q = q + torch.cat([g.q_proj.bias for g in groups]).to(q.dtype)
        q = (
            q.view(batch_size, length, num_groups, num_heads, head_dim)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch_size, num_groups * num_heads, length, head_dim)
        )
        q = groups[0].rope(q, pos[:, lo:hi])
        # Query lo + j may attend key t <= lo + j, i.e. t - j <= lo: the
        # tril with diagonal lo of a [length, hi] mask.
        causal = torch.ones(length, hi, dtype=torch.bool, device=q.device).tril(diagonal=lo)
        out = F.scaled_dot_product_attention(
            q, k[:, :, :hi], v[:, :, :hi], attn_mask=causal, dropout_p=0.0,
            scale=1.0 / head_dim,
        )  # [B, G*H, length, hd]
        out = (
            out.reshape(batch_size, num_groups, num_heads, length, head_dim)
            .permute(1, 0, 3, 2, 4)
            .reshape(num_groups, rows, inner_dim)
        )
        wo = torch.stack([g.out_proj.weight for g in groups])  # [G, dim, e]
        out_c = torch.bmm(out, wo.transpose(1, 2).to(out.dtype))  # [G, rows, dim]
        if groups[0].out_proj.bias is not None:
            bo = torch.stack([g.out_proj.bias for g in groups])  # [G, dim]
            out_c = out_c + bo.to(out_c.dtype).unsqueeze(1)
        gates = group_router_output.gates[:, lo:hi].permute(2, 0, 1).reshape(
            num_groups, rows, 1
        )
        gated = out_c.to(norm_h.dtype) * gates.to(norm_h.dtype)
        per_token = gated.permute(1, 0, 2)  # [rows, G, dim]
        index = group_router_output.topk_indices[:, lo:hi].reshape(
            rows, -1, 1
        ).expand(-1, -1, dim)
        return per_token.gather(1, index).view(batch_size, length, -1, dim)

    def column_expert_half(
        self,
        *,
        block_sources: list[Tensor],
        streams: Float[Tensor, "batch_size seq_len slots dim"] | None,
        attn_update: Float[Tensor, "batch_size seq_len k dim"],
        column_plan: _ColumnStreamPlan,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        depth_key_cache: DepthKeyCache | None = None,
        agg: Float[Tensor, "batch_size seq_len dim"] | None = None,
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len slots dim"],
        Float[Tensor, "batch_size seq_len dim"] | None,
        dict[str, Any],
    ]:
        """Second half of :meth:`_forward_column_slot_updates`: experts + combine.

        Folds ``attn_update`` into the streams / running aggregate, runs the
        MLP-side depth read, the norm, the expert update and the final slot
        add. See :meth:`column_attn_half` for why the layer is split.

        Args:
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            streams (Float[Tensor, "batch_size seq_len slots dim"] | None):
                The streams the layer ENTERED with (None on the block's first
                layer); the attention update is folded in here.
            attn_update (Float[Tensor, "batch_size seq_len k dim"]): The
                first half's output.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real-token
                mask.
            group_router_output (RouterOutput): Block-level group routing.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.
            agg (Float[Tensor, "batch_size seq_len dim"] | None): Threaded
                running aggregate (head-only reads).
            router_noise (dict[str, Tensor] | None): Pre-drawn router noise
                factors for the expert pool.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len slots dim"],
            Float[Tensor, "batch_size seq_len dim"] | None, dict[str, Any]]:
            The extended streams, the new running aggregate (head-only reads;
            None otherwise), and the layer aux.
        """

        dense = self._dense_fastpath
        running: Tensor | None = None
        if self._column_head_only:
            # The head-only read consumes the running aggregate only; mean is
            # linear, so thread it as [batch, seq, dim] without touching the
            # streams (see forward_column).
            running = (
                agg
                if agg is not None
                else (None if streams is None else streams.mean(dim=2))
            )
            attn_mean = attn_update.mean(dim=2)
            mlp_sources = [
                *block_sources,
                attn_mean if running is None else running + attn_mean,
            ]
        else:
            streams = attn_update if streams is None else streams + attn_update
            mlp_sources = _column_read_sources(
                block_sources,
                streams,
                self._column_aggregate,
                head_only=False,
                deviation_sources=self._column_deviation,
            )

        h = self.kimi_mlp_read(mlp_sources, key_cache=depth_key_cache)
        norm_h = self.mlp_norm(h)
        private_slots, public_update, expert_aux = self._expert_update_column(
            norm_h,
            valid_token_mask,
            group_router_output,
            column_plan,
            dense=dense,
            slot_layout=True,
            router_noise=router_noise,
        )
        # [batch, seq, k, dim] (pooled) or [batch, seq, 1, dim] (broadcast).
        expert_update: Tensor | None = (
            None if private_slots is None else self._drop_path(private_slots)
        )
        if public_update is not None:
            public = self._drop_path(public_update).unsqueeze(2)
            expert_update = (
                public if expert_update is None else expert_update + public
            )
        assert expert_update is not None

        if self._column_head_only:
            update = attn_update + expert_update
            streams = update if streams is None else streams + update
            new_agg = update.mean(dim=2)
            if running is not None:
                new_agg = running + new_agg
            return streams, new_agg, expert_aux
        streams = streams + expert_update
        return streams, None, expert_aux

    def init_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RoutedTransformerLayerKVCache:
        """Initialize causal KV-cache state for this layer."""

        return RoutedTransformerLayerKVCache(
            group_caches=[
                group_attn.init_kv_cache(
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
                for group_attn in self.attention_groups
            ]
        )

    def forward_step(
        self,
        x: Float[Tensor, "batch_size dim"],
        valid_token_mask: Bool[Tensor, "batch_size"],
        group_router_output: RouterOutput,
        cache: RoutedTransformerLayerKVCache,
        pos: Int[Tensor, "batch_size"],
    ) -> tuple[
        Float[Tensor, "batch_size dim"],
        RoutedTransformerLayerKVCache,
        dict[str, Any],
    ]:
        """Process one incremental token with cached history.

        Args:
            x (Float[Tensor, "batch_size dim"]): New token embedding.
            valid_token_mask (Bool[Tensor, "batch_size"]): Valid-token mask.
            group_router_output (RouterOutput): Per-token group routing.
            cache (RoutedTransformerLayerKVCache): Layer cache.
            pos (Int[Tensor, "batch_size"]): Integer position of the new token.

        Returns:
            tuple[Float[Tensor, "batch_size dim"], RoutedTransformerLayerKVCache,
            dict[str, Any]]: Updated token, new cache, and aux data.
        """

        if self.expert_pool is not None:
            raise NotImplementedError(
                "Expert-pool layers do not support incremental (KV-cache) "
                "decoding yet; use the full training forward."
            )
        norm_x = self.attn_norm(x)
        attn_update = torch.zeros_like(x)
        new_group_caches: list[AttentionGroupKVCache] = []
        for group_idx, group_attn in enumerate(self.attention_groups):
            group_output, group_cache = group_attn.forward_step(
                norm_x,
                cache.group_caches[group_idx],
                group_router_output.active_mask[:, group_idx],
                pos,
            )
            attn_update = (
                attn_update
                + group_output
                * group_router_output.gates[
                    :,
                    group_idx,
                ].unsqueeze(-1)
            )
            new_group_caches.append(group_cache)

        x = self._apply_residual(x, attn_update)

        norm_x = self.mlp_norm(x)
        expert_router_output = _squeeze_router_output(
            self.expert_router(
                norm_x.unsqueeze(1), valid_token_mask.unsqueeze(1)
            )
        )
        expert_update = self._compute_expert_residual(
            norm_x,
            group_router_output,
            expert_router_output,
        )
        for dense_shared in self.dense_shared_experts:
            expert_update = expert_update + dense_shared(norm_x)
        x = self._apply_residual(x, expert_update)
        return (
            x,
            RoutedTransformerLayerKVCache(group_caches=new_group_caches),
            {"expert_router": _router_output_to_aux(expert_router_output)},
        )

    @torch.no_grad()
    def update_detached_parameters(self, aux: dict[str, Any]) -> None:
        """Apply detached expert-router updates after an optimizer step.

        With an expert pool the ``"expert_router"`` aux carries the PUBLIC
        pool router's output (absent in private-only mode, which has no
        public router) and ``"expert_pool_private"`` one aux per private pool
        (its ``valid_token_mask`` restricted to the group's member tokens), so
        loss-free balancing nudges every pool router that exists.

        Args:
            aux (dict[str, Any]): Layer auxiliary outputs.
        """

        if self.expert_pool is not None:
            if self.expert_pool.public_router is not None:
                _update_router_from_aux(
                    self.expert_pool.public_router, aux.get("expert_router")
                )
            private_aux = aux.get("expert_pool_private")
            if isinstance(private_aux, list):
                for private_router, router_aux in zip(
                    self.expert_pool.private_routers, private_aux
                ):
                    _update_router_from_aux(private_router, router_aux)
            return
        _update_router_from_aux(self.expert_router, aux.get("expert_router"))


class NestorBlock(nn.Module):
    """Hierarchical Nestor block with routed layers and local pooling."""

    input_dim: int
    output_dim: int
    pooling_stride: int
    causal: bool
    num_groups: int
    num_groups_per_token: int
    num_experts_per_group: int
    num_shared_experts: int
    num_experts_per_group_per_token: int
    block_mlp_hidden_dims: list[int]
    block_mlp_activation: str
    router_score_mlp_hidden_dims: list[int]
    router_score_mlp_activation: str
    router_noise_mlp_hidden_dims: list[int]
    router_noise_mlp_activation: str
    loss_free_balancing: bool
    bias_update_rate: float
    attention_token_dispatch: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pooling_stride: int,
        block_num_layers: int,
        num_groups: int,
        num_groups_per_token: int,
        router_num_experts_per_group: int,
        router_num_shared_experts: int,
        router_num_experts_per_group_per_token: int,
        block_num_heads: int,
        block_head_dim: int,
        block_mlp_hidden_dims: list[int],
        block_mlp_activation: str,
        router_score_mlp_hidden_dims: list[int],
        router_score_mlp_activation: str,
        router_noise_mlp_hidden_dims: list[int],
        router_noise_mlp_activation: str,
        *,
        layer_num_experts: list[int] | None = None,
        depth_scale: float,
        rope_base: float = 10000.0,
        causal: bool = False,
        loss_free_balancing: bool = False,
        bias_update_rate: float = 0.0,
        pooling_keep_ratio: float | None = None,
        attention_token_dispatch: bool = False,
        router_num_dense_shared_experts: int = 0,
        residual_mode: str = "convex",
        block_residual_mode: str = "projection",
        kimi_max_sources: int | None = None,
        column_include_aggregate: bool = False,
        column_head_only: bool = False,
        column_deviation_sources: bool = False,
        keeps_columns: bool = True,
        column_trailing_layers: int | None = None,
        drop_path_rate: float = 0.0,
        summary_dim: int | None = None,
        xblock_consumes_summaries: bool = True,
        xblock_emits_summary: bool = True,
        num_global_tokens: int = 0,
        pooling_mode: str = "mean",
        apply_output_projection: bool = True,
        expert_pool_public_experts: int = 0,
        expert_pool_public_top_k: int = 1,
        expert_pool_private_experts: int = 0,
        expert_pool_private_top_k: int = 1,
        expert_pool_active_budget: float | None = None,
        expert_pool_public_budget: float | None = None,
        expert_pool_private_gate: str = "renorm",
    ) -> None:
        """Initialize one hierarchical Nestor block.

        Args:
            input_dim (int): Input token width.
            output_dim (int): Output token width after projection.
            pooling_stride (int): Local pooling stride after the block.
            block_num_layers (int): Number of routed transformer layers.
            num_groups (int): Attention groups.
            num_groups_per_token (int): Top-k active groups per token.
            router_num_experts_per_group (int): Experts per group (the uniform
                count, and the fallback when ``layer_num_experts`` is None).
            layer_num_experts (list[int] | None): Per-layer routed-expert counts
                (length ``block_num_layers``) for a depth-varying MoE sparsity
                schedule. None builds every layer at
                ``router_num_experts_per_group``.
            router_num_shared_experts (int): Shared experts across groups.
            router_num_experts_per_group_per_token (int): Top-k experts per token.
            block_num_heads (int): Heads per attention group.
            block_head_dim (int): Width of each attention head.
            block_mlp_hidden_dims (list[int]): Expert MLP hidden widths.
            block_mlp_activation (str): Expert MLP activation.
            router_score_mlp_hidden_dims (list[int]): Score-router hidden widths.
            router_score_mlp_activation (str): Score-router activation.
            router_noise_mlp_hidden_dims (list[int]): Noise-router hidden widths.
            router_noise_mlp_activation (str): Noise-router activation.
            depth_scale (float): Convex residual weight ``1/D`` for every layer.
            rope_base (float): RoPE frequency base ``theta_l^rope`` for this block.
            causal (bool): Whether to enforce prefix-stable execution.
            loss_free_balancing (bool): Whether detached bias updates are used.
            bias_update_rate (float): Detached bias update step size.
            xblock_consumes_summaries (bool): Under ``attn_res_xblock``, whether
                this block reads preceding blocks' summaries (False for the
                first block, which has none).
            xblock_emits_summary (bool): Under ``attn_res_xblock``, whether this
                block produces a summary for later blocks (False for the last
                block, whose summary would have no reader -- its parameters
                would break DDP without find_unused_parameters).
            column_include_aggregate (bool): Under
                ``block_residual_mode='attn_res_column'``, whether this block
                also carries and exports the aggregate of its ``k`` column
                streams, contributing ``k + 1`` depth sources. No-op for a
                single-column block.
            apply_output_projection (bool): Whether this block owns the legacy
                pool-and-project boundary transition. Block AttnRes disables it
                and consumes the raw block output at ``input_dim`` instead.
            block_residual_mode (str): Between-block transition mode of the
                owning backbone, threaded to every layer. ``"attn_res_kimi"``
                enables the per-sublayer Kimi depth reads; other values change
                nothing.
            kimi_max_sources (int | None): Largest depth-read source count of
                the owning backbone (``num_blocks + 1``); required under
                ``block_residual_mode="attn_res_kimi"``, ignored otherwise.
            expert_pool_public_experts (int): Public expert-pool size per
                layer; 0 (default) drops the public path -- private-only
                pooling with a private pool, no pooling without one.
            expert_pool_public_top_k (int): Public experts per token.
            expert_pool_private_experts (int): Private expert-pool size per
                attention group; > 0 replaces every layer's expert banks with
                an ExpertPoolMLP. 0 (default) disables.
            expert_pool_active_budget (float | None): Active expert width in
                multiples of ``dim``, overriding the pool's 2d/4d contract.
            expert_pool_public_budget (float | None): Public family's active
                width in multiples of ``dim`` (``None`` = the 2d contract).
            expert_pool_private_top_k (int): Private experts per selected
                attention group.
            expert_pool_private_gate (str): Private-router gate rule,
                ``"renorm"`` or ``"switch"`` (see ``Router.gate_mode``).
        """

        super().__init__()
        if block_num_layers < 1:
            raise ValueError(
                "Each Nestor block must contain at least one layer."
            )
        if pooling_stride < 1:
            raise ValueError("pooling_stride must be a positive integer.")
        if not 1 <= num_groups_per_token <= num_groups:
            raise ValueError("num_groups_per_token must be in [1, num_groups].")
        # Per-layer expert schedule (depth-varying MoE sparsity). None => every
        # layer uses router_num_experts_per_group; otherwise layer i is built
        # with layer_num_experts[i] routed experts. The routed top-k and shared
        # count must fit the SMALLEST layer in the schedule.
        if layer_num_experts is None:
            layer_num_experts = [router_num_experts_per_group] * block_num_layers
        if len(layer_num_experts) != block_num_layers:
            raise ValueError(
                "layer_num_experts must have block_num_layers entries "
                f"({block_num_layers}), got {len(layer_num_experts)}."
            )
        self.layer_num_experts = [int(e) for e in layer_num_experts]
        min_layer_experts = min(self.layer_num_experts)
        if not 0 <= router_num_shared_experts <= min_layer_experts:
            raise ValueError(
                "router_num_shared_experts must be in [0, min(layer_num_experts)]."
            )
        if not 1 <= router_num_experts_per_group_per_token <= min_layer_experts:
            raise ValueError(
                "router_num_experts_per_group_per_token must be in "
                "[1, min(layer_num_experts)]."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling_stride = pooling_stride
        self.pooling_keep_ratio = pooling_keep_ratio
        self.causal = causal
        self.num_groups = num_groups
        self.num_groups_per_token = num_groups_per_token
        # Kept in sync with RoutedTransformerLayer._column_aggregate: the layers
        # read the aggregate, the block exports it (same single-column no-op).
        self._column_aggregate = (
            column_include_aggregate and num_groups_per_token > 1
        )
        self._column_deviation = column_deviation_sources
        # False => this block runs the merged recurrence and exports one
        # source (see NestorConfig.block_column_blocks).
        self._keeps_columns = keeps_columns
        self._column_trailing = column_trailing_layers
        self.num_experts_per_group = router_num_experts_per_group
        self.num_shared_experts = router_num_shared_experts
        self.num_dense_shared_experts = router_num_dense_shared_experts
        self.num_experts_per_group_per_token = (
            router_num_experts_per_group_per_token
        )
        self.block_mlp_hidden_dims = list(block_mlp_hidden_dims)
        self.block_mlp_activation = block_mlp_activation
        self.router_score_mlp_hidden_dims = list(router_score_mlp_hidden_dims)
        self.router_score_mlp_activation = router_score_mlp_activation
        self.router_noise_mlp_hidden_dims = list(router_noise_mlp_hidden_dims)
        self.router_noise_mlp_activation = router_noise_mlp_activation
        self.loss_free_balancing = loss_free_balancing
        self.bias_update_rate = bias_update_rate
        self.attention_token_dispatch = attention_token_dispatch
        self.expert_pool_public_experts = expert_pool_public_experts
        self.expert_pool_public_top_k = expert_pool_public_top_k
        self.expert_pool_private_experts = expert_pool_private_experts
        self.expert_pool_private_top_k = expert_pool_private_top_k
        self.expert_pool_active_budget = expert_pool_active_budget
        self.expert_pool_public_budget = expert_pool_public_budget
        self.expert_pool_private_gate = expert_pool_private_gate
        if residual_mode not in ("convex", "attn_res", "attn_res_xblock"):
            raise ValueError(
                f"Unknown residual_mode '{residual_mode}'. Expected 'convex', "
                "'attn_res', or 'attn_res_xblock'."
            )
        if residual_mode == "attn_res_xblock" and causal:
            # The cross-block summary is a global pool over all tokens, which
            # would leak future positions under a causal mask.
            raise ValueError(
                "residual_mode='attn_res_xblock' is non-causal only "
                "(global cross-block summaries break causality)."
            )
        self.residual_mode = residual_mode
        self._uses_attn_res = residual_mode in ("attn_res", "attn_res_xblock")
        self.summary_dim = input_dim if summary_dim is None else summary_dim
        # Global/register tokens prepended to the sequence; forced into every
        # attention group (see _force_global_tokens_all_groups) and bypassed by
        # token pooling. Backbone-level count, identical for every block.
        self._num_global = num_global_tokens
        # Token-pool operator (see NestorConfig.pooling_mode).
        if pooling_mode not in ("mean", "concat"):
            raise ValueError(
                f"pooling_mode must be 'mean' or 'concat', got {pooling_mode!r}."
            )
        self._concat_pool = pooling_mode == "concat" and pooling_stride > 1
        if self._concat_pool and pooling_keep_ratio is not None:
            raise ValueError(
                "pooling_mode='concat' requires integer strides "
                "(pooling_keep_ratios is fractional, has no fixed window)."
            )
        if self._concat_pool and num_global_tokens > 0:
            raise ValueError(
                "pooling_mode='concat' is incompatible with num_global_tokens>0 "
                "(d-wide global tokens cannot share the S*d->d' projection)."
            )
        if not apply_output_projection and (
            pooling_stride != 1 or pooling_keep_ratio not in (None, 1.0)
        ):
            raise ValueError(
                "Disabling the output projection requires constant token count."
            )
        self.apply_output_projection = apply_output_projection

        self.group_router = Router(
            feature_dim=input_dim,
            num_choices=num_groups,
            num_choices_per_token=num_groups_per_token,
            score_mlp_hidden_dims=router_score_mlp_hidden_dims,
            score_mlp_activation=router_score_mlp_activation,
            noise_mlp_hidden_dims=router_noise_mlp_hidden_dims,
            noise_mlp_activation=router_noise_mlp_activation,
            # The group router is computed once and reused by all M_l layers, so
            # it receives M_l branch gradients and its rate carries 1/sqrt(M_l).
            score_lr_mul=block_num_layers**-0.5,
            loss_free_balancing=loss_free_balancing,
            bias_update_rate=bias_update_rate,
        )
        # NESTOR_DENSE_NO_ROUTER=1: a single attention group makes the group
        # gates exactly 1.0 with exactly zero gradient (softmax over one logit),
        # so the group router is dead compute — skip running it and freeze its
        # parameters. Mirrors the per-layer expert-router skip; see
        # RoutedTransformerLayer._forward_dense.
        import os as _os

        self._dense_noroute = (
            num_groups == 1
            and _os.environ.get("NESTOR_DENSE_NO_ROUTER") == "1"
        )
        if self._dense_noroute:
            self.group_router.requires_grad_(False)
        # Stochastic-depth schedule: drop-path rate rises linearly with depth
        # within the block (timm rule). Single-block models (the dense ViT) get
        # the exact DeiT schedule 0 -> drop_path_rate across all layers.
        _dpr = [
            drop_path_rate * i / max(1, block_num_layers - 1)
            for i in range(block_num_layers)
        ]
        self.layers = cast(
            list[RoutedTransformerLayer],
            nn.ModuleList(
                [
                    RoutedTransformerLayer(
                        dim=input_dim,
                        num_experts_per_group=self.layer_num_experts[_i],
                        num_shared_experts=router_num_shared_experts,
                        num_dense_shared_experts=(
                            router_num_dense_shared_experts
                        ),
                        num_experts_per_group_per_token=(
                            router_num_experts_per_group_per_token
                        ),
                        num_groups=num_groups,
                        num_heads=block_num_heads,
                        head_dim=block_head_dim,
                        mlp_hidden_dims=block_mlp_hidden_dims,
                        mlp_activation=block_mlp_activation,
                        router_score_mlp_hidden_dims=router_score_mlp_hidden_dims,
                        router_score_mlp_activation=router_score_mlp_activation,
                        router_noise_mlp_hidden_dims=router_noise_mlp_hidden_dims,
                        router_noise_mlp_activation=router_noise_mlp_activation,
                        depth_scale=depth_scale,
                        rope_base=rope_base,
                        loss_free_balancing=loss_free_balancing,
                        bias_update_rate=bias_update_rate,
                        attention_token_dispatch=attention_token_dispatch,
                        # Layers only distinguish convex vs depth-attention; the
                        # cross-block wiring lives on the block, not the layer.
                        residual_mode=(
                            "attn_res" if self._uses_attn_res else "convex"
                        ),
                        block_residual_mode=block_residual_mode,
                        kimi_max_sources=kimi_max_sources,
                        column_include_aggregate=column_include_aggregate,
                        column_head_only=column_head_only,
                        column_deviation_sources=column_deviation_sources,
                        keeps_columns=keeps_columns,
                        column_trailing_layers=column_trailing_layers,
                        drop_path_rate=_dpr[_i],
                        causal=causal,
                        num_groups_per_token=num_groups_per_token,
                        expert_pool_public_experts=expert_pool_public_experts,
                        expert_pool_public_top_k=expert_pool_public_top_k,
                        expert_pool_private_experts=(
                            expert_pool_private_experts
                        ),
                        expert_pool_private_top_k=expert_pool_private_top_k,
                        expert_pool_active_budget=expert_pool_active_budget,
                        expert_pool_public_budget=expert_pool_public_budget,
                        expert_pool_private_gate=expert_pool_private_gate,
                    )
                    for _i in range(block_num_layers)
                ]
            ),
        )
        # Block-level batched dispatch attention: group membership is decided
        # once per block, so the compaction plan (one .item() sync) is shared
        # by all layers; the layers' _dispatch_full_attn flag already encodes
        # token dispatch + not causal + NESTOR_DENSE_FULL_ATTN. Top-1 group
        # routing only: the gather-only dispatch needs each token to own
        # exactly one real compact row. Global tokens are members of every
        # group (multi-membership), so they also disable the batched plan ->
        # the per-group gathered loop, which handles >1 group per token.
        self._attn_dispatch_plan = (
            num_groups > 1
            and num_groups_per_token == 1
            and num_global_tokens == 0
            and bool(getattr(self.layers[0], "_dispatch_full_attn", False))
        )
        # Pooled, RMS-normalized features are projected up as an in-stream map
        # with fan-in d_l and fan-out d_{l+1}; Muon applies the sqrt(d_{l+1}/d_l)
        # update scale that the scaled semi-orthogonal init meets at step zero.
        # Concat pooling feeds the S concatenated tokens (S*d wide, each
        # component still ~unit RMS) into this projection so it learns the
        # window combination; mean pooling feeds the d-wide pooled token.
        proj_in_dim = input_dim * pooling_stride if self._concat_pool else input_dim
        self.output_proj: nn.Module
        if apply_output_projection:
            self.output_proj = ScaledLinear(
                proj_in_dim, output_dim, role="projection"
            )
        else:
            self.output_proj = nn.Identity()
        if self._uses_attn_res:
            # Final depth aggregation over the block's full history (Fig 1b top),
            # then RMSNorm to restore Theta(1) RMS before pooling -- a uniform
            # convex combination of M sublayer outputs has RMS ~1/sqrt(M) at init,
            # whereas pooling / output_proj assume a Theta(1) input.
            self.out_query = ScaledLinear(
                input_dim, 1, role="score", bias=False, zero_init=True
            )
            self.out_key_norm = RMSNorm(input_dim, bias=False)
            self.out_norm = RMSNorm(input_dim)
        self.xblock_consumes_summaries = xblock_consumes_summaries
        self.xblock_emits_summary = xblock_emits_summary
        if residual_mode == "attn_res_xblock":
            # Cross-block AttnRes (Level 2): each preceding NestorBlock supplies
            # one global summary vector at the canonical ``summary_dim``.
            # ``summary_out_proj`` + ``summary_norm`` produce this block's summary
            # (mean-pool -> project -> RMSNorm), and ``summary_in_proj`` reads the
            # prior summaries into this block's width as extra depth sources. The
            # token-axis mismatch across blocks is sidestepped because summaries
            # are per-sequence (one vector), broadcast over the current tokens.
            # The modules are positional: the first block has no summaries to
            # read and the last block's summary has no reader, so those modules
            # are not created -- a parameter with no path to the loss breaks
            # DDP without find_unused_parameters (dropped for perf in a3b04a8).
            if xblock_emits_summary:
                self.summary_out_proj = ScaledLinear(
                    input_dim, self.summary_dim, role="projection", bias=False
                )
                self.summary_norm = RMSNorm(self.summary_dim)
            if xblock_consumes_summaries:
                self.summary_in_proj = ScaledLinear(
                    self.summary_dim, input_dim, role="projection", bias=False
                )
        self.register_buffer(
            "_cached_causal_mask",
            torch.empty((0, 0), dtype=torch.bool),
            persistent=False,
        )

    def _compose_attn_mask(
        self,
        valid_mask: Bool[Tensor, "batch_size seq_len"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
    ) -> Bool[Tensor, "batch_size seq_len seq_len"] | None:
        """Combine caller attention masks with validity and causality masks.

        ``attn_mask=None`` is the trivial-mask contract (the adapter
        guarantees every token is valid and encodes no pair structure): the
        composed mask stays ``None`` and causality is applied analytically by
        the None-mask attention paths. Composing here would materialize the
        O(S^2) tril for nothing -- the cached tril alone is 16 GiB at S=131072.
        """

        if attn_mask is None:
            return None
        combined = attn_mask & valid_mask[:, :, None] & valid_mask[:, None, :]
        if self.causal:
            seq_len = combined.shape[-1]
            causal_mask = self._cached_causal_mask
            if (
                causal_mask.device != combined.device
                or causal_mask.shape[0] < seq_len
            ):
                causal_mask = torch.tril(
                    torch.ones(
                        (seq_len, seq_len),
                        dtype=torch.bool,
                        device=combined.device,
                    )
                )
                self._cached_causal_mask = causal_mask
            combined = combined & causal_mask[:seq_len, :seq_len].unsqueeze(0)
        return combined

    def _fractional_pool_tokens(
        self,
        x: Float[Tensor, "batch_size seq_len input_dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"],
    ) -> tuple[
        Float[Tensor, "batch_size pooled_seq_len input_dim"],
        Int[Tensor, "batch_size pooled_seq_len"],
        Bool[Tensor, "batch_size pooled_seq_len pooled_seq_len"],
    ]:
        """Adaptive-average-pool to keep ``pooling_keep_ratio`` of the tokens.

        Used for fully-valid (non-padded, non-causal) sequences, i.e. image
        classification, where every token is real and attention is full. Each
        block exit merges the sequence into ``round(L * ratio)`` evenly-sized
        segments via ``adaptive_avg_pool1d``; pooled tokens are all valid.
        """

        assert not self.causal, "Fractional pooling is non-causal only."
        assert attn_mask is not None, (
            "fractional pooling requires a dense attn_mask "
            "(trivial-mask contract models must not pool)."
        )
        batch_size, seq_len, _ = x.shape
        assert self.pooling_keep_ratio is not None
        target_len = max(1, round(seq_len * self.pooling_keep_ratio))
        if target_len >= seq_len:
            valid_mask = pos != PADDING_TOKEN_POS
            return x, pos, self._compose_attn_mask(valid_mask, attn_mask)

        pooled_x = F.adaptive_avg_pool1d(
            x.transpose(1, 2), target_len
        ).transpose(1, 2)
        pooled_pos = (
            torch.arange(target_len, dtype=pos.dtype, device=pos.device)
            .unsqueeze(0)
            .expand(batch_size, target_len)
        )
        pooled_valid = torch.ones(
            (batch_size, target_len), dtype=torch.bool, device=x.device
        )
        full_mask = torch.ones(
            (batch_size, target_len, target_len),
            dtype=torch.bool,
            device=x.device,
        )
        return pooled_x, pooled_pos, self._compose_attn_mask(
            pooled_valid, full_mask
        )

    def _pool_tokens(
        self,
        x: Float[Tensor, "batch_size seq_len input_dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
    ) -> tuple[
        Float[Tensor, "batch_size pooled_seq_len input_dim"],
        Int[Tensor, "batch_size pooled_seq_len"],
        Bool[Tensor, "batch_size pooled_seq_len pooled_seq_len"] | None,
    ]:
        """Average-pool fixed windows between blocks (mask None passes through
        only on the stride-1 identity path).

        The implementation follows the local-pooling behavior described in
        ``implementation.tex``. Positions are represented by the rounded mean
        source position of each pooled window.
        """

        batch_size, seq_len, dim = x.shape
        if self.pooling_keep_ratio is not None and self.pooling_keep_ratio < 1.0:
            return self._fractional_pool_tokens(x, pos, attn_mask)
        if self.pooling_stride == 1:
            # Window of one: pooled_x == x * valid (sum over one token divided
            # by sqrt(1)), pooled_pos == pos, and the representative-index
            # gathers reduce to the identity on attn_mask. Skipping them avoids
            # two [B, S, S] gathers per block and lets the whole mask chain be
            # dead-code-eliminated on the maskless dense path.
            valid = pos != PADDING_TOKEN_POS
            return x * valid.unsqueeze(-1).to(dtype=x.dtype), pos, attn_mask
        # Real pooling gathers the [B, S, S] mask, so it needs a dense one;
        # the trivial-mask contract (None) never reaches a pooling model.
        assert attn_mask is not None, (
            "pooling with stride > 1 requires a dense attn_mask "
            "(trivial-mask contract models must not pool)."
        )
        valid_mask = pos != PADDING_TOKEN_POS
        output_seq_len = (
            seq_len + self.pooling_stride - 1
        ) // self.pooling_stride
        pad_len = output_seq_len * self.pooling_stride - seq_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            pos = F.pad(pos, (0, pad_len), value=PADDING_TOKEN_POS)
            valid_mask = F.pad(valid_mask, (0, pad_len), value=False)

        window_x = x.reshape(
            batch_size, output_seq_len, self.pooling_stride, dim
        )
        window_pos = pos.reshape(
            batch_size, output_seq_len, self.pooling_stride
        )
        window_valid = valid_mask.reshape(
            batch_size, output_seq_len, self.pooling_stride
        )

        weights = window_valid.unsqueeze(-1).to(dtype=window_x.dtype)
        if self._concat_pool:
            # Concat pooling: stack the S window tokens into one S*d vector
            # (invalid slots zeroed) and let output_proj (S*d -> d') learn the
            # combination. No averaging loss; each component keeps unit RMS so
            # no sqrt(|W|) correction is needed (unlike mean pooling).
            pooled_x = (window_x * weights).reshape(
                batch_size, output_seq_len, self.pooling_stride * dim
            )
        else:
            # RMS pooling: dividing by sqrt(|W|) cancels the sqrt(|W|) growth of
            # a sum of weakly correlated unit-RMS tokens, so the next block sees
            # Theta(1) RMS input even when the stride is part of the scaling family.
            pooled_x = (window_x * weights).sum(dim=2) / weights.sum(
                dim=2
            ).clamp_min(1.0).sqrt()
        pooled_valid = window_valid.any(dim=2)

        valid_counts = (
            window_valid.to(dtype=torch.float32).sum(dim=2).clamp_min(1.0)
        )
        mean_pos = (
            torch.where(
                window_valid,
                window_pos,
                torch.zeros_like(window_pos),
            )
            .to(dtype=torch.float32)
            .sum(dim=2)
            / valid_counts
        )
        pooled_pos = torch.where(
            pooled_valid,
            torch.round(mean_pos).to(dtype=pos.dtype),
            torch.full(
                (batch_size, output_seq_len),
                fill_value=PADDING_TOKEN_POS,
                dtype=pos.dtype,
                device=pos.device,
            ),
        )

        window_offsets = torch.arange(
            self.pooling_stride,
            dtype=torch.long,
            device=x.device,
        ).view(1, 1, -1)
        last_valid_offsets = (
            window_valid.to(dtype=torch.long) * (window_offsets + 1)
        ).argmax(dim=2)
        representative_indices = (
            torch.arange(
                output_seq_len, dtype=torch.long, device=x.device
            ).view(1, -1)
            * self.pooling_stride
            + last_valid_offsets
        )

        pooled_attn_mask = attn_mask.gather(
            dim=1,
            index=representative_indices.unsqueeze(-1).expand(-1, -1, seq_len),
        ).gather(
            dim=2,
            index=representative_indices.unsqueeze(1).expand(
                -1, output_seq_len, -1
            ),
        )

        return (
            pooled_x,
            pooled_pos,
            self._compose_attn_mask(pooled_valid, pooled_attn_mask),
        )

    def init_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> NestorBlockKVCache:
        """Initialize cache state for one block.

        Args:
            batch_size (int): Batch size for streaming inference.
            device (torch.device): Cache device.
            dtype (torch.dtype): Cache dtype.

        Returns:
            NestorBlockKVCache: Empty block cache.
        """

        return NestorBlockKVCache(
            layer_caches=[
                layer.init_kv_cache(
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
                for layer in self.layers
            ],
            pooling_cache=LocalPoolingKVCache(
                token_sum=torch.zeros(
                    (batch_size, self.input_dim),
                    dtype=dtype,
                    device=device,
                ),
                token_count=torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
                position_sum=torch.zeros(
                    batch_size, dtype=torch.float32, device=device
                ),
            ),
        )

    def _pool_token_step(
        self,
        x: Float[Tensor, "batch_size input_dim"],
        pos: Int[Tensor, "batch_size"],
        pooling_cache: LocalPoolingKVCache | None,
    ) -> tuple[
        Float[Tensor, "batch_size input_dim"],
        Int[Tensor, "batch_size"],
        LocalPoolingKVCache | None,
    ]:
        """Add one token to the current local-pooling window."""

        assert pooling_cache is not None
        valid_mask = pos != PADDING_TOKEN_POS
        next_sum = pooling_cache.token_sum + x * valid_mask.unsqueeze(-1).to(
            dtype=x.dtype
        )
        next_count = pooling_cache.token_count + valid_mask.to(dtype=torch.long)
        next_position_sum = pooling_cache.position_sum + torch.where(
            valid_mask,
            pos.to(dtype=torch.float32),
            torch.zeros_like(pooling_cache.position_sum),
        )

        emit_mask = valid_mask & (next_count == self.pooling_stride)
        pooled_x = torch.zeros_like(x)
        pooled_pos = torch.full_like(pos, fill_value=PADDING_TOKEN_POS)
        if emit_mask.any():
            denom = next_count[emit_mask].unsqueeze(-1).to(dtype=x.dtype)
            pooled_x[emit_mask] = next_sum[emit_mask] / denom.sqrt()
            pooled_pos[emit_mask] = torch.round(
                next_position_sum[emit_mask]
                / next_count[emit_mask].to(dtype=torch.float32)
            ).to(dtype=pos.dtype)
            next_sum[emit_mask] = 0
            next_count[emit_mask] = 0
            next_position_sum[emit_mask] = 0

        return (
            pooled_x,
            pooled_pos,
            LocalPoolingKVCache(
                token_sum=next_sum,
                token_count=next_count,
                position_sum=next_position_sum,
            ),
        )

    def flush(
        self,
        cache: NestorBlockKVCache,
    ) -> tuple[
        Float[Tensor, "batch_size output_dim"],
        Int[Tensor, "batch_size"],
        NestorBlockKVCache,
    ]:
        """Flush a trailing partial local-pooling window.

        Args:
            cache (NestorBlockKVCache): Block cache to flush.

        Returns:
            tuple[Float[Tensor, "batch_size output_dim"], Int[Tensor, "batch_size"],
            NestorBlockKVCache]: Flushed token, position, and reset cache.
        """

        assert cache.pooling_cache is not None
        emit_mask = cache.pooling_cache.token_count > 0
        pooled_x = torch.zeros_like(cache.pooling_cache.token_sum)
        pooled_pos = torch.full_like(
            cache.pooling_cache.token_count,
            fill_value=PADDING_TOKEN_POS,
        )
        if emit_mask.any():
            denom = (
                cache.pooling_cache.token_count[emit_mask]
                .unsqueeze(-1)
                .to(dtype=cache.pooling_cache.token_sum.dtype)
            )
            pooled_x[emit_mask] = (
                cache.pooling_cache.token_sum[emit_mask] / denom.sqrt()
            )
            pooled_pos[emit_mask] = torch.round(
                cache.pooling_cache.position_sum[emit_mask]
                / cache.pooling_cache.token_count[emit_mask].to(
                    dtype=torch.float32
                )
            ).to(dtype=pooled_pos.dtype)

        next_cache = NestorBlockKVCache(
            layer_caches=cache.layer_caches,
            pooling_cache=LocalPoolingKVCache(
                token_sum=torch.zeros_like(cache.pooling_cache.token_sum),
                token_count=torch.zeros_like(cache.pooling_cache.token_count),
                position_sum=torch.zeros_like(cache.pooling_cache.position_sum),
            ),
        )
        pooled_x = self.output_proj(pooled_x)
        pooled_x = pooled_x * (pooled_pos != PADDING_TOKEN_POS).unsqueeze(
            -1
        ).to(dtype=pooled_x.dtype)
        return pooled_x, pooled_pos, next_cache

    def _summarize_block(
        self,
        x: Float[Tensor, "batch_size seq_len input_dim"],
        valid_mask: Bool[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size summary_dim"]:
        """Global per-sequence summary of the block output for cross-block AttnRes.

        Mean-pools the aggregated block output over valid tokens, projects it to
        the canonical ``summary_dim``, and RMSNorms to Theta(1) so prior-block
        summaries enter later blocks' depth attention at a comparable scale.
        """

        weights = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.summary_norm(self.summary_out_proj(pooled))

    def _column_layer_index(self, layer: nn.Module) -> int:
        """Position of ``layer`` in ``self.layers`` (a checkpoint-friendly handle).

        Args:
            layer (nn.Module): One of this block's layers.

        Returns:
            int: Its index in ``self.layers``.
        """

        for index, candidate in enumerate(self.layers):
            if candidate is layer:
                return index
        raise ValueError("layer is not one of this block's layers")

    def _column_layers_region(
        self,
        *,
        layer_indices: list[int],
        noises: list[dict[str, Tensor] | None] | None,
        block_sources: list[Tensor],
        streams: Float[Tensor, "batch_size seq_len slots dim"] | None,
        agg: Float[Tensor, "batch_size seq_len dim"] | None,
        column_plan: _ColumnStreamPlan,
        base_attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
        dispatch_plan: _AttentionDispatchPlan | _PackedAttentionPlan | None,
        depth_key_cache: DepthKeyCache | None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len slots dim"],
        Float[Tensor, "batch_size seq_len dim"] | None,
        list[dict[str, Any]],
    ]:
        """Run the block's column layers back to back (the column-mode loop).

        Each layer is one recompute region under ``NESTOR_ACTIVATION_CHECKPOINT=1``
        (two under ``..._SPLIT=1``, :func:`_activation_checkpoint_split_enabled`).
        Under ``..._BLOCK=1`` the caller wraps this whole method in one more,
        outer region (:func:`_activation_checkpoint_block_enabled`), which is
        why the layers are addressed by index and the noises can be handed in
        pre-drawn: the outer recompute must not draw RNG.

        Args:
            layer_indices (list[int]): Indices into ``self.layers`` of the
                column layers, in depth order.
            noises (list[dict[str, Tensor] | None] | None): Pre-drawn router
                noise per layer (the BLOCK mode); None draws each layer's noise
                right before the layer, as the plain path always did.
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``, in depth order.
            streams (Float[Tensor, "batch_size seq_len slots dim"] | None):
                Running per-column partial sums at entry (None at block entry).
            agg (Float[Tensor, "batch_size seq_len dim"] | None): Running
                aggregate at entry (the merged prefix; head-only reads).
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            base_attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None):
                Attention mask.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real-token
                mask.
            group_router_output (RouterOutput): Block-level group routing.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            dispatch_plan (_AttentionDispatchPlan | _PackedAttentionPlan |
                None): Block-level attention plan, if any.
            depth_key_cache (DepthKeyCache | None): Cache of the completed
                sources' unit-RMS keys.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len slots dim"],
            Float[Tensor, "batch_size seq_len dim"] | None,
            list[dict[str, Any]]]: The streams and aggregate after the last
            layer, and the per-layer aux dicts in layer order.
        """

        layers_aux: list[dict[str, Any]] = []
        split_layers = (
            _activation_checkpoint_enabled()
            and _activation_checkpoint_split_enabled()
        )
        lean = split_layers and _activation_checkpoint_lean_enabled()
        for position, index in enumerate(layer_indices):
            layer = self.layers[index]
            # Drawn BEFORE either region so the RNG stream is consumed in the
            # same order whether or not the layer is split.
            router_noise = (
                noises[position]
                if noises is not None
                else self._layer_router_noise(layer, valid_token_mask)
            )
            if lean and layer._column_slot_updates and layer._column_head_only:
                # The lean split (see _activation_checkpoint_lean_enabled):
                # the regions get the means, the stream adds and the
                # aggregate update happen out here. Every expression below is
                # column_expert_half's head-only branch, verbatim, so the
                # values are the plain path's.
                stream_mean = None if streams is None else streams.mean(dim=2)
                running = agg if agg is not None else stream_mean
                tail_chunks = _tail_query_chunks()
                if tail_chunks > 1 and getattr(layer, "_dense_tail_attn", False):
                    # The dense tail's attention half as one K/V region plus
                    # one region per query slice (see _tail_query_chunks).
                    if getattr(layer, "drop_path_rate", 0.0) > 0.0 and layer.training:
                        raise RuntimeError(
                            "NESTOR_DENSE_TAIL_QUERY_CHUNKS needs drop-path 0."
                        )
                    norm_h, tail_k, tail_v = _maybe_checkpoint(
                        layer.column_tail_kv_lean,
                        block_sources=block_sources,
                        stream_mean=stream_mean,
                        pos=pos,
                        depth_key_cache=depth_key_cache,
                    )
                    seq_len = valid_token_mask.shape[1]
                    n_slices = min(tail_chunks, seq_len)
                    assembler = _SeqAssembler(seq_len)
                    for i in range(n_slices):
                        lo, hi = (i * seq_len) // n_slices, ((i + 1) * seq_len) // n_slices
                        assembler.put(
                            _maybe_checkpoint(
                                layer.column_tail_attn_slice_lean,
                                norm_h=norm_h,
                                k=tail_k,
                                v=tail_v,
                                group_router_output=group_router_output,
                                pos=pos,
                                lo=lo,
                                hi=hi,
                            ),
                            lo,
                            hi,
                        )
                    attn_update = layer._drop_path(assembler.result())
                    del assembler, norm_h, tail_k, tail_v
                else:
                    attn_update = _maybe_checkpoint(
                        layer.column_attn_half_lean,
                        block_sources=block_sources,
                        stream_mean=stream_mean,
                        column_plan=column_plan,
                        base_attn_mask=base_attn_mask,
                        group_router_output=group_router_output,
                        pos=pos,
                        dispatch_plan=dispatch_plan,
                        depth_key_cache=depth_key_cache,
                    )
                attn_mean = attn_update.mean(dim=2)
                chunks = _expert_half_chunks()
                if chunks > 1:
                    expert_update, layer_aux = self._column_expert_half_lean_chunked(
                        layer,
                        chunks=chunks,
                        router_noise=router_noise,
                        block_sources=block_sources,
                        running=running,
                        attn_mean=attn_mean,
                        column_plan=column_plan,
                        valid_token_mask=valid_token_mask,
                        group_router_output=group_router_output,
                    )
                else:
                    expert_update, layer_aux = _maybe_checkpoint(
                        layer.column_expert_half_lean,
                        router_noise=router_noise,
                        block_sources=block_sources,
                        running=running,
                        attn_mean=attn_mean,
                        column_plan=column_plan,
                        valid_token_mask=valid_token_mask,
                        group_router_output=group_router_output,
                        depth_key_cache=depth_key_cache,
                    )
                # The same expressions as column_expert_half's head-only
                # branch, in an order that keeps at most three [S, k, dim]
                # streams alive: the two updates die once summed, the running
                # stream is extended in place (nothing has saved it -- the
                # regions took its mean, whose backward needs no input -- so
                # the in-place add is autograd-legal and bitwise the same).
                update = attn_update + expert_update
                del attn_update, expert_update
                new_agg = update.mean(dim=2)
                streams = update if streams is None else streams.add_(update)
                del update
                agg = new_agg if running is None else running + new_agg
                layers_aux.append(layer_aux)
                continue
            if split_layers and layer._column_slot_updates:
                # Two recompute regions per layer (see
                # _activation_checkpoint_split_enabled); the attention
                # update between them is one more checkpoint boundary.
                attn_update = _maybe_checkpoint(
                    layer.column_attn_half,
                    block_sources=block_sources,
                    streams=streams,
                    column_plan=column_plan,
                    base_attn_mask=base_attn_mask,
                    group_router_output=group_router_output,
                    pos=pos,
                    dispatch_plan=dispatch_plan,
                    depth_key_cache=depth_key_cache,
                )
                chunks = _expert_half_chunks()
                if chunks > 1:
                    streams, agg, layer_aux = self._column_expert_half_chunked(
                        layer,
                        chunks=chunks,
                        router_noise=router_noise,
                        block_sources=block_sources,
                        streams=streams,
                        attn_update=attn_update,
                        column_plan=column_plan,
                        valid_token_mask=valid_token_mask,
                        group_router_output=group_router_output,
                        agg=agg,
                    )
                else:
                    streams, agg, layer_aux = _maybe_checkpoint(
                        layer.column_expert_half,
                        router_noise=router_noise,
                        block_sources=block_sources,
                        streams=streams,
                        attn_update=attn_update,
                        column_plan=column_plan,
                        valid_token_mask=valid_token_mask,
                        group_router_output=group_router_output,
                        depth_key_cache=depth_key_cache,
                        agg=agg,
                    )
                layers_aux.append(layer_aux)
                continue
            streams, agg, layer_aux = _maybe_checkpoint(
                layer.forward_column,
                router_noise=router_noise,
                block_sources=block_sources,
                streams=streams,
                column_plan=column_plan,
                base_attn_mask=base_attn_mask,
                valid_token_mask=valid_token_mask,
                group_router_output=group_router_output,
                pos=pos,
                dispatch_plan=dispatch_plan,
                depth_key_cache=depth_key_cache,
                agg=agg,
            )
            layers_aux.append(layer_aux)
        assert streams is not None
        return streams, agg, layers_aux

    def _column_expert_half_chunked(
        self,
        layer: nn.Module,
        *,
        chunks: int,
        router_noise: dict[str, Tensor] | None,
        block_sources: list[Tensor],
        streams: Float[Tensor, "batch_size seq_len slots dim"] | None,
        attn_update: Float[Tensor, "batch_size seq_len k dim"],
        column_plan: _ColumnStreamPlan,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
        agg: Float[Tensor, "batch_size seq_len dim"] | None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len slots dim"],
        Float[Tensor, "batch_size seq_len dim"] | None,
        dict[str, Any],
    ]:
        """One layer's expert half as ``chunks`` recompute regions over sequence slices.

        See :func:`_expert_half_chunks`. Every per-token input (the sources,
        the streams, the attention update, the slot map, the group routing,
        the pre-drawn router noise, the running aggregate) is sliced along the
        sequence, :meth:`RoutedTransformerLayer.column_expert_half` runs on
        each slice under its own :func:`_maybe_checkpoint`, and the slices'
        streams, aggregates and aux are concatenated back. The depth key
        cache is not threaded (a slice cannot match its sources by identity;
        checkpointing disables the cache anyway).

        Args:
            layer (nn.Module): The :class:`RoutedTransformerLayer`.
            chunks (int): Number of contiguous sequence slices (``>= 2``).
            router_noise (dict[str, Tensor] | None): Pre-drawn router noise
                for the whole sequence, ``[batch, seq, ...]`` per slab.
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``.
            streams (Float[Tensor, "batch_size seq_len slots dim"] | None):
                Streams at layer entry (None on the block's first layer).
            attn_update (Float[Tensor, "batch_size seq_len k dim"]): The
                attention half's output.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real-token
                mask.
            group_router_output (RouterOutput): Block-level group routing.
            agg (Float[Tensor, "batch_size seq_len dim"] | None): Running
                aggregate at entry (head-only reads).

        Returns:
            tuple[Float[Tensor, "batch_size seq_len slots dim"],
            Float[Tensor, "batch_size seq_len dim"] | None, dict[str, Any]]:
            Exactly what :meth:`RoutedTransformerLayer.column_expert_half`
            returns for the whole sequence.
        """

        if getattr(layer, "drop_path_rate", 0.0) > 0.0 and layer.training:
            raise RuntimeError(
                "NESTOR_EXPERT_HALF_CHUNKS needs drop-path 0: the per-sample "
                "mask would be drawn once per slice."
            )
        seq_len = valid_token_mask.shape[1]
        chunks = min(chunks, seq_len)
        out_streams: list[Tensor] = []
        out_agg: list[Tensor | None] = []
        out_aux: list[dict[str, Any]] = []
        for i in range(chunks):
            lo, hi = (i * seq_len) // chunks, ((i + 1) * seq_len) // chunks
            sl = slice(lo, hi)
            piece_streams, piece_agg, piece_aux = _maybe_checkpoint(
                layer.column_expert_half,
                router_noise=(
                    None
                    if router_noise is None
                    else {key: slab[:, sl] for key, slab in router_noise.items()}
                ),
                block_sources=[source[:, sl] for source in block_sources],
                streams=None if streams is None else streams[:, sl],
                attn_update=attn_update[:, sl],
                column_plan=_ColumnStreamPlan(column_plan.slot_of[:, sl], column_plan.num_slots),
                valid_token_mask=valid_token_mask[:, sl],
                group_router_output=RouterOutput(*(field[:, sl] for field in group_router_output)),
                depth_key_cache=None,
                agg=None if agg is None else agg[:, sl],
            )
            out_streams.append(piece_streams)
            out_agg.append(piece_agg)
            out_aux.append(piece_aux)
        new_streams = torch.cat(out_streams, dim=1)
        new_agg = None if out_agg[0] is None else torch.cat(out_agg, dim=1)  # type: ignore[arg-type]
        return new_streams, new_agg, _cat_aux_along_seq(out_aux)

    def _column_expert_half_lean_chunked(
        self,
        layer: nn.Module,
        *,
        chunks: int,
        router_noise: dict[str, Tensor] | None,
        block_sources: list[Tensor],
        running: Float[Tensor, "batch_size seq_len dim"] | None,
        attn_mean: Float[Tensor, "batch_size seq_len dim"],
        column_plan: _ColumnStreamPlan,
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_router_output: RouterOutput,
    ) -> tuple[Float[Tensor, "batch_size seq_len *k dim"], dict[str, Any]]:
        """:meth:`_column_expert_half_chunked` for the lean split (:func:`_activation_checkpoint_lean_enabled`).

        Slices every per-token input along the sequence, runs
        :meth:`RoutedTransformerLayer.column_expert_half_lean` on each slice
        under its own :func:`_maybe_checkpoint`, and concatenates the slices'
        expert updates and aux. The expert update is per token, so the
        concatenation is exactly the whole-sequence update.

        Args:
            layer (nn.Module): The :class:`RoutedTransformerLayer`.
            chunks (int): Number of contiguous sequence slices (``>= 2``).
            router_noise (dict[str, Tensor] | None): Pre-drawn router noise
                for the whole sequence, ``[batch, seq, ...]`` per slab.
            block_sources (list[Tensor]): Completed depth sources
                ``[batch, seq, dim]``.
            running (Float[Tensor, "batch_size seq_len dim"] | None): Running
                aggregate at layer entry (None on the block's first layer).
            attn_mean (Float[Tensor, "batch_size seq_len dim"]): The attention
                update's mean over slots.
            column_plan (_ColumnStreamPlan): Block-level column -> slot map.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Real-token
                mask.
            group_router_output (RouterOutput): Block-level group routing.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len *k dim"], dict[str, Any]]:
            Exactly what :meth:`RoutedTransformerLayer.column_expert_half_lean`
            returns for the whole sequence.
        """

        if getattr(layer, "drop_path_rate", 0.0) > 0.0 and layer.training:
            raise RuntimeError(
                "NESTOR_EXPERT_HALF_CHUNKS needs drop-path 0: the per-sample "
                "mask would be drawn once per slice."
            )
        seq_len = valid_token_mask.shape[1]
        chunks = min(chunks, seq_len)
        assembler = _SeqAssembler(seq_len)
        out_aux: list[dict[str, Any]] = []
        for i in range(chunks):
            lo, hi = (i * seq_len) // chunks, ((i + 1) * seq_len) // chunks
            sl = slice(lo, hi)
            piece_update, piece_aux = _maybe_checkpoint(
                layer.column_expert_half_lean,
                router_noise=(
                    None
                    if router_noise is None
                    else {key: slab[:, sl] for key, slab in router_noise.items()}
                ),
                block_sources=[source[:, sl] for source in block_sources],
                running=None if running is None else running[:, sl],
                attn_mean=attn_mean[:, sl],
                column_plan=_ColumnStreamPlan(column_plan.slot_of[:, sl], column_plan.num_slots),
                valid_token_mask=valid_token_mask[:, sl],
                group_router_output=RouterOutput(*(field[:, sl] for field in group_router_output)),
                depth_key_cache=None,
            )
            assembler.put(piece_update, lo, hi)
            del piece_update
            out_aux.append(piece_aux)
        return assembler.result(), _cat_aux_along_seq(out_aux)

    def _layer_dispatch_plan(
        self,
        group_router_output: RouterOutput,
        pos: Int[Tensor, "batch_size seq_len"],
    ) -> _AttentionDispatchPlan | _PackedAttentionPlan | None:
        """Build whichever block-level attention plan this block's layers use.

        Group membership is decided once per block, so anything derived from
        it is computed here and shared by every layer: the capacity-padded
        top-1 dispatch plan (:func:`_build_attention_dispatch_plan`) or the
        layer-packed varlen plan (:func:`_build_packed_attention_plan`). The
        two are mutually exclusive by construction (the former needs a
        non-causal model, the latter a causal one). Layers that use neither
        get None.

        Args:
            group_router_output (RouterOutput): The block's group routing.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.

        Returns:
            _AttentionDispatchPlan | _PackedAttentionPlan | None: The plan.
        """

        rope = self.layers[0].attention_groups[0].rope
        if self._attn_dispatch_plan:
            return _build_attention_dispatch_plan(group_router_output, pos, rope)
        if getattr(self.layers[0], "_attn_packed_varlen", False):
            return _build_packed_attention_plan(
                group_router_output, pos, rope, self.num_groups
            )
        return None

    def _layer_router_noise(
        self, layer: nn.Module, valid_mask: Bool[Tensor, "batch_size seq_len"]
    ) -> dict[str, Tensor] | None:
        """Pre-draw one layer's expert-router noise outside its checkpoint.

        Args:
            layer (nn.Module): The layer about to run (a
                :class:`RoutedTransformerLayer`).
            valid_mask (Bool[Tensor, "batch_size seq_len"]): Gives the batch
                shape and device.

        Returns:
            dict[str, Tensor] | None: See
            :meth:`RoutedTransformerLayer.draw_router_noise`.
        """

        draw = getattr(layer, "draw_router_noise", None)
        if draw is None or not layer.training:
            return None
        device = valid_mask.device
        dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else torch.get_default_dtype()
        )
        return draw(valid_mask.shape[0], valid_mask.shape[1], device, dtype)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len input_dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
        block_summaries: list[Tensor] | None = None,
        kimi_blocks: list[Tensor] | None = None,
        column_sources: list[Tensor] | None = None,
        depth_key_cache: DepthKeyCache | None = None,
    ) -> NestorBlockOutput:
        """Run one hierarchical block.

        Args:
            x (Float[Tensor, "batch_size seq_len input_dim"]): Input sequence.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            attn_mask (Bool[Tensor, "batch_size seq_len seq_len"]): Attention mask.
            block_summaries (list[Tensor] | None): Under
                ``residual_mode='attn_res_xblock'``, the global summaries
                ``[batch, summary_dim]`` of every preceding block; ignored
                otherwise.
            kimi_blocks (list[Tensor] | None): Under
                ``block_residual_mode='attn_res_kimi'``, the completed block
                representations ``[batch, seq, input_dim]`` in depth order
                (index 0 is the embedding); None otherwise. When set, every
                sublayer reads a depth-attention mixture of these sources plus
                the running partial sum, and the block's output is the partial
                sum of its sublayer updates (paper Eq. 3-6).
            column_sources (list[Tensor] | None): Under
                ``block_residual_mode='attn_res_column'``, the completed depth
                sources ``[batch, seq, input_dim]`` (the embedding followed by
                every earlier block's exports: its ``k_b`` column streams, and
                their aggregate under ``block_column_include_aggregate``); None
                otherwise. When set, the block runs
                :meth:`RoutedTransformerLayer.forward_column` and returns its
                ``k`` streams in ``NestorBlockOutput.streams``. Mutually
                exclusive with ``kimi_blocks``.
            depth_key_cache (DepthKeyCache | None): Under either depth-read
                mode, the backbone's cache of the unit-RMS keys of
                ``kimi_blocks`` / ``column_sources``. Those sources do not
                change while this block runs, and the key map is weight-free,
                so all ``2 * num_layers`` reads share one normalization per
                source instead of one each. None recomputes them per read.

        Returns:
            NestorBlockOutput: Pooled output sequence, aux data, and (for
            cross-block AttnRes) this block's global summary.
        """

        valid_mask = pos != PADDING_TOKEN_POS
        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        attn_mask = self._compose_attn_mask(valid_mask, attn_mask)
        if self._dense_noroute:
            group_router_output = _constant_router_output(
                x, valid_mask, self.num_groups
            )
        else:
            group_router_output = self.group_router(x, valid_mask)
            # Force the prepended global/register tokens into every group so
            # they attend to (and are attended by) the whole sequence across
            # all groups. No-op for single-group blocks. (_dense_noroute is
            # single-group only, so its constant output already covers them.)
            group_router_output = _force_global_tokens_all_groups(
                group_router_output, self._num_global, self.num_groups
            )

        layers_aux: list[dict[str, Any]] = []
        summary: Tensor | None = None
        streams: Tensor | None = None
        if column_sources is not None:
            # Column-wise residual streams: the block's group router (already
            # run above) fixes each token's k columns for the whole block, so
            # the column -> slot map and the dispatch plan are built ONCE and
            # reused by every layer. Each token then carries k parallel partial
            # sums instead of one; see
            # :meth:`RoutedTransformerLayer.forward_column`.
            assert kimi_blocks is None, (
                "column_sources and kimi_blocks are mutually exclusive."
            )
            column_plan = _build_column_stream_plan(
                group_router_output.topk_indices, self.num_groups
            )
            dispatch_plan = self._layer_dispatch_plan(group_router_output, pos)
            # Snapshot the backbone's depth-source list. It is a single list
            # object that Nestor.forward EXTENDS after each block, so by the
            # time a checkpointed layer recomputes, the live list would carry
            # later blocks' exports too. The copy is shallow and per block.
            block_sources = list(column_sources)
            if self._keeps_columns:
                agg: Tensor | None = None
                # Merged prefix: with block_column_trailing_layers = n, only
                # the last n layers track per-column streams. The prefix runs
                # the ordinary recurrence and hands its partial forward as the
                # aggregate, so the block's residual path is unchanged and only
                # the DEVIATIONS narrow to the trailing window.
                column_layers = list(self.layers)
                prefix: list = []
                if (
                    self._column_trailing is not None
                    and self._column_trailing < len(column_layers)
                ):
                    split = len(column_layers) - self._column_trailing
                    prefix, column_layers = (
                        column_layers[:split],
                        column_layers[split:],
                    )
                partial_prefix: Tensor | None = None
                for layer in prefix:
                    partial_prefix, layer_aux = _maybe_checkpoint(
                        layer.forward_kimi,
                        router_noise=self._layer_router_noise(layer, valid_mask),
                        kimi_blocks=block_sources,
                        partial=partial_prefix,
                        base_attn_mask=attn_mask,
                        valid_token_mask=valid_mask,
                        group_router_output=group_router_output,
                        pos=pos,
                        dispatch_plan=dispatch_plan,
                        depth_key_cache=depth_key_cache,
                    )
                    layers_aux.append(layer_aux)
                agg = partial_prefix
                if _activation_checkpoint_enabled() and _activation_checkpoint_block_enabled():
                    # One OUTER recompute region for the block's column layers
                    # (see _activation_checkpoint_block_enabled); the per-layer
                    # / per-half regions nest inside it. Every layer's router
                    # noise is drawn here, in layer order, before the region.
                    noises = [
                        self._layer_router_noise(layer, valid_mask)
                        for layer in column_layers
                    ]
                    streams, agg, column_aux = _maybe_checkpoint(
                        self._column_layers_region,
                        layer_indices=[
                            self._column_layer_index(layer) for layer in column_layers
                        ],
                        noises=noises,
                        block_sources=block_sources,
                        streams=streams,
                        agg=agg,
                        column_plan=column_plan,
                        base_attn_mask=attn_mask,
                        valid_token_mask=valid_mask,
                        group_router_output=group_router_output,
                        pos=pos,
                        dispatch_plan=dispatch_plan,
                        depth_key_cache=depth_key_cache,
                    )
                    layers_aux.extend(column_aux)
                else:
                    streams, agg, column_aux = self._column_layers_region(
                        layer_indices=[
                            self._column_layer_index(layer) for layer in column_layers
                        ],
                        noises=None,
                        block_sources=block_sources,
                        streams=streams,
                        agg=agg,
                        column_plan=column_plan,
                        base_attn_mask=attn_mask,
                        valid_token_mask=valid_mask,
                        group_router_output=group_router_output,
                        pos=pos,
                        dispatch_plan=dispatch_plan,
                        depth_key_cache=depth_key_cache,
                    )
                    layers_aux.extend(column_aux)
                assert streams is not None
                # The block emits k depth sources, not one tensor; ``x`` carries
                # their MEAN purely so the next block's group router (and the
                # validity masking / pooling below) still sees one token-shaped
                # representation. With k == 1 the mean is the stream itself, so
                # this matches the Kimi path exactly.
                # The block's own aggregate is the merged prefix plus the
                # tracked window; the deviations below are re-based on the
                # window's mean alone, so they still sum to exactly zero.
                window_mean = streams.mean(dim=2)
                x = (
                    window_mean
                    if partial_prefix is None
                    else partial_prefix + window_mean
                )
                if self._column_aggregate:
                    # Export that same aggregate as a further depth source, in
                    # the slot axis so the boundary bookkeeping below (exit
                    # masking, unbind into the backbone's source list) needs no
                    # special case. It is the block-wise residual
                    # ``attn_res_kimi`` would have exported, now sitting beside
                    # its k column-wise parts. Under deviation sources the k
                    # parts are re-based to ``s_j - a``: the raw set is linearly
                    # redundant (their sum is ``k * a``), so exporting
                    # deviations gives the reads an aggregate-plus-differences
                    # basis instead of k near-copies of the block direction.
                    if self._column_deviation:
                        streams = streams - window_mean.unsqueeze(2)
                    streams = torch.cat([streams, x.unsqueeze(2)], dim=2)
            else:
                # Dropped from ``block_column_blocks``: run the ordinary MERGED
                # recurrence (exactly what ``attn_res_kimi`` does) and export
                # ONE source. This block never allocates the per-layer
                # ``[batch, seq, k, dim]`` stream state at all, which is both
                # the residual step-time gap and the memory that makes column
                # mode the first arm to OOM at scale. Worth doing only where
                # the head read had learned to weight that block's columns near
                # zero -- see ``NestorConfig.block_column_blocks``.
                partial: Tensor | None = None
                for layer in self.layers:
                    partial, layer_aux = _maybe_checkpoint(
                        layer.forward_kimi,
                        router_noise=self._layer_router_noise(layer, valid_mask),
                        kimi_blocks=block_sources,
                        partial=partial,
                        base_attn_mask=attn_mask,
                        valid_token_mask=valid_mask,
                        group_router_output=group_router_output,
                        pos=pos,
                        dispatch_plan=dispatch_plan,
                        depth_key_cache=depth_key_cache,
                    )
                    layers_aux.append(layer_aux)
                assert partial is not None
                # One slot, so the shared block exit below is unchanged.
                x = partial
                streams = partial.unsqueeze(2)
        elif kimi_blocks is not None:
            # Kimi block AttnRes: the residual stream is replaced by the
            # running SUM of sublayer updates (``partial``); every sublayer's
            # input is a depth-attention read over the completed block
            # representations plus ``partial``. State is threaded functionally
            # through :meth:`RoutedTransformerLayer.forward_kimi` (no module-
            # attribute mutation), so dynamo traces the whole block as one
            # graph. Everything else (routers, masks, dispatch) is the
            # standard layer path.
            dispatch_plan = self._layer_dispatch_plan(group_router_output, pos)
            partial: Tensor | None = None
            # See the column branch: the backbone extends this list after each
            # block, so a recomputed layer must see the block-entry snapshot.
            block_sources = list(kimi_blocks)
            for layer in self.layers:
                partial, layer_aux = _maybe_checkpoint(
                    layer.forward_kimi,
                    router_noise=self._layer_router_noise(layer, valid_mask),
                    kimi_blocks=block_sources,
                    partial=partial,
                    base_attn_mask=attn_mask,
                    valid_token_mask=valid_mask,
                    group_router_output=group_router_output,
                    pos=pos,
                    dispatch_plan=dispatch_plan,
                    depth_key_cache=depth_key_cache,
                )
                layers_aux.append(layer_aux)
            assert partial is not None
            x = partial
        elif self._uses_attn_res:
            # AttnRes: thread the depth history (block input + each sublayer
            # output) in place of a single residual stream. For cross-block
            # AttnRes, prepend each preceding block's global summary (projected
            # into this block's width and broadcast over tokens) as extra depth
            # sources, then aggregate the full history into the block output.
            history = DepthHistory()
            if self.residual_mode == "attn_res_xblock" and block_summaries:
                assert self.xblock_consumes_summaries, (
                    "Block received summaries but was built with "
                    "xblock_consumes_summaries=False."
                )
                seq_len = x.shape[1]
                for prior in block_summaries:
                    source = self.summary_in_proj(prior)
                    history.append(source.unsqueeze(1).expand(-1, seq_len, -1))
            history.append(x)
            dispatch_plan = self._layer_dispatch_plan(group_router_output, pos)
            for layer in self.layers:
                history, layer_aux = _maybe_checkpoint(
                    layer.forward_attn_res,
                    router_noise=self._layer_router_noise(layer, valid_mask),
                    history=history,
                    base_attn_mask=attn_mask,
                    valid_token_mask=valid_mask,
                    group_router_output=group_router_output,
                    pos=pos,
                    dispatch_plan=dispatch_plan,
                )
                layers_aux.append(layer_aux)
            x = self.out_norm(
                depth_attention(history, self.out_query, self.out_key_norm)
            )
            if (
                self.residual_mode == "attn_res_xblock"
                and self.xblock_emits_summary
            ):
                summary = self._summarize_block(x, valid_mask)
        else:
            dispatch_plan = self._layer_dispatch_plan(group_router_output, pos)
            for layer in self.layers:
                x, layer_aux = _maybe_checkpoint(
                    layer,
                    router_noise=self._layer_router_noise(layer, valid_mask),
                    x=x,
                    base_attn_mask=attn_mask,
                    valid_token_mask=valid_mask,
                    group_router_output=group_router_output,
                    pos=pos,
                    dispatch_plan=dispatch_plan,
                )
                layers_aux.append(layer_aux)

        if self._num_global > 0:
            # Global/register tokens are not spatial -- carry them through the
            # block exit unchanged while pooling only the patch tail, then
            # re-prepend. The base attn mask is rebuilt all-true at the new
            # length (validity is re-composed on entry to the next block).
            g = self._num_global
            gx, px = x[:, :g], x[:, g:]
            gpos, ppos = pos[:, :g], pos[:, g:]
            px, ppos, _ = self._pool_tokens(
                px, ppos, attn_mask[:, g:, g:]
            )
            x = torch.cat([gx, px], dim=1)
            pos = torch.cat([gpos, ppos], dim=1)
            new_len = x.shape[1]
            attn_mask = torch.ones(
                (x.shape[0], new_len, new_len),
                dtype=torch.bool,
                device=x.device,
            )
        else:
            x, pos, attn_mask = self._pool_tokens(x, pos, attn_mask)
        x = self.output_proj(x)
        exit_mask = (pos != PADDING_TOKEN_POS).unsqueeze(-1)
        x = x * exit_mask.to(dtype=x.dtype)
        if streams is not None:
            # Column mode runs at constant token count with no boundary
            # projection (both enforced by the config checks), so the block
            # exit reduces to the same padding zero-out ``_pool_tokens`` /
            # ``output_proj`` apply to ``x``.
            streams = streams * exit_mask.unsqueeze(-1).to(dtype=streams.dtype)
        return NestorBlockOutput(
            x=x,
            pos=pos,
            attn_mask=attn_mask,
            aux={
                "group_router": _router_output_to_aux(group_router_output),
                "layers": layers_aux,
                "valid_token_mask": valid_mask,
            },
            summary=summary,
            streams=streams,
        )

    def forward_step(
        self,
        x: Float[Tensor, "batch_size input_dim"],
        pos: Int[Tensor, "batch_size"],
        cache: NestorBlockKVCache,
    ) -> tuple[
        Float[Tensor, "batch_size output_dim"],
        Int[Tensor, "batch_size"],
        NestorBlockKVCache,
        dict[str, Any],
    ]:
        """Run one incremental step through the block.

        Args:
            x (Float[Tensor, "batch_size input_dim"]): New token embedding.
            pos (Int[Tensor, "batch_size"]): New token position.
            cache (NestorBlockKVCache): Block cache.

        Returns:
            tuple[Float[Tensor, "batch_size output_dim"], Int[Tensor, "batch_size"],
            NestorBlockKVCache, dict[str, Any]]: Emitted token, emitted position,
            updated cache, and aux data.
        """

        if self._uses_attn_res:
            raise NotImplementedError(
                "Incremental forward_step (KV-cache decode) is not supported "
                f"under residual_mode='{self.residual_mode}'. AttnRes targets "
                "the non-causal image path; the depth-history decode schedule "
                "(paper Algorithm 1) is not implemented."
            )

        valid_mask = pos != PADDING_TOKEN_POS
        x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        group_router_output = _squeeze_router_output(
            self.group_router(x.unsqueeze(1), valid_mask.unsqueeze(1))
        )

        next_layer_caches: list[RoutedTransformerLayerKVCache] = []
        layers_aux: list[dict[str, Any]] = []
        for layer, layer_cache in zip(self.layers, cache.layer_caches):
            x, next_layer_cache, layer_aux = layer.forward_step(
                x=x,
                valid_token_mask=valid_mask,
                group_router_output=group_router_output,
                cache=layer_cache,
                pos=pos,
            )
            next_layer_caches.append(next_layer_cache)
            layers_aux.append(layer_aux)

        pooled_x, pooled_pos, next_pooling_cache = self._pool_token_step(
            x,
            pos,
            cache.pooling_cache,
        )
        next_x = self.output_proj(pooled_x)
        next_x = next_x * (pooled_pos != PADDING_TOKEN_POS).unsqueeze(-1).to(
            dtype=next_x.dtype
        )
        return (
            next_x,
            pooled_pos,
            NestorBlockKVCache(
                layer_caches=next_layer_caches,
                pooling_cache=next_pooling_cache,
            ),
            {
                "group_router": _router_output_to_aux(group_router_output),
                "layers": layers_aux,
                "valid_token_mask": valid_mask,
            },
        )

    @torch.no_grad()
    def update_detached_parameters(self, aux: dict[str, Any]) -> None:
        """Apply detached router updates from the merged aux data."""

        group_router_aux = aux.get("group_router")
        if isinstance(group_router_aux, dict):
            topk_indices = group_router_aux.get("topk_indices")
            valid_token_mask = group_router_aux.get("valid_token_mask")
            if topk_indices is not None and valid_token_mask is not None:
                self.group_router.update_detached_parameters(
                    topk_indices,
                    valid_token_mask,
                )

        for layer, layer_aux in zip(self.layers, aux.get("layers", [])):
            if isinstance(layer_aux, dict):
                layer.update_detached_parameters(
                    cast(dict[str, Any], layer_aux)
                )

    def to_config(self) -> dict[str, Any]:
        """Serialize the block back into a dictionary config."""

        first_layer = self.layers[0]
        first_group = first_layer.attention_groups[0]
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "pooling_stride": self.pooling_stride,
            "block_num_layers": len(self.layers),
            "num_groups": self.num_groups,
            "num_groups_per_token": self.num_groups_per_token,
            "block_num_heads": first_group.num_heads,
            "block_head_dim": first_group.head_dim,
            "block_rope_base": first_group.rope.base,
            "block_mlp_hidden_dims": self.block_mlp_hidden_dims,
            "block_mlp_activation": self.block_mlp_activation,
            "router_num_experts_per_group": self.num_experts_per_group,
            "layer_num_experts": list(self.layer_num_experts),
            "router_num_shared_experts": self.num_shared_experts,
            "router_num_dense_shared_experts": self.num_dense_shared_experts,
            "router_num_experts_per_group_per_token": (
                self.num_experts_per_group_per_token
            ),
            "router_score_mlp_hidden_dims": self.router_score_mlp_hidden_dims,
            "router_score_mlp_activation": self.router_score_mlp_activation,
            "router_noise_mlp_hidden_dims": self.router_noise_mlp_hidden_dims,
            "router_noise_mlp_activation": self.router_noise_mlp_activation,
            "loss_free_balancing": self.loss_free_balancing,
            "bias_update_rate": self.bias_update_rate,
            "attention_token_dispatch": self.attention_token_dispatch,
            "expert_pool_public_experts": self.expert_pool_public_experts,
            "expert_pool_public_top_k": self.expert_pool_public_top_k,
            "expert_pool_private_experts": self.expert_pool_private_experts,
            "expert_pool_private_top_k": self.expert_pool_private_top_k,
            "expert_pool_active_budget": self.expert_pool_active_budget,
            "expert_pool_public_budget": self.expert_pool_public_budget,
            "expert_pool_private_gate": self.expert_pool_private_gate,
        }

    @staticmethod
    def from_config(
        config: dict[str, Any],
        *,
        depth_scale: float,
        causal: bool = False,
        residual_mode: str = "convex",
        block_residual_mode: str = "projection",
        kimi_max_sources: int | None = None,
        column_include_aggregate: bool = False,
        column_head_only: bool = False,
        column_deviation_sources: bool = False,
        keeps_columns: bool = True,
        column_trailing_layers: int | None = None,
        drop_path_rate: float = 0.0,
        summary_dim: int | None = None,
        xblock_consumes_summaries: bool = True,
        xblock_emits_summary: bool = True,
        num_global_tokens: int = 0,
        pooling_mode: str = "mean",
        apply_output_projection: bool = True,
    ) -> "NestorBlock":
        """Construct a block from a dictionary config.

        Args:
            config (dict[str, Any]): One expanded block dictionary.
            depth_scale (float): Convex residual weight ``1/D`` for the model.
            causal (bool): Whether to enforce prefix-stable execution.
            residual_mode (str): ``"convex"`` (default), ``"attn_res"`` (intra-
                block Attention Residuals), or ``"attn_res_xblock"`` (additionally
                attend over preceding blocks' global summaries).
            summary_dim (int | None): Canonical width of cross-block summaries
                (the model input width); only used for ``"attn_res_xblock"``.
            block_residual_mode (str): Between-block transition mode of the
                owning backbone; ``"attn_res_kimi"`` and ``"attn_res_column"``
                build the per-sublayer depth reads.
            kimi_max_sources (int | None): Largest depth-read source count
                (``num_blocks + 1``, or ``1 + sum_b k_b`` for
                ``"attn_res_column"``, one more per multi-column block under
                ``column_include_aggregate``); required under both depth-read
                modes.
            column_include_aggregate (bool): See :class:`NestorBlock`; adds the
                aggregated block-wise residual to the column-wise ones.
            xblock_consumes_summaries (bool): See :class:`NestorBlock`; False
                for the model's first block.
            xblock_emits_summary (bool): See :class:`NestorBlock`; False for
                the model's last block.
            apply_output_projection (bool): Whether the block owns its legacy
                boundary projection. Block AttnRes disables it.
        """

        return NestorBlock(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            pooling_stride=config["pooling_stride"],
            pooling_keep_ratio=config.get("pooling_keep_ratio"),
            block_num_layers=config["block_num_layers"],
            num_groups=config["num_groups"],
            num_groups_per_token=config["num_groups_per_token"],
            router_num_experts_per_group=config["router_num_experts_per_group"],
            layer_num_experts=config.get("layer_num_experts"),
            router_num_shared_experts=config["router_num_shared_experts"],
            router_num_dense_shared_experts=config.get(
                "router_num_dense_shared_experts", 0
            ),
            router_num_experts_per_group_per_token=config[
                "router_num_experts_per_group_per_token"
            ],
            block_num_heads=config["block_num_heads"],
            block_head_dim=config["block_head_dim"],
            block_mlp_hidden_dims=list(config["block_mlp_hidden_dims"]),
            block_mlp_activation=config["block_mlp_activation"],
            router_score_mlp_hidden_dims=list(
                config["router_score_mlp_hidden_dims"]
            ),
            router_score_mlp_activation=config["router_score_mlp_activation"],
            router_noise_mlp_hidden_dims=list(
                config["router_noise_mlp_hidden_dims"]
            ),
            router_noise_mlp_activation=config["router_noise_mlp_activation"],
            rope_base=config["block_rope_base"],
            depth_scale=depth_scale,
            causal=causal,
            loss_free_balancing=config["loss_free_balancing"],
            bias_update_rate=config["bias_update_rate"],
            attention_token_dispatch=config.get(
                "attention_token_dispatch", False
            ),
            expert_pool_public_experts=config.get(
                "expert_pool_public_experts", 0
            ),
            expert_pool_public_top_k=config.get(
                "expert_pool_public_top_k", 1
            ),
            expert_pool_private_experts=config.get(
                "expert_pool_private_experts", 0
            ),
            expert_pool_active_budget=config.get(
                "expert_pool_active_budget", None
            ),
            expert_pool_public_budget=config.get(
                "expert_pool_public_budget", None
            ),
            expert_pool_private_top_k=config.get(
                "expert_pool_private_top_k", 1
            ),
            expert_pool_private_gate=config.get(
                "expert_pool_private_gate", "renorm"
            ),
            residual_mode=residual_mode,
            block_residual_mode=block_residual_mode,
            kimi_max_sources=kimi_max_sources,
            column_include_aggregate=column_include_aggregate,
            column_head_only=column_head_only,
            column_deviation_sources=column_deviation_sources,
            keeps_columns=keeps_columns,
            column_trailing_layers=column_trailing_layers,
            drop_path_rate=drop_path_rate,
            summary_dim=summary_dim,
            xblock_consumes_summaries=xblock_consumes_summaries,
            xblock_emits_summary=xblock_emits_summary,
            num_global_tokens=num_global_tokens,
            pooling_mode=pooling_mode,
            apply_output_projection=apply_output_projection,
        )


# ---------------------------------------------------------------------------
# Top-level Nestor backbone.
# ---------------------------------------------------------------------------


@register_backbone("nestor")
class Nestor(Backbone):
    """Hierarchical routed Nestor backbone.

    The backbone is a stack of hierarchical blocks. Each block computes a
    group router once, reuses those group assignments across all layers in the
    block, then either locally pools/projects into the next token space or uses
    per-token attention over all earlier block outputs at the boundary.

    Attributes:
        config (NestorConfig): Backbone configuration.
        causal (bool): Whether the backbone runs prefix-stable/causal.
        blocks (list[NestorBlock]): The hierarchical blocks in depth order.
        block_attention_residuals (list[BlockAttentionResidual]): Per-block
            boundary reads under ``block_residual_mode='attn_res'``; empty
            otherwise.
        depth_key_cache_enabled (bool): Whether the depth-read modes share one
            :class:`DepthKeyCache` across a forward (exact, and much cheaper).
    """

    config: NestorConfig
    causal: bool
    blocks: list[NestorBlock]
    block_attention_residuals: list[BlockAttentionResidual]
    depth_key_cache_enabled: bool

    def __init__(self, config: NestorConfig) -> None:
        """Initialize Nestor from a config.

        Args:
            config (NestorConfig): Backbone configuration.
        """

        super().__init__()
        self.config = config

        expanded_config = config.to_dict()
        self.causal = expanded_config["causal"]
        residual_mode = expanded_config.get("residual_mode", "convex")
        self.block_residual_mode = expanded_config.get(
            "block_residual_mode", "projection"
        )
        self.block_attn_res_include_embedding = expanded_config.get(
            "block_attn_res_include_embedding", True
        )
        self.block_column_include_aggregate = expanded_config.get(
            "block_column_include_aggregate", False
        )
        self.block_column_head_only = expanded_config.get(
            "block_column_head_only", False
        )
        self.block_column_head_query_per_source = expanded_config.get(
            "block_column_head_query_per_source", False
        )
        self.block_column_deviation_sources = expanded_config.get(
            "block_column_deviation_sources", False
        )
        self.block_column_blocks = expanded_config.get(
            "block_column_blocks", None
        )
        self.block_column_trailing_layers = expanded_config.get(
            "block_column_trailing_layers", None
        )
        self._column_block_set: set[int] | None = (
            None
            if self.block_column_blocks is None
            else set(self.block_column_blocks)
        )
        block_configs = expanded_config["block_configs"]
        # The residual stream is D = 2 * sum_l M_l branches deep (attention and
        # feed-forward per layer); the convex residual weight is 1/D for every
        # branch, so no learning rate carries a depth factor. Under
        # residual_mode="attn_res" the convex residual is replaced by depth
        # attention, so depth_scale is unused but harmless.
        total_layers = sum(bc["block_num_layers"] for bc in block_configs)
        depth_scale = 1.0 / (2 * total_layers)
        # Largest depth-read source count. The Kimi mode exports ONE tensor per
        # block (plus the embedding b_0); the column mode exports one per
        # routed column, so block b contributes k_b = num_groups_per_token[b],
        # plus their aggregate under block_column_include_aggregate (which
        # single-column blocks skip -- their aggregate is their one stream),
        # and blocks outside block_column_blocks contribute one.
        if self.block_residual_mode == "attn_res_column":
            # A block dropped from block_column_blocks runs the merged
            # recurrence and contributes exactly ONE source, like a k == 1
            # block does.
            keeps = self._block_keeps_columns
            self.max_depth_sources = 1 + sum(
                (
                    bc["num_groups_per_token"]
                    + int(
                        self.block_column_include_aggregate
                        and bc["num_groups_per_token"] > 1
                    )
                )
                if keeps(idx)
                else 1
                for idx, bc in enumerate(block_configs)
            )
        else:
            self.max_depth_sources = len(block_configs) + 1
        # Global/register tokens: a small set of learned tokens prepended to the
        # sequence (Nestor.forward), members of every attention group so they
        # carry an all-to-all channel on top of the cheap local grouping. 0 off.
        self.num_global = int(getattr(config, "num_global_tokens", 0))
        if self.num_global > 0:
            if self.causal:
                raise ValueError(
                    "num_global_tokens is non-causal only (prepended global "
                    "tokens would break prefix-stable causal execution)."
                )
            self.global_tokens = nn.Parameter(
                torch.zeros(1, self.num_global, config.input_dim)
            )
            nn.init.trunc_normal_(self.global_tokens, std=0.02)
        self.blocks = cast(
            list[NestorBlock],
            nn.ModuleList(
                [
                    NestorBlock.from_config(
                        block_config,
                        depth_scale=depth_scale,
                        causal=self.causal,
                        residual_mode=residual_mode,
                        block_residual_mode=self.block_residual_mode,
                        kimi_max_sources=self.max_depth_sources,
                        column_include_aggregate=(
                            self.block_column_include_aggregate
                        ),
                        column_head_only=self.block_column_head_only,
                        column_deviation_sources=(
                            self.block_column_deviation_sources
                        ),
                        keeps_columns=self._block_keeps_columns(block_idx),
                        column_trailing_layers=(
                            self.block_column_trailing_layers
                        ),
                        drop_path_rate=config.drop_path_rate,
                        summary_dim=config.input_dim,
                        # First block has no summaries to read; last block's
                        # summary would have no reader (unused params kill DDP).
                        xblock_consumes_summaries=block_idx > 0,
                        xblock_emits_summary=(
                            block_idx < len(block_configs) - 1
                        ),
                        num_global_tokens=self.num_global,
                        pooling_mode=getattr(config, "pooling_mode", "mean"),
                        apply_output_projection=(
                            self.block_residual_mode == "projection"
                        ),
                    )
                    for block_idx, block_config in enumerate(block_configs)
                ]
            ),
        )
        if self.block_residual_mode == "attn_res":
            # Block l emits at the width it received. Residual l maps all raw
            # outputs through l, plus optional O^(0), into feature_dims[l].
            block_output_widths = [config.input_dim, *config.feature_dims[:-1]]
            self.block_attention_residuals = cast(
                list[BlockAttentionResidual],
                nn.ModuleList(
                    [
                        BlockAttentionResidual(
                            source_widths=(
                                (
                                    [config.input_dim]
                                    if self.block_attn_res_include_embedding
                                    else []
                                )
                                + block_output_widths[: block_idx + 1]
                            ),
                            target_width=config.feature_dims[block_idx],
                        )
                        for block_idx in range(config.num_blocks)
                    ]
                ),
            )
        else:
            self.block_attention_residuals = cast(
                list[BlockAttentionResidual], nn.ModuleList()
            )
        if self.block_residual_mode in ("attn_res_kimi", "attn_res_column"):
            # Final head read (paper Fig. 2): a dedicated zero-init pseudo-
            # query attends over ALL completed block representations
            # (including the embedding b_0) to produce the head input. Under
            # ``attn_res_column`` those representations are the individual
            # column streams, so the head can prefer one column of one block
            # over another instead of only their block-level mixture -- and
            # under ``block_column_include_aggregate`` it keeps that mixture as
            # a source too, so it can address either.
            self.final_depth_read = DepthAttentionRead(
                config.output_dim,
                max_sources=self.max_depth_sources,
                per_source_queries=self.block_column_head_query_per_source,
            )
        mass_lambda = config.block_column_mass_lambda
        if self.block_residual_mode == "attn_res_column" and (
            config.block_column_gain_init != 1.0 or mass_lambda is not None
        ):
            # Depth-slot semantics are POSITIONAL and identical for every read
            # of one backbone: slot 0 is the embedding, then each block's k
            # columns followed (multi-column blocks, under the aggregate flag)
            # by their aggregate -- and a block's live streams occupy exactly
            # the positions its exports will take. So one gain vector serves
            # all reads, each slicing its prefix. The embedding and
            # single-column blocks (whose one stream IS the merged direction)
            # always keep 1.0. A multi-column block's slots get either the
            # constant column init (aggregate stays at 1.0), or the
            # mass-conserving split (columns ``lambda / k``, aggregate
            # ``1 - lambda``) that keeps every block's total init mass at
            # exactly 1 regardless of its k.
            gain_init = torch.ones(self.max_depth_sources)
            slot = 1  # skip the embedding
            for block_idx, bc in enumerate(block_configs):
                if not self._block_keeps_columns(block_idx):
                    # Merged recurrence: one slot, and it IS the block
                    # direction, so it keeps gain 1.0 like an aggregate.
                    slot += 1
                    continue
                top_k = bc["num_groups_per_token"]
                if top_k > 1:
                    gain_init[slot:slot + top_k] = (
                        mass_lambda / top_k
                        if mass_lambda is not None
                        else config.block_column_gain_init
                    )
                    slot += top_k
                    if self.block_column_include_aggregate:
                        if mass_lambda is not None:
                            gain_init[slot] = 1.0 - mass_lambda
                        slot += 1
                else:
                    slot += 1  # single stream == merged direction, keeps 1.0
            assert slot == self.max_depth_sources, (
                f"gain-init layout covered {slot} slots, expected "
                f"{self.max_depth_sources}."
            )
            # Under head-only columns the sublayer reads use the NARROW
            # (kimi-shaped) source layout, where slot 1 is an aggregate, not a
            # column -- applying the full-layout vector there would zero the
            # wrong slots. Their unit init IS the kimi init; only the head
            # read carries the column layout.
            targets = (
                [self.final_depth_read]
                if self.block_column_head_only
                else [
                    module
                    for module in self.modules()
                    if isinstance(module, DepthAttentionRead)
                ]
            )
            with torch.no_grad():
                for module in targets:
                    module.value_gains.copy_(gain_init)
        import os as _os

        # Share the weight-free key SCALE of the completed depth sources across
        # every read of a forward (see DepthKeyCache): exact, and the column
        # mode's dominant read cost now that the read itself never materializes
        # a key. ``NESTOR_NO_DEPTH_KEY_CACHE=1`` falls back to per-read
        # reduction (debug / A-B only).
        self.depth_key_cache_enabled = (
            _os.environ.get("NESTOR_NO_DEPTH_KEY_CACHE") != "1"
        )
        self.final_norm: nn.Module = nn.Identity()

    @property
    def output_dim(self) -> int:
        """Return the final block width."""

        return self.config.output_dim

    def fsdp2_wrap_order(self) -> list[nn.Module]:
        """Shard each block before the backbone root."""

        return [*self.blocks, *self.block_attention_residuals, self]

    def init_kv_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NestorKVCache:
        """Initialize the hierarchical KV cache for causal decoding.

        Args:
            batch_size (int): Batch size for streaming inference.
            device (torch.device | None): Optional device override.
            dtype (torch.dtype | None): Optional dtype override.

        Returns:
            NestorKVCache: Empty hierarchical cache.
        """

        if not self.causal:
            raise ValueError(
                "KV cache is only available for causal Nestor models."
            )

        parameter = next(self.parameters())
        cache_device = device if device is not None else parameter.device
        cache_dtype = dtype if dtype is not None else parameter.dtype
        return NestorKVCache(
            block_caches=[
                block.init_kv_cache(
                    batch_size, device=cache_device, dtype=cache_dtype
                )
                for block in self.blocks
            ]
        )


    def _block_keeps_columns(self, block_idx: int) -> bool:
        """Whether block ``block_idx`` maintains per-column residual streams.

        Under ``block_residual_mode='attn_res_column'`` every block keeps them
        by default. ``block_column_blocks`` narrows that to a chosen subset;
        the rest run the merged (kimi-style) recurrence with one stream and
        export one source, which costs a block nothing when the head read had
        learned to weight its columns near zero anyway.

        Args:
            block_idx (int): Index of the block in depth order.

        Returns:
            bool: True when the block stays column-wise.
        """

        if self._column_block_set is None:
            return True
        return block_idx in self._column_block_set

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len input_dim"],
        pos: Int[Tensor, "batch_size seq_len"],
        attn_mask: Bool[Tensor, "batch_size seq_len seq_len"] | None,
    ) -> ModelOutput:
        """Run the hierarchical backbone on a token sequence.

        Args:
            x (Float[Tensor, "batch_size seq_len input_dim"]): Token embeddings.
            pos (Int[Tensor, "batch_size seq_len"]): Token positions.
            attn_mask (Bool[Tensor, "batch_size seq_len seq_len"] | None):
                Attention mask. ``None`` is the trivial-mask contract: every
                token is valid and the only pair structure is the backbone's
                own causality, applied analytically by the maskless attention
                paths (no O(S^2) tensor is ever built -- required for
                long-context runs, where a dense mask is 16+ GiB).

        Returns:
            ModelOutput: Final token sequence in ``pred`` plus per-block aux data.
        """

        # Prepend learned global/register tokens. They take positions
        # 0..num_global-1 (patches shift up by num_global; RoPE is relative, so
        # a uniform patch shift is harmless) and an all-true base mask. Each
        # block forces them into every attention group; they bypass pooling and
        # are stripped before the head below.
        if self.num_global > 0:
            batch_size = x.shape[0]
            globals_x = self.global_tokens.to(dtype=x.dtype).expand(
                batch_size, -1, -1
            )
            x = torch.cat([globals_x, x], dim=1)
            global_pos = (
                torch.arange(self.num_global, device=pos.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            shifted = torch.where(
                pos == PADDING_TOKEN_POS, pos, pos + self.num_global
            )
            pos = torch.cat([global_pos, shifted], dim=1)
            new_len = x.shape[1]
            attn_mask = torch.ones(
                (batch_size, new_len, new_len),
                dtype=torch.bool,
                device=x.device,
            )

        embedding_output = x
        block_aux: list[dict[str, Any]] = []
        # Cross-block AttnRes (residual_mode='attn_res_xblock') threads each
        # block's global summary forward; stays empty for the other modes.
        block_summaries: list[Tensor] = []
        block_outputs: list[Tensor] = []
        # Kimi block AttnRes: completed block representations in depth order,
        # starting from the embedding b_0; None for the other modes.
        kimi_blocks: list[Tensor] | None = (
            [embedding_output]
            if self.block_residual_mode == "attn_res_kimi"
            else None
        )
        # Column-wise residual streams: same source list, except each block
        # contributes its k column streams instead of one merged output (plus
        # the merged one under block_column_include_aggregate).
        column_sources: list[Tensor] | None = (
            [embedding_output]
            if self.block_residual_mode == "attn_res_column"
            else None
        )
        # Head-only columns: the sublayer reads consume THIS narrow list
        # (embedding + one aggregate per completed block, kimi's shape) while
        # the full ``column_sources`` above feeds only the final head read.
        narrow_sources: list[Tensor] | None = (
            [embedding_output]
            if column_sources is not None and self.block_column_head_only
            else None
        )
        # Both depth-read modes re-read the completed sources from every
        # sublayer of every block (33 reads over 369 source-slots for the
        # (3, 5, 7, 1) column ladder), and the key map is weight-free, so one
        # shared cache normalizes each source exactly once per forward and
        # retains one key for the backward instead of one per read. The cache
        # is grown in lockstep with the source list below; the reads check by
        # identity that the two agree.
        depth_source_list = (
            kimi_blocks
            if kimi_blocks is not None
            else (narrow_sources if narrow_sources is not None else column_sources)
        )
        # Activation checkpointing forces the cache off: its source list grows
        # between blocks, so a recomputed early block would be handed later
        # blocks' sources and the read's identity check would reject them. The
        # cache only saves recomputing a per-token scalar, which is the trade
        # checkpointing is already making. Decided per forward because the
        # checkpointing flag is read per layer call, not fixed at construction.
        depth_key_cache = (
            DepthKeyCache(depth_source_list, eps=self.final_depth_read.eps)
            if depth_source_list is not None
            and self.depth_key_cache_enabled
            and not _activation_checkpoint_enabled()
            else None
        )
        for block_idx, block in enumerate(self.blocks):
            block_output = block(
                x,
                pos,
                attn_mask,
                block_summaries=block_summaries,
                kimi_blocks=kimi_blocks,
                column_sources=(
                    narrow_sources
                    if narrow_sources is not None
                    else column_sources
                ),
                depth_key_cache=depth_key_cache,
            )
            x = block_output.x
            pos = block_output.pos
            attn_mask = block_output.attn_mask
            block_aux.append(block_output.aux)
            if block_output.summary is not None:
                block_summaries = block_summaries + [block_output.summary]
            if self.block_residual_mode == "attn_res":
                block_outputs.append(x)
                sources = (
                    [embedding_output, *block_outputs]
                    if self.block_attn_res_include_embedding
                    else block_outputs
                )
                x = self.block_attention_residuals[block_idx](sources)
            elif kimi_blocks is not None:
                kimi_blocks.append(x)
                if depth_key_cache is not None:
                    depth_key_cache.extend([x])
            elif column_sources is not None:
                assert block_output.streams is not None
                # Unbind ONCE: the cache identifies its sources by object
                # identity, and a second unbind would make new views.
                #
                # Passing the contiguous slots as ONE grouped entry instead --
                # which DepthAttentionRead supports and which is ~4x cheaper per
                # source when a single group spans all 40 -- measured WORSE in
                # production (the head-only read's overhead rose from +6.80%
                # to +9.06%). Sources here do not form one wide group: they
                # arrive as the embedding plus six per-block groups of width
                # 4-9, so batching spans only ~6 sources while the grouped path
                # adds an out-of-place [batch, seq, dim] accumulate per group
                # that this in-place addcmul_ chain avoids. At those widths the
                # overhead exceeds the batching win.
                exported = block_output.streams.unbind(dim=2)
                column_sources.extend(exported)
                if narrow_sources is not None:
                    # Head-only: sublayer reads see one source per block --
                    # the aggregate, exported as the LAST slot (a k == 1
                    # block's only slot IS its aggregate). The cache serves
                    # those reads, so it grows with the narrow list.
                    narrow_sources.append(exported[-1])
                    if depth_key_cache is not None:
                        depth_key_cache.extend([exported[-1]])
                elif depth_key_cache is not None:
                    depth_key_cache.extend(exported)

        if kimi_blocks is not None:
            # Final head input: dedicated depth read over all completed block
            # representations (paper Fig. 2), then the usual final_norm/head.
            # Every source is cached by now, so this read normalizes nothing.
            x = self.final_depth_read(kimi_blocks, key_cache=depth_key_cache)
        elif column_sources is not None:
            # Same read, but over every block's individual column streams.
            # Under head-only columns the cache tracks the NARROW list, so the
            # one wide read runs uncached -- its ~40 scale reductions are
            # trivial next to the 32 sublayer reads the cache does serve.
            x = self.final_depth_read(
                column_sources,
                key_cache=(
                    depth_key_cache if narrow_sources is None else None
                ),
            )

        # Strip the global/register tokens before the head -- they are a pure
        # all-to-all communication channel, not classification features, so the
        # head pools only the (pooled) patch tokens, as in every non-global run.
        if self.num_global > 0:
            x = x[:, self.num_global:]
            pos = pos[:, self.num_global:]

        x = self.final_norm(x)

        return ModelOutput(
            pred=x,
            aux={
                "blocks": block_aux,
                "hidden_states": x,
                "positions": pos,
            },
        )

    def forward_step(
        self,
        x: Float[Tensor, "batch_size input_dim"],
        pos: Int[Tensor, "batch_size"],
        cache: NestorKVCache,
    ) -> tuple[ModelOutput, NestorKVCache]:
        """Run one incremental causal step through the hierarchy.

        Args:
            x (Float[Tensor, "batch_size input_dim"]): New token embedding.
            pos (Int[Tensor, "batch_size"]): New token position.
            cache (NestorKVCache): Existing hierarchical cache.

        Returns:
            tuple[ModelOutput, NestorKVCache]: One-step output token and updated
            cache.
        """

        if not self.causal:
            raise ValueError(
                "forward_step is only available for causal Nestor."
            )
        if self.block_residual_mode in ("attn_res_kimi", "attn_res_column"):
            # The depth-history decode schedule (paper Algorithm 1, two-phase
            # online-softmax merge) is not implemented.
            raise NotImplementedError(
                f"{self.block_residual_mode} supports training/full-sequence "
                "forward only."
            )

        block_aux: list[dict[str, Any]] = []
        embedding_output = x
        block_outputs: list[Tensor] = []
        current_x = x
        current_pos = pos
        next_block_caches = []
        for block_idx, (block, block_cache) in enumerate(
            zip(self.blocks, cache.block_caches)
        ):
            current_x, current_pos, next_block_cache, aux = block.forward_step(
                current_x,
                current_pos,
                block_cache,
            )
            next_block_caches.append(next_block_cache)
            block_aux.append(aux)
            if self.block_residual_mode == "attn_res":
                block_outputs.append(current_x)
                sources = (
                    [embedding_output, *block_outputs]
                    if self.block_attn_res_include_embedding
                    else block_outputs
                )
                current_x = self.block_attention_residuals[block_idx](sources)

        output_valid = (current_pos != PADDING_TOKEN_POS).unsqueeze(1)
        current_tokens = self.final_norm(current_x).unsqueeze(1)
        current_positions = current_pos.unsqueeze(1)
        return ModelOutput(
            pred=current_tokens,
            aux={
                "blocks": block_aux,
                "hidden_states": current_tokens,
                "positions": current_positions,
                "output_valid": output_valid,
            },
        ), NestorKVCache(block_caches=next_block_caches)

    def finalize_kv_cache(
        self,
        cache: NestorKVCache,
    ) -> tuple[ModelOutput, NestorKVCache]:
        """Flush trailing partial pooling windows after streaming inference.

        Args:
            cache (NestorKVCache): Hierarchical cache to flush.

        Returns:
            tuple[ModelOutput, NestorKVCache]: Remaining emitted tokens and the
            reset cache state.
        """

        if not self.causal:
            raise ValueError(
                "finalize_kv_cache is only available for causal Nestor."
            )

        first_group_cache = (
            cache.block_caches[0].layer_caches[0].group_caches[0]
        )
        batch_size = first_group_cache.keys.shape[0]
        device = first_group_cache.keys.device
        dtype = first_group_cache.keys.dtype
        if self.block_residual_mode == "attn_res":
            # Block AttnRes requires stride-one transitions, so forward_step
            # emits every valid token immediately and no pooling tail exists.
            empty = torch.zeros(
                (batch_size, 0, self.output_dim),
                dtype=dtype,
                device=device,
            )
            positions = torch.zeros(
                (batch_size, 0), dtype=torch.long, device=device
            )
            return ModelOutput(
                pred=empty,
                aux={
                    "hidden_states": empty,
                    "positions": positions,
                    "output_valid": positions != PADDING_TOKEN_POS,
                },
            ), cache
        next_cache = cache
        emitted_tokens: list[Tensor] = []
        emitted_positions: list[Tensor] = []
        while True:
            progress = False
            current_x = torch.zeros(
                (batch_size, self.config.input_dim),
                dtype=dtype,
                device=device,
            )
            current_pos = torch.full(
                (batch_size,),
                fill_value=PADDING_TOKEN_POS,
                dtype=torch.long,
                device=device,
            )
            updated_block_caches = []
            for block_idx, block in enumerate(self.blocks):
                if (current_pos != PADDING_TOKEN_POS).any():
                    current_x, current_pos, block_cache, _ = block.forward_step(
                        current_x,
                        current_pos,
                        next_cache.block_caches[block_idx],
                    )
                else:
                    current_x, current_pos, block_cache = block.flush(
                        next_cache.block_caches[block_idx]
                    )
                updated_block_caches.append(block_cache)
                if (current_pos != PADDING_TOKEN_POS).any():
                    progress = True

            if not progress:
                break

            emitted_tokens.append(current_x)
            emitted_positions.append(current_pos)
            next_cache = NestorKVCache(block_caches=updated_block_caches)

        if emitted_tokens:
            pred = self.final_norm(torch.stack(emitted_tokens, dim=1))
            positions = torch.stack(emitted_positions, dim=1)
        else:
            pred = torch.zeros(
                (batch_size, 0, self.output_dim),
                dtype=dtype,
                device=device,
            )
            positions = torch.zeros(
                (batch_size, 0), dtype=torch.long, device=device
            )
        return ModelOutput(
            pred=pred,
            aux={
                "hidden_states": pred,
                "positions": positions,
                "output_valid": positions != PADDING_TOKEN_POS,
            },
        ), NestorKVCache(block_caches=next_cache.block_caches)

    @torch.no_grad()
    def update_detached_parameters(self, aux: dict[str, object]) -> None:
        """Apply detached router updates after an optimizer step.

        Args:
            aux (dict[str, object]): Backbone auxiliary outputs for the current
                optimizer step.
        """

        blocks_aux = aux.get("blocks", [])
        if not isinstance(blocks_aux, list):
            return
        for block, block_aux in zip(self.blocks, blocks_aux):
            if isinstance(block_aux, dict):
                block.update_detached_parameters(
                    cast(dict[str, Any], block_aux)
                )

    def to_config(self) -> NestorConfig:
        """Return the original config."""

        return self.config

    @classmethod
    def from_config(cls, config: BackboneConfig) -> "Nestor":
        """Construct Nestor from a generic backbone config."""

        if not isinstance(config, NestorConfig):
            raise TypeError(
                f"{cls.__name__}.from_config expected NestorConfig, got "
                f"{type(config).__name__}."
            )
        return cls(config=config)
