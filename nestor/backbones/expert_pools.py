"""Public/private expert-pool MLP for grouped-prefix Nestor layers.

A pooled layer replaces its ordinary expert machinery with two sparse pools:

* **Public pool**: ``P`` experts shared by ALL tokens, selected top-``k_p``
  by a layer-local router. OPTIONAL -- ``P = 0`` builds a *private-only*
  layer (see below).
* **Private pools**: one independent pool of ``Q`` experts per attention
  group; a token routes top-``k_q`` inside EACH of the ``k_a`` attention
  groups it selected, and each group's contribution is weighted by the
  token's normalized attention-group gate.

For every token the MLP update is::

    MLP(h) = alpha_public  * sum_e public_gate_e  * PublicExpert_e(h)
           + alpha_private * sum_g attn_gate_g
                           * sum_e private_gate_(g,e) * PrivateExpert_(g,e)(h)

where ``(alpha_public, alpha_private)`` is the softmax of a learned 2-logit
path gate on ``h`` -- BOTH paths are always evaluated (no path selection) and
the gate receives no balancing pressure.

**Private-only mode** (``num_public_experts == 0``): every attention column
draws experts ONLY from its own private pool. No public router, no public
experts, and no path gate exist at all (a softmax over one path is a constant
1.0, so the gate would be an unused parameter and break DDP's reducer), and
the update collapses to the second line above with ``alpha_private = 1``.
Public-only (``P > 0, Q == 0``) is not implemented.

Experts are bias-free SwiGLU maps whose intermediate widths keep the SP-50
active-width contract at any top-k (each *present* family gets an equal share
of the ``4d`` active width one dense SwiGLU spends, so the active total stays
~``4d`` either way)::

    m_public  = round_to_8(2d / k_p)                    # both pools
    m_private = round_to_8(2d / (k_a * k_q))            # both pools
    m_private = round_to_8(4d / (k_a * k_q))            # private-only

**Per-group output** (:meth:`ExpertPoolMLP.forward_per_group`): the private
family is computed group-by-group anyway, so the same forward can hand back
``[batch, seq, G, dim]`` instead of the summed ``[batch, seq, dim]`` at ZERO
extra cost. ``block_residual_mode='attn_res_column'`` uses that to send each
attention group's private experts into that group's own residual stream while
the (genuinely shared) public family broadcasts to all of them. Every other
residual mode calls :meth:`ExpertPoolMLP.forward` and is unaffected.

**Balancing:** instead of a
sequence-granular Switch aux loss, EVERY pool router family reuses
nestor's native loss-free balancing (the detached choice-bias mechanism the
g8t3 attention router already uses, ``router_loss_free_balancing`` /
``router_bias_update_rate``), so pooled arms stay aux-loss-free like every
other nestor g8t3 arm. Each private router's bias statistics only count
tokens routed into its attention group (the validity mask it receives is
``valid & member``).

**Dispatch:** every pool runs through the shared static-shape grouped-GEMM
dispatch (:func:`~nestor.backbones.grouped_mm.grouped_expert_rows`), which is
the default wherever it is usable (CUDA tensors + ``torch._grouped_mm``). The
per-expert sparse loop below is kept as the CPU fallback and as a debugging
reference behind ``NESTOR_POOL_GROUPED_MM=0``. Per pooled layer that turns
``P + G * Q`` sparse dispatches (each with a ``nonzero`` device->host sync and
its own tiny GEMMs) into **2** dispatches / 6 grouped GEMMs and no syncs (**1**
dispatch / 3 GEMMs in private-only mode), by flattening the ``G x Q`` private
experts -- all of the same width -- into one bank indexed by
``group_idx * Q + expert_idx``.

Import direction: this module imports :class:`Router` from
``nestor.backbones.nestor`` eagerly, and ``nestor.py`` imports
:class:`ExpertPoolMLP` lazily inside ``RoutedTransformerLayer.__init__`` --
the lazy direction breaks the cycle. ``grouped_mm`` imports neither, so both
call sites take it at module scope.
"""

from __future__ import annotations

import os

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from nestor.backbones.nestor import Router, RouterOutput
from nestor.backbones.batched_router import (
    batched_routers_enabled,
    check_batchable,
    route_private_batched,
)
from nestor.backbones.expert_pool_dispatch import (  # noqa: F401 (re-exports)
    _accumulate_sparse_expert_bank,
    _grouped_expert_bank,
    private_expert_hidden_dim,
    public_expert_hidden_dim,
    round_to_multiple_of_8,
)
from nestor.backbones.expert_bank_stack import StackedBanksMixin
from nestor.backbones.grouped_mm import fused_gate_requested, grouped_expert_combine
from nestor.backbones.triton_expert import (
    triton_cat_cast_requested,
    triton_expert_available,
    triton_expert_requested,
)
from nestor.backbones.triton_dispatch import (
    triton_index_available,
    triton_index_requested,
)
from nestor.scaling.blocks import GatedScaledMLP, ScaledLinear


class ExpertPoolMLP(StackedBanksMixin, nn.Module):
    """Public/private expert-pool MLP replacing a routed layer's expert banks.

    See the module docstring for the update equation, width contract, and the
    loss-free-balancing choice above.

    Attributes:
        dim (int): Token feature width ``d``.
        num_groups (int): Attention groups ``G`` (one private pool each).
        num_groups_per_token (int): Active attention groups per token ``k_a``.
        num_public_experts (int): Public pool size ``P`` (0 = private-only).
        public_top_k (int): Public experts active per token ``k_p``.
        num_private_experts (int): Private pool size per group ``Q``.
        private_top_k (int): Private experts active per selected group ``k_q``.
        private_gate_mode (str): Gate rule of the private routers,
            ``"renorm"`` or ``"switch"`` (see ``Router.gate_mode``).
        has_public_pool (bool): Whether the public family exists (``P > 0``).
            ``False`` puts the layer in private-only mode: ``public_router``,
            ``public_experts`` and ``path_gate`` are all ``None``.
        public_hidden_dim (int): Public-expert SwiGLU intermediate width (0 in
            private-only mode).
        private_hidden_dim (int): Private-expert SwiGLU intermediate width.
        public_router (Router | None): Layer-local top-``k_p`` router over
            ``P``; ``None`` in private-only mode.
        private_routers (list[Router]): One top-``k_q`` router over ``Q`` per
            attention group.
        public_experts (list[GatedScaledMLP] | None): ``P`` bias-free SwiGLU
            experts; ``None`` in private-only mode.
        private_experts (list[list[GatedScaledMLP]]): ``G x Q`` bias-free
            SwiGLU experts.
        path_gate (ScaledLinear | None): Zero-init 2-logit score head on
            ``h``; softmax gives ``(alpha_public, alpha_private)`` (exactly
            0.5/0.5 at step 0). Always-on: no top-k, no balancing. ``None`` in
            private-only mode, where there is only one path to weight.
        grouped_dispatch (bool | None): Dispatch mode. ``None`` (the default)
            means *auto*: grouped GEMMs whenever the inputs are CUDA tensors
            and ``torch._grouped_mm`` exists, else the sparse loop. ``True``
            forces the grouped path (used by tests to exercise it on CPU,
            where ``segment_mm`` takes its host-side fallback), ``False``
            forces the sparse loop. ``NESTOR_POOL_GROUPED_MM=0`` in the
            environment sets ``False`` at construction.
        fused_gate (bool): ``NESTOR_GROUPED_MM_FUSED_GATE`` -- up + gate as
            one grouped GEMM on the grouped path.
        triton_index (bool): ``NESTOR_TRITON_INDEX_BUILD`` -- the fused
            dispatch-map kernel (integer-exact).
        triton_fused (bool): ``NESTOR_TRITON_EXPERT_DISPATCH`` (default
            follows ``triton_index``) -- fused Triton dispatch gather, SwiGLU
            and weighted slot combine around the grouped GEMMs.
        triton_cat_cast (bool): ``NESTOR_TRITON_EXPERT_CATCAST`` (default 1)
            -- the fused cat+cast of the stacked up|gate bank on that path.
        batched_routers (bool): ``NESTOR_BATCHED_ROUTERS`` (default 0) -- the
            ``G`` private routers run as ONE batched router per layer (round
            5a, :mod:`nestor.backbones.batched_router`); exact up to GEMM
            accumulation order, parameters untouched (resume-safe).
    """

    dim: int
    num_groups: int
    num_groups_per_token: int
    num_public_experts: int
    public_top_k: int
    num_private_experts: int
    private_top_k: int
    private_gate_mode: str
    has_public_pool: bool
    public_hidden_dim: int
    private_hidden_dim: int
    grouped_dispatch: bool | None

    def __init__(
        self,
        dim: int,
        num_groups: int,
        num_groups_per_token: int,
        num_public_experts: int,
        public_top_k: int,
        num_private_experts: int,
        private_top_k: int,
        router_score_mlp_hidden_dims: list[int],
        router_score_mlp_activation: str,
        router_noise_mlp_hidden_dims: list[int],
        router_noise_mlp_activation: str,
        *,
        loss_free_balancing: bool = False,
        bias_update_rate: float = 0.0,
        active_budget: float | None = None,
        public_budget: float | None = None,
        private_gate_mode: str = "renorm",
    ) -> None:
        """Initialize the expert pools, their routers, and the path gate.

        Args:
            dim (int): Token feature width.
            num_groups (int): Attention groups (one private pool per group).
            num_groups_per_token (int): Top-k of the block's attention-group
                router (``k_a``); sets the private expert width.
            num_public_experts (int): Public pool size ``P``; 0 builds a
                private-only layer (no public path, no path gate).
            public_top_k (int): Public experts per token ``k_p``.
            num_private_experts (int): Private pool size per group ``Q`` (> 0).
            private_top_k (int): Private experts per selected group ``k_q``.
            router_score_mlp_hidden_dims (list[int]): Score-MLP hidden widths
                shared by all pool routers.
            router_score_mlp_activation (str): Score-MLP activation.
            router_noise_mlp_hidden_dims (list[int]): Noise-MLP hidden widths
                shared by all pool routers.
            router_noise_mlp_activation (str): Noise-MLP activation.
            loss_free_balancing (bool): Whether detached bias updates are used
                for the public and private routers.
            bias_update_rate (float): Detached bias update step size.
            active_budget (float | None): Private family's active width in
                multiples of ``dim`` (``None`` = the ``2d``/``4d`` contract;
                see :func:`private_expert_hidden_dim`).
            public_budget (float | None): Public family's active width in
                multiples of ``dim`` (``None`` = the ``2d`` contract; see
                :func:`public_expert_hidden_dim`).
            private_gate_mode (str): Gate rule of the private routers.
                ``"renorm"`` softmaxes the selected ``k_q`` scores -- with the
                default ``k_q = 1`` that gate is the constant 1.0, so the
                private routers get NO gradient and route as a random hash.
                ``"switch"`` uses each selected expert's full-softmax
                probability, which keeps a top-1 private router trainable.

        Raises:
            ValueError: If the private pool is empty (public-only pools are not
                implemented) or the group geometry is inconsistent.
        """

        super().__init__()
        if num_private_experts < 1:
            raise ValueError(
                "ExpertPoolMLP requires a private pool: num_private_experts "
                "must be > 0 (public-only pools are not implemented; set both "
                "counts to 0 on the layer to disable pooling entirely)."
            )
        if not 1 <= num_groups_per_token <= num_groups:
            raise ValueError(
                "num_groups_per_token must be in [1, num_groups]."
            )

        self.dim = dim
        self.num_groups = num_groups
        self.num_groups_per_token = num_groups_per_token
        self.num_public_experts = num_public_experts
        self.public_top_k = public_top_k
        self.num_private_experts = num_private_experts
        self.private_top_k = private_top_k
        self.private_gate_mode = private_gate_mode
        self.has_public_pool = num_public_experts > 0
        self.public_hidden_dim = (
            public_expert_hidden_dim(dim, public_top_k, active_budget=public_budget)
            if self.has_public_pool
            else 0
        )
        # With no public family to share with, the private experts take the
        # whole 4d active budget (see private_expert_hidden_dim).
        self.private_hidden_dim = private_expert_hidden_dim(
            dim,
            num_groups_per_token,
            private_top_k,
            with_public_pool=self.has_public_pool,
            active_budget=active_budget,
        )

        def _make_router(
            num_choices: int, top_k: int, gate_mode: str = "renorm"
        ) -> Router:
            # Layer-local (not shared across layers), so no score_lr_mul.
            return Router(
                feature_dim=dim,
                num_choices=num_choices,
                num_choices_per_token=top_k,
                score_mlp_hidden_dims=router_score_mlp_hidden_dims,
                score_mlp_activation=router_score_mlp_activation,
                noise_mlp_hidden_dims=router_noise_mlp_hidden_dims,
                noise_mlp_activation=router_noise_mlp_activation,
                loss_free_balancing=loss_free_balancing,
                bias_update_rate=bias_update_rate,
                gate_mode=gate_mode,
            )

        def _make_expert(hidden_dim: int) -> GatedScaledMLP:
            # Bias-free SwiGLU (the doc's 3-matrix expert: gate/up/down).
            return GatedScaledMLP(
                input_dim=dim,
                hidden_dims=[hidden_dim],
                output_dim=dim,
                activation="silu",
                bias=False,
            )

        # Private-only mode builds NOTHING public: an absent path gate is a
        # constant alpha_private = 1, and a 2-logit gate over one path would
        # be a parameter with no gradient (breaks DDP's reducer).
        self.public_router = (
            _make_router(num_public_experts, public_top_k)
            if self.has_public_pool
            else None
        )
        self.private_routers = nn.ModuleList(
            [
                _make_router(
                    num_private_experts, private_top_k, gate_mode=private_gate_mode
                )
                for _ in range(num_groups)
            ]
        )
        self.public_experts = (
            nn.ModuleList(
                [
                    _make_expert(self.public_hidden_dim)
                    for _ in range(num_public_experts)
                ]
            )
            if self.has_public_pool
            else None
        )
        self.private_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _make_expert(self.private_hidden_dim)
                        for _ in range(num_private_experts)
                    ]
                )
                for _ in range(num_groups)
            ]
        )
        self.path_gate = (
            ScaledLinear(dim, 2, role="score", bias=False, zero_init=True)
            if self.has_public_pool
            else None
        )
        # Grouped GEMMs are the default everywhere they are usable; the env
        # flag is a debugging opt-OUT, read once at construction.
        self.grouped_dispatch = (
            False
            if os.environ.get("NESTOR_POOL_GROUPED_MM", "1") == "0"
            else None
        )
        # NESTOR_GROUPED_MM_FUSED_GATE=1: up + gate as one grouped GEMM on the
        # grouped path (SwiGLU banks only; a no-op for 2-layer experts). Read
        # once here so twins built under different settings can coexist.
        self.fused_gate = fused_gate_requested()
        # NESTOR_TRITON_INDEX_BUILD=1: the grouped dispatch builds its per-row
        # index maps in ONE Triton kernel (integer-exact; launch count only).
        self.triton_index = triton_index_requested() and triton_index_available()
        # NESTOR_TRITON_EXPERT_DISPATCH (default: follows INDEX_BUILD): the
        # dispatch gather, the SwiGLU and the weighted slot combine around the
        # grouped GEMMs run as fused Triton kernels (exact; bias-free SwiGLU
        # banks, which the pools are). See nestor/backbones/triton_expert.py.
        self.triton_fused = triton_expert_requested() and triton_expert_available()
        # NESTOR_TRITON_EXPERT_CATCAST (default 1): the fused cat+cast of the
        # stacked up|gate bank on the Triton expert-dispatch path (profiling knob).
        self.triton_cat_cast = triton_cat_cast_requested()
        # NESTOR_BATCHED_ROUTERS=1: the G private routers evaluate
        # as one batched router per layer (one d -> G*h GEMM + one batched
        # matmul per head, one topk/softmax/scatter on [B,S,G,Q]) instead of
        # G eager Router calls + 2G stack copies -- the per-rank-1 trace's
        # biggest eager island. Same parameters, per-group outputs are views.
        self.batched_routers = batched_routers_enabled() and num_groups > 1
        if self.batched_routers:
            check_batchable(list(self.private_routers))
        self._declare_expert_banks(  # stacked by consolidate_stacked_expert_banks()
            [("stacked_public", self.public_experts), ("stacked_private", self._private_bank())])

    def _use_grouped_dispatch(
        self, x: Float[Tensor, "num_tokens dim"]
    ) -> bool:
        """Whether this forward should take the grouped-GEMM path.

        Args:
            x (Float[Tensor, "num_tokens dim"]): The flattened pool input
                (only its device is read).

        Returns:
            bool: ``grouped_dispatch`` when it is set explicitly, else the
            auto rule (CUDA tensors and a ``torch._grouped_mm`` build).
        """

        if self.grouped_dispatch is not None:
            return self.grouped_dispatch
        return x.is_cuda and hasattr(torch, "_grouped_mm")

    def _public_slots(
        self, public_output: RouterOutput, num_tokens: int
    ) -> tuple[
        Int[Tensor, "num_tokens public_top_k"],
        Float[Tensor, "num_tokens public_top_k"],
    ]:
        """Bank index and combine weight of each public ``(token, slot)`` pair.

        Straight top-k dispatch: the ``k_p`` selected experts of every token,
        weighted by their (already validity-masked) router gates -- padding
        tokens dispatch rows of weight exactly zero.

        Args:
            public_output (RouterOutput): Public router output.
            num_tokens (int): Flattened token count ``B * S``.

        Returns:
            tuple[Int[Tensor, "num_tokens public_top_k"],
            Float[Tensor, "num_tokens public_top_k"]]: Expert indices into
            ``public_experts`` and their gate weights.
        """

        expert_ids = public_output.topk_indices.reshape(
            num_tokens, self.public_top_k
        )
        gates = public_output.gates.reshape(num_tokens, self.num_public_experts)
        return expert_ids, gates.gather(1, expert_ids)

    def _private_slots(
        self,
        private_outputs: list[RouterOutput],
        flat_group_gates: Float[Tensor, "num_tokens num_groups"],
        num_tokens: int,
        selected: Int[Tensor, "num_tokens k_a"] | None = None,
        stacked: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[
        Int[Tensor, "num_tokens k_a*k_q"],
        Float[Tensor, "num_tokens k_a*k_q"],
        Int[Tensor, "num_tokens k_a"],
    ]:
        """Bank index and combine weight of each private ``(token, slot)`` pair.

        The ``G x Q`` private experts all share ``private_hidden_dim``, so they
        flatten into ONE bank indexed by ``group_idx * Q + expert_idx`` and the
        whole private family dispatches in a single grouped GEMM. Each token
        contributes ``k_a * k_q`` rows -- its selected attention groups times
        that group's selected experts -- and each row's weight is
        ``private_gate * attention_group_gate``, exactly the product the sparse
        loop forms per group.

        The ``k_a`` groups are taken as the largest attention gates, which is
        exactly the membership set: ``group_gates`` comes from the block's
        top-``k_a`` group router, so at most ``k_a`` of its entries are
        non-zero (zero for every non-member and for every padding token).
        Should a caller ever pass MORE than ``k_a`` active groups (only
        ``_force_global_tokens_all_groups``, a vision-only path that pools
        never see -- global tokens are non-causal-only), the smallest-gate
        groups would be dropped; the sparse fallback stays exact there.

        Args:
            private_outputs (list[RouterOutput]): One router output per group,
                in group order (gates already masked to ``valid & member``).
            flat_group_gates (Float[Tensor, "num_tokens num_groups"]):
                Normalized attention-group gates.
            num_tokens (int): Flattened token count ``B * S``.
            selected (Int[Tensor, "num_tokens k_a"] | None): The token's
                ``k_a`` attention groups in a caller-chosen SLOT order
                (the group router's ``topk_indices``, which is also the order
                of the column-stream slots). ``None`` recovers them as the
                ``k_a`` largest gates -- the same set, since gates are zero
                outside the router's top-``k_a`` -- in descending-gate order.
                Only the row order differs, and every consumer either sums
                over slots or reads this order back through the returned
                ``selected``.
            stacked (tuple[Tensor, Tensor] | None): The batched routers'
                already-stacked ``gates`` (``[..., G, Q]``) and
                ``topk_indices`` (``[..., G, k_q]``); ``None`` stacks the
                per-group outputs here.

        Returns:
            tuple[Int[Tensor, "num_tokens k_a*k_q"],
            Float[Tensor, "num_tokens k_a*k_q"], Int[Tensor, "num_tokens
            k_a"]]: Indices into the flattened ``G x Q`` bank, their combined
            weights, and the ``k_a`` selected attention groups themselves (in
            slot-major order, so row ``j * k_q + e`` of the first two belongs
            to group ``selected[:, j]``) -- the last is what
            :meth:`_private_update_per_group` needs to un-flatten the bank's
            rows back onto their groups.
        """

        num_slots = self.num_groups_per_token * self.private_top_k
        # [num_tokens, G, Q] gates / [num_tokens, G, k_q] choices. Stacking
        # ALL groups' gates keeps every private router in the autograd graph
        # even when its group has no member token (DDP-safety, gather's
        # backward fills the un-gathered entries with zeros).
        if stacked is None:
            gates = torch.stack([out.gates for out in private_outputs], dim=-2)
            choices = torch.stack(
                [out.topk_indices for out in private_outputs], dim=-2
            )
        else:
            gates, choices = stacked
        gates = gates.reshape(
            num_tokens, self.num_groups, self.num_private_experts
        )
        choices = choices.reshape(num_tokens, self.num_groups, self.private_top_k)

        if selected is None:
            selected = torch.topk(
                flat_group_gates, self.num_groups_per_token, dim=-1
            ).indices  # [num_tokens, k_a]
        else:
            selected = selected.reshape(num_tokens, self.num_groups_per_token)
        group_gate = flat_group_gates.gather(1, selected)  # [num_tokens, k_a]
        expand = selected.unsqueeze(-1)
        chosen = choices.gather(
            1, expand.expand(-1, -1, self.private_top_k)
        )  # [num_tokens, k_a, k_q]
        chosen_gates = gates.gather(
            1, expand.expand(-1, -1, self.num_private_experts)
        ).gather(2, chosen)
        expert_ids = expand * self.num_private_experts + chosen
        weights = chosen_gates * group_gate.unsqueeze(-1)
        return (
            expert_ids.reshape(num_tokens, num_slots),
            weights.reshape(num_tokens, num_slots),
            selected,
        )

    def _private_bank(self) -> list[nn.Module]:
        """Flatten the ``G x Q`` private experts into one ``group * Q + e`` bank.

        Returns:
            list[nn.Module]: All private experts, indexed exactly the way
            :meth:`_private_slots` builds its ids.
        """

        return [expert for bank in self.private_experts for expert in bank]

    def _route(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_active_mask: Bool[Tensor, "batch_size seq_len num_groups"],
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[RouterOutput | None, list[RouterOutput], tuple[Tensor, Tensor] | None]:
        """Run every pool router, public first then the ``G`` private ones.

        The order is part of the contract: routers draw noise, so any caller
        that wants to reproduce another's numerics must run them identically.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized token
                features.
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of real
                (non-padding) tokens.
            group_active_mask (Bool[Tensor, "batch_size seq_len num_groups"]):
                Attention-group membership.
            router_noise (dict[str, Tensor] | None): Pre-drawn noise factors
                from :meth:`draw_router_noise` (``"public"`` ``[B, S, P]``,
                ``"private"`` ``[B, S, G, Q]``); ``None`` lets each router
                draw its own.

        Returns:
            tuple[RouterOutput | None, list[RouterOutput], tuple[Tensor,
            Tensor] | None]: Public router output (``None`` in private-only
            mode), one output per private pool (each masked to ``valid &
            member``), and -- on the batched-router path -- the stacked
            private ``(gates, topk_indices)`` for :meth:`_private_slots`
            (``None`` on the sequential path).
        """

        public_noise = None if router_noise is None else router_noise.get("public")
        private_noise = None if router_noise is None else router_noise.get("private")
        public_output = (
            self.public_router(x, valid_token_mask, noise=public_noise)
            if self.public_router is not None
            else None
        )
        if self.batched_routers:
            private_outputs, gates, choices = route_private_batched(
                list(self.private_routers), x, valid_token_mask,
                group_active_mask, training=self.training, noise=private_noise,
            )
            return public_output, private_outputs, (gates, choices)
        return public_output, [
            self.private_routers[group_idx](
                x,
                valid_token_mask & group_active_mask[:, :, group_idx],
                noise=None if private_noise is None else private_noise[:, :, group_idx],
            )
            for group_idx in range(self.num_groups)
        ], None

    def draw_router_noise(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """Pre-draw every pool router's exploration-noise factor (train mode).

        Drawn by the block OUTSIDE the layer's activation checkpoint and passed
        back in through ``router_noise`` so a recompute routes exactly as the
        forward did (see :meth:`Router.draw_noise`).

        Args:
            batch_size (int): Batch size.
            seq_len (int): Sequence length.
            device (torch.device): Device of the router scores.
            dtype (torch.dtype): Dtype of the router scores.

        Returns:
            dict[str, Tensor]: ``"private"`` ``[B, S, G, Q]`` (one slab per
            private router, in group order) and, with a public pool,
            ``"public"`` ``[B, S, P]``.
        """

        noise = {
            "private": torch.randn(
                batch_size, seq_len, self.num_groups, self.num_private_experts,
                device=device, dtype=dtype,
            )
        }
        if self.public_router is not None:
            noise["public"] = self.public_router.draw_noise(
                batch_size, seq_len, device, dtype
            )
        return noise

    def _public_update(
        self,
        flat_x: Float[Tensor, "num_tokens dim"],
        public_output: RouterOutput,
        num_tokens: int,
    ) -> Float[Tensor, "num_tokens dim"]:
        """Gate-weighted public-pool output, on either dispatch path.

        Args:
            flat_x (Float[Tensor, "num_tokens dim"]): Flattened pool input.
            public_output (RouterOutput): Public router output.
            num_tokens (int): Flattened token count ``B * S``.

        Returns:
            Float[Tensor, "num_tokens dim"]: ``sum_e public_gate_e *
            PublicExpert_e(x)`` (ungated by the path gate).
        """

        if self._use_grouped_dispatch(flat_x):
            return _grouped_expert_bank(
                flat_x,
                list(self.public_experts),
                *self._public_slots(public_output, num_tokens),
                fused_gate=self.fused_gate, stacked=self.stacked_public,
                triton_index=self.triton_index, triton_fused=self.triton_fused,
                triton_cat_cast=self.triton_cat_cast,
            )
        public_update = torch.zeros_like(flat_x)
        _accumulate_sparse_expert_bank(
            public_update,
            flat_x,
            list(self.public_experts),
            public_output.gates.reshape(num_tokens, self.num_public_experts),
        )
        return public_update

    def _private_update(
        self,
        flat_x: Float[Tensor, "num_tokens dim"],
        private_outputs: list[RouterOutput],
        flat_group_gates: Float[Tensor, "num_tokens num_groups"],
        num_tokens: int,
        stacked: tuple[Tensor, Tensor] | None = None,
    ) -> Float[Tensor, "num_tokens dim"]:
        """Private-pool output SUMMED over the token's attention groups.

        Args:
            flat_x (Float[Tensor, "num_tokens dim"]): Flattened pool input.
            private_outputs (list[RouterOutput]): One router output per group.
            flat_group_gates (Float[Tensor, "num_tokens num_groups"]):
                Normalized attention-group gates.
            num_tokens (int): Flattened token count ``B * S``.

        Returns:
            Float[Tensor, "num_tokens dim"]: ``sum_g attn_gate_g * sum_e
            private_gate_(g,e) * PrivateExpert_(g,e)(x)`` (ungated by the path
            gate).
        """

        if self._use_grouped_dispatch(flat_x):
            expert_ids, weights, _ = self._private_slots(
                private_outputs, flat_group_gates, num_tokens, stacked=stacked
            )
            # All G x Q private experts share one width, so they flatten into
            # a single bank -> the whole private family is ONE grouped GEMM.
            return _grouped_expert_bank(
                flat_x,
                self._private_bank(),
                expert_ids,
                weights,
                fused_gate=self.fused_gate, stacked=self.stacked_private,
                triton_index=self.triton_index, triton_fused=self.triton_fused,
                triton_cat_cast=self.triton_cat_cast,
            )
        private_update = torch.zeros_like(flat_x)
        for group_idx, router_output in enumerate(private_outputs):
            # Private gates are already zeroed outside the group's members;
            # the attention gate additionally weights each group's pool.
            weights = router_output.gates.reshape(
                num_tokens, self.num_private_experts
            ) * flat_group_gates[:, group_idx : group_idx + 1]
            _accumulate_sparse_expert_bank(
                private_update,
                flat_x,
                list(self.private_experts[group_idx]),
                weights,
            )
        return private_update

    def _private_update_per_group(
        self,
        flat_x: Float[Tensor, "num_tokens dim"],
        private_outputs: list[RouterOutput],
        flat_group_gates: Float[Tensor, "num_tokens num_groups"],
        num_tokens: int,
        stacked: tuple[Tensor, Tensor] | None = None,
    ) -> Float[Tensor, "num_tokens num_groups dim"]:
        """Private-pool output kept SEPARATE per attention group.

        Same experts, same gates, same dispatched rows as
        :meth:`_private_update` -- and therefore exactly the same FLOPs. Only
        the final reduction differs: instead of collapsing every dispatched
        ``(group, expert)`` row into one ``[num_tokens, dim]`` tensor, the rows
        are summed over the ``k_q`` experts of each group and left on the group
        axis. Summing the result over ``num_groups`` reproduces
        :meth:`_private_update` up to float reassociation (the sum is split as
        ``sum_g sum_e`` instead of one flat ``sum_(g,e)``).

        Groups a token did not select carry no dispatched row at all (grouped
        path) or an exactly-zero attention gate (sparse path), so their entry is
        exactly zero either way.

        Args:
            flat_x (Float[Tensor, "num_tokens dim"]): Flattened pool input.
            private_outputs (list[RouterOutput]): One router output per group.
            flat_group_gates (Float[Tensor, "num_tokens num_groups"]):
                Normalized attention-group gates.
            num_tokens (int): Flattened token count ``B * S``.

        Returns:
            Float[Tensor, "num_tokens num_groups dim"]: Entry ``g`` is
            ``attn_gate_g * sum_e private_gate_(g,e) * PrivateExpert_(g,e)(x)``.
        """

        if self._use_grouped_dispatch(flat_x):
            expert_ids, weights, selected = self._private_slots(
                private_outputs, flat_group_gates, num_tokens, stacked=stacked
            )
            per_slot = self._grouped_private_per_slot(
                flat_x, expert_ids, weights, num_tokens
            )
            # Slot-major rows carry their group in `selected`; scatter them back
            # onto the group axis (top-k indices are distinct, so no collision).
            index = selected.unsqueeze(-1).expand(-1, -1, self.dim)
            return flat_x.new_zeros(
                num_tokens, self.num_groups, self.dim
            ).scatter_add(1, index, per_slot)
        per_group: list[Tensor] = []
        for group_idx, router_output in enumerate(private_outputs):
            weights = router_output.gates.reshape(
                num_tokens, self.num_private_experts
            ) * flat_group_gates[:, group_idx : group_idx + 1]
            group_update = torch.zeros_like(flat_x)
            _accumulate_sparse_expert_bank(
                group_update,
                flat_x,
                list(self.private_experts[group_idx]),
                weights,
            )
            per_group.append(group_update)
        return torch.stack(per_group, dim=1)

    def _grouped_private_per_slot(
        self,
        flat_x: Float[Tensor, "num_tokens dim"],
        expert_ids: Int[Tensor, "num_tokens k_a*k_q"],
        weights: Float[Tensor, "num_tokens k_a*k_q"],
        num_tokens: int,
    ) -> Float[Tensor, "num_tokens k_a dim"]:
        """Grouped-GEMM private rows, collapsed over each slot's ``k_q`` experts.

        Shared by the per-group and per-slot layouts: the bank's ``(token,
        slot, expert)`` rows are weighted and summed over the ``k_q`` experts
        of each attention slot, leaving a ``[num_tokens, k_a, dim]`` tensor in
        the slot order of ``expert_ids``.

        Args:
            flat_x (Float[Tensor, "num_tokens dim"]): Flattened pool input.
            expert_ids (Int[Tensor, "num_tokens k_a*k_q"]): Flattened-bank
                index of each dispatched row (from :meth:`_private_slots`).
            weights (Float[Tensor, "num_tokens k_a*k_q"]): Combine weight of
                each dispatched row.
            num_tokens (int): Flattened token count ``B * S``.

        Returns:
            Float[Tensor, "num_tokens k_a dim"]: Slot ``j`` holds
            ``attn_gate_g * sum_e private_gate_(g,e) * PrivateExpert_(g,e)(x)``
            for the group ``g`` in slot ``j``.
        """

        # [num_tokens, k_a, dim]: weight every row and collapse only the k_q
        # experts of a group (fused with the dispatch under triton_fused).
        return grouped_expert_combine(
            flat_x, self._private_bank(), expert_ids, weights, self.private_top_k,
            fused_gate=self.fused_gate, stacked=self.stacked_private,
            triton_index=self.triton_index, triton_fused=self.triton_fused,
                triton_cat_cast=self.triton_cat_cast,
        )

    def _private_update_per_slot(
        self,
        flat_x: Float[Tensor, "num_tokens dim"],
        private_outputs: list[RouterOutput],
        flat_group_gates: Float[Tensor, "num_tokens num_groups"],
        num_tokens: int,
        slot_groups: Int[Tensor, "num_tokens k_a"],
        stacked: tuple[Tensor, Tensor] | None = None,
    ) -> Float[Tensor, "num_tokens k_a dim"]:
        """Private-pool output laid out by the token's attention SLOTS.

        The slot-layout twin of :meth:`_private_update_per_group`: the same
        experts on the same rows with the same weights, but the result is
        indexed by slot ``j`` (the group ``slot_groups[:, j]``) instead of by
        group -- which is exactly the layout of the column-wise residual
        streams (``slot j`` of token ``t`` belongs to ``topk_indices[t, j]``,
        see :func:`~nestor.backbones.nestor._build_column_stream_plan`). The
        grouped path already produces its rows slot-major, so this is what it
        computes BEFORE the group-axis scatter; the ``[num_tokens, num_groups,
        dim]`` intermediate (``G / k_a`` times wider than the streams, mostly
        zeros) and its scatter never exist. Gathering the per-group tensor at
        ``slot_groups`` reproduces it exactly.

        Args:
            flat_x (Float[Tensor, "num_tokens dim"]): Flattened pool input.
            private_outputs (list[RouterOutput]): One router output per group.
            flat_group_gates (Float[Tensor, "num_tokens num_groups"]):
                Normalized attention-group gates.
            num_tokens (int): Flattened token count ``B * S``.
            slot_groups (Int[Tensor, "num_tokens k_a"]): The group held in
                each slot -- the block's group-router ``topk_indices``.

        Returns:
            Float[Tensor, "num_tokens k_a dim"]: Entry ``j`` is
            ``attn_gate_g * sum_e private_gate_(g,e) * PrivateExpert_(g,e)(x)``
            with ``g = slot_groups[:, j]``.
        """

        if self._use_grouped_dispatch(flat_x):
            expert_ids, weights, _ = self._private_slots(
                private_outputs, flat_group_gates, num_tokens,
                selected=slot_groups, stacked=stacked,
            )
            return self._grouped_private_per_slot(
                flat_x, expert_ids, weights, num_tokens
            )
        per_group = self._private_update_per_group(
            flat_x, private_outputs, flat_group_gates, num_tokens
        )  # sparse loop: [num_tokens, num_groups, dim]
        index = slot_groups.reshape(num_tokens, -1).unsqueeze(-1).expand(
            -1, -1, self.dim
        )
        return per_group.gather(1, index)

    def forward_per_slot(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_gates: Float[Tensor, "batch_size seq_len num_groups"],
        group_active_mask: Bool[Tensor, "batch_size seq_len num_groups"],
        slot_groups: Int[Tensor, "batch_size seq_len k_a"],
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len dim"] | None,
        Float[Tensor, "batch_size seq_len k_a dim"],
        RouterOutput | None,
        list[RouterOutput],
    ]:
        """:meth:`forward_per_group` with the private family in SLOT layout.

        Identical routers, experts, rows, weights and path gate as
        :meth:`forward_per_group`; only the private family's layout differs:
        ``[batch, seq, k_a, dim]`` indexed by the token's attention slots
        (``slot_groups``, the group router's ``topk_indices``) instead of
        ``[batch, seq, num_groups, dim]`` indexed by group. That is the layout
        of the column-wise residual streams, so a column-mode layer adds it
        straight into its ``k`` streams -- no group-wide intermediate, no
        scatter. ``forward_per_group(...)[1].gather(2, slot_groups)`` equals
        this output exactly.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized token
                features (the layer's ``mlp_norm`` output).
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of real
                (non-padding) tokens.
            group_gates (Float[Tensor, "batch_size seq_len num_groups"]):
                Normalized attention-group gates.
            group_active_mask (Bool[Tensor, "batch_size seq_len num_groups"]):
                Attention-group membership from the same router output.
            slot_groups (Int[Tensor, "batch_size seq_len k_a"]): The group
                held in each stream slot (the group router's
                ``topk_indices``).

        Returns:
            tuple[Float[Tensor, "batch_size seq_len dim"] | None,
            Float[Tensor, "batch_size seq_len k_a dim"], RouterOutput | None,
            list[RouterOutput]]: The path-gated public update (``None`` in
            private-only mode), the path-gated per-slot private updates, the
            public router output, and one router output per private pool.
        """

        flat_x = x.reshape(-1, self.dim)
        num_tokens = flat_x.shape[0]
        flat_group_gates = group_gates.reshape(num_tokens, self.num_groups)

        public_output, private_outputs, stacked = self._route(
            x, valid_token_mask, group_active_mask, router_noise=router_noise
        )
        public_update = (
            self._public_update(flat_x, public_output, num_tokens).reshape_as(x)
            if public_output is not None
            else None
        )
        private_per_slot = self._private_update_per_slot(
            flat_x,
            private_outputs,
            flat_group_gates,
            num_tokens,
            slot_groups.reshape(num_tokens, -1),
            stacked=stacked,
        ).view(x.shape[0], x.shape[1], -1, self.dim)

        if public_update is None:
            return (
                None,
                private_per_slot.to(x.dtype),
                public_output,
                private_outputs,
            )
        path_weights = F.softmax(self.path_gate(x).float(), dim=-1).to(
            x.dtype
        )  # [batch_size, seq_len, 2]
        return (
            (path_weights[..., 0:1] * public_update).to(x.dtype),
            (
                path_weights[..., 1:2].unsqueeze(-1) * private_per_slot
            ).to(x.dtype),
            public_output,
            private_outputs,
        )

    def forward_per_group(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_gates: Float[Tensor, "batch_size seq_len num_groups"],
        group_active_mask: Bool[Tensor, "batch_size seq_len num_groups"],
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len dim"] | None,
        Float[Tensor, "batch_size seq_len num_groups dim"],
        RouterOutput | None,
        list[RouterOutput],
    ]:
        """:meth:`forward`, but with the private family left on its group axis.

        The column-wise residual mode
        (``block_residual_mode='attn_res_column'``) keeps one residual stream
        per routed attention group, so it needs the pool's update SPLIT the way
        the pool already computes it: the public family is genuinely shared and
        stays one tensor, while each private pool's contribution belongs to its
        own attention group -- exactly the per-group signal
        :meth:`forward` sums away on its last line.

        Identical work to :meth:`forward`: the same routers run in the same
        order, the same experts are evaluated on the same rows, and the SAME
        2-logit path gate is applied BEFORE the split (``alpha_public`` scales
        the public tensor, ``alpha_private`` every private group). ZERO extra
        FLOPs -- only the reduction over groups is dropped. Consequently
        ``public + private.sum(dim=2)`` equals :meth:`forward`'s update up to
        float reassociation.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized token
                features (the layer's ``mlp_norm`` output).
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of real
                (non-padding) tokens.
            group_gates (Float[Tensor, "batch_size seq_len num_groups"]):
                Normalized attention-group gates (zero outside the token's
                top-``k_a`` groups).
            group_active_mask (Bool[Tensor, "batch_size seq_len num_groups"]):
                Attention-group membership from the same router output.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len dim"] | None,
            Float[Tensor, "batch_size seq_len num_groups dim"], RouterOutput |
            None, list[RouterOutput]]: The path-gated public update (``None`` in
            private-only mode, where there is no public family), the path-gated
            per-group private updates, the public router output, and one router
            output per private pool.
        """

        flat_x = x.reshape(-1, self.dim)
        num_tokens = flat_x.shape[0]
        flat_group_gates = group_gates.reshape(num_tokens, self.num_groups)

        public_output, private_outputs, stacked = self._route(
            x, valid_token_mask, group_active_mask, router_noise=router_noise
        )
        public_update = (
            self._public_update(flat_x, public_output, num_tokens).reshape_as(x)
            if public_output is not None
            else None
        )
        private_per_group = self._private_update_per_group(
            flat_x, private_outputs, flat_group_gates, num_tokens, stacked=stacked
        ).view(x.shape[0], x.shape[1], self.num_groups, self.dim)

        if public_update is None:
            # Private-only: one path, so its weight is the constant 1.
            return (
                None,
                private_per_group.to(x.dtype),
                public_output,
                private_outputs,
            )
        # Same 2-logit path gate as forward(), applied before the split.
        path_weights = F.softmax(self.path_gate(x).float(), dim=-1).to(
            x.dtype
        )  # [batch_size, seq_len, 2]
        return (
            (path_weights[..., 0:1] * public_update).to(x.dtype),
            (
                path_weights[..., 1:2].unsqueeze(-1) * private_per_group
            ).to(x.dtype),
            public_output,
            private_outputs,
        )

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len dim"],
        valid_token_mask: Bool[Tensor, "batch_size seq_len"],
        group_gates: Float[Tensor, "batch_size seq_len num_groups"],
        group_active_mask: Bool[Tensor, "batch_size seq_len num_groups"],
        router_noise: dict[str, Tensor] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len dim"],
        RouterOutput | None,
        list[RouterOutput],
    ]:
        """Compute the pooled MLP update for a pre-normalized input.

        Runs every pool through the grouped-GEMM dispatch when
        :meth:`_use_grouped_dispatch` allows it (the default on GPU) and
        through the per-expert sparse loop otherwise; the two agree up to GEMM
        reassociation. The router outputs are identical either way.

        Args:
            x (Float[Tensor, "batch_size seq_len dim"]): Pre-normalized token
                features (the layer's ``mlp_norm`` output).
            valid_token_mask (Bool[Tensor, "batch_size seq_len"]): Mask of
                real (non-padding) tokens.
            group_gates (Float[Tensor, "batch_size seq_len num_groups"]):
                Normalized attention-group gates from the block's group
                router (zero outside the token's top-``k_a`` groups).
            group_active_mask (Bool[Tensor, "batch_size seq_len num_groups"]):
                Attention-group membership from the same router output.

        Returns:
            tuple[Float[Tensor, "batch_size seq_len dim"], RouterOutput | None,
            list[RouterOutput]]: The pooled MLP update, the public router
            output (``None`` in private-only mode), and one router output per
            private pool (each private output's ``valid_token_mask`` is
            ``valid & member`` so balancing statistics only count tokens
            routed into that group).
        """

        flat_x = x.reshape(-1, self.dim)
        num_tokens = flat_x.shape[0]
        flat_group_gates = group_gates.reshape(num_tokens, self.num_groups)

        # Routers first (their order fixes the noise RNG draws), dispatch
        # after -- the dispatch itself consumes no randomness.
        public_output, private_outputs, stacked = self._route(
            x, valid_token_mask, group_active_mask, router_noise=router_noise
        )

        # `public_update is None` <=> private-only mode, all the way down.
        public_update = (
            self._public_update(flat_x, public_output, num_tokens)
            if public_output is not None
            else None
        )
        private_update = self._private_update(
            flat_x, private_outputs, flat_group_gates, num_tokens, stacked=stacked
        )

        if public_update is None:
            # Private-only: one path, so its weight is the constant 1.
            update = private_update.reshape_as(x)
        else:
            # 2-logit path gate -> (alpha_public, alpha_private). Both paths
            # are always evaluated; fp32 softmax matches the router numerics.
            path_weights = F.softmax(self.path_gate(x).float(), dim=-1).to(
                x.dtype
            )  # [batch_size, seq_len, 2]
            update = path_weights[..., 0:1] * public_update.reshape_as(
                x
            ) + path_weights[..., 1:2] * private_update.reshape_as(x)
        return update.to(x.dtype), public_output, private_outputs
