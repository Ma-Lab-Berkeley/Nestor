"""Post-construction transforms for the paper's exact muP parameterization."""

from __future__ import annotations

import math
import os

import torch
from torch import nn

from nestor.adapters.text import TextAdapter
from nestor.backbones.nestor import Nestor, Router
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.heads.language_modeling import LanguageModelingHead
from nestor.scaling.blocks import (
    ParamScaling,
    RMSNorm,
    ScaledLinear,
    ScaledMLP,
    set_scaling,
)


def _normal_init(linear: ScaledLinear, std: float) -> None:
    """Initialize a bias-free linear map from a centered normal distribution.

    Args:
        linear (ScaledLinear): Linear map to initialize.
        std (float): Entrywise standard deviation.
    """

    with torch.no_grad():
        nn.init.normal_(linear.weight, mean=0.0, std=std)
    linear.bias = None


def _tag_adamw(linear: ScaledLinear, lr_mul: float) -> None:
    """Assign the reference AdamW learning-rate group to a linear weight.

    Args:
        linear (ScaledLinear): Linear map whose weight is tagged.
        lr_mul (float): Multiplier on the base AdamW learning rate.
    """

    set_scaling(
        linear.weight,
        ParamScaling("adamw", lr_mul=lr_mul, weight_decay=True),
    )


def _configure_router(
    router: Router,
    *,
    std: float,
    lr_mul: float,
    deterministic: bool,
    initialize: bool,
    configure_optimizer: bool,
) -> None:
    """Make one router match the reference bias-free direct linear gate.

    Args:
        router (Router): Router to modify.
        std (float): Router matrix initialization standard deviation.
        lr_mul (float): AdamW learning-rate multiplier.
        deterministic (bool): Whether to remove the learned noise network.
        initialize (bool): Whether to reset the router matrix.
        configure_optimizer (bool): Whether to retag the router matrix.
    """

    if deterministic:
        router.score_mlp = ScaledMLP(
            input_dim=router.feature_dim,
            hidden_dims=[],
            output_dim=router.num_choices,
            activation="gelu",
            final_role="score",
            bias=False,
        )
        router.noise_mlp = None

    score = router.score_mlp.layers[-1]
    if initialize:
        _normal_init(score, std)
    if router.num_choices == 1:
        # A single-choice router's softmax gate is constantly 1.0, so its
        # weight can never receive a gradient. Freeze it: a trainable-but-
        # unused parameter breaks DDP's reducer on the dense fast path.
        router.requires_grad_(False)
        return
    if configure_optimizer:
        _tag_adamw(score, lr_mul)


def configure_reference_mup_model(
    adapter: nn.Module,
    backbone: nn.Module,
    head: nn.Module,
    config: ReferenceMupConfig,
) -> None:
    """Apply the paper's muP parameterization to a sequence model in place.

    This intentionally targets the text/Nestor/language-modeling composition
    used by the DCLM reproduction. Explicit type checks prevent a partial,
    silently incorrect transform on unrelated models.

    Args:
        adapter (nn.Module): Expected :class:`TextAdapter`.
        backbone (nn.Module): Expected :class:`Nestor`.
        head (nn.Module): Expected :class:`LanguageModelingHead`.
        config (ReferenceMupConfig): Enabled transformations and constants.

    Raises:
        TypeError: If the model is not the supported sequence composition.
        ValueError: If the architecture transform sees a non-constant-width
            or pooling block.
    """

    if not isinstance(adapter, TextAdapter):
        raise TypeError("Reference muP requires TextAdapter.")
    if not isinstance(backbone, Nestor):
        raise TypeError("Reference muP requires Nestor.")
    if not isinstance(head, LanguageModelingHead):
        raise TypeError("Reference muP requires LanguageModelingHead.")

    base_std = config.initializer_range / math.sqrt(config.width_multiplier)
    output_std = base_std / math.sqrt(2 * config.num_layers)
    matrix_lr_mul = 1.0 / config.width_multiplier
    # MUON PAIRING. With ``matrix_optimizer="muon"`` the in-stream matrices keep
    # the spectral Muon tags set at construction (retagging them to AdamW would
    # leave :class:`MuonAdamW` with an EMPTY Muon partition — the whole model
    # would silently train as AdamW at the small companion rate). Only the
    # non-matrix reference tags are applied: score heads read by a softmax
    # (routers, pool path gates) go to AdamW at the BASE rate — their native
    # ``1/d_in`` multiplier on top of Muon's small ``lr_adamw`` would freeze a
    # deterministic single-matrix router outright — and norm gains keep the
    # usual no-decay group.
    muon_matrices = config.matrix_optimizer == "muon"

    def _tag_matrix(linear: ScaledLinear) -> None:
        """Retag a matrix for reference AdamW, or keep its native Muon tag."""

        if not muon_matrices:
            _tag_adamw(linear, matrix_lr_mul)

    # ROUTER LEARNING RATE -- suspected reference mismatch, gated for testing.
    #
    # The reference implementation applies its muP lr scaling (lr / m_width)
    # by marking a parameter. It marks the attention projections, the MLP
    # matrices and the shared-expert matrices -- and
    # NEVER a MoE router/gate. An unmarked parameter keeps the BASE lr, so
    # the reference router trains at 0.01 where nestor trains it at
    # 0.01 / m_width = 0.00125, i.e. 8x slower.
    #
    # That is the leading explanation for nestor's reference-muP MoE arms
    # collapsing while the same architecture trains fine on the default recipe:
    # a 1b MoE hit CE 4.28 at 6k steps and 5.34 at 8k against the reference's
    # 2.93 at 6k and final 2.7388, and 200m dense-MoE probes sit at
    # 5.2-5.4 regardless of balancer. A router 8x too slow cannot specialise,
    # and the damage scales with expert count -- which is why an E=4 arm
    # survives and E=22/36 do not.
    #
    # NESTOR_MUP_ROUTER_BASE_LR=1 restores the reference behaviour. Default off
    # so no existing arm changes until this is confirmed.
    router_lr_mul = (
        1.0
        if muon_matrices
        or os.environ.get("NESTOR_MUP_ROUTER_BASE_LR", "0") == "1"
        else matrix_lr_mul
    )
    # Same reasoning applied to INIT. Nestor inits every matrix at
    # ``initializer_range / sqrt(m_width)`` (0.0354 at range 0.1, m_width 8).
    # A parameter the reference does not treat as a muP matrix keeps the plain
    # ``initializer_range`` (0.1), i.e. 2.83x wider. If the router is unmarked
    # there, it is unmarked for both lr AND init, so the two flags are usually
    # set together; they are separate so each can be attributed.
    router_std = (
        config.initializer_range
        if os.environ.get("NESTOR_MUP_ROUTER_BASE_INIT", "0") == "1"
        else base_std
    )

    if config.enabled("architecture"):
        for block in backbone.blocks:
            if block.pooling_stride != 1 or block.input_dim != block.output_dim:
                raise ValueError(
                    "Reference architecture requires stride-1 constant-width blocks."
                )
            block.output_proj = nn.Identity()
            if block.num_groups == 1:
                block._dense_noroute = True
                block.group_router.requires_grad_(False)
        backbone.final_norm = RMSNorm(
            backbone.output_dim,
            eps=config.norm_eps,
            bias=False,
        )
        for module in backbone.modules():
            if isinstance(module, RMSNorm):
                module.eps = config.norm_eps
                module.bias = None
            elif isinstance(module, ScaledLinear):
                module.bias = None

    if config.enabled("residual"):
        for block in backbone.blocks:
            for layer in block.layers:
                layer.reference_residual_multiplier = config.residual_multiplier

    initialize = config.enabled("initialization")
    configure_optimizer = config.enabled("optimizer")
    deterministic_router = config.enabled("router")
    for block in backbone.blocks:
        for layer in block.layers:
            for attention in layer.attention_groups:
                for linear in (attention.q_proj, attention.k_proj, attention.v_proj):
                    if initialize:
                        _normal_init(linear, base_std)
                    if configure_optimizer:
                        _tag_matrix(linear)
                if initialize:
                    _normal_init(attention.out_proj, output_std)
                if configure_optimizer:
                    _tag_matrix(attention.out_proj)

            if layer.expert_router is not None:
                _configure_router(
                    layer.expert_router,
                    std=router_std,
                    lr_mul=router_lr_mul,
                    deterministic=deterministic_router,
                    initialize=initialize,
                    configure_optimizer=configure_optimizer,
                )
            expert_banks = [
                *layer.shared_experts,
                *(expert for bank in layer.group_specific_experts for expert in bank),
                *layer.dense_shared_experts,
            ]
            pool = layer.expert_pool
            if pool is not None:
                # Public/private expert pools: every pool router gets the
                # reference bias-free deterministic gate treatment, every pool
                # expert the same up/gate (base) and down (output) init as the
                # standard banks, and the 2-logit path gate -- a score head
                # read by a softmax -- the router-matrix init and lr group.
                # A private-only pool has no public router, no public experts
                # and no path gate, so each of those is skipped when absent.
                pool_routers = list(pool.private_routers)
                if pool.public_router is not None:
                    # Public first: keeps the RNG draw order (and so the
                    # initialization) of both-pools models bit-identical.
                    pool_routers.insert(0, pool.public_router)
                for pool_router in pool_routers:
                    _configure_router(
                        pool_router,
                        std=router_std,
                        lr_mul=router_lr_mul,
                        deterministic=deterministic_router,
                        initialize=initialize,
                        configure_optimizer=configure_optimizer,
                    )
                if pool.public_experts is not None:
                    expert_banks.extend(pool.public_experts)
                for pool_bank in pool.private_experts:
                    expert_banks.extend(pool_bank)
                if pool.path_gate is not None:
                    if initialize:
                        _normal_init(pool.path_gate, base_std)
                    if configure_optimizer:
                        # The 2-logit path gate is read by a softmax — a score
                        # head, not an in-stream matrix — so under a Muon
                        # target it joins the routers at the base AdamW rate.
                        _tag_adamw(
                            pool.path_gate,
                            router_lr_mul if muon_matrices else matrix_lr_mul,
                        )
            for expert in expert_banks:
                up, down = expert.layers
                for linear, std in ((up, base_std), (down, output_std)):
                    if initialize:
                        _normal_init(linear, std)
                    if configure_optimizer:
                        _tag_matrix(linear)
                gate = getattr(expert, "gate", None)
                if isinstance(gate, ScaledLinear):
                    if initialize:
                        _normal_init(gate, base_std)
                    if configure_optimizer:
                        _tag_matrix(gate)

    if configure_optimizer and config.group_router_lr == "base":
        # The block GROUP (attention-column) router is the one router the
        # reference transform leaves native: its 128->G score and noise
        # read-outs carry ``score_lr_mul / d_in`` and so train at
        # ``lr_adamw * M_l^-0.5 / 128`` (~1e-6 under the Muon pairing) -- the
        # same "effectively frozen" regime the router_lr_mul comment above
        # describes for expert routers. ``group_router_lr="base"`` gives the
        # read-outs the expert routers' base-rate treatment. The Muon-owned
        # 640->128 hidden layer is untouched; 1-group blocks are frozen anyway.
        for block in backbone.blocks:
            if block.num_groups == 1:
                continue
            for mlp in (block.group_router.score_mlp, block.group_router.noise_mlp):
                if mlp is None:
                    continue
                _tag_adamw(mlp.layers[-1], 1.0)

    if config.enabled("embedding"):
        with torch.no_grad():
            nn.init.normal_(
                adapter.embed.weight,
                mean=0.0,
                std=config.initializer_range,
            )
        set_scaling(
            adapter.embed.weight,
            ParamScaling("adamw", lr_mul=1.0, weight_decay=True),
        )
        adapter.norm = nn.Identity()
        adapter.embedding_multiplier = config.embedding_multiplier

    if config.enabled("logits"):
        if len(head.projection.layers) != 1:
            raise ValueError("Reference LM head must be a single linear map.")
        lm_head = head.projection.layers[0]
        _normal_init(lm_head, config.initializer_range)
        _tag_adamw(lm_head, 1.0)
        head.logit_multiplier = 1.0 / config.width_multiplier

    if configure_optimizer:
        for module in backbone.modules():
            if isinstance(module, RMSNorm):
                set_scaling(
                    module.weight,
                    ParamScaling("adamw", lr_mul=1.0, weight_decay=False),
                )
