"""Optimizer and scheduler utilities."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

from torch import nn
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    StepLR,
    LambdaLR,
    SequentialLR,
)

from nestor.configs.optim import (
    AdamConfig,
    AdamWConfig,
    MuonAdamWConfig,
    OptimizerConfig,
    SGDConfig,
)
from nestor.configs.sched import (
    SchedulerConfig,
    CosineAnnealingConfig,
    PowerLawConfig,
    StepLRConfig,
    WSDConfig,
)
from nestor.scaling.blocks import get_scaling
from nestor.scaling.muon import MuonAdamW


def build_optimizer(config: OptimizerConfig, model: nn.Module) -> Optimizer:
    """
    Build an optimizer from configuration.

    Args:
        config (OptimizerConfig): Optimizer configuration.
        model (nn.Module): Model to optimize.

    Returns:
        Optimizer: The constructed optimizer.
    """
    params = model.parameters()

    if isinstance(config, MuonAdamWConfig):
        return _build_muon_adamw(config, model)
    elif isinstance(config, AdamWConfig):
        return _build_mup_adamw(config, model)
    elif isinstance(config, AdamConfig):
        return Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
    elif isinstance(config, SGDConfig):
        return SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


def _build_mup_adamw(config: AdamWConfig, model: nn.Module) -> AdamW:
    """Build a pure-AdamW optimizer that honours the muP scaling tags.

    This is the all-AdamW counterpart of :func:`_build_muon_adamw`, the closest
    nestor analogue of the reference ``params_group_method: mup`` + ``TorchAdamW``
    recipe. Score heads (the router/readout, including the LM unembedding) are
    grouped at ``lr * (1/d_in)`` so their softmax-Lipschitz output stays
    ``Theta(1)`` as width grows; norm gains, biases, and embeddings get no weight
    decay; matrix (in-stream/projection) weights — which the spectral muP would
    hand to Muon — train at the **base** AdamW rate. (NOT ``1/d_in``: nestor's muP
    uses an *absolute* fan-in scale, so a ``1/d_in`` multiplier puts the matrix LR
    at ~``lr/d_in`` ≈ 6e-6 at d=1600 — effectively frozen, so the model only learns
    through the embeddings. Measured: DCLM repro@8000 held-out CE 5.34 with 1/d_in
    vs 3.55 for Muon at the same step. The base rate ≈ Muon's effective spectral
    LR and trains.) Untagged parameters fall back to the base group with a warning.
    ``NESTOR_DROP_MUP=1`` ignores the tags entirely (one plain AdamW group),
    reproducing a vanilla run.

    Args:
        config (AdamWConfig): AdamW configuration (base ``lr``, decay, betas, eps).
        model (nn.Module): Model whose parameters carry scaling tags.

    Returns:
        AdamW: AdamW with per-group learning rates and weight-decay flags.
    """
    import os as _os

    _drop_mup = _os.environ.get("NESTOR_DROP_MUP") == "1"

    # Group key: (lr_multiplier, weight_decay_enabled).
    groups: dict[tuple[float, bool], list[nn.Parameter]] = defaultdict(list)
    untagged: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scaling = None if _drop_mup else get_scaling(param)
        if scaling is None:
            untagged.append(name)
            groups[(1.0, True)].append(param)
        elif scaling.optimizer == "adamw":
            # Score heads carry lr_mul = 1/d_in; norms/biases/embeddings carry
            # weight_decay=False. Both are encoded in the tag.
            groups[(round(scaling.lr_mul, 10), scaling.weight_decay)].append(param)
        elif scaling.optimizer == "muon":
            # In-stream/projection matrices: train at the BASE AdamW rate with
            # decay. nestor's muP uses an absolute 1/d_in scale, so a 1/d_in lr_mul
            # froze the matrices (~6e-6 at d=1600); Muon avoids this via its
            # spectral scale, and the base rate is the AdamW analogue that trains.
            groups[(1.0, True)].append(param)
        else:
            raise ValueError(
                f"Parameter '{name}' has unknown optimizer '{scaling.optimizer}'."
            )

    if untagged:
        warnings.warn(
            f"muP-AdamW: {len(untagged)} parameter(s) had no scaling tag and were "
            f"placed in the base group (e.g. {untagged[0]}).",
            stacklevel=2,
        )

    param_groups: list[dict[str, Any]] = []
    for (lr_mul, decay), group_params in sorted(groups.items()):
        param_groups.append(
            {
                "params": group_params,
                "lr": config.lr * lr_mul,
                "weight_decay": config.weight_decay if decay else 0.0,
            }
        )

    if not param_groups:
        raise ValueError("AdamW received a model with no trainable parameters.")

    # NESTOR_ADAMW_FUSED=1: one fused CUDA kernel per step instead of the
    # foreach multi-tensor path. Identical math (same update rule, still fp32
    # state); measured as most of the 0.12 s/step `update` phase at 745M
    # params. Opt-in because fused requires all params on CUDA.
    import os

    fused_kwargs: dict[str, Any] = (
        {"fused": True} if os.environ.get("NESTOR_ADAMW_FUSED") == "1" else {}
    )

    # NESTOR_ZERO1=1: shard the optimizer STATE across DDP ranks (ZeRO-1).
    # AdamW's fp32 moments are 8 bytes/param -- ~37 GB/rank at the 1b arm's
    # ~4.6B stored params, the bulk of the static memory that makes per-rank 8
    # OOM there (the peak is byte-identical with every activation/loss-side
    # lever applied, so static state is what remains). Params and grads stay
    # replicated; the update math is unchanged; every muP param group keeps
    # its LR/decay. Checkpoints: save() consolidates to rank 0 first, so the
    # saved format matches plain AdamW's and loads either way.
    import torch.distributed as _dist

    if (
        os.environ.get("NESTOR_ZERO1") == "1"
        and _dist.is_available()
        and _dist.is_initialized()
        and _dist.get_world_size() > 1
    ):
        from torch.distributed.optim import ZeroRedundancyOptimizer

        first, *rest = param_groups
        zero = ZeroRedundancyOptimizer(
            first["params"],
            optimizer_class=AdamW,
            lr=first["lr"],
            betas=config.betas,
            eps=config.eps,
            weight_decay=first["weight_decay"],
            **fused_kwargs,
        )
        for group in rest:
            zero.add_param_group(
                {
                    "params": group["params"],
                    "lr": group["lr"],
                    "weight_decay": group["weight_decay"],
                    "betas": config.betas,
                    "eps": config.eps,
                    **fused_kwargs,
                }
            )
        return zero

    return AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        **fused_kwargs,
    )


def _build_muon_adamw(config: MuonAdamWConfig, model: nn.Module) -> MuonAdamW:
    """Partition parameters by their scaling tags into Muon and AdamW groups.

    Muon-owned parameters are grouped by their spectral scale ``sqrt(d_out/d_in)``
    so each group's learning rate is ``eta_M * scale`` (the spectral condition on
    the update). AdamW-owned parameters are grouped by their learning-rate
    multiplier and weight-decay flag, so score heads land at ``eta_A / d_in`` and
    norms/biases at ``eta_A`` with no decay. Untagged parameters fall back to a
    plain AdamW group (with a warning), so partial migrations still train.

    Args:
        config (MuonAdamWConfig): Optimizer configuration.
        model (nn.Module): Model whose parameters carry scaling tags.

    Returns:
        MuonAdamW: The constructed combined optimizer.
    """

    muon_groups: dict[float, list[nn.Parameter]] = defaultdict(list)
    adamw_groups: dict[tuple[float, bool], list[nn.Parameter]] = defaultdict(list)
    untagged: list[str] = []

    # NESTOR_DROP_MUP=1 ignores the muP/Muon scaling tags so every param lands in
    # the default AdamW group -- exactly what FSDP2 did silently (its DTensor
    # wrap stripped the tags), which is the regime the original sweep was tuned
    # on and which gives ~0.72 top-1 vs ~0.57 for real-muP Muon. Lets the fast
    # DDP path reproduce the original optimizer.
    import os as _os

    _drop_mup = _os.environ.get("NESTOR_DROP_MUP") == "1"

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scaling = None if _drop_mup else get_scaling(param)
        if scaling is None:
            untagged.append(name)
            adamw_groups[(1.0, True)].append(param)
        elif scaling.optimizer == "muon":
            muon_groups[round(scaling.muon_scale, 8)].append(param)
        elif scaling.optimizer == "adamw":
            adamw_groups[(round(scaling.lr_mul, 10), scaling.weight_decay)].append(
                param
            )
        else:
            raise ValueError(
                f"Parameter '{name}' has unknown optimizer "
                f"'{scaling.optimizer}'."
            )

    if untagged:
        warnings.warn(
            "MuonAdamW: "
            f"{len(untagged)} parameter(s) had no scaling tag and were assigned "
            f"to a default AdamW group (e.g. {untagged[0]}). Build them with the "
            "scaling blocks to control their init and learning rate.",
            stacklevel=2,
        )

    if not muon_groups and not _drop_mup:
        # Every matrix was tagged AdamW — e.g. the reference muP "optimizer"
        # component ran with matrix_optimizer="adamw" — so the "Muon" run would
        # silently train as pure AdamW at the (small) companion lr_adamw.
        warnings.warn(
            "MuonAdamW: the Muon partition is EMPTY — no parameter carries a "
            "'muon' scaling tag, so every matrix will train under AdamW at "
            f"lr_adamw={config.lr_adamw}. If this model uses reference muP, set "
            "ReferenceMupConfig(matrix_optimizer='muon') so the matrices keep "
            "their spectral Muon tags.",
            stacklevel=2,
        )

    param_groups: list[dict[str, Any]] = []
    for scale, group_params in sorted(muon_groups.items()):
        param_groups.append(
            {
                "params": group_params,
                "method": "muon",
                "lr": config.lr_muon * scale,
                "weight_decay": config.weight_decay,
                "momentum": config.muon_momentum,
                "nesterov": config.muon_nesterov,
                "ns_steps": config.muon_ns_steps,
            }
        )
    for (lr_mul, decay), group_params in sorted(adamw_groups.items()):
        param_groups.append(
            {
                "params": group_params,
                "method": "adamw",
                "lr": config.lr_adamw * lr_mul,
                "betas": config.betas,
                "eps": config.eps,
                "weight_decay": config.weight_decay if decay else 0.0,
            }
        )

    if not param_groups:
        raise ValueError("MuonAdamW received a model with no trainable parameters.")

    optimizer = MuonAdamW(
        param_groups,
        lr=config.lr_adamw,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        momentum=config.muon_momentum,
        nesterov=config.muon_nesterov,
        ns_steps=config.muon_ns_steps,
    )
    # Say in the log whether NESTOR_MUON_SHARD actually took effect: it needs an
    # initialized process group with world_size > 1, and a silent fallback to
    # the replicated update would otherwise be invisible.
    if _os.environ.get("NESTOR_MUON_SHARD") == "1":
        import logging as _logging

        from nestor.utils.distributed import is_main_process as _is_main

        if _is_main():
            _logging.getLogger("nestor.optim").info(
                "[muon] sharded update %s (rank 0 owns %d of %d Muon matrices "
                "across %d ranks)",
                "ON" if optimizer._shard else "OFF (no process group)",
                len(optimizer._owned[optimizer._rank]) if optimizer._shard else 0,
                len(optimizer._owner),
                optimizer._world,
            )
    return optimizer


def build_scheduler(
    config: SchedulerConfig | None,
    optimizer: Optimizer,
    num_training_steps: int | None = None,
) -> LRScheduler | None:
    """
    Build a learning rate scheduler from configuration.

    Args:
        config (SchedulerConfig | None): Scheduler configuration.
        optimizer (Optimizer): Optimizer to schedule.
        num_training_steps (int | None): Total training steps (for cosine).

    Returns:
        LRScheduler | None: The constructed scheduler, or None.
    """
    if config is None:
        return None

    if isinstance(config, CosineAnnealingConfig):
        T_max = num_training_steps if num_training_steps else config.T_max
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.eta_min,
        )

        if config.warmup_steps > 0:
            return _create_warmup_scheduler(
                optimizer, base_scheduler, config.warmup_steps
            )
        return base_scheduler

    elif isinstance(config, StepLRConfig):
        base_scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )

        if config.warmup_steps > 0:
            return _create_warmup_scheduler(
                optimizer, base_scheduler, config.warmup_steps
            )
        return base_scheduler

    elif isinstance(config, WSDConfig):
        total = num_training_steps if num_training_steps else config.total_steps
        return _create_wsd_scheduler(optimizer, config, total)

    elif isinstance(config, PowerLawConfig):
        return _create_power_law_scheduler(optimizer, config)

    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")


def _create_power_law_scheduler(
    optimizer: Optimizer,
    config: PowerLawConfig,
) -> LRScheduler:
    """Build the reference power-law schedule as a common LR multiplier.

    Args:
        optimizer (Optimizer): Optimizer whose parameter groups are scheduled.
        config (PowerLawConfig): Power-law constants and warmup boundary.

    Returns:
        LRScheduler: Lambda scheduler matching the reference implementation.
    """

    warmup = max(0, config.warmup_steps)
    warmup_peak = min(
        1.0,
        (config.a / config.base_lr) * ((max(1, warmup) * config.c) ** config.b),
    )

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step <= warmup:
            return warmup_peak * step / warmup
        return min(
            1.0,
            (config.a / config.base_lr) * ((max(1, step) * config.c) ** config.b),
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _create_wsd_scheduler(
    optimizer: Optimizer, config: WSDConfig, total_steps: int
) -> LRScheduler:
    """Warmup-Stable-Decay schedule as a single multiplicative ``LambdaLR``.

    The returned lambda maps step -> lr/lr_base in ``[eta_min/lr_base, 1]``: linear
    warmup, constant 1.0 through the stable phase, then a ``decay_shape`` decay to
    the ``eta_min`` floor over the final ``decay_steps``. Applied uniformly to all
    param groups so muP per-group base LRs keep their ratios.
    """
    import math

    warmup = max(0, config.warmup_steps)
    decay = max(1, config.decay_steps)
    decay_start = max(warmup, total_steps - decay)
    # Floor expressed relative to the (largest) base LR so the multiplier lands
    # the absolute LR at config.eta_min at the end of decay.
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    lr_ref = max(base_lrs) if base_lrs else 1.0
    floor = config.eta_min / lr_ref if lr_ref > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / decay)
        if config.decay_shape == "linear":
            scale = 1.0 - progress
        elif config.decay_shape == "cosine":
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:  # one_minus_sqrt (MiniCPM/WSD default)
            scale = 1.0 - math.sqrt(progress)
        return floor + (1.0 - floor) * scale

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _create_warmup_scheduler(
    optimizer: Optimizer,
    base_scheduler: LRScheduler,
    warmup_steps: int,
) -> LRScheduler:
    """
    Create a scheduler with linear warmup.

    Args:
        optimizer (Optimizer): The optimizer.
        base_scheduler (LRScheduler): Base scheduler after warmup.
        warmup_steps (int): Number of warmup steps.

    Returns:
        LRScheduler: Scheduler with warmup.
    """
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: step / max(1, warmup_steps),
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps],
    )
