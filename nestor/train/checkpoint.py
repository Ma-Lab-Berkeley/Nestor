"""Checkpoint utilities for saving and loading training state."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nestor.utils.distributed import (
    is_fsdp2_model,
    is_main_process,
)


def get_git_sha() -> str | None:
    """Get the current git commit SHA.

    Returns:
        str | None: Git SHA or ``None`` if unavailable.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def filter_init_state(
    ckpt_state: dict[str, Any],
    current_state: dict[str, Any],
    parameter_names: set[str],
    *,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    """Select the checkpoint entries a non-strict weight-init load may apply.

    The non-strict path (``DCLM_INIT_CHECKPOINT``: a fresh optimizer and
    schedule over pretrained weights) keeps every checkpoint entry whose name
    AND shape match the freshly built model, and drops the rest (typically
    length-dependent buffers such as RoPE caches). Dropping is fine for
    buffers but fatal for parameters: a model built in a different layout
    than the file -- an unstacked model fed a ``NESTOR_STACKED_EXPERT_BANKS=1``
    checkpoint, whose expert weights live under other names -- would train
    from random experts while logging only a "skipped keys" line. So the
    filter refuses to leave any parameter of the model uninitialised unless
    ``allow_partial`` is set.

    Args:
        ckpt_state (dict[str, Any]): The checkpoint's ``model`` state dict.
        current_state (dict[str, Any]): ``model.state_dict()`` of the built model.
        parameter_names (set[str]): Names of the model's parameters
            (``dict(model.named_parameters())`` keys), the entries that must be covered.
        allow_partial (bool): Tolerate uncovered parameters (a deliberately new module).

    Returns:
        tuple[dict[str, Any], set[str]]: The entries to load, and the checkpoint
        keys skipped (name unknown or shape mismatch).

    Raises:
        RuntimeError: When a parameter of the model is not covered and
            ``allow_partial`` is False.
    """

    filtered = {
        k: v
        for k, v in ckpt_state.items()
        if k in current_state and tuple(v.shape) == tuple(current_state[k].shape)
    }
    skipped = set(ckpt_state) - set(filtered)
    uncovered = sorted(parameter_names - set(filtered))
    if uncovered and not allow_partial:
        shown = ", ".join(uncovered[:8]) + (" ..." if len(uncovered) > 8 else "")
        raise RuntimeError(
            f"init checkpoint leaves {len(uncovered)} model parameter(s) uninitialised "
            f"(layout mismatch? e.g. NESTOR_STACKED_EXPERT_BANKS differs from the file): {shown}. "
            "Set DCLM_INIT_ALLOW_PARTIAL=1 only for a deliberately new module."
        )
    return filtered, skipped


def _unwrap_compiled(model: nn.Module) -> nn.Module:
    """Strip torch.compile (``_orig_mod``) and DDP (``.module``) wrappers.

    Handles either nesting order (``compile(DDP(m))`` or ``DDP(compile(m))``) so
    checkpoints save/load the bare module's canonical parameter names regardless
    of how the model was wrapped for distribution/compilation.
    """
    from torch.nn.parallel import DistributedDataParallel

    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, DistributedDataParallel):
            model = model.module
        else:
            return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    step: int = 0,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path (Path): Destination checkpoint path.
        model (nn.Module): Possibly wrapped model to save.
        optimizer (Optimizer | None): Optimizer state to save.
        scheduler (LRScheduler | None): Scheduler state to save.
        step (int): Current training step.
        epoch (int): Current epoch.
        config (dict[str, Any] | None): Optional serialized config.
        metrics (dict[str, float] | None): Optional scalar metrics.
    """

    path = Path(path)
    optimizer_state: dict[str, Any] | None = None
    base = _unwrap_compiled(model)

    if is_fsdp2_model(base):
        model_state, optimizer_state = _get_fsdp2_state_dict(
            model=base,
            optimizer=optimizer,
        )
        if not is_main_process():
            return
    else:
        # Sharded optimizers (ZeRO-1 ``ZeroRedundancyOptimizer``, the sharded
        # ``MuonAdamW`` under NESTOR_MUON_SHARD) keep their state split across
        # ranks; consolidation is a COLLECTIVE, so it must run on every rank
        # BEFORE the rank-0 gate. Both consolidate to the unsharded format, so
        # saved checkpoints load into either a sharded or an unsharded optimizer.
        consolidate = getattr(optimizer, "consolidate_state_dict", None)
        if callable(consolidate):
            consolidate(to=0)
        if not is_main_process():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        model_state = base.state_dict()
        # bf16-master parameters are saved as their consolidated fp32 masters
        # (the format every eval / resume path reads).
        from nestor.scaling.bf16_master import fp32_model_state
        model_state = fp32_model_state(base, model_state)
        if optimizer is not None:
            optimizer_state = optimizer.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model": model_state,
        "step": step,
        "epoch": epoch,
        "git_sha": get_git_sha(),
    }
    if optimizer_state is not None:
        checkpoint["optimizer"] = optimizer_state
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if config is not None:
        checkpoint["config"] = config
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path (Path): Checkpoint path.
        model (nn.Module | None): Possibly wrapped model to load.
        optimizer (Optimizer | None): Optimizer to restore.
        scheduler (LRScheduler | None): Scheduler to restore.
        device (torch.device | str): Device mapping for ``torch.load``.
        strict (bool): Whether to enforce strict model-state loading. Even
            when False, missing PARAMETERS are an error (they would train from
            random init); only missing buffers are tolerated.

    Returns:
        dict[str, Any]: Raw checkpoint dictionary.

    Raises:
        RuntimeError: When a non-strict load would leave parameters
            uninitialized (an architecture / construction-flag mismatch).
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        base = _unwrap_compiled(model)
        if is_fsdp2_model(base):
            _load_fsdp2_state_dict(
                model=base,
                optimizer=optimizer,
                model_state=checkpoint["model"],
                optimizer_state=checkpoint.get("optimizer", {}),
                strict=strict,
            )
        else:
            incompatible = base.load_state_dict(checkpoint["model"], strict=strict)
            if not strict:
                # A NON-strict load exists to initialize from a checkpoint of
                # the SAME architecture under a different data/context config
                # (the long-context continued-pretraining stage). Silently missing PARAMETERS
                # would leave them at random init and train a corrupted model
                # -- exactly what an expert-bank ownership mismatch
                # (NESTOR_STACKED_EXPERT_BANKS set for one run and not the
                # other) produces, since it renames every pool weight. Fail
                # loudly instead. Buffers may legitimately be absent.
                names = {name for name, _ in base.named_parameters()}
                absent = [key for key in incompatible.missing_keys if key in names]
                if absent:
                    raise RuntimeError(
                        f"{len(absent)} PARAMETERS are missing from {path} "
                        f"(e.g. {absent[:3]}), so a non-strict load would leave "
                        "them randomly initialized. The checkpoint does not "
                        "match this model -- check the experiment and the "
                        "construction-time NESTOR_* flags "
                        "(NESTOR_STACKED_EXPERT_BANKS and NESTOR_DENSE_NO_ROUTER "
                        "both change the parameter set)."
                    )
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def _get_fsdp2_state_dict(
    model: nn.Module,
    optimizer: Optimizer | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Collect a full FSDP2 state dict.

    Args:
        model (nn.Module): FSDP2-wrapped model root.
        optimizer (Optimizer | None): Optimizer bound to the sharded model.

    Returns:
        tuple[dict[str, Any], dict[str, Any] | None]: Full model and optimizer
            state dicts using canonical parameter names.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
    )

    model_state, optimizer_state = get_state_dict(
        model,
        optimizer if optimizer is not None else (),
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    return model_state, optimizer_state if optimizer is not None else None


def _load_fsdp2_state_dict(
    model: nn.Module,
    optimizer: Optimizer | None,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    *,
    strict: bool,
) -> None:
    """Load a full state dict into an FSDP2-wrapped model.

    Args:
        model (nn.Module): FSDP2-wrapped model root.
        optimizer (Optimizer | None): Optimizer bound to the sharded model.
        model_state (dict[str, Any]): Full model state dict.
        optimizer_state (dict[str, Any]): Full optimizer state dict.
        strict (bool): Whether to enforce strict model-state loading.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_state_dict,
    )

    set_state_dict(
        model,
        optimizer if optimizer is not None else (),
        model_state_dict=model_state,
        optim_state_dict=optimizer_state if optimizer is not None else {},
        options=StateDictOptions(full_state_dict=True, strict=strict),
    )
