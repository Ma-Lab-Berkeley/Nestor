"""Rebuild a registered experiment's model and load its checkpoint.

Every evaluation script starts the same way: rebuild the exact training
architecture from the experiment registry (plus the ``DCLM_*`` / ``NESTOR_*``
environment the run used) and load ``model_*.pt`` strictly enough that a
registry/env mismatch cannot silently evaluate random weights. This module is
that shared prologue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import nestor.experiments.pretrain  # noqa: F401  (triggers @register_experiment)
from nestor.backbones.expert_bank_stack import consolidate_stacked_expert_banks
from nestor.experiments.registry import get_experiment
from nestor.models import Model
from nestor.train.checkpoint import remap_legacy_bank_keys

__all__ = ["load_experiment_checkpoint", "remap_legacy_bank_keys"]

#: Key fragments only a stacked-expert-bank run writes (see
#: :class:`~nestor.backbones.expert_bank_stack.StackedExpertBank`).
_STACKED_BANK_MARKERS = (
    "stacked_private.w_",
    "stacked_public.w_",
    "stacked_moe_bank.w_",
)


def load_experiment_checkpoint(
    experiment: str,
    checkpoint: Path,
    *,
    device: torch.device | str,
    allow_partial: bool = False,
) -> tuple[Model, dict[str, Any]]:
    """Instantiate ``experiment``'s model and load ``checkpoint`` into it.

    Args:
        experiment (str): Registered experiment name (``nestor_200m``, ...).
        checkpoint (Path): ``model_*.pt`` written by the trainer (a dict with a
            ``model`` state dict, or a bare state dict).
        device (torch.device | str): Device to place the model on.
        allow_partial (bool): Tolerate checkpoint PARAMETERS missing from the
            rebuilt model. Off by default: a mismatch would otherwise evaluate
            randomly initialised weights. Buffers (e.g. non-persistent RoPE
            tables) may legitimately be absent.

    Note:
        A checkpoint written with ``NESTOR_STACKED_EXPERT_BANKS=1`` names its
        expert pools' weights differently; that is detected from the file and
        the rebuilt model is consolidated to match, so eval jobs need not set
        the flag. Legacy 200m baseline checkpoints are renamed on the fly (see
        :func:`remap_legacy_bank_keys`).

    Returns:
        tuple[Model, dict[str, Any]]: The model in eval mode on ``device``, and
        ``{"step", "missing", "unexpected", "num_params"}``.

    Raises:
        SystemExit: When parameters are missing and ``allow_partial`` is False.
    """

    exp = get_experiment(experiment)()
    model = Model.from_config(exp.jobs[0].model)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    # A run trained with NESTOR_STACKED_EXPERT_BANKS=1 stores each expert pool
    # bank as stacked owners, so its parameter NAMES differ from a model built
    # without the flag. Detect it from the checkpoint instead of requiring the
    # eval job to reproduce the training env: if the file is stacked and this
    # model is not, consolidate before loading. Values are identical either
    # way, so the reverse case (unstacked file, stacked model) is handled by
    # simply not consolidating -- the model is rebuilt fresh here.
    if any(marker in key for key in state for marker in _STACKED_BANK_MARKERS):
        consolidate_stacked_expert_banks(model, force=True)
    state = remap_legacy_bank_keys(state, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    param_names = {name for name, _ in model.named_parameters()}
    missing_params = [key for key in missing if key in param_names]
    if missing_params and not allow_partial:
        raise SystemExit(
            f"[eval] {len(missing_params)} PARAMETERS missing from the checkpoint "
            f"(e.g. {missing_params[:3]}); the rebuilt model does not match. "
            "Check --experiment and the DCLM_*/NESTOR_* env, or pass "
            "--allow-partial to evaluate anyway."
        )
    model.to(device).eval()
    info = {
        "step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    return model, info
