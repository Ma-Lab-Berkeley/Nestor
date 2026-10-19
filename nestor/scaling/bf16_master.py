"""bf16 parameters with sharded fp32 masters and fp32 gradient accumulators.

Under ``NESTOR_BF16_MASTER=1`` every
Muon-tagged matrix parameter (attention projections, expert banks, read-outs
-- the parameters the forward already consumes as bf16 casts under autocast)
is stored as a **bf16 leaf**:

* the forward uses it directly (no per-op autocast copy, no per-micro-batch
  stacked-bank ``cat+cast``: the 1b per-rank-1 trace charged 11.5 ms fwd +
  19 ms bwd per 4k-token micro-batch to those casts);
* autograd hands back a bf16 gradient per micro-batch, which a
  post-accumulate-grad hook adds into an **fp32 accumulator** and drops --
  exactly the fp32 ``.grad`` accumulation of the fp32-parameter path (the
  gradient is the bf16 GEMM output either way, summed in fp32);
* only the **owner rank** of the sharded :class:`~nestor.scaling.muon.MuonAdamW`
  keeps the parameter's fp32 master (and momentum); at the optimizer step the
  accumulators are reduced to their owners (:func:`reduce_to_owners`, half the
  bytes of an all-reduce), the owner steps the master and writes it back as
  bf16, and the existing owner broadcast then carries bf16 (half the bytes).

Static memory per rank at 5.5B parameters: bf16 params 11 GB + fp32
accumulators 22 GB + master/momentum shards ~1.4 GB (32 ranks) against fp32
params 22 + fp32 grads 22 + momentum shard 0.7 today. Exact up to reduction
order: the values the forward sees are the same bf16 roundings of the same
fp32 master, and every accumulation is fp32.

DDP leaves these parameters alone (they are added to the reducer's ignore
list, like the fused head), so ``no_sync`` / bucketing never see bf16 grads.
AdamW parameters (embeddings, heads, norms, biases) stay fp32 and replicated.

Checkpoints keep the fp32 format: :meth:`MuonAdamW.consolidate_state_dict`
gathers the masters, and :func:`fp32_model_state` substitutes them for the
bf16 tensors in the model state dict on the saving rank.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from nestor.scaling.blocks import get_scaling

MASTER_ATTR = "_fp32_master"
ACC_ATTR = "_fp32_grad_acc"


def bf16_master_enabled() -> bool:
    """Whether ``NESTOR_BF16_MASTER=1`` is set.

    Returns:
        bool: True iff the bf16-master parameter mode is requested.
    """

    return os.environ.get("NESTOR_BF16_MASTER", "0") == "1"


def is_bf16_master_param(param: Tensor) -> bool:
    """Whether ``param`` was converted by :func:`convert_muon_params_to_bf16`.

    Args:
        param (Tensor): A model parameter.

    Returns:
        bool: True iff the parameter carries an fp32 gradient accumulator.
    """

    return hasattr(param, ACC_ATTR)


def convert_muon_params_to_bf16(model: nn.Module) -> list[str]:
    """Turn every Muon-tagged parameter into a bf16 leaf with fp32 side state.

    Must run BEFORE the model is wrapped in DDP or compiled. Each converted
    parameter keeps (as attributes) its fp32 master (``_fp32_master``, from
    the fp32 init -- no rounding at step 0) and a zeroed fp32 accumulator
    (``_fp32_grad_acc``), and gets a post-accumulate-grad hook that folds each
    micro-batch's bf16 gradient into the accumulator and frees ``.grad``.
    The optimizer later drops the masters of parameters this rank does not
    own (:meth:`MuonAdamW._adopt_bf16_masters`).

    Args:
        model (nn.Module): The unwrapped model.

    Returns:
        list[str]: Fully qualified names of the converted parameters (for
            DDP's ignore list).
    """

    names: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scaling = get_scaling(param)
        if scaling is None or scaling.optimizer != "muon":
            continue
        if param.dtype != torch.float32:
            continue
        master = param.data.detach().clone()          # fp32, the init as is
        param.data = param.data.to(torch.bfloat16)     # the leaf the forward uses
        setattr(param, MASTER_ATTR, master)
        setattr(param, ACC_ATTR, torch.zeros_like(master))
        param._acc_touched = False
        param.register_post_accumulate_grad_hook(_fold_grad_into_accumulator)
        names.append(name)
    return names


@torch.no_grad()
def _fold_grad_into_accumulator(param: Tensor) -> None:
    """Post-accumulate-grad hook: ``acc += grad.float()``; free the bf16 grad.

    Args:
        param (Tensor): The bf16 parameter whose ``.grad`` was just produced.
    """

    grad = param.grad
    if grad is None:
        return
    getattr(param, ACC_ATTR).add_(grad.to(torch.float32))
    param._acc_touched = True   # the optimizer steps (and resets) touched parameters only
    param.grad = None


def bf16_master_params(model: nn.Module) -> list[Tensor]:
    """The converted parameters of ``model`` (DDP / compile wrappers peeled).

    Args:
        model (nn.Module): Any wrapping of the trained model.

    Returns:
        list[Tensor]: Parameters carrying an fp32 accumulator.
    """

    return [p for p in model.parameters() if is_bf16_master_param(p)]


@torch.no_grad()
def reduce_to_owners(optimizer: Any) -> None:
    """Reduce every bf16-master accumulator to the rank that owns its master.

    One flat fp32 buffer per owner (the owner's parameters in the optimizer's
    broadcast order) and one ``dist.reduce(SUM)`` each; the owner divides by
    the world size (the mean DDP would have produced). Non-owners' accumulators
    are zeroed afterwards -- they are re-filled by the next window. Unsharded
    (or single-process) runs need no communication: the accumulator IS the
    mean gradient. Call once per optimizer step, after the window's last
    backward, before clipping.

    Args:
        optimizer (Any): The :class:`MuonAdamW` instance (uses its
            ``_shard`` / ``_owned`` / ``_rank`` / ``_world``).
    """

    if not getattr(optimizer, "_shard", False):
        return
    world = optimizer._world
    for owner in range(world):
        params = [p for p in optimizer._owned[owner] if is_bf16_master_param(p)]
        if not params:
            continue
        accs = [getattr(p, ACC_ATTR) for p in params]
        buffer = torch.cat([a.reshape(-1) for a in accs])
        dist.reduce(buffer, dst=owner, op=dist.ReduceOp.SUM)
        if owner == optimizer._rank:
            buffer.div_(world)
            torch._foreach_copy_(accs, MuonAdamWViews.flat_views(buffer, accs))
        else:
            torch._foreach_zero_(accs)


class MuonAdamWViews:
    """Flat-buffer view helper shared with :class:`MuonAdamW` (kept import-free)."""

    @staticmethod
    def flat_views(buffer: Tensor, tensors: list[Tensor]) -> list[Tensor]:
        """Views of ``buffer`` shaped like ``tensors`` (their concatenation order).

        Args:
            buffer (Tensor): ``[sum(t.numel())]`` flat buffer.
            tensors (list[Tensor]): Tensors whose shapes partition the buffer.

        Returns:
            list[Tensor]: One view per tensor, shaped like it.
        """

        return [
            chunk.view_as(t)
            for chunk, t in zip(buffer.split([t.numel() for t in tensors]), tensors)
        ]


@torch.no_grad()
def clip_grad_norm_(
    model: nn.Module, optimizer: Any, max_norm: float, norm_type: float = 2.0
) -> Tensor:
    """Global-norm gradient clipping across bf16-master shards and fp32 params.

    The global norm is ``sqrt(sum over this rank's OWNED accumulators + sum
    over the replicated fp32 parameters' grads)`` with the owned part summed
    across ranks (one scalar all-reduce); every rank then scales its owned
    accumulators and the replicated grads by the same factor. Replicated grads
    are identical on every rank after DDP's all-reduce, so they are counted
    once. Only ``norm_type=2`` is supported (the trainer's default).

    Args:
        model (nn.Module): The training model (any wrapping).
        optimizer (Any): The :class:`MuonAdamW` instance.
        max_norm (float): Clip threshold.
        norm_type (float): Must be 2.0.

    Returns:
        Tensor: The global (pre-clip) gradient norm, as
        ``torch.nn.utils.clip_grad_norm_`` returns it.
    """

    if norm_type != 2.0:
        raise ValueError("bf16-master clipping supports the 2-norm only.")
    owned_accs = [
        getattr(p, ACC_ATTR)
        for p in model.parameters()
        if is_bf16_master_param(p) and optimizer._owns(p)
    ]
    replicated_grads = [
        p.grad for p in model.parameters()
        if not is_bf16_master_param(p) and p.grad is not None
    ]
    device = (owned_accs or replicated_grads)[0].device
    owned_sq = (
        torch.stack(torch._foreach_norm(owned_accs)).square().sum()
        if owned_accs else torch.zeros((), device=device)
    )
    if getattr(optimizer, "_shard", False):
        dist.all_reduce(owned_sq, op=dist.ReduceOp.SUM)
    rep_sq = (
        torch.stack(torch._foreach_norm(replicated_grads)).square().sum()
        if replicated_grads else torch.zeros((), device=device)
    )
    total = (owned_sq + rep_sq).sqrt()
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    if owned_accs:
        torch._foreach_mul_(owned_accs, coef)
    if replicated_grads:
        torch._foreach_mul_(replicated_grads, coef)
    return total


def fp32_model_state(model: nn.Module, state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Substitute consolidated fp32 masters for bf16 tensors in a model state dict.

    Call on the saving rank after :meth:`MuonAdamW.consolidate_state_dict`
    (which leaves every master on that rank as ``_fp32_master``) and BEFORE
    ``optimizer.state_dict()`` (which releases the non-owned masters again to
    free their memory); keeps the on-disk checkpoint in the fp32 format every
    eval / resume path reads. ``save_checkpoint`` follows this order.

    Args:
        model (nn.Module): The unwrapped model.
        state (dict[str, Tensor]): Its ``state_dict()``.

    Returns:
        dict[str, Tensor]: The state with fp32 masters where available.
    """

    out = dict(state)
    for name, param in model.named_parameters():
        master = getattr(param, MASTER_ATTR, None)
        if master is not None and name in out:
            out[name] = master.detach().clone()
    return out
