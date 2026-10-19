"""Muon, paired with AdamW, as a single optimizer for Nestor.

Muon realizes the spectral condition of ``paper/nestor.tex`` directly: it
orthogonalizes the momentum with a few Newton--Schulz iterations into a step
close to the polar factor ``U V^T`` (spectral norm ``Theta(eta)``), which scaled
by ``sqrt(d_out/d_in)`` is exactly ``||Delta W||_* = Theta(sqrt(d_out/d_in))`` at
any shape. We therefore run Muon on the in-stream matrices and the between-block
projection, and AdamW on what has no such matrix structure -- embeddings, score
heads, norm gains, and biases.

:class:`MuonAdamW` is a single :class:`torch.optim.Optimizer` whose parameter
groups carry a ``method`` field (``"muon"`` or ``"adamw"``). Keeping it one real
optimizer (rather than two wrapped optimizers) lets the FSDP2 distributed
checkpoint utilities, the learning-rate schedulers, and checkpoint save/load in
:mod:`nestor.train` work unchanged.

Note on sharding: the Newton--Schulz orthogonalization needs the full 2D matrix.
Replicated parameters (single device or DDP) are orthogonalized directly; an
FSDP2-sharded ``DTensor`` parameter is all-gathered to its full tensor,
orthogonalized identically on every rank, and resharded back to the parameter's
placement (see :func:`_orthogonalize_update`). The all-gather is the standard
communication cost of running Muon under FSDP.

Under DDP the parameters and gradients are replicated, so by default EVERY rank
orthogonalizes EVERY matrix -- ``world_size`` times the Newton--Schulz work and
a full momentum buffer per rank. ``NESTOR_MUON_SHARD=1`` (or ``shard=True``)
turns on the ZeRO-1-style split of that work (:class:`MuonAdamW`): each Muon
parameter has one OWNER rank that keeps its momentum, orthogonalizes it and
writes the update; the owners then broadcast their parameters so every replica
is bit-identical again. Same math, ``1/world_size`` of the optimizer compute and
momentum memory per rank, plus one broadcast of the Muon parameters per step.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import Tensor

from nestor.scaling.bf16_master import ACC_ATTR, MASTER_ATTR, is_bf16_master_param
from torch.optim import Optimizer

try:
    from torch.distributed.tensor import DTensor, distribute_tensor

    _DTENSOR_AVAILABLE = True
except Exception:  # pragma: no cover - older PyTorch without DTensor
    DTensor = None  # type: ignore[assignment]
    distribute_tensor = None  # type: ignore[assignment]
    _DTENSOR_AVAILABLE = False


def _orthogonalize_update(direction: Tensor, ns_steps: int) -> Tensor:
    """Orthogonalize a (possibly sharded) 2D update.

    Newton--Schulz needs the full matrix. For a replicated parameter this is a
    direct call; for an FSDP2-sharded ``DTensor`` it all-gathers to the full
    tensor, orthogonalizes identically on every rank (the iteration is
    deterministic), and reshards the result back to the parameter's placement.

    Args:
        direction (Tensor): The momentum direction (plain tensor or ``DTensor``).
        ns_steps (int): Newton--Schulz iterations.

    Returns:
        Tensor: The orthogonalized update, matching ``direction``'s sharding.
    """

    if _DTENSOR_AVAILABLE and isinstance(direction, DTensor):
        full = direction.full_tensor()
        ortho = zeropower_via_newtonschulz5(full, ns_steps)
        return distribute_tensor(
            ortho, direction.device_mesh, direction.placements
        )
    return zeropower_via_newtonschulz5(direction, ns_steps)


def zeropower_via_newtonschulz5(grad: Tensor, steps: int) -> Tensor:
    """Orthogonalize a 2D matrix via the quintic Newton--Schulz iteration.

    Returns a matrix close to the polar factor ``U V^T`` of ``grad`` (all
    singular values driven toward one), using the coefficients of
    Jordan (2024). Runs in bfloat16 for speed and is numerically stable for the
    half-precision range typical of momentum buffers.

    Args:
        grad (Tensor): A 2D matrix (e.g. a momentum buffer).
        steps (int): Number of Newton--Schulz iterations.

    Returns:
        Tensor: The orthogonalized matrix, in the dtype of ``grad``.
    """

    if grad.ndim != 2:
        raise ValueError("Newton--Schulz orthogonalization expects a 2D matrix.")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    return x.to(grad.dtype)


def zeropower_via_newtonschulz5_batched(grad: Tensor, steps: int) -> Tensor:
    """Batched quintic Newton--Schulz over a stack of equally-shaped matrices.

    ``grad`` is ``[batch, m, n]``; each ``[m, n]`` slice is orthogonalized
    independently and identically to :func:`zeropower_via_newtonschulz5`, but the
    whole stack runs as batched matmuls (one kernel per step instead of one per
    matrix). This collapses a same-shape expert bank's per-matrix Newton--Schulz
    into a single batched call -- the dominant Muon cost for MoE, where the
    optimizer step scales with the (large) parameter count, not the active FLOPs.

    Args:
        grad (Tensor): A ``[batch, m, n]`` stack of momentum buffers.
        steps (int): Number of Newton--Schulz iterations.

    Returns:
        Tensor: The orthogonalized stack, in the dtype of ``grad``.
    """

    if grad.ndim != 3:
        raise ValueError(
            "Batched Newton--Schulz expects a 3D [batch, m, n] stack."
        )
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    # Same shape for every slice, so one transpose decision covers the batch.
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    return x.to(grad.dtype)


def _orthogonalize_updates(
    directions: list[Tensor], ns_steps: int
) -> list[Tensor]:
    """Orthogonalize a list of 2D updates, batching same-shape matrices.

    Directions that share MATRIX shape (the trailing two dims), dtype and
    sharding are concatenated and orthogonalized with a single batched
    Newton--Schulz; a lone 2D matrix falls back to the per-matrix path.
    Numerically equivalent (within bf16) to calling
    :func:`_orthogonalize_update` on each matrix, but turns ``E`` tiny matmuls
    into one batched ``bmm`` -- the win that brings the MoE optimizer step
    toward the dense baseline's.

    A 3D direction is a STACKED EXPERT BANK ``[E, m, n]``
    (``NESTOR_STACKED_EXPERT_BANKS``): its ``E`` slices are ``E`` experts'
    matrices, each orthogonalized independently, so it joins the bucket of its
    matrix shape as ``E`` slices.

    Args:
        directions (list[Tensor]): Per-parameter momentum directions (plain
            tensors, stacked banks, or ``DTensor``s).
        ns_steps (int): Newton--Schulz iterations.

    Returns:
        list[Tensor]: Orthogonalized updates, aligned with ``directions``.
    """

    buckets: dict[Any, list[int]] = defaultdict(list)
    for i, d in enumerate(directions):
        is_dt = _DTENSOR_AVAILABLE and isinstance(d, DTensor)
        placements = repr(d.placements) if is_dt else None
        # Keyed by the MATRIX shape, so a stacked bank ``[E, m, n]`` shares a
        # bucket with the plain ``[m, n]`` matrices of the same shape: the
        # batched iteration normalizes and orthogonalizes each slice
        # independently, so ``E`` slices and one matrix mix freely.
        buckets[(tuple(d.shape[-2:]), d.dtype, is_dt, placements)].append(i)

    updates: list[Tensor | None] = [None] * len(directions)
    for (_, _, is_dt, _), idxs in buckets.items():
        if len(idxs) == 1 and directions[idxs[0]].ndim == 2:
            i = idxs[0]
            updates[i] = _orthogonalize_update(directions[i], ns_steps)
            continue
        if is_dt:
            # All-gather each shard to its full matrix, batch the (replicated)
            # Newton--Schulz compute, then reshard each result to its placement.
            # FSDP2 shards plain matrices only, so these are all 2D.
            fulls = [directions[i].full_tensor() for i in idxs]
            ortho = zeropower_via_newtonschulz5_batched(
                torch.stack(fulls, 0), ns_steps
            )
            for j, i in enumerate(idxs):
                d = directions[i]
                updates[i] = distribute_tensor(
                    ortho[j], d.device_mesh, d.placements
                )
            continue
        # ONE batched iteration for the whole bucket: every 2D matrix enters
        # as a single slice and every stacked bank as its ``E`` slices, so a
        # model with stacked banks costs one Newton--Schulz call per matrix
        # SHAPE rather than one per parameter (162 banks x 15 matmuls per step
        # otherwise).
        counts = [1 if directions[i].ndim == 2 else directions[i].shape[0] for i in idxs]
        ortho = zeropower_via_newtonschulz5_batched(
            torch.cat(
                [
                    directions[i] if directions[i].ndim == 3
                    else directions[i].unsqueeze(0)
                    for i in idxs
                ],
                0,
            ),
            ns_steps,
        )
        for i, chunk in zip(idxs, ortho.split(counts)):
            updates[i] = chunk if directions[i].ndim == 3 else chunk.squeeze(0)
    return updates  # type: ignore[return-value]


class MuonAdamW(Optimizer):
    """A combined Muon + AdamW optimizer keyed by a per-group ``method``.

    Each parameter group must carry ``method="muon"`` or ``method="adamw"`` and a
    group ``lr`` already scaled for that parameter (Muon groups bake in the
    spectral ``sqrt(d_out/d_in)`` factor, AdamW score-head groups bake in
    ``1/d_in``). All other hyperparameters fall back to the constructor defaults.

    Sharded mode (``shard=True`` or ``NESTOR_MUON_SHARD=1``, active only under
    an initialized process group with ``world_size > 1``, i.e. DDP where every
    rank holds the full all-reduced gradient): every Muon parameter is assigned
    one owner rank (round-robin inside each shape bucket, so every rank's
    batched Newton--Schulz covers the same shapes and the work is balanced);
    only the owner keeps the parameter's momentum buffer and applies its
    update, and :meth:`step` ends with every owner broadcasting its parameters
    (one flat fp32 buffer per owner) so all replicas agree bit for bit. AdamW
    parameters (embeddings, heads, norms, biases) stay replicated and are
    updated on every rank as before. Checkpoints: call
    :meth:`consolidate_state_dict` on EVERY rank before ``state_dict()`` on the
    destination rank; the saved format is the unsharded one, so a checkpoint
    loads into either mode (:meth:`load_state_dict` drops the momentum of
    parameters this rank does not own).

    Args:
        params (Any): Parameter groups (each a dict with ``params`` and ``method``).
        lr (float): Default learning rate (used by groups that omit ``lr``).
        betas (tuple[float, float]): AdamW moment decay rates.
        eps (float): AdamW denominator epsilon.
        weight_decay (float): Decoupled weight decay.
        momentum (float): Muon momentum.
        nesterov (bool): Whether Muon uses Nesterov momentum.
        ns_steps (int): Newton--Schulz iterations per Muon step.
        shard (bool | None): Shard the Muon update across DDP ranks; ``None``
            reads ``NESTOR_MUON_SHARD``.

    Attributes:
        _shard (bool): Whether the sharded Muon update is active.
        _rank (int): This process's rank (0 when unsharded).
        _world (int): Process-group size (1 when unsharded).
        _owner (dict[Tensor, int]): Owner rank of every Muon parameter.
        _owned (list[list[Tensor]]): Muon parameters per owner rank, in a
            fixed order shared by every rank (the broadcast layout).
        _consolidated (dict[Tensor, Tensor]): Momentum buffers of parameters
            this rank does not own, gathered by :meth:`consolidate_state_dict`
            for the next :meth:`state_dict` call and released afterwards.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        shard: bool | None = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            method="adamw",
        )
        super().__init__(params, defaults)
        if shard is None:
            shard = os.environ.get("NESTOR_MUON_SHARD") == "1"
        in_group = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )
        self._shard: bool = bool(shard and in_group)
        self._rank: int = dist.get_rank() if self._shard else 0
        self._world: int = dist.get_world_size() if self._shard else 1
        self._owner: dict[Tensor, int] = {}
        self._owned: list[list[Tensor]] = [[] for _ in range(self._world)]
        self._consolidated: dict[Tensor, Tensor] = {}
        self._consolidated_masters: dict[Tensor, Tensor] = {}
        self._assign_owners()
        self._adopt_bf16_masters()

    def _assign_owners(self) -> None:
        """Assign every Muon parameter an owner rank (deterministic on all ranks).

        Parameters are bucketed by ``(shape, dtype)`` in parameter-group order
        (identical on every rank, since the groups are built from the same
        module) and dealt round-robin across ranks with one running cursor, so
        each rank owns an equal share of every same-shape bank (its batched
        Newton--Schulz keeps the same shapes) and singleton shapes still spread
        evenly. Unsharded, every parameter is owned by rank 0 (this rank).
        """

        buckets: dict[tuple[Any, ...], list[Tensor]] = defaultdict(list)
        for group in self.param_groups:
            if group.get("method") != "muon":
                continue
            for p in group["params"]:
                buckets[(tuple(p.shape), p.dtype, p.device)].append(p)
        cursor = 0
        for params in buckets.values():
            for p in params:
                owner = cursor % self._world
                self._owner[p] = owner
                self._owned[owner].append(p)
                cursor += 1

    def _adopt_bf16_masters(self) -> None:
        """Keep the fp32 master of every OWNED bf16-master parameter; drop the rest.

        :func:`nestor.scaling.bf16_master.convert_muon_params_to_bf16` leaves
        an fp32 master on every converted parameter (the fp32 init itself, not
        a bf16 rounding of it); ownership is only known here. Owners move
        theirs into ``state[p]["master"]`` (checkpointed with the momentum);
        non-owners free theirs.
        """

        for group in self.param_groups:
            if group.get("method") != "muon":
                continue
            for p in group["params"]:
                master = getattr(p, MASTER_ATTR, None)
                if master is None:
                    continue
                if self._owns(p):
                    self.state[p]["master"] = master
                else:
                    delattr(p, MASTER_ATTR)

    def _owns(self, p: Tensor) -> bool:
        """Whether this rank applies the Muon update of ``p``."""

        return not self._shard or self._owner.get(p, self._rank) == self._rank

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply one Muon step to Muon groups and one AdamW step to AdamW groups."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            method = group["method"]
            if method == "muon":
                self._muon_step(group)
            elif method == "adamw":
                self._adamw_step(group)
            else:
                raise ValueError(f"Unknown optimizer method '{method}'.")
        if self._shard:
            self._broadcast_muon_params()
        return loss

    @staticmethod
    def _flat_views(buffer: Tensor, params: list[Tensor]) -> list[Tensor]:
        """Views of a flat buffer shaped like ``params`` (their concatenation order).

        Args:
            buffer (Tensor): ``[sum(p.numel())]`` flat buffer.
            params (list[Tensor]): Tensors whose shapes partition the buffer.

        Returns:
            list[Tensor]: One view per parameter, shaped like it.
        """

        return [
            chunk.view_as(p)
            for chunk, p in zip(buffer.split([p.numel() for p in params]), params)
        ]

    @torch.no_grad()
    def _broadcast_muon_params(self) -> None:
        """Every owner broadcasts its Muon parameters; replicas overwrite theirs.

        One flat buffer per owner (``torch.cat`` of the owner's parameters,
        ``1/world_size`` of the Muon parameters), one ``broadcast`` each, one
        ``_foreach_copy_`` to unpack on the receivers. Sequential per owner so
        only one shard buffer is alive at a time.
        """

        for owner in range(self._world):
            params = self._owned[owner]
            if not params:
                continue
            if owner == self._rank:
                buffer = torch.cat([p.reshape(-1) for p in params])
            else:
                buffer = torch.empty(
                    sum(p.numel() for p in params),
                    dtype=params[0].dtype,
                    device=params[0].device,
                )
            dist.broadcast(buffer, src=owner)
            if owner != self._rank:
                torch._foreach_copy_(params, self._flat_views(buffer, params))

    @torch.no_grad()
    def _consolidate_masters(self, to: int) -> None:
        """Gather the fp32 masters of bf16-master parameters onto rank ``to``.

        Owners broadcast their masters (one flat fp32 buffer per owner); rank
        ``to`` attaches each as ``_fp32_master`` on the parameter (so
        :func:`nestor.scaling.bf16_master.fp32_model_state` can write the fp32
        checkpoint) and keeps it for the next :meth:`state_dict`; other ranks
        discard. A no-op when no parameter is in bf16-master mode.

        Args:
            to (int): Destination rank.
        """

        for owner in range(self._world):
            params = [p for p in self._owned[owner] if is_bf16_master_param(p)]
            if not params:
                continue
            if owner == self._rank:
                buffer = torch.cat([self.state[p]["master"].reshape(-1) for p in params])
            else:
                buffer = torch.empty(
                    sum(p.numel() for p in params), dtype=torch.float32,
                    device=params[0].device,
                )
            dist.broadcast(buffer, src=owner)
            if self._rank == to and owner != self._rank:
                for p, view in zip(params, self._flat_views(buffer, params)):
                    master = view.clone()
                    setattr(p, MASTER_ATTR, master)
                    self._consolidated_masters[p] = master

    @torch.no_grad()
    def consolidate_state_dict(self, to: int = 0) -> None:
        """Gather every Muon momentum buffer onto rank ``to`` (a COLLECTIVE).

        Call on every rank right before ``state_dict()`` on rank ``to``: the
        owners broadcast their momentum shards (a parameter never stepped on
        its owner contributes zeros, exactly what its lazy init would produce),
        rank ``to`` keeps the ones it does not own for the next
        :meth:`state_dict`, every other rank discards them. No-op unsharded.

        Args:
            to (int): Destination rank.
        """

        if not self._shard:
            return
        self._consolidate_masters(to)
        for owner in range(self._world):
            params = self._owned[owner]
            if not params:
                continue
            if owner == self._rank:
                buffer = torch.cat(
                    [
                        self.state[p]["momentum_buffer"].reshape(-1)
                        if "momentum_buffer" in self.state[p]
                        else torch.zeros_like(p, dtype=torch.float32).reshape(-1)
                        for p in params
                    ]
                )
            else:
                buffer = torch.empty(
                    sum(p.numel() for p in params),
                    dtype=torch.float32,  # momentum is fp32 even for bf16 leaves
                    device=params[0].device,
                )
            dist.broadcast(buffer, src=owner)
            if self._rank == to and owner != self._rank:
                for p, view in zip(params, self._flat_views(buffer, params)):
                    self._consolidated[p] = view.clone()

    def state_dict(self) -> dict[str, Any]:
        """The unsharded optimizer state (momentum for every Muon parameter).

        Sharded, the momentum of parameters this rank does not own comes from
        the preceding :meth:`consolidate_state_dict`; it is spliced into
        ``self.state`` for the duration of the call and released afterwards.
        Without a preceding consolidation the state covers only this rank's
        shard (still a loadable, if partial, state).

        Returns:
            dict[str, Any]: The optimizer state dict in torch's format.
        """

        injected = [p for p in self._consolidated if p not in self.state]
        for p in injected:
            self.state[p] = {"momentum_buffer": self._consolidated[p]}
        for p, master in self._consolidated_masters.items():
            self.state.setdefault(p, {})["master"] = master
        try:
            return super().state_dict()
        finally:
            for p in injected:
                del self.state[p]
            for p in self._consolidated_masters:
                if p in self.state and p not in injected:
                    self.state[p].pop("master", None)
                    if not self.state[p]:
                        del self.state[p]
                if not self._owns(p) and hasattr(p, MASTER_ATTR):
                    delattr(p, MASTER_ATTR)
            self._consolidated = {}
            self._consolidated_masters = {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load an (unsharded or partial) state; drop momentum this rank does not own.

        Args:
            state_dict (dict[str, Any]): State produced by :meth:`state_dict`.
        """

        # torch's loader casts floating state to the PARAMETER's dtype (and
        # calls its own policy hook, not a subclass's), which would round the
        # fp32 master / momentum of a bf16 leaf. Stash those fp32 tensors from
        # the incoming state first and write them back after the load.
        from itertools import chain

        saved_ids = list(chain.from_iterable(g["params"] for g in state_dict["param_groups"]))
        live = list(chain.from_iterable(g["params"] for g in self.param_groups))
        stash: dict[Tensor, dict[str, Tensor]] = {}
        for saved_id, p in zip(saved_ids, live):
            if not is_bf16_master_param(p):
                continue
            entry = state_dict["state"].get(saved_id)
            if not isinstance(entry, dict):
                continue
            keep = {
                k: v.detach().to(device=p.device, dtype=torch.float32).clone()
                for k, v in entry.items()
                if k in ("master", "momentum_buffer") and torch.is_tensor(v)
            }
            if keep:
                stash[p] = keep
        super().load_state_dict(state_dict)
        for p in list(self.state.keys()):
            if self._shard and p in self._owner and not self._owns(p):
                del self.state[p]
                continue
            if p in stash:
                self.state[p].update(stash[p])
            if is_bf16_master_param(p) and "master" in self.state[p]:
                # A resumed master supersedes the one adopted at construction
                # (which was the bf16-rounded value the model loaded).
                master = self.state[p]["master"]
                setattr(p, MASTER_ATTR, master)
                p.data.copy_(master)

    def _muon_step(self, group: dict[str, Any]) -> None:
        """Apply one Muon step to all parameters in the group with ``foreach``.

        The momentum bookkeeping and the parameter writes are batched across the
        group with ``torch._foreach_*`` (one kernel launch instead of one per
        parameter). The Newton--Schulz orthogonalization is batched across
        same-shape matrices (expert banks) via :func:`_orthogonalize_updates`.
        Parameters whose gradient is ``None`` this
        step (e.g. an unselected MoE expert) are skipped, so their momentum is
        held rather than decayed.
        """

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        params: list[Tensor] = []
        grads: list[Tensor] = []
        bufs: list[Tensor] = []
        bf16_leaves: list[Tensor] = []   # bf16 parameters whose fp32 master is stepped
        bf16_accs: list[Tensor] = []     # their fp32 accumulators (zeroed after use)
        for p in group["params"]:
            if not self._owns(p):
                continue
            if is_bf16_master_param(p):
                # The gradient is the fp32 accumulator (already reduced to this
                # owner); a parameter no micro-batch touched keeps its momentum
                # exactly as a ``grad is None`` parameter does below.
                if not getattr(p, "_acc_touched", False):
                    continue
                grad = getattr(p, ACC_ATTR)
                state = self.state[p]
                master = state["master"]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(master)
                params.append(master)
                grads.append(grad)
                bufs.append(state["momentum_buffer"])
                bf16_leaves.append(p)
                bf16_accs.append(grad)
                continue
            if p.grad is None:
                continue
            if p.grad.ndim not in (2, 3):
                raise ValueError(
                    "Muon parameters must be 2D matrices (or [E, m, n] stacked "
                    f"expert banks); got shape {tuple(p.grad.shape)}."
                )
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            params.append(p)
            grads.append(p.grad)
            bufs.append(state["momentum_buffer"])
        if not params:
            return
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads)
        if nesterov:
            directions = torch._foreach_add(grads, bufs, alpha=momentum)
        else:
            directions = bufs
        updates = _orthogonalize_updates(list(directions), ns_steps)
        if weight_decay != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)
        torch._foreach_add_(params, updates, alpha=-lr)
        if bf16_leaves:
            # Write the stepped masters back as the bf16 leaves the forward uses
            # (the rounding the fp32-parameter path applies under autocast) and
            # clear the consumed accumulators.
            torch._foreach_copy_(
                bf16_leaves, [self.state[p]["master"] for p in bf16_leaves]
            )
            torch._foreach_zero_(bf16_accs)
            for p in bf16_leaves:
                p._acc_touched = False

    def _adamw_step(self, group: dict[str, Any]) -> None:
        """Apply one AdamW step to all parameters in the group with ``foreach``.

        Mirrors torch's multi-tensor AdamW: the moment updates, bias correction,
        and parameter writes are batched with ``torch._foreach_*``. Step counts
        are tracked per parameter so that parameters skipped on some steps (e.g.
        sparse MoE experts) get the correct bias correction.
        """

        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        params: list[Tensor] = []
        grads: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        steps: list[int] = []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])
        if not params:
            return

        if weight_decay != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)

        # exp_avg    <- beta1 * exp_avg    + (1 - beta1) * grad
        # exp_avg_sq <- beta2 * exp_avg_sq + (1 - beta2) * grad^2   (batched)
        torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1.0 - beta2)

        # Per-parameter bias correction (step counts may differ across the group).
        bias_correction1 = [1.0 - beta1**s for s in steps]
        bias_correction2_sqrt = [math.sqrt(1.0 - beta2**s) for s in steps]

        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, bias_correction2_sqrt)
        torch._foreach_add_(denom, eps)

        step_size = [-lr / bc1 for bc1 in bias_correction1]
        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)
