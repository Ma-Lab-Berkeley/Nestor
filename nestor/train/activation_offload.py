"""Spill saved activations to host memory once the GPU is nearly full.

An escape valve, not a speed feature. ``torch.autograd.graph.saved_tensors_hooks``
intercepts every tensor the forward stashes for its backward; this pairs a
``pack`` that copies a tensor to pinned host memory with an ``unpack`` that
brings it back, and applies the pair only while allocated GPU memory is above a
threshold. Below the threshold the hooks are pass-through, so a step that fits
pays nothing but the predicate.

MEASURED: THIS DOES NOT RAISE THE BATCH CEILING. USE CHECKPOINTING INSTEAD.
---------------------------------------------------------------------------
Kept because it is the natural thing to reach for and the result is worth
recording, not because it works. On one H200 at seq 4096, largest per-rank
micro-batch that survives a real forward/backward
(``scripts/dclm_pretrain/max_batch_probe.py``):

    arm                  plain   +checkpoint   +offload        +both
    sched_lin8_colres        8            21   host OOM-kill      16
    sched_lin16_colres       5            19   host OOM-kill      16

Offload alone did not degrade, it **crashed the host**: it pinned its way
through the job's 300 GiB cgroup and was OOM-killed. Combined with
checkpointing it *lowered* the ceiling (16 against 21 and 19), because the
pinned host buffers and hook bookkeeping cost more than the spill relieves.

The reason is structural. Nestor's peak is not a spike: measured on
a 200m column arm at batch 4, the forward hands **54.8 GiB** to
the backward and the backward's own workspace adds only ~12 GiB on top. There
is no transient to shave -- crossing a threshold means spilling from a large,
persistently held pool, and every spilled byte crosses PCIe twice per step.

``NESTOR_ACTIVATION_CHECKPOINT=1``
(:func:`nestor.backbones.nestor._maybe_checkpoint`) attacks the same pool by
recomputing it, which costs arithmetic the GPU has spare instead of bandwidth it
does not: **2.6-3.8x the per-rank batch for 18-30% more time per sequence.**

Enable with ``NESTOR_ACTIVATION_OFFLOAD=1``; tune the trigger with
``NESTOR_ACTIVATION_OFFLOAD_GIB`` (default: 80% of the device), the smallest
tensor worth moving with ``NESTOR_ACTIVATION_OFFLOAD_MIN_MIB`` (default 1 MiB --
below that, per-tensor latency dominates the transfer), and the pinned-host
ceiling with ``NESTOR_ACTIVATION_OFFLOAD_MAX_GIB`` (default 16, which is what
keeps the failure on the GPU instead of on the host).
"""

from __future__ import annotations

import os
import weakref
from contextlib import nullcontext
from typing import Any, ContextManager

import torch
from torch import Tensor

GIB = 2 ** 30
MIB = 2 ** 20

_SHARE_STORAGE_RATIO = 16
"""int: Spill a view's WHOLE storage (shareable) only if it is at most this many times the view."""


class ActivationOffloadStats:
    """Running totals describing what an offload window actually moved.

    Attributes:
        packed (int): Tensors copied to host memory.
        passed (int): Tensors left on device because the predicate said the
            step had room, or because they were below the size floor.
        bytes_moved (int): Total bytes copied to the host (counted once; the
            return trip moves the same amount again).
        shared (int): Saves that reused an existing host copy of the same
            tensor (:class:`_HostCopyCache`) instead of copying it again.
    """

    packed: int
    passed: int
    bytes_moved: int
    shared: int

    def __init__(self) -> None:
        """Start every counter at zero."""

        self.packed = 0
        self.passed = 0
        self.bytes_moved = 0
        self.shared = 0

    def summary(self) -> str:
        """Return a one-line description of the window.

        Returns:
            str: Counts and moved volume, or a note that nothing was offloaded.
        """

        if self.packed == 0:
            return (
                f"offload: nothing moved ({self.passed} tensors stayed resident)"
            )
        return (
            f"offload: {self.packed} tensors / "
            f"{self.bytes_moved / GIB:.2f} GiB to host "
            f"({self.passed} stayed resident, {self.shared} saves shared a copy)"
        )


class _OffloadRef:
    """A saved tensor living in pinned host memory, plus how to restore it.

    Attributes:
        host (Tensor): Pinned CPU copy of the original.
        device (torch.device): Device the tensor must return to.
    """

    __slots__ = ("host", "device", "__weakref__")

    def __init__(self, host: Tensor, device: torch.device) -> None:
        """Record a host copy and where it came from.

        Args:
            host (Tensor): Pinned CPU tensor holding the values.
            device (torch.device): Device to restore to on unpack.
        """

        self.host = host
        self.device = device


class _HostCopyCache:
    """Host copies keyed by STORAGE identity, so every view of one buffer is spilled once.

    Under checkpointing a recompute region saves its INPUTS, and the column
    backbone hands the same buffers to every one of its ~66 regions: the
    block's depth sources (~46 at 128k, each a view of a block's exported
    ``[batch, seq, k + 1, dim]`` buffer) go to every attention half whole and
    to every expert-half chunk as a sequence slice. Without this cache each
    save copied its view to pinned host memory again (1b at 128k: 2.4 TB of
    host RSS, the job OOM-killed by the host) -- and keying on
    the view would still copy every slice. So the cache keys on the storage
    (pointer, byte size, dtype, version counter) and holds ONE host copy of
    the whole storage; a save records its view (offset, shape, stride) into
    that copy and the unpack rebuilds exactly that view. Two saves share a
    copy only when they would read identical bytes (an in-place write bumps
    the version counter, which views share, and gets a fresh copy). Entries
    are weak: a host copy lives exactly as long as some saved-tensor payload
    still refers to it, i.e. until the last region that saved a view of it
    has run its backward.

    Attributes:
        _refs (weakref.WeakValueDictionary[tuple, _OffloadRef]): The live
            host copies by storage key.
    """

    def __init__(self) -> None:
        """Start empty."""

        self._refs: weakref.WeakValueDictionary[tuple[Any, ...], _OffloadRef] = (
            weakref.WeakValueDictionary()
        )

    @staticmethod
    def key(tensor: Tensor) -> tuple[Any, ...]:
        """The identity under which a saved tensor's storage bytes are shared.

        Args:
            tensor (Tensor): The tensor autograd wants to save.

        Returns:
            tuple[Any, ...]: ``(storage pointer, storage bytes, dtype, version)``.
        """

        storage = tensor.untyped_storage()
        return (storage.data_ptr(), storage.nbytes(), tensor.dtype, tensor._version)

    def get(self, tensor: Tensor) -> _OffloadRef | None:
        """Return the live host copy of ``tensor``'s storage, if any.

        Args:
            tensor (Tensor): The tensor autograd wants to save.

        Returns:
            _OffloadRef | None: An existing handle, or None.
        """

        return self._refs.get(self.key(tensor))

    def put(self, tensor: Tensor, ref: _OffloadRef) -> None:
        """Register ``ref`` as the host copy of ``tensor``'s storage.

        Args:
            tensor (Tensor): A tensor over the storage that was just copied.
            ref (_OffloadRef): The host handle of the whole storage.
        """

        self._refs[self.key(tensor)] = ref

    def __len__(self) -> int:
        """int: Live entries (weak: drops as payloads are released)."""

        return len(self._refs)


class _OffloadView:
    """A saved tensor as a view into a shared host copy of its storage.

    Attributes:
        ref (_OffloadRef): The storage's host copy (shared with other views).
        offset (int): Storage offset of the view, in elements.
        shape (tuple[int, ...]): View shape.
        stride (tuple[int, ...]): View strides, in elements.
    """

    __slots__ = ("ref", "offset", "shape", "stride")

    def __init__(self, ref: _OffloadRef, tensor: Tensor) -> None:
        """Record the view geometry of ``tensor`` over ``ref``'s storage copy.

        Args:
            ref (_OffloadRef): Host copy of the whole storage.
            tensor (Tensor): The saved view.
        """

        self.ref = ref
        self.offset = tensor.storage_offset()
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())

    def restore(self) -> Tensor:
        """Rebuild the view on its device (the bytes of the saved view only).

        Returns:
            Tensor: The saved values, contiguous, on the original device.
        """

        host_view = torch.as_strided(self.ref.host, self.shape, self.stride, self.offset)
        if not host_view.is_contiguous():
            # A strided H2D copy would gather element-wise; gather on the
            # host once (the slice's bytes only) and ship one block.
            host_view = host_view.contiguous()
        return host_view.to(self.ref.device, non_blocking=host_view.is_pinned())


def _storage_as_flat(tensor: Tensor) -> Tensor:
    """A 1-D view of ``tensor``'s whole storage in its dtype.

    Args:
        tensor (Tensor): Any tensor.

    Returns:
        Tensor: ``[storage bytes / element size]`` over the same storage.
    """

    storage = tensor.untyped_storage()
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    flat.set_(storage, 0, (storage.nbytes() // tensor.element_size(),))
    return flat


def _default_threshold_bytes() -> int:
    """Return the allocated-bytes level above which spilling starts.

    Defaults to 80% of the visible device, which leaves the backward its
    workspace; ``NESTOR_ACTIVATION_OFFLOAD_GIB`` overrides it.

    Returns:
        int: Threshold in bytes.
    """

    override = os.environ.get("NESTOR_ACTIVATION_OFFLOAD_GIB")
    if override:
        return int(float(override) * GIB)
    if not torch.cuda.is_available():
        return 1 << 62  # effectively never
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return int(total * 0.8)


def activation_offload(
    stats: ActivationOffloadStats | None = None,
) -> ContextManager[Any]:
    """Offload saved activations while the device is above the threshold.

    Returns a no-op context unless ``NESTOR_ACTIVATION_OFFLOAD=1``, so callers
    can wrap a forward unconditionally.

    The predicate is evaluated per saved tensor, at pack time, against live
    allocated memory. A step that stays under the threshold therefore keeps
    every activation on device and pays only the predicate; a step that climbs
    past it spills from that point on, which is where the pressure is.

    Args:
        stats (ActivationOffloadStats | None): Collector to record what moved.
            None allocates a throwaway one.

    Returns:
        ContextManager[Any]: Context to wrap the forward (and only the forward
        -- the unpack hook fires during the backward regardless).
    """

    if os.environ.get("NESTOR_ACTIVATION_OFFLOAD", "0") != "1":
        return nullcontext()

    tally = stats if stats is not None else ActivationOffloadStats()
    threshold = _default_threshold_bytes()
    floor = int(float(os.environ.get(
        "NESTOR_ACTIVATION_OFFLOAD_MIN_MIB", "1"
    )) * MIB)
    # Hard cap on pinned host memory. Without one this is not merely slow, it
    # is unsafe: left uncapped on the schedule arms it pinned its way through a
    # 300 GiB cgroup and the job was OOM-killed by the host, taking the whole
    # step down rather than degrading. Past the cap, pack passes through and the
    # step fails on the GPU the way it would have anyway.
    budget = int(float(os.environ.get(
        "NESTOR_ACTIVATION_OFFLOAD_MAX_GIB", "16"
    )) * GIB)

    cache = _HostCopyCache()

    def pack(tensor: Tensor) -> Tensor | _OffloadRef:
        """Copy one saved tensor to the host if the device is under pressure.

        A tensor whose bytes are already on the host (saved before by another
        region) reuses that copy regardless of the threshold and budget: it
        costs no new host memory and lets the device copy go.

        Args:
            tensor (Tensor): Tensor the autograd graph wants to save.

        Returns:
            Tensor | _OffloadRef: The tensor itself, or a host-resident handle.
        """

        if not tensor.is_cuda or tensor.numel() * tensor.element_size() < floor:
            tally.passed += 1
            return tensor
        shared = cache.get(tensor)
        if shared is not None:
            tally.shared += 1
            return _OffloadView(shared, tensor)
        if torch.cuda.memory_allocated() < threshold or tally.bytes_moved >= budget:
            tally.passed += 1
            return tensor
        view_bytes = tensor.numel() * tensor.element_size()
        storage_bytes = tensor.untyped_storage().nbytes()
        if storage_bytes <= _SHARE_STORAGE_RATIO * view_bytes:
            # Spill the whole storage once; this and every later view of it
            # (the other depth sources of the same block, the expert-half
            # chunks' slices) share the copy.
            flat = _storage_as_flat(tensor)
            host = torch.empty(flat.shape, dtype=flat.dtype, device="cpu", pin_memory=True)
            host.copy_(flat, non_blocking=True)
            tally.packed += 1
            tally.bytes_moved += storage_bytes
            ref = _OffloadRef(host, tensor.device)
            cache.put(tensor, ref)
            return _OffloadView(ref, tensor)
        # A small view of a much larger buffer: spill just the view.
        host = torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
        )
        host.copy_(tensor, non_blocking=True)
        tally.packed += 1
        tally.bytes_moved += view_bytes
        return _OffloadRef(host, tensor.device)

    def unpack(payload: Tensor | _OffloadRef) -> Tensor:
        """Restore a saved tensor for the backward.

        Args:
            payload (Tensor | _OffloadRef): Whatever ``pack`` returned.

        Returns:
            Tensor: The tensor, back on its original device.
        """

        if isinstance(payload, _OffloadView):
            return payload.restore()
        if isinstance(payload, _OffloadRef):
            return payload.host.to(payload.device, non_blocking=True)
        return payload

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
