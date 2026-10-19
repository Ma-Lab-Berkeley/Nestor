"""Allocator snapshot at the moment of a CUDA out-of-memory error.

A memory snapshot taken after a *completed* step attributes the live bytes at
that step's peak. A run that never completes a step -- the 500m column arm at
128k, which OOMs in the first backward -- leaves nothing to attribute, and a
snapshot from a shorter length does not extrapolate when some item is
superlinear in the sequence length.

``NESTOR_DIAG_OOM_SNAPSHOT=1`` closes that gap: the allocator history is
recorded from the start of training and, the first time the caching allocator
fails an allocation, a snapshot is dumped to
``<run_dir>/diag_oom_rank<r>.pickle`` on every rank (the observer fires inside
the allocator, before the ``OutOfMemoryError`` propagates, so the live blocks
are exactly the ones that did not fit). Load it with
``torch.cuda.memory._snapshot`` tooling: the instant of maximum live
bytes in the trace IS the OOM. Diagnostic only -- it changes no arithmetic and
is inert unless the variable is set.

Exact contract of the observer callback (torch ``_cuda_attach_out_of_memory_observer``):
``(device: int, alloc: int, device_allocated: int, device_free: int) -> None``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import torch

MAX_ENTRIES = 400_000
"""int: Allocator-history ring size (alloc / free events with stacks)."""


class OomSnapshotDumper:
    """Dump one allocator snapshot the first time it is invoked.

    Attributes:
        path (Path): Where the snapshot pickle goes.
        dump_fn (Callable[[str], None]): Writes the snapshot given the path
            (``torch.cuda.memory._dump_snapshot`` in production; a test passes
            its own).
        logger (logging.Logger): Where the outcome is reported.
        fired (bool): True once a dump was attempted (later calls are no-ops:
            the allocator may retry and fail several times per OOM).
        suspended (bool): While True, calls are ignored WITHOUT consuming the
            one shot -- the trainer sets it around the MFU calibration, whose
            probe forward+backward OOMs by design at long S and is caught
            (early runs spent the shot there).
    """

    def __init__(
        self,
        path: Path,
        dump_fn: Callable[[str], None],
        logger: logging.Logger | None = None,
    ) -> None:
        self.path = path
        self.dump_fn = dump_fn
        self.logger = logger or logging.getLogger(__name__)
        self.fired = False
        self.suspended = False

    def __call__(self, *args: object) -> None:
        """Dump once; swallow every error (a diagnostic must never mask the OOM).

        Args:
            *args (object): The observer's ``(device, alloc, device_allocated,
                device_free)`` -- logged when present, otherwise ignored.
        """

        if self.fired or self.suspended:
            return
        self.fired = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.dump_fn(str(self.path))
            if len(args) == 4:
                device, alloc, allocated, free = args
                self.logger.warning(
                    "[OOM-SNAPSHOT] device %s: failed to allocate %.2f GiB with %.2f GiB "
                    "allocated and %.2f GiB free -> %s",
                    device, float(alloc) / 2**30, float(allocated) / 2**30,
                    float(free) / 2**30, self.path,
                )
            else:
                self.logger.warning("[OOM-SNAPSHOT] snapshot -> %s", self.path)
        except Exception as exc:  # noqa: BLE001 -- never break the OOM path
            self.logger.warning("[OOM-SNAPSHOT] dump failed: %s", exc)


def install_oom_snapshot(
    run_dir: Path,
    rank: int,
    logger: logging.Logger | None = None,
) -> OomSnapshotDumper | None:
    """Arm the OOM-time snapshot if ``NESTOR_DIAG_OOM_SNAPSHOT=1`` and CUDA is up.

    Records the allocator history (``MAX_ENTRIES`` events with Python stacks)
    and attaches an out-of-memory observer that dumps
    ``run_dir / f"diag_oom_rank{rank}.pickle"`` once. Returns the dumper so the
    caller can also invoke it from an ``except torch.OutOfMemoryError`` path
    (the fallback when this torch build has no observer hook).

    Args:
        run_dir (Path): The run directory the snapshot goes into.
        rank (int): This process's rank (one file per rank).
        logger (logging.Logger | None): Reporting.

    Returns:
        OomSnapshotDumper | None: The armed dumper, or None when inert (env
        unset, no CUDA, or the history API missing).
    """

    log = logger or logging.getLogger(__name__)
    if os.environ.get("NESTOR_DIAG_OOM_SNAPSHOT") != "1" or not torch.cuda.is_available():
        return None
    record = getattr(torch.cuda.memory, "_record_memory_history", None)
    dump = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if record is None or dump is None:
        log.warning("[OOM-SNAPSHOT] torch %s has no allocator history API; inert", torch.__version__)
        return None
    record(max_entries=MAX_ENTRIES)
    dumper = OomSnapshotDumper(run_dir / f"diag_oom_rank{rank}.pickle", dump, log)
    attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
    if attach is not None:
        attach(dumper)
        log.info("[OOM-SNAPSHOT] armed (observer) -> %s", dumper.path)
    else:
        log.info("[OOM-SNAPSHOT] armed (exception path only; no observer in torch %s) -> %s",
                 torch.__version__, dumper.path)
    return dumper
