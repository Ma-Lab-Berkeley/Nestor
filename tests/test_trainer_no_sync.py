"""Tests for deferred DDP gradient synchronization under accumulation.

:func:`nestor.train.trainer.grad_sync_context` wraps each micro-batch's
forward/backward: under ``NESTOR_DDP_NO_SYNC=1`` every micro-batch but the last
of an accumulation window runs inside ``DistributedDataParallel.no_sync()`` so
the window's gradients are all-reduced once. The tests run a real single-rank
gloo process group and a real DDP wrapper (no mocks) and check the context
toggles DDP's ``require_backward_grad_sync`` exactly when it should, and that
the accumulated gradient equals the per-micro-batch-synced one.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nestor.train.trainer import grad_sync_context


@contextlib.contextmanager
def _single_rank_group():
    """A one-process gloo group, torn down afterwards."""

    previous = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class GradSyncContextTest:
    """``grad_sync_context`` defers exactly the non-final micro-batches."""

    def test_null_context_without_ddp(self) -> None:
        """A plain module never gets a no_sync context, flag or not."""

        model = torch.nn.Linear(4, 2)
        for enabled in (False, True):
            for last in (False, True):
                ctx = grad_sync_context(model, enabled, last)
                assert isinstance(ctx, contextlib.nullcontext)

    def test_ddp_toggles_only_when_enabled_and_not_last(self) -> None:
        """Inside the context DDP does not sync; outside it does again."""

        with _single_rank_group():
            model = DDP(torch.nn.Linear(4, 2))
            assert model.require_backward_grad_sync is True
            with grad_sync_context(model, True, last_in_window=False):
                assert model.require_backward_grad_sync is False
            assert model.require_backward_grad_sync is True
            with grad_sync_context(model, True, last_in_window=True):
                assert model.require_backward_grad_sync is True
            with grad_sync_context(model, False, last_in_window=False):
                assert model.require_backward_grad_sync is True

    def test_compiled_wrapper_reaches_inner_ddp(self) -> None:
        """``torch.compile(DDP(m))`` (the pipeline's wrap order) still defers.

        The trainer's model is an ``OptimizedModule`` around the DDP module;
        the context must toggle the INNER DDP's flag, not fall through to a
        null context (which silently reduced every micro-batch).
        Constructing the wrapper traces nothing, so this stays CPU-cheap.
        """

        with _single_rank_group():
            ddp = DDP(torch.nn.Linear(4, 2))
            compiled = torch.compile(ddp)
            assert not isinstance(compiled, DDP)
            assert ddp.require_backward_grad_sync is True
            with grad_sync_context(compiled, True, last_in_window=False):
                assert ddp.require_backward_grad_sync is False
            assert ddp.require_backward_grad_sync is True
            with grad_sync_context(compiled, True, last_in_window=True):
                assert ddp.require_backward_grad_sync is True
            with grad_sync_context(compiled, False, last_in_window=False):
                assert ddp.require_backward_grad_sync is True
            # Compile around a plain module: still nothing to defer.
            plain = torch.compile(torch.nn.Linear(4, 2))
            ctx = grad_sync_context(plain, True, last_in_window=False)
            assert isinstance(ctx, contextlib.nullcontext)

    def test_accumulated_gradient_matches_synced_windows(self) -> None:
        """Deferring the sync leaves the window's gradient unchanged."""

        with _single_rank_group():
            torch.manual_seed(0)
            base = torch.nn.Linear(4, 2)
            synced = DDP(base)
            torch.manual_seed(1)
            batches = [torch.randn(3, 4) for _ in range(4)]

            synced.zero_grad(set_to_none=True)
            for batch in batches:
                with grad_sync_context(synced, False, last_in_window=False):
                    (synced(batch) ** 2).sum().div(4).backward()
            expected = [p.grad.clone() for p in synced.parameters()]

            synced.zero_grad(set_to_none=True)
            for idx, batch in enumerate(batches):
                with grad_sync_context(synced, True, last_in_window=idx == 3):
                    (synced(batch) ** 2).sum().div(4).backward()
            actual = [p.grad.clone() for p in synced.parameters()]
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
