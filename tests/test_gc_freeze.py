"""Tests for the trainer's garbage-collector freeze (``NESTOR_GC_FREEZE``).

:func:`nestor.train.trainer.freeze_gc` runs one full collection and then moves
every surviving object into the cyclic collector's permanent generation, so
that later collections in the training loop no longer traverse the long-lived
objects of the run (module tree, optimizer state, compiled frames).

The tests use the real ``gc`` module (no mocks) and pin the three properties
the trainer relies on:

* garbage pending at the call is collected, not frozen (a frozen object is
  never reclaimed by the cyclic collector);
* objects alive at the call are exempt from later collections;
* collection stays enabled: cyclic garbage created afterwards is reclaimed.

Every test undoes the freeze (``gc.unfreeze``) so the test process is left as
it was found.
"""

from __future__ import annotations

import gc
import weakref

from nestor.train.trainer import freeze_gc


class _Node:
    """A plain object that can sit in a reference cycle and be weak-referenced.

    Attributes:
        other (_Node | None): The cycle partner.
    """

    def __init__(self) -> None:
        self.other: _Node | None = None


def _cycle() -> tuple[_Node, weakref.ref]:
    """Two nodes pointing at each other.

    Returns:
        tuple[_Node, weakref.ref]: One node of the cycle and a weak reference
            to it (dead once the cycle has been reclaimed).
    """

    first, second = _Node(), _Node()
    first.other, second.other = second, first
    return first, weakref.ref(first)


class FreezeGcTest:
    """``freeze_gc`` = collect, then freeze; collection stays on."""

    def test_returns_permanent_generation_size(self) -> None:
        """The return value is the collector's own freeze count, and positive."""

        try:
            frozen = freeze_gc()
            assert frozen == gc.get_freeze_count()
            assert frozen > 0
            assert gc.isenabled(), "freeze_gc must not disable collection"
        finally:
            gc.unfreeze()
        assert gc.get_freeze_count() == 0

    def test_pending_garbage_is_collected_not_frozen(self) -> None:
        """An unreachable cycle that exists at the call is reclaimed by it."""

        was_enabled = gc.isenabled()
        gc.disable()  # keep the cycle pending until freeze_gc itself collects
        try:
            node, ref = _cycle()
            del node
            assert ref() is not None, "the cycle should still be pending"
            freeze_gc()
            assert ref() is None, "pending garbage was frozen instead of collected"
        finally:
            gc.unfreeze()
            if was_enabled:
                gc.enable()

    def test_objects_alive_at_the_call_are_exempt_afterwards(self) -> None:
        """A cycle alive at the freeze outlives later collections once dropped.

        This is the behaviour that makes the freeze cheap -- the collector no
        longer looks at these objects -- and the reason it runs once, after
        the first step, on objects that live for the whole run. ``unfreeze``
        hands the cycle back and the next collection reclaims it.
        """

        node, ref = _cycle()
        try:
            freeze_gc()
            del node
            gc.collect()
            assert ref() is not None, "a frozen object was traversed and collected"
        finally:
            gc.unfreeze()
        gc.collect()
        assert ref() is None

    def test_garbage_created_after_the_freeze_is_still_collected(self) -> None:
        """Collection keeps working for objects created after the freeze."""

        try:
            freeze_gc()
            node, ref = _cycle()
            del node
            gc.collect()
            assert ref() is None
        finally:
            gc.unfreeze()
