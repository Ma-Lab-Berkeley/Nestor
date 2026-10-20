"""Tests for the OOM-time allocator snapshot (nestor/train/oom_snapshot.py).

CPU-only: the dumper's once-only / never-raise contract with a real file
writer, and the installer's inertness without the env variable or CUDA.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import torch

from nestor.train.oom_snapshot import OomSnapshotDumper, install_oom_snapshot


def _write_marker(path: str) -> None:
    """A stand-in snapshot writer: writes the path's name into the file."""

    Path(path).write_text(Path(path).name)


class OomSnapshotDumperTest:
    """OomSnapshotDumper: dumps once, reports the observer args, never raises."""

    def test_dumps_once_with_observer_args(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "diag_oom_rank0.pickle"
        dumper = OomSnapshotDumper(path, _write_marker, logging.getLogger("t"))
        assert dumper.fired is False
        dumper(0, 3 * 2**30, 131 * 2**30, 3 * 2**30)
        assert dumper.fired is True
        assert path.read_text() == "diag_oom_rank0.pickle"
        # A second OOM (the allocator retries) must not rewrite or raise.
        path.write_text("kept")
        dumper(0, 1, 1, 1)
        assert path.read_text() == "kept"

    def test_dumps_without_args_from_the_exception_path(self, tmp_path: Path) -> None:
        path = tmp_path / "diag_oom_rank3.pickle"
        dumper = OomSnapshotDumper(path, _write_marker)
        dumper()
        assert path.exists() and dumper.fired

    def test_suspended_calls_do_not_consume_the_shot(self, tmp_path: Path) -> None:
        path = tmp_path / "diag_oom_rank0.pickle"
        dumper = OomSnapshotDumper(path, _write_marker)
        dumper.suspended = True
        dumper(0, 1, 1, 1)  # the calibration's caught OOM
        assert not dumper.fired and not path.exists()
        dumper.suspended = False
        dumper(0, 1, 1, 1)  # the real one
        assert dumper.fired and path.exists()

    def test_a_failing_writer_is_swallowed(self, tmp_path: Path) -> None:
        def boom(_: str) -> None:
            raise OSError("disk full")

        dumper = OomSnapshotDumper(tmp_path / "x.pickle", boom)
        dumper(0, 1, 1, 1)  # must not raise: the OOM itself is what matters
        assert dumper.fired


class InstallOomSnapshotTest:
    """install_oom_snapshot: inert unless NESTOR_DIAG_OOM_SNAPSHOT=1 and CUDA is available."""

    def test_inert_without_env(self, tmp_path: Path) -> None:
        saved = os.environ.pop("NESTOR_DIAG_OOM_SNAPSHOT", None)
        try:
            assert install_oom_snapshot(tmp_path, 0) is None
        finally:
            if saved is not None:
                os.environ["NESTOR_DIAG_OOM_SNAPSHOT"] = saved

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only contract")
    def test_inert_without_cuda_even_with_env(self, tmp_path: Path) -> None:
        saved = os.environ.get("NESTOR_DIAG_OOM_SNAPSHOT")
        os.environ["NESTOR_DIAG_OOM_SNAPSHOT"] = "1"
        try:
            assert install_oom_snapshot(tmp_path, 0) is None
        finally:
            if saved is None:
                del os.environ["NESTOR_DIAG_OOM_SNAPSHOT"]
            else:
                os.environ["NESTOR_DIAG_OOM_SNAPSHOT"] = saved
