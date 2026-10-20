"""Distributed data-loading regression tests."""

from __future__ import annotations

import socket

from nestor.configs.distributed import DistributedConfig
from nestor.utils.distributed import (
    cleanup_distributed,
    get_per_rank_batch_size,
    init_distributed,
    is_distributed,
)


def _find_free_port() -> int:
    """Find a free localhost TCP port for distributed test setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class DistributedBatchSizingTest:
    """Tests for global-to-local batch-size conversion."""

    def test_returns_same_batch_size_for_single_process(self) -> None:
        """Single-process runs should keep the configured batch size."""
        assert get_per_rank_batch_size(512, world_size=1) == 512

    def test_divides_global_batch_size_evenly(self) -> None:
        """Distributed runs should divide the global batch size by world size."""
        assert get_per_rank_batch_size(512, world_size=2) == 256
        assert get_per_rank_batch_size(768, world_size=3) == 256

    def test_raises_for_uneven_batch_size_split(self) -> None:
        """Global batch size must divide evenly across ranks."""
        try:
            get_per_rank_batch_size(513, world_size=2)
        except ValueError as e:
            assert "divisible" in str(e)
        else:
            raise AssertionError(
                "Expected get_per_rank_batch_size to raise ValueError."
            )


class DistributedLifecycleTest:
    """Tests for process-group lifecycle alignment with job config."""

    def test_disabled_init_cleans_existing_process_group(self) -> None:
        """Calling init with enabled=False should clean up stale distributed state."""
        port = _find_free_port()
        try:
            init_distributed(
                DistributedConfig(
                    enabled=True,
                    backend="gloo",
                    init_method=f"tcp://127.0.0.1:{port}",
                    rank=0,
                    local_rank=0,
                    world_size=1,
                )
            )
            assert is_distributed()

            init_distributed(DistributedConfig(enabled=False))
            assert not is_distributed()
        finally:
            cleanup_distributed()
