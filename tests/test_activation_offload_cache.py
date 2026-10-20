"""The activation offload's host-copy cache (nestor/train/activation_offload.py).

A buffer saved by several recompute regions -- the column backbone's depth
sources, views of a block's exported ``[batch, seq, k + 1, dim]`` buffer,
saved whole by every attention half and as sequence slices by every
expert-half chunk -- must be copied to the host once, not once per save
(1b at 128k: 2.4 TB of host RSS, OOM-killed). The cache keys on the STORAGE
plus the version counter, so every view of identical bytes shares one copy,
and a saved view is restored from that copy exactly. CPU-only: the key, the
cache and the view restore are exercised directly.
"""

from __future__ import annotations

import gc

import torch

from nestor.train.activation_offload import (
    _HostCopyCache,
    _OffloadRef,
    _OffloadView,
    _storage_as_flat,
)


def _host_copy(tensor: torch.Tensor) -> _OffloadRef:
    """What ``pack`` builds on a miss: a copy of the whole storage."""

    return _OffloadRef(_storage_as_flat(tensor).clone(), tensor.device)


class HostCopyCacheTest:
    """_HostCopyCache: every view of one storage -> one copy; another storage / a write -> another."""

    def test_views_of_one_storage_share_one_copy(self) -> None:
        cache = _HostCopyCache()
        x = torch.randn(1, 8, 3, 4)          # a block's export: [batch, seq, k + 1, dim]
        assert cache.get(x) is None
        ref = _host_copy(x)
        cache.put(x, ref)
        for view in (x, x.unbind(dim=2)[1], x.unbind(dim=2)[2][:, 2:5], x.view(8, 12), x[:, 3:]):
            assert cache.get(view) is ref
        assert len(cache) == 1

    def test_other_storage_or_dtype_does_not_share(self) -> None:
        cache = _HostCopyCache()
        x = torch.randn(4, 3)
        cache.put(x, _host_copy(x))
        assert cache.get(x.clone()) is None                  # another storage
        assert cache.get(x.to(torch.float64)) is None        # another dtype (and storage)

    def test_in_place_write_gets_a_fresh_copy(self) -> None:
        cache = _HostCopyCache()
        x = torch.randn(4, 3)
        ref = _host_copy(x)
        cache.put(x, ref)
        x[1:].add_(1.0)                        # bumps the version counter shared by every view
        assert cache.get(x) is None and cache.get(x[:2]) is None
        ref2 = _host_copy(x)
        cache.put(x, ref2)
        assert cache.get(x) is ref2 and ref2 is not ref

    def test_entries_are_weak(self) -> None:
        cache = _HostCopyCache()
        x = torch.randn(4, 3)
        ref = _host_copy(x)
        cache.put(x, ref)
        assert len(cache) == 1
        del ref
        gc.collect()
        assert cache.get(x) is None and len(cache) == 0


class OffloadViewRestoreTest:
    """_OffloadView.restore: a saved view comes back with exactly its values."""

    def test_restores_whole_slices_and_unbind_views(self) -> None:
        x = torch.randn(1, 8, 3, 4)
        ref = _host_copy(x)
        for view in (x, x[:, 2:6], x.unbind(dim=2)[2], x.unbind(dim=2)[0][:, 3:7], x.view(8, 12)[1:5]):
            restored = _OffloadView(ref, view).restore()
            assert restored.shape == view.shape
            torch.testing.assert_close(restored, view)
            assert restored.is_contiguous()

    def test_restore_reads_the_host_copy_not_the_live_tensor(self) -> None:
        x = torch.randn(4, 3)
        ref = _host_copy(x)
        payload = _OffloadView(ref, x[1:3])
        expected = x[1:3].clone()
        x.zero_()
        torch.testing.assert_close(payload.restore(), expected)
