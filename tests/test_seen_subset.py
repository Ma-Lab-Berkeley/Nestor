"""Tests for ``nestor.data.seen_subset``.

The builder claims the union of windows a fresh multi-rank run consumes by
optimizer step ``T`` equals the epoch-0 ``DistributedSampler`` permutation
prefix ``perm[: T * seqs_per_step]``, independent of world size / per-rank
batch / grad accumulation. The reference implementation here IS torch's
``DistributedSampler`` + ``DataLoader``: it simulates every rank's loader and
consumes micro-batches exactly as the trainer does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nestor.data.seen_subset import (
    build_replay_mix,
    build_seen_subset_bin,
    multiset_extra_indices,
    seen_window_indices,
)


class _IndexDataset(Dataset):
    """Map-style dataset whose item ``i`` is the index ``i`` itself.

    Attributes:
        length (int): Number of items.
    """

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


def _reference_consumption_order(
    num_windows: int, world_size: int, per_rank_batch: int, grad_accum: int, steps: int
) -> list[tuple[int, int]]:
    """(window, optimizer step) pairs consumed, via the real sampler + loader.

    Mirrors the training pipeline's loader (``DistributedSampler`` at the
    default seed, per-rank ``DataLoader`` with ``drop_last=True``) and the
    trainer (``grad_accum`` micro-batches per optimizer step per rank, stopping
    mid-epoch at ``max_steps``).

    Args:
        num_windows (int): Dataset length.
        world_size (int): Ranks.
        per_rank_batch (int): Per-rank micro-batch size.
        grad_accum (int): Micro-batches per optimizer step per rank.
        steps (int): Optimizer steps trained.

    Returns:
        list[tuple[int, int]]: One ``(window index, step)`` per consumed row,
            all ranks pooled.
    """

    rows: list[tuple[int, int]] = []
    for rank in range(world_size):
        sampler = DistributedSampler(_IndexDataset(num_windows), num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(_IndexDataset(num_windows), batch_size=per_rank_batch, sampler=sampler, drop_last=True)
        for batch_number, batch in enumerate(loader):
            if batch_number >= steps * grad_accum:
                break
            rows.extend((int(i), batch_number // grad_accum) for i in batch)
    return rows


def _reference_consumed_windows(
    num_windows: int, world_size: int, per_rank_batch: int, grad_accum: int, steps: int
) -> set[int]:
    """Union of the windows consumed by step ``steps``, via the real sampler + loader."""

    return {w for w, _ in _reference_consumption_order(num_windows, world_size, per_rank_batch, grad_accum, steps)}


def _random_bin(path: Path, num_windows: int, block_size: int, tail: int, seed: int) -> np.ndarray:
    """Write a random uint32 token bin of ``num_windows`` windows plus an unaligned tail."""

    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 100_000, size=num_windows * block_size + tail, dtype=np.uint32)  # (num_tokens,)
    tokens.tofile(path)
    return tokens


class SeenWindowIndicesTest:
    """``seen_window_indices`` == what DistributedSampler-driven ranks consume."""

    @pytest.mark.parametrize(
        ("world_size", "per_rank_batch", "grad_accum"),
        [
            (4, 4, 8),  # 8k: 128 sequences per step on 4 ranks
            (4, 2, 8),  # 16k: 64
            (4, 1, 8),  # 32k: 32
            (2, 1, 8),  # 64k: 16
            (8, 1, 4),  # 32k on a whole node
            (3, 2, 2),  # padding case: num_windows % world != 0
        ],
    )
    def test_matches_real_sampler_consumption(self, world_size: int, per_rank_batch: int, grad_accum: int) -> None:
        num_windows = 997  # prime: exercises sampler padding for every world
        seqs_per_step = world_size * per_rank_batch * grad_accum
        steps = num_windows // (2 * seqs_per_step)  # a mid-epoch stop
        assert steps >= 2
        reference = _reference_consumed_windows(num_windows, world_size, per_rank_batch, grad_accum, steps)
        claimed = seen_window_indices(num_windows, steps, seqs_per_step)
        assert len(claimed) == steps * seqs_per_step
        assert set(claimed.tolist()) == reference

    def test_prefix_is_world_shape_invariant(self) -> None:
        # The same (steps, seqs_per_step) consumed under different world shapes
        # yields the same SET: the runs of every model at a length saw the same
        # windows whatever their GPU count and per-rank batch.
        num_windows = 1024
        claimed = set(seen_window_indices(num_windows, 5, 32).tolist())
        for world_size, per_rank_batch in [(4, 1), (2, 2), (8, 1), (16, 2)]:
            grad_accum = 32 // (world_size * per_rank_batch)
            assert _reference_consumed_windows(num_windows, world_size, per_rank_batch, grad_accum, 5) == claimed

    def test_wrap_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="wrapped"):
            seen_window_indices(num_windows=10, steps=2, seqs_per_step=8)


class BuildSeenSubsetBinTest:
    """The written bin holds exactly the seen windows' tokens plus a sane sidecar."""

    def test_written_tokens_match_selected_windows(self, tmp_path: Path) -> None:
        block_size, num_windows = 8, 200
        source = tmp_path / "full.bin"
        tokens = _random_bin(source, num_windows, block_size, tail=5, seed=7)
        # steps = ceil(400 / 32) = 13 of 4 sequences -> 52 windows
        out = tmp_path / "capped.bin"
        meta = build_seen_subset_bin(source, out, block_size=block_size, cpt_tokens=400, tokens_per_step=32)
        expected_indices = np.sort(seen_window_indices(num_windows, 13, 4))
        written = np.fromfile(out, dtype=np.uint32).reshape(-1, block_size)  # (52, block_size)
        expected = np.stack([tokens[i * block_size : (i + 1) * block_size] for i in expected_indices])
        assert written.shape == (52, block_size)
        assert np.array_equal(written, expected)
        sidecar = json.loads((tmp_path / "capped.bin.meta.json").read_text())
        assert sidecar["num_tokens"] == 52 * block_size == meta["num_tokens"]
        assert sidecar["num_windows"] == 52 and sidecar["unique_windows"] == 52
        assert sidecar["steps"] == 13 and sidecar["seqs_per_step"] == 4
        assert sidecar["source_num_windows"] == num_windows
        assert sidecar["token_dtype"] == "uint32" and sidecar["eot_token"] == 100257

    def test_rejects_misaligned_tokens_per_step(self, tmp_path: Path) -> None:
        source = tmp_path / "full.bin"
        np.zeros(64, dtype=np.uint32).tofile(source)
        with pytest.raises(ValueError, match="not a multiple"):
            build_seen_subset_bin(source, tmp_path / "capped.bin", block_size=8, cpt_tokens=32, tokens_per_step=12)


class MultisetSubsetBinTest:
    """Repeated documents: the bin = seen set + spread second copies; ``T`` steps = one exact epoch."""

    def test_multiset_holds_seen_set_plus_second_copies(self, tmp_path: Path) -> None:
        block_size, num_windows = 8, 4000
        source = tmp_path / "full.bin"
        tokens = _random_bin(source, num_windows, block_size, tail=3, seed=3)
        # steps = ceil(3200 / 32) = 100 of 4 sequences -> 400 seen; T = 150 -> 200 extras
        out = tmp_path / "ms.bin"
        meta = build_seen_subset_bin(
            source, out, block_size=block_size, cpt_tokens=3200, tokens_per_step=32, multiset_steps=150
        )
        seen = np.sort(seen_window_indices(num_windows, 100, 4))
        written = np.fromfile(out, dtype=np.uint32).reshape(-1, block_size)
        assert written.shape == (150 * 4, block_size)
        # Recover window indices from the tokens (windows are distinct with
        # overwhelming probability at this token range) and count copies.
        by_content = {tokens[i * block_size : (i + 1) * block_size].tobytes(): int(i) for i in seen}
        recovered = [by_content[row.tobytes()] for row in written]
        counts = {w: recovered.count(w) for w in set(recovered)}
        assert set(counts) == set(seen.tolist())
        assert max(counts.values()) == 2
        assert sum(c == 2 for c in counts.values()) == 200
        assert meta["unique_windows"] == 400 and meta["extra_windows"] == 200
        assert meta["multiset_steps"] == 150 and meta["num_windows"] == 600

    def test_extras_are_seen_windows_without_replacement(self) -> None:
        seen = np.arange(10, 50, dtype=np.int64)
        extra = multiset_extra_indices(seen, 25, seed=1)
        assert extra.size == 25 and len(set(extra.tolist())) == 25
        assert set(extra.tolist()) <= set(seen.tolist())
        assert np.array_equal(extra, np.sort(extra))
        assert np.array_equal(extra, multiset_extra_indices(seen, 25, seed=1))  # reproducible
        with pytest.raises(ValueError, match="third copy"):
            multiset_extra_indices(seen, 41)

    def test_multiset_below_the_seen_steps_is_rejected(self, tmp_path: Path) -> None:
        source = tmp_path / "full.bin"
        _random_bin(source, 400, 8, tail=0, seed=5)
        with pytest.raises(ValueError, match="multiset_steps"):
            build_seen_subset_bin(
                source, tmp_path / "ms.bin", block_size=8, cpt_tokens=3200, tokens_per_step=32, multiset_steps=50
            )

    @pytest.mark.parametrize(("world_size", "per_rank_batch", "grad_accum"), [(4, 1, 32), (8, 1, 16), (4, 4, 8), (2, 1, 8)])
    def test_t_steps_is_one_exact_epoch(self, world_size: int, per_rank_batch: int, grad_accum: int) -> None:
        # A T-step run over the multiset bin consumes every row exactly once
        # under every world shape the launchers use (seqs_per_step fixed).
        seqs = world_size * per_rank_batch * grad_accum
        t = 27
        rows = _reference_consumption_order(t * seqs, world_size, per_rank_batch, grad_accum, t)
        assert sorted(w for w, _ in rows) == list(range(t * seqs))

    def test_second_copies_are_spread_not_appended(self, tmp_path: Path) -> None:
        # Under the trainer's own permutation of the multiset bin the two
        # copies of a window land at independent uniform positions: the
        # fraction of pairs with BOTH copies in the last 10 % of steps is
        # ~1 % (an appended second pass puts 100 % of them there), and the
        # mean step gap is ~T / 3.
        block_size, num_windows = 4, 20000
        source = tmp_path / "full.bin"
        _random_bin(source, num_windows, block_size, tail=0, seed=11)
        seqs, steps, t = 16, 500, 600  # 8000 seen rows + 1600 second copies
        out = tmp_path / "ms.bin"
        build_seen_subset_bin(
            source, out, block_size=block_size, cpt_tokens=steps * seqs * block_size,
            tokens_per_step=seqs * block_size, multiset_steps=t,
        )
        written = np.fromfile(out, dtype=np.uint32).reshape(-1, block_size)
        first_row = {written[r].tobytes(): r for r in range(steps * seqs)}
        pairs = [(first_row[written[r].tobytes()], r) for r in range(steps * seqs, t * seqs)]
        step_of = dict(_reference_consumption_order(t * seqs, 4, 1, seqs // 4, t))
        tail = int(0.9 * t)
        both_in_tail = np.mean([step_of[a] >= tail and step_of[b] >= tail for a, b in pairs])
        gaps = np.array([abs(step_of[a] - step_of[b]) for a, b in pairs])
        assert both_in_tail < 0.03
        assert abs(gaps.mean() - t / 3) < 0.05 * t


class ReplayMixTest:
    """A replay mixture = the target prefix followed by the replay prefix, both seen sets."""

    def test_mixture_is_the_two_prefixes(self, tmp_path: Path) -> None:
        block_size, tps = 8, 32  # 4 sequences per step
        target = tmp_path / "target.bin"
        replay = tmp_path / "replay.bin"
        target_tokens = _random_bin(target, 300, block_size, tail=2, seed=21)
        replay_tokens = _random_bin(replay, 400, block_size, tail=6, seed=22)
        out = tmp_path / "mix.bin"
        meta = build_replay_mix(
            target, replay, out, block_size=block_size, target_share_pct=75, mix_tokens=1600, tokens_per_step=tps
        )
        # target: 1200 tokens -> 38 steps -> 152 windows; replay: 400 tokens -> 13 steps -> 52 windows
        t_idx = np.sort(seen_window_indices(300, 38, 4))
        r_idx = np.sort(seen_window_indices(400, 13, 4))
        expected = np.concatenate(
            [
                np.stack([target_tokens[i * block_size : (i + 1) * block_size] for i in t_idx]),
                np.stack([replay_tokens[i * block_size : (i + 1) * block_size] for i in r_idx]),
            ]
        )  # (204, block_size)
        written = np.fromfile(out, dtype=np.uint32).reshape(-1, block_size)
        assert np.array_equal(written, expected)
        assert meta["num_windows"] == 204 and meta["num_tokens"] == 204 * block_size
        parts = {p["part"]: p for p in meta["sources"]}
        assert parts["target"]["windows"] == 152 and parts["target"]["steps"] == 38
        assert parts["replay"]["windows"] == 52 and parts["replay"]["steps"] == 13
        assert parts["target"]["share_pct"] == 75 and parts["replay"]["share_pct"] == 25
        assert abs(152 / 204 - 0.75) < 0.01
        sidecar = json.loads((tmp_path / "mix.bin.meta.json").read_text())
        assert sidecar["target_share_pct"] == 75 and sidecar["num_windows"] == 204

    def test_shares_outside_the_open_interval_are_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "t.bin"
        replay = tmp_path / "r.bin"
        _random_bin(target, 100, 8, tail=0, seed=1)
        _random_bin(replay, 100, 8, tail=0, seed=2)
        for share in (0, 100, -5):
            with pytest.raises(ValueError, match="strictly between"):
                build_replay_mix(target, replay, tmp_path / "m.bin", block_size=8, target_share_pct=share, mix_tokens=800, tokens_per_step=32)
