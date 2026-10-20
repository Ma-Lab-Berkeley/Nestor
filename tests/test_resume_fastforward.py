"""Test for the resume dataloader fast-forward arithmetic.

``Trainer.run`` restores ``global_step`` from a checkpoint but ``for batch in
train_loader`` restarts at the first batch of the epoch. For single-epoch
corpora (DCLM ``.bin`` runs) a mid-run resume therefore re-read data from the
start and never reached the untouched half of its budget. The fix skips the
micro-batches already consumed this epoch:

    skip = (global_step * grad_accum_steps) % batches_per_epoch

This test pins that arithmetic's invariants directly, since they are what the
fix relies on:

* the skip is a whole number of optimizer steps (multiple of ``grad_accum``),
  so gradient-accumulation alignment survives the resume;
* it is invariant to per-rank batch-shape changes across legs at fixed
  tokens/step (the exact situation that damaged the 500m pair);
* multi-epoch positions reduce correctly modulo the epoch length.
"""

from __future__ import annotations


def _skip(global_step: int, grad_accum: int, batches_per_epoch: int) -> int:
    """The fix's skip count (mirrors trainer.run)."""

    return (global_step * grad_accum) % batches_per_epoch


class ResumeFastForwardTest:
    """Invariants of the fast-forward arithmetic."""

    def test_skip_is_whole_optimizer_steps(self) -> None:
        """Skip must be a multiple of grad_accum, or accumulation desyncs."""

        for step, accum, bpe in [
            (12000, 2, 200_000),
            (10000, 4, 95_368),
            (2000, 8, 190_736),
            (7, 3, 1000),
        ]:
            if bpe % accum == 0:
                assert _skip(step, accum, bpe) % accum == 0

    def test_invariant_to_batch_shape_at_fixed_tokens_per_step(self) -> None:
        """Legs with different (per-rank batch, accum) but identical
        tokens/step must skip the same number of SAMPLES.

        This is the m3 (per-rank 8, accum 4) -> m5 (per-rank 16, accum 2)
        switch: micro-batch counts differ, sample counts must not.
        """

        step = 12000
        dataset_samples = 25_600_000
        world = 8
        # (global_batch, grad_accum) with global_batch*accum == 256 seq/step
        for gb, ga in [(64, 4), (128, 2), (32, 8), (256, 1)]:
            per_rank = gb // world
            bpe = dataset_samples // world // per_rank  # loader len per rank
            skipped_samples = _skip(step, ga, bpe) * per_rank
            assert skipped_samples == (step * 256 // world) % (
                dataset_samples // world
            ), f"shape ({gb},{ga}) skips a different data position"

    def test_multi_epoch_position_reduces_mod_epoch(self) -> None:
        """After k full epochs the skip equals the within-epoch remainder."""

        bpe, accum = 1000, 2
        for step in (0, 250, 500, 1250, 2750):
            consumed = step * accum
            assert _skip(step, accum, bpe) == consumed % bpe

    def test_fresh_run_skips_nothing(self) -> None:
        """global_step == 0 must not skip a single batch."""

        assert _skip(0, 8, 123_456) == 0
