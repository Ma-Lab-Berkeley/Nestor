"""Split activation checkpointing of column layers (NESTOR_ACTIVATION_CHECKPOINT_SPLIT).

``NESTOR_ACTIVATION_CHECKPOINT_SPLIT=1`` checkpoints a slot-updates column
layer as two recompute regions (``column_attn_half`` / ``column_expert_half``)
instead of one, so the backward holds half a layer's activations at a time
(the 1b column arm at 64k). The two
halves run the same ops in the same order as the unsplit layer, so a pooled
column backbone in TRAIN mode must give identical outputs and gradients with
the split on, the split off, and checkpointing off altogether -- including
the head-only column read, whose running aggregate is threaded across the
split boundary.
"""

from __future__ import annotations

import os

import torch

from nestor.backbones import Nestor
from nestor.backbones.nestor import NestorConfig


def _pooled_column_config(head_only: bool, tail: bool = False) -> NestorConfig:
    """Two causal blocks: a 4-column pooled block (public 2 + private 2 per column) then a dense one.

    Args:
        head_only (bool): Use the paper arms' head-only read (column attention
            residual + head-only aggregate reads + deviation sources), whose
            running aggregate ``agg`` is threaded across the split boundary;
            False = the plain Kimi depth read.
        tail (bool): Make the 4-column block a DENSE TAIL (every token selects
            all 4 columns, ``k == G``), the block the query-chunked attention
            half applies to under ``NESTOR_DENSE_TAIL_ATTN=1``.

    Returns:
        NestorConfig: The small column-mode configuration.
    """

    # head_only = the paper arms' read (nestor.experiments.pretrain.NESTOR_READ:
    # column attention residual, per-block aggregates as read sources,
    # head-only sublayer reads, deviation sources), at the default column gain.
    read_kwargs: dict[str, object] = (
        {
            "block_residual_mode": "attn_res_column",
            "block_column_include_aggregate": True,
            "block_column_head_only": True,
            "block_column_deviation_sources": True,
        }
        if head_only
        else {"block_residual_mode": "attn_res_kimi"}
    )
    return NestorConfig(
        input_dim=32, num_blocks=2, feature_dims=[32, 32], causal=True,
        pooling_strides=[1, 1], block_num_layers=[2, 2],
        num_groups=[4, 1], num_groups_per_token=[4 if tail else 2, 1],
        block_num_heads=2, block_head_dims=16, attention_token_dispatch=True,
        block_mlp_hidden_dims=[[24], [24]], block_mlp_activations="swiglu",
        **read_kwargs,  # type: ignore[arg-type]
        router_num_experts_per_group=[1, 1], router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_loss_free_balancing=True, router_bias_update_rate=1e-3,
        router_score_mlp_hidden_dims=[[8], [8]], router_noise_mlp_hidden_dims=[[8], [8]],
        block_expert_pool_public_experts=[2, 0], block_expert_pool_public_top_k=[1, 1],
        block_expert_pool_private_experts=[2, 0], block_expert_pool_private_top_k=[1, 1],
    )


def _run(
    checkpoint: bool, split: bool, head_only: bool, seed: int, block: bool = False, chunks: int = 1,
    lean: bool = False, tail: bool = False, tail_chunks: int = 1,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One seeded forward/backward of the small column backbone under the given flags.

    Args:
        checkpoint (bool): ``NESTOR_ACTIVATION_CHECKPOINT``.
        split (bool): ``NESTOR_ACTIVATION_CHECKPOINT_SPLIT``.
        head_only (bool): Config variant, see :func:`_pooled_column_config`.
        seed (int): Seed for the parameters, the input and the router noise.
        block (bool): ``NESTOR_ACTIVATION_CHECKPOINT_BLOCK`` (one outer region
            per block around the per-layer / per-half regions).
        chunks (int): ``NESTOR_EXPERT_HALF_CHUNKS`` (sequence slices per
            expert half under the split; 1 = off).
        lean (bool): ``NESTOR_ACTIVATION_CHECKPOINT_LEAN`` (the regions take
            the stream means, the stream adds run outside them).
        tail (bool): Config variant with a dense-tail (``k == G``) block, run
            under ``NESTOR_DENSE_TAIL_ATTN=1`` (the fused all-columns path).
        tail_chunks (int): ``NESTOR_DENSE_TAIL_QUERY_CHUNKS`` (query slices
            of the dense tail's attention half under the lean split; 1 = off).

    Returns:
        tuple[torch.Tensor, list[torch.Tensor]]: Output features
            ``[2, 6, 32]`` and every parameter gradient, in parameter order.
    """

    flags = {
        "NESTOR_COLUMN_SLOT_UPDATES": "1",
        "NESTOR_ACTIVATION_CHECKPOINT": "1" if checkpoint else "0",
        "NESTOR_ACTIVATION_CHECKPOINT_SPLIT": "1" if split else "0",
        "NESTOR_ACTIVATION_CHECKPOINT_BLOCK": "1" if block else "0",
        "NESTOR_EXPERT_HALF_CHUNKS": str(chunks),
        "NESTOR_ACTIVATION_CHECKPOINT_LEAN": "1" if lean else "0",
        "NESTOR_DENSE_TAIL_ATTN": "1" if tail else "0",
        "NESTOR_DENSE_TAIL_QUERY_CHUNKS": str(tail_chunks),
    }
    previous = {key: os.environ.get(key) for key in flags}
    os.environ.update(flags)
    try:
        torch.manual_seed(seed)
        model = Nestor(_pooled_column_config(head_only, tail)).train()
        torch.manual_seed(seed + 1)
        x = torch.randn(2, 6, 32)
        pos = torch.arange(6).unsqueeze(0).expand(2, 6)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        torch.manual_seed(seed + 2)
        feats = model(x, pos, attn_mask).pred
        feats.float().pow(2).mean().backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        return feats.detach().clone(), grads
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class CheckpointSplitEquivalenceTest:
    """Split checkpoint == unsplit checkpoint == no checkpoint, outputs and grads."""

    def _assert_same(self, a: tuple[torch.Tensor, list[torch.Tensor]], b: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
        torch.testing.assert_close(b[0], a[0], rtol=1e-6, atol=1e-6)
        assert len(a[1]) == len(b[1]) and len(a[1]) > 0
        for ga, gb in zip(a[1], b[1]):
            torch.testing.assert_close(gb, ga, rtol=1e-5, atol=1e-6)

    def test_split_equals_unsplit_checkpoint(self) -> None:
        """Same seed -> same routing; the split only moves the recompute boundary."""
        self._assert_same(_run(True, False, False, 31), _run(True, True, False, 31))

    def test_split_equals_no_checkpoint(self) -> None:
        """The split path's forward and grads match the plain (no recompute) path."""
        self._assert_same(_run(False, False, False, 32), _run(True, True, False, 32))

    def test_split_head_only_read(self) -> None:
        """The head-only variant threads the running aggregate across the split."""
        self._assert_same(_run(True, False, True, 33), _run(True, True, True, 33))
        self._assert_same(_run(False, False, True, 34), _run(True, True, True, 34))

    def test_block_region_equals_plain(self) -> None:
        """One outer region per block (nesting the per-layer regions) == the plain path."""
        self._assert_same(_run(False, False, False, 41), _run(True, False, False, 41, block=True))
        self._assert_same(_run(False, False, True, 42), _run(True, False, True, 42, block=True))

    def test_block_region_over_split_equals_plain(self) -> None:
        """Block region nesting the per-half regions (the 64k recipe) == the plain path, both reads."""
        self._assert_same(_run(False, False, False, 43), _run(True, True, False, 43, block=True))
        self._assert_same(_run(False, False, True, 44), _run(True, True, True, 44, block=True))
        self._assert_same(_run(True, False, True, 45), _run(True, True, True, 45, block=True))

    def test_split_flag_ignored_without_checkpointing(self) -> None:
        """SPLIT=1 with checkpointing off is the plain path (the flag is a checkpoint mode)."""
        self._assert_same(_run(False, False, False, 35), _run(False, True, False, 35))



class ExpertHalfChunkEquivalenceTest:
    """NESTOR_EXPERT_HALF_CHUNKS: the expert half over sequence slices == the plain path."""

    def _assert_same(self, a: tuple[torch.Tensor, list[torch.Tensor]], b: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
        torch.testing.assert_close(b[0], a[0], rtol=1e-6, atol=1e-6)
        assert len(a[1]) == len(b[1]) and len(a[1]) > 0
        for ga, gb in zip(a[1], b[1]):
            torch.testing.assert_close(gb, ga, rtol=1e-5, atol=1e-6)

    def test_chunked_split_equals_plain(self) -> None:
        """Two and four slices of a 6-token sequence (uneven at 4) == no checkpoint, Kimi read."""
        self._assert_same(_run(False, False, False, 51), _run(True, True, False, 51, chunks=2))
        self._assert_same(_run(False, False, False, 52), _run(True, True, False, 52, chunks=4))

    def test_chunked_split_head_only_read(self) -> None:
        """The head-only read threads the running aggregate through every slice."""
        self._assert_same(_run(False, False, True, 53), _run(True, True, True, 53, chunks=2))
        self._assert_same(_run(True, True, True, 54), _run(True, True, True, 54, chunks=3))

    def test_chunked_under_block_region(self) -> None:
        """The 64k recipe: block region over split halves over slices == plain, both reads."""
        self._assert_same(_run(False, False, False, 55), _run(True, True, False, 55, block=True, chunks=2))
        self._assert_same(_run(False, False, True, 56), _run(True, True, True, 56, block=True, chunks=4))

    def test_chunks_ignored_without_split(self) -> None:
        """The flag is a mode of the split path: without it the layer is one region."""
        self._assert_same(_run(True, False, False, 57), _run(True, False, False, 57, chunks=4))
        self._assert_same(_run(False, False, False, 58), _run(False, True, False, 58, chunks=4))


class LeanSplitEquivalenceTest:
    """NESTOR_ACTIVATION_CHECKPOINT_LEAN: regions fed the means, adds outside == the plain path."""

    def _assert_same(self, a: tuple[torch.Tensor, list[torch.Tensor]], b: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
        torch.testing.assert_close(b[0], a[0], rtol=1e-6, atol=1e-6)
        assert len(a[1]) == len(b[1]) and len(a[1]) > 0
        for ga, gb in zip(a[1], b[1]):
            torch.testing.assert_close(gb, ga, rtol=1e-5, atol=1e-6)

    def test_lean_split_head_only_equals_plain(self) -> None:
        """The head-only read (the paper arms' read): lean split == no checkpoint == plain split."""
        self._assert_same(_run(False, False, True, 61), _run(True, True, True, 61, lean=True))
        self._assert_same(_run(True, True, True, 62), _run(True, True, True, 62, lean=True))

    def test_lean_under_block_region_with_chunks(self) -> None:
        """The 128k recipe: block region over lean halves over slices == plain."""
        self._assert_same(_run(False, False, True, 63), _run(True, True, True, 63, block=True, lean=True))
        self._assert_same(_run(False, False, True, 64), _run(True, True, True, 64, block=True, chunks=4, lean=True))
        self._assert_same(_run(False, False, True, 65), _run(True, True, True, 65, chunks=3, lean=True))

    def test_lean_falls_back_for_the_kimi_read(self) -> None:
        """A full (Kimi) column read reads the streams inside the layer: the plain split runs, still exact."""
        self._assert_same(_run(False, False, False, 66), _run(True, True, False, 66, lean=True))
        self._assert_same(_run(False, False, False, 67), _run(True, True, False, 67, block=True, chunks=2, lean=True))

    def test_lean_ignored_without_split(self) -> None:
        """A mode of the split path: without SPLIT (or checkpointing) the flag changes nothing."""
        self._assert_same(_run(True, False, True, 68), _run(True, False, True, 68, lean=True))
        self._assert_same(_run(False, False, True, 69), _run(False, True, True, 69, lean=True))


class TailQueryChunkEquivalenceTest:
    """NESTOR_DENSE_TAIL_QUERY_CHUNKS: the dense tail's attention half over query slices == plain."""

    def _assert_same(self, a: tuple[torch.Tensor, list[torch.Tensor]], b: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
        torch.testing.assert_close(b[0], a[0], rtol=1e-5, atol=1e-6)
        assert len(a[1]) == len(b[1]) and len(a[1]) > 0
        for ga, gb in zip(a[1], b[1]):
            torch.testing.assert_close(gb, ga, rtol=1e-4, atol=1e-6)

    def test_tail_path_itself_is_exercised(self) -> None:
        """The k == G block takes the fused all-columns path (else these tests test nothing)."""
        import os
        os.environ["NESTOR_DENSE_TAIL_ATTN"] = "1"
        try:
            model = Nestor(_pooled_column_config(True, tail=True))
            assert any(getattr(layer, "_dense_tail_attn", False) for layer in model.blocks[0].layers)
        finally:
            os.environ.pop("NESTOR_DENSE_TAIL_ATTN", None)

    def test_query_slices_equal_plain(self) -> None:
        """2 and 4 slices of a 6-token sequence (uneven at 4) under the lean split == no checkpoint."""
        self._assert_same(_run(False, False, True, 71, tail=True), _run(True, True, True, 71, lean=True, tail=True, tail_chunks=2))
        self._assert_same(_run(False, False, True, 72, tail=True), _run(True, True, True, 72, lean=True, tail=True, tail_chunks=4))

    def test_query_slices_with_expert_chunks_and_block(self) -> None:
        """The 1b 128k recipe shape: lean + expert chunks + query slices, with and without the block region."""
        self._assert_same(_run(False, False, True, 73, tail=True), _run(True, True, True, 73, lean=True, tail=True, tail_chunks=3, chunks=2))
        self._assert_same(_run(False, False, True, 74, tail=True), _run(True, True, True, 74, lean=True, tail=True, tail_chunks=2, chunks=3, block=True))

    def test_query_slices_ignored_without_lean_or_tail(self) -> None:
        """A mode of the lean split on dense-tail layers only: otherwise the flag changes nothing."""
        self._assert_same(_run(True, True, True, 75, tail=True), _run(True, True, True, 75, tail=True, tail_chunks=4))
        self._assert_same(_run(True, True, True, 76, lean=True), _run(True, True, True, 76, lean=True, tail_chunks=4))
