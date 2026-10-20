"""The window set a run consumed, as a corpus of its own.

Two of the continued-training corpora are built from what another run saw:

* **Repeated documents at equal FLOPs** (``nestor.continual.extension``):
  the equal-FLOP run of Nestor trains for ``T > 477`` steps on the sequences
  the equal-token run consumed, with ``(T - 477)`` steps' worth of them drawn
  again without replacement, as ONE shuffled pass over the multiset
  (:func:`build_seen_subset_bin` with ``multiset_steps``).
* **Replay mixtures** (``nestor.continual.adaptation``): a target-domain
  corpus whose last 25 % is replaced by DCLM sequences the model already saw
  during context extension, each part a prefix of the sampler's permutation
  of its source (:func:`build_replay_mix`).

Both rest on one fact about the trainer: it shuffles a map-style dataset with
``DistributedSampler(dataset, shuffle=True)`` at the default ``seed=0`` (the
experiment's seed seeds the model, not the sampler) and stops at ``max_steps``
inside epoch 0, having consumed exactly ``steps * grad_accum * per_rank_batch``
rows per rank. Rank ``r``'s ``j``-th sample is ``perm[r + j * world]`` of the
epoch-0 permutation ``torch.randperm(num_windows, generator=manual_seed(0))``
(tail padding only ever appends past the consumed prefix), so the UNION across
ranks is the permutation prefix ``perm[: steps * seqs_per_step]`` --
independent of the world size and per-rank shape, which the launchers vary
while holding ``seqs_per_step`` fixed. The runs of every model at a length
therefore consumed the SAME window set.

Window order on disk is a read-locality choice only: the consuming run
reshuffles the bin with its own permutation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from nestor.data.token_documents import open_token_bin, read_bin_meta


def seen_window_indices(
    num_windows: int,
    steps: int,
    seqs_per_step: int,
    seed: int = 0,
    epoch: int = 0,
) -> np.ndarray:
    """Window indices a fresh run consumed by optimizer step ``steps``.

    Args:
        num_windows (int): Windows in the source corpus
            (``num_tokens // block_size``, the map-style dataset length).
        steps (int): Optimizer steps trained (``ceil(tokens / tokens_per_step)``).
        seqs_per_step (int): Sequences per optimizer step across all ranks
            (``world * per_rank_batch * grad_accum``).
        seed (int): ``DistributedSampler`` seed (the trainer leaves the default 0).
        epoch (int): Sampler epoch (0 for a fresh run inside epoch 0).

    Returns:
        np.ndarray: ``int64`` window indices in consumption order (the
            epoch-``epoch`` permutation prefix). Shape: (steps * seqs_per_step,).

    Raises:
        ValueError: When the run would have wrapped past one epoch (its seen
            set is then the whole corpus, not a prefix).
    """

    consumed = steps * seqs_per_step
    if consumed > num_windows:
        raise ValueError(
            f"run consumed {consumed} windows > corpus {num_windows}: the run wrapped, "
            "its seen set is the whole corpus"
        )
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(num_windows, generator=generator)  # (num_windows,)
    return permutation[:consumed].numpy().astype(np.int64)


def multiset_extra_indices(seen: np.ndarray, extra_windows: int, seed: int = 1) -> np.ndarray:
    """Second copies that make a ``T``-step run one epoch of the enlarged bin.

    Draws ``extra_windows`` of the seen windows WITHOUT replacement (each
    window appears at most twice in the multiset bin), with its own fixed
    seed so the choice is reproducible and independent of the sampler seed.

    Args:
        seen (np.ndarray): ``int64`` seen window indices (any order; only the
            set matters). Shape: (num_seen,).
        extra_windows (int): Second copies to draw; at most ``num_seen``.
        seed (int): Generator seed for the draw.

    Returns:
        np.ndarray: ``int64`` window indices, ascending. Shape: (extra_windows,).

    Raises:
        ValueError: When a third copy of some window would be needed.
    """

    if extra_windows > seen.size:
        raise ValueError(
            f"{extra_windows} extra windows > {seen.size} seen: the multiset would need a "
            "third copy of some window (multiset_steps > 2 x steps)"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(int(seen.size), generator=generator).numpy()  # (num_seen,)
    return np.sort(seen[order[:extra_windows]]).astype(np.int64)


def _write_windows(handle: BinaryIO, tokens: np.ndarray, indices: np.ndarray, block_size: int) -> None:
    """Append the windows ``indices`` of a token stream to an open file.

    Args:
        handle (BinaryIO): Destination opened for binary writing.
        tokens (np.ndarray): Flat token stream (a memmap). Shape: (num_tokens,).
        indices (np.ndarray): Window indices to copy. Shape: (num_out,).
        block_size (int): Tokens per window.
    """

    for index in indices:
        start = int(index) * block_size
        np.asarray(tokens[start : start + block_size]).tofile(handle)  # (block_size,)


def _steps_for(tokens: int, tokens_per_step: int) -> int:
    """``ceil(tokens / tokens_per_step)``: the trainer's step count for a token budget."""

    return -(-tokens // tokens_per_step)


def build_seen_subset_bin(
    source_bin: Path,
    output_bin: Path,
    *,
    block_size: int,
    cpt_tokens: int,
    tokens_per_step: int,
    seed: int = 0,
    multiset_steps: int | None = None,
    multiset_seed: int = 1,
) -> dict[str, Any]:
    """Write the windows a run over ``source_bin`` consumed as a new flat token bin.

    Args:
        source_bin (Path): Full corpus ``.bin`` the run trained on.
        output_bin (Path): Destination ``.bin``; a ``.meta.json`` sidecar with
            provenance is written next to it.
        block_size (int): Tokens per training window (the context length).
        cpt_tokens (int): The run's token budget, from which
            ``steps = ceil(cpt_tokens / tokens_per_step)``.
        tokens_per_step (int): Tokens per optimizer step.
        seed (int): Sampler seed (0 = the trainer's default).
        multiset_steps (int | None): The equal-FLOP step count ``T``:
            ``(T - steps) * seqs_per_step`` seen windows are appended as second
            copies so the bin holds exactly ``T * seqs_per_step`` windows and a
            ``T``-step run is one epoch of it. ``None`` = the plain seen set.
        multiset_seed (int): Seed of the second-copy draw.

    Returns:
        dict[str, Any]: The sidecar metadata written.

    Raises:
        ValueError: When ``tokens_per_step`` is not a multiple of ``block_size``
            or ``multiset_steps`` is below the run's own step count.
    """

    if tokens_per_step % block_size != 0:
        raise ValueError(f"tokens_per_step {tokens_per_step} not a multiple of block_size {block_size}")
    seqs_per_step = tokens_per_step // block_size
    steps = _steps_for(cpt_tokens, tokens_per_step)

    tokens, eot = open_token_bin(source_bin)  # (num_tokens,)
    num_windows = int(tokens.shape[0]) // block_size
    indices_sorted = np.sort(seen_window_indices(num_windows, steps, seqs_per_step, seed=seed))
    extra = np.zeros(0, dtype=np.int64)  # (extra_windows,)
    if multiset_steps is not None:
        if multiset_steps < steps:
            raise ValueError(f"multiset_steps {multiset_steps} < steps {steps}")
        extra = multiset_extra_indices(
            indices_sorted, (multiset_steps - steps) * seqs_per_step, seed=multiset_seed
        )
    to_write = np.concatenate([indices_sorted, extra])  # (num_windows_out,)

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(output_bin, "wb") as handle:
        _write_windows(handle, tokens, to_write, block_size)

    source_meta = read_bin_meta(source_bin)
    meta: dict[str, Any] = {
        "token_dtype": str(tokens.dtype),
        "tokenizer": source_meta.get("tokenizer", "cl100k_base"),
        "eot_token": eot,
        "num_tokens": int(to_write.size) * block_size,
        "block_size": block_size,
        "num_windows": int(to_write.size),
        "unique_windows": int(indices_sorted.size),
        "source_bin": str(source_bin),
        "source_num_windows": num_windows,
        "sampler_seed": seed,
        "steps": steps,
        "seqs_per_step": seqs_per_step,
        "multiset_steps": multiset_steps,
        "extra_windows": int(extra.size),
        "multiset_seed": multiset_seed if multiset_steps is not None else None,
        "torch_version": torch.__version__,
        "note": (
            "epoch-0 DistributedSampler(seed=0) permutation prefix: the window set a "
            f"{steps}-step run over the source consumed"
            + (
                f"; multiset: + {int(extra.size)} second copies drawn without replacement "
                f"so a {multiset_steps}-step run is one epoch"
                if multiset_steps is not None
                else ""
            )
        ),
    }
    Path(f"{output_bin}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def build_replay_mix(
    target_bin: Path,
    replay_bin: Path,
    output_bin: Path,
    *,
    block_size: int,
    target_share_pct: int,
    mix_tokens: int,
    tokens_per_step: int,
    seed: int = 0,
) -> dict[str, Any]:
    """A target-domain / replay mixture of nested seen-window prefixes.

    The target part is the first ``target_share_pct`` % of the window set the
    pure-target run consumes (so every share sees a SUBSET of the documents
    the no-replay run saw); the replay part is the first
    ``100 - target_share_pct`` % of the window set the context-extension runs
    consumed over ``replay_bin`` (true replay of seen data). Each part is
    sized at ``mix_tokens x share`` in whole steps, so the realised share is
    within half a percent of the nominal one.

    Args:
        target_bin (Path): The target corpus ``.bin``.
        replay_bin (Path): The replay source ``.bin`` (the extension corpus).
        output_bin (Path): Destination ``.bin`` (+ ``.meta.json`` sidecar).
        block_size (int): Tokens per training window.
        target_share_pct (int): Target-domain share in percent, ``0 < p < 100``.
        mix_tokens (int): Nominal total size of the mixture in tokens.
        tokens_per_step (int): Tokens per optimizer step of the runs whose
            seen sets are taken.
        seed (int): Sampler seed of those runs (0 = the trainer's default).

    Returns:
        dict[str, Any]: The sidecar metadata written.

    Raises:
        ValueError: On a share outside ``(0, 100)`` or a dtype / tokenizer /
            end-of-text mismatch between the two sources.
    """

    if not 0 < target_share_pct < 100:
        raise ValueError(f"target_share_pct must be strictly between 0 and 100, got {target_share_pct}")
    if tokens_per_step % block_size != 0:
        raise ValueError(f"tokens_per_step {tokens_per_step} not a multiple of block_size {block_size}")
    seqs_per_step = tokens_per_step // block_size
    target_tokens, target_eot = open_token_bin(target_bin)
    replay_tokens, replay_eot = open_token_bin(replay_bin)
    target_meta, replay_meta = read_bin_meta(target_bin), read_bin_meta(replay_bin)
    if target_tokens.dtype != replay_tokens.dtype or target_eot != replay_eot:
        raise ValueError("target and replay bins disagree on token dtype or end-of-text id")
    if target_meta.get("tokenizer", "cl100k_base") != replay_meta.get("tokenizer", "cl100k_base"):
        raise ValueError("target and replay bins were written by different tokenizers")

    parts: list[dict[str, Any]] = []
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(output_bin, "wb") as handle:
        for name, tokens, path, pct in (
            ("target", target_tokens, target_bin, target_share_pct),
            ("replay", replay_tokens, replay_bin, 100 - target_share_pct),
        ):
            steps = _steps_for(mix_tokens * pct // 100, tokens_per_step)
            num_windows = int(tokens.shape[0]) // block_size
            indices = np.sort(seen_window_indices(num_windows, steps, seqs_per_step, seed=seed))
            _write_windows(handle, tokens, indices, block_size)
            parts.append(
                {
                    "part": name,
                    "source_bin": str(path),
                    "share_pct": pct,
                    "steps": steps,
                    "windows": int(indices.size),
                    "tokens": int(indices.size) * block_size,
                }
            )

    num_windows_out = sum(int(p["windows"]) for p in parts)
    meta: dict[str, Any] = {
        "token_dtype": str(target_tokens.dtype),
        "tokenizer": target_meta.get("tokenizer", "cl100k_base"),
        "eot_token": target_eot,
        "num_tokens": num_windows_out * block_size,
        "block_size": block_size,
        "num_windows": num_windows_out,
        "target_share_pct": target_share_pct,
        "mix_tokens": mix_tokens,
        "seqs_per_step": seqs_per_step,
        "sampler_seed": seed,
        "sources": parts,
        "torch_version": torch.__version__,
        "note": (
            "replay mixture: each part is the epoch-0 DistributedSampler(seed=0) "
            "permutation prefix of its source; the consuming run reshuffles the whole bin"
        ),
    }
    Path(f"{output_bin}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
