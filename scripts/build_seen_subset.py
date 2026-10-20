#!/usr/bin/env python3
"""Build the repeated-documents corpus of an equal-FLOP extension run.

The ``equal_flops_repeated`` setting of the length ladder trains Nestor for
``T`` steps on the sequences the 477-step equal-token run consumed, with
``(T - 477)`` steps' worth of them drawn again without replacement, as one
shuffled pass over the multiset (``nestor.data.seen_subset``). This writes
that corpus next to the extension corpus it is cut from.

The protocol form derives everything from the registered cell and the length::

    python scripts/build_seen_subset.py --experiment nestor_200m --length 32768 \\
        --data-dir data/longctx

writes ``data/longctx/dclm_longdoc_ge32768_train_repeated_T570.bin`` (with the
default MoE reference; ``--reference dense`` reproduces the paper's 200M
budgets). The explicit form takes every number::

    python scripts/build_seen_subset.py --bin corpus.bin --out subset.bin \\
        --block-size 32768 --cpt-tokens 500000000 --tokens-per-step 1048576 \\
        [--multiset-steps 570]

Without ``--multiset-steps`` the plain seen set is written (the windows the
equal-token run consumed, each once).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nestor.continual.extension import (
    EXTENSION_TOKENS,
    LENGTHS,
    REFERENCE_FAMILY,
    TOKENS_PER_STEP,
    ExtensionRun,
)
from nestor.data.seen_subset import build_seen_subset_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    protocol = parser.add_argument_group("protocol form")
    protocol.add_argument("--experiment", help="registered Nestor cell, e.g. nestor_200m")
    protocol.add_argument("--length", type=int, choices=LENGTHS, help="target context length")
    protocol.add_argument("--reference", default=REFERENCE_FAMILY, choices=("moe", "dense"))
    protocol.add_argument("--data-dir", type=Path, default=Path("data/longctx"))
    explicit = parser.add_argument_group("explicit form")
    explicit.add_argument("--bin", type=Path, help="source corpus .bin")
    explicit.add_argument("--out", type=Path, help="destination .bin")
    explicit.add_argument("--block-size", type=int)
    explicit.add_argument("--cpt-tokens", type=int, default=EXTENSION_TOKENS)
    explicit.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    explicit.add_argument("--multiset-steps", type=int, default=None, help="append second copies so a run of this many steps is one epoch")
    parser.add_argument("--seed", type=int, default=0, help="the trainer's sampler seed")
    parser.add_argument("--multiset-seed", type=int, default=1)
    args = parser.parse_args()

    if args.experiment is not None:
        if args.length is None:
            parser.error("--length is required with --experiment")
        run = ExtensionRun(args.experiment, args.length, "equal_flops_repeated", args.reference)
        source = ExtensionRun(args.experiment, args.length, "equal_tokens").corpus(args.data_dir)
        out = run.corpus(args.data_dir)
        block_size, multiset_steps = run.seq_len, run.steps
        cpt_tokens, tokens_per_step = EXTENSION_TOKENS, TOKENS_PER_STEP
        print(f"[seen] {run.describe()}")
    else:
        if args.bin is None or args.out is None or args.block_size is None:
            parser.error("the explicit form needs --bin, --out and --block-size")
        source, out, block_size = args.bin, args.out, args.block_size
        cpt_tokens, tokens_per_step, multiset_steps = args.cpt_tokens, args.tokens_per_step, args.multiset_steps

    meta = build_seen_subset_bin(
        source,
        out,
        block_size=block_size,
        cpt_tokens=cpt_tokens,
        tokens_per_step=tokens_per_step,
        seed=args.seed,
        multiset_steps=multiset_steps,
        multiset_seed=args.multiset_seed,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
