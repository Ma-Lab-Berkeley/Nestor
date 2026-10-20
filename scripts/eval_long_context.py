#!/usr/bin/env python3
"""Held-out loss at a context length, by position, for a trained checkpoint.

The read of the long-context and domain-adaptation experiments: the model is
rebuilt at ``--block-size`` with rotary base ``--rope-base`` (the run's own),
every held-out sequence is scored once in full precision, and the next-token
cross-entropy is averaged over all prediction targets of all sequences
(excluding the first token of each). The per-position curve, its centred
moving average and 1K-position bins are written to ``--out``.

Two ways to cut sequences from an end-of-text-separated token ``.bin``:

* ``--mode documents`` (default): the first ``--block-size`` tokens of each
  document of at least that many tokens -- the DCLM long-document held-out
  sets and the PG-19 test books;
* ``--mode packed``: consecutive non-overlapping windows of the stream, as
  training reads it -- the page-length corpora (OpenWebMath, AlgebraicStack,
  FineWeb-2), whose documents are far shorter than the context.

Example::

    python scripts/eval_long_context.py --experiment nestor_500m \\
        --checkpoint outputs/continual/nestor_500m_ext32768_equal_tokens/.../model_final.pt \\
        --bin data/longctx/dclm_longdoc_ge32768_val.bin --block-size 32768 \\
        --rope-base 2000000 --max-docs 200 --out results/nestor_500m_32k.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="Registered experiment whose model to rebuild")
    ap.add_argument("--checkpoint", required=True, type=Path, help="Path to model_*.pt")
    ap.add_argument("--bin", required=True, type=Path, help="End-of-text-separated token .bin (held-out)")
    ap.add_argument("--mode", choices=["documents", "packed"], default="documents")
    ap.add_argument("--block-size", type=int, default=32768, help="Sequence length S (the model is rebuilt at it)")
    ap.add_argument("--rope-base", type=int, default=None, help="Rotary base of the run (default: the registry's)")
    ap.add_argument("--min-doc-tokens", type=int, default=None, help="documents mode: minimum length (default S)")
    ap.add_argument("--max-docs", type=int, default=None, help="Cap on sequences scored")
    ap.add_argument("--max-tokens", type=int, default=None, help="Cap on target tokens scored")
    ap.add_argument("--head-chunk", type=int, default=4096, help="Targets per unembedding call (bounds logits memory)")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (the paper's reads are fp32)")
    ap.add_argument("--ma-window", type=int, default=101)
    ap.add_argument("--bin-size", type=int, default=1024)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="JSON output path")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    # The registry reads the context length and rotary base from the
    # environment at import time, so they are set before anything of nestor
    # that imports the registry is loaded.
    os.environ["DCLM_BLOCK_SIZE"] = str(args.block_size)
    if args.rope_base is not None:
        os.environ["DCLM_ROPE_BASE"] = str(args.rope_base)

    import numpy as np
    import torch

    from nestor.data.token_documents import document_spans, load_document, open_token_bin, select_documents
    from nestor.eval.checkpoint import load_experiment_checkpoint
    from nestor.eval.long_context import PerPositionLoss, sequence_nll

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_experiment_checkpoint(
        args.experiment, args.checkpoint, device=device, allow_partial=args.allow_partial
    )
    print(f"[longctx] loaded {args.checkpoint} | step={info['step']} | params={info['num_params']/1e6:.1f}M")

    tokens, eot = open_token_bin(args.bin)
    if args.mode == "documents":
        spans = select_documents(
            document_spans(tokens, eot),
            min_tokens=args.min_doc_tokens or args.block_size,
            max_docs=args.max_docs,
        )
        print(f"[longctx] {len(spans)} documents >= {args.min_doc_tokens or args.block_size} tokens")
        sequences = (load_document(tokens, span, max_tokens=args.block_size) for span in spans)
    else:
        num_windows = len(tokens) // args.block_size
        if args.max_docs is not None:
            num_windows = min(num_windows, args.max_docs)
        print(f"[longctx] {num_windows} packed windows of {args.block_size}")
        sequences = (
            np.array(tokens[i * args.block_size : (i + 1) * args.block_size], dtype=np.int64)
            for i in range(num_windows)
        )

    acc = PerPositionLoss(args.block_size)
    scored = 0
    t0 = time.time()
    with torch.no_grad():
        for seq in sequences:
            ids = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, S)
            # ONE backbone pass per sequence; the unembedding runs on slices of
            # --head-chunk targets, so a 32K sequence never materialises a
            # (32K, vocab) logits tensor.
            nll, _ = sequence_nll(model, ids, head_chunk=args.head_chunk, autocast_bf16=args.bf16)
            acc.add(nll, 1)
            scored += 1
            if scored % 10 == 0:
                print(
                    f"[longctx] {scored} sequences, {acc.tokens:,} tokens, "
                    f"ce={acc.cross_entropy():.4f}, {time.time()-t0:.0f}s",
                    flush=True,
                )
            if args.max_tokens is not None and acc.tokens >= args.max_tokens:
                break

    result = {
        "experiment": args.experiment,
        "checkpoint": str(args.checkpoint),
        "bin": str(args.bin),
        "mode": args.mode,
        "block_size": args.block_size,
        "rope_base": args.rope_base,
        "sequences": scored,
        **acc.to_dict(ma_window=args.ma_window, bin_size=args.bin_size),
    }
    print(
        f"[longctx] DONE: sequences={scored} tokens={acc.tokens:,} "
        f"cross_entropy={acc.cross_entropy():.4f} | {time.time()-t0:.0f}s"
    )
    binned = " ".join(f"{b['start']//1024}k:{b['cross_entropy']:.3f}" for b in result["binned"] if b["tokens"])
    print(f"[longctx] binned CE by position: {binned}")
    print(f"[longctx] RESULT cross_entropy={acc.cross_entropy():.4f}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"[longctx] wrote {args.out}")


if __name__ == "__main__":
    main()
