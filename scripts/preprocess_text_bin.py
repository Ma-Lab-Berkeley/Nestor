#!/usr/bin/env python3
"""Tokenize a text corpus (parquet / jsonl / jsonl.gz / jsonl.zst / txt) into a token ``.bin``.

The corpora of the domain-adaptation experiments -- PG-19 (parquet),
OpenWebMath and AlgebraicStack (Proof-Pile-2, ``jsonl.zst``) and the German
subset of FineWeb-2 (parquet) -- become the same flat, end-of-text separated
cl100k stream that ``scripts/preprocess_dclm.py`` writes and the trainer and
evaluators read, with an optional per-document minimum length and truncation.

Examples::

    # PG-19 test split -> the held-out books
    python scripts/preprocess_text_bin.py --input "raw/pg19/data/test-*.parquet" \\
        --out data/longctx/pg19_test.bin --workers 8

    # OpenWebMath, the first 12 training shards
    python scripts/preprocess_text_bin.py --input raw/proof-pile-2/open-web-math/train/shard-00{00..11}.jsonl.zst \\
        --out data/longctx/owm_train.bin --workers 32

Parquet inputs need ``pyarrow`` (``uv sync --extra data``).
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

from nestor.data.token_documents import (
    encode_documents,
    init_tokenizer_worker,
    iter_document_texts,
    write_bin_meta,
)


def iter_batches(
    files: list[Path], *, batch_size: int, text_key: str, max_docs: int | None
) -> Iterator[list[str]]:
    """Group the documents of all files into encode batches.

    Args:
        files (list[Path]): Input files, in order.
        batch_size (int): Documents per batch.
        text_key (str): Column / field holding the document text.
        max_docs (int | None): Stop after this many documents.

    Yields:
        list[str]: Up to ``batch_size`` document texts.
    """

    batch: list[str] = []
    seen = 0
    for path in files:
        for text in iter_document_texts(path, text_key):
            batch.append(text)
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if max_docs is not None and seen >= max_docs:
                if batch:
                    yield batch
                return
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="Files or globs (parquet / jsonl[.gz|.zst] / txt)")
    parser.add_argument("--out", required=True, type=Path, help="Output .bin path")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--min-doc-tokens", type=int, default=1, help="Drop documents shorter than this")
    parser.add_argument("--max-doc-tokens", type=int, default=None, help="Truncate documents to this many tokens")
    parser.add_argument("--max-docs", type=int, default=None, help="Stop after reading this many documents")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64, help="Docs per encode task")
    args = parser.parse_args()

    files = sorted(
        {Path(p) for pattern in args.input for p in glob.glob(pattern)}
        | {Path(p) for p in args.input if Path(p).is_file()}
    )
    if not files:
        print(f"No input files matched {args.input}", file=sys.stderr)
        sys.exit(1)
    print(
        f"[prep] {len(files)} files -> {args.out} "
        f"(min_doc_tokens={args.min_doc_tokens}, max_doc_tokens={args.max_doc_tokens})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    total_docs = 0
    t0 = time.time()
    last_log = t0
    with open(args.out, "wb") as out_fh, Pool(
        processes=args.workers,
        initializer=init_tokenizer_worker,
        initargs=(args.tokenizer, args.min_doc_tokens, args.max_doc_tokens),
    ) as pool:
        batches = iter_batches(files, batch_size=args.batch_size, text_key=args.text_key, max_docs=args.max_docs)
        # imap keeps submission order, so the on-disk stream is reproducible.
        for arr, kept in pool.imap(encode_documents, batches, chunksize=1):
            arr.tofile(out_fh)
            total_tokens += int(arr.shape[0])
            total_docs += kept
            now = time.time()
            if now - last_log >= 30:
                print(f"[prep] {total_tokens/1e9:.3f}B tokens, {total_docs:,} docs, {now-t0:.0f}s", flush=True)
                last_log = now

    write_bin_meta(
        args.out,
        num_tokens=total_tokens,
        num_docs=total_docs,
        tokenizer_name=args.tokenizer,
        extra={
            "inputs": [str(f) for f in files][:50],
            "num_input_files": len(files),
            "min_doc_tokens": args.min_doc_tokens,
            "max_doc_tokens": args.max_doc_tokens,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )
    print(f"[prep] DONE: {total_tokens:,} tokens, {total_docs:,} docs -> {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
