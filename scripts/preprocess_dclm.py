#!/usr/bin/env python3
"""Tokenize DCLM-baseline ``jsonl.zst`` shards into one flat token ``.bin``.

Reads the shards of a local DCLM-baseline copy
(``<dclm_root>/global-shard_XX_of_10/local-shard_Y_of_10/shard_*.jsonl.zst``)
and emits the simplest possible artifact: one contiguous ``numpy`` array of
cl100k token ids (end-of-text separated) plus a small ``.meta.json`` sidecar.
``nestor.data.datasets.DclmTokenDataset`` memory-maps that file, so a 10B+
token corpus never enters RAM.

Examples
--------
Smoke corpus (~300M tokens, validate the pipeline)::

    python scripts/preprocess_dclm.py --dclm-root /data/dclm-baseline-1.0 \
        --out data/dclm/dclm_smoke.bin --target-tokens 300_000_000 --workers 32

Full 200m-scale corpus (~11B tokens = 2.5x Chinchilla, with margin)::

    python scripts/preprocess_dclm.py --dclm-root /data/dclm-baseline-1.0 \
        --out data/dclm/dclm_train.bin --target-tokens 11_000_000_000 --workers 48

Held-out split: run again with ``--global-shards`` set to shards the training
corpus never read (the training corpus reads the sorted shard list from the
first global shard on; the paper's held-out set comes from the last one).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - dependency hint
    print("zstandard is required: `uv pip install zstandard`", file=sys.stderr)
    raise

from nestor.data.token_documents import (  # noqa: E402
    encode_documents as _encode_batch,
    init_tokenizer_worker as _init_worker,
    write_bin_meta,
)

DEFAULT_TOKENIZER = "cl100k_base"


def _iter_shard_files(
    dclm_root: str,
    *,
    global_shards: list[int] | None,
    file_limit: int | None,
) -> list[str]:
    """Return a deterministic, sorted list of jsonl.zst shard paths to read."""
    files: list[str] = []
    globals_present = sorted(
        d for d in os.listdir(dclm_root) if d.startswith("global-shard_")
    )
    for gdir in globals_present:
        if global_shards is not None:
            # e.g. "global-shard_03_of_10" -> 3
            try:
                gnum = int(gdir.split("_")[1])
            except (IndexError, ValueError):
                continue
            if gnum not in global_shards:
                continue
        gpath = os.path.join(dclm_root, gdir)
        if not os.path.isdir(gpath):
            continue
        for ldir in sorted(os.listdir(gpath)):
            lpath = os.path.join(gpath, ldir)
            if not os.path.isdir(lpath):
                continue
            for fn in sorted(os.listdir(lpath)):
                if fn.endswith(".jsonl.zst"):
                    files.append(os.path.join(lpath, fn))
    if file_limit is not None:
        files = files[:file_limit]
    return files


def _iter_text_batches(files: list[str], *, batch_size: int, text_key: str):
    """Yield lists of up to ``batch_size`` document texts from the shards."""
    dctx = zstd.ZstdDecompressor()
    batch: list[str] = []
    for path in files:
        with open(path, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get(text_key)
                if not text:
                    continue
                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dclm-root",
        required=True,
        help="Root dir holding global-shard_XX_of_10/local-shard_Y_of_10/*.jsonl.zst",
    )
    parser.add_argument("--out", required=True, help="Output .bin path")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Stop after writing at least this many tokens (None = read all shards)",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tiktoken encoding name")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512, help="Docs per encode task")
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--min-doc-tokens",
        type=int,
        default=1,
        help="Drop documents shorter than this many tokens",
    )
    parser.add_argument(
        "--max-doc-tokens", type=int, default=None, help="Truncate documents to this many tokens"
    )
    parser.add_argument(
        "--global-shards",
        type=str,
        default=None,
        help="Comma-separated global-shard numbers to read, e.g. '1,2'. Default: all.",
    )
    parser.add_argument(
        "--file-limit", type=int, default=None, help="Cap on number of shard files"
    )
    args = parser.parse_args()

    global_shards = (
        [int(x) for x in args.global_shards.split(",")] if args.global_shards else None
    )
    files = _iter_shard_files(
        args.dclm_root, global_shards=global_shards, file_limit=args.file_limit
    )
    if not files:
        print(f"No .jsonl.zst shards found under {args.dclm_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[prep] {len(files)} shard files; target_tokens={args.target_tokens}")

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    total_tokens = 0
    total_docs = 0
    t0 = time.time()
    last_log = t0

    # Workers encode batches in parallel; the main process appends the resulting
    # uint32 arrays to a single contiguous file in submission order (imap keeps
    # order), so the on-disk token stream is reproducible.
    with open(out_path, "wb") as out_fh:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.tokenizer, args.min_doc_tokens, args.max_doc_tokens),
        ) as pool:
            batches = _iter_text_batches(
                files, batch_size=args.batch_size, text_key=args.text_key
            )
            for arr, kept in pool.imap(_encode_batch, batches, chunksize=4):
                arr.tofile(out_fh)
                total_tokens += int(arr.shape[0])
                total_docs += kept
                now = time.time()
                if now - last_log >= 30:
                    rate = total_tokens / max(now - t0, 1e-6)
                    print(
                        f"[prep] {total_tokens/1e9:.3f}B tokens, "
                        f"{rate/1e6:.2f}M tok/s, {now-t0:.0f}s elapsed",
                        flush=True,
                    )
                    last_log = now
                if args.target_tokens is not None and total_tokens >= args.target_tokens:
                    break
            pool.terminate()

    write_bin_meta(
        out_path,
        num_tokens=total_tokens,
        num_docs=total_docs,
        tokenizer_name=args.tokenizer,
        extra={
            "dclm_root": args.dclm_root,
            "global_shards": args.global_shards or "all",
            "min_doc_tokens": args.min_doc_tokens,
            "max_doc_tokens": args.max_doc_tokens,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )

    print(
        f"[prep] DONE: {total_tokens:,} tokens ({total_tokens/1e9:.3f}B) -> {out_path} "
        f"in {time.time()-t0:.0f}s"
    )
    print(f"[prep] wrote {out_path}.meta.json")


if __name__ == "__main__":
    main()
