"""Documents inside an end-of-text-separated flat token stream.

The DCLM preprocessor writes one contiguous ``uint32`` array in which every
document is terminated by the tokenizer's end-of-text id, so document
boundaries are recoverable from the stream alone. This module holds the two
sides of that contract: scanning a ``.bin`` back into ``[start, end)`` document
spans (for document-level evaluation such as per-position loss and test-time
training on held-out books), and the tokenizer helpers that write such streams
with a per-document length filter (TTT-E2E's "discard every document shorter
than the context length").

No ``torch`` import: the preprocessing scripts run on CPU nodes.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

# cl100k_base end-of-text id (tiktoken.get_encoding("cl100k_base").eot_token).
DEFAULT_EOT: int = 100257


def read_bin_meta(bin_path: Path) -> dict[str, Any]:
    """Read the ``<bin>.meta.json`` sidecar written by the preprocessors.

    Args:
        bin_path (Path): Token ``.bin`` path.

    Returns:
        dict[str, Any]: Sidecar contents, or ``{}`` when there is no sidecar.
    """

    meta_path = Path(f"{bin_path}.meta.json")
    return json.loads(meta_path.read_text()) if meta_path.is_file() else {}


def open_token_bin(bin_path: Path) -> tuple[np.memmap, int]:
    """Memory-map a token ``.bin`` and return it with its end-of-text id.

    Args:
        bin_path (Path): Token ``.bin`` path. The dtype and end-of-text id come
            from the sidecar; without one they default to ``uint32`` / cl100k.

    Returns:
        tuple[np.memmap, int]: Read-only token stream of shape ``(num_tokens,)``
        and the end-of-text id that terminates every document.
    """

    meta = read_bin_meta(bin_path)
    dtype = np.uint16 if meta.get("token_dtype") == "uint16" else np.uint32
    eot = int(meta.get("eot_token", DEFAULT_EOT))
    return np.memmap(bin_path, dtype=dtype, mode="r"), eot


def document_spans(
    tokens: np.ndarray,
    eot_token: int,
    chunk_tokens: int = 1 << 28,
) -> np.ndarray:
    """``[start, end)`` spans of the eot-terminated documents in a token stream.

    A span excludes its terminating end-of-text token. Tokens after the last
    end-of-text (an unterminated tail) are not a document. Consecutive
    end-of-text tokens produce empty spans, which are kept so the count matches
    the number of separators; filter them with :func:`select_documents`.

    Args:
        tokens (np.ndarray): Flat token stream (a ``np.memmap`` is fine; it is
            read ``chunk_tokens`` at a time). Shape: (num_tokens,).
        eot_token (int): End-of-text id that terminates every document.
        chunk_tokens (int): Tokens read per scan step (bounds peak RAM).

    Returns:
        np.ndarray: ``int64`` spans in stream order. Shape: (num_docs, 2).
    """

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    doc_start = 0
    for offset in range(0, len(tokens), chunk_tokens):
        chunk = np.asarray(tokens[offset : offset + chunk_tokens])
        eot_idx = np.flatnonzero(chunk == eot_token).astype(np.int64) + offset
        if len(eot_idx) == 0:
            continue
        chunk_starts = np.concatenate(([doc_start], eot_idx[:-1] + 1))
        starts.append(chunk_starts)
        ends.append(eot_idx)
        doc_start = int(eot_idx[-1]) + 1
    if not starts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack((np.concatenate(starts), np.concatenate(ends)), axis=1)


def select_documents(
    spans: np.ndarray,
    *,
    min_tokens: int = 1,
    max_docs: int | None = None,
) -> np.ndarray:
    """Keep the documents of at least ``min_tokens`` tokens, in stream order.

    Args:
        spans (np.ndarray): Document spans from :func:`document_spans`.
            Shape: (num_docs, 2).
        min_tokens (int): Minimum document length in tokens (eot excluded).
        max_docs (int | None): Optional cap on the number of documents kept.

    Returns:
        np.ndarray: The selected spans. Shape: (num_selected, 2).
    """

    lengths = spans[:, 1] - spans[:, 0]
    selected = spans[lengths >= max(min_tokens, 1)]
    if max_docs is not None:
        selected = selected[:max_docs]
    return selected


def load_document(
    tokens: np.ndarray,
    span: np.ndarray,
    *,
    max_tokens: int | None = None,
) -> np.ndarray:
    """Copy one document out of the stream as an ``int64`` array.

    Args:
        tokens (np.ndarray): Flat token stream. Shape: (num_tokens,).
        span (np.ndarray): ``[start, end)`` of the document. Shape: (2,).
        max_tokens (int | None): Truncate the document to its first
            ``max_tokens`` tokens (TTT-E2E / LaCT score fixed-length prefixes).

    Returns:
        np.ndarray: Document tokens. Shape: (min(len, max_tokens),).
    """

    start, end = int(span[0]), int(span[1])
    if max_tokens is not None:
        end = min(end, start + max_tokens)
    return np.array(tokens[start:end], dtype=np.int64, copy=True)


# --- tokenizer side: writing eot-separated streams with a length filter -------

_ENC: Any = None
_MIN_DOC_TOKENS: int = 0
_MAX_DOC_TOKENS: int | None = None


def init_tokenizer_worker(
    tokenizer_name: str,
    min_doc_tokens: int = 0,
    max_doc_tokens: int | None = None,
) -> None:
    """Pool initializer: one ``tiktoken`` encoding per worker plus the filters.

    Args:
        tokenizer_name (str): ``tiktoken`` encoding name (``cl100k_base``).
        min_doc_tokens (int): Documents shorter than this (in tokens, eot
            excluded) are dropped by :func:`encode_documents`.
        max_doc_tokens (int | None): Documents are truncated to this many
            tokens before the eot is appended (``None`` = no truncation).
    """

    global _ENC, _MIN_DOC_TOKENS, _MAX_DOC_TOKENS
    import tiktoken

    _ENC = tiktoken.get_encoding(tokenizer_name)
    _MIN_DOC_TOKENS = int(min_doc_tokens)
    _MAX_DOC_TOKENS = max_doc_tokens


def encode_documents(texts: list[str]) -> tuple[np.ndarray, int]:
    """Encode a batch of documents into one eot-separated ``uint32`` array.

    Must run after :func:`init_tokenizer_worker` in the same process. Empty
    documents and documents shorter than the worker's ``min_doc_tokens`` are
    dropped; longer ones are truncated to ``max_doc_tokens`` when set.

    Args:
        texts (list[str]): Raw document texts.

    Returns:
        tuple[np.ndarray, int]: Token ids with one eot after every kept document
        (shape ``(num_tokens,)``) and the number of documents kept.
    """

    assert _ENC is not None, "call init_tokenizer_worker first"
    eot = _ENC.eot_token
    out: list[int] = []
    kept = 0
    for ids in _ENC.encode_ordinary_batch(texts):
        if _MAX_DOC_TOKENS is not None:
            ids = ids[:_MAX_DOC_TOKENS]
        if len(ids) < max(_MIN_DOC_TOKENS, 1):
            continue
        out.extend(ids)
        out.append(eot)
        kept += 1
    return np.asarray(out, dtype=np.uint32), kept


def write_bin_meta(
    bin_path: Path,
    *,
    num_tokens: int,
    num_docs: int,
    tokenizer_name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the ``<bin>.meta.json`` sidecar (dtype, counts, eot id).

    Args:
        bin_path (Path): Token ``.bin`` path the sidecar describes.
        num_tokens (int): Tokens written (eot tokens included).
        num_docs (int): Documents written.
        tokenizer_name (str): ``tiktoken`` encoding name used.
        extra (dict[str, Any] | None): Additional fields to record.
    """

    import tiktoken

    meta: dict[str, Any] = {
        "token_dtype": "uint32",
        "num_tokens": int(num_tokens),
        "num_docs": int(num_docs),
        "tokenizer": tokenizer_name,
        "eot_token": tiktoken.get_encoding(tokenizer_name).eot_token,
    }
    if extra:
        meta.update(extra)
    Path(f"{bin_path}.meta.json").write_text(json.dumps(meta, indent=2))


# --- reading raw text corpora ------------------------------------------------


def iter_document_texts(path: Path, text_key: str = "text") -> Iterator[str]:
    """Yield the documents of one raw corpus file.

    The released corpora come in four shapes: parquet (PG-19, FineWeb-2),
    ``jsonl.zst`` (DCLM-baseline, Proof-Pile-2), plain or gzipped ``jsonl``,
    and ``txt`` (one document per file). Parquet needs ``pyarrow``
    (``uv sync --extra data``); the other formats need nothing beyond the
    base install.

    Args:
        path (Path): ``.parquet``, ``.jsonl``, ``.jsonl.gz``, ``.json.gz``,
            ``.jsonl.zst`` or ``.txt``.
        text_key (str): Column / field holding the document text.

    Yields:
        str: Non-empty document texts, in file order.

    Raises:
        ValueError: On an unsupported file type.
    """

    name = path.name
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=[text_key]):
            for text in batch.column(0).to_pylist():
                if text:
                    yield text
    elif name.endswith((".jsonl", ".jsonl.gz", ".json.gz", ".jsonl.zst")):
        if name.endswith(".gz"):
            handle: Any = gzip.open(path, "rt", encoding="utf-8")
        elif name.endswith(".zst"):
            import zstandard as zstd

            handle = io.TextIOWrapper(
                zstd.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8"
            )
        else:
            handle = open(path, encoding="utf-8")
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get(text_key)
                except json.JSONDecodeError:
                    continue
                if text:
                    yield text
    elif name.endswith(".txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if text:
            yield text
    else:
        raise ValueError(f"unsupported input type: {path}")
