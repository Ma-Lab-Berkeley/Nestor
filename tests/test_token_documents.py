"""Tests for ``nestor.data.token_documents`` against plain reference implementations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nestor.data.token_documents import (
    document_spans,
    encode_documents,
    init_tokenizer_worker,
    load_document,
    open_token_bin,
    read_bin_meta,
    select_documents,
    write_bin_meta,
)


def reference_document_spans(tokens: list[int], eot: int) -> list[tuple[int, int]]:
    """Slow, obviously-correct spans: walk the stream, cut at every eot."""

    spans: list[tuple[int, int]] = []
    start = 0
    for i, tok in enumerate(tokens):
        if tok == eot:
            spans.append((start, i))
            start = i + 1
    return spans


class DocumentSpansTest:
    """``document_spans`` must match the reference for every chunk size."""

    def test_matches_reference_across_chunk_sizes(self) -> None:
        rng = np.random.default_rng(1)
        eot = 9
        stream = rng.integers(0, 12, size=1013).tolist()
        stream[40:43] = [eot, eot, eot]  # empty documents
        stream[-2:] = [1, 2]  # unterminated tail
        expected = reference_document_spans(stream, eot)
        arr = np.asarray(stream, dtype=np.uint32)
        for chunk in (1, 2, 5, 64, 1013, 4096):
            got = document_spans(arr, eot, chunk_tokens=chunk)
            assert got.tolist() == [list(s) for s in expected], f"chunk_tokens={chunk}"

    def test_no_eot_gives_no_spans(self) -> None:
        arr = np.asarray([1, 2, 3], dtype=np.uint32)
        assert document_spans(arr, 9).shape == (0, 2)


class SelectAndLoadTest:
    """Filtering by length and loading with truncation."""

    def test_select_and_load(self) -> None:
        eot = 7
        stream = [1, 2, 3, eot, 4, eot, 5, 5, 5, 5, 5, eot]
        arr = np.asarray(stream, dtype=np.uint32)
        spans = document_spans(arr, eot)
        assert spans.tolist() == [[0, 3], [4, 5], [6, 11]]
        kept = select_documents(spans, min_tokens=3)
        assert kept.tolist() == [[0, 3], [6, 11]]
        assert select_documents(spans, min_tokens=3, max_docs=1).tolist() == [[0, 3]]
        assert load_document(arr, kept[1]).tolist() == [5, 5, 5, 5, 5]
        assert load_document(arr, kept[1], max_tokens=2).tolist() == [5, 5]
        assert load_document(arr, kept[1]).dtype == np.int64


class EncodeDocumentsTest:
    """The worker encoder applies the min / max document-length filters."""

    def test_filters_and_eot_separation(self) -> None:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        texts = ["alpha beta gamma delta epsilon zeta eta", "hi", "", "one two three"]
        init_tokenizer_worker("gpt2", min_doc_tokens=3, max_doc_tokens=5)
        arr, kept = encode_documents(texts)
        expected: list[int] = []
        expected_kept = 0
        for text in texts:
            ids = enc.encode_ordinary(text)[:5]
            if len(ids) >= 3:
                expected.extend(ids + [enc.eot_token])
                expected_kept += 1
        assert kept == expected_kept == 2
        assert arr.dtype == np.uint32
        assert arr.tolist() == expected

    def test_meta_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "toy.bin"
            np.asarray([1, 2, 50256, 3, 50256], dtype=np.uint32).tofile(bin_path)
            write_bin_meta(bin_path, num_tokens=5, num_docs=2, tokenizer_name="gpt2", extra={"k": 1})
            meta = read_bin_meta(bin_path)
            assert meta["num_tokens"] == 5 and meta["num_docs"] == 2 and meta["k"] == 1
            assert meta["eot_token"] == 50256
            tokens, eot = open_token_bin(bin_path)
            assert eot == 50256 and len(tokens) == 5
            assert document_spans(tokens, eot).tolist() == [[0, 2], [3, 4]]

    def test_encode_without_init_raises(self) -> None:
        import nestor.data.token_documents as mod

        saved = mod._ENC
        mod._ENC = None
        try:
            with pytest.raises(AssertionError):
                encode_documents(["x"])
        finally:
            mod._ENC = saved


class IterDocumentTextsTest:
    """``iter_document_texts`` reads the raw corpus formats the preprocessors accept."""

    def test_jsonl_plain_and_gzipped_and_txt(self, tmp_path) -> None:
        import gzip

        from nestor.data.token_documents import iter_document_texts

        rows = ['{"text": "alpha"}', "", '{"text": ""}', "not json", '{"other": "x"}', '{"text": "beta"}']
        (tmp_path / "a.jsonl").write_text("\n".join(rows) + "\n")
        with gzip.open(tmp_path / "b.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write('{"text": "gamma"}\n{"text": "delta"}\n')
        (tmp_path / "c.txt").write_text("epsilon")
        (tmp_path / "d.txt").write_text("")
        assert list(iter_document_texts(tmp_path / "a.jsonl")) == ["alpha", "beta"]
        assert list(iter_document_texts(tmp_path / "b.jsonl.gz")) == ["gamma", "delta"]
        assert list(iter_document_texts(tmp_path / "c.txt")) == ["epsilon"]
        assert list(iter_document_texts(tmp_path / "d.txt")) == []
        (tmp_path / "e.jsonl").write_text('{"body": "zeta"}\n')
        assert list(iter_document_texts(tmp_path / "e.jsonl", text_key="body")) == ["zeta"]

    def test_unsupported_suffix_is_rejected(self, tmp_path) -> None:
        from nestor.data.token_documents import iter_document_texts

        (tmp_path / "x.csv").write_text("a,b\n")
        try:
            list(iter_document_texts(tmp_path / "x.csv"))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
