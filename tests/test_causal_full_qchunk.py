"""Tests for the query-chunked fp32 causal attention fallback in ``nestor/backbones/nestor.py``.

The fallback exists for head dimensions that no fused fp32 kernel accepts
(the 100m arms' 30), where SDPA's math kernel materialises ``[B, H, S, S]``.
Reference implementation: ``F.scaled_dot_product_attention(is_causal=True)``
itself, and a plain explicit softmax over a ``tril`` mask.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("DCLM_DATA_PATH", "/dev/null")

from nestor.backbones.nestor import (
    causal_attention_query_chunked,
    causal_full_query_chunk,
)


def _reference_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Explicit masked softmax attention; ``q``/``k``/``v`` are ``[B, H, S, D]``."""
    scores = (q @ k.transpose(-1, -2)) * scale  # [B, H, S, S]
    seq_len = q.shape[2]
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


class CausalQueryChunkedAttentionTest:
    """Chunked == fused causal SDPA == explicit reference, for every block layout."""

    @pytest.mark.parametrize(("seq_len", "chunk"), [(300, 64), (256, 64), (5, 64), (301, 1), (128, 128)])
    def test_matches_is_causal_sdpa(self, seq_len: int, chunk: int) -> None:
        torch.manual_seed(0)
        batch_size, num_heads, head_dim = 2, 3, 30  # the 100m head_dim: no fused fp32 kernel
        q, k, v = (torch.randn(batch_size, num_heads, seq_len, head_dim) for _ in range(3))
        scale = 1.0 / head_dim
        fused = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        chunked = causal_attention_query_chunked(q, k, v, scale=scale, chunk=chunk)
        assert chunked.shape == fused.shape
        assert torch.allclose(chunked, fused, atol=1e-5, rtol=1e-5)
        assert torch.allclose(chunked, _reference_causal(q, k, v, scale), atol=1e-5, rtol=1e-5)

    def test_env_knob(self) -> None:
        old = os.environ.get("NESTOR_CAUSAL_FULL_QCHUNK")
        try:
            os.environ.pop("NESTOR_CAUSAL_FULL_QCHUNK", None)
            assert causal_full_query_chunk() == 1024
            os.environ["NESTOR_CAUSAL_FULL_QCHUNK"] = "0"
            assert causal_full_query_chunk() == 0
            os.environ["NESTOR_CAUSAL_FULL_QCHUNK"] = "256"
            assert causal_full_query_chunk() == 256
        finally:
            if old is None:
                os.environ.pop("NESTOR_CAUSAL_FULL_QCHUNK", None)
            else:
                os.environ["NESTOR_CAUSAL_FULL_QCHUNK"] = old
