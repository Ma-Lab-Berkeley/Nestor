"""Equivalence tests for the compact-causal gathered-attention fast path.

``NESTOR_GATHERED_CAUSAL_FLASH=1`` lets
:meth:`nestor.backbones.nestor.GroupRoutedAttention.forward_gathered` drop both
``[B, cap, L]`` / ``[B, cap, cap]`` mask gathers (and lets its caller drop the
dense ``[B, S, S]`` group-mask build) whenever the gathered pair mask is
provably plain causality in compact space. The flag must not change any output,
and a module whose mask is *not* compact-causal must permanently fall back.

The flag is read once per module in ``__init__``, so every test builds two
models -- one with the flag, one without -- and copies the weights across so the
only difference is the attention path.
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.nestor import GroupRoutedAttention
from nestor.constants import PADDING_TOKEN_POS

_FLAG = "NESTOR_GATHERED_CAUSAL_FLASH"


class _CausalFlashEnv:
    """Context manager that sets/clears ``NESTOR_GATHERED_CAUSAL_FLASH``.

    Not a mock: it flips the real environment variable that the production
    modules read in ``__init__``, and restores the previous value on exit.

    Attributes:
        enabled (bool): Whether the flag is set to ``"1"`` inside the block.
        previous (str | None): Value restored on exit.
    """

    enabled: bool
    previous: str | None

    def __init__(self, enabled: bool) -> None:
        """Initialize the context manager.

        Args:
            enabled (bool): Whether to enable the fast path inside the block.
        """

        self.enabled = enabled
        self.previous = None

    def __enter__(self) -> Self:
        """Set the flag and remember the previous value."""

        self.previous = os.environ.get(_FLAG)
        if self.enabled:
            os.environ[_FLAG] = "1"
        else:
            os.environ.pop(_FLAG, None)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore the previous environment value."""

        if self.previous is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = self.previous


def _build_attention_pair(
    *, dim: int = 32, num_heads: int = 2, head_dim: int = 16
) -> tuple[GroupRoutedAttention, GroupRoutedAttention]:
    """Build weight-identical (baseline, fast-path) attention groups.

    Args:
        dim (int): Feature width.
        num_heads (int): Attention heads.
        head_dim (int): Per-head width.

    Returns:
        tuple[GroupRoutedAttention, GroupRoutedAttention]: The baseline module
        (flag off) and the fast-path module (flag on) with copied weights.
    """

    torch.manual_seed(0)
    with _CausalFlashEnv(False):
        baseline = GroupRoutedAttention(dim, num_heads, head_dim).eval()
    with _CausalFlashEnv(True):
        fast = GroupRoutedAttention(dim, num_heads, head_dim).eval()
    fast.load_state_dict(baseline.state_dict())
    return baseline, fast


def _causal_group_inputs(
    *,
    batch_size: int = 3,
    seq_len: int = 9,
    dim: int = 32,
    num_padded: int = 0,
    seed: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a causal token-dispatch input set for one attention group.

    Mirrors what ``RoutedTransformerLayer._attn_update_general`` hands to the
    gathered path: the group mask is ``causal & valid (x) valid & members (x)
    members`` and members never include padding tokens (the router zeroes their
    gates).

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        dim (int): Feature width.
        num_padded (int): Trailing padded positions in the LAST sequence.
        seed (int): Seed for the random draws.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: ``x`` ``[B, S, dim]``, ``pos``
        ``[B, S]``, the group mask ``[B, S, S]``, and ``members`` ``[B, S]``.
    """

    torch.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    if num_padded > 0:
        pos[-1, seq_len - num_padded :] = PADDING_TOKEN_POS
    valid = pos != PADDING_TOKEN_POS  # [B, S]

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    base = causal.unsqueeze(0) & valid[:, :, None] & valid[:, None, :]

    members = (torch.rand(batch_size, seq_len) > 0.35) & valid
    members[:, 0] = True  # keep every sequence non-empty
    group_mask = base & members[:, :, None] & members[:, None, :]
    return x, pos, group_mask, members


def _causal_dispatch_config(*, causal: bool = True) -> NestorConfig:
    """A tiny causal grouped-prefix config with attention token dispatch.

    Block 0 has 4 attention groups (top-2 per token) routed through the
    gathered path; block 1 is dense. Strides are 1 so token positions (and
    therefore padding) survive to the end.

    Args:
        causal (bool): Whether the backbone is causal.

    Returns:
        NestorConfig: The tiny config.
    """

    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[32, 32],
        causal=causal,
        pooling_strides=[1, 1],
        block_num_layers=[2, 1],
        num_groups=[4, 1],
        num_groups_per_token=[2, 1],
        block_num_heads=2,
        block_head_dims=16,
        attention_token_dispatch=True,
        block_mlp_hidden_dims=[[24], [24]],
        block_mlp_activations="gelu",
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_score_mlp_hidden_dims=[[8], [8]],
        router_noise_mlp_hidden_dims=[[8], [8]],
    )


def _build_model_pair(config: NestorConfig) -> tuple[Nestor, Nestor]:
    """Build weight-identical (baseline, fast-path) backbones.

    Args:
        config (NestorConfig): Backbone config used for both models.

    Returns:
        tuple[Nestor, Nestor]: Baseline (flag off) and fast-path (flag on)
        backbones with identical parameters, both in eval mode.
    """

    torch.manual_seed(7)
    with _CausalFlashEnv(False):
        baseline = Nestor(config).eval()
    with _CausalFlashEnv(True):
        fast = Nestor(config).eval()
    fast.load_state_dict(baseline.state_dict())
    return baseline, fast


def _prefix_groups(model: Nestor) -> list[GroupRoutedAttention]:
    """Collect the token-dispatch attention groups of the first block.

    Args:
        model (Nestor): Backbone built from :func:`_causal_dispatch_config`.

    Returns:
        list[GroupRoutedAttention]: Every attention group in block 0.
    """

    return [
        group
        for layer in model.blocks[0].layers
        for group in layer.attention_groups
    ]


class GatheredCausalFlashModuleTest:
    """Module-level equivalence for ``GroupRoutedAttention.forward_gathered``."""

    def test_fully_causal_matches_masked_path(self) -> None:
        """Flag-on output equals the masked gather path with no padding."""

        baseline, fast = _build_attention_pair()
        x, pos, group_mask, members = _causal_group_inputs()

        with torch.no_grad():
            expected = baseline.forward_gathered(x, group_mask, pos, members)
            actual = fast.forward_gathered(x, group_mask, pos, members)

        assert fast._gathered_causal_flash_ok is True
        assert baseline._gathered_causal_flash_ok is None
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_padded_tokens_match_masked_path(self) -> None:
        """Padding (invalid trailing tokens) keeps the fast path exact."""

        baseline, fast = _build_attention_pair()
        x, pos, group_mask, members = _causal_group_inputs(num_padded=3)
        assert not members[-1, -1]

        with torch.no_grad():
            expected = baseline.forward_gathered(x, group_mask, pos, members)
            actual = fast.forward_gathered(x, group_mask, pos, members)

        assert fast.gathered_causal_flash_ok
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_verified_module_accepts_none_mask(self) -> None:
        """After verification the mask is redundant and may be dropped."""

        baseline, fast = _build_attention_pair()
        x, pos, group_mask, members = _causal_group_inputs(seed=4)

        with torch.no_grad():
            expected = baseline.forward_gathered(x, group_mask, pos, members)
            fast.forward_gathered(x, group_mask, pos, members)  # verifies
            actual = fast.forward_gathered(x, None, pos, members)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_unverified_module_rejects_none_mask(self) -> None:
        """Passing no mask before verification is a contract violation."""

        _, fast = _build_attention_pair()
        x, pos, _, members = _causal_group_inputs(seed=5)

        try:
            fast.forward_gathered(x, None, pos, members)
        except ValueError as error:
            assert "requires attn_mask" in str(error)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for an unverified module")

    def test_non_causal_mask_falls_back_permanently(self) -> None:
        """A non-causal mask must revert the module to the masked path."""

        baseline, fast = _build_attention_pair()
        x, pos, _, members = _causal_group_inputs(seed=6)
        full = torch.ones(
            x.shape[0], x.shape[1], x.shape[1], dtype=torch.bool
        )
        group_mask = full & members[:, :, None] & members[:, None, :]

        with torch.no_grad():
            expected = baseline.forward_gathered(x, group_mask, pos, members)
            actual = fast.forward_gathered(x, group_mask, pos, members)
            again = fast.forward_gathered(x, group_mask, pos, members)

        assert fast._gathered_causal_flash_ok is False
        assert not fast.gathered_causal_flash_ok
        # Same kernel, same inputs -> bit-exact, not merely close.
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(again, expected, rtol=0.0, atol=0.0)

    def test_empty_group_leaves_verdict_undecided(self) -> None:
        """The empty-group guard returns before any verification happens."""

        _, fast = _build_attention_pair()
        x, pos, _, _ = _causal_group_inputs(seed=8)
        members = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
        group_mask = torch.zeros(
            x.shape[0], x.shape[1], x.shape[1], dtype=torch.bool
        )

        with torch.no_grad():
            out = fast.forward_gathered(x, group_mask, pos, members)

        assert fast._gathered_causal_flash_ok is None
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(out) == 0


class GatheredCausalFlashBackboneTest:
    """End-to-end equivalence inside the Nestor backbone."""

    def test_causal_backbone_matches_baseline(self) -> None:
        """Flag-on backbone matches flag-off, before AND after verification."""

        config = _causal_dispatch_config()
        baseline, fast = _build_model_pair(config)
        torch.manual_seed(9)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            # First forward verifies; the second exercises the mask-free
            # caller branch (group_mask=None).
            first = fast(x, pos, attn_mask).pred
            second = fast(x, pos, attn_mask).pred

        groups = _prefix_groups(fast)
        assert groups and all(g.gathered_causal_flash_ok for g in groups)
        torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)

    def test_causal_backbone_with_padding_matches_baseline(self) -> None:
        """Padded sequences stay equivalent on the fast path."""

        config = _causal_dispatch_config()
        baseline, fast = _build_model_pair(config)
        torch.manual_seed(10)
        x = torch.randn(3, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(3, 1)
        pos[1, 6:] = PADDING_TOKEN_POS
        pos[2, 4:] = PADDING_TOKEN_POS
        attn_mask = torch.ones(3, 8, 8, dtype=torch.bool)
        valid = (pos != PADDING_TOKEN_POS).unsqueeze(-1)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            fast(x, pos, attn_mask)  # verification forward
            actual = fast(x, pos, attn_mask).pred

        assert all(g.gathered_causal_flash_ok for g in _prefix_groups(fast))
        # Padded rows are re-zeroed at the block exit, so compare everything
        # but assert the real rows are finite (both paths).
        assert torch.isfinite(expected).all()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual * valid, expected * valid, rtol=1e-5, atol=1e-6
        )

    def test_non_causal_backbone_falls_back(self) -> None:
        """A non-causal backbone must keep the masked gather path, bit-exact."""

        config = _causal_dispatch_config(causal=False)
        baseline, fast = _build_model_pair(config)
        torch.manual_seed(11)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            first = fast(x, pos, attn_mask).pred
            second = fast(x, pos, attn_mask).pred

        groups = _prefix_groups(fast)
        assert groups and all(
            g._gathered_causal_flash_ok is False for g in groups
        )
        torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)

    def test_compiled_fast_path_tracks_eager_baseline(self) -> None:
        """The flag is meant for compiled training: trace it cold and warm.

        The first compiled forward traces with an UNDECIDED verdict (mask
        gather + verification), the second with a verified one (mask-free), so
        this covers the recompile that the verdict flip triggers.
        """

        try:
            probe = torch.compile(lambda t: t * 2.0)
            probe(torch.ones(2))
        except Exception as error:  # pragma: no cover - environment-specific
            pytest.skip(f"torch.compile unavailable: {error}")

        config = _causal_dispatch_config()
        baseline, fast = _build_model_pair(config)
        torch.manual_seed(13)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        compiled = torch.compile(fast, fullgraph=False)
        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            cold = compiled(x, pos, attn_mask).pred
            warm = compiled(x, pos, attn_mask).pred

        assert all(g.gathered_causal_flash_ok for g in _prefix_groups(fast))
        # CPU inductor reorders reductions; a loose tolerance is expected.
        torch.testing.assert_close(cold, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(warm, expected, rtol=1e-4, atol=1e-4)

    def test_backward_is_finite_on_fast_path(self) -> None:
        """Gradients flow through the fast path without NaNs."""

        config = _causal_dispatch_config()
        _, fast = _build_model_pair(config)
        fast.train()
        torch.manual_seed(12)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        fast(x, pos, attn_mask).pred.square().mean().backward()
        for group in _prefix_groups(fast):
            assert group.q_proj.weight.grad is not None
            assert torch.isfinite(group.q_proj.weight.grad).all()
