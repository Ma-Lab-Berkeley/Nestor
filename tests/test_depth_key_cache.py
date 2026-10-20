"""Tests for the shared depth-read key cache (:class:`DepthKeyCache`).

The cache is pure common-subexpression elimination: the Kimi key map is
weight-free, so the unit-RMS key of a depth source is read-independent and the
sources that stay fixed across a block's (or a whole forward's) reads can be
normalized once. These tests pin the two properties that matter -- the cached
forward is bit-identical to the uncached one, and each fixed source really is
normalized exactly once instead of once per read.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.attention_residual import (
    DepthAttentionRead,
    DepthKeyCache,
    unit_rms_key,
)
from nestor.constants import PADDING_TOKEN_POS


class _RsqrtCounter(TorchDispatchMode):
    """Count RMS normalizations, weighted by how many sources each covers.

    Every unit-RMS depth key costs exactly one ``rsqrt`` (see
    :func:`nestor.backbones.attention_residual.unit_rms_key`), as does every
    ``RMSNorm``. The norm count is identical with and without the key cache, so
    the DIFFERENCE between two otherwise identical forwards is exactly the
    number of depth-source normalizations the cache eliminated.

    A GROUPED source entry normalizes ``width`` sources in ONE ``rsqrt`` call
    (its result is ``[batch, seq, width, 1]`` rather than ``[batch, seq, 1]``),
    so a raw call count would silently under-report the work as grouping
    changes. Each call is therefore weighted by the size of its source axis,
    which makes the tally grouping-invariant and keeps it measuring what it
    claims to: normalizations PER SOURCE.

    Attributes:
        count (int): Source-weighted number of ``aten.rsqrt`` calls so far.
    """

    count: int

    def __init__(self) -> None:
        """Initialize the counter at zero."""

        super().__init__()
        self.count = 0

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Tally ``aten.rsqrt`` and run the operator unchanged.

        Args:
            func (Any): The dispatched ``OpOverload``.
            types (Any): Tensor subclass types involved.
            args (tuple[Any, ...]): Positional operator arguments.
            kwargs (dict[str, Any] | None): Keyword operator arguments.

        Returns:
            Any: Whatever the operator returns.
        """

        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.rsqrt.default:
            # Weight by the source axis: a [batch, seq, width, 1] scale is one
            # call but ``width`` source normalizations.
            self.count += (
                result.shape[-2]
                if isinstance(result, torch.Tensor) and result.dim() == 4
                else 1
            )
        return result


def _create_depth_read_config(
    *,
    block_residual_mode: str,
    num_groups: list[int],
    num_groups_per_token: list[int],
    block_num_layers: list[int],
) -> NestorConfig:
    """Create a tiny constant-width Nestor config in a depth-read mode.

    Args:
        block_residual_mode (str): ``"attn_res_kimi"`` or ``"attn_res_column"``.
        num_groups (list[int]): Attention groups per block.
        num_groups_per_token (list[int]): Active groups per token, per block.
        block_num_layers (list[int]): Layers per block.

    Returns:
        NestorConfig: Causal, stride-1, width-16 config with the requested
        depth-read topology.
    """

    num_blocks = len(num_groups)
    return NestorConfig(
        input_dim=16,
        num_blocks=num_blocks,
        feature_dims=[16] * num_blocks,
        pooling_strides=[1] * num_blocks,
        causal=True,
        block_num_layers=block_num_layers,
        num_groups=num_groups,
        num_groups_per_token=num_groups_per_token,
        block_num_heads=[1] * num_blocks,
        block_head_dims=[8] * num_blocks,
        block_mlp_hidden_dims=[[16]] * num_blocks,
        router_num_experts_per_group=[1] * num_blocks,
        router_num_shared_experts=[1] * num_blocks,
        router_num_experts_per_group_per_token=[1] * num_blocks,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode=block_residual_mode,
    )


def _randomize_depth_reads(model: Nestor, seed: int) -> None:
    """Give every depth read a non-degenerate query and value gains.

    A zero query with unit gains collapses every read to the uniform mean of
    unit-RMS sources, which would hide per-read weighting bugs.

    Args:
        model (Nestor): Backbone in a depth-read block residual mode.
        seed (int): ``torch.manual_seed`` value used before the draws.
    """

    torch.manual_seed(seed)
    with torch.no_grad():
        for block in model.blocks:
            for layer in block.layers:
                for read in (layer.kimi_attn_read, layer.kimi_mlp_read):
                    read.query.normal_()
                    read.value_gains.uniform_(0.5, 1.5)
        model.final_depth_read.query.normal_()
        model.final_depth_read.value_gains.uniform_(0.5, 1.5)


class DepthKeyCacheTest:
    """Unit checks on the cache itself and on the read that consumes it."""

    def test_cached_read_matches_uncached_bitwise(self) -> None:
        """A fully cached prefix must not move a single bit of the mixture."""

        torch.manual_seed(0)
        read = DepthAttentionRead(12, max_sources=4)
        with torch.no_grad():
            read.query.normal_()
            read.value_gains.uniform_(0.5, 1.5)
        sources = [torch.randn(2, 5, 12) for _ in range(4)]
        cache = DepthKeyCache(sources, eps=read.eps)

        expected = read(sources)
        actual = read(sources, key_cache=cache)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_partial_prefix_matches_uncached_bitwise(self) -> None:
        """The uncached tail is the case every real block forward hits."""

        torch.manual_seed(1)
        read = DepthAttentionRead(12, max_sources=5)
        with torch.no_grad():
            read.query.normal_()
            read.value_gains.uniform_(0.5, 1.5)
        fixed = [torch.randn(2, 3, 12) for _ in range(3)]
        tail = [torch.randn(2, 3, 12) for _ in range(2)]
        cache = DepthKeyCache(fixed, eps=read.eps)

        expected = read([*fixed, *tail])
        actual = read([*fixed, *tail], key_cache=cache)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_two_reads_share_one_normalization_per_source(self) -> None:
        """The point of the cache: n normalizations, not n per read."""

        torch.manual_seed(2)
        first = DepthAttentionRead(12, max_sources=3)
        second = DepthAttentionRead(12, max_sources=3)
        sources = [torch.randn(2, 4, 12) for _ in range(3)]
        cache = DepthKeyCache(sources, eps=first.eps)

        with torch.no_grad():
            uncached = _RsqrtCounter()
            with uncached:
                first(sources)
                second(sources)
            cached = _RsqrtCounter()
            with cached:
                first(sources, key_cache=cache)
                second(sources, key_cache=cache)

        assert uncached.count == 6  # 3 sources x 2 reads
        assert cached.count == 3  # 3 sources, shared
        # And the two reads literally consumed the same scale tensors.
        scales = cache.scales()
        assert all(a is b for a, b in zip(scales, cache.scales()))

    def test_cached_scales_are_per_token_not_full_width(self) -> None:
        """The cache holds the key's SCALAR factor, not the key itself.

        This is what makes the depth read cheap to retain: a source of shape
        ``[batch, seq, dim]`` contributes a ``[batch, seq, 1]`` scale, so the
        cache is ``dim`` times smaller than the keys it stands in for.
        """

        torch.manual_seed(3)
        read = DepthAttentionRead(12, max_sources=2)
        sources = [torch.randn(2, 4, 12) for _ in range(2)]
        cache = DepthKeyCache(sources, eps=read.eps)

        scales = cache.scales()
        assert all(s.shape == (2, 4, 1) for s in scales)
        assert all(s.dtype == torch.float32 for s in scales)
        # And the scale is exactly the factor the explicit key map applies.
        for source, scale in zip(sources, scales):
            torch.testing.assert_close(
                source * scale,
                unit_rms_key(source, eps=read.eps, dtype=torch.float32),
                rtol=0.0,
                atol=0.0,
            )

    def test_extend_keeps_earlier_scales_and_reduces_only_the_new(
        self,
    ) -> None:
        """Growing the source list must not re-reduce the old sources."""

        torch.manual_seed(3)
        read = DepthAttentionRead(12, max_sources=4)
        sources = [torch.randn(2, 4, 12) for _ in range(2)]
        cache = DepthKeyCache(sources, eps=read.eps)
        first_scales = list(cache.scales())

        appended = [torch.randn(2, 4, 12) for _ in range(2)]
        cache.extend(appended)
        with torch.no_grad():
            counter = _RsqrtCounter()
            with counter:
                grown_scales = cache.scales()

        assert counter.count == 2  # only the two appended sources
        assert cache.num_sources == 4
        assert all(a is b for a, b in zip(first_scales, grown_scales))
        torch.testing.assert_close(
            appended[0] * grown_scales[2],
            unit_rms_key(appended[0], eps=read.eps, dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_mixed_dtype_sources_match_uncached_bitwise(self) -> None:
        """The cache must key on the read's PROMOTED mixture dtype."""

        torch.manual_seed(4)
        read = DepthAttentionRead(12, max_sources=3)
        with torch.no_grad():
            read.query.normal_()
            read.value_gains.uniform_(0.5, 1.5)
        # An fp32 embedding plus bf16 block outputs: exactly the autocast
        # situation, where the read promotes to fp32.
        fixed = [torch.randn(2, 3, 12), torch.randn(2, 3, 12).bfloat16()]
        tail = [torch.randn(2, 3, 12).bfloat16()]
        cache = DepthKeyCache(fixed, eps=read.eps)

        expected = read([*fixed, *tail])
        actual = read([*fixed, *tail], key_cache=cache)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_bfloat16_sources_match_uncached_bitwise(self) -> None:
        """An all-bf16 read caches bf16 keys, still bit-for-bit."""

        torch.manual_seed(5)
        read = DepthAttentionRead(12, max_sources=3)
        with torch.no_grad():
            read.query.normal_()
            read.value_gains.uniform_(0.5, 1.5)
        sources = [torch.randn(2, 3, 12).bfloat16() for _ in range(3)]
        cache = DepthKeyCache(sources, eps=read.eps)

        expected = read(sources)
        actual = read(sources, key_cache=cache)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_gradients_match_the_uncached_read(self) -> None:
        """One shared key must deliver the same gradient to every source.

        The forward is bit-exact, but gradients are not: the cached key
        receives one accumulated gradient from both reads and backprops through
        a single RMSNorm, where the uncached form backprops twice and sums into
        the source, so the additions re-associate. The read computes its RMS
        statistic in fp32 BY DESIGN (``source.float()``), so that reassociation
        is capped at fp32 round-off even in this fp64 model -- measured max
        deviation is ~2e-7 relative to the gradient's scale, i.e. ~2 ULP of
        fp32.
        """

        torch.manual_seed(6)
        read = DepthAttentionRead(12, max_sources=3).double()
        with torch.no_grad():
            read.query.normal_()
            read.value_gains.uniform_(0.5, 1.5)
        raw = [torch.randn(2, 4, 12, dtype=torch.float64) for _ in range(3)]

        grads: list[list[Tensor]] = []
        for use_cache in (False, True):
            sources = [source.clone().requires_grad_(True) for source in raw]
            cache = DepthKeyCache(sources, eps=read.eps) if use_cache else None
            loss = (
                read(sources, key_cache=cache).square().sum()
                + read(sources, key_cache=cache).sin().sum()
            )
            loss.backward()
            grads.append([source.grad for source in sources])

        for uncached_grad, cached_grad in zip(*grads):
            torch.testing.assert_close(
                cached_grad, uncached_grad, rtol=1e-5, atol=1e-5
            )

    def test_rejects_cache_longer_than_the_source_list(self) -> None:
        """A cache covering more sources than were passed is a caller bug."""

        read = DepthAttentionRead(12, max_sources=3)
        sources = [torch.randn(1, 2, 12) for _ in range(3)]
        cache = DepthKeyCache(sources, eps=read.eps)
        with pytest.raises(ValueError, match="covers 3 sources"):
            read(sources[:2], key_cache=cache)

    def test_rejects_eps_mismatch(self) -> None:
        """Different eps means different keys, so refuse to share them."""

        read = DepthAttentionRead(12, max_sources=2)
        sources = [torch.randn(1, 2, 12) for _ in range(2)]
        cache = DepthKeyCache(sources, eps=read.eps * 10)
        with pytest.raises(ValueError, match="eps"):
            read(sources, key_cache=cache)

    def test_rejects_source_identity_drift(self) -> None:
        """The cache is keyed by object identity, and says so loudly."""

        read = DepthAttentionRead(12, max_sources=2)
        sources = [torch.randn(1, 2, 12) for _ in range(2)]
        cache = DepthKeyCache(sources, eps=read.eps)
        drifted = [sources[0], sources[1].clone()]
        with pytest.raises(ValueError, match="leading sources"):
            read(drifted, key_cache=cache)


class NestorDepthKeyCacheEquivalenceTest:
    """The backbone must be unchanged, bit for bit, by the cache."""

    def test_column_forward_matches_uncached_bitwise(self) -> None:
        """Column mode: 2 blocks x 2 layers, routed to 2 of 4 columns."""

        torch.manual_seed(10)
        config = _create_depth_read_config(
            block_residual_mode="attn_res_column",
            num_groups=[4, 4],
            num_groups_per_token=[2, 2],
            block_num_layers=[2, 2],
        )
        model = Nestor(config)
        _randomize_depth_reads(model, seed=11)
        model.eval()

        torch.manual_seed(12)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1).clone()
        pos[1, 4:] = PADDING_TOKEN_POS
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            assert model.depth_key_cache_enabled
            cached = model(x, pos, attn_mask).pred
            model.depth_key_cache_enabled = False
            uncached = model(x, pos, attn_mask).pred
        torch.testing.assert_close(cached, uncached, rtol=0.0, atol=0.0)

    def test_kimi_forward_matches_uncached_bitwise(self) -> None:
        """Kimi mode shares the same fixed-prefix structure (n <= blocks + 1)."""

        torch.manual_seed(13)
        config = _create_depth_read_config(
            block_residual_mode="attn_res_kimi",
            num_groups=[4, 4],
            num_groups_per_token=[2, 2],
            block_num_layers=[2, 2],
        )
        model = Nestor(config)
        _randomize_depth_reads(model, seed=14)
        model.eval()

        torch.manual_seed(15)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            cached = model(x, pos, attn_mask).pred
            model.depth_key_cache_enabled = False
            uncached = model(x, pos, attn_mask).pred
        torch.testing.assert_close(cached, uncached, rtol=0.0, atol=0.0)

    def test_column_backward_matches_uncached(self) -> None:
        """Gradients agree to fp32 round-off (the depth sums re-associate).

        The forward is bit-identical; the backward is not, because a shared key
        accumulates both reads' gradients before backpropagating through one
        RMSNorm. The read's RMS statistic runs in fp32 by design, so the model
        is fp64 only to keep every OTHER term exact -- the residual deviation
        is a couple of fp32 ULPs of the gradient scale.
        """

        torch.manual_seed(16)
        config = _create_depth_read_config(
            block_residual_mode="attn_res_column",
            num_groups=[4, 4],
            num_groups_per_token=[2, 2],
            block_num_layers=[2, 2],
        )
        model = Nestor(config).double()
        _randomize_depth_reads(model, seed=17)
        model.eval()

        torch.manual_seed(18)
        raw = torch.randn(2, 5, config.input_dim, dtype=torch.float64)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        grads: list[dict[str, Tensor]] = []
        for enabled in (True, False):
            model.depth_key_cache_enabled = enabled
            model.zero_grad(set_to_none=True)
            x = raw.clone().requires_grad_(True)
            model(x, pos, attn_mask).pred.square().sum().backward()
            grads.append(
                {
                    "input": x.grad,
                    **{
                        name: parameter.grad
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                    },
                }
            )

        cached_grads, uncached_grads = grads
        assert cached_grads.keys() == uncached_grads.keys()
        for name, cached_grad in cached_grads.items():
            torch.testing.assert_close(
                cached_grad,
                uncached_grads[name],
                rtol=1e-5,
                atol=1e-5,
                msg=lambda text, name=name: f"{name}: {text}",
            )

    def test_ladder_normalizes_each_source_once_per_forward(self) -> None:
        """Op-count regression guard on the production schedule's shape.

        The 4-block ``(3, 5, 7, 1)`` ladder with 4 layers per block runs 33
        depth reads. Uncached they touch ``sum_reads n_read = 369`` sources;
        with the cache the 17 fixed sources (embedding + every block's exported
        columns) are normalized once each and only the 112 within-block partial
        streams stay per-read, for 129 -- a saving of exactly 240.
        """

        torch.manual_seed(19)
        config = _create_depth_read_config(
            block_residual_mode="attn_res_column",
            num_groups=[8, 8, 8, 1],
            num_groups_per_token=[3, 5, 7, 1],
            block_num_layers=[4, 4, 4, 4],
        )
        model = Nestor(config)
        model.eval()

        x = torch.randn(1, 4, config.input_dim)
        pos = torch.arange(4).expand(1, -1)
        attn_mask = torch.ones(1, 4, 4, dtype=torch.bool)

        counts: list[int] = []
        with torch.no_grad():
            for enabled in (True, False):
                model.depth_key_cache_enabled = enabled
                counter = _RsqrtCounter()
                with counter:
                    model(x, pos, attn_mask)
                counts.append(counter.count)

        cached_count, uncached_count = counts
        assert uncached_count - cached_count == 240

    @pytest.mark.parametrize("mode", ["attn_res_column", "attn_res_kimi"])
    def test_repeated_forwards_are_identical(self, mode: str) -> None:
        """The cache must not leak state between forwards."""

        torch.manual_seed(20)
        config = _create_depth_read_config(
            block_residual_mode=mode,
            num_groups=[4, 4],
            num_groups_per_token=[2, 2],
            block_num_layers=[2, 2],
        )
        model = Nestor(config)
        _randomize_depth_reads(model, seed=21)
        model.eval()

        torch.manual_seed(22)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            first = model(x, pos, attn_mask).pred
            second = model(x, pos, attn_mask).pred
            other = model(x + 1.0, pos, attn_mask).pred
            third = model(x, pos, attn_mask).pred
        torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)
        torch.testing.assert_close(third, first, rtol=0.0, atol=0.0)
        assert not torch.allclose(other, first)
