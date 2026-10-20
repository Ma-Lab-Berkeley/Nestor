"""Reference and integration tests for the aggregated column-wise residual.

Covers ``block_column_include_aggregate=True`` on top of
``block_residual_mode='attn_res_column'``: a block routed to ``k > 1``
attention columns keeps its ``k`` column-wise residual streams AND their
aggregate, so it contributes ``k + 1`` depth sources instead of ``k``. The
aggregate is the merged block-wise direction ``attn_res_kimi`` would have
exported as the block's only source, which the column mode alone cannot
reconstruct: the depth read normalizes every source by its OWN RMS, so no
setting of its query and gains recovers the merged direction from the parts.

Blocks routed to a single column are left alone -- their aggregate is their
one stream -- so the extra source only ever appears where it is not a
duplicate.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.attention_residual import DepthAttentionRead
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.constants import PADDING_TOKEN_POS
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig


def _create_aggregate_config(
    *,
    causal: bool = True,
    include_aggregate: bool = True,
    num_groups: list[int] | None = None,
    num_groups_per_token: list[int] | None = None,
) -> NestorConfig:
    """Create a tiny two-block column-stream config with the aggregate on.

    Block 0 is grouped (4 columns, top-2) so it carries the aggregate, and
    block 1 is dense (1 column) so it does not -- both branches of the
    single-column no-op are exercised by one model.

    Args:
        causal (bool): Whether the backbone runs causally.
        include_aggregate (bool): Value of
            ``block_column_include_aggregate``; False builds the plain column
            twin used as the A/B reference.
        num_groups (list[int] | None): Per-block column count; defaults to
            ``[4, 1]``.
        num_groups_per_token (list[int] | None): Per-block ``k``; defaults to
            ``[2, 1]``.

    Returns:
        NestorConfig: Constant-width config in column-stream mode.
    """

    return NestorConfig(
        input_dim=16,
        num_blocks=2,
        feature_dims=[16, 16],
        pooling_strides=[1, 1],
        causal=causal,
        block_num_layers=[2, 1],
        num_groups=[4, 1] if num_groups is None else num_groups,
        num_groups_per_token=(
            [2, 1] if num_groups_per_token is None else num_groups_per_token
        ),
        block_num_heads=[1, 1],
        block_head_dims=[8, 8],
        block_mlp_hidden_dims=[[16], [16]],
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=[1, 1],
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="attn_res_column",
        block_column_include_aggregate=include_aggregate,
    )


def _randomize_depth_reads(model: Nestor, seed: int) -> None:
    """Give every depth read a seeded non-degenerate query and value gains.

    Zero queries and unit gains make every read a uniform mean of unit-RMS
    sources, which would hide source-ordering bugs (the aggregate landing in
    the wrong slot of ``value_gains``, say); the fixed seed keeps two
    identically built models identical after randomization.

    Args:
        model (Nestor): Backbone in column-stream mode.
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


def _reference_aggregate_backbone_forward(
    model: Nestor,
    x: Tensor,
    pos: Tensor,
    attn_mask: Tensor,
) -> Tensor:
    """Hand-rolled column recurrence that also carries the block aggregate.

    Threads the exported-source list and this block's ``k`` partial streams as
    ORDINARY PYTHON LISTS, with an explicit ``(batch, token, slot)`` loop for
    the column -> slot scatter, calling the model's OWN submodules (depth
    reads, norms, routers, attention groups, expert helpers) for every
    operation. Slow, but obviously faithful to the stated semantics:

    - sublayer input     = one shared depth read over ``[*sources, *streams,
      mean(streams)]`` (streams and aggregate alike omitted on a block's first
      layer, where the streams are still all-zero),
    - attention          = column ``c``'s gated output lands in the slot the
      token holds ``c`` in, and nowhere else,
    - MLP                = one column-agnostic update broadcast to all slots,
    - block export       = the ``k`` streams THEN their aggregate,
    - head input         = final depth read over every exported source.

    A block routed to a single column contributes no aggregate: it would
    duplicate that column exactly.

    Assumes ``model.num_global == 0`` (enforced by the config checks), stride-1
    pooling, no expert pool, and eval mode (no drop-path or router noise).

    Args:
        model (Nestor): Backbone built with
            ``block_residual_mode='attn_res_column'`` and
            ``block_column_include_aggregate=True``.
        x (Tensor): Token embeddings ``[batch_size, seq_len, input_dim]``.
        pos (Tensor): Token positions ``[batch_size, seq_len]``.
        attn_mask (Tensor): Base attention mask
            ``[batch_size, seq_len, seq_len]``.

    Returns:
        Tensor: Backbone output ``[batch_size, seq_len, output_dim]``.
    """

    sources: list[Tensor] = [x]
    for block in model.blocks:
        valid_mask = pos != PADDING_TOKEN_POS
        block_in = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        composed_mask = block._compose_attn_mask(valid_mask, attn_mask)
        router = block.group_router(block_in, valid_mask)
        top_k = block.num_groups_per_token
        topk_indices = router.topk_indices  # [batch, seq, top_k]
        batch_size, seq_len = topk_indices.shape[:2]
        # Single-column blocks have nothing to aggregate.
        carries_aggregate = top_k > 1

        streams: list[Tensor] | None = None  # top_k x [batch, seq, dim]
        for layer in block.layers:
            read_sources = list(sources)
            if streams is not None:
                read_sources.extend(streams)
                if carries_aggregate:
                    read_sources.append(_mean_aggregate(streams))
            h = layer.kimi_attn_read(read_sources)
            norm_h = layer.attn_norm(h)

            # Every column's gated attention output, computed one at a time.
            per_column: list[Tensor] = []  # num_groups x [batch, seq, dim]
            for group_idx, group_attn in enumerate(layer.attention_groups):
                if layer._dense_fastpath:
                    per_column.append(
                        layer._attn_update_dense(
                            norm_h, composed_mask, router, pos
                        )
                    )
                    continue
                members = router.active_mask[:, :, group_idx]
                gate = router.gates[..., group_idx].unsqueeze(-1)
                group_mask = (
                    composed_mask & members[:, :, None] & members[:, None, :]
                )
                per_column.append(group_attn(norm_h, group_mask, pos) * gate)

            # Explicit per-(token, slot) scatter: slot j of token (b, t) takes
            # column topk_indices[b, t, j]'s output and nothing else.
            attn_update = [
                torch.zeros_like(norm_h) for _ in range(top_k)
            ]  # top_k x [batch, seq, dim]
            for b in range(batch_size):
                for t in range(seq_len):
                    for slot in range(top_k):
                        column = int(topk_indices[b, t, slot])
                        attn_update[slot][b, t] = per_column[column][b, t]
            streams = (
                attn_update
                if streams is None
                else [
                    stream + update
                    for stream, update in zip(streams, attn_update)
                ]
            )

            mlp_sources = [*sources, *streams]
            if carries_aggregate:
                mlp_sources.append(_mean_aggregate(streams))
            h = layer.kimi_mlp_read(mlp_sources)
            norm_h = layer.mlp_norm(h)
            if layer._dense_fastpath:
                expert_update, _ = layer._expert_update_dense(
                    norm_h, valid_mask, router
                )
            else:
                expert_update, _ = layer._expert_update_general(
                    norm_h, valid_mask, router
                )
            streams = [stream + expert_update for stream in streams]

        assert streams is not None
        exit_weights = (pos != PADDING_TOKEN_POS).unsqueeze(-1)
        streams = [
            stream * exit_weights.to(dtype=stream.dtype) for stream in streams
        ]
        # Only the next block's group router consumes the collapsed view; the
        # exported aggregate is a source in its own right.
        x = torch.stack(streams, dim=2).mean(dim=2)
        x = x * exit_weights.to(dtype=x.dtype)
        sources.extend(streams)
        if carries_aggregate:
            sources.append(_mean_aggregate(streams))
    return model.final_norm(model.final_depth_read(sources))


def _mean_aggregate(streams: list[Tensor]) -> Tensor:
    """Mean of a block's column streams -- the production aggregate.

    Args:
        streams (list[Tensor]): ``k`` streams of shape
            ``[batch_size, seq_len, dim]``.

    Returns:
        Tensor: ``[batch_size, seq_len, dim]`` mean over the ``k`` streams.
    """

    return torch.stack(streams, dim=0).mean(dim=0)


class DepthAttentionReadScaleTest:
    """Why the aggregate may be a mean: the read only sees directions."""

    def test_rescaling_one_source_leaves_the_mixture_alone(self) -> None:
        """A source enters the read only through its unit-RMS direction.

        This is what lets :func:`_column_read_sources` aggregate with a mean
        instead of the sum ``attn_res_kimi`` accumulates: the key map divides
        every source by its own RMS, so both the depth logit and the mixture
        weight absorb a positive rescaling. Exact only up to the RMSNorm
        epsilon floor, which is why the sources here are at Theta(1) RMS --
        the regime the read is normalized for. On a source small enough for
        ``eps`` to matter the two choices do differ slightly, and the mean is
        then the safer one: it keeps the aggregate on the same scale as the
        streams it summarizes, so the floor treats every source alike.
        """

        torch.manual_seed(70)
        read = DepthAttentionRead(dim=8, max_sources=3)
        with torch.no_grad():
            read.query.normal_()
            read.value_gains.uniform_(0.5, 1.5)
        columns = [torch.randn(2, 4, 8) for _ in range(2)]
        mean = torch.stack(columns, dim=0).mean(dim=0)

        with torch.no_grad():
            from_mean = read([*columns, mean])
            from_sum = read([*columns, mean * 2.0])
        torch.testing.assert_close(from_mean, from_sum, rtol=1e-5, atol=1e-6)


class NestorColumnAggregateConfigTest:
    """Config validation and depth-source bookkeeping."""

    def test_rejects_aggregate_outside_column_mode(self) -> None:
        """The flag is meaningless without column-wise streams to aggregate."""

        config = _create_aggregate_config()
        config.block_residual_mode = "attn_res_kimi"
        config.num_groups_per_token = [1, 1]
        with pytest.raises(
            ValueError, match="block_column_include_aggregate only applies"
        ):
            Nestor(config)

    def test_source_count_adds_one_per_multi_column_block(self) -> None:
        """Only ``k > 1`` blocks contribute the extra aggregated source."""

        plain = Nestor(_create_aggregate_config(include_aggregate=False))
        aggregated = Nestor(_create_aggregate_config())

        # Blocks are k=2 then k=1: 1 + 2 + 1 = 4 sources without the
        # aggregate, and 1 + (2 + 1) + 1 = 5 with it (block 1 abstains).
        assert plain.max_depth_sources == 4
        assert aggregated.max_depth_sources == 5
        assert aggregated.final_depth_read.max_sources == 5

    def test_single_column_block_abstains(self) -> None:
        """A ``k == 1`` block's aggregate would duplicate its one stream."""

        model = Nestor(_create_aggregate_config())

        assert model.blocks[0]._column_aggregate
        assert not model.blocks[1]._column_aggregate
        for block in model.blocks:
            # Blocks and their layers must agree, or the reads and the export
            # would disagree on the source list.
            assert all(
                layer._column_aggregate == block._column_aggregate
                for layer in block.layers
            )

    def test_every_block_multi_column_adds_one_source_each(self) -> None:
        """Two grouped blocks contribute two aggregates."""

        config = _create_aggregate_config(
            num_groups=[4, 4], num_groups_per_token=[2, 3]
        )
        model = Nestor(config)

        assert model.max_depth_sources == 1 + (2 + 1) + (3 + 1)

    def test_config_round_trips(self) -> None:
        """``to_config`` must carry the flag back into a buildable model."""

        torch.manual_seed(80)
        model = Nestor(_create_aggregate_config())
        rebuilt = Nestor(model.to_config())

        assert model.to_config().block_column_include_aggregate
        assert rebuilt.max_depth_sources == model.max_depth_sources
        assert sum(p.numel() for p in rebuilt.parameters()) == sum(
            p.numel() for p in model.parameters()
        )


class NestorColumnAggregateExportTest:
    """What a block hands back at its boundary."""

    def test_multi_column_block_exports_k_plus_one_streams(self) -> None:
        """Block 0 exports its 2 columns then their mean; block 1 exports 1."""

        torch.manual_seed(81)
        config = _create_aggregate_config()
        model = Nestor(config).eval()
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            grouped = model.blocks[0](
                x, pos, attn_mask, column_sources=[x]
            )
        assert grouped.streams is not None
        assert grouped.streams.shape == (2, 5, 3, config.input_dim)
        # Last slot is the aggregate of the preceding k.
        torch.testing.assert_close(
            grouped.streams[:, :, 2],
            grouped.streams[:, :, :2].mean(dim=2),
            rtol=0.0,
            atol=0.0,
        )
        # ...and it is what the next block's group router receives.
        torch.testing.assert_close(
            grouped.x, grouped.streams[:, :, 2], rtol=0.0, atol=0.0
        )

        with torch.no_grad():
            dense = model.blocks[1](
                x, pos, attn_mask, column_sources=[x]
            )
        assert dense.streams is not None
        assert dense.streams.shape == (2, 5, 1, config.input_dim)

    def test_pooled_block_exports_the_aggregate_too(self) -> None:
        """Expert pools change what fills the streams, not the export shape."""

        torch.manual_seed(82)
        config = _create_aggregate_config()
        config.block_expert_pool_public_experts = [2, 2]
        config.block_expert_pool_public_top_k = [1, 1]
        config.block_expert_pool_private_experts = [3, 3]
        config.block_expert_pool_private_top_k = [1, 1]
        model = Nestor(config).eval()
        assert all(
            layer.expert_pool is not None
            for block in model.blocks
            for layer in block.layers
        )
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            output = model.blocks[0](x, pos, attn_mask, column_sources=[x])
        assert output.streams is not None
        assert output.streams.shape == (2, 5, 3, config.input_dim)
        torch.testing.assert_close(
            output.streams[:, :, 2],
            output.streams[:, :, :2].mean(dim=2),
            rtol=0.0,
            atol=0.0,
        )

    def test_padded_tokens_stay_zero_in_the_aggregate(self) -> None:
        """The exported aggregate is masked like the streams it summarizes."""

        torch.manual_seed(83)
        config = _create_aggregate_config()
        model = Nestor(config).eval()
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1).clone()
        pos[1, 3:] = PADDING_TOKEN_POS
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            output = model.blocks[0](x, pos, attn_mask, column_sources=[x])
        assert output.streams is not None
        assert torch.count_nonzero(output.streams[1, 3:]) == 0


class NestorColumnAggregateReferenceTest:
    """Bit-exact checks against the explicit-loop recurrence."""

    def test_forward_matches_handrolled_recurrence_exactly(self) -> None:
        """Production forward must equal the reference loop at atol 0.

        Both sides run the SAME submodules in the same order and reduce with
        the same associativity (each slot receives exactly one attention
        contribution per layer, so the production ``scatter_add`` onto a zero
        buffer is a plain assignment), so float non-associativity cannot bite
        and exact equality is the right bar.
        """

        torch.manual_seed(84)
        config = _create_aggregate_config()
        model = Nestor(config)
        torch.manual_seed(84)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_depth_reads(candidate, seed=85)
            candidate.eval()
        # Block 0 must exercise the general (multi-column) path and block 1
        # the k == 1 degenerate one, so both branches are covered.
        assert not any(
            layer._dense_fastpath for layer in model.blocks[0].layers
        )
        assert all(layer._dense_fastpath for layer in model.blocks[1].layers)

        torch.manual_seed(86)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
            expected = _reference_aggregate_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_padded_forward_matches_reference(self) -> None:
        """Padding must not leak through the aggregate either."""

        torch.manual_seed(87)
        config = _create_aggregate_config()
        model = Nestor(config)
        torch.manual_seed(87)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_depth_reads(candidate, seed=88)
            candidate.eval()

        torch.manual_seed(89)
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1).clone()
        pos[1, 4:] = PADDING_TOKEN_POS
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
            expected = _reference_aggregate_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_aggregate_changes_the_computed_function(self) -> None:
        """Guard against a silently dead flag.

        Two models built from the same seed differ ONLY in the flag, so any
        output difference is the extra source. It must be real: the aggregate
        joins the same shared softmax as everything else, so it redistributes
        depth-mixture mass rather than being appended inertly.
        """

        torch.manual_seed(93)
        aggregated = Nestor(_create_aggregate_config())
        torch.manual_seed(93)
        plain = Nestor(_create_aggregate_config(include_aggregate=False))
        for candidate in (aggregated, plain):
            candidate.eval()

        torch.manual_seed(94)
        x = torch.randn(2, 5, 16)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            with_aggregate = aggregated(x, pos, attn_mask).pred
            without = plain(x, pos, attn_mask).pred
        assert not torch.allclose(with_aggregate, without, rtol=1e-4, atol=1e-4)


class NestorColumnAggregateIntegrationTest:
    """End-to-end wiring and gradients."""

    def test_backward_reaches_the_aggregate_source(self) -> None:
        """The aggregate's own value gain must carry gradient.

        Its index in the head read is fixed by the export order (embedding,
        block 0's two columns, block 0's aggregate, block 1's one stream), so
        a wrongly ordered source list shows up as a dead gain here.
        """

        torch.manual_seed(95)
        config = _create_aggregate_config()
        model = Nestor(config)
        _randomize_depth_reads(model, seed=96)
        model.eval()
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

        output = model(x, pos, attn_mask)
        output.pred.square().mean().backward()

        gains_grad = model.final_depth_read.value_gains.grad
        assert gains_grad is not None
        assert torch.count_nonzero(gains_grad[:5]) == 5
        for block in model.blocks:
            for layer in block.layers:
                for read in (layer.kimi_attn_read, layer.kimi_mlp_read):
                    assert read.value_gains.grad is not None

    def test_forward_shape_and_finiteness(self) -> None:
        """The backbone still emits one token-shaped tensor for the head."""

        torch.manual_seed(97)
        config = _create_aggregate_config()
        model = Nestor(config).eval()
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            output = model(x, pos, attn_mask)
        assert output.pred.shape == (2, 5, config.output_dim)
        assert torch.isfinite(output.pred).all()

    def test_layers_stay_stateless_across_forwards(self) -> None:
        """Functional threading: a second forward must be identical."""

        torch.manual_seed(98)
        config = _create_aggregate_config()
        model = Nestor(config).eval()
        _randomize_depth_reads(model, seed=99)
        x = torch.randn(2, 4, config.input_dim)
        pos = torch.arange(4).expand(2, -1)
        attn_mask = torch.ones(2, 4, 4, dtype=torch.bool)

        with torch.no_grad():
            first = model(x, pos, attn_mask).pred
            second = model(x, pos, attn_mask).pred
        assert torch.equal(first, second)

    def test_reference_mup_transform_composes(self) -> None:
        """The reference muP transform must apply cleanly and train."""

        torch.manual_seed(100)
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(embed_dim=16, vocab_size=64),
            backbone_config=_create_aggregate_config(causal=True),
            head_config=LanguageModelingHeadConfig(vocab_size=64),
            reference_mup=ReferenceMupConfig(num_layers=3),
        )
        model = Model.from_config(config)
        assert isinstance(model, SequenceModel)

        token_ids = torch.randint(0, 64, (2, 5))
        output = model(token_ids)
        assert output.pred.shape == (2, 5, 64)
        assert torch.isfinite(output.pred).all()
        output.pred.square().mean().backward()
        assert model.backbone.final_depth_read.query.grad is not None


class NestorColumnAggregateCompileTest:
    """torch.compile smoke test for the extra per-read source."""

    def test_compile_forward_backward_matches_eager(self) -> None:
        """Compiled fwd+bwd must run without error and track eager closely."""

        # Probe compile availability on a trivial function first so a missing
        # CPU inductor toolchain skips instead of failing, while real tracing
        # bugs in the model still fail loudly below.
        try:
            probe = torch.compile(lambda t: t * 2.0)
            probe(torch.ones(2))
        except Exception as error:  # pragma: no cover - environment-specific
            pytest.skip(f"torch.compile unavailable: {error}")

        torch.manual_seed(101)
        config = _create_aggregate_config()
        model = Nestor(config)
        _randomize_depth_reads(model, seed=102)
        model.eval()

        x = torch.randn(2, 4, config.input_dim)
        pos = torch.arange(4).expand(2, -1)
        attn_mask = torch.ones(2, 4, 4, dtype=torch.bool)

        with torch.no_grad():
            eager_pred = model(x, pos, attn_mask).pred

        compiled = torch.compile(model, fullgraph=False)
        pred = compiled(x, pos, attn_mask).pred
        # CPU inductor reorders reductions; a loose tolerance is expected.
        torch.testing.assert_close(pred, eager_pred, rtol=1e-4, atol=1e-4)

        pred.square().mean().backward()
        for block in model.blocks:
            for layer in block.layers:
                assert layer.kimi_attn_read.query.grad is not None
                assert layer.kimi_mlp_read.query.grad is not None
        assert model.final_depth_read.query.grad is not None
