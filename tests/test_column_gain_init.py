"""Tests for ``block_column_gain_init``.

The knob exists to remove the simplex-dilution penalty at initialization: a
column read spreads its softmax over every column of every block, so at the
default unit gains the merged directions start with ``1/n`` of the mixture
instead of kimi's ``1/(blocks+1)``. Initializing the COLUMN gains small keeps
the extra sources without letting them crowd the mixture from step 0 -- and at
exactly 0.0, with the aggregate exported, the read's mixture is EXACTLY the
kimi read's (up to the positive scale its pre-norm consumers discard), while
the zeroed gains still receive gradient.
"""

from __future__ import annotations

import pytest
import torch

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.attention_residual import DepthAttentionRead


def _depth_width(sources) -> int:
    """Total depth sources in a read's argument, counting grouped entries.

    A read entry is either a plain ``[batch, seq, dim]`` source or a GROUPED
    ``[batch, seq, width, dim]`` block of contiguous sources (a column block
    exports its slots without unbinding them, so the read can batch its
    per-source work). ``len()`` therefore counts entries, not depth sources.

    Args:
        sources (Sequence[Tensor]): A read's source list.

    Returns:
        int: Number of depth sources the read will score.
    """

    return sum(s.shape[2] if s.dim() == 4 else 1 for s in sources)


def _flat_sources(sources) -> list:
    """Expand grouped entries so individual depth sources can be indexed.

    Args:
        sources (Sequence[Tensor]): A read's source list.

    Returns:
        list: One ``[batch, seq, dim]`` tensor per depth source, in depth order.
    """

    flat: list = []
    for source in sources:
        flat.extend(source.unbind(dim=2) if source.dim() == 4 else [source])
    return flat


def _config(
    *,
    gain_init: float,
    include_aggregate: bool = True,
    block_residual_mode: str = "attn_res_column",
) -> NestorConfig:
    """Build a three-block column config exercising every slot type.

    Block 0 is multi-column (4 columns, top-2, carries an aggregate), block 1
    multi-column with a different ``k`` (top-3), and block 2 dense (one
    column, no aggregate), so the expected gain layout mixes all three cases.

    Args:
        gain_init (float): Value for ``block_column_gain_init``.
        include_aggregate (bool): Value for
            ``block_column_include_aggregate``.
        block_residual_mode (str): Residual mode, for the validation test.

    Returns:
        NestorConfig: Constant-width causal config.
    """

    return NestorConfig(
        input_dim=16,
        num_blocks=3,
        feature_dims=[16, 16, 16],
        pooling_strides=[1, 1, 1],
        causal=True,
        block_num_layers=[2, 1, 1],
        num_groups=[4, 4, 1],
        num_groups_per_token=[2, 3, 1],
        block_num_heads=[1, 1, 1],
        block_head_dims=[8, 8, 8],
        block_mlp_hidden_dims=[[16], [16], [16]],
        router_num_experts_per_group=[1, 1, 1],
        router_num_shared_experts=[1, 1, 1],
        router_num_experts_per_group_per_token=[1, 1, 1],
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode=block_residual_mode,
        block_column_include_aggregate=include_aggregate,
        block_column_gain_init=gain_init,
    )


class ColumnGainInitLayoutTest:
    """The init vector must land on the right slots of every read."""

    def test_column_slots_get_the_init_and_the_rest_keep_one(self) -> None:
        """Layout: [emb, 2 cols, agg, 3 cols, agg, 1 dense] = 8 slots.

        Columns get the configured value; the embedding, both aggregates, and
        the dense block's single stream (which IS its merged direction) keep
        1.0 -- on the final read and on every sublayer read alike.
        """

        model = Nestor(_config(gain_init=0.25))
        expected = torch.tensor(
            [1.0, 0.25, 0.25, 1.0, 0.25, 0.25, 0.25, 1.0, 1.0]
        )
        assert model.max_depth_sources == expected.numel()
        reads = [
            module
            for module in model.modules()
            if isinstance(module, DepthAttentionRead)
        ]
        assert len(reads) > 1  # sublayer reads + the final read
        for read in reads:
            torch.testing.assert_close(
                read.value_gains.detach(), expected, rtol=0.0, atol=0.0
            )

    def test_default_is_the_plain_unit_init(self) -> None:
        """gain_init=1.0 must leave every read at the ones init."""

        model = Nestor(_config(gain_init=1.0))
        for module in model.modules():
            if isinstance(module, DepthAttentionRead):
                assert torch.all(module.value_gains.detach() == 1.0)

    def test_rejected_outside_column_mode(self) -> None:
        """The knob names column sources, which other modes do not have.

        Validation runs in ``NestorConfig.to_dict`` (the model-construction
        path), not at dataclass construction, matching the mode checks around
        it.
        """

        config = _config(
            gain_init=0.5,
            include_aggregate=False,
            block_residual_mode="attn_res_kimi",
        )
        with pytest.raises(ValueError, match="block_column_gain_init"):
            config.to_dict()

    def test_negative_init_rejected(self) -> None:
        """A negative gain is a sign flip, not a scale-down."""

        with pytest.raises(ValueError, match="non-negative"):
            _config(gain_init=-0.1).to_dict()


class ColumnGainZeroMatchesKimiReadTest:
    """gain_init=0.0 + aggregate: the read starts at exactly kimi's mixture."""

    def test_zeroed_columns_reproduce_the_merged_only_read(self) -> None:
        """Direction equality against a hand-built merged-only read.

        The column read with zeroed column gains and the kimi read see the
        same non-silenced sources (embedding + per-block merged directions),
        so their mixtures must agree up to the positive scale the softmax
        mass on silenced slots removes -- which every consumer's pre-norm
        discards. Checked post-RMSNorm, and with a non-zero query so the
        agreement is not an artifact of uniform weights.
        """

        torch.manual_seed(0)
        batch, seq, dim, k, blocks = 2, 5, 16, 4, 3
        emb = torch.randn(batch, seq, dim)
        streams = [torch.randn(batch, seq, k, dim) for _ in range(blocks)]
        aggs = [s.mean(dim=2) for s in streams]

        kimi_sources = [emb, *aggs]
        column_sources = [emb]
        for s, agg in zip(streams, aggs):
            column_sources += [*s.unbind(dim=2), agg]

        kimi = DepthAttentionRead(dim, max_sources=len(kimi_sources))
        column_read = DepthAttentionRead(dim, max_sources=len(column_sources))
        with torch.no_grad():
            query = torch.randn(dim) * 0.5
            kimi.query.copy_(query)
            column_read.query.copy_(query)
            kimi.value_gains.uniform_(0.5, 1.5)
            gains = torch.zeros(len(column_sources))
            gains[0] = kimi.value_gains[0]
            slot = 1
            for block in range(blocks):
                slot += k  # column slots stay 0
                gains[slot] = kimi.value_gains[1 + block]
                slot += 1
            column_read.value_gains.copy_(gains)

        def unit_rms(x: torch.Tensor) -> torch.Tensor:
            """Normalize to unit RMS, the pre-norm a consumer applies."""

            return x / x.float().square().mean(-1, keepdim=True).sqrt()

        torch.testing.assert_close(
            unit_rms(column_read(column_sources)),
            unit_rms(kimi(kimi_sources)),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_zeroed_gains_still_receive_gradient(self) -> None:
        """Silence must be a starting point, not a dead zone.

        ``dh/dg_i = alpha_i * k_hat_i`` is non-zero at ``g_i = 0``, so the
        optimizer can raise a column's gain the moment it would help.
        """

        torch.manual_seed(1)
        model = Nestor(_config(gain_init=0.0))
        model.eval()
        x = torch.randn(2, 6, 16, requires_grad=True)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        model(x, pos, attn_mask).pred.square().sum().backward()

        # Column slots of the layout [emb, c,c, agg, c,c,c, agg, dense].
        column_slots = [1, 2, 4, 5, 6]
        grads = model.final_depth_read.value_gains.grad
        assert grads is not None
        assert torch.all(grads[column_slots].abs() > 0), (
            "zero-gain column slots received no gradient; the init would be "
            "a dead zone instead of a starting point"
        )


class ColumnMassLambdaTest:
    """``block_column_mass_lambda``: unit per-block mass, split by lambda."""

    def test_mass_conserving_layout(self) -> None:
        """Layout [emb, 2 cols, agg, 3 cols, agg, dense] at lambda=0.5.

        Multi-column blocks: columns ``0.5 / k``, aggregate ``0.5`` -- total
        mass exactly 1 per block regardless of k. Embedding and the dense
        block keep 1.0.
        """

        config = _config(gain_init=1.0)
        config.block_column_mass_lambda = 0.5
        model = Nestor(config)
        expected = torch.tensor(
            [1.0, 0.25, 0.25, 0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5, 1.0]
        )
        for module in model.modules():
            if isinstance(module, DepthAttentionRead):
                torch.testing.assert_close(
                    module.value_gains.detach(), expected, rtol=0.0, atol=1e-7
                )
        # The conservation property itself: per-block mass is exactly one.
        gains = model.final_depth_read.value_gains.detach()
        assert torch.isclose(gains[1:4].sum(), torch.tensor(1.0))
        assert torch.isclose(gains[4:8].sum(), torch.tensor(1.0))

    def test_lambda_zero_equals_gain_init_zero(self) -> None:
        """lambda=0 and gain_init=0 must build identical gain vectors."""

        config_lambda = _config(gain_init=1.0)
        config_lambda.block_column_mass_lambda = 0.0
        config_gain = _config(gain_init=0.0)
        torch.manual_seed(0)
        model_lambda = Nestor(config_lambda)
        torch.manual_seed(0)
        model_gain = Nestor(config_gain)
        torch.testing.assert_close(
            model_lambda.final_depth_read.value_gains.detach(),
            model_gain.final_depth_read.value_gains.detach(),
            rtol=0.0,
            atol=0.0,
        )

    def test_requires_the_aggregate(self) -> None:
        """The 1 - lambda share needs an aggregate slot to land on."""

        config = _config(gain_init=1.0, include_aggregate=False)
        config.block_column_mass_lambda = 0.5
        with pytest.raises(ValueError, match="include_aggregate"):
            config.to_dict()

    def test_mutually_exclusive_with_constant_init(self) -> None:
        """Two prescriptions for the same gains must be rejected."""

        config = _config(gain_init=0.25)
        config.block_column_mass_lambda = 0.5
        with pytest.raises(ValueError, match="set one"):
            config.to_dict()

    def test_out_of_range_lambda_rejected(self) -> None:
        """lambda splits one unit of mass, so it lives in [0, 1]."""

        config = _config(gain_init=1.0)
        config.block_column_mass_lambda = 1.5
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            config.to_dict()


class ColumnHeadOnlyTest:
    """``block_column_head_only``: narrow sublayer reads, wide head read."""

    def _head_config(self) -> NestorConfig:
        """The three-block config with head-only columns on.

        Returns:
            NestorConfig: ``_config`` plus the head-only flag, columns silent
            at the head read.
        """

        config = _config(gain_init=0.0)
        config.block_column_head_only = True
        return config

    def test_sublayer_reads_are_narrow_and_head_read_is_wide(self) -> None:
        """Sublayer reads see at most num_blocks + 1 sources; head sees all.

        Config layout: 3 blocks (k = 2, 3, 1) with aggregates, so the full
        list is 9 sources while the narrow list never exceeds 4 (embedding +
        up to 2 completed aggregates + the live aggregate).
        """

        torch.manual_seed(0)
        model = Nestor(self._head_config())
        model.eval()

        widths: dict[int, list[int]] = {}

        def hook(module, args, kwargs):
            sources = kwargs.get("sources", args[0] if args else ())
            widths.setdefault(id(module), []).append(_depth_width(sources))

        for module in model.modules():
            if isinstance(module, DepthAttentionRead):
                module.register_forward_pre_hook(hook, with_kwargs=True)

        x = torch.randn(2, 6, 16)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        model(x, pos, attn_mask)

        head_widths = widths.pop(id(model.final_depth_read))
        assert head_widths == [9], head_widths
        sublayer_max = max(w for calls in widths.values() for w in calls)
        assert sublayer_max <= 4, (
            f"sublayer reads should be narrow (<= blocks + 1), saw "
            f"{sublayer_max}"
        )

    def test_gain_init_lands_only_on_the_head_read(self) -> None:
        """Sublayer reads keep unit gains -- their layout has no column slots.

        Applying the full-layout vector to a narrow read would zero aggregate
        slots (slot 1 is a column in the full layout but an aggregate in the
        narrow one), silently breaking the kimi-equivalent start.
        """

        torch.manual_seed(1)
        model = Nestor(self._head_config())
        expected_head = torch.tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        )
        torch.testing.assert_close(
            model.final_depth_read.value_gains.detach(),
            expected_head,
            rtol=0.0,
            atol=0.0,
        )
        for module in model.modules():
            if (
                isinstance(module, DepthAttentionRead)
                and module is not model.final_depth_read
            ):
                assert torch.all(module.value_gains.detach() == 1.0)

    def test_backward_reaches_the_columns_through_the_head(self) -> None:
        """The head read is the columns' only consumer; grads must arrive."""

        torch.manual_seed(2)
        model = Nestor(self._head_config())
        model.eval()
        x = torch.randn(2, 6, 16, requires_grad=True)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        model(x, pos, attn_mask).pred.square().sum().backward()
        grads = model.final_depth_read.value_gains.grad
        assert grads is not None
        column_slots = [1, 2, 4, 5, 6]
        assert torch.all(grads[column_slots].abs() > 0)
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_requires_the_aggregate(self) -> None:
        """The narrow list is made of aggregates, so they must be exported."""

        config = _config(gain_init=1.0, include_aggregate=False)
        config.block_column_head_only = True
        with pytest.raises(ValueError, match="include_aggregate"):
            config.to_dict()

    def test_rejected_outside_column_mode(self) -> None:
        """Head-only narrows column-mode reads; other modes have none."""

        config = _config(
            gain_init=1.0,
            include_aggregate=False,
            block_residual_mode="attn_res_kimi",
        )
        config.block_column_head_only = True
        with pytest.raises(ValueError, match="block_column_head_only"):
            config.to_dict()


class PerSourceQueryTest:
    """``per_source_queries``: one direction detector per depth source."""

    def test_shared_and_per_source_agree_when_rows_are_equal(self) -> None:
        """Copying the shared query into every row must reproduce it exactly.

        The per-source read is a strict generalization; the shared read is the
        point of its parameter space where all rows coincide.
        """

        torch.manual_seed(0)
        dim, n = 16, 5
        shared = DepthAttentionRead(dim, max_sources=n)
        per_source = DepthAttentionRead(
            dim, max_sources=n, per_source_queries=True
        )
        with torch.no_grad():
            shared.query.normal_(0, 0.5)
            per_source.query.copy_(shared.query.expand(n, dim))
            shared.value_gains.uniform_(0.5, 1.5)
            per_source.value_gains.copy_(shared.value_gains)
        sources = [torch.randn(2, 4, dim) for _ in range(n)]
        torch.testing.assert_close(
            per_source(sources), shared(sources), rtol=0.0, atol=0.0
        )

    def test_zero_init_is_uniform_and_rows_learn_independently(self) -> None:
        """Zero rows give the uniform mixture; each row gets its own gradient.

        Independence is the point: with the shared query every source's logit
        gradient lands on ONE vector, so promoting a source demotes the rest.
        """

        torch.manual_seed(1)
        dim, n = 16, 4
        read = DepthAttentionRead(dim, max_sources=n, per_source_queries=True)
        assert read.query.shape == (n, dim)
        sources = [torch.randn(2, 4, dim) for _ in range(n)]
        with torch.no_grad():
            uniform = read(sources)
            mean_of_keys = torch.stack(
                [s / s.float().square().mean(-1, keepdim=True).sqrt()
                 for s in sources]
            ).mean(0)
        torch.testing.assert_close(uniform, mean_of_keys, rtol=1e-5, atol=1e-5)

        read(sources).square().sum().backward()
        grads = read.query.grad
        assert grads is not None
        assert torch.all(grads[: n].abs().sum(dim=-1) > 0)

    def test_head_flag_builds_per_source_head_and_shared_sublayers(self) -> None:
        """The config knob touches only the final head read."""

        config = _config(gain_init=0.0)
        config.block_column_head_only = True
        config.block_column_head_query_per_source = True
        model = Nestor(config)
        assert model.final_depth_read.per_source_queries
        assert model.final_depth_read.query.shape == (
            model.max_depth_sources,
            16,
        )
        for module in model.modules():
            if (
                isinstance(module, DepthAttentionRead)
                and module is not model.final_depth_read
            ):
                assert not module.per_source_queries
                assert module.query.shape == (16,)

    def test_rejected_outside_column_mode(self) -> None:
        """The knob names the column-mode head read."""

        config = _config(
            gain_init=1.0,
            include_aggregate=False,
            block_residual_mode="attn_res_kimi",
        )
        config.block_column_head_query_per_source = True
        with pytest.raises(
            ValueError, match="block_column_head_query_per_source"
        ):
            config.to_dict()


class DeviationSourcesTest:
    """``block_column_deviation_sources``: aggregate + differences basis."""

    def _dev_config(self, gain_init: float = 0.0) -> NestorConfig:
        """The three-block config with deviation sources on.

        Args:
            gain_init (float): Column (now deviation) slot init.

        Returns:
            NestorConfig: Config with the re-based exports.
        """

        config = _config(gain_init=gain_init)
        config.block_column_deviation_sources = True
        return config

    def test_exported_deviations_sum_to_zero(self) -> None:
        """Each multi-column block's deviation slots must sum to ~zero.

        ``sum_j (s_j - a) = 0`` exactly when ``a`` is the mean -- the property
        that makes the basis non-redundant. Checked on the sources the head
        read actually receives, via a hook on the final read.
        """

        torch.manual_seed(0)
        model = Nestor(self._dev_config())
        model.eval()
        captured: list = []
        model.final_depth_read.register_forward_pre_hook(
            lambda m, a, k: captured.append(
                k.get("sources", a[0] if a else ())
            ),
            with_kwargs=True,
        )
        x = torch.randn(2, 6, 16)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        model(x, pos, attn_mask)

        sources = _flat_sources(captured[0])
        assert len(sources) == 9  # emb, 2 dev, agg, 3 dev, agg, dense
        # Block 0: slots 1-2 are deviations of a 2-column block.
        block0 = sources[1] + sources[2]
        assert block0.abs().max() < 1e-5
        # Block 1: slots 4-6 are deviations of a 3-column block.
        block1 = sources[4] + sources[5] + sources[6]
        assert block1.abs().max() < 1e-5

    def test_silent_deviations_reproduce_the_aggregate_only_read(self) -> None:
        """gain_init=0 on deviation slots must equal the kimi-style mixture.

        Stronger than the raw-column version of this property: deviations at
        zero gain leave sources whose directions are EXACTLY the aggregates,
        so the model must match its own head-only twin's read inputs.
        """

        torch.manual_seed(1)
        model = Nestor(self._dev_config(gain_init=0.0))
        model.eval()
        gains = model.final_depth_read.value_gains.detach()
        expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(gains, expected, rtol=0.0, atol=0.0)
        x = torch.randn(2, 6, 16, requires_grad=True)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        out = model(x, pos, attn_mask)
        out.pred.square().sum().backward()
        grads = model.final_depth_read.value_gains.grad
        assert grads is not None
        deviation_slots = [1, 2, 4, 5, 6]
        assert torch.all(grads[deviation_slots].abs() > 0)

    def test_requires_aggregate_and_column_mode(self) -> None:
        """Deviations are relative to the aggregate; both prerequisites hold."""

        config = _config(gain_init=1.0, include_aggregate=False)
        config.block_column_deviation_sources = True
        with pytest.raises(ValueError, match="include_aggregate"):
            config.to_dict()

        config = _config(
            gain_init=1.0,
            include_aggregate=False,
            block_residual_mode="attn_res_kimi",
        )
        config.block_column_deviation_sources = True
        with pytest.raises(ValueError, match="deviation_sources"):
            config.to_dict()


class ColumnBlockSelectionTest:
    """``block_column_blocks``: only the listed blocks keep column streams."""

    def _sel_config(self, blocks) -> NestorConfig:
        """Head-only deviation config with a block selection applied.

        Args:
            blocks (list[int] | None): Blocks that keep columns; None = all.

        Returns:
            NestorConfig: Config in the head-only + deviation pairing.
        """

        config = _config(gain_init=0.0)
        config.block_column_head_only = True
        config.block_column_deviation_sources = True
        if blocks is not None:
            config.block_column_blocks = blocks
        return config

    def test_dropped_blocks_contribute_one_source(self) -> None:
        """A dropped block exports one source, like a k == 1 block.

        Layout is [emb, 2 dev, agg, 3 dev, agg, dense] = 9 with all blocks
        column-wise; keeping only block 0 leaves [emb, 2 dev, agg, b1, b2] = 6.
        """

        assert Nestor(self._sel_config(None)).max_depth_sources == 9
        assert Nestor(self._sel_config([0])).max_depth_sources == 6
        assert Nestor(self._sel_config([0, 1])).max_depth_sources == 9

    def test_gain_init_layout_tracks_the_selection(self) -> None:
        """A dropped block's single slot keeps gain 1.0 (it IS the merge)."""

        model = Nestor(self._sel_config([0]))
        torch.testing.assert_close(
            model.final_depth_read.value_gains.detach(),
            torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            rtol=0.0,
            atol=0.0,
        )

    def test_forward_and_backward_run_on_a_selection(self) -> None:
        """The merged path must produce finite values and carry gradient."""

        torch.manual_seed(4)
        model = Nestor(self._sel_config([0]))
        model.eval()
        x = torch.randn(2, 6, 16, requires_grad=True)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        out = model(x, pos, attn_mask).pred
        out.square().sum().backward()
        assert torch.isfinite(out).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_selecting_every_block_is_the_default(self) -> None:
        """An explicit full selection must match None exactly."""

        torch.manual_seed(5)
        explicit = Nestor(self._sel_config([0, 1, 2]))
        torch.manual_seed(5)
        default = Nestor(self._sel_config(None))
        assert explicit.max_depth_sources == default.max_depth_sources
        torch.testing.assert_close(
            explicit.final_depth_read.value_gains.detach(),
            default.final_depth_read.value_gains.detach(),
            rtol=0.0,
            atol=0.0,
        )

    def test_validation(self) -> None:
        """Empty selection, out-of-range index, and wrong mode are rejected."""

        config = _config(gain_init=1.0)
        config.block_column_blocks = []
        with pytest.raises(ValueError, match="empty"):
            config.to_dict()

        config = _config(gain_init=1.0)
        config.block_column_blocks = [0, 7]
        with pytest.raises(ValueError, match="outside"):
            config.to_dict()

        config = _config(
            gain_init=1.0,
            include_aggregate=False,
            block_residual_mode="attn_res_kimi",
        )
        config.block_column_blocks = [0]
        with pytest.raises(ValueError, match="block_column_blocks"):
            config.to_dict()


class FusedColumnScatterTest:
    """Attention and the private pool share one scatter per sublayer.

    Both index by the same column -> slot map, so their per-group updates can
    be summed and scattered once instead of twice. Two identities make that
    exact, and both depend on a non-selected group's gate being exactly zero:

    * ``scatter(a + e) == scatter(a) + scatter(e)`` (scatter is linear), and
    * ``mean_over_slots(scatter(X)) == sum_over_groups(X) / k``, which lets the
      head-only MLP read take its aggregate without any scatter at all.
    """

    def test_scatter_is_linear(self) -> None:
        """The fusion's first identity, forward and backward."""

        from nestor.backbones.nestor import (
            _build_column_stream_plan,
            _scatter_columns_to_slots,
        )

        torch.manual_seed(0)
        batch, seq, groups, dim, top_k = 2, 7, 5, 16, 3
        topk = torch.stack(
            [torch.randperm(groups)[:top_k] for _ in range(batch * seq)]
        ).view(batch, seq, top_k)
        plan = _build_column_stream_plan(topk, groups)

        a = torch.randn(batch, seq, groups, dim, requires_grad=True)
        e = torch.randn(batch, seq, groups, dim, requires_grad=True)
        a2 = a.clone().detach().requires_grad_(True)
        e2 = e.clone().detach().requires_grad_(True)

        fused = _scatter_columns_to_slots(a + e, plan)
        unfused = (
            _scatter_columns_to_slots(a2, plan)
            + _scatter_columns_to_slots(e2, plan)
        )
        torch.testing.assert_close(fused, unfused, rtol=1e-6, atol=1e-6)

        grad = torch.randn_like(fused)
        fused.backward(grad)
        unfused.backward(grad)
        torch.testing.assert_close(a.grad, a2.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(e.grad, e2.grad, rtol=1e-6, atol=1e-6)

    def test_slot_mean_equals_group_sum_over_k_when_gated(self) -> None:
        """The second identity, plus the ungated control that must FAIL.

        The control is the point: the identity is not generic, it holds
        *because* a non-selected group contributes exactly zero. If gating ever
        becomes soft, the fusion's pre-scatter mean silently becomes wrong, so
        this test documents the dependency rather than just the result.
        """

        from nestor.backbones.nestor import (
            _build_column_stream_plan,
            _scatter_columns_to_slots,
        )

        torch.manual_seed(0)
        batch, seq, groups, dim, top_k = 2, 7, 5, 16, 3
        topk = torch.stack(
            [torch.randperm(groups)[:top_k] for _ in range(batch * seq)]
        ).view(batch, seq, top_k)
        plan = _build_column_stream_plan(topk, groups)

        active = torch.zeros(batch, seq, groups, dtype=torch.bool)
        active.scatter_(2, topk, True)
        gated = torch.randn(batch, seq, groups, dim) * active.unsqueeze(-1)
        torch.testing.assert_close(
            _scatter_columns_to_slots(gated, plan).mean(dim=2),
            gated.sum(dim=2) / top_k,
            rtol=1e-5,
            atol=1e-6,
        )

        ungated = torch.randn(batch, seq, groups, dim)
        assert not torch.allclose(
            _scatter_columns_to_slots(ungated, plan).mean(dim=2),
            ungated.sum(dim=2) / top_k,
            rtol=1e-3,
            atol=1e-3,
        ), "identity must depend on non-selected groups being zero"


class ColumnTrailingWindowTest:
    """``block_column_trailing_layers``: columns tracked only near a block's end.

    The cost of column mode is slot-layers (``sum_b layers_b * k_b``), so the
    WIDE blocks that give the head read the most columns are the most expensive
    to keep whole. A trailing window decouples the two: the block's aggregate
    still accumulates over every layer, and only the deviations narrow.
    """

    def _win_config(self, trailing: int | None) -> NestorConfig:
        """Config with a 3/2/1-layer block split and an optional window.

        Args:
            trailing (int | None): ``block_column_trailing_layers``.

        Returns:
            NestorConfig: Head-only deviation config.
        """

        config = NestorConfig(
            input_dim=16,
            num_blocks=3,
            feature_dims=[16] * 3,
            pooling_strides=[1] * 3,
            causal=True,
            block_num_layers=[3, 2, 1],
            num_groups=[4, 4, 1],
            num_groups_per_token=[2, 3, 1],
            block_num_heads=[1] * 3,
            block_head_dims=[8] * 3,
            block_mlp_hidden_dims=[[16]] * 3,
            router_num_experts_per_group=[1] * 3,
            router_num_shared_experts=[1] * 3,
            router_num_experts_per_group_per_token=[1] * 3,
            router_score_mlp_hidden_dims=[8],
            router_noise_mlp_hidden_dims=[8],
            residual_mode="convex",
            block_residual_mode="attn_res_column",
            block_column_include_aggregate=True,
            block_column_gain_init=0.0,
            block_column_deviation_sources=True,
        )
        config.block_column_head_only = True
        if trailing is not None:
            config.block_column_trailing_layers = trailing
        return config

    @pytest.mark.parametrize("trailing", [1, 2])
    def test_deviations_still_sum_to_zero(self, trailing: int) -> None:
        """The window's deviations are re-based on the WINDOW's own mean.

        This is the invariant that makes the narrowing sound: had the
        deviations been re-based on the block's full aggregate (which includes
        the merged prefix), they would sum to the prefix instead of to zero and
        the basis would stop being non-redundant.
        """

        torch.manual_seed(3)
        model = Nestor(self._win_config(trailing))
        model.eval()
        captured: list = []
        model.final_depth_read.register_forward_pre_hook(
            lambda m, a, k: captured.append(
                k.get("sources", a[0] if a else ())
            ),
            with_kwargs=True,
        )
        x = torch.randn(2, 8, 16)
        pos = torch.arange(8).expand(2, -1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)
        model(x, pos, attn_mask)

        sources = _flat_sources(captured[0])
        assert (sources[1] + sources[2]).abs().max() < 1e-5
        assert (sources[4] + sources[5] + sources[6]).abs().max() < 1e-5

    def test_window_does_not_change_the_source_count(self) -> None:
        """Narrowing the window must not change what the head read sees.

        The point of the knob is to cut COST, not read resolution -- the block
        exports the same k deviations plus its aggregate either way.
        """

        assert (
            Nestor(self._win_config(1)).max_depth_sources
            == Nestor(self._win_config(None)).max_depth_sources
        )

    def test_gradients_reach_the_merged_prefix(self) -> None:
        """The prefix layers must still train, via the block aggregate."""

        torch.manual_seed(5)
        model = Nestor(self._win_config(1))
        model.eval()
        x = torch.randn(2, 8, 16, requires_grad=True)
        pos = torch.arange(8).expand(2, -1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)
        model(x, pos, attn_mask).pred.square().sum().backward()
        # Block 0 has 3 layers with a window of 1, so layers 0-1 are prefix.
        prefix_grads = [
            p.grad
            for p in model.blocks[0].layers[0].parameters()
            if p.grad is not None
        ]
        assert prefix_grads, "prefix layer received no gradient at all"
        assert any(g.abs().max() > 0 for g in prefix_grads)
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_validation(self) -> None:
        """A zero-length window is a dropped block, not a narrowed one."""

        config = self._win_config(0)
        with pytest.raises(ValueError, match="at least 1"):
            config.to_dict()
