"""Reference and integration tests for the Kimi-faithful block AttnRes mode."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.attention_residual import DepthAttentionRead, _depth_softmax
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.constants import PADDING_TOKEN_POS
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig


def _reference_depth_attention_read(
    module: DepthAttentionRead,
    sources: list[Tensor],
) -> Tensor:
    """Slow, loop-based transcription of the muP-adapted depth read.

    Paper Eq. 3-6 (weight-free RMSNorm keys, one shared pseudo-query, softmax
    over the depth axis) plus this repo's two adaptations: values are the
    unit-RMS keys scaled by ``value_gains``, and the depth logits carry the
    ``1/sqrt(dim)`` temperature. Written with explicit Python loops and a
    hand-rolled softmax so correctness is obvious by inspection.

    Args:
        module (DepthAttentionRead): Module supplying ``query``, ``value_gains``
            and ``eps``.
        sources (list[Tensor]): Depth sources, each ``[batch, seq, dim]``.

    Returns:
        Tensor: Depth-attended mixture ``[batch, seq, dim]``.
    """

    keys: list[Tensor] = []  # n x [batch, seq, dim]
    logits: list[Tensor] = []  # n x [batch, seq]
    for source in sources:
        rms = (
            source.float().square().mean(dim=-1, keepdim=True) + module.eps
        ).sqrt()  # [batch, seq, 1]
        key = source.float() / rms  # [batch, seq, dim]
        keys.append(key)
        logits.append(
            (key * module.query.float()).sum(dim=-1) / module.dim ** 0.5
        )

    shift = torch.stack(logits, dim=0).max(dim=0).values  # [batch, seq]
    weights = [(logit - shift).exp() for logit in logits]  # n x [batch, seq]
    denominator = torch.stack(weights, dim=0).sum(dim=0)  # [batch, seq]

    mixture = torch.zeros_like(keys[0])  # [batch, seq, dim]
    for index, (key, weight) in enumerate(zip(keys, weights)):
        alpha = (weight / denominator).unsqueeze(-1)  # [batch, seq, 1]
        mixture = mixture + key * module.value_gains[index].float() * alpha
    return mixture.to(sources[0].dtype)


def _reference_stacked_depth_read(
    module: DepthAttentionRead,
    sources: list[Tensor],
) -> Tensor:
    """The pre-streaming ``DepthAttentionRead.forward``, written out verbatim.

    Materializes the ``[n, batch, seq, dim]`` stack (and the stacked keys and
    weighted values) exactly as the module used to, so the streaming
    implementation can be checked against the formulation it replaced.

    Args:
        module (DepthAttentionRead): Module supplying the parameters.
        sources (list[Tensor]): Depth sources, each ``[batch, seq, dim]``.

    Returns:
        Tensor: Depth-attended mixture ``[batch, seq, dim]``.
    """

    stacked = torch.stack(list(sources), dim=0)  # [n, batch, seq, dim]
    inv_rms = torch.rsqrt(
        stacked.float().square().mean(dim=-1, keepdim=True) + module.eps
    )  # [n, batch, seq, 1]
    keys = (stacked.float() * inv_rms).to(stacked.dtype)
    logits = torch.einsum("d,nbtd->nbt", module.query.to(keys.dtype), keys)
    logits = logits / (module.dim ** 0.5)
    alpha = logits.float().softmax(dim=0).to(stacked.dtype)  # [n, batch, seq]
    gains = module.value_gains[: stacked.shape[0]].to(keys.dtype)  # [n]
    values = keys * gains.view(-1, 1, 1, 1)  # [n, batch, seq, dim]
    return (values * alpha.unsqueeze(-1)).sum(dim=0)


def _randomize_depth_read(module: DepthAttentionRead, seed: int) -> None:
    """Give every parameter of a depth read a non-degenerate value.

    A zero query and unit gains collapse the read to a plain uniform mean of
    unit-RMS sources, which would hide weighting bugs.

    Args:
        module (DepthAttentionRead): Module to randomize in place.
        seed (int): ``torch.manual_seed`` value used before the draws.
    """

    torch.manual_seed(seed)
    with torch.no_grad():
        module.query.normal_()
        module.value_gains.uniform_(0.5, 1.5)


def _reference_kimi_backbone_forward(
    model: Nestor,
    x: Tensor,
    pos: Tensor,
    attn_mask: Tensor,
) -> Tensor:
    """Hand-rolled Kimi block AttnRes recurrence (paper Eq. 3-6 + Fig. 2).

    Threads the completed-block list ``kimi_blocks`` and the within-block
    partial sum explicitly, calling the model's OWN submodules (depth reads,
    norms, routers, sublayer-update helpers, pooling) for every operation, so
    the production forward must match it bit-for-bit. Slow but obviously
    faithful to the paper recurrence:

    - sublayer input  = depth read over ``[*kimi_blocks, partial]``,
    - ``partial``     = running sum of the block's sublayer updates,
    - block output    = ``partial`` (pooled/masked at the block exit),
    - head input      = final depth read over all block representations.

    Assumes ``model.num_global == 0`` (enforced by the kimi config checks)
    and eval mode (no drop-path randomness).

    Args:
        model (Nestor): Backbone built with
            ``block_residual_mode='attn_res_kimi'``.
        x (Tensor): Token embeddings ``[batch_size, seq_len, input_dim]``.
        pos (Tensor): Token positions ``[batch_size, seq_len]``.
        attn_mask (Tensor): Base attention mask
            ``[batch_size, seq_len, seq_len]``.

    Returns:
        Tensor: Backbone output ``[batch_size, seq_len, output_dim]``.
    """

    kimi_blocks: list[Tensor] = [x]
    for block in model.blocks:
        valid_mask = pos != PADDING_TOKEN_POS
        block_in = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        composed_mask = block._compose_attn_mask(valid_mask, attn_mask)
        group_router_output = block.group_router(block_in, valid_mask)
        partial: Tensor | None = None
        for layer in block.layers:
            sources = (
                kimi_blocks if partial is None else [*kimi_blocks, partial]
            )
            h = layer.kimi_attn_read(sources)
            if layer._dense_fastpath:
                attn_update = layer._attn_update_dense(
                    layer.attn_norm(h), composed_mask, group_router_output, pos
                )
            else:
                attn_update = layer._attn_update_general(
                    layer.attn_norm(h), composed_mask, group_router_output, pos
                )
            partial = (
                attn_update if partial is None else partial + attn_update
            )
            h = layer.kimi_mlp_read([*kimi_blocks, partial])
            if layer._dense_fastpath:
                expert_update, _ = layer._expert_update_dense(
                    layer.mlp_norm(h), valid_mask, group_router_output
                )
            else:
                expert_update, _ = layer._expert_update_general(
                    layer.mlp_norm(h), valid_mask, group_router_output
                )
            partial = partial + expert_update
        assert partial is not None
        x, pos, attn_mask = block._pool_tokens(partial, pos, attn_mask)
        x = block.output_proj(x)
        x = x * (pos != PADDING_TOKEN_POS).unsqueeze(-1).to(dtype=x.dtype)
        kimi_blocks.append(x)
    return model.final_norm(model.final_depth_read(kimi_blocks))


def _randomize_kimi_queries(model: Nestor, seed: int) -> None:
    """Give every depth read a seeded non-degenerate query and value gains.

    Zero queries and unit gains make every depth read a uniform mean of
    unit-RMS sources, which would hide weighting bugs; the fixed seed keeps two
    identically built models identical after randomization.

    Args:
        model (Nestor): Backbone with ``block_residual_mode='attn_res_kimi'``.
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


def _create_kimi_config(*, causal: bool = True) -> NestorConfig:
    """Create a tiny constant-width dense Nestor config in Kimi AttnRes mode."""

    return NestorConfig(
        input_dim=16,
        num_blocks=3,
        feature_dims=[16, 16, 16],
        pooling_strides=[1, 1, 1],
        causal=causal,
        block_num_layers=[1, 1, 1],
        num_groups=[1, 1, 1],
        num_groups_per_token=[1, 1, 1],
        block_num_heads=[1, 1, 1],
        block_head_dims=[8, 8, 8],
        block_mlp_hidden_dims=[[16], [16], [16]],
        router_num_experts_per_group=[1, 1, 1],
        router_num_shared_experts=[1, 1, 1],
        router_num_experts_per_group_per_token=[1, 1, 1],
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="attn_res_kimi",
    )


def _create_kimi_moe_config(*, causal: bool = True) -> NestorConfig:
    """Create a tiny multi-group / multi-expert (general-path) Kimi config."""

    return NestorConfig(
        input_dim=16,
        num_blocks=2,
        feature_dims=[16, 16],
        pooling_strides=[1, 1],
        causal=causal,
        block_num_layers=[2, 1],
        num_groups=[2, 2],
        num_groups_per_token=[1, 1],
        block_num_heads=[1, 1],
        block_head_dims=[8, 8],
        block_mlp_hidden_dims=[[16], [16]],
        router_num_experts_per_group=[2, 2],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=[1, 1],
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="attn_res_kimi",
    )


class DepthAttentionReadReferenceTest:
    """Numerical checks against a slow transcription of the depth read."""

    def test_forward_matches_loop_reference(self) -> None:
        """The production forward should match the explicit-loop reference."""

        torch.manual_seed(0)
        module = DepthAttentionRead(8, max_sources=4)
        _randomize_depth_read(module, seed=1)
        torch.manual_seed(2)
        sources = [torch.randn(2, 5, 8) for _ in range(4)]

        output = module(sources)
        reference = _reference_depth_attention_read(module, sources)
        torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    def test_zero_init_query_is_uniform_gain_scaled_mean(self) -> None:
        """A fresh query must average the gain-scaled unit-RMS sources."""

        torch.manual_seed(1)
        module = DepthAttentionRead(6, max_sources=3)
        with torch.no_grad():
            # Non-unit gains so the expected mixture is not just the mean of
            # the source directions.
            module.value_gains.copy_(torch.tensor([0.5, 1.0, 2.0]))
        sources = [torch.randn(2, 4, 6) for _ in range(3)]

        unit_rms = torch.stack(
            [
                source
                * torch.rsqrt(
                    source.square().mean(dim=-1, keepdim=True) + module.eps
                )
                for source in sources
            ],
            dim=0,
        )  # [3, batch, seq, dim]
        gains = module.value_gains.detach().view(-1, 1, 1, 1)
        expected = (unit_rms * gains).mean(dim=0)
        torch.testing.assert_close(module(sources), expected, rtol=0.0, atol=1e-6)

    def test_gradient_flows_to_query_and_value_gains(self) -> None:
        """Both learned tensors must receive non-zero gradients."""

        torch.manual_seed(2)
        module = DepthAttentionRead(6, max_sources=3)
        sources = [torch.randn(2, 4, 6) for _ in range(3)]

        module(sources).square().sum().backward()
        assert module.query.grad is not None
        assert torch.count_nonzero(module.query.grad) > 0
        assert module.value_gains.grad is not None
        assert torch.count_nonzero(module.value_gains.grad) > 0

    def test_unused_value_gains_stay_gradient_free(self) -> None:
        """Gains beyond ``len(sources)`` are sliced off, so they get no grad."""

        torch.manual_seed(3)
        module = DepthAttentionRead(6, max_sources=4)
        sources = [torch.randn(2, 4, 6) for _ in range(2)]

        module(sources).square().sum().backward()
        assert module.value_gains.grad is not None
        assert torch.count_nonzero(module.value_gains.grad[2:]) == 0

    def test_rejects_empty_sources(self) -> None:
        """At least one depth source is part of the API contract."""

        module = DepthAttentionRead(4, max_sources=2)
        with pytest.raises(ValueError, match="at least one source"):
            module([])

    def test_rejects_non_positive_max_sources(self) -> None:
        """The gain bank must be allocatable, so ``max_sources`` >= 1."""

        with pytest.raises(ValueError, match="max_sources must be positive"):
            DepthAttentionRead(4, max_sources=0)


class DepthAttentionReadStreamingEquivalenceTest:
    """The streamed read must match the old stacked formulation everywhere."""

    def test_matches_stacked_reference(self) -> None:
        """The streamed read must reproduce the stacked formulation."""

        torch.manual_seed(40)
        module = DepthAttentionRead(12, max_sources=6)
        _randomize_depth_read(module, seed=41)
        torch.manual_seed(42)
        sources = [torch.randn(3, 5, 12) for _ in range(5)]

        output = module(sources)
        reference = _reference_stacked_depth_read(module, sources)
        assert output.dtype == reference.dtype
        torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)

    def test_single_source_matches_stacked_reference(self) -> None:
        """The degenerate one-source read (alpha == 1) stays equivalent."""

        torch.manual_seed(43)
        module = DepthAttentionRead(8, max_sources=3)
        _randomize_depth_read(module, seed=44)
        torch.manual_seed(45)
        sources = [torch.randn(2, 4, 8)]

        torch.testing.assert_close(
            module(sources),
            _reference_stacked_depth_read(module, sources),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_matches_stacked_reference_under_bf16_autocast(self) -> None:
        """bf16 autocast with MIXED source dtypes keeps stack's promotion.

        The real backbone feeds this read an fp32 residual entry point plus
        bf16 block outputs; ``torch.stack`` promoted the mixture to fp32, so
        the factored version must promote identically.

        Values agree only to bf16 precision here, and that is the intended
        contract rather than a slack tolerance. The stacked reference builds an
        explicit fp32 key per source, so it dots the query against an fp32
        tensor even for a bf16 source; the factored read never materializes a
        key, so it dots the query against the bf16 source itself. Upcasting a
        bf16 stream is information-free -- the only thing the reference gains is
        that the QUERY is not rounded to bf16, worth ~2^-8 relative on the
        logit, which is below the bf16 rounding the mixed VALUES already carry.
        Buying it back would mean retaining an fp32 copy of every source, which
        is exactly the cost the factored form exists to remove.
        """

        torch.manual_seed(46)
        module = DepthAttentionRead(16, max_sources=4)
        _randomize_depth_read(module, seed=47)
        torch.manual_seed(48)
        sources = [
            torch.randn(2, 4, 16),
            torch.randn(2, 4, 16).to(torch.bfloat16),
            torch.randn(2, 4, 16).to(torch.bfloat16),
        ]

        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(sources)
            reference = _reference_stacked_depth_read(module, sources)
        assert output.dtype == torch.float32
        assert reference.dtype == torch.float32
        bf16_eps = torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(
            output, reference, rtol=4 * bf16_eps, atol=4 * bf16_eps
        )

    def test_all_fp32_sources_match_the_stacked_reference_tightly(self) -> None:
        """No source is downcast when every source is already fp32.

        The guard on the tolerance above: the bf16 gap there comes from the
        bf16 SOURCES, not from the factorization. Hand the same read an
        all-fp32 list and every per-source dot product runs in fp32, so the
        factored and explicit-key forms must agree to fp32 rounding.
        """

        torch.manual_seed(46)
        module = DepthAttentionRead(16, max_sources=4)
        _randomize_depth_read(module, seed=47)
        torch.manual_seed(48)
        sources = [torch.randn(2, 4, 16) for _ in range(3)]

        output = module(sources)
        reference = _reference_stacked_depth_read(module, sources)
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)

    def test_matches_stacked_reference_in_bfloat16(self) -> None:
        """An all-bf16 source list mixes in bf16, as the stacked version did.

        Exact equality is not expected here: the streamed read folds the value
        gains into the (fp32) depth weights and accumulates with a fused
        ``addcmul_``, so a product that the stacked version rounded to bf16
        twice is now rounded once. Both effects are last-bit only, so this
        bounds the gap at two ULP of the largest reference element. (The real
        backbone promotes this read to fp32 -- see the autocast test -- where
        the two formulations agree to fp32 rounding noise.)
        """

        torch.manual_seed(49)
        module = DepthAttentionRead(16, max_sources=4)
        _randomize_depth_read(module, seed=50)
        torch.manual_seed(51)
        sources = [
            torch.randn(2, 4, 16).to(torch.bfloat16) for _ in range(4)
        ]

        output = module(sources)
        reference = _reference_stacked_depth_read(module, sources)
        assert output.dtype == torch.bfloat16
        two_ulp = (
            2.0
            * torch.finfo(torch.bfloat16).eps
            * float(reference.detach().float().abs().max())
        )
        torch.testing.assert_close(output, reference, rtol=0.0, atol=two_ulp)

    def test_non_contiguous_sources_match_stacked_reference(self) -> None:
        """Strided source views survive the GEMV reshape in the logit path.

        The streamed read flattens each key to ``[batch*seq, dim]`` for the
        matrix-vector logit product; a non-contiguous source must therefore
        still produce the stacked reference's answer.
        """

        torch.manual_seed(55)
        module = DepthAttentionRead(8, max_sources=3)
        _randomize_depth_read(module, seed=56)
        torch.manual_seed(57)
        # Slice a wider buffer on the feature axis => strided [batch, seq, dim].
        wide = torch.randn(2, 5, 3, 16)
        sources = [wide[:, :, idx, :8] for idx in range(3)]
        assert not sources[0].is_contiguous()

        torch.testing.assert_close(
            module(sources),
            _reference_stacked_depth_read(module, sources),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gradients_match_stacked_reference(self) -> None:
        """Parameter gradients must match the stacked formulation too."""

        torch.manual_seed(52)
        module = DepthAttentionRead(10, max_sources=4)
        _randomize_depth_read(module, seed=53)
        torch.manual_seed(54)
        sources = [torch.randn(2, 3, 10, requires_grad=True) for _ in range(4)]

        module(sources).square().sum().backward()
        streamed = {
            name: param.grad.clone()
            for name, param in module.named_parameters()
            if param.grad is not None
        }
        streamed_inputs = [source.grad.clone() for source in sources]
        module.zero_grad(set_to_none=True)
        for source in sources:
            source.grad = None

        _reference_stacked_depth_read(module, sources).square().sum().backward()
        for name, param in module.named_parameters():
            torch.testing.assert_close(
                streamed[name], param.grad, rtol=1e-5, atol=1e-6
            )
        for streamed_grad, source in zip(streamed_inputs, sources):
            torch.testing.assert_close(
                streamed_grad, source.grad, rtol=1e-5, atol=1e-6
            )


class NestorKimiBlockAttentionResidualTest:
    """End-to-end checks for the ``attn_res_kimi`` backbone wiring."""

    def test_forward_backward_and_stateless_layers(self) -> None:
        """Forward shape/finiteness, query gradients, and stateless layers."""

        torch.manual_seed(3)
        config = _create_kimi_config()
        model = Nestor(config)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        output = model(x, pos, attn_mask)
        assert output.pred.shape == (2, 5, config.output_dim)
        assert torch.isfinite(output.pred).all()
        output.pred.square().mean().backward()

        for block in model.blocks:
            for layer in block.layers:
                assert layer.kimi_attn_read.query.grad is not None
                assert layer.kimi_mlp_read.query.grad is not None
                # The Kimi path threads state functionally (torch.compile-
                # friendly); the old transient-attribute machinery must stay
                # deleted.
                assert not hasattr(layer, "_kimi_sources")
                assert not hasattr(layer, "_kimi_partial")
                assert not hasattr(layer, "_kimi_stage")
        assert model.final_depth_read.query.grad is not None
        assert len(model.block_attention_residuals) == 0

    def test_repeated_forward_is_identical(self) -> None:
        """Transient per-forward state must not leak into a second forward."""

        torch.manual_seed(4)
        config = _create_kimi_config()
        model = Nestor(config).eval()
        # Non-trivial depth weights so state leakage could actually show up.
        with torch.no_grad():
            for block in model.blocks:
                for layer in block.layers:
                    layer.kimi_attn_read.query.normal_()
                    layer.kimi_mlp_read.query.normal_()
            model.final_depth_read.query.normal_()
        x = torch.randn(2, 4, config.input_dim)
        pos = torch.arange(4).expand(2, -1)
        attn_mask = torch.ones(2, 4, 4, dtype=torch.bool)

        with torch.no_grad():
            first = model(x, pos, attn_mask).pred
            second = model(x, pos, attn_mask).pred
        assert torch.equal(first, second)

    def test_forward_step_raises(self) -> None:
        """Incremental KV-cache decode is explicitly unsupported."""

        torch.manual_seed(5)
        config = _create_kimi_config(causal=True)
        model = Nestor(config).eval()
        x = torch.randn(2, 3, config.input_dim)
        pos = torch.arange(3).expand(2, -1)
        cache = model.init_kv_cache(batch_size=2)

        with pytest.raises(NotImplementedError, match="attn_res_kimi"):
            model.forward_step(x[:, 0], pos[:, 0], cache)

    def test_rejects_varying_width(self) -> None:
        """Raw (projection-free) depth reads require one shared width."""

        config = _create_kimi_config()
        config.feature_dims = [16, 24, 16]
        with pytest.raises(ValueError, match="requires constant width"):
            Nestor(config)

    def test_rejects_token_pooling(self) -> None:
        """Per-token depth mixing requires aligned token positions."""

        config = _create_kimi_config()
        config.pooling_strides = [2, 1, 1]
        with pytest.raises(ValueError, match="requires constant token count"):
            Nestor(config)

    def test_reference_mup_transform_composes(self) -> None:
        """The reference muP transform must apply cleanly on a Kimi model."""

        torch.manual_seed(6)
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(embed_dim=16, vocab_size=64),
            backbone_config=_create_kimi_config(causal=True),
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


class NestorKimiFunctionalEquivalenceTest:
    """Bit-exact checks of the functional threading against the paper loop."""

    def test_dense_path_matches_handrolled_recurrence_exactly(self) -> None:
        """Dense fast-path forward must equal the reference loop at atol 0."""

        torch.manual_seed(20)
        config = _create_kimi_config()
        # Two layers in the first block so a layer receives a non-None
        # partial sum in its attention read (single-layer blocks never do).
        config.block_num_layers = [2, 1, 1]
        model = Nestor(config)
        torch.manual_seed(20)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_kimi_queries(candidate, seed=21)
            candidate.eval()
        assert all(
            layer._dense_fastpath
            for block in model.blocks
            for layer in block.layers
        )

        torch.manual_seed(22)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        torch.manual_seed(23)
        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
        torch.manual_seed(23)
        with torch.no_grad():
            expected = _reference_kimi_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_general_path_matches_handrolled_recurrence_exactly(self) -> None:
        """General (multi-group MoE) forward must equal the reference loop."""

        torch.manual_seed(24)
        config = _create_kimi_moe_config()
        model = Nestor(config)
        torch.manual_seed(24)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_kimi_queries(candidate, seed=25)
            candidate.eval()
        assert not any(
            layer._dense_fastpath
            for block in model.blocks
            for layer in block.layers
        )

        torch.manual_seed(26)
        x = torch.randn(2, 4, config.input_dim)
        pos = torch.arange(4).expand(2, -1)
        attn_mask = torch.ones(2, 4, 4, dtype=torch.bool)

        torch.manual_seed(27)
        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
        torch.manual_seed(27)
        with torch.no_grad():
            expected = _reference_kimi_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class NestorKimiCompileTest:
    """torch.compile smoke test for the functional Kimi threading."""

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

        torch.manual_seed(30)
        config = _create_kimi_config()
        config.block_num_layers = [2, 1, 1]
        model = Nestor(config)
        _randomize_kimi_queries(model, seed=31)
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


class DepthSoftmaxHelperTest:
    """The extracted depth-softmax step (the graph-break point).

    ``_depth_softmax`` was factored out of :meth:`DepthAttentionRead.forward` so
    ``NESTOR_DEPTH_READ_EAGER=1`` can graph-break there when the source count
    exceeds what Inductor's persistent reduction can fit in shared memory. The
    extraction must be numerically inert.
    """

    def reference_depth_softmax(
        self,
        depth_logits: Tensor,  # [batch, seq, n]
        gains: Tensor,  # [n]
    ) -> Tensor:
        """Slow, explicit per-position softmax over the depth axis.

        Args:
            depth_logits (Tensor): ``[batch, seq, n]`` logits.
            gains (Tensor): ``[n]`` per-source value gains.

        Returns:
            Tensor: ``[batch, seq, n]`` gain-scaled fp32 weights.
        """

        batch, seq, n = depth_logits.shape
        out = torch.zeros(batch, seq, n, dtype=torch.float32)
        for b in range(batch):
            for t in range(seq):
                column = depth_logits[b, t, :].float()
                shifted = column - column.max()
                probs = shifted.exp() / shifted.exp().sum()
                for i in range(n):
                    out[b, t, i] = probs[i] * float(gains[i])
        return out

    def test_matches_reference(self) -> None:
        """Helper reproduces an explicit per-position softmax."""

        torch.manual_seed(0)
        logits = torch.randn(2, 5, 6)
        gains = torch.randn(6).abs() + 0.1
        expected = self.reference_depth_softmax(logits, gains)
        assert torch.allclose(_depth_softmax(logits, gains), expected, atol=1e-6)

    def test_weights_form_a_simplex_before_gains(self) -> None:
        """With unit gains the weights sum to one at every position."""

        torch.manual_seed(1)
        logits = torch.randn(3, 4, 9)
        weights = _depth_softmax(logits, torch.ones(9))
        assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 4), atol=1e-6)

    def test_wide_source_stack_is_supported(self) -> None:
        """A 35-source read -- the widest arm in the schedule series -- works.

        This is the shape that killed Inductor codegen on H200
        (``lin16_colres``), so it is worth pinning as a supported input.
        """

        torch.manual_seed(2)
        logits = torch.randn(2, 8, 35)
        weights = _depth_softmax(logits, torch.ones(35))
        assert weights.shape == (2, 8, 35)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-6)
        assert torch.isfinite(weights).all()


class DepthReadAutocastDtypeFlagTest:
    """``NESTOR_DEPTH_READ_AUTOCAST=1`` narrows the mixture to bf16.

    The read's output dtype is the promotion over its sources, and the real
    backbone passes one fp32 source (the embedding) beside bf16 block states,
    so every mixture comes out fp32 and every read retains an fp32
    ``[batch, seq, dim]`` activation for the norm that consumes it. The flag
    opts a fresh run into mixing at the autocast dtype instead.

    The flag is read once at import, like ``NESTOR_DEPTH_READ_EAGER``, so these
    run the real module in a subprocess rather than reaching into its state.
    """

    def run_probe(self, flag: str) -> str:
        """Report the read's output dtype under autocast with the flag set.

        Args:
            flag (str): Value for ``NESTOR_DEPTH_READ_AUTOCAST``.

        Returns:
            str: Stripped stdout of the probe, i.e. the output dtype's name.
        """

        import os
        import subprocess
        import sys

        probe = (
            "import torch\n"
            "from nestor.backbones.attention_residual import DepthAttentionRead\n"
            "torch.manual_seed(0)\n"
            "read = DepthAttentionRead(16, max_sources=4)\n"
            "sources = [torch.randn(2, 4, 16),\n"
            "           torch.randn(2, 4, 16).to(torch.bfloat16)]\n"
            "with torch.autocast('cpu', dtype=torch.bfloat16):\n"
            "    print(read(sources).dtype)\n"
        )
        env = {**os.environ, "NESTOR_DEPTH_READ_AUTOCAST": flag}
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return result.stdout.strip()

    def test_default_keeps_the_promoted_fp32_mixture(self) -> None:
        """Unset, the read still promotes to fp32 -- in-flight runs are safe."""

        assert self.run_probe("0") == "torch.float32"

    def test_flag_mixes_at_the_autocast_dtype(self) -> None:
        """Set, one fp32 source no longer promotes the whole mixture."""

        assert self.run_probe("1") == "torch.bfloat16"
