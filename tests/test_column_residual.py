"""Reference and integration tests for column-wise residual streams.

Covers ``block_residual_mode='attn_res_column'``: a Nestor block routed to
``k = num_groups_per_token`` attention groups ("columns") carries ``k``
parallel residual streams, each column writing only into its own, and exports
all ``k`` as separate depth-read sources at the block boundary.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.nestor import _build_column_stream_plan
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.constants import PADDING_TOKEN_POS
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig


def _create_column_config(
    *,
    causal: bool = True,
    num_groups: list[int] | None = None,
    num_groups_per_token: list[int] | None = None,
    block_num_layers: list[int] | None = None,
) -> NestorConfig:
    """Create a tiny two-block routed Nestor config in column-stream mode.

    Block 0 is grouped (4 columns, top-2 by default) and block 1 is dense
    (1 column) -- the g8t3 shape the production arm uses, at toy size.

    Args:
        causal (bool): Whether the backbone runs causally.
        num_groups (list[int] | None): Per-block column count; defaults to
            ``[4, 1]``.
        num_groups_per_token (list[int] | None): Per-block ``k``; defaults to
            ``[2, 1]``.
        block_num_layers (list[int] | None): Layers per block; defaults to
            ``[2, 1]``.

    Returns:
        NestorConfig: Constant-width config with
        ``block_residual_mode='attn_res_column'``.
    """

    return NestorConfig(
        input_dim=16,
        num_blocks=2,
        feature_dims=[16, 16],
        pooling_strides=[1, 1],
        causal=causal,
        block_num_layers=[2, 1] if block_num_layers is None else block_num_layers,
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
    )


def _create_pooled_column_config(
    *,
    public_experts: int = 2,
    private_experts: int = 2,
    private_top_k: int = 1,
    block_residual_mode: str = "attn_res_column",
) -> NestorConfig:
    """Create the tiny column config with public/private expert pools.

    Every layer's MLP becomes an
    :class:`~nestor.backbones.expert_pools.ExpertPoolMLP`: a shared public pool
    plus one private pool per attention group (``public_experts=0`` builds the
    private-only variant the production arms use).

    Args:
        public_experts (int): Public pool size; 0 = private-only mode.
        private_experts (int): Private pool size per attention group.
        private_top_k (int): Private experts active per selected group.
        block_residual_mode (str): Residual topology to build; the pooled
            ``attn_res_kimi`` model is the non-regression reference.

    Returns:
        NestorConfig: Pooled variant of :func:`_create_column_config`.
    """

    config = _create_column_config()
    config.block_residual_mode = block_residual_mode
    config.block_expert_pool_public_experts = [public_experts, public_experts]
    config.block_expert_pool_public_top_k = [1, 1]
    config.block_expert_pool_private_experts = [private_experts, private_experts]
    config.block_expert_pool_private_top_k = [private_top_k, private_top_k]
    return config


def _reference_pool_parts(
    layer: torch.nn.Module,
    norm_h: Tensor,
    valid_mask: Tensor,
    router: object,
) -> tuple[Tensor, list[Tensor]]:
    """Hand-rolled pool equation, kept split into its public/private parts.

    Spells out the update from ``nestor/backbones/expert_pools.py``::

        MLP(h) = alpha_public  * sum_e public_gate_e  * PublicExpert_e(h)
               + alpha_private * sum_g attn_gate_g
                               * sum_e private_gate_(g,e) * PrivateExpert_(g,e)(h)

    with dense per-expert evaluation, calling the pool's OWN routers, experts
    and path gate. Routers run public-first then in group order, matching
    ``ExpertPoolMLP._route`` (irrelevant in eval, where no noise is drawn, but
    it keeps the reference honest).

    Args:
        layer (torch.nn.Module): A ``RoutedTransformerLayer`` with a pool.
        norm_h (Tensor): Pre-normalized MLP input ``[batch, seq, dim]``.
        valid_mask (Tensor): Real-token mask ``[batch, seq]``.
        router (object): The block's group ``RouterOutput``.

    Returns:
        tuple[Tensor, list[Tensor]]: The path-gated public update
        ``[batch, seq, dim]`` (all-zero in private-only mode) and the
        path-gated per-attention-group private updates, ``num_groups`` tensors
        of ``[batch, seq, dim]``.
    """

    pool = layer.expert_pool

    public = torch.zeros_like(norm_h)
    if pool.public_router is not None:
        public_output = pool.public_router(norm_h, valid_mask)
        for expert_idx, expert in enumerate(pool.public_experts):
            public = public + expert(norm_h) * public_output.gates[
                ..., expert_idx : expert_idx + 1
            ]

    per_column: list[Tensor] = []  # num_groups x [batch, seq, dim]
    for group_idx in range(pool.num_groups):
        private_output = pool.private_routers[group_idx](
            norm_h, valid_mask & router.active_mask[:, :, group_idx]
        )
        group_update = torch.zeros_like(norm_h)
        for expert_idx, expert in enumerate(pool.private_experts[group_idx]):
            group_update = group_update + expert(norm_h) * private_output.gates[
                ..., expert_idx : expert_idx + 1
            ]
        per_column.append(
            group_update * router.gates[..., group_idx : group_idx + 1]
        )

    if pool.path_gate is not None:
        path_weights = torch.softmax(pool.path_gate(norm_h).float(), dim=-1).to(
            norm_h.dtype
        )
        public = public * path_weights[..., 0:1]
        per_column = [column * path_weights[..., 1:2] for column in per_column]
    return public, per_column


def _reference_pool_slot_updates(
    layer: torch.nn.Module,
    norm_h: Tensor,
    valid_mask: Tensor,
    router: object,
    topk_indices: Tensor,
) -> list[Tensor]:
    """Column-wise layout of :func:`_reference_pool_parts`, slot by slot.

    - the PUBLIC family is shared, so its update is added to EVERY slot;
    - the PRIVATE pool of attention group ``c`` is added ONLY to the slot the
      token holds column ``c`` in, via an explicit ``(batch, token, slot)``
      loop.

    Args:
        layer (torch.nn.Module): A ``RoutedTransformerLayer`` with a pool.
        norm_h (Tensor): Pre-normalized MLP input ``[batch, seq, dim]``.
        valid_mask (Tensor): Real-token mask ``[batch, seq]``.
        router (object): The block's group ``RouterOutput``.
        topk_indices (Tensor): Selected columns ``[batch, seq, top_k]``.

    Returns:
        list[Tensor]: ``top_k`` updates of shape ``[batch, seq, dim]``.
    """

    public, per_column = _reference_pool_parts(
        layer, norm_h, valid_mask, router
    )
    batch_size, seq_len, top_k = topk_indices.shape
    slot_updates = [public.clone() for _ in range(top_k)]
    for batch_idx in range(batch_size):
        for token_idx in range(seq_len):
            for slot in range(top_k):
                column = int(topk_indices[batch_idx, token_idx, slot])
                slot_updates[slot][batch_idx, token_idx] += per_column[column][
                    batch_idx, token_idx
                ]
    return slot_updates


def _randomize_column_queries(model: Nestor, seed: int) -> None:
    """Give every depth read a seeded non-degenerate query and value gains.

    Zero queries and unit gains make every depth read a uniform mean of
    unit-RMS sources, which would hide slot-weighting bugs; the fixed seed
    keeps two identically built models identical after randomization.

    Args:
        model (Nestor): Backbone with
            ``block_residual_mode='attn_res_column'``.
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


def _reference_column_backbone_forward(
    model: Nestor,
    x: Tensor,
    pos: Tensor,
    attn_mask: Tensor,
) -> Tensor:
    """Hand-rolled column-wise residual-stream recurrence.

    Threads the exported-source list and this block's ``k`` partial streams as
    ORDINARY PYTHON LISTS, with an explicit ``(batch, token, slot)`` loop for
    the column -> slot scatter, calling the model's OWN submodules (depth
    reads, norms, routers, attention groups, expert helpers) for every
    operation. Slow, but obviously faithful to the stated semantics:

    - sublayer input     = one shared depth read over
      ``[*sources, *streams]`` (streams omitted on a block's first layer,
      where they are still all-zero, exactly as ``forward_kimi`` omits its
      ``partial``),
    - attention          = column ``c``'s gated output lands in the slot the
      token holds ``c`` in, and nowhere else,
    - MLP                = one column-agnostic update broadcast to all slots,
    - block export       = the ``k`` streams, appended to the source list,
    - head input         = final depth read over every exported source.

    Assumes ``model.num_global == 0`` (enforced by the config checks), stride-1
    pooling and eval mode (no drop-path or router noise).

    Args:
        model (Nestor): Backbone built with
            ``block_residual_mode='attn_res_column'``.
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

        streams: list[Tensor] | None = None  # top_k x [batch, seq, dim]
        for layer in block.layers:
            read_sources = sources if streams is None else [*sources, *streams]
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
                    composed_mask
                    & members[:, :, None]
                    & members[:, None, :]
                )
                per_column.append(
                    group_attn(norm_h, group_mask, pos) * gate
                )

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

            h = layer.kimi_mlp_read([*sources, *streams])
            norm_h = layer.mlp_norm(h)
            if layer.expert_pool is not None:
                # Pooled MLP: the private experts are per attention column, so
                # each slot gets its own column's pool plus the shared public
                # part (see _reference_pool_slot_updates).
                streams = [
                    stream + update
                    for stream, update in zip(
                        streams,
                        _reference_pool_slot_updates(
                            layer, norm_h, valid_mask, router, topk_indices
                        ),
                    )
                ]
                continue
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
        # Only the next block's group router consumes the collapsed view.
        x = torch.stack(streams, dim=2).mean(dim=2)
        x = x * exit_weights.to(dtype=x.dtype)
        sources.extend(streams)
    return model.final_norm(model.final_depth_read(sources))


class ColumnStreamPlanTest:
    """Unit checks for the column -> stream-slot inversion."""

    def test_slot_map_inverts_topk_indices(self) -> None:
        """Every selected column maps to its slot; the rest to the sentinel."""

        topk_indices = torch.tensor(
            [[[2, 0], [1, 3], [3, 2]]]
        )  # [1, 3, 2]
        plan = _build_column_stream_plan(topk_indices, num_groups=4)

        assert plan.num_slots == 2
        expected = torch.tensor(
            [
                [
                    # token 0 picked columns (2, 0) -> slots (1, 0)
                    [1, 2, 0, 2],
                    # token 1 picked columns (1, 3) -> slots (0, 1)
                    [2, 0, 2, 1],
                    # token 2 picked columns (3, 2) -> slots (1, 0)
                    [2, 2, 1, 0],
                ]
            ]
        )
        assert torch.equal(plan.slot_of, expected)

    def test_sentinel_is_out_of_slot_range(self) -> None:
        """The sentinel must index the drop bin, i.e. equal ``num_slots``."""

        topk_indices = torch.zeros(2, 3, 1, dtype=torch.long)
        plan = _build_column_stream_plan(topk_indices, num_groups=5)

        assert plan.num_slots == 1
        assert int(plan.slot_of[:, :, 0].max()) == 0
        assert int(plan.slot_of[:, :, 1:].min()) == 1


class NestorColumnResidualConfigTest:
    """Config-level validation of the Nestor-specific mode."""

    def test_rejects_model_without_a_multi_column_block(self) -> None:
        """A model whose every block has k == 1 IS ``attn_res_kimi``."""

        config = _create_column_config(
            num_groups=[1, 1], num_groups_per_token=[1, 1]
        )
        with pytest.raises(ValueError, match="Nestor-specific") as error:
            Nestor(config)
        assert "attn_res_kimi" in str(error.value)

    def test_rejects_top1_routed_model(self) -> None:
        """Several columns but top-1 routing is still one stream per block."""

        config = _create_column_config(
            num_groups=[4, 1], num_groups_per_token=[1, 1]
        )
        with pytest.raises(ValueError, match="num_groups_per_token > 1"):
            Nestor(config)

    def test_rejects_varying_width(self) -> None:
        """Raw (projection-free) depth reads require one shared width."""

        config = _create_column_config()
        config.feature_dims = [16, 24]
        with pytest.raises(ValueError, match="requires constant width"):
            Nestor(config)

    def test_rejects_token_pooling(self) -> None:
        """Per-token depth mixing requires aligned token positions."""

        config = _create_column_config()
        config.pooling_strides = [2, 1]
        with pytest.raises(ValueError, match="requires constant token count"):
            Nestor(config)

    def test_rejects_global_tokens(self) -> None:
        """Global/register tokens are members of every column at once."""

        config = _create_column_config(causal=False)
        config.num_global_tokens = 2
        with pytest.raises(ValueError, match="num_global_tokens"):
            Nestor(config)

    def test_rejects_non_convex_residual_mode(self) -> None:
        """The per-sublayer depth reads replace the within-block residual."""

        config = _create_column_config()
        config.residual_mode = "attn_res"
        with pytest.raises(ValueError, match="requires residual_mode='convex'"):
            Nestor(config)

    def test_depth_source_count_is_one_plus_sum_of_top_k(self) -> None:
        """Every block exports ``k_b`` sources, plus the embedding b_0."""

        config = _create_column_config()
        model = Nestor(config)

        # Blocks are k=2 then k=1 -> 1 + 2 + 1 = 4 sources at the final read.
        assert model.max_depth_sources == 4
        assert model.final_depth_read.max_sources == 4


class NestorColumnResidualReferenceTest:
    """Bit-exact checks against the explicit-loop column recurrence."""

    def test_forward_matches_handrolled_recurrence_exactly(self) -> None:
        """Production forward must equal the reference loop at atol 0.

        Both sides run the SAME submodules in the same order and reduce with
        the same associativity (each slot receives exactly one attention
        contribution per layer, so the production ``scatter_add`` onto a zero
        buffer is a plain assignment), so float non-associativity cannot bite
        and exact equality is the right bar.
        """

        torch.manual_seed(20)
        config = _create_column_config()
        model = Nestor(config)
        torch.manual_seed(20)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_column_queries(candidate, seed=21)
            candidate.eval()
        # Block 0 must exercise the general (multi-column) path and block 1
        # the k == 1 degenerate one, so both branches are covered.
        assert not any(
            layer._dense_fastpath for layer in model.blocks[0].layers
        )
        assert all(layer._dense_fastpath for layer in model.blocks[1].layers)

        torch.manual_seed(22)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
            expected = _reference_column_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_token_dispatch_forward_matches_reference_exactly(self) -> None:
        """Token dispatch changes the attention kernel, not the semantics."""

        torch.manual_seed(23)
        config = _create_column_config()
        config.attention_token_dispatch = True
        model = Nestor(config)
        torch.manual_seed(23)
        reference_config = _create_column_config()
        reference_model = Nestor(reference_config)
        for candidate in (model, reference_model):
            _randomize_column_queries(candidate, seed=24)
            candidate.eval()

        torch.manual_seed(25)
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
            expected = _reference_column_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        # The gathered path compacts member tokens before the projections, so
        # the GEMMs see a different row order and reassociate in the last bit.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_pooled_forward_matches_handrolled_recurrence(self) -> None:
        """End-to-end check that private experts ride their column's stream.

        Same recurrence as above, but every layer's MLP is an expert pool, so
        the reference scatters each attention group's private pool into that
        group's slot and broadcasts only the public part.
        """

        config = _create_pooled_column_config(
            public_experts=2, private_experts=3
        )
        torch.manual_seed(29)
        model = Nestor(config)
        torch.manual_seed(29)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_column_queries(candidate, seed=30)
            candidate.eval()
        assert all(
            layer.expert_pool is not None
            for block in model.blocks
            for layer in block.layers
        )

        torch.manual_seed(31)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
            expected = _reference_column_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        # The reference evaluates every expert densely on all tokens and sums
        # over experts; production evaluates each expert on its selected rows
        # only (different GEMM shape, different summation order).
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_padded_forward_matches_reference(self) -> None:
        """Padding tokens must stay zero in every exported stream."""

        torch.manual_seed(26)
        config = _create_column_config()
        model = Nestor(config)
        torch.manual_seed(26)
        reference_model = Nestor(config)
        for candidate in (model, reference_model):
            _randomize_column_queries(candidate, seed=27)
            candidate.eval()

        torch.manual_seed(28)
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1).clone()
        pos[1, 3:] = PADDING_TOKEN_POS
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            actual = model(x, pos, attn_mask).pred
            expected = _reference_column_backbone_forward(
                reference_model, x, pos, attn_mask
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class NestorColumnFastPathEquivalenceTest:
    """The speed-stack attention paths must not change column semantics."""

    def test_grouped_projection_bmm_matches_per_group_loop(self) -> None:
        """NESTOR_GROUPED_ATTN_BMM batches the projections, not the scatter.

        The batched path replaces the final per-group ``scatter_add`` with one
        widened ``index_add`` over ``(token, slot)`` cells; both must produce
        the same streams. The env flags are the real ones the production
        modules read in ``__init__`` (no monkeypatching), restored afterwards.
        """

        import os

        config = _create_column_config()
        config.attention_token_dispatch = True
        previous = os.environ.get("NESTOR_GROUPED_ATTN_BMM")
        try:
            os.environ["NESTOR_GROUPED_ATTN_BMM"] = "1"
            torch.manual_seed(50)
            batched = Nestor(config)
            os.environ.pop("NESTOR_GROUPED_ATTN_BMM")
            torch.manual_seed(50)
            looped = Nestor(config)
        finally:
            if previous is None:
                os.environ.pop("NESTOR_GROUPED_ATTN_BMM", None)
            else:
                os.environ["NESTOR_GROUPED_ATTN_BMM"] = previous
        assert all(layer._grouped_proj_bmm for layer in batched.blocks[0].layers)
        assert not any(
            layer._grouped_proj_bmm for layer in looped.blocks[0].layers
        )
        for candidate in (batched, looped):
            _randomize_column_queries(candidate, seed=51)
            candidate.eval()

        torch.manual_seed(52)
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

        with torch.no_grad():
            actual = batched(x, pos, attn_mask).pred
            expected = looped(x, pos, attn_mask).pred
        # Stacking the per-group weights into one bmm reassociates the GEMM.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_top1_batched_dispatch_block_matches_masked_path(self) -> None:
        """A k == 1 grouped block may take the top-1 batched dispatch plan.

        ``NESTOR_DENSE_FULL_ATTN=1`` + token dispatch + no padding routes such
        a block through ``_batched_dispatch_attention``, whose output is a
        single column per token and therefore a single stream slot.
        """

        import os

        config = _create_column_config(
            causal=False,
            num_groups=[4, 4],
            num_groups_per_token=[1, 2],
        )
        config.attention_token_dispatch = True
        previous = os.environ.get("NESTOR_DENSE_FULL_ATTN")
        try:
            os.environ["NESTOR_DENSE_FULL_ATTN"] = "1"
            torch.manual_seed(53)
            batched = Nestor(config)
            os.environ.pop("NESTOR_DENSE_FULL_ATTN")
            torch.manual_seed(53)
            masked = Nestor(config)
        finally:
            if previous is None:
                os.environ.pop("NESTOR_DENSE_FULL_ATTN", None)
            else:
                os.environ["NESTOR_DENSE_FULL_ATTN"] = previous
        assert batched.blocks[0]._attn_dispatch_plan
        assert not masked.blocks[0]._attn_dispatch_plan
        for candidate in (batched, masked):
            _randomize_column_queries(candidate, seed=54)
            candidate.eval()

        torch.manual_seed(55)
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

        with torch.no_grad():
            actual = batched(x, pos, attn_mask).pred
            expected = masked(x, pos, attn_mask).pred
        assert torch.isfinite(actual).all()
        # Same no-padding equivalence contract as the non-column paths.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


class NestorColumnStreamIsolationTest:
    """Each column must write into its own stream and no other."""

    def test_zeroing_one_column_zeroes_exactly_its_slots(self) -> None:
        """A dead column leaves exactly the slots it owns at their init."""

        torch.manual_seed(30)
        config = _create_column_config(block_num_layers=[1, 1])
        model = Nestor(config).eval()
        block = model.blocks[0]
        layer = block.layers[0]
        dead_column = 2

        torch.manual_seed(31)
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1)
        valid_mask = pos != PADDING_TOKEN_POS
        attn_mask = block._compose_attn_mask(
            valid_mask, torch.ones(2, 6, 6, dtype=torch.bool)
        )

        with torch.no_grad():
            router = block.group_router(x, valid_mask)
            plan = _build_column_stream_plan(
                router.topk_indices, block.num_groups
            )
            norm_x = layer.attn_norm(x)
            live = layer._attn_update_general(
                norm_x, attn_mask, router, pos, column_plan=plan
            )
            # Sanity: with every column alive no slot may still be at its
            # zero init, otherwise the assertion below would be vacuous.
            assert bool((live.abs().sum(dim=-1) > 0).all())

            layer.attention_groups[dead_column].out_proj.weight.zero_()
            if layer.attention_groups[dead_column].out_proj.bias is not None:
                layer.attention_groups[dead_column].out_proj.bias.zero_()
            dead = layer._attn_update_general(
                norm_x, attn_mask, router, pos, column_plan=plan
            )

        owned_by_dead = router.topk_indices == dead_column  # [batch, seq, k]
        untouched = dead.abs().sum(dim=-1) == 0  # [batch, seq, k]
        assert torch.equal(untouched, owned_by_dead)
        # Slots owned by live columns must be bit-identical to before.
        torch.testing.assert_close(
            dead[~owned_by_dead], live[~owned_by_dead], rtol=0.0, atol=0.0
        )


class NestorColumnResidualIntegrationTest:
    """End-to-end wiring, gradients and unsupported-path checks."""

    def test_forward_shape_and_finiteness(self) -> None:
        """The backbone still emits one token-shaped tensor for the head."""

        torch.manual_seed(40)
        config = _create_column_config()
        model = Nestor(config).eval()
        x = torch.randn(2, 5, config.input_dim)
        pos = torch.arange(5).expand(2, -1)
        attn_mask = torch.ones(2, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            output = model(x, pos, attn_mask)
        assert output.pred.shape == (2, 5, config.output_dim)
        assert torch.isfinite(output.pred).all()

    def test_backward_reaches_reads_columns_and_mlps(self) -> None:
        """Grads must reach every depth read, column and expert MLP."""

        torch.manual_seed(41)
        config = _create_column_config()
        model = Nestor(config)
        _randomize_column_queries(model, seed=42)
        model.eval()
        x = torch.randn(2, 6, config.input_dim)
        pos = torch.arange(6).expand(2, -1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

        output = model(x, pos, attn_mask)
        output.pred.square().mean().backward()

        assert model.final_depth_read.query.grad is not None
        assert torch.count_nonzero(model.final_depth_read.query.grad) > 0
        for block_idx, block in enumerate(model.blocks):
            # A column no token routed to is dead compute (the per-group loop
            # skips it), so only populated columns can carry gradient.
            populated = output.aux["blocks"][block_idx]["group_router"][
                "active_mask"
            ].any(dim=0).any(dim=0)  # [num_groups]
            assert bool(populated.any())
            for layer_idx, layer in enumerate(block.layers):
                for read_idx, read in enumerate(
                    (layer.kimi_attn_read, layer.kimi_mlp_read)
                ):
                    assert read.query.grad is not None
                    if block_idx == 0 and layer_idx == 0 and read_idx == 0:
                        # The very first attention read sees ONE source (the
                        # embedding): a one-source softmax is the constant 1,
                        # so its query legitimately carries zero gradient.
                        continue
                    assert torch.count_nonzero(read.query.grad) > 0
                scored = 0  # columns whose query/key path carried gradient
                for group_idx, group_attn in enumerate(
                    layer.attention_groups
                ):
                    for projection in (
                        group_attn.q_proj,
                        group_attn.k_proj,
                        group_attn.v_proj,
                        group_attn.out_proj,
                    ):
                        assert projection.weight.grad is not None
                    if not bool(populated[group_idx]):
                        continue
                    # The value/output path is always live for a populated
                    # column; the query/key path is not, because a column with
                    # a single member per sequence softmaxes over one key and
                    # its attention weights are constant 1 by construction.
                    assert torch.count_nonzero(group_attn.v_proj.weight.grad) > 0
                    assert (
                        torch.count_nonzero(group_attn.out_proj.weight.grad) > 0
                    )
                    scored += int(
                        torch.count_nonzero(group_attn.q_proj.weight.grad) > 0
                        and torch.count_nonzero(group_attn.k_proj.weight.grad)
                        > 0
                    )
                assert scored > 0
                for expert in layer.shared_experts:
                    grads = [
                        parameter.grad
                        for parameter in expert.parameters()
                        if parameter.requires_grad
                    ]
                    assert grads and all(
                        grad is not None and torch.count_nonzero(grad) > 0
                        for grad in grads
                    )

    def test_layers_stay_stateless_across_forwards(self) -> None:
        """Functional threading: a second forward must be identical."""

        torch.manual_seed(43)
        config = _create_column_config()
        model = Nestor(config).eval()
        _randomize_column_queries(model, seed=44)
        x = torch.randn(2, 4, config.input_dim)
        pos = torch.arange(4).expand(2, -1)
        attn_mask = torch.ones(2, 4, 4, dtype=torch.bool)

        with torch.no_grad():
            first = model(x, pos, attn_mask).pred
            second = model(x, pos, attn_mask).pred
        assert torch.equal(first, second)
        for block in model.blocks:
            for layer in block.layers:
                assert not hasattr(layer, "_column_streams")

    def test_forward_step_raises(self) -> None:
        """Incremental KV-cache decode is explicitly unsupported."""

        torch.manual_seed(45)
        config = _create_column_config(causal=True)
        model = Nestor(config).eval()
        x = torch.randn(2, 3, config.input_dim)
        pos = torch.arange(3).expand(2, -1)
        cache = model.init_kv_cache(batch_size=2)

        with pytest.raises(NotImplementedError, match="attn_res_column"):
            model.forward_step(x[:, 0], pos[:, 0], cache)

    def test_config_round_trips(self) -> None:
        """``to_config`` must round-trip the mode back into a buildable model."""

        torch.manual_seed(47)
        model = Nestor(_create_column_config())
        rebuilt = Nestor(model.to_config())

        assert model.to_config().block_residual_mode == "attn_res_column"
        assert rebuilt.max_depth_sources == model.max_depth_sources
        assert sum(p.numel() for p in rebuilt.parameters()) == sum(
            p.numel() for p in model.parameters()
        )

    def test_reference_mup_transform_composes(self) -> None:
        """The reference muP transform must apply cleanly and train."""

        torch.manual_seed(46)
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(embed_dim=16, vocab_size=64),
            backbone_config=_create_column_config(causal=True),
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


class NestorColumnCompileTest:
    """torch.compile smoke test for the functional column threading."""

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

        torch.manual_seed(60)
        config = _create_column_config()
        model = Nestor(config)
        _randomize_column_queries(model, seed=61)
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
