"""Expert-pool behavior under column-wise residual streams.

Companion to ``tests/test_column_residual.py`` (which owns the pool-free
semantics and the hand-rolled backbone recurrence, including its pooled
end-to-end case). This file covers the sublayer contract that
``block_residual_mode='attn_res_column'`` adds on top of an
:class:`~nestor.backbones.expert_pools.ExpertPoolMLP`: each attention group's
PRIVATE experts write into that group's own residual stream, while the shared
public pool broadcasts to all of them -- and every other residual mode keeps
calling ``ExpertPoolMLP.forward`` unchanged.

The pool reference implementations are imported from the companion module (the
one exception the repo's self-contained-test rule allows).
"""

from __future__ import annotations

import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.nestor import _build_column_stream_plan
from nestor.constants import PADDING_TOKEN_POS
from tests.test_column_residual import (
    _create_column_config,
    _create_pooled_column_config,
    _reference_pool_parts,
    _reference_pool_slot_updates,
)


class NestorColumnPooledExpertTest:
    """Private expert pools must follow their attention column's stream."""

    def _pooled_layer_inputs(
        self, config: NestorConfig, seed: int, seq_len: int = 6
    ) -> tuple[Nestor, Tensor, Tensor, Tensor, object, object]:
        """Build a pooled model and everything block 0's layer 0 MLP needs.

        Args:
            config (NestorConfig): Pooled column config to build.
            seed (int): Seed for both the model and the input draw.
            seq_len (int): Sequence length of the probe batch.

        Returns:
            tuple[Nestor, Tensor, Tensor, Tensor, object, object]: The model,
            its first routed layer, the pre-normalized MLP input
            ``[batch, seq, dim]``, the valid-token mask, the block's group
            router output, and the column plan.
        """

        torch.manual_seed(seed)
        model = Nestor(config).eval()
        block = model.blocks[0]
        layer = block.layers[0]

        torch.manual_seed(seed + 1)
        x = torch.randn(2, seq_len, config.input_dim)
        pos = torch.arange(seq_len).expand(2, -1)
        valid_mask = pos != PADDING_TOKEN_POS
        router = block.group_router(x, valid_mask)
        plan = _build_column_stream_plan(router.topk_indices, block.num_groups)
        return model, layer, layer.mlp_norm(x), valid_mask, router, plan

    def test_zeroing_one_private_bank_zeroes_exactly_its_slots(self) -> None:
        """A dead private pool changes exactly the slots its column owns.

        The MLP analogue of
        :meth:`NestorColumnStreamIsolationTest.
        test_zeroing_one_column_zeroes_exactly_its_slots`: killing attention
        group ``c``'s private experts must leave every slot NOT owning column
        ``c`` bit-identical, and must move every slot that does own it.
        """

        _model, layer, norm_x, valid_mask, router, plan = (
            self._pooled_layer_inputs(
                _create_pooled_column_config(public_experts=0), seed=70
            )
        )
        dead_column = 2

        with torch.no_grad():
            live, _ = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )
            for expert in layer.expert_pool.private_experts[dead_column]:
                expert.layers[-1].weight.zero_()
            dead, _ = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )

        owns_dead = (router.topk_indices == dead_column)  # [batch, seq, k]
        assert bool(owns_dead.any()), "no token routed to the dead column"
        # Private-only pool: a slot's whole update comes from its column.
        assert bool((live.abs().sum(dim=-1) > 0).all())
        assert bool((dead[owns_dead].abs().sum(dim=-1) == 0).all())
        torch.testing.assert_close(
            dead[~owns_dead], live[~owns_dead], rtol=0.0, atol=0.0
        )

    def test_public_part_is_identical_across_every_slot(self) -> None:
        """The public pool is shared, so it must broadcast unchanged.

        With every private bank dead, all ``k`` slots must hold exactly the
        same (path-gated) public update.
        """

        _model, layer, norm_x, valid_mask, router, plan = (
            self._pooled_layer_inputs(
                _create_pooled_column_config(public_experts=2), seed=71
            )
        )

        with torch.no_grad():
            for bank in layer.expert_pool.private_experts:
                for expert in bank:
                    expert.layers[-1].weight.zero_()
            update, _ = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )

        assert update.shape[2] == plan.num_slots
        assert bool((update.abs().sum(dim=-1) > 0).all())
        for slot in range(1, plan.num_slots):
            torch.testing.assert_close(
                update[:, :, slot], update[:, :, 0], rtol=0.0, atol=0.0
            )

    def test_no_pool_column_update_broadcasts_the_shared_mlp(self) -> None:
        """Without a pool the MLP stays column-agnostic (unchanged behavior)."""

        _model, layer, norm_x, valid_mask, router, plan = (
            self._pooled_layer_inputs(_create_column_config(), seed=72)
        )
        assert layer.expert_pool is None

        with torch.no_grad():
            column_update, column_aux = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )
            shared_update, shared_aux = layer._expert_update(
                norm_x, valid_mask, router, dense=False
            )

        # A length-1 slot axis: it broadcasts onto all k streams, exactly as
        # the pre-pool implementation's explicit ``unsqueeze(2)`` did.
        assert column_update.shape == (*shared_update.shape[:2], 1, 16)
        torch.testing.assert_close(
            column_update[:, :, 0], shared_update, rtol=0.0, atol=0.0
        )
        assert column_aux.keys() == shared_aux.keys()

    def test_column_update_matches_reference_loop(self) -> None:
        """The pooled column update must equal the hand-rolled scatter.

        Reference tolerance rather than atol 0: the reference evaluates every
        expert densely on all tokens and sums over experts, while production
        evaluates each expert on its selected rows only -- a different GEMM
        shape and a different summation order, so the last bits legitimately
        differ.
        """

        _model, layer, norm_x, valid_mask, router, plan = (
            self._pooled_layer_inputs(
                _create_pooled_column_config(
                    public_experts=2, private_experts=3
                ),
                seed=73,
            )
        )

        with torch.no_grad():
            actual, _ = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )
            expected = torch.stack(
                _reference_pool_slot_updates(
                    layer, norm_x, valid_mask, router, router.topk_indices
                ),
                dim=2,
            )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_grouped_dispatch_matches_sparse_per_group(self) -> None:
        """Both pool dispatch paths must produce the same per-column split."""

        _model, layer, norm_x, valid_mask, router, plan = (
            self._pooled_layer_inputs(
                _create_pooled_column_config(
                    public_experts=2, private_experts=3
                ),
                seed=74,
            )
        )
        pool = layer.expert_pool

        with torch.no_grad():
            pool.grouped_dispatch = False
            sparse, _ = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )
            pool.grouped_dispatch = True
            grouped, _ = layer._expert_update_column(
                norm_x, valid_mask, router, plan, dense=False
            )
        # Grouped GEMMs reassociate the dispatched rows (documented in
        # ExpertPoolMLP.forward).
        torch.testing.assert_close(grouped, sparse, rtol=1e-6, atol=1e-6)

    def test_forward_per_group_sums_to_pool_forward(self) -> None:
        """The split must be a pure re-layout of :meth:`forward`'s update.

        This is the non-regression guard for every OTHER residual mode:
        ``attn_res_kimi`` and ``projection`` call ``ExpertPoolMLP.forward``,
        so as long as summing the per-group split reproduces it, the split
        added no work and changed no semantics.
        """

        for public_experts in (0, 2):
            _model, layer, norm_x, valid_mask, router, _ = (
                self._pooled_layer_inputs(
                    _create_pooled_column_config(
                        public_experts=public_experts, private_experts=3
                    ),
                    seed=75 + public_experts,
                )
            )
            pool = layer.expert_pool
            for grouped in (False, True):
                pool.grouped_dispatch = grouped
                with torch.no_grad():
                    merged, public_output, private_outputs = pool(
                        norm_x, valid_mask, router.gates, router.active_mask
                    )
                    (
                        split_public,
                        split_private,
                        split_public_output,
                        split_private_outputs,
                    ) = pool.forward_per_group(
                        norm_x, valid_mask, router.gates, router.active_mask
                    )
                total = split_private.sum(dim=2)
                if split_public is not None:
                    total = total + split_public
                # Only the reduction order differs (sum_g sum_e vs one flat
                # sum over the (g, e) slots).
                torch.testing.assert_close(
                    total, merged, rtol=1e-6, atol=1e-6
                )
                assert (public_output is None) == (
                    split_public_output is None
                )
                assert (split_public is None) == (public_output is None)
                assert len(split_private_outputs) == len(private_outputs)

    def test_kimi_mode_pool_output_matches_reference_equation(self) -> None:
        """Pooled ``attn_res_kimi`` still implements the documented equation.

        ``forward_kimi`` reaches the pool through ``ExpertPoolMLP.forward``,
        the code path the per-column split must NOT have touched; this checks
        it against the module docstring's equation written out by hand.
        """

        config = _create_pooled_column_config(
            public_experts=2,
            private_experts=3,
            block_residual_mode="attn_res_kimi",
        )
        model, layer, norm_x, valid_mask, router, _ = (
            self._pooled_layer_inputs(config, seed=77)
        )
        assert model.block_residual_mode == "attn_res_kimi"

        with torch.no_grad():
            actual, aux = layer._expert_update(
                norm_x, valid_mask, router, dense=False
            )
            public, per_column = _reference_pool_parts(
                layer, norm_x, valid_mask, router
            )
        expected = public + torch.stack(per_column, dim=0).sum(dim=0)
        assert "expert_router" in aux and "expert_pool_private" in aux
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

