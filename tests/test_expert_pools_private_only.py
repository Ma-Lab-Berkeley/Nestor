"""Tests for the PRIVATE-ONLY expert-pool MLP (``num_public_experts == 0``).

Sibling of ``test_expert_pools.py`` (which covers the canonical
public+private ``p4k1_q4k1`` design). Here each attention column draws its
experts only from its own private pool: no shared/public pool, no 2-logit path
gate, and the private experts widen to the whole ``4d`` active budget. Covers
the width contract, a slow per-token reference of the private-only equation,
gradient flow / DDP-safety (no parameter may be left without a grad), the
2-block backbone integration, and composition with the reference-muP
transform.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from nestor.backbones import Nestor
from nestor.backbones.expert_pools import (
    ExpertPoolMLP,
    private_expert_hidden_dim,
)
from nestor.backbones.nestor import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig
from nestor.scaling.blocks import get_scaling

#: Tolerance for grouped-GEMM vs sparse/reference comparisons: the grouped
#: path reassociates the same fp32 arithmetic (one padded matmul per expert
#: bank instead of one per expert, slot sums instead of ``index_add_``), so
#: agreement is to a few ULPs rather than bitwise.
_GROUPED_RTOL = 1e-5
_GROUPED_ATOL = 1e-6


def _make_private_pool(
    *,
    dim: int = 16,
    num_groups: int = 4,
    num_groups_per_token: int = 2,
    num_private_experts: int = 3,
    private_top_k: int = 1,
    seed: int = 0,
) -> ExpertPoolMLP:
    """Build a small seeded private-only pool with randomized router biases.

    The loss-free choice biases start at zero; they are randomized here so the
    reference comparison exercises the biased top-k selection.

    Args:
        dim (int): Token feature width.
        num_groups (int): Attention groups (one private pool each).
        num_groups_per_token (int): Active attention groups per token.
        num_private_experts (int): Private pool size per group.
        private_top_k (int): Private experts per selected group.
        seed (int): ``torch.manual_seed`` value.

    Returns:
        ExpertPoolMLP: The randomized private-only pool module.
    """

    torch.manual_seed(seed)
    pool = ExpertPoolMLP(
        dim=dim,
        num_groups=num_groups,
        num_groups_per_token=num_groups_per_token,
        num_public_experts=0,
        public_top_k=1,
        num_private_experts=num_private_experts,
        private_top_k=private_top_k,
        router_score_mlp_hidden_dims=[8],
        router_score_mlp_activation="gelu",
        router_noise_mlp_hidden_dims=[8],
        router_noise_mlp_activation="gelu",
        loss_free_balancing=True,
        bias_update_rate=1e-3,
    )
    with torch.no_grad():
        for router in pool.private_routers:
            router.choice_bias.normal_(std=0.1)
    return pool


def _make_group_routing(
    batch_size: int,
    seq_len: int,
    num_groups: int,
    num_groups_per_token: int,
    *,
    seed: int = 1,
    dead_groups: tuple[int, ...] = (),
) -> tuple[
    Tensor,  # gates: (batch_size, seq_len, num_groups)
    Tensor,  # active_mask: (batch_size, seq_len, num_groups) bool
]:
    """Fabricate a top-k attention-group routing (normalized gates + mask).

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        num_groups (int): Total attention groups.
        num_groups_per_token (int): Active groups per token.
        seed (int): ``torch.manual_seed`` value.
        dead_groups (tuple[int, ...]): Groups no token may select (their scores
            are driven to ``-inf``), so their private pools see ZERO member
            tokens.

    Returns:
        tuple[Tensor, Tensor]: Gates ``(batch_size, seq_len, num_groups)``
        (softmax over each token's top-k groups, zero elsewhere) and the
        boolean membership mask of the same shape.
    """

    torch.manual_seed(seed)
    scores = torch.randn(batch_size, seq_len, num_groups)
    for group_idx in dead_groups:
        scores[:, :, group_idx] = -float("inf")
    topk = scores.topk(num_groups_per_token, dim=-1).indices
    gates = torch.zeros_like(scores).scatter(
        -1, topk, torch.softmax(scores.gather(-1, topk), dim=-1)
    )
    return gates, gates > 0


def reference_private_only_pool_forward(
    pool: ExpertPoolMLP,
    x: Tensor,  # (batch_size, seq_len, dim)
    valid_token_mask: Tensor,  # (batch_size, seq_len) bool
    group_gates: Tensor,  # (batch_size, seq_len, num_groups)
    group_active_mask: Tensor,  # (batch_size, seq_len, num_groups) bool
) -> Tensor:
    """Slow per-token reference of the private-only equation (eval mode).

    Explicit python loops over tokens, groups and experts, implementing::

        MLP(h) = sum_g attn_gate_g * sum_e private_gate_(g,e)
                                     * PrivateExpert_(g,e)(h)

    with NO path-gate factor (there is only one path). Routing math is
    re-derived from each router's score MLP and choice bias (top-k on the
    biased scores, softmax gates over the raw selected scores), matching
    ``Router.forward`` in eval mode.

    Args:
        pool (ExpertPoolMLP): Private-only pool module in eval mode.
        x (Tensor): Pre-normalized inputs ``(batch_size, seq_len, dim)``.
        valid_token_mask (Tensor): ``(batch_size, seq_len)`` bool.
        group_gates (Tensor): ``(batch_size, seq_len, num_groups)``.
        group_active_mask (Tensor): ``(batch_size, seq_len, num_groups)``.

    Returns:
        Tensor: The pooled MLP update ``(batch_size, seq_len, dim)``.
    """

    def routed_sum(router, experts, h: Tensor) -> Tensor:
        scores = router.score_mlp(h)
        biased = scores + router.choice_bias
        topk = torch.topk(biased, k=router.num_choices_per_token).indices
        gates = torch.softmax(scores[topk].float(), dim=-1).to(scores.dtype)
        out = torch.zeros_like(h)
        for slot, expert_idx in enumerate(topk.tolist()):
            out = out + gates[slot] * experts[expert_idx](h)
        return out

    batch_size, seq_len, _ = x.shape
    out = torch.zeros_like(x)
    for b in range(batch_size):
        for s in range(seq_len):
            h = x[b, s]
            if not valid_token_mask[b, s]:
                continue
            private = torch.zeros_like(h)
            for g in range(pool.num_groups):
                if not group_active_mask[b, s, g]:
                    continue
                private = private + group_gates[b, s, g] * routed_sum(
                    pool.private_routers[g], pool.private_experts[g], h
                )
            out[b, s] = private
    return out


class PrivateOnlyExpertPoolWidthsTest:
    """The private-only width contract ``m_private = round8(4d/(k_a*k_q))``."""

    def test_ladder_width_table_at_d640(self) -> None:
        """d=640: k_a of 3/5/7/1 -> 856/512/368/2560, each active ~4d=2560."""

        widths = {
            top_k: private_expert_hidden_dim(
                640, top_k, 1, with_public_pool=False
            )
            for top_k in (3, 5, 7, 1)
        }
        assert widths == {3: 856, 5: 512, 7: 368, 1: 2560}
        # Active intermediate = k_a * k_q * m_private ~= 4d (rounding only).
        actives = {top_k: top_k * width for top_k, width in widths.items()}
        assert actives == {3: 2568, 5: 2560, 7: 2576, 1: 2560}
        assert all(abs(active - 2560) <= 16 for active in actives.values())

    def test_private_top_k_splits_the_same_budget(self) -> None:
        """``k_q > 1`` divides the width; the active total still tracks 4d."""

        assert private_expert_hidden_dim(640, 3, 2, with_public_pool=False) == 424
        assert abs(3 * 2 * 424 - 2560) <= 24
        assert private_expert_hidden_dim(1280, 3, 1, with_public_pool=False) == 1704

    def test_both_pools_contract_is_unchanged(self) -> None:
        """The default (``with_public_pool=True``) still halves the budget."""

        assert private_expert_hidden_dim(640, 3, 1) == 424
        assert private_expert_hidden_dim(640, 3, 1, with_public_pool=True) == 424
        assert private_expert_hidden_dim(960, 3, 1) == 640
        assert private_expert_hidden_dim(1280, 3, 1) == 856
        # Exactly the 4d/2d ratio, up to the multiple-of-8 rounding.
        assert private_expert_hidden_dim(640, 5, 1, with_public_pool=False) == 512
        assert private_expert_hidden_dim(640, 5, 1) == 256


class PrivateOnlyExpertPoolStructureTest:
    """No public modules and no path gate may exist (DDP safety)."""

    def test_no_public_modules_and_no_path_gate(self) -> None:
        """Public router/experts/path gate are ``None``, privates are built."""

        pool = _make_private_pool(seed=40)
        assert pool.has_public_pool is False
        assert pool.public_router is None
        assert pool.public_experts is None
        assert pool.path_gate is None
        assert pool.num_public_experts == 0
        assert pool.public_hidden_dim == 0
        assert len(pool.private_routers) == pool.num_groups
        assert len(pool.private_experts) == pool.num_groups
        assert all(
            len(bank) == pool.num_private_experts
            for bank in pool.private_experts
        )
        # 4d contract: dim 16, k_a 2, k_q 1 -> round8(64/2) = 32.
        assert pool.private_hidden_dim == 32
        names = [name for name, _ in pool.named_parameters()]
        assert names, "the private-only pool must own parameters"
        assert not [
            name
            for name in names
            if "public" in name or "path_gate" in name
        ]

    def test_public_only_pool_is_rejected(self) -> None:
        """A public pool without a private one is not implemented."""

        try:
            ExpertPoolMLP(
                dim=16,
                num_groups=4,
                num_groups_per_token=2,
                num_public_experts=3,
                public_top_k=1,
                num_private_experts=0,
                private_top_k=1,
                router_score_mlp_hidden_dims=[8],
                router_score_mlp_activation="gelu",
                router_noise_mlp_hidden_dims=[8],
                router_noise_mlp_activation="gelu",
            )
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for a public-only pool")


class PrivateOnlyExpertPoolReferenceTest:
    """Module forward vs the slow per-token private-only reference.

    Both dispatch paths (grouped GEMM and the per-expert sparse loop) are
    checked against the reference, so each is validated against the
    mathematical definition rather than only against the other.
    """

    def test_matches_reference_top1(self) -> None:
        """Canonical ``q*k1`` geometry with padding tokens matches exactly."""

        for grouped in (False, True):
            pool = _make_private_pool(seed=41)
            pool.eval()
            pool.grouped_dispatch = grouped
            torch.manual_seed(42)
            x = torch.randn(2, 7, 16)
            valid = torch.ones(2, 7, dtype=torch.bool)
            valid[1, -2:] = False  # exercise padding
            gates, active = _make_group_routing(2, 7, 4, 2, seed=43)
            gates = gates * valid.unsqueeze(-1).to(gates.dtype)
            active = active & valid.unsqueeze(-1)

            with torch.no_grad():
                out, public_output, private_outputs = pool(
                    x, valid, gates, active
                )
                ref = reference_private_only_pool_forward(
                    pool, x, valid, gates, active
                )
            torch.testing.assert_close(
                out, ref, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
            )
            assert public_output is None
            assert len(private_outputs) == pool.num_groups
            for g, private_output in enumerate(private_outputs):
                assert torch.equal(
                    private_output.valid_token_mask, valid & active[:, :, g]
                )

    def test_matches_reference_top2_with_dead_group(self) -> None:
        """Top-2 private routing, ``k_a=3``, and a zero-member group match."""

        for grouped in (False, True):
            pool = _make_private_pool(
                num_groups=4,
                num_groups_per_token=3,
                num_private_experts=4,
                private_top_k=2,
                seed=44,
            )
            pool.eval()
            pool.grouped_dispatch = grouped
            torch.manual_seed(45)
            x = torch.randn(2, 9, 16)
            valid = torch.ones(2, 9, dtype=torch.bool)
            valid[0, -3:] = False
            gates, active = _make_group_routing(
                2, 9, 4, 3, seed=46, dead_groups=(3,)
            )
            gates = gates * valid.unsqueeze(-1).to(gates.dtype)
            active = active & valid.unsqueeze(-1)
            assert not active[:, :, 3].any()

            with torch.no_grad():
                out, _, _ = pool(x, valid, gates, active)
                ref = reference_private_only_pool_forward(
                    pool, x, valid, gates, active
                )
            torch.testing.assert_close(
                out, ref, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
            )

    def test_grouped_matches_sparse_under_bf16_autocast(self) -> None:
        """bf16 autocast with an fp32 residual stream agrees on both paths."""

        pool = _make_private_pool(seed=47)
        pool.eval()
        torch.manual_seed(48)
        x = torch.randn(2, 6, 16)  # fp32 residual stream
        valid = torch.ones(2, 6, dtype=torch.bool)
        gates, active = _make_group_routing(2, 6, 4, 2, seed=49)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                pool.grouped_dispatch = False
                sparse, _, _ = pool(x, valid, gates, active)
                pool.grouped_dispatch = True
                grouped, _, _ = pool(x, valid, gates, active)
        assert grouped.dtype == x.dtype
        torch.testing.assert_close(
            grouped, sparse, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
        )


class PrivateOnlyExpertPoolGradientsTest:
    """Gradient flow / DDP-safety of the private-only parameters."""

    def test_every_parameter_receives_a_gradient(self) -> None:
        """No parameter is left without a grad -- on BOTH dispatch paths.

        DDP's reducer expects every trainable parameter to be touched by the
        backward, so an unused path gate (or an expert nothing selects) would
        hang training. The hard cases are covered: two attention groups with
        NO member token, and choice biases pinning every router to expert 0.
        """

        for grouped in (False, True):
            pool = _make_private_pool(seed=50)
            pool.train()
            pool.grouped_dispatch = grouped
            with torch.no_grad():
                for router in pool.private_routers:
                    router.choice_bias.zero_()
                    router.choice_bias[0] = 1e3
            torch.manual_seed(51)
            x = torch.randn(2, 8, 16, requires_grad=True)
            valid = torch.ones(2, 8, dtype=torch.bool)
            valid[1, -2:] = False
            gates, active = _make_group_routing(
                2, 8, 4, 2, seed=52, dead_groups=(2, 3)
            )
            gates = gates * valid.unsqueeze(-1).to(gates.dtype)
            active = active & valid.unsqueeze(-1)
            assert not active[:, :, 2].any() and not active[:, :, 3].any()

            out, _, _ = pool(x, valid, gates, active)
            out.square().sum().backward()
            for name, parameter in pool.named_parameters():
                assert parameter.grad is not None, f"no grad for {name}"
                assert torch.isfinite(parameter.grad).all(), f"bad grad {name}"
            assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_router_and_expert_gradients_nonzero_with_top2(self) -> None:
        """With top-2 private gates every router and expert gets real signal.

        (With top-1 the renormalized gate is the constant 1.0 -- nestor's
        native router semantics -- so router score heads receive exactly-zero
        gradient and learn through noise + loss-free balancing only.)
        """

        pool = _make_private_pool(
            num_private_experts=4, private_top_k=2, seed=53
        )
        pool.train()
        torch.manual_seed(54)
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 16)
        valid = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Round-robin membership so every private pool has member tokens.
        active = torch.zeros(
            batch_size, seq_len, pool.num_groups, dtype=torch.bool
        )
        for s in range(seq_len):
            active[:, s, s % pool.num_groups] = True
            active[:, s, (s + 1) % pool.num_groups] = True
        gates = active.to(x.dtype) / 2.0

        out, _, _ = pool(x, valid, gates, active)
        out.square().sum().backward()
        for g, router in enumerate(pool.private_routers):
            score = router.score_mlp.layers[-1].weight
            assert score.grad is not None, f"no grad for private router {g}"
            assert score.grad.abs().sum() > 0, f"zero grad, private router {g}"
        for g, bank in enumerate(pool.private_experts):
            for e, expert in enumerate(bank):
                for linear in (expert.layers[0], expert.layers[1], expert.gate):
                    assert linear.weight.grad is not None, f"no grad {g}/{e}"
                    assert linear.weight.grad.abs().sum() > 0, (
                        f"zero grad for private expert {g}/{e}"
                    )


def _private_only_config() -> NestorConfig:
    """A tiny 2-block grouped-prefix config with PRIVATE-ONLY pools.

    Block 0: 4 attention groups top-2 with token dispatch, loss-free
    balancing, and a 2-expert private pool per group (no public pool). Block 1:
    dense, unpooled. Between-block transition is ``attn_res_kimi``.

    Returns:
        NestorConfig: The tiny causal config.
    """

    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[32, 32],
        causal=True,
        pooling_strides=[1, 1],
        block_num_layers=[2, 2],
        num_groups=[4, 1],
        num_groups_per_token=[2, 1],
        block_num_heads=2,
        block_head_dims=16,
        attention_token_dispatch=True,
        block_mlp_hidden_dims=[[24], [24]],
        block_mlp_activations="swiglu",
        block_residual_mode="attn_res_kimi",
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_loss_free_balancing=True,
        router_bias_update_rate=1e-3,
        router_score_mlp_hidden_dims=[[8], [8]],
        router_noise_mlp_hidden_dims=[[8], [8]],
        block_expert_pool_public_experts=[0, 0],
        block_expert_pool_public_top_k=[1, 1],
        block_expert_pool_private_experts=[2, 0],
        block_expert_pool_private_top_k=[1, 1],
    )


class PrivateOnlyExpertPoolIntegrationTest:
    """Private-only pooled prefix inside the full Nestor backbone."""

    def test_forward_backward_and_determinism(self) -> None:
        """Forward+backward is finite, DDP-safe, and eval-deterministic."""

        torch.manual_seed(55)
        model = Nestor(_private_only_config())
        prefix_layer = model.blocks[0].layers[0]
        assert prefix_layer.expert_pool is not None
        assert prefix_layer.expert_pool.path_gate is None
        assert prefix_layer.expert_pool.public_router is None
        assert prefix_layer.expert_router is None
        assert len(prefix_layer.shared_experts) == 0
        assert all(
            len(bank) == 0 for bank in prefix_layer.group_specific_experts
        )
        suffix_layer = model.blocks[1].layers[0]
        assert suffix_layer.expert_pool is None
        assert suffix_layer.expert_router is not None
        # 4d contract: dim 32, k_a 2, k_q 1 -> round8(128/2) = 64.
        assert prefix_layer.expert_pool.private_hidden_dim == 64

        x = torch.randn(2, 6, 32)
        pos = torch.arange(6).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        output = model(x, pos, attn_mask)
        assert torch.isfinite(output.pred).all()
        output.pred.square().mean().backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, f"no grad for {name}"
            assert torch.isfinite(parameter.grad).all(), f"bad grad {name}"

        model.eval()
        with torch.no_grad():
            first = model(x, pos, attn_mask).pred
            second = model(x, pos, attn_mask).pred
        assert torch.equal(first, second)

    def test_aux_omits_the_public_router_entry(self) -> None:
        """A pooled layer reports only ``expert_pool_private``.

        The absent ``"expert_router"`` entry must be a no-op for every aux
        consumer (losses, routing metrics, the trainer's grad-accum merge).
        """

        torch.manual_seed(56)
        model = Nestor(_private_only_config())
        x = torch.randn(2, 6, 32)
        pos = torch.arange(6).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        output = model(x, pos, attn_mask)

        pooled_aux: dict[str, Any] = output.aux["blocks"][0]["layers"][0]
        assert "expert_router" not in pooled_aux
        assert len(pooled_aux["expert_pool_private"]) == 4
        unpooled_aux: dict[str, Any] = output.aux["blocks"][1]["layers"][0]
        assert isinstance(unpooled_aux["expert_router"], dict)
        assert "expert_pool_private" not in unpooled_aux

    def test_detached_balancing_updates_private_routers(self) -> None:
        """Loss-free balancing still reaches the private routers.

        Regression: the layer used to read the PUBLIC router's aux first and
        return early when it was missing, which would have skipped every
        private router in this mode.
        """

        torch.manual_seed(57)
        model = Nestor(_private_only_config())
        x = torch.randn(2, 6, 32)
        pos = torch.arange(6).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        output = model(x, pos, attn_mask)
        pool = model.blocks[0].layers[0].expert_pool
        assert pool is not None
        assert all(
            torch.all(router.choice_bias == 0)
            for router in pool.private_routers
        )
        model.update_detached_parameters(output.aux)
        moved = [
            router.choice_bias.abs().sum().item()
            for router in pool.private_routers
        ]
        assert any(value > 0 for value in moved)

    def test_grouped_dispatch_matches_sparse_in_the_backbone(self) -> None:
        """Both dispatch paths give the same model, incl. a 1-group pool."""

        config = _private_only_config()
        config.block_expert_pool_private_experts = [2, 2]
        torch.manual_seed(58)
        model = Nestor(config)
        pools = [
            layer.expert_pool
            for block in model.blocks
            for layer in block.layers
            if layer.expert_pool is not None
        ]
        assert len(pools) == 4
        assert [pool.num_groups for pool in pools] == [4, 4, 1, 1]
        # The dense tail block has k_a = 1, so its private experts carry the
        # full 4d width (32 * 4 = 128) while the grouped prefix splits it.
        assert [pool.private_hidden_dim for pool in pools] == [64, 64, 128, 128]

        x = torch.randn(2, 6, 32)
        pos = torch.arange(6).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            for pool in pools:
                pool.grouped_dispatch = False
            sparse = model(x, pos, attn_mask).pred
            for pool in pools:
                pool.grouped_dispatch = True
            grouped = model(x, pos, attn_mask).pred
        torch.testing.assert_close(
            grouped, sparse, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
        )

        model.train()
        model(x, pos, attn_mask).pred.square().mean().backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, f"no grad for {name}"
            assert torch.isfinite(parameter.grad).all(), f"bad grad {name}"

    def test_config_validation_of_the_private_only_mode(self) -> None:
        """(0, >0) is legal, (>0, 0) is not, (0, 0) disables pooling."""

        # Private-only builds.
        assert Nestor(_private_only_config()) is not None

        # Public-only is not implemented.
        config = _private_only_config()
        config.block_expert_pool_public_experts = [2, 0]
        config.block_expert_pool_private_experts = [0, 0]
        try:
            config.to_dict()
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for a public-only pool")

        # Neither pool: the layers keep their ordinary MLP machinery.
        config = _private_only_config()
        config.block_expert_pool_private_experts = [0, 0]
        model = Nestor(config)
        assert all(
            layer.expert_pool is None and layer.expert_router is not None
            for block in model.blocks
            for layer in block.layers
        )

        # Pools still may not sit on top of a configured MoE.
        config = _private_only_config()
        config.router_num_experts_per_group = [2, 1]
        try:
            config.to_dict()
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for pool over MoE")


class PrivateOnlyExpertPoolReferenceMupTest:
    """Composition with the reference-muP post-construction transform."""

    def test_configure_reference_mup_model(self) -> None:
        """The transform reaches every private router and expert, gate-free."""

        torch.manual_seed(59)
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(embed_dim=32, vocab_size=64),
            backbone_config=_private_only_config(),
            head_config=LanguageModelingHeadConfig(vocab_size=64),
            reference_mup=ReferenceMupConfig(num_layers=4),
        )
        model = Model.from_config(config)
        assert isinstance(model, SequenceModel)
        pool = model.backbone.blocks[0].layers[0].expert_pool
        assert pool is not None
        assert pool.public_router is None and pool.path_gate is None
        # Deterministic bias-free reference routers on every private router.
        for router in pool.private_routers:
            assert router.noise_mlp is None
            assert router.score_mlp.layers[-1].bias is None
            scaling = get_scaling(router.score_mlp.layers[-1].weight)
            assert scaling is not None and scaling.optimizer == "adamw"
            assert scaling.lr_mul == 1.0 / 8.0
        # Every private expert carries the matrix AdamW group.
        for bank in pool.private_experts:
            for expert in bank:
                for linear in (expert.layers[0], expert.layers[1], expert.gate):
                    scaling = get_scaling(linear.weight)
                    assert scaling is not None and scaling.optimizer == "adamw"
                    assert scaling.lr_mul == 1.0 / 8.0

        token_ids = torch.randint(0, 64, (2, 8))
        output = model(token_ids)
        assert torch.isfinite(output.pred).all()
        output.pred.float().square().mean().backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, f"no grad for {name}"
            assert torch.isfinite(parameter.grad).all(), f"bad grad {name}"
