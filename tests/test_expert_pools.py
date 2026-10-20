"""Tests for the public/private expert-pool MLP.

Covers the pool width contract, a slow
per-token reference of the pool equation, gradient flow / DDP-safety, the
2-block grouped-prefix integration (with the Kimi AttnRes recipe), and
composition with the reference-muP transform.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from nestor.backbones import Nestor
from nestor.backbones.expert_pools import (
    ExpertPoolMLP,
    private_expert_hidden_dim,
    public_expert_hidden_dim,
)
from nestor.backbones.nestor import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig
from nestor.scaling.blocks import get_scaling


def _make_pool(
    *,
    dim: int = 16,
    num_groups: int = 4,
    num_groups_per_token: int = 2,
    num_public_experts: int = 3,
    public_top_k: int = 1,
    num_private_experts: int = 3,
    private_top_k: int = 1,
    seed: int = 0,
) -> ExpertPoolMLP:
    """Build a small seeded pool with randomized path gate and router biases.

    The path gate is zero-init (alphas exactly 0.5) and the loss-free choice
    biases start at zero; both are randomized here so the reference comparison
    exercises the biased top-k selection and a non-trivial path mixture.

    Args:
        dim (int): Token feature width.
        num_groups (int): Attention groups (one private pool each).
        num_groups_per_token (int): Active attention groups per token.
        num_public_experts (int): Public pool size.
        public_top_k (int): Public experts per token.
        num_private_experts (int): Private pool size per group.
        private_top_k (int): Private experts per selected group.
        seed (int): ``torch.manual_seed`` value.

    Returns:
        ExpertPoolMLP: The randomized pool module.
    """

    torch.manual_seed(seed)
    pool = ExpertPoolMLP(
        dim=dim,
        num_groups=num_groups,
        num_groups_per_token=num_groups_per_token,
        num_public_experts=num_public_experts,
        public_top_k=public_top_k,
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
        pool.path_gate.weight.normal_()
        pool.public_router.choice_bias.normal_(std=0.1)
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
        dead_groups (tuple[int, ...]): Groups no token may select (their
            scores are driven to ``-inf``), so their private pools see ZERO
            member tokens.

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


def reference_expert_pool_forward(
    pool: ExpertPoolMLP,
    x: Tensor,  # (batch_size, seq_len, dim)
    valid_token_mask: Tensor,  # (batch_size, seq_len) bool
    group_gates: Tensor,  # (batch_size, seq_len, num_groups)
    group_active_mask: Tensor,  # (batch_size, seq_len, num_groups) bool
) -> Tensor:
    """Slow per-token reference of the pool equation (eval mode, no noise).

    Explicit python loops over tokens, groups, and experts. Routing math is
    re-derived from the routers' score MLPs and choice biases (top-k on the
    biased scores, softmax gates over the raw selected scores), matching
    ``Router.forward`` in eval mode.

    Args:
        pool (ExpertPoolMLP): Pool module in eval mode.
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
            alphas = torch.softmax(pool.path_gate(h).float(), dim=-1).to(
                h.dtype
            )
            public = torch.zeros_like(h)
            private = torch.zeros_like(h)
            if valid_token_mask[b, s]:
                public = routed_sum(pool.public_router, pool.public_experts, h)
                for g in range(pool.num_groups):
                    if not group_active_mask[b, s, g]:
                        continue
                    private = private + group_gates[b, s, g] * routed_sum(
                        pool.private_routers[g], pool.private_experts[g], h
                    )
            out[b, s] = alphas[0] * public + alphas[1] * private
    return out


class ExpertPoolWidthsTest:
    """The doc's width table (SP-50 active contract, multiples of 8)."""

    def test_doc_table_widths(self) -> None:
        """d=640 -> 1280/424, d=960 -> 1920/640 (and 1b: 2560/856)."""

        assert public_expert_hidden_dim(640, 1) == 1280
        assert private_expert_hidden_dim(640, 3, 1) == 424
        assert public_expert_hidden_dim(960, 1) == 1920
        assert private_expert_hidden_dim(960, 3, 1) == 640
        assert public_expert_hidden_dim(1280, 1) == 2560
        assert private_expert_hidden_dim(1280, 3, 1) == 856

    def test_top_k_narrows_active_width(self) -> None:
        """Raising expert top-k divides each family's per-expert width."""

        assert public_expert_hidden_dim(640, 2) == 640
        assert private_expert_hidden_dim(640, 3, 2) == 216  # 1280/6 -> 213.3
        # Aggregate active width stays ~2d per family.
        assert 2 * public_expert_hidden_dim(640, 2) == 1280
        assert abs(3 * 2 * private_expert_hidden_dim(640, 3, 2) - 1280) <= 24

    def test_public_budget_rescales_the_public_family(self) -> None:
        """``active_budget`` on the public width mirrors the private knob (2.5d = 1.25 + 1.25)."""

        assert public_expert_hidden_dim(640, 2, active_budget=None) == 640
        assert public_expert_hidden_dim(640, 2, active_budget=1.25) == 400
        assert public_expert_hidden_dim(960, 2, active_budget=1.25) == 600
        assert private_expert_hidden_dim(640, 3, 1, active_budget=1.25) == 264  # 266.7
        assert private_expert_hidden_dim(640, 8, 1, active_budget=1.25) == 96  # 100 -> 96
        public = 2 * public_expert_hidden_dim(640, 2, active_budget=1.25)
        private = 3 * private_expert_hidden_dim(640, 3, 1, active_budget=1.25)
        assert abs(public + private - 2.5 * 640) <= 16


#: Tolerance for grouped-GEMM vs sparse/reference comparisons. The grouped
#: path reassociates the same fp32 arithmetic: every expert's rows are packed
#: into one (16-row-aligned, zero-padded) matmul instead of one matmul per
#: expert over its own token slice, and the per-token combine sums the routed
#: slots instead of accumulating them with ``index_add_``. Both reorder fp32
#: sums, so agreement is to a few ULPs of the ~1-magnitude outputs, not bitwise
#: (measured max |diff| ~7e-7 on these shapes).
_GROUPED_RTOL = 1e-5
_GROUPED_ATOL = 1e-6


class ExpertPoolMLPReferenceTest:
    """Module forward vs the slow per-token reference implementation.

    Both dispatch paths are checked against the reference, so the grouped
    path is validated against the mathematical definition of the pool
    equation, not only against the sparse implementation.
    """

    def test_matches_reference_top1(self) -> None:
        """Canonical p*k1_q*k1 geometry matches the reference exactly."""

        for grouped in (False, True):
            pool = _make_pool(seed=0)
            pool.eval()
            pool.grouped_dispatch = grouped
            torch.manual_seed(2)
            x = torch.randn(2, 7, 16)
            valid = torch.ones(2, 7, dtype=torch.bool)
            valid[1, -2:] = False  # exercise padding
            gates, active = _make_group_routing(2, 7, 4, 2, seed=3)
            gates = gates * valid.unsqueeze(-1).to(gates.dtype)
            active = active & valid.unsqueeze(-1)

            with torch.no_grad():
                out, public_output, private_outputs = pool(
                    x, valid, gates, active
                )
                ref = reference_expert_pool_forward(
                    pool, x, valid, gates, active
                )
            torch.testing.assert_close(
                out, ref, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
            )
            assert len(private_outputs) == pool.num_groups
            # Private balancing statistics only count member tokens.
            for g, private_output in enumerate(private_outputs):
                assert torch.equal(
                    private_output.valid_token_mask, valid & active[:, :, g]
                )
            assert torch.equal(public_output.valid_token_mask, valid)

    def test_matches_reference_top2(self) -> None:
        """Top-2 expert selection (non-constant gates) also matches."""

        for grouped in (False, True):
            pool = _make_pool(
                num_public_experts=4,
                public_top_k=2,
                num_private_experts=4,
                private_top_k=2,
                seed=4,
            )
            pool.eval()
            pool.grouped_dispatch = grouped
            torch.manual_seed(5)
            x = torch.randn(2, 5, 16)
            valid = torch.ones(2, 5, dtype=torch.bool)
            gates, active = _make_group_routing(2, 5, 4, 2, seed=6)

            with torch.no_grad():
                out, _, _ = pool(x, valid, gates, active)
                ref = reference_expert_pool_forward(
                    pool, x, valid, gates, active
                )
            torch.testing.assert_close(
                out, ref, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
            )


class ExpertPoolGroupedDispatchTest:
    """Grouped-GEMM dispatch vs the per-expert sparse loop.

    The grouped path is the production default (GPU); the sparse loop is the
    CPU fallback and the ``NESTOR_POOL_GROUPED_MM=0`` debug path. Forcing
    ``grouped_dispatch`` runs both on CPU, where ``segment_mm`` takes its
    host-side segment loop.
    """

    def test_matches_sparse_top1_with_padding_and_dead_group(self) -> None:
        """Top-1/top-1 with padding tokens and a zero-member attention group."""

        pool = _make_pool(seed=20)  # randomized choice biases + path gate
        pool.eval()
        torch.manual_seed(21)
        x = torch.randn(3, 11, 16)
        valid = torch.ones(3, 11, dtype=torch.bool)
        valid[0, -4:] = False
        valid[2, -1:] = False
        # Group 3 is never selected -> its private pool gets no tokens.
        gates, active = _make_group_routing(
            3, 11, 4, 2, seed=22, dead_groups=(3,)
        )
        gates = gates * valid.unsqueeze(-1).to(gates.dtype)
        active = active & valid.unsqueeze(-1)
        assert not active[:, :, 3].any()

        with torch.no_grad():
            pool.grouped_dispatch = False
            sparse, sparse_public, sparse_private = pool(
                x, valid, gates, active
            )
            pool.grouped_dispatch = True
            grouped, grouped_public, grouped_private = pool(
                x, valid, gates, active
            )
        torch.testing.assert_close(
            grouped, sparse, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
        )
        # The RouterOutputs the trainer consumes must be bit-identical: the
        # dispatch changes only how the experts are evaluated.
        for field in range(len(sparse_public)):
            assert torch.equal(sparse_public[field], grouped_public[field])
        for sparse_out, grouped_out in zip(sparse_private, grouped_private):
            for field in range(len(sparse_out)):
                assert torch.equal(sparse_out[field], grouped_out[field])

    def test_matches_sparse_top_k_greater_than_one(self) -> None:
        """Top-2 public and top-2 private (non-constant expert gates)."""

        pool = _make_pool(
            num_groups=4,
            num_groups_per_token=3,
            num_public_experts=5,
            public_top_k=2,
            num_private_experts=4,
            private_top_k=2,
            seed=23,
        )
        pool.eval()
        torch.manual_seed(24)
        x = torch.randn(2, 9, 16)
        valid = torch.ones(2, 9, dtype=torch.bool)
        valid[1, -3:] = False
        gates, active = _make_group_routing(2, 9, 4, 3, seed=25)
        gates = gates * valid.unsqueeze(-1).to(gates.dtype)
        active = active & valid.unsqueeze(-1)

        with torch.no_grad():
            pool.grouped_dispatch = False
            sparse, _, _ = pool(x, valid, gates, active)
            pool.grouped_dispatch = True
            grouped, _, _ = pool(x, valid, gates, active)
            ref = reference_expert_pool_forward(pool, x, valid, gates, active)
        torch.testing.assert_close(
            grouped, sparse, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
        )
        torch.testing.assert_close(
            grouped, ref, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
        )

    def test_matches_sparse_when_all_groups_are_selected(self) -> None:
        """The degenerate ``k_a == num_groups`` geometry (every group active)."""

        pool = _make_pool(
            num_groups=3, num_groups_per_token=3, seed=26
        )
        pool.eval()
        torch.manual_seed(27)
        x = torch.randn(2, 6, 16)
        valid = torch.ones(2, 6, dtype=torch.bool)
        gates, active = _make_group_routing(2, 6, 3, 3, seed=28)

        with torch.no_grad():
            pool.grouped_dispatch = False
            sparse, _, _ = pool(x, valid, gates, active)
            pool.grouped_dispatch = True
            grouped, _, _ = pool(x, valid, gates, active)
        torch.testing.assert_close(
            grouped, sparse, rtol=_GROUPED_RTOL, atol=_GROUPED_ATOL
        )

    def test_matches_sparse_under_bf16_autocast(self) -> None:
        """bf16 autocast with an fp32 residual stream agrees on both paths.

        The grouped path casts inputs/weights to the autocast dtype of the
        input's OWN device (fp32 accumulation inside the GEMM), forms the
        per-slot product in that dtype, and sums the slots in the fp32 input
        dtype -- exactly the sparse path's ``weighted_out.to(residual.dtype)``
        accumulation, so on CPU the two agree bitwise here.
        """

        pool = _make_pool(seed=29)
        pool.eval()
        torch.manual_seed(30)
        x = torch.randn(2, 6, 16)
        valid = torch.ones(2, 6, dtype=torch.bool)
        gates, active = _make_group_routing(2, 6, 4, 2, seed=31)

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

    def test_grouped_path_is_ddp_safe(self) -> None:
        """Every trainable parameter ends the step with a finite grad.

        Covers the two cases the sparse path needed its keepalive branch for:
        a private pool whose attention group has NO member token, and experts
        that no token selects (the choice bias pins routing to expert 0). The
        grouped path stacks every expert weight and every private router's
        gates, so both join the graph unconditionally.
        """

        pool = _make_pool(seed=32)
        pool.train()
        pool.grouped_dispatch = True
        with torch.no_grad():
            # Pin every router to its expert 0: experts 1..Q-1 get no tokens.
            pool.public_router.choice_bias.zero_()
            pool.public_router.choice_bias[0] = 1e3
            for router in pool.private_routers:
                router.choice_bias.zero_()
                router.choice_bias[0] = 1e3
        torch.manual_seed(33)
        x = torch.randn(2, 8, 16, requires_grad=True)
        valid = torch.ones(2, 8, dtype=torch.bool)
        valid[1, -2:] = False
        gates, active = _make_group_routing(
            2, 8, 4, 2, seed=34, dead_groups=(2, 3)
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

    def test_gradients_match_the_sparse_path(self) -> None:
        """Backward agrees too, not just the forward.

        Run in eval mode, so the exploration-noise MLPs are out of the graph
        on BOTH paths (their grads stay ``None`` either way -- the training
        DDP contract is covered by ``test_grouped_path_is_ddp_safe``).
        """

        grads: dict[bool, dict[str, Tensor | None]] = {}
        for grouped in (False, True):
            pool = _make_pool(seed=35)
            pool.eval()  # deterministic routing (no exploration noise)
            pool.grouped_dispatch = grouped
            torch.manual_seed(36)
            x = torch.randn(2, 7, 16)
            valid = torch.ones(2, 7, dtype=torch.bool)
            valid[0, -2:] = False
            gates, active = _make_group_routing(2, 7, 4, 2, seed=37)
            gates = gates * valid.unsqueeze(-1).to(gates.dtype)
            active = active & valid.unsqueeze(-1)

            out, _, _ = pool(x, valid, gates, active)
            out.square().sum().backward()
            grads[grouped] = {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in pool.named_parameters()
            }
        assert {name for name, g in grads[False].items() if g is None} == {
            name for name, g in grads[True].items() if g is None
        }
        for name, sparse_grad in grads[False].items():
            if sparse_grad is None:
                continue
            torch.testing.assert_close(
                grads[True][name],
                sparse_grad,
                rtol=_GROUPED_RTOL,
                atol=_GROUPED_ATOL,
            )

    def test_enablement_defaults_and_env_opt_out(self) -> None:
        """Auto by default (grouped on CUDA), opt out with the env flag.

        ``grouped_dispatch is None`` means "decide per forward": grouped when
        the inputs are CUDA tensors and ``torch._grouped_mm`` exists, sparse
        otherwise -- so CPU runs (tests, the equivalence scripts) fall back
        automatically and no launcher has to opt in.
        """

        pool = _make_pool(seed=38)
        assert pool.grouped_dispatch is None
        cpu_x = torch.zeros(1, pool.dim)
        assert pool._use_grouped_dispatch(cpu_x) is False

        previous = os.environ.get("NESTOR_POOL_GROUPED_MM")
        try:
            os.environ["NESTOR_POOL_GROUPED_MM"] = "0"
            opted_out = _make_pool(seed=38)
        finally:
            if previous is None:
                os.environ.pop("NESTOR_POOL_GROUPED_MM", None)
            else:
                os.environ["NESTOR_POOL_GROUPED_MM"] = previous
        assert opted_out.grouped_dispatch is False
        assert opted_out._use_grouped_dispatch(cpu_x) is False


class ExpertPoolMLPGradientsTest:
    """Gradient flow / DDP-safety of the pool parameters."""

    def test_all_parameters_receive_gradients(self) -> None:
        """Every pool parameter joins the graph (grad not None, finite)."""

        pool = _make_pool(seed=7)
        pool.train()
        torch.manual_seed(8)
        x = torch.randn(2, 9, 16, requires_grad=True)
        valid = torch.ones(2, 9, dtype=torch.bool)
        gates, active = _make_group_routing(2, 9, 4, 2, seed=9)

        out, _, _ = pool(x, valid, gates, active)
        out.square().sum().backward()
        for name, parameter in pool.named_parameters():
            assert parameter.grad is not None, f"no grad for {name}"
            assert torch.isfinite(parameter.grad).all(), f"bad grad {name}"
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_router_and_path_gate_gradients_nonzero_with_top2(self) -> None:
        """With top-2 gates the routers and path gate get NONZERO gradients.

        (With top-1 the renormalized gate is the constant 1.0 -- nestor's
        native router semantics -- so router score heads receive exactly-zero
        gradient and learn through noise + loss-free balancing only; top-2
        exercises the real gradient path.)
        """

        pool = _make_pool(
            num_public_experts=4,
            public_top_k=2,
            num_private_experts=4,
            private_top_k=2,
            seed=10,
        )
        pool.train()
        torch.manual_seed(11)
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
        assert pool.path_gate.weight.grad is not None
        assert pool.path_gate.weight.grad.abs().sum() > 0
        public_score = pool.public_router.score_mlp.layers[-1].weight
        assert public_score.grad is not None
        assert public_score.grad.abs().sum() > 0
        for g, router in enumerate(pool.private_routers):
            score = router.score_mlp.layers[-1].weight
            assert score.grad is not None, f"no grad for private router {g}"
            assert score.grad.abs().sum() > 0, f"zero grad, private router {g}"
        # Expert weights: the selected experts of each family carry signal.
        public_grad = sum(
            expert.layers[0].weight.grad.abs().sum()
            for expert in pool.public_experts
        )
        private_grad = sum(
            expert.layers[0].weight.grad.abs().sum()
            for bank in pool.private_experts
            for expert in bank
        )
        assert public_grad > 0
        assert private_grad > 0

    def test_rejects_one_sided_pools(self) -> None:
        """Public without private (and vice versa) must raise."""

        try:
            _make_pool(num_private_experts=0)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for one-sided pool")


def _pooled_kimi_config() -> NestorConfig:
    """A tiny 2-block grouped-prefix config with pools + Kimi block AttnRes.

    Block 0: 4 attention groups top-2 with token dispatch, loss-free
    balancing, and a 2-public/2-private expert pool. Block 1: dense (no
    pool). Between-block transition is ``attn_res_kimi``.

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
        block_expert_pool_public_experts=[2, 0],
        block_expert_pool_public_top_k=[1, 1],
        block_expert_pool_private_experts=[2, 0],
        block_expert_pool_private_top_k=[1, 1],
    )


class ExpertPoolIntegrationTest:
    """Pooled grouped prefix inside the full Nestor backbone."""

    def test_kimi_v3_forward_backward_and_determinism(self) -> None:
        """Forward+backward is finite, DDP-safe, and eval-deterministic."""

        torch.manual_seed(12)
        model = Nestor(_pooled_kimi_config())
        prefix_layer = model.blocks[0].layers[0]
        assert prefix_layer.expert_pool is not None
        assert prefix_layer.expert_router is None
        assert len(prefix_layer.shared_experts) == 0
        assert all(
            len(bank) == 0 for bank in prefix_layer.group_specific_experts
        )
        suffix_layer = model.blocks[1].layers[0]
        assert suffix_layer.expert_pool is None
        assert suffix_layer.expert_router is not None

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

    def test_detached_balancing_updates_pool_routers(self) -> None:
        """The merged aux drives loss-free bias updates on all pool routers."""

        torch.manual_seed(13)
        model = Nestor(_pooled_kimi_config())
        x = torch.randn(2, 6, 32)
        pos = torch.arange(6).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
        output = model(x, pos, attn_mask)
        layer = model.blocks[0].layers[0]
        pool = layer.expert_pool
        assert pool is not None
        assert torch.all(pool.public_router.choice_bias == 0)
        model.update_detached_parameters(output.aux)
        # The 2-expert public pool cannot be exactly balanced on an even
        # token count only by chance; with 12 tokens over 2 experts the bias
        # moves unless assignment is exactly 6/6. Just require SOME pool
        # router bias to have moved (rate 1e-3, sign steps).
        moved = [
            router.choice_bias.abs().sum().item()
            for router in (pool.public_router, *pool.private_routers)
        ]
        assert any(value > 0 for value in moved)

    def test_grouped_dispatch_matches_sparse_in_the_backbone(self) -> None:
        """Both dispatch paths give the same model, incl. a 1-group pool.

        Pools are enabled on BOTH blocks here, so the production arm's dense
        tail block (``num_groups == 1``, where the group router is a
        fabricated constant output) goes through the grouped path too.
        """

        config = _pooled_kimi_config()
        config.block_expert_pool_public_experts = [2, 2]
        config.block_expert_pool_private_experts = [2, 2]
        torch.manual_seed(15)
        model = Nestor(config)
        pools = [
            layer.expert_pool
            for block in model.blocks
            for layer in block.layers
            if layer.expert_pool is not None
        ]
        assert len(pools) == 4
        assert [pool.num_groups for pool in pools] == [4, 4, 1, 1]

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

    def test_pool_config_validation(self) -> None:
        """One-sided pools and pool-over-MoE configs are rejected."""

        config = _pooled_kimi_config()
        config.block_expert_pool_private_experts = [0, 0]
        try:
            config.to_dict()
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for one-sided pool")

        config = _pooled_kimi_config()
        config.router_num_experts_per_group = [2, 1]
        try:
            config.to_dict()
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected ValueError for pool over MoE")


class ExpertPoolReferenceMupTest:
    """Composition with the reference-muP post-construction transform."""

    def test_configure_reference_mup_model(self) -> None:
        """The transform reaches pool routers, experts, and the path gate."""

        torch.manual_seed(14)
        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(embed_dim=32, vocab_size=64),
            backbone_config=_pooled_kimi_config(),
            head_config=LanguageModelingHeadConfig(vocab_size=64),
            reference_mup=ReferenceMupConfig(num_layers=4),
        )
        model = Model.from_config(config)
        assert isinstance(model, SequenceModel)
        pool = model.backbone.blocks[0].layers[0].expert_pool
        assert pool is not None
        # Deterministic bias-free reference routers on every pool router.
        for router in (pool.public_router, *pool.private_routers):
            assert router.noise_mlp is None
            assert router.score_mlp.layers[-1].bias is None
            scaling = get_scaling(router.score_mlp.layers[-1].weight)
            assert scaling is not None and scaling.optimizer == "adamw"
            assert scaling.lr_mul == 1.0 / 8.0
        # Pool experts and the path gate carry the matrix AdamW group.
        for expert in (pool.public_experts[0], pool.private_experts[0][0]):
            for linear in (expert.layers[0], expert.layers[1], expert.gate):
                scaling = get_scaling(linear.weight)
                assert scaling is not None and scaling.optimizer == "adamw"
                assert scaling.lr_mul == 1.0 / 8.0
        gate_scaling = get_scaling(pool.path_gate.weight)
        assert gate_scaling is not None and gate_scaling.optimizer == "adamw"
        assert gate_scaling.lr_mul == 1.0 / 8.0

        token_ids = torch.randint(0, 64, (2, 8))
        output = model(token_ids)
        assert torch.isfinite(output.pred).all()
        output.pred.float().square().mean().backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, f"no grad for {name}"
            assert torch.isfinite(parameter.grad).all(), f"bad grad {name}"

class ExpertPoolAutocastTest:
    """Regression tests for mixed-precision execution."""

    def test_bf16_autocast_forward_matches_dtype(self) -> None:
        """Pool runs under bf16 autocast with an fp32 residual stream.

        Regression: the sparse accumulators follow the (fp32) residual stream
        while the experts emit bf16 under autocast, so ``index_add_`` raised
        "self (Float) and source (BFloat16) must have the same scalar type"
        on the first GPU step. CPU bf16 autocast reproduces it exactly.
        """

        pool = _make_pool(seed=5)
        batch_size, seq_len = 2, 6
        torch.manual_seed(5)
        x = torch.randn(batch_size, seq_len, pool.dim)  # fp32 residual stream
        group_gates = torch.rand(batch_size, seq_len, pool.num_groups)
        group_gates = group_gates / group_gates.sum(dim=-1, keepdim=True)
        group_active = torch.ones(
            batch_size, seq_len, pool.num_groups, dtype=torch.bool
        )
        valid = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            update, _, _ = pool(x, valid, group_gates, group_active)

        assert update.shape == x.shape
        assert update.dtype == x.dtype
        assert torch.isfinite(update.float()).all()
        update.float().square().mean().backward()
        assert pool.path_gate.weight.grad is not None
