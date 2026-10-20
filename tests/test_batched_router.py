"""The ``G`` private pool routers evaluated as ONE batched router.

Reference = the sequential path (``Router.forward`` per group, exactly as
``ExpertPoolMLP._route`` runs it with ``NESTOR_BATCHED_ROUTERS`` unset). Pins:
the batched heads equal the per-router MLPs; every ``RouterOutput`` field of
every group matches (scores / probs / gates numerically, top-k choices and
masks exactly) for both gate modes; the stacked tensors the dispatch consumes
equal the sequential ``torch.stack``; the whole pool forward and its router
parameter gradients agree between the two paths; training noise keeps the
routers' shapes, masking and anneal schedule; and mismatched routers refuse to
batch.
"""

from __future__ import annotations

import copy

import torch

from nestor.backbones.batched_router import (
    batched_mlp,
    check_batchable,
    route_private_batched,
)
from nestor.backbones.expert_pools import ExpertPoolMLP


def _make_pool(*, gate_mode: str = "switch", seed: int = 0) -> ExpertPoolMLP:
    """A small seeded pool with randomized choice biases (sequential routers).

    Args:
        gate_mode (str): Private gate rule, ``"switch"`` or ``"renorm"``.
        seed (int): ``torch.manual_seed`` value.

    Returns:
        ExpertPoolMLP: The pool, on the sequential router path.
    """

    torch.manual_seed(seed)
    pool = ExpertPoolMLP(
        dim=24,
        num_groups=4,
        num_groups_per_token=2,
        num_public_experts=3,
        public_top_k=2,
        num_private_experts=5,
        private_top_k=2,
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activation="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activation="gelu",
        loss_free_balancing=True,
        bias_update_rate=1e-3,
        private_gate_mode=gate_mode,
    )
    with torch.no_grad():
        for router in pool.private_routers:
            router.choice_bias.normal_(std=0.1)
    return pool


def _inputs(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random tokens, a validity mask with padding, and a membership mask.

    Args:
        seed (int): ``torch.manual_seed`` value.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``x [2, 7, 24]``,
        ``valid [2, 7]`` (last two tokens of image 1 are padding) and
        ``member [2, 7, 4]`` with exactly two groups per token.
    """

    torch.manual_seed(seed)
    x = torch.randn(2, 7, 24)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1, 5:] = False
    member = torch.zeros(2, 7, 4, dtype=torch.bool)
    picks = torch.rand(2, 7, 4).topk(2, dim=-1).indices
    member.scatter_(-1, picks, True)
    return x, valid, member


def _sequential(pool: ExpertPoolMLP, x, valid, member) -> list:
    """The reference: one ``Router.forward`` per group, masked to its members."""

    return [
        pool.private_routers[g](x, valid & member[:, :, g])
        for g in range(pool.num_groups)
    ]


class BatchedMlpTest:
    """The stacked heads reproduce the per-head ``ScaledMLP`` outputs."""

    def test_matches_each_head(self) -> None:
        """``batched_mlp(x, heads)[:, g]`` equals ``heads[g](x)``."""

        pool = _make_pool()
        x = torch.randn(11, 24)
        heads = [r.score_mlp for r in pool.private_routers]
        out = batched_mlp(x, heads)
        assert out.shape == (11, 4, 5)
        for g, head in enumerate(heads):
            torch.testing.assert_close(out[:, g], head(x), rtol=1e-6, atol=1e-6)


class BatchedRouteEquivalenceTest:
    """Every per-group ``RouterOutput`` field matches the sequential path (eval)."""

    def _check(self, gate_mode: str) -> None:
        pool = _make_pool(gate_mode=gate_mode).eval()
        x, valid, member = _inputs()
        reference = _sequential(pool, x, valid, member)
        outputs, gates, choices = route_private_batched(
            list(pool.private_routers), x, valid, member, training=False
        )
        assert len(outputs) == pool.num_groups
        for ref, out in zip(reference, outputs):
            torch.testing.assert_close(out.scores, ref.scores, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(
                out.softmax_probs, ref.softmax_probs, rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(out.gates, ref.gates, rtol=1e-6, atol=1e-6)
            assert torch.equal(out.topk_indices, ref.topk_indices)
            assert torch.equal(out.active_mask, ref.active_mask)
            assert torch.equal(out.valid_token_mask, ref.valid_token_mask)
        # The stacked tensors are what _private_slots would have stacked.
        torch.testing.assert_close(
            gates, torch.stack([r.gates for r in reference], dim=-2),
            rtol=1e-6, atol=1e-6,
        )
        assert torch.equal(
            choices, torch.stack([r.topk_indices for r in reference], dim=-2)
        )

    def test_switch_gate(self) -> None:
        """The production gate (every hybrid arm): full-softmax switch gates."""

        self._check("switch")

    def test_renorm_gate(self) -> None:
        """The legacy renormalized gate."""

        self._check("renorm")

    def test_gates_are_zero_outside_valid_members(self) -> None:
        """Masking to ``valid & member`` is applied per group."""

        pool = _make_pool().eval()
        x, valid, member = _inputs()
        outputs, gates, _ = route_private_batched(
            list(pool.private_routers), x, valid, member, training=False
        )
        allowed = valid.unsqueeze(-1) & member  # [B, S, G]
        assert torch.all(gates[~allowed] == 0)
        for g, out in enumerate(outputs):
            assert torch.all(out.gates[~allowed[:, :, g]] == 0)
            assert torch.all(out.gates[allowed[:, :, g]].sum(-1) > 0)


class BatchedPoolForwardTest:
    """The whole pool agrees between the two router paths, forward and backward."""

    def _pair(self) -> tuple[ExpertPoolMLP, ExpertPoolMLP]:
        sequential = _make_pool()
        batched = copy.deepcopy(sequential)
        batched.batched_routers = True   # the construction-time flag, set directly
        check_batchable(list(batched.private_routers))
        return sequential, batched

    def test_forward_matches(self) -> None:
        """``forward`` (grouped dispatch on CPU) agrees on both paths in eval."""

        sequential, batched = self._pair()
        sequential.eval()
        batched.eval()
        x, valid, member = _inputs()
        group_gates = torch.rand(2, 7, 4) * member
        group_gates = group_gates / group_gates.sum(-1, keepdim=True)
        for pool in (sequential, batched):
            pool.grouped_dispatch = True
        out_seq, pub_seq, priv_seq = sequential(x, valid, group_gates, member)
        out_bat, pub_bat, priv_bat = batched(x, valid, group_gates, member)
        torch.testing.assert_close(out_bat, out_seq, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(pub_bat.gates, pub_seq.gates)
        for a, b in zip(priv_seq, priv_bat):
            assert torch.equal(a.topk_indices, b.topk_indices)
            torch.testing.assert_close(b.gates, a.gates, rtol=1e-6, atol=1e-6)

    def test_forward_per_slot_matches(self) -> None:
        """``forward_per_slot`` (the column-mode entry point) agrees as well."""

        sequential, batched = self._pair()
        sequential.eval()
        batched.eval()
        x, valid, member = _inputs()
        slot_groups = member.float().topk(2, dim=-1).indices  # [B, S, k_a]
        group_gates = torch.rand(2, 7, 4) * member
        group_gates = group_gates / group_gates.sum(-1, keepdim=True)
        for pool in (sequential, batched):
            pool.grouped_dispatch = True
        pub_seq, priv_seq, *_ = sequential.forward_per_slot(
            x, valid, group_gates, member, slot_groups
        )
        pub_bat, priv_bat, *_ = batched.forward_per_slot(
            x, valid, group_gates, member, slot_groups
        )
        torch.testing.assert_close(pub_bat, pub_seq, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(priv_bat, priv_seq, rtol=1e-5, atol=1e-5)

    def test_router_gradients_match(self) -> None:
        """Gradients reach the routers' own parameters identically on both paths."""

        sequential, batched = self._pair()
        sequential.eval()
        batched.eval()
        x, valid, member = _inputs()
        group_gates = torch.rand(2, 7, 4) * member
        group_gates = group_gates / group_gates.sum(-1, keepdim=True)
        for pool in (sequential, batched):
            pool.grouped_dispatch = True
            out, _, _ = pool(x, valid, group_gates, member)
            out.square().sum().backward()
        for r_seq, r_bat in zip(sequential.private_routers, batched.private_routers):
            for p_seq, p_bat in zip(r_seq.parameters(), r_bat.parameters()):
                assert (p_seq.grad is None) == (p_bat.grad is None)
                if p_seq.grad is not None:
                    torch.testing.assert_close(p_bat.grad, p_seq.grad, rtol=1e-4, atol=1e-5)


class BatchedRouteTrainingTest:
    """Training mode: noise applied, masks respected, anneal counters advance."""

    def test_training_noise_shapes_and_anneal(self) -> None:
        """Noise keeps the layout; each router's anneal counter advances by one."""

        pool = _make_pool().train()
        x, valid, member = _inputs()
        for router in pool.private_routers:
            # The anneal schedule is normally armed at construction from
            # NESTOR_ROUTER_NOISE_ANNEAL_STEPS; arm it the same way here.
            router._noise_anneal_steps = 10
            router.register_buffer("_noise_steps", torch.zeros((), dtype=torch.long))
        before = [int(r._noise_steps) for r in pool.private_routers]
        torch.manual_seed(3)
        outputs, gates, choices = route_private_batched(
            list(pool.private_routers), x, valid, member, training=True
        )
        assert gates.shape == (2, 7, 4, 5) and choices.shape == (2, 7, 4, 2)
        allowed = valid.unsqueeze(-1) & member
        assert torch.all(gates[~allowed] == 0)
        assert [int(r._noise_steps) for r in pool.private_routers] == [
            b + 1 for b in before
        ]
        # Noise changes the scores relative to the noiseless pass.
        clean, _, _ = route_private_batched(
            list(pool.private_routers), x, valid, member, training=False
        )
        assert not torch.allclose(outputs[0].scores, clean[0].scores)


class CheckBatchableTest:
    """Routers of different shape or gate mode refuse to batch."""

    def test_mismatch_raises(self) -> None:
        """A router with a different choice count is rejected."""

        pool = _make_pool()
        other = _make_pool(gate_mode="renorm")
        routers = [pool.private_routers[0], other.private_routers[0]]
        try:
            check_batchable(routers)
        except ValueError:
            return
        raise AssertionError("check_batchable accepted mismatched gate modes")
