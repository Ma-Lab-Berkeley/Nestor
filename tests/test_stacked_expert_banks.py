"""Stacked expert banks (``NESTOR_STACKED_EXPERT_BANKS=1``): exact ownership swap.

Reference = the same ``ExpertPoolMLP`` built with the flag off (separate
per-expert 2-D parameters). Stacked, the pool must produce the same forward
and the same gradients (viewed per expert), survive ``.to()`` (views are
re-derived), expose each bank once in ``state_dict``, and Muon must update a
``[E, m, n]`` bank exactly as it updates the ``E`` matrices separately.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor, nn

from nestor.backbones.expert_bank_stack import (
    StackedExpertBank,
    consolidate_stacked_expert_banks,
)
from nestor.backbones.expert_pools import ExpertPoolMLP
from nestor.scaling.blocks import GatedScaledMLP
from nestor.scaling.muon import MuonAdamW

DIM, GROUPS, K, PUBLIC, PRIVATE = 32, 4, 2, 4, 3


def _build_pool(stacked: bool, seed: int = 0) -> ExpertPoolMLP:
    """A small hybrid pool (public 4 top-2, private 3 per column, switch gate)."""

    key = "NESTOR_STACKED_EXPERT_BANKS"
    previous = os.environ.get(key)
    os.environ[key] = "1" if stacked else "0"
    try:
        torch.manual_seed(seed)
        pool = ExpertPoolMLP(
            dim=DIM,
            num_groups=GROUPS,
            num_groups_per_token=K,
            num_public_experts=PUBLIC,
            public_top_k=2,
            num_private_experts=PRIVATE,
            private_top_k=2,
            router_score_mlp_hidden_dims=[],
            router_score_mlp_activation="gelu",
            router_noise_mlp_hidden_dims=[],
            router_noise_mlp_activation="gelu",
            private_gate_mode="switch",
        )
        # The model builder does this after the reference-muP pass; here the
        # pool is the whole model.
        consolidate_stacked_expert_banks(pool)
        return pool
    finally:
        if previous is None:
            del os.environ[key]
        else:
            os.environ[key] = previous


def _inputs(seed: int = 1) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """A batch, its group gates/membership and slot groups (top-k of G)."""

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 5, DIM, generator=gen)
    valid = torch.ones(2, 5, dtype=torch.bool)
    scores = torch.rand(2, 5, GROUPS, generator=gen)
    slot_groups = scores.topk(K, dim=-1).indices  # [2, 5, K]
    active = torch.zeros(2, 5, GROUPS, dtype=torch.bool).scatter(-1, slot_groups, True)
    gates = torch.zeros(2, 5, GROUPS).scatter(-1, slot_groups, 1.0 / K)
    return x, valid, gates, active, slot_groups


def _forward(pool: ExpertPoolMLP) -> Tensor:
    """Sum of the pool's per-slot outputs (public + private) as one tensor."""

    x, valid, gates, active, slot_groups = _inputs()
    pool.eval()  # routers: no noise
    public, private, _, _ = pool.forward_per_slot(x, valid, gates, active, slot_groups)
    return public + private.sum(2)


class StackedExpertBankTest:
    """Owner-level checks on one bank."""

    def test_layout_values_and_views(self) -> None:
        torch.manual_seed(0)
        experts = [GatedScaledMLP(8, [6], 8, bias=False) for _ in range(3)]
        reference = [
            (e.layers[0].weight.clone(), e.gate.weight.clone(), e.layers[1].weight.clone())
            for e in experts
        ]
        bank = StackedExpertBank(experts)
        assert tuple(bank.w_up.shape) == (3, 8, 6)
        assert tuple(bank.w_gate.shape) == (3, 8, 6)
        assert tuple(bank.w_down.shape) == (3, 6, 8)
        for i, (up, gate, down) in enumerate(reference):
            assert torch.equal(bank.w_up[i].t(), up)
            assert torch.equal(bank.w_gate[i].t(), gate)
            assert torch.equal(bank.w_down[i].t(), down)
            # The experts now read views of the owners, not parameters.
            assert "weight" not in experts[i].layers[0]._parameters
            assert torch.equal(experts[i].layers[0].weight, up)
        # Gradients through an expert's forward land on the stacked owner.
        x = torch.randn(4, 8)
        experts[1](x).sum().backward()
        assert bank.w_up.grad is not None
        assert bank.w_up.grad[0].abs().sum() == 0 and bank.w_up.grad[1].abs().sum() > 0

    def test_biased_experts_get_bias_banks(self) -> None:
        """Biased banks (the NATIVE shared bank) stack their biases too.

        Owner-level equivalence and tagging live in
        ``tests/test_native_stacked_banks.py``; here only the structural
        contract: bias owners exist, bias-free banks have none.
        """

        experts = [GatedScaledMLP(8, [6], 8, bias=True) for _ in range(2)]
        bank = StackedExpertBank(experts)
        assert tuple(bank.b_up.shape) == (2, 6)
        assert tuple(bank.b_down.shape) == (2, 8)
        free = StackedExpertBank(
            [GatedScaledMLP(8, [6], 8, bias=False) for _ in range(2)]
        )
        assert free.b_up is None and free.b_down is None

    def test_rejects_mixed_bias_presence(self) -> None:
        experts = [
            GatedScaledMLP(8, [6], 8, bias=True),
            GatedScaledMLP(8, [6], 8, bias=False),
        ]
        with pytest.raises(ValueError):
            StackedExpertBank(experts)


class StackedPoolEquivalenceTest:
    """The pool with stacked banks equals the per-expert reference (sparse CPU path)."""

    def test_consolidation_is_deferred_past_later_init_passes(self) -> None:
        """A re-init of the per-expert weights BEFORE consolidation is honoured.

        Regression: consolidating in ``__init__`` handed the reference-muP
        pass transposed views, so ``nn.init.normal_`` laid the same random
        draws out in a different element order (+0.006 loss at step 20 on
        a validation run). The builder now stacks last.
        """

        os.environ["NESTOR_STACKED_EXPERT_BANKS"] = "1"
        try:
            torch.manual_seed(3)
            pool = _build_pool(stacked=False)  # flag read again below
            assert pool.stacked_private is None  # declared, not stacked
            torch.manual_seed(11)
            for e in pool._private_bank():
                nn.init.normal_(e.layers[0].weight, std=0.02)
            reference = torch.stack([e.layers[0].weight.t().clone() for e in pool._private_bank()])
            assert consolidate_stacked_expert_banks(pool) == 2
            assert torch.equal(pool.stacked_private.w_up, reference)
            assert consolidate_stacked_expert_banks(pool) == 0  # idempotent
        finally:
            del os.environ["NESTOR_STACKED_EXPERT_BANKS"]

    def test_forward_and_gradients_match(self) -> None:
        reference = _build_pool(stacked=False)
        stacked = _build_pool(stacked=True)
        assert stacked.stacked_private is not None and stacked.stacked_public is not None
        assert reference.stacked_private is None
        out_ref = _forward(reference)
        out_new = _forward(stacked)
        assert torch.allclose(out_ref, out_new, atol=1e-6)
        out_ref.pow(2).sum().backward()
        out_new.pow(2).sum().backward()
        private_ref = reference._private_bank()
        up_ref = torch.stack([e.layers[0].weight.grad.t() for e in private_ref])
        gate_ref = torch.stack([e.gate.weight.grad.t() for e in private_ref])
        down_ref = torch.stack([e.layers[1].weight.grad.t() for e in private_ref])
        assert torch.allclose(stacked.stacked_private.w_up.grad, up_ref, atol=1e-6)
        assert torch.allclose(stacked.stacked_private.w_gate.grad, gate_ref, atol=1e-6)
        assert torch.allclose(stacked.stacked_private.w_down.grad, down_ref, atol=1e-6)
        pub_ref = torch.stack([e.layers[1].weight.grad.t() for e in reference.public_experts])
        assert torch.allclose(stacked.stacked_public.w_down.grad, pub_ref, atol=1e-6)
        # Router / path-gate parameters are untouched and agree too.
        for (name, p_ref), (_, p_new) in zip(
            reference.named_parameters(), stacked.named_parameters()
        ):
            if "router" in name or "path_gate" in name:
                assert (p_ref.grad is None) == (p_new.grad is None), name
                if p_ref.grad is not None:
                    assert torch.allclose(p_ref.grad, p_new.grad, atol=1e-6), name

    def test_parameter_set_and_count(self) -> None:
        reference = _build_pool(stacked=False)
        stacked = _build_pool(stacked=True)
        n_ref = sum(p.numel() for p in reference.parameters())
        n_new = sum(p.numel() for p in stacked.parameters())
        assert n_ref == n_new
        keys = set(stacked.state_dict().keys())
        assert "stacked_private.w_up" in keys and "stacked_public.w_down" in keys
        assert not any(k.startswith("private_experts.") for k in keys)
        # Every expert bank appears exactly once: 3 owners per bank.
        n_tensors_new = sum(1 for _ in stacked.parameters())
        n_expert_tensors_ref = 3 * (PUBLIC + GROUPS * PRIVATE)
        n_tensors_ref = sum(1 for _ in reference.parameters())
        assert n_tensors_new == n_tensors_ref - n_expert_tensors_ref + 6

    def test_views_survive_apply_and_state_dict_round_trip(self) -> None:
        stacked = _build_pool(stacked=True)
        out_before = _forward(stacked)
        moved = stacked.to(torch.float64)
        assert moved.stacked_private.w_up.dtype == torch.float64
        expert = moved._private_bank()[2]
        assert expert.layers[0].weight.dtype == torch.float64
        assert torch.equal(expert.layers[0].weight, moved.stacked_private.w_up[2].t())
        fresh = _build_pool(stacked=True, seed=5)
        fresh.load_state_dict(stacked.to(torch.float32).state_dict())
        assert torch.allclose(_forward(fresh), out_before, atol=1e-6)


class StackedCheckpointCompatibilityTest:
    """An eval job must load a stacked checkpoint without reproducing the env."""

    def test_marker_and_cross_load(self) -> None:
        from nestor.eval.checkpoint import _STACKED_BANK_MARKERS

        def is_stacked(keys) -> bool:
            """The detection load_experiment_checkpoint runs on a checkpoint's keys."""

            return any(m in k for k in keys for m in _STACKED_BANK_MARKERS)

        stacked = _build_pool(stacked=True)
        state = stacked.state_dict()
        assert is_stacked(state)  # also at module top level (no dotted prefix)
        assert is_stacked({f"backbone.blocks.0.layers.0.expert_pool.{k}" for k in state})
        assert not is_stacked(_build_pool(stacked=False).state_dict())
        # What load_experiment_checkpoint does when the file is stacked and the model is not.
        fresh = _build_pool(stacked=False, seed=9)
        assert consolidate_stacked_expert_banks(fresh, force=True) == 2
        missing, unexpected = fresh.load_state_dict(state, strict=False)
        assert not missing and not unexpected
        assert torch.allclose(_forward(fresh), _forward(stacked), atol=1e-6)


class MuonStackedBankTest:
    """Muon on an ``[E, m, n]`` parameter == Muon on the ``E`` matrices separately."""

    def test_three_d_update_matches_per_matrix(self) -> None:
        torch.manual_seed(0)
        mats = [nn.Parameter(torch.randn(12, 7)) for _ in range(4)]
        bank = nn.Parameter(torch.stack([m.detach() for m in mats]).clone())
        grads = [torch.randn(12, 7) for _ in range(4)]
        for m, g in zip(mats, grads):
            m.grad = g.clone()
        bank.grad = torch.stack(grads).clone()
        group = {"method": "muon", "lr": 0.05, "weight_decay": 0.01, "momentum": 0.9,
                     "nesterov": True, "ns_steps": 5}
        opt_ref = MuonAdamW([{"params": mats, **group}])
        opt_new = MuonAdamW([{"params": [bank], **group}])
        for _ in range(3):
            opt_ref.step()
            opt_new.step()
            for m, g in zip(mats, grads):
                m.grad = (g * 0.5).clone()
            bank.grad = torch.stack([g * 0.5 for g in grads]).clone()
        for i, m in enumerate(mats):
            assert torch.allclose(bank[i], m, rtol=1e-3, atol=1e-4)

    def test_mixed_2d_and_3d_bucket_matches_per_matrix(self) -> None:
        """A [E,m,n] bank and plain [m,n] matrices of that shape share one batched call.

        Regression: orthogonalizing each bank separately cost one
        Newton--Schulz call per bank (162 x 15 matmuls per step on the 500m
        arm). They are now concatenated per matrix shape, which
        must not change any update.
        """

        from nestor.scaling.muon import _orthogonalize_updates

        torch.manual_seed(0)
        bank_a = torch.randn(3, 10, 6)
        plain = torch.randn(10, 6)
        bank_b = torch.randn(2, 10, 6)
        other = torch.randn(6, 10)  # different shape -> its own bucket
        batched = _orthogonalize_updates([bank_a, plain, bank_b, other], 5)
        per_matrix = _orthogonalize_updates(
            [bank_a[0], bank_a[1], bank_a[2], plain, bank_b[0], bank_b[1], other], 5
        )
        assert [tuple(u.shape) for u in batched] == [
            (3, 10, 6), (10, 6), (2, 10, 6), (6, 10)
        ]
        flat = [batched[0][0], batched[0][1], batched[0][2], batched[1],
                batched[2][0], batched[2][1], batched[3]]
        for got, want in zip(flat, per_matrix):
            assert torch.allclose(got, want, rtol=1e-3, atol=1e-4)

    def test_rejects_other_ranks(self) -> None:
        p = nn.Parameter(torch.randn(3, 4, 5, 6))
        p.grad = torch.randn_like(p)
        opt = MuonAdamW([{"params": [p], "method": "muon", "lr": 0.1}])
        with pytest.raises(ValueError):
            opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMM path is CUDA-only")
class StackedGroupedPathCudaTest:
    """On CUDA the grouped-GEMM path with stacked owners matches the per-expert one."""

    def test_grouped_forward_backward_match(self) -> None:
        reference = _build_pool(stacked=False).cuda()
        stacked = _build_pool(stacked=True).cuda()
        x, valid, gates, active, slot_groups = (t.cuda() for t in _inputs())
        for pool in (reference, stacked):
            pool.eval()
            assert pool._use_grouped_dispatch(x.reshape(-1, DIM))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_ref = sum(
                t for t in reference.forward_per_slot(x, valid, gates, active, slot_groups)[:2]
                if t is not None and t.dim() == 3
            ) + reference.forward_per_slot(x, valid, gates, active, slot_groups)[1].sum(2)
            out_new = sum(
                t for t in stacked.forward_per_slot(x, valid, gates, active, slot_groups)[:2]
                if t is not None and t.dim() == 3
            ) + stacked.forward_per_slot(x, valid, gates, active, slot_groups)[1].sum(2)
        assert torch.allclose(out_ref.float(), out_new.float(), rtol=2e-2, atol=2e-2)
        out_ref.float().pow(2).sum().backward()
        out_new.float().pow(2).sum().backward()
        up_ref = torch.stack([e.layers[0].weight.grad.t() for e in reference._private_bank()])
        assert torch.allclose(
            stacked.stacked_private.w_up.grad, up_ref, rtol=5e-2, atol=5e-2
        )


class NonStrictLoadGuardTest:
    """A non-strict load must not silently leave parameters at random init."""

    def test_missing_parameters_raise(self, tmp_path) -> None:
        import pytest as _pytest

        from nestor.train.checkpoint import load_checkpoint

        stacked = _build_pool(stacked=True)
        path = tmp_path / "model_stacked.pt"
        torch.save({"model": stacked.state_dict(), "step": 1}, path)
        plain = _build_pool(stacked=False)
        with _pytest.raises(RuntimeError, match="PARAMETERS are missing"):
            load_checkpoint(path, model=plain, strict=False)

    def test_matching_model_loads(self, tmp_path) -> None:
        from nestor.train.checkpoint import load_checkpoint

        source = _build_pool(stacked=True)
        path = tmp_path / "model_stacked.pt"
        torch.save({"model": source.state_dict(), "step": 1}, path)
        target = _build_pool(stacked=True, seed=4)
        load_checkpoint(path, model=target, strict=False)
        assert torch.allclose(_forward(target), _forward(source), atol=1e-6)
