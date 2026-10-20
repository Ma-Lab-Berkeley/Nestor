"""Equivalence tests for the fused Triton expert dispatch.

``NESTOR_TRITON_EXPERT_DISPATCH=1`` replaces the eager chain around the
grouped expert GEMMs — the dispatch gather, the SwiGLU on the fused up|gate
GEMM output, and the weighted slot combine — with fused Triton kernels
(``nestor/backbones/triton_expert.py``). The reference throughout is the
eager chain of ``grouped_expert_rows`` plus the call sites' weighting
(``grouped_expert_combine(..., triton_fused=False)``), held to the bar every
exact fast path meets: bf16 reassociation / transcendental-ulp noise on the
forward, relative-Frobenius on gradients.

Function-level classes pin each kernel pair against its eager chain on
synthetic layouts (alignment slack, rows not block-aligned, odd widths);
the end-to-end class runs the whole bank both ways under bf16 autocast in
every layout the pools use (fused gate on/off, stacked owners, Triton index
build, per-slot grouping); the pool-level class builds ``ExpertPoolMLP``
twins under the flag. CUDA + triton only (the flag-reader test runs anywhere).
"""

from __future__ import annotations

import copy
import os
from typing import Self

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from nestor.backbones.expert_bank_stack import (
    StackedExpertBank,
    consolidate_stacked_expert_banks,
)
from nestor.backbones.expert_pools import ExpertPoolMLP
from nestor.backbones.grouped_mm import (
    _CombineGather,
    _dispatch_plan,
    grouped_expert_combine,
)
from nestor.backbones.triton_expert import (
    bank_is_fusable,
    combine_rows,
    swiglu,
    swiglu_fused,
    triton_expert_available,
    triton_expert_requested,
)
from nestor.scaling.blocks import GatedScaledMLP

_CUDA = pytest.mark.skipif(
    not triton_expert_available(), reason="needs CUDA + triton"
)

# Two bf16 ulps of relative headroom (silu's exp may differ by an fp32 ulp
# from ATen's before the bf16 rounding; products are otherwise identical).
BF16_RTOL = 2.0**-6
BF16_ATOL = 1e-3


class _FlagEnv:
    """Set/restore real env flags around a block (no monkeypatching).

    Attributes:
        values (dict[str, str | None]): Flag -> value inside the block
            (``None`` unsets it).
        previous (dict[str, str | None]): Restored on exit.
    """

    def __init__(self, **values: str | None) -> None:
        self.values: dict[str, str | None] = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> Self:
        for key, val in self.values.items():
            self.previous[key] = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        return self

    def __exit__(self, *exc: object) -> None:
        for key, prev in self.previous.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _swiglu_bank(
    seed: int, num_experts: int, dim: int, hidden: int, bias: bool = False
) -> list[GatedScaledMLP]:
    """A SwiGLU expert bank on CUDA under a fixed seed (fp32 parameters).

    Args:
        seed (int): Parameter draw seed.
        num_experts (int): Bank size.
        dim (int): Input/output width.
        hidden (int): Expert hidden width.
        bias (bool): Up/down biases (the pools are bias-free).

    Returns:
        list[GatedScaledMLP]: The bank.
    """

    torch.manual_seed(seed)
    bank = [
        GatedScaledMLP(dim, [hidden], dim, activation="silu", bias=bias)
        for _ in range(num_experts)
    ]
    for expert in bank:
        for param in expert.parameters():
            with torch.no_grad():
                param.add_(0.3 * torch.randn_like(param))
    return [e.cuda() for e in bank]


def _routing(
    seed: int, num_tokens: int, slots: int, num_experts: int
) -> tuple[Tensor, Tensor]:
    """Random ``(token, slot)`` bank ids and combine weights (some exactly 0).

    Args:
        seed (int): RNG seed.
        num_tokens (int): Tokens.
        slots (int): Slots per token.
        num_experts (int): Bank size.

    Returns:
        tuple[Tensor, Tensor]: ``expert_ids`` ``[num_tokens, slots]`` (int64)
        and fp32 ``weights`` of the same shape, on CUDA.
    """

    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, num_experts, (num_tokens, slots), generator=gen)
    weights = torch.rand(num_tokens, slots, generator=gen)
    weights[torch.rand(num_tokens, slots, generator=gen) < 0.15] = 0.0
    return ids.cuda(), weights.cuda()


def _rel_close(actual: Tensor, expected: Tensor, tol: float, name: str) -> None:
    """Relative-Frobenius agreement with an absolute floor for tiny tensors."""

    err = (actual.float() - expected.float()).norm().item()
    ref = expected.float().norm().item()
    assert err <= tol * ref + 1e-6, f"{name}: ||diff|| {err:.4g} > {tol} x {ref:.4g}"


@_CUDA
class TritonSwiGLUFnTest:
    """Fused ``silu(gate) * up`` == the eager chain, values and grads."""

    def _check(self, rows: int, hidden: int, fused_layout: bool) -> None:
        torch.manual_seed(rows * 31 + hidden)
        if fused_layout:
            ug_ref = (torch.randn(rows, 2 * hidden, device="cuda") * 3).to(torch.bfloat16).requires_grad_(True)
            ug_new = ug_ref.detach().clone().requires_grad_(True)
            ref = F.silu(ug_ref[:, hidden:]) * ug_ref[:, :hidden]
            new = swiglu_fused(ug_new, hidden)
            inputs = [(ug_ref, ug_new)]
        else:
            up_ref = (torch.randn(rows, hidden, device="cuda") * 3).to(torch.bfloat16).requires_grad_(True)
            gate_ref = (torch.randn(rows, hidden, device="cuda") * 3).to(torch.bfloat16).requires_grad_(True)
            up_new = up_ref.detach().clone().requires_grad_(True)
            gate_new = gate_ref.detach().clone().requires_grad_(True)
            ref = F.silu(gate_ref) * up_ref
            new = swiglu(up_new, gate_new)
            inputs = [(up_ref, up_new), (gate_ref, gate_new)]
        assert new.dtype == ref.dtype and new.shape == ref.shape
        torch.testing.assert_close(new.float(), ref.float(), rtol=BF16_RTOL, atol=BF16_ATOL)
        grad = torch.randn_like(ref, dtype=torch.float32)
        (ref.float() * grad).sum().backward()
        (new.float() * grad).sum().backward()
        for t_ref, t_new in inputs:
            torch.testing.assert_close(
                t_new.grad.float(), t_ref.grad.float(), rtol=BF16_RTOL, atol=BF16_ATOL
            )

    def test_fused_layout(self) -> None:
        """The up|gate GEMM layout, rows not block-aligned, hidden not a multiple of 128."""

        self._check(rows=1000, hidden=200, fused_layout=True)

    def test_fused_layout_small_width(self) -> None:
        self._check(rows=37, hidden=12, fused_layout=True)

    def test_separate_layout(self) -> None:
        self._check(rows=1000, hidden=200, fused_layout=False)


@_CUDA
class TritonCatCastFnTest:
    """Fused cat+cast == ``torch.cat(...).to(bf16)`` bit for bit, grads too."""

    def test_values_and_contiguous_grads(self) -> None:
        from nestor.backbones.triton_expert import cat_cast

        torch.manual_seed(5)
        a_ref = torch.randn(6, 40, 24, device="cuda", requires_grad=True)
        b_ref = torch.randn(6, 40, 24, device="cuda", requires_grad=True)
        a_new = a_ref.detach().clone().requires_grad_(True)
        b_new = b_ref.detach().clone().requires_grad_(True)
        ref = torch.cat([a_ref, b_ref], dim=-1).to(torch.bfloat16)
        new = cat_cast(a_new, b_new, torch.bfloat16)
        assert new.shape == ref.shape and new.dtype == ref.dtype and new.is_contiguous()
        assert torch.equal(new, ref)
        cot = torch.randn_like(ref, dtype=torch.float32)
        (ref.float() * cot).sum().backward()
        (new.float() * cot).sum().backward()
        assert torch.equal(a_new.grad, a_ref.grad) and torch.equal(b_new.grad, b_ref.grad)
        assert a_new.grad.is_contiguous() and b_new.grad.is_contiguous()


@_CUDA
class TritonCombineRowsFnTest:
    """Fused combine == gather + weight + cast + group-sum, values and grads."""

    def _check(self, num_tokens: int, slots: int, group: int, num_banks: int, dim: int, out_dtype: torch.dtype) -> None:
        ids, weights = _routing(seed=num_tokens + slots, num_tokens=num_tokens, slots=slots, num_experts=num_banks)
        plan = _dispatch_plan(ids.reshape(-1), num_banks, slots, torch.bfloat16, triton_index=False)
        torch.manual_seed(7)
        y_ref = (torch.randn(plan.rows_pad, dim, device="cuda") * 2).to(torch.bfloat16).requires_grad_(True)
        y_new = y_ref.detach().clone().requires_grad_(True)
        w_ref = weights.reshape(-1).clone().requires_grad_(True)
        w_new = weights.reshape(-1).clone().requires_grad_(True)

        rows = _CombineGather.apply(y_ref, plan.row_dest, plan.pad_src_row, plan.pad_valid)
        ref = (rows * w_ref.reshape(-1, 1)).to(out_dtype).view(num_tokens, slots // group, group, dim).sum(2)
        new = combine_rows(
            y_new, plan.row_dest, plan.pad_src_row, plan.pad_valid, w_new, group, out_dtype
        ).view(num_tokens, slots // group, dim)
        assert new.dtype == out_dtype
        tol = {"rtol": 1e-5, "atol": 1e-5} if out_dtype == torch.float32 else {"rtol": BF16_RTOL, "atol": BF16_ATOL}
        torch.testing.assert_close(new.float(), ref.float(), **tol)

        grad = torch.randn_like(ref, dtype=torch.float32)
        (ref.float() * grad).sum().backward()
        (new.float() * grad).sum().backward()
        torch.testing.assert_close(y_new.grad.float(), y_ref.grad.float(), rtol=BF16_RTOL, atol=1e-6)
        slack = plan.pad_valid == 0
        assert slack.any() and (y_new.grad[slack] == 0).all()
        torch.testing.assert_close(w_new.grad, w_ref.grad, rtol=1e-4, atol=1e-4)

    def test_full_combine_fp32(self) -> None:
        """group == slots: the plain weighted sum (public pool / native bank)."""

        self._check(num_tokens=300, slots=2, group=2, num_banks=8, dim=40, out_dtype=torch.float32)

    def test_per_slot_combine_fp32(self) -> None:
        """group == k_q < slots: the private pools' per-attention-slot layout."""

        self._check(num_tokens=257, slots=6, group=2, num_banks=24, dim=40, out_dtype=torch.float32)

    def test_bf16_out(self) -> None:
        self._check(num_tokens=300, slots=4, group=4, num_banks=8, dim=200, out_dtype=torch.bfloat16)


@_CUDA
class GroupedExpertCombineTest:
    """``grouped_expert_combine`` fused == eager for every pool layout."""

    def _run(
        self,
        bank: list[GatedScaledMLP],
        x: Tensor,
        ids: Tensor,
        weights: Tensor,
        group: int,
        *,
        triton_fused: bool,
        fused_gate: bool,
        stacked: StackedExpertBank | None,
        triton_index: bool,
    ) -> tuple[Tensor, list[Tensor]]:
        x = x.detach().clone().requires_grad_(True)
        weights = weights.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = grouped_expert_combine(
                x, bank, ids, weights, group,
                fused_gate=fused_gate, stacked=stacked,
                triton_index=triton_index, triton_fused=triton_fused,
            )
        torch.manual_seed(3)
        (out.float() * torch.randn_like(out, dtype=torch.float32)).sum().backward()
        params = list(stacked.parameters()) if stacked is not None else [p for e in bank for p in e.parameters()]
        return out.detach(), [x.grad, weights.grad, *[p.grad for p in params]]

    def _check(self, *, fused_gate: bool, stacked: bool, triton_index: bool, group: int, slots: int) -> None:
        dim, hidden, num_experts, num_tokens = 40, 24, 6, 150
        bank_ref = _swiglu_bank(seed=5, num_experts=num_experts, dim=dim, hidden=hidden)
        bank_new = copy.deepcopy(bank_ref)
        stack_ref = StackedExpertBank(bank_ref) if stacked else None
        stack_new = StackedExpertBank(bank_new) if stacked else None
        torch.manual_seed(11)
        x = torch.randn(num_tokens, dim, device="cuda")
        ids, weights = _routing(seed=8, num_tokens=num_tokens, slots=slots, num_experts=num_experts)
        out_ref, grads_ref = self._run(
            bank_ref, x, ids, weights, group,
            triton_fused=False, fused_gate=fused_gate, stacked=stack_ref, triton_index=triton_index,
        )
        out_new, grads_new = self._run(
            bank_new, x, ids, weights, group,
            triton_fused=True, fused_gate=fused_gate, stacked=stack_new, triton_index=triton_index,
        )
        assert out_new.shape == (num_tokens, slots // group, dim) and out_new.dtype == torch.float32
        torch.testing.assert_close(out_new, out_ref, rtol=BF16_RTOL, atol=1e-2)
        assert len(grads_ref) == len(grads_new)
        for i, (g_ref, g_new) in enumerate(zip(grads_ref, grads_new)):
            assert g_ref is not None and g_new is not None, i
            _rel_close(g_new, g_ref, 2e-2, f"grad {i}")

    def test_fused_gate_per_expert(self) -> None:
        self._check(fused_gate=True, stacked=False, triton_index=False, group=2, slots=2)

    def test_unfused_gate_per_expert(self) -> None:
        self._check(fused_gate=False, stacked=False, triton_index=False, group=2, slots=2)

    def test_fused_gate_stacked_triton_index_per_slot(self) -> None:
        """The production layout: stacked owners, fused gate, Triton index build, k_q grouping."""

        self._check(fused_gate=True, stacked=True, triton_index=True, group=2, slots=6)

    def test_biased_bank_falls_back_bit_exactly(self) -> None:
        """A biased bank is not fusable: the flag-on call IS the eager chain."""

        bank = _swiglu_bank(seed=2, num_experts=4, dim=16, hidden=8, bias=True)
        assert not bank_is_fusable(bank, fused_gate=True)
        torch.manual_seed(1)
        x = torch.randn(50, 16, device="cuda")
        ids, weights = _routing(seed=4, num_tokens=50, slots=2, num_experts=4)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            off = grouped_expert_combine(x, bank, ids, weights, 2, fused_gate=True, triton_fused=False)
            on = grouped_expert_combine(x, bank, ids, weights, 2, fused_gate=True, triton_fused=True)
        assert torch.equal(on, off)


def _build_pool(seed: int, **env: str | None) -> ExpertPoolMLP:
    """A small hybrid pool on CUDA built under ``env`` flags (read at construction)."""

    with _FlagEnv(**env):
        torch.manual_seed(seed)
        pool = ExpertPoolMLP(
            dim=40,
            num_groups=4,
            num_groups_per_token=2,
            num_public_experts=4,
            public_top_k=2,
            num_private_experts=3,
            private_top_k=2,
            router_score_mlp_hidden_dims=[],
            router_score_mlp_activation="gelu",
            router_noise_mlp_hidden_dims=[],
            router_noise_mlp_activation="gelu",
            private_gate_mode="switch",
        )
        consolidate_stacked_expert_banks(pool)
    pool.grouped_dispatch = True
    return pool.cuda().eval()


def _pool_inputs(seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """A batch, its group gates/membership and slot groups (top-2 of 4) on CUDA."""

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 70, 40, generator=gen)
    valid = torch.ones(3, 70, dtype=torch.bool)
    scores = torch.rand(3, 70, 4, generator=gen)
    slot_groups = scores.topk(2, dim=-1).indices
    active = torch.zeros(3, 70, 4, dtype=torch.bool).scatter(-1, slot_groups, True)
    gates = torch.zeros(3, 70, 4).scatter(-1, slot_groups, 0.5)
    return tuple(t.cuda() for t in (x, valid, gates, active, slot_groups))


@_CUDA
class ExpertPoolTritonFusedTest:
    """``ExpertPoolMLP`` flag twins agree on ``forward_per_slot`` and grads."""

    def _run(self, pool: ExpertPoolMLP) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        x, valid, gates, active, slot_groups = _pool_inputs(seed=21)
        x = x.requires_grad_(True)
        pool.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            public, private, _, _ = pool.forward_per_slot(x, valid, gates, active, slot_groups)
        torch.manual_seed(22)
        loss = (public.float() * torch.randn_like(public, dtype=torch.float32)).sum()
        loss = loss + (private.float() * torch.randn_like(private, dtype=torch.float32)).sum()
        loss.backward()
        grads = {n: p.grad.detach().float().clone() for n, p in pool.named_parameters() if p.grad is not None}
        return public.detach().float(), private.detach().float(), grads | {"__x__": x.grad.detach().float()}

    def _check(self, stacked: str) -> None:
        base = {"NESTOR_STACKED_EXPERT_BANKS": stacked, "NESTOR_GROUPED_MM_FUSED_GATE": "1", "NESTOR_TRITON_INDEX_BUILD": "1"}
        ref = _build_pool(seed=9, **base, NESTOR_TRITON_EXPERT_DISPATCH="0")
        new = _build_pool(seed=9, **base, NESTOR_TRITON_EXPERT_DISPATCH="1")
        assert not ref.triton_fused and new.triton_fused
        new.load_state_dict(ref.state_dict())
        pub_ref, priv_ref, g_ref = self._run(ref)
        pub_new, priv_new, g_new = self._run(new)
        torch.testing.assert_close(pub_new, pub_ref, rtol=BF16_RTOL, atol=1e-2)
        torch.testing.assert_close(priv_new, priv_ref, rtol=BF16_RTOL, atol=1e-2)
        assert g_ref.keys() == g_new.keys()
        for name in g_ref:
            _rel_close(g_new[name], g_ref[name], 3e-2, name)

    def test_stacked_banks(self) -> None:
        self._check(stacked="1")

    def test_per_expert_banks(self) -> None:
        self._check(stacked="0")


class _Group(torch.nn.Module):
    """A stand-in attention group: q/k/v/out projections with ``weight``."""

    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, inner, bias=False)
        self.k_proj = torch.nn.Linear(dim, inner, bias=False)
        self.v_proj = torch.nn.Linear(dim, inner, bias=False)
        self.out_proj = torch.nn.Linear(inner, dim, bias=False)


class BatchedHeadpadStacksTest:
    """``batched=True`` builds the SAME head-padded stacks and grads (CPU)."""

    def _check(self, head_dim: int) -> None:
        from nestor.backbones.triton_packed import stack_wo_headpad, stack_wqkv_headpad

        dim, num_heads = 24, 3
        hdp = head_dim + ((-head_dim) % 8)
        torch.manual_seed(head_dim)
        groups_ref = [_Group(dim, num_heads * head_dim) for _ in range(4)]
        groups_new = copy.deepcopy(groups_ref)
        for builder in (stack_wqkv_headpad, stack_wo_headpad):
            ref = builder(groups_ref, dim, num_heads, head_dim, hdp, torch.bfloat16)
            new = builder(groups_new, dim, num_heads, head_dim, hdp, torch.bfloat16, batched=True)
            assert new.shape == ref.shape and new.is_contiguous()
            assert torch.equal(new, ref)
            torch.manual_seed(1)
            cot = torch.randn_like(ref, dtype=torch.float32)
            (ref.float() * cot).sum().backward()
            (new.float() * cot).sum().backward()
        for g_ref, g_new in zip(groups_ref, groups_new):
            for (name, p_ref), (_, p_new) in zip(g_ref.named_parameters(), g_new.named_parameters()):
                assert p_ref.grad is not None and torch.equal(p_new.grad, p_ref.grad), name

    def test_padded_head_dim(self) -> None:
        self._check(head_dim=10)

    def test_aligned_head_dim(self) -> None:
        self._check(head_dim=8)


@_CUDA
class TritonExpertDispatchModelTest:
    """A column model under the expert-dispatch flag == the packed reference."""

    def test_model_matches_packed_reference(self) -> None:
        from tests.test_triton_packed_attn import _assert_agree, _build, _config

        packed = {
            "NESTOR_ATTN_PACKED_VARLEN": "1",
            "NESTOR_COLUMN_SLOT_UPDATES": "1",
            "NESTOR_TRITON_PACKED_ATTN": "1",
            "NESTOR_GROUPED_MM_FUSED_GATE": "1",
            "NESTOR_TRITON_INDEX_BUILD": "1",
        }
        cfg = _config(num_heads=3, head_dim=10)
        reference = _build(cfg, **packed, NESTOR_TRITON_EXPERT_DISPATCH="0")
        candidate = _build(cfg, **packed, NESTOR_TRITON_EXPERT_DISPATCH="1")
        assert all(layer._batched_attn_stacks for block in candidate.blocks for layer in block.layers)
        assert all(
            m.triton_fused for m in candidate.modules() if isinstance(m, ExpertPoolMLP)
        )
        _assert_agree(reference, candidate, seed=23)


class FlagReaderTest:
    """The flag follows ``NESTOR_TRITON_INDEX_BUILD`` unless set explicitly."""

    def test_default_follows_index_build(self) -> None:
        with _FlagEnv(NESTOR_TRITON_EXPERT_DISPATCH=None, NESTOR_TRITON_INDEX_BUILD=None):
            assert not triton_expert_requested()
        with _FlagEnv(NESTOR_TRITON_EXPERT_DISPATCH=None, NESTOR_TRITON_INDEX_BUILD="1"):
            assert triton_expert_requested()
        with _FlagEnv(NESTOR_TRITON_EXPERT_DISPATCH="0", NESTOR_TRITON_INDEX_BUILD="1"):
            assert not triton_expert_requested()
        with _FlagEnv(NESTOR_TRITON_EXPERT_DISPATCH="1", NESTOR_TRITON_INDEX_BUILD=None):
            assert triton_expert_requested()
