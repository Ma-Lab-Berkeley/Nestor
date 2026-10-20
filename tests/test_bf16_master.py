"""bf16 Muon leaves with fp32 masters and fp32 gradient accumulators.

Reference = the fp32-parameter path: an fp32 model trained under bf16 autocast
(so its forward consumes bf16 casts of its fp32 parameters and accumulates
fp32 gradients across the window) with the unsharded ``MuonAdamW``. The
converted twin must (1) keep masters equal to the reference parameters and
leaves equal to their bf16 roundings after every step, with the same clipped
gradient norm; (2) under two gloo ranks with a real DDP wrapper (the converted
parameters on its ignore list) leave every rank with identical leaves that
match the single-process reference after reduce-to-owners + shard-aware
clipping + the owners' bf16 broadcast; (3) consolidate the masters into an
fp32 checkpoint (``fp32_model_state``) and restore them on load.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nestor.scaling.bf16_master import (
    ACC_ATTR,
    MASTER_ATTR,
    clip_grad_norm_,
    convert_muon_params_to_bf16,
    fp32_model_state,
    is_bf16_master_param,
    reduce_to_owners,
)
from nestor.scaling.blocks import ParamScaling, set_scaling
from nestor.scaling.muon import MuonAdamW

STEPS = 3
ACCUM = 2
BATCH = 8
MAX_NORM = 0.5


def _find_free_port() -> int:
    """A free localhost TCP port for the gloo rendezvous."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _TinyMixture(nn.Module):
    """Four same-shape Muon 'experts', two Muon projections, an AdamW embedding and bias.

    Attributes:
        experts (nn.ParameterList): Four ``[16, 24]`` Muon matrices (one
            shape bucket -> two per owner under two ranks).
        proj_a (nn.Parameter): ``[24, 32]`` Muon matrix.
        proj_b (nn.Parameter): ``[32, 8]`` Muon matrix.
        embed (nn.Parameter): ``[10, 16]`` AdamW table.
        bias (nn.Parameter): ``[8]`` AdamW bias.
    """

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ParameterList(
            [nn.Parameter(torch.randn(16, 24) * 0.2) for _ in range(4)]
        )
        self.proj_a = nn.Parameter(torch.randn(24, 32) * 0.2)
        self.proj_b = nn.Parameter(torch.randn(32, 8) * 0.2)
        self.embed = nn.Parameter(torch.randn(10, 16) * 0.5)
        self.bias = nn.Parameter(torch.zeros(8))
        for p in [*self.experts, self.proj_a, self.proj_b]:
            set_scaling(p, ParamScaling("muon", muon_scale=1.0))
        set_scaling(self.embed, ParamScaling("adamw", weight_decay=False))
        set_scaling(self.bias, ParamScaling("adamw", weight_decay=False))

    def forward(self, idx: Tensor) -> Tensor:
        """``[batch]`` token ids -> ``[batch, 8]`` (matmuls under autocast)."""

        h = self.embed[idx]  # [batch, 16] fp32
        mixed = sum(torch.tanh(torch.nn.functional.linear(h, e.t())) for e in self.experts)
        out = torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(mixed, self.proj_a.t())), self.proj_b.t())
        return out.float() + self.bias


def _build_optimizer(model: nn.Module, shard: bool) -> MuonAdamW:
    """Muon on the matrices, AdamW on the embedding and bias."""

    muon = list(model.experts) + [model.proj_a, model.proj_b]
    adamw = [model.embed, model.bias]
    return MuonAdamW(
        [
            {"params": muon, "method": "muon", "lr": 0.02, "weight_decay": 0.01,
             "momentum": 0.95, "nesterov": True, "ns_steps": 5},
            {"params": adamw, "method": "adamw", "lr": 0.01, "betas": (0.9, 0.95),
             "eps": 1e-8, "weight_decay": 0.0},
        ],
        shard=shard,
    )


def _data() -> list[list[tuple[Tensor, Tensor]]]:
    """``STEPS`` windows of ``ACCUM`` fixed micro-batches."""

    gen = torch.Generator().manual_seed(1234)
    return [
        [
            (torch.randint(0, 10, (BATCH,), generator=gen), torch.randn(BATCH, 8, generator=gen))
            for _ in range(ACCUM)
        ]
        for _ in range(STEPS)
    ]


def _loss(model: nn.Module, idx: Tensor, target: Tensor) -> Tensor:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = model(idx)
    return (pred.float() - target).pow(2).mean() / ACCUM


def _reference() -> tuple[dict[str, Tensor], list[float]]:
    """The fp32-parameter path: params after every step + clipped norms."""

    torch.manual_seed(7)
    model = _TinyMixture()
    optimizer = _build_optimizer(model, shard=False)
    norms: list[float] = []
    for window in _data():
        for idx, target in window:
            _loss(model, idx, target).backward()
        norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {n: p.detach().clone() for n, p in model.named_parameters()}, norms


class Bf16MasterSingleProcessTest:
    """Converted twin == fp32 reference: masters, leaves, clipped norms, resume."""

    def test_masters_track_the_fp32_reference(self) -> None:
        ref_params, ref_norms = _reference()
        torch.manual_seed(7)
        model = _TinyMixture()
        names = convert_muon_params_to_bf16(model)
        assert len(names) == 6 and all("embed" not in n and "bias" not in n for n in names)
        for p in [*model.experts, model.proj_a, model.proj_b]:
            assert p.dtype == torch.bfloat16 and is_bf16_master_param(p)
        optimizer = _build_optimizer(model, shard=False)
        for p in [*model.experts, model.proj_a, model.proj_b]:
            assert optimizer.state[p]["master"].dtype == torch.float32
        norms: list[float] = []
        for window in _data():
            for idx, target in window:
                _loss(model, idx, target).backward()
                # The hook folded the bf16 grad into the fp32 accumulator.
                assert model.proj_a.grad is None and model.proj_a._acc_touched
            reduce_to_owners(optimizer)  # no-op unsharded
            norms.append(float(clip_grad_norm_(model, optimizer, MAX_NORM)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for p in [*model.experts, model.proj_a, model.proj_b]:
                assert torch.all(getattr(p, ACC_ATTR) == 0)
                assert torch.equal(p.detach(), optimizer.state[p]["master"].to(torch.bfloat16))
        for got, want in zip(norms, ref_norms):
            assert abs(got - want) < 1e-5 * max(1.0, want), (got, want)
        muon_masters = {n: optimizer.state[p]["master"] for n, p in model.named_parameters() if is_bf16_master_param(p)}
        assert len(muon_masters) == 6
        for name, master in muon_masters.items():
            torch.testing.assert_close(master, ref_params[name], rtol=1e-6, atol=1e-6)
        for name in ("embed", "bias"):
            torch.testing.assert_close(dict(model.named_parameters())[name].detach(), ref_params[name], rtol=1e-6, atol=1e-6)
        # fp32 checkpoint format and resume.
        state = fp32_model_state(model, model.state_dict())
        assert all(v.dtype == torch.float32 for v in state.values())
        torch.manual_seed(7)
        fresh = _TinyMixture()
        convert_muon_params_to_bf16(fresh)
        fresh.load_state_dict(state)
        fresh_opt = _build_optimizer(fresh, shard=False)
        fresh_opt.load_state_dict(optimizer.state_dict())
        for p, q in zip(fresh.parameters(), model.parameters()):
            assert torch.equal(p.detach(), q.detach())
        for name, p in fresh.named_parameters():
            if is_bf16_master_param(p):
                assert torch.equal(fresh_opt.state[p]["master"], muon_masters[name])


def _worker(rank: int, port: int, queue: mp.SimpleQueue) -> None:
    """One DDP rank in bf16-master mode; reports leaves, masters and the checkpoint."""

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        model = _TinyMixture()
        names = convert_muon_params_to_bf16(model)
        DDP._set_params_and_buffers_to_ignore_for_model(model, names)
        ddp = DDP(model)
        optimizer = _build_optimizer(model, shard=True)
        assert optimizer._shard
        muon = [*model.experts, model.proj_a, model.proj_b]
        assert all((MASTER_ATTR in p.__dict__) == optimizer._owns(p) for p in muon)
        norms: list[float] = []
        for window in _data():
            for i, (idx, target) in enumerate(window):
                ctx = ddp.no_sync() if i < ACCUM - 1 else torch.enable_grad()
                with ctx:
                    _loss(ddp, idx, target).backward()
            reduce_to_owners(optimizer)
            norms.append(float(clip_grad_norm_(model, optimizer, MAX_NORM)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        optimizer.consolidate_state_dict(to=0)
        # Model state FIRST (the consolidated masters are attached to the
        # parameters until optimizer.state_dict() releases them) -- the order
        # save_checkpoint uses.
        state = fp32_model_state(model, model.state_dict()) if rank == 0 else None
        saved = optimizer.state_dict() if rank == 0 else None
        queue.put((
            rank,
            [p.detach().float().numpy().copy() for p in model.parameters()],
            norms,
            None if saved is None else {i: saved["state"][i]["master"].numpy().copy() for i in range(6)},
            None if state is None else {k: v.numpy().copy() for k, v in state.items()},
        ))
    finally:
        dist.destroy_process_group()


class Bf16MasterTwoRanksTest:
    """Two gloo ranks: replica agreement, reference match, fp32 checkpoint."""

    def test_two_ranks_match_reference(self) -> None:
        ref_params, ref_norms = _reference()
        port = _find_free_port()
        queue: mp.SimpleQueue = mp.get_context("spawn").SimpleQueue()
        mp.spawn(fn=_worker, args=(port, queue), nprocs=2, join=True)
        results = {}
        while len(results) < 2:
            rank, params, norms, masters, state = queue.get()
            results[rank] = (params, norms, masters, state)
        p0, n0, masters, state = results[0]
        p1, n1, _, _ = results[1]
        for a, b in zip(p0, p1):
            assert (a == b).all()                       # replicas agree bit for bit
        for got, want in zip(n0, ref_norms):
            assert abs(got - want) < 1e-5 * max(1.0, want), (got, want)
        assert n0 == n1
        muon_names = [f"experts.{i}" for i in range(4)] + ["proj_a", "proj_b"]  # the Muon group's order
        for i, name in enumerate(muon_names):
            torch.testing.assert_close(torch.from_numpy(masters[i]), ref_params[name], rtol=1e-6, atol=1e-6)
        for value in state.values():
            assert value.dtype.name == "float32"
        for i, name in enumerate(muon_names):
            assert (state[name] == masters[i]).all()
