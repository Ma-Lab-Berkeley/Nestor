"""Sharded Muon update (``MuonAdamW(shard=True)`` / ``NESTOR_MUON_SHARD=1``).

Reference = the unsharded optimizer stepping the same model on the same data
in one process. Sharded, a two-rank gloo group with a real DDP wrapper (no
mocks) must (1) leave every rank with bit-identical parameters (the owners'
broadcast), (2) match the reference within bf16 Newton--Schulz noise,
(3) keep momentum only for owned parameters, and (4) consolidate to the
unsharded checkpoint format and load back pruned.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nestor.scaling.muon import MuonAdamW

STEPS = 4
BATCH = 8


def _find_free_port() -> int:
    """Find a free localhost TCP port for the gloo rendezvous."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _TinyMixture(nn.Module):
    """Six same-shape 'experts', two singleton projections, an embedding, a bias.

    Attributes:
        experts (nn.ParameterList): Six ``[16, 24]`` matrices (one Muon
            shape bucket, so the batched Newton--Schulz path is exercised and
            the round-robin owner split lands three per rank).
        proj_a (nn.Parameter): ``[24, 32]`` singleton Muon matrix.
        proj_b (nn.Parameter): ``[32, 8]`` singleton Muon matrix.
        embed (nn.Parameter): ``[10, 16]`` AdamW table.
        bias (nn.Parameter): ``[8]`` AdamW bias.
    """

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ParameterList(
            [nn.Parameter(torch.randn(16, 24) * 0.2) for _ in range(6)]
        )
        self.proj_a = nn.Parameter(torch.randn(24, 32) * 0.2)
        self.proj_b = nn.Parameter(torch.randn(32, 8) * 0.2)
        self.embed = nn.Parameter(torch.randn(10, 16) * 0.5)
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, idx: Tensor) -> Tensor:
        """``[batch]`` token ids -> ``[batch, 8]`` outputs."""

        h = self.embed[idx]  # [batch, 16]
        mixed = sum(torch.tanh(h @ e) for e in self.experts)  # [batch, 24]
        return torch.tanh(mixed @ self.proj_a) @ self.proj_b + self.bias


def _build_optimizer(model: nn.Module, shard: bool) -> MuonAdamW:
    """Muon on the matrices, AdamW on the embedding and bias."""

    muon = list(model.experts) + [model.proj_a, model.proj_b]
    adamw = [model.embed, model.bias]
    return MuonAdamW(
        [
            {
                "params": muon, "method": "muon", "lr": 0.02,
                "weight_decay": 0.01, "momentum": 0.95, "nesterov": True,
                "ns_steps": 5,
            },
            {
                "params": adamw, "method": "adamw", "lr": 0.01,
                "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            },
        ],
        shard=shard,
    )


def _data() -> list[tuple[Tensor, Tensor]]:
    """Fixed (ids, targets) per step, identical everywhere it is called."""

    gen = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randint(0, 10, (BATCH,), generator=gen),
            torch.randn(BATCH, 8, generator=gen),
        )
        for _ in range(STEPS)
    ]


def _run(model: nn.Module, optimizer: MuonAdamW) -> None:
    """Train ``STEPS`` steps of MSE on the fixed data."""

    for idx, target in _data():
        loss = (model(idx) - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _reference() -> tuple[list[Tensor], MuonAdamW, _TinyMixture]:
    """Unsharded single-process run: final parameters + the optimizer."""

    torch.manual_seed(7)
    model = _TinyMixture()
    optimizer = _build_optimizer(model, shard=False)
    _run(model, optimizer)
    return [p.detach().clone() for p in model.parameters()], optimizer, model


def _shard_worker(rank: int, port: int, queue: mp.SimpleQueue) -> None:
    """One DDP rank: sharded Muon, then consolidate / save / reload."""

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(7)
        model = _TinyMixture()
        ddp = DDP(model)
        optimizer = _build_optimizer(model, shard=True)
        assert optimizer._shard, "two gloo ranks should activate sharding"
        for idx, target in _data():
            loss = (ddp(idx) - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        owned = [p for p in optimizer._owned[rank]]
        muon_params = list(model.experts) + [model.proj_a, model.proj_b]
        stated = [p for p in muon_params if "momentum_buffer" in optimizer.state.get(p, {})]
        # Consolidate (collective) then state_dict on rank 0 only, as the
        # checkpoint code does.
        optimizer.consolidate_state_dict(to=0)
        saved = optimizer.state_dict() if rank == 0 else None
        if saved is not None:
            # Broadcast the saved state to rank 1 so it can test loading it.
            payload = [saved]
        else:
            payload = [None]
        dist.broadcast_object_list(payload, src=0)
        loaded_state_count = None
        loaded_matches_owned = None
        if rank == 1:
            fresh = _build_optimizer(model, shard=True)
            fresh.load_state_dict(payload[0])
            loaded_state_count = sum(
                1 for p in muon_params if "momentum_buffer" in fresh.state.get(p, {})
            )
            loaded_matches_owned = all(
                torch.equal(
                    fresh.state[p]["momentum_buffer"],
                    optimizer.state[p]["momentum_buffer"],
                )
                for p in fresh._owned[1]
            )
        # Ship numpy copies: a shared-memory tensor put on the queue can be
        # freed with the worker before the parent reads it.
        queue.put(
            (
                rank,
                [p.detach().numpy().copy() for p in model.parameters()],
                len(owned),
                len(stated),
                # rank 0's consolidated momentum per Muon param (index order)
                (
                    [
                        saved["state"][i]["momentum_buffer"].numpy().copy()
                        for i in range(len(muon_params))
                    ]
                    if saved is not None
                    else None
                ),
                loaded_state_count,
                loaded_matches_owned,
                # after state_dict() the injected foreign momentum is released
                sum(1 for p in muon_params if "momentum_buffer" in optimizer.state.get(p, {})),
            )
        )
    finally:
        dist.destroy_process_group()


class MuonShardSingleProcessTest:
    """Without a process group ``shard=True`` is exactly the unsharded optimizer."""

    def test_identical_without_process_group(self) -> None:
        ref_params, ref_opt, _ = _reference()
        torch.manual_seed(7)
        model = _TinyMixture()
        optimizer = _build_optimizer(model, shard=True)
        assert not optimizer._shard
        _run(model, optimizer)
        for p, q in zip(model.parameters(), ref_params):
            assert torch.equal(p.detach(), q)
        optimizer.consolidate_state_dict(to=0)  # no-op
        sd = optimizer.state_dict()
        ref_sd = ref_opt.state_dict()
        assert sd["state"].keys() == ref_sd["state"].keys()
        for key in sd["state"]:
            for name, value in sd["state"][key].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, ref_sd["state"][key][name])

    def test_env_flag_reads_at_construction(self) -> None:
        os.environ["NESTOR_MUON_SHARD"] = "1"
        try:
            torch.manual_seed(7)
            optimizer = _build_optimizer(_TinyMixture(), shard=None)
        finally:
            del os.environ["NESTOR_MUON_SHARD"]
        # No process group -> requested but inactive; every param owned here.
        assert not optimizer._shard
        assert all(optimizer._owns(p) for p in optimizer._owner)


class MuonShardTwoRanksTest:
    """Two gloo ranks under DDP: exact replica agreement, reference match, checkpoint format."""

    def test_two_rank_sharded_matches_reference(self) -> None:
        ref_params, ref_opt, ref_model = _reference()
        port = _find_free_port()
        ctx = mp.get_context("spawn")
        queue: mp.SimpleQueue = ctx.SimpleQueue()
        mp.spawn(fn=_shard_worker, args=(port, queue), nprocs=2, join=True)
        results = {r[0]: r for r in (queue.get() for _ in range(2))}
        params0, owned0, stated0, momentum0, _, _, after0 = results[0][1:]
        params1, owned1, stated1, _, loaded_count, loaded_ok, after1 = results[1][1:]
        params0 = [torch.from_numpy(a) for a in params0]
        params1 = [torch.from_numpy(a) for a in params1]
        momentum0 = [torch.from_numpy(a) for a in momentum0]

        # (1) The owners' broadcast leaves both replicas bit-identical.
        for a, b in zip(params0, params1):
            assert torch.equal(a, b)
        # (2) Same math as the unsharded reference (bf16 Newton--Schulz noise).
        for a, ref in zip(params0, ref_params):
            assert torch.allclose(a, ref, rtol=1e-3, atol=1e-4), (a - ref).abs().max()
        # (3) Eight Muon matrices dealt 4/4; momentum kept only for owned ones.
        assert owned0 == 4 and owned1 == 4
        assert stated0 == 4 and stated1 == 4
        assert after0 == 4 and after1 == 4  # consolidation released again
        # (4) Consolidated state carries every momentum, matching the reference.
        muon_ref = list(ref_model.experts) + [ref_model.proj_a, ref_model.proj_b]
        assert len(momentum0) == 8
        for buf, p in zip(momentum0, muon_ref):
            assert torch.allclose(
                buf, ref_opt.state[p]["momentum_buffer"], rtol=1e-3, atol=1e-4
            )
        # Loading the unsharded state into a sharded optimizer keeps its shard.
        assert loaded_count == 4
        assert loaded_ok
