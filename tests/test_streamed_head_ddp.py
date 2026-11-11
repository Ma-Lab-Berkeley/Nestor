"""The streamed head+CE under DDP: the head leaves the reducer, its gradient is
averaged by hand.

``NESTOR_STREAMED_HEAD_CE=1`` applies the LM head inside the loss, outside
``DDP.forward``, through a custom autograd function, so the head's parameters
cannot be tracked by DDP's reducer. As for the checkpointed in-loss head
(``tests/test_fused_head_ddp.py``),
:func:`nestor.utils.distributed.wrap_model_for_distributed` excludes them from
DDP and :func:`nestor.utils.distributed.reduce_ignored_grads` averages their
gradients across ranks once per step.

Pinned here on gloo with two processes and DIFFERENT batches per rank:

* the DDP wrapper ignores exactly the head's parameters;
* after backward + ``reduce_ignored_grads`` every rank holds, for the head AND
  for the DDP-reduced body, the mean of the two per-rank gradients that a plain
  (unwrapped, standard-path) model computes on each batch -- the reference
  implementation, i.e. exactly what DDP would have produced had the head run
  inside the model forward.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor


def _build_model():
    """Tiny causal grouped token-dispatch LM with an LM head, fixed seed 13."""

    from nestor.backbones import NestorConfig
    from nestor.configs.adapters import TextAdapterConfig
    from nestor.heads import LanguageModelingHeadConfig
    from nestor.models import Model, SequenceModelConfig

    config = SequenceModelConfig(
        adapter_config=TextAdapterConfig(embed_dim=32, vocab_size=64),
        backbone_config=NestorConfig(
            input_dim=32,
            num_blocks=1,
            feature_dims=[32],
            pooling_strides=[1],
            causal=True,
            block_num_layers=2,
            num_groups=[4],
            num_groups_per_token=[2],
            block_num_heads=2,
            block_head_dims=16,
            attention_token_dispatch=True,
            block_mlp_hidden_dims=[[24]],
            block_mlp_activations="gelu",
            router_num_experts_per_group=[1],
            router_num_shared_experts=[1],
            router_num_experts_per_group_per_token=1,
            router_score_mlp_hidden_dims=[[8]],
            router_noise_mlp_hidden_dims=[[8]],
        ),
        head_config=LanguageModelingHeadConfig(vocab_size=64),
    )
    torch.manual_seed(13)
    return Model.from_config(config).train()


def _batch(rank: int) -> Tensor:
    """A rank-specific token batch (so the two ranks' gradients differ).

    Args:
        rank (int): Rank whose batch to draw.

    Returns:
        Tensor: Token ids. Shape: ``(2, 40)``.
    """

    torch.manual_seed(200 + rank)
    return torch.randint(0, 64, (2, 40))


def _local_reference_grads(rank: int) -> dict[str, Tensor]:
    """Reference: plain (unwrapped) model, STANDARD head->loss path.

    Router noise is switched off so the two processes and the reference agree
    exactly; both in-loss flags are cleared for the duration of the call.

    Args:
        rank (int): Rank whose batch to use.

    Returns:
        dict[str, Tensor]: Gradient of every parameter on ``rank``'s batch.
    """

    from nestor.configs.losses import CausalLMLossConfig
    from nestor.train.losses import CausalLMLoss

    saved = {
        key: os.environ.pop(key, None)
        for key in ("NESTOR_STREAMED_HEAD_CE", "NESTOR_FUSED_HEAD_CE")
    }
    try:
        model = _build_model()
        for module in model.modules():  # no router noise: deterministic across procs
            if hasattr(module, "noise_mlp"):
                module.noise_mlp = None
        loss_fn = CausalLMLoss(CausalLMLossConfig())
        tokens = _batch(rank)
        out = model(tokens)
        assert "fused_head_module" not in out.aux, "reference must use real logits"
        loss_fn(out.pred, tokens, out.aux).total.backward()
        return {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _ddp_worker(rank: int, world_size: int, results: dict) -> None:
    """Wrap with DDP under the streamed flag, run one step, compare to the mean
    of the per-rank reference gradients. Module-level so ``mp.spawn`` can
    pickle it.

    Args:
        rank (int): This process's rank.
        world_size (int): Number of processes.
        results (dict): Manager dict; ``results["rank<r>"]`` is ``"OK"`` or an
            error description.
    """

    import torch.distributed as dist

    from nestor.configs.distributed import DistributedConfig
    from nestor.configs.losses import CausalLMLossConfig
    from nestor.train.losses import CausalLMLoss
    from nestor.utils.distributed import (
        fused_head_param_names,
        reduce_ignored_grads,
        wrap_model_for_distributed,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29553")
    os.environ["NESTOR_STREAMED_HEAD_CE"] = "1"
    os.environ["NESTOR_CE_CHUNK_ROWS"] = "16"
    os.environ["NESTOR_DDP"] = "1"
    os.environ["NESTOR_DDP_STATIC_GRAPH"] = "0"
    os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model = _build_model()
        for module in model.modules():
            if hasattr(module, "noise_mlp"):
                module.noise_mlp = None
        head_names = fused_head_param_names(model)
        ddp = wrap_model_for_distributed(
            model,
            DistributedConfig(enabled=True, backend="gloo"),
            device=torch.device("cpu"),
        )
        assert set(ddp.parameters_to_ignore) == set(head_names), (
            f"ignored {sorted(ddp.parameters_to_ignore)} != head {sorted(head_names)}"
        )
        assert head_names, "model has no head parameters"

        loss_fn = CausalLMLoss(CausalLMLossConfig())
        tokens = _batch(rank)
        out = ddp(tokens)
        assert "fused_head_module" in out.aux, "the streamed flag did not engage"
        loss = loss_fn(out.pred, tokens, out.aux).total
        sum_node = type(loss.grad_fn.next_functions[0][0]).__name__
        assert "StreamedHeadCE" in sum_node, sum_node
        loss.backward()
        reduce_ignored_grads(ddp)

        expected = _local_reference_grads(0)
        other = _local_reference_grads(1)
        for name in expected:
            expected[name] = (expected[name] + other[name]) / 2
        got = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert set(got) == set(expected), "gradient key sets differ"
        for name, ref in expected.items():
            torch.testing.assert_close(
                got[name], ref, rtol=1e-4, atol=1e-6,
                msg=lambda m, name=name: f"rank {rank} {name}: {m}",
            )
        # The head really is outside the reducer: its grads are the two-rank
        # mean only because reduce_ignored_grads averaged them.
        assert any(n.startswith("head.") for n in got)
        results[f"rank{rank}"] = "OK"
    except Exception as err:  # report instead of hanging the spawn
        results[f"rank{rank}"] = f"{type(err).__name__}: {err}"
    finally:
        dist.destroy_process_group()


class StreamedHeadDdpTest:
    """The streamed head is DDP-safe: ignored by the reducer, averaged by hand."""

    def test_head_ignored_and_gradients_match_two_rank_mean(self) -> None:
        """Two gloo ranks, different batches: every gradient is the rank mean."""

        import torch.multiprocessing as mp

        if not torch.distributed.is_available():
            pytest.skip("torch.distributed unavailable")
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(_ddp_worker, args=(2, results), nprocs=2, join=True)
        for rank in (0, 1):
            assert results.get(f"rank{rank}") == "OK", results.get(f"rank{rank}")
