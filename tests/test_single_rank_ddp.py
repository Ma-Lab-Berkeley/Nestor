"""One rank: ``NESTOR_DDP=1`` leaves the model bare.

With a process group of a single rank (``torchrun --nproc_per_node=1``) a
``DistributedDataParallel`` wrapper has no peer, but still copies every
gradient into the reducer's buckets, all-reduces it with itself, broadcasts
the buffers and makes ``torch.compile`` split its graphs at bucket boundaries.
:func:`nestor.utils.distributed.wrap_model_for_distributed` therefore returns
the bare model on one rank; ``NESTOR_DDP_SINGLE_RANK=1`` keeps the wrapper.

The tests run a real one-process gloo group and real wrappers (no mocks):

* the bare model comes back on one rank, the DDP wrapper when forced;
* nothing is skipped without ``NESTOR_DDP=1`` or without a process group;
* the bare model's gradients equal the single-rank DDP wrapper's -- the
  reference implementation, i.e. the behaviour before the skip existed.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from nestor.configs.distributed import DistributedConfig
from nestor.configs.losses import CausalLMLossConfig
from nestor.train.losses import CausalLMLoss
from nestor.utils.distributed import (
    single_rank_ddp_skipped,
    wrap_model_for_distributed,
)

_FLAGS: tuple[str, ...] = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "NESTOR_DDP",
    "NESTOR_DDP_SINGLE_RANK",
    "NESTOR_STREAMED_HEAD_CE",
    "NESTOR_FUSED_HEAD_CE",
)


@contextlib.contextmanager
def _env(**values: str | None) -> Iterator[None]:
    """Set (or, with ``None``, unset) env flags; restore every flag afterwards.

    Args:
        **values (str | None): Flag -> value inside the block.
    """

    previous = {key: os.environ.get(key) for key in _FLAGS}
    try:
        for key in _FLAGS:  # start from a clean slate for the flags under test
            if key.startswith("NESTOR_"):
                os.environ.pop(key, None)
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _single_rank_group() -> Iterator[None]:
    """A one-process gloo group, torn down afterwards."""

    with _env(MASTER_ADDR="127.0.0.1", MASTER_PORT="29577"):
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            yield
        finally:
            dist.destroy_process_group()


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
    model = Model.from_config(config).train()
    for module in model.modules():  # no router noise: runs are comparable
        if hasattr(module, "noise_mlp"):
            module.noise_mlp = None
    return model


def _wrapped_grads(force_wrapper: bool) -> tuple[bool, dict[str, Tensor]]:
    """Wrap the tiny model on one rank, run one step, return its gradients.

    Must be called inside :func:`_single_rank_group`.

    Args:
        force_wrapper (bool): Whether ``NESTOR_DDP_SINGLE_RANK=1`` is set.

    Returns:
        tuple[bool, dict[str, Tensor]]: Whether the result of
            ``wrap_model_for_distributed`` is a DDP wrapper, and the gradient
            of every parameter of the underlying model.
    """

    with _env(NESTOR_DDP="1", NESTOR_DDP_SINGLE_RANK="1" if force_wrapper else None):
        model = _build_model()
        wrapped = wrap_model_for_distributed(
            model,
            DistributedConfig(enabled=True, backend="gloo"),
            device=torch.device("cpu"),
        )
        torch.manual_seed(7)
        tokens = torch.randint(0, 64, (3, 40))
        out = wrapped(tokens)
        CausalLMLoss(CausalLMLossConfig())(out.pred, tokens, out.aux).total.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return isinstance(wrapped, DDP), grads


class SingleRankDdpSkipTest:
    """``wrap_model_for_distributed`` on a one-rank process group."""

    def test_bare_model_returned_on_one_rank(self) -> None:
        """NESTOR_DDP=1, world size 1: the very same module comes back."""

        with _single_rank_group(), _env(NESTOR_DDP="1"):
            assert single_rank_ddp_skipped()
            model = _build_model()
            wrapped = wrap_model_for_distributed(
                model,
                DistributedConfig(enabled=True, backend="gloo"),
                device=torch.device("cpu"),
            )
            assert wrapped is model
            assert not isinstance(wrapped, DDP)

    def test_wrapper_kept_when_forced(self) -> None:
        """NESTOR_DDP_SINGLE_RANK=1 keeps the DistributedDataParallel wrapper."""

        with _single_rank_group(), _env(NESTOR_DDP="1", NESTOR_DDP_SINGLE_RANK="1"):
            assert not single_rank_ddp_skipped()
            model = _build_model()
            wrapped = wrap_model_for_distributed(
                model,
                DistributedConfig(enabled=True, backend="gloo"),
                device=torch.device("cpu"),
            )
            assert isinstance(wrapped, DDP)
            assert wrapped.module is model

    def test_not_skipped_without_the_ddp_flag(self) -> None:
        """Without NESTOR_DDP=1 the helper is False (FSDP2 keeps its own path)."""

        with _single_rank_group(), _env():
            assert not single_rank_ddp_skipped()

    def test_not_skipped_without_a_process_group(self) -> None:
        """No process group: nothing to skip; a disabled config returns the model."""

        with _env(NESTOR_DDP="1"):
            assert not dist.is_initialized()
            assert not single_rank_ddp_skipped()
            model = _build_model()
            wrapped = wrap_model_for_distributed(
                model, DistributedConfig(enabled=False), device=torch.device("cpu")
            )
            assert wrapped is model

    def test_bare_gradients_equal_single_rank_ddp(self) -> None:
        """The bare model reproduces the one-rank DDP wrapper's gradients.

        The mean over one rank is that rank's own gradient, so the two must
        agree exactly (same kernels, same order).
        """

        with _single_rank_group():
            is_ddp_forced, reference = _wrapped_grads(force_wrapper=True)
            is_ddp_bare, got = _wrapped_grads(force_wrapper=False)
        assert is_ddp_forced and not is_ddp_bare
        assert got.keys() == reference.keys()
        for name, ref in reference.items():
            torch.testing.assert_close(
                got[name], ref, rtol=0.0, atol=0.0,
                msg=lambda m, _n=name: f"{_n}: {m}",
            )

    def test_streamed_head_on_bare_single_rank_model(self) -> None:
        """Streamed head + one rank: no wrapper, no ignore list, head gets grads."""

        with _single_rank_group(), _env(NESTOR_DDP="1", NESTOR_STREAMED_HEAD_CE="1"):
            model = _build_model()
            wrapped = wrap_model_for_distributed(
                model,
                DistributedConfig(enabled=True, backend="gloo"),
                device=torch.device("cpu"),
            )
            assert wrapped is model
            torch.manual_seed(7)
            tokens = torch.randint(0, 64, (3, 40))
            out = wrapped(tokens)
            CausalLMLoss(CausalLMLossConfig())(
                out.pred, tokens, out.aux
            ).total.backward()
            head_grads = [
                p.grad for n, p in model.named_parameters() if n.startswith("head.")
            ]
            assert head_grads and all(
                g is not None and g.abs().sum() > 0 for g in head_grads
            )
