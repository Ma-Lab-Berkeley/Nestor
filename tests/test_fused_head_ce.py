"""Equivalence tests for the fused head+CE path (``NESTOR_FUSED_HEAD_CE``).

With the flag set (train mode, grad on), ``SequenceModel.forward`` skips the
unembedding and hands the loss the hidden states plus the head via ``aux``;
``CausalLMLoss`` projects hidden -> logits per row chunk inside a checkpointed
closure, so the full ``[tokens, vocab]`` logits tensor is never materialized.
That tensor is the forward-phase allocation that OOMs the 1b arm at per-rank 8.

The math must be identical: the projection is per-token, so it commutes with
the loss's target shift and with row chunking, and sum reduction makes the
chunk split exact. These tests pin the loss value and EVERY parameter gradient
(head included -- its grads now arrive via the loss graph rather than the
model forward) against the standard path, and check the eval-mode bypass that
keeps held-out metrics on real logits.

CPU-runnable: the tiny model uses the default per-group dispatch loop.
"""

from __future__ import annotations

import os
from typing import Self

import torch

from nestor.configs.losses import CausalLMLossConfig
from nestor.train.losses import CausalLMLoss


class _FlagEnv:
    """Set/restore env flags (no monkeypatching).

    Attributes:
        values (dict[str, str]): Flag -> value inside the block.
        previous (dict[str, str | None]): Restored on exit (None -> unset).
    """

    def __init__(self, **values: str) -> None:
        self.values: dict[str, str] = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> Self:
        for key, val in self.values.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = val
        return self

    def __exit__(self, *exc: object) -> None:
        for key, prev in self.previous.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _build_model():
    """Tiny causal grouped token-dispatch LM, fixed seed 13.

    Returns:
        SequenceModel: Model in train mode (the fused path is train-only).
    """

    from nestor.backbones import NestorConfig
    from nestor.configs.adapters import TextAdapterConfig
    from nestor.heads import LanguageModelingHeadConfig
    from nestor.models import Model, SequenceModel, SequenceModelConfig

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
    model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model.train()


def _run(fused: bool, chunk_rows: int = 16):
    """One forward+loss+backward; returns (loss, {name: grad}).

    Fixed seeds for weights, tokens, and router noise, so the flag is the only
    difference between calls.

    Args:
        fused (bool): Whether ``NESTOR_FUSED_HEAD_CE`` is set.
        chunk_rows (int): Chunk size for the fused CE.

    Returns:
        tuple[Tensor, dict[str, Tensor]]: Total loss and per-parameter grads.
    """

    model = _build_model()
    loss_fn = CausalLMLoss(CausalLMLossConfig())
    torch.manual_seed(7)
    tokens = torch.randint(0, 64, (3, 40))
    with _FlagEnv(
        NESTOR_FUSED_HEAD_CE="1" if fused else "0",
        NESTOR_CE_CHUNK_ROWS=str(chunk_rows),
    ):
        torch.manual_seed(99)  # router noise draw
        out = model(tokens)
        loss = loss_fn(out.pred, tokens, out.aux).total
    loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), grads


class FusedHeadCeTest:
    """The fused path must reproduce the standard head->loss pipeline."""

    def test_loss_and_all_grads_match(self) -> None:
        """Loss and every parameter gradient (head included) agree."""

        ref_loss, ref_grads = _run(fused=False)
        got_loss, got_grads = _run(fused=True)
        torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert got_grads.keys() == ref_grads.keys(), (
            "fused path changed WHICH parameters receive gradients"
        )
        for name in ref_grads:
            torch.testing.assert_close(
                got_grads[name],
                ref_grads[name],
                rtol=1e-4,
                atol=1e-5,
                msg=lambda m, _n=name: f"{_n}: {m}",
            )

    def test_head_receives_grads_via_loss(self) -> None:
        """The head parameters must get nonzero grads on the fused path."""

        _, grads = _run(fused=True)
        head_grads = [g for n, g in grads.items() if "head" in n]
        assert head_grads, "no head parameters received gradients"
        assert any(g.abs().sum() > 0 for g in head_grads)

    def test_eval_mode_bypasses_fusion(self) -> None:
        """model.eval() must return REAL logits even with the flag set.

        Held-out evals read ``pred`` as logits; the fused path would hand them
        hidden states of the wrong width.
        """

        model = _build_model().eval()
        tokens = torch.randint(0, 64, (2, 17))
        with _FlagEnv(NESTOR_FUSED_HEAD_CE="1"), torch.no_grad():
            out = model(tokens)
        assert out.pred.shape[-1] == 64, "eval pred is not vocab-width logits"
        assert "fused_head_module" not in out.aux
