"""Equivalence tests for the streamed head+CE path (``NESTOR_STREAMED_HEAD_CE``).

``nestor.train.streamed_head_ce.streamed_head_ce`` reads the cross entropy of
a linear readout one row chunk at a time and differentiates the readout by
hand in the same pass: it never materializes the ``[tokens, vocab]`` logits
and recomputes nothing in backward. Its contract is to equal, in value and in
every gradient,

    F.cross_entropy(F.linear(hidden, weight, bias) * multiplier, targets,
                    ignore_index=..., label_smoothing=..., reduction="sum")

which is the reference implementation here (plain autograd, fp64).

The tests pin

* the function against that reference across chunk sizes (one row per chunk,
  ragged last chunk, one chunk), label smoothing, the logit multiplier, a
  missing bias and ignored targets;
* the hand-written backward against finite differences (``gradcheck``);
* the bf16 pitfall the backward guards against: the loss scale must be applied
  in fp32, never rounded to the gradient's storage dtype;
* the model-level wiring: with the flag set, ``SequenceModel`` + ``CausalLMLoss``
  reproduce the loss and EVERY parameter gradient of the standard path, eval
  mode still returns real logits, the flag takes precedence over the
  checkpointed in-loss head, and a head with hidden layers falls back to it.

CPU-runnable: the tiny model uses the default per-group dispatch loop.
"""

from __future__ import annotations

import os
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor

from nestor.configs.losses import CausalLMLossConfig
from nestor.train.losses import CausalLMLoss
from nestor.train.streamed_head_ce import (
    fused_head_mode,
    streamable_projection,
    streamed_head_ce,
)

IGNORE: int = -100


class _FlagEnv:
    """Set/restore env flags (no monkeypatching).

    Attributes:
        values (dict[str, str | None]): Flag -> value inside the block
            (``None`` unsets the flag).
        previous (dict[str, str | None]): Restored on exit (None -> unset).
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


def _reference(
    hidden: Tensor,
    targets: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    logit_multiplier: float,
    label_smoothing: float,
) -> Tensor:
    """Reference implementation: full logits, then PyTorch's cross entropy.

    Args:
        hidden (Tensor): Hidden states. Shape: ``(rows, d)``.
        targets (Tensor): Target ids or ``IGNORE``. Shape: ``(rows,)``.
        weight (Tensor): Readout weight. Shape: ``(vocab, d)``.
        bias (Tensor | None): Readout bias. Shape: ``(vocab,)``.
        logit_multiplier (float): Scalar applied to the logits.
        label_smoothing (float): Label smoothing.

    Returns:
        Tensor: Summed loss over the non-ignored rows. Shape: ``()``.
    """

    logits = F.linear(hidden, weight, bias) * logit_multiplier  # [rows, vocab]
    return F.cross_entropy(
        logits,
        targets,
        ignore_index=IGNORE,
        label_smoothing=label_smoothing,
        reduction="sum",
    )


def _problem(
    rows: int, width: int, vocab: int, *, dtype: torch.dtype, with_bias: bool, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """A random readout problem with a few ignored targets.

    Args:
        rows (int): Number of token rows.
        width (int): Hidden width ``d``.
        vocab (int): Vocabulary size.
        dtype (torch.dtype): Floating dtype of the leaves.
        with_bias (bool): Whether the readout has a bias.
        seed (int): Generator seed.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor | None]: ``hidden`` ``(rows, d)``,
            ``targets`` ``(rows,)`` (every fifth row ignored), ``weight``
            ``(vocab, d)`` and ``bias`` ``(vocab,)`` or ``None``; the float
            tensors are leaves that require grad.
    """

    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, width, generator=generator, dtype=dtype).requires_grad_()
    weight = torch.randn(vocab, width, generator=generator, dtype=dtype).requires_grad_()
    bias = (
        torch.randn(vocab, generator=generator, dtype=dtype).requires_grad_()
        if with_bias
        else None
    )
    targets = torch.randint(0, vocab, (rows,), generator=generator)
    targets[::5] = IGNORE
    return hidden, targets, weight, bias


def _grads(loss: Tensor, leaves: list[Tensor]) -> list[Tensor]:
    """Gradients of ``loss`` with respect to ``leaves`` (fresh graph each call)."""

    return list(torch.autograd.grad(loss, leaves))


class StreamedHeadCeFunctionTest:
    """``streamed_head_ce`` equals full-logits cross entropy, value and grads."""

    def test_matches_reference_fp64(self) -> None:
        """Loss and all three gradients agree to fp64 rounding in every mode."""

        rows, width, vocab = 37, 6, 23
        for chunk_rows in (1, 5, 36, 37, 1000):
            for label_smoothing in (0.0, 0.1):
                for multiplier in (1.0, 0.7):
                    for with_bias in (True, False):
                        hidden, targets, weight, bias = _problem(
                            rows, width, vocab,
                            dtype=torch.float64, with_bias=with_bias, seed=3,
                        )
                        leaves = [hidden, weight] + ([bias] if with_bias else [])
                        ref = _reference(
                            hidden, targets, weight, bias,
                            logit_multiplier=multiplier,
                            label_smoothing=label_smoothing,
                        )
                        got = streamed_head_ce(
                            hidden, targets, weight, bias,
                            logit_multiplier=multiplier,
                            chunk_rows=chunk_rows,
                            ignore_index=IGNORE,
                            label_smoothing=label_smoothing,
                        )
                        case = (
                            f"chunk={chunk_rows} eps={label_smoothing} "
                            f"mult={multiplier} bias={with_bias}"
                        )
                        torch.testing.assert_close(
                            got, ref, rtol=1e-12, atol=1e-12, msg=lambda m: f"{case}: {m}"
                        )
                        for g, r in zip(_grads(got, leaves), _grads(ref, leaves)):
                            torch.testing.assert_close(
                                g, r, rtol=1e-10, atol=1e-12,
                                msg=lambda m: f"{case}: {m}",
                            )

    def test_matches_reference_fp32(self) -> None:
        """The fp32 path (no autocast) agrees to fp32 rounding."""

        hidden, targets, weight, bias = _problem(
            64, 16, 101, dtype=torch.float32, with_bias=True, seed=11
        )
        leaves = [hidden, weight, bias]
        ref = _reference(
            hidden, targets, weight, bias, logit_multiplier=1.0, label_smoothing=0.0
        )
        got = streamed_head_ce(
            hidden, targets, weight, bias, chunk_rows=24, ignore_index=IGNORE
        )
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
        for g, r in zip(_grads(got, leaves), _grads(ref, leaves)):
            assert g.dtype == torch.float32
            torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-5)

    def test_gradcheck(self) -> None:
        """The hand-written backward matches finite differences."""

        hidden, targets, weight, bias = _problem(
            7, 3, 5, dtype=torch.float64, with_bias=True, seed=5
        )

        def _loss(h: Tensor, w: Tensor, b: Tensor) -> Tensor:
            """Streamed loss as a function of the three differentiable inputs."""

            return streamed_head_ce(
                h, targets, w, b,
                logit_multiplier=1.3,
                chunk_rows=3,
                ignore_index=IGNORE,
                label_smoothing=0.05,
            )

        assert torch.autograd.gradcheck(_loss, (hidden, weight, bias), atol=1e-6)

    def test_ignored_rows_contribute_nothing(self) -> None:
        """All-ignored input: zero loss, zero gradients (no NaN from the mask)."""

        hidden, targets, weight, bias = _problem(
            12, 4, 9, dtype=torch.float64, with_bias=True, seed=2
        )
        targets = torch.full_like(targets, IGNORE)
        loss = streamed_head_ce(
            hidden, targets, weight, bias, chunk_rows=5, ignore_index=IGNORE
        )
        assert loss.item() == 0.0
        for grad in _grads(loss, [hidden, weight, bias]):
            assert torch.count_nonzero(grad).item() == 0

    def test_frozen_weight_gets_no_gradient(self) -> None:
        """A frozen readout still sends the right gradient to the hidden states."""

        hidden, targets, weight, bias = _problem(
            20, 5, 13, dtype=torch.float64, with_bias=True, seed=8
        )
        ref_grad = _grads(
            _reference(
                hidden, targets, weight, bias,
                logit_multiplier=1.0, label_smoothing=0.0,
            ),
            [hidden],
        )[0]
        frozen_weight = weight.detach()
        frozen_bias = bias.detach()
        loss = streamed_head_ce(
            hidden, targets, frozen_weight, frozen_bias,
            chunk_rows=7, ignore_index=IGNORE,
        )
        loss.backward()
        torch.testing.assert_close(hidden.grad, ref_grad, rtol=1e-10, atol=1e-12)
        assert frozen_weight.grad is None and frozen_bias.grad is None

    def test_no_grad_call_returns_the_loss(self) -> None:
        """Under ``no_grad`` the function still returns the loss value."""

        hidden, targets, weight, bias = _problem(
            15, 4, 11, dtype=torch.float64, with_bias=True, seed=4
        )
        ref = _reference(
            hidden, targets, weight, bias, logit_multiplier=1.0, label_smoothing=0.0
        )
        with torch.no_grad():
            got = streamed_head_ce(
                hidden, targets, weight, bias, chunk_rows=4, ignore_index=IGNORE
            )
        assert not got.requires_grad
        torch.testing.assert_close(got, ref.detach(), rtol=1e-12, atol=1e-12)

    def test_shape_errors(self) -> None:
        """Mismatched shapes are rejected before any work."""

        import pytest

        hidden, targets, weight, bias = _problem(
            6, 4, 9, dtype=torch.float32, with_bias=True, seed=1
        )
        with pytest.raises(ValueError):
            streamed_head_ce(hidden.unsqueeze(0), targets, weight, bias)
        with pytest.raises(ValueError):
            streamed_head_ce(hidden, targets[:-1], weight, bias)
        with pytest.raises(ValueError):
            streamed_head_ce(hidden, targets, weight[:, :-1], bias)


class StreamedHeadCeAutocastTest:
    """bf16 autocast: logits and gradient GEMMs in bf16, statistics in fp32."""

    def test_bf16_autocast_tracks_the_fp32_reference(self) -> None:
        """Under bf16 autocast the result stays within bf16 rounding of fp32."""

        hidden, targets, weight, bias = _problem(
            48, 16, 61, dtype=torch.float32, with_bias=True, seed=21
        )
        leaves = [hidden, weight, bias]
        ref = _reference(
            hidden, targets, weight, bias, logit_multiplier=1.0, label_smoothing=0.0
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            got = streamed_head_ce(
                hidden, targets, weight, bias, chunk_rows=16, ignore_index=IGNORE
            )
        assert got.dtype == torch.float32, "the loss statistic must stay fp32"
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=1e-2)
        for g, r in zip(_grads(got, leaves), _grads(ref, leaves)):
            assert g.dtype == torch.float32, "leaf gradients keep the leaf dtype"
            # bf16 carries 8 mantissa bits: compare in aggregate, not per entry.
            relative = (g - r).norm() / r.norm()
            assert relative.item() < 2e-2, f"relative gradient error {relative.item()}"

    def test_loss_scale_is_applied_in_fp32(self) -> None:
        """The backward scale must not be rounded to bf16.

        The hidden gradient is stored in bf16. A 0-dim fp32 scalar does not
        promote a bf16 tensor, so ``grad_output * stored`` would silently
        round the loss scale (``1 / (tokens * grad_accum)`` in training) to 8
        mantissa bits -- a systematic gradient-scale error of up to 0.4 %.
        With the scale applied in fp32, the gradient of ``s * loss`` is
        exactly ``s`` times the gradient of ``loss`` up to fp32 rounding.
        """

        scale = 1.0 / 3.0  # not representable in bf16: rounds to 0.333984375
        results: list[Tensor] = []
        for factor in (1.0, scale):
            hidden, targets, weight, bias = _problem(
                32, 8, 29, dtype=torch.float32, with_bias=True, seed=17
            )
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = streamed_head_ce(
                    hidden, targets, weight, bias, chunk_rows=8, ignore_index=IGNORE
                )
            results.append(_grads(loss * factor, [hidden])[0])
        unscaled, scaled = results
        torch.testing.assert_close(scaled, unscaled * scale, rtol=1e-6, atol=0.0)


def _build_model(hidden_dims: list[int] | None = None):
    """Tiny causal grouped token-dispatch LM, fixed seed 13.

    Args:
        hidden_dims (list[int] | None): Hidden widths of the LM head (empty or
            ``None`` = the usual single linear readout).

    Returns:
        SequenceModel: Model in train mode (the in-loss head is train-only).
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
        head_config=LanguageModelingHeadConfig(
            vocab_size=64, hidden_dims=list(hidden_dims or [])
        ),
    )
    torch.manual_seed(13)
    model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model.train()


def _run(
    *,
    streamed: bool,
    fused: bool = False,
    chunk_rows: int = 16,
    hidden_dims: list[int] | None = None,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor], str]:
    """One forward+loss+backward of the tiny model under the given flags.

    Fixed seeds for weights, tokens, and router noise, so the flags are the
    only difference between calls.

    Args:
        streamed (bool): Whether ``NESTOR_STREAMED_HEAD_CE`` is set.
        fused (bool): Whether ``NESTOR_FUSED_HEAD_CE`` is set.
        chunk_rows (int): ``NESTOR_CE_CHUNK_ROWS``.
        hidden_dims (list[int] | None): Hidden widths of the LM head.
        label_smoothing (float): Label smoothing of the loss.

    Returns:
        tuple[Tensor, dict[str, Tensor], str]: Total loss, per-parameter
            gradients, and the name of the autograd node that produced the
            summed loss (identifies which path ran).
    """

    model = _build_model(hidden_dims)
    loss_fn = CausalLMLoss(CausalLMLossConfig(label_smoothing=label_smoothing))
    torch.manual_seed(7)
    tokens = torch.randint(0, 64, (3, 40))
    with _FlagEnv(
        NESTOR_STREAMED_HEAD_CE="1" if streamed else "0",
        NESTOR_FUSED_HEAD_CE="1" if fused else "0",
        NESTOR_CE_CHUNK_ROWS=str(chunk_rows),
    ):
        torch.manual_seed(99)  # router noise draw
        out = model(tokens)
        loss = loss_fn(out.pred, tokens, out.aux).total
    # total = loss_sum / num_tokens: the first input of the division is the
    # summed loss, whose grad_fn names the path.
    sum_node = type(loss.grad_fn.next_functions[0][0]).__name__
    loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), grads, sum_node


class StreamedHeadModelTest:
    """The streamed path must reproduce the standard head->loss pipeline."""

    def test_loss_and_all_grads_match(self) -> None:
        """Loss and every parameter gradient (head included) agree."""

        for label_smoothing in (0.0, 0.1):
            ref_loss, ref_grads, ref_node = _run(
                streamed=False, label_smoothing=label_smoothing
            )
            got_loss, got_grads, got_node = _run(
                streamed=True, label_smoothing=label_smoothing
            )
            assert "StreamedHeadCE" in got_node, got_node
            assert "StreamedHeadCE" not in ref_node, ref_node
            torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
            assert got_grads.keys() == ref_grads.keys(), (
                "streamed path changed WHICH parameters receive gradients"
            )
            for name in ref_grads:
                torch.testing.assert_close(
                    got_grads[name],
                    ref_grads[name],
                    rtol=1e-4,
                    atol=1e-5,
                    msg=lambda m, _n=name: f"{_n} (eps={label_smoothing}): {m}",
                )

    def test_chunk_size_does_not_change_the_result(self) -> None:
        """One row per chunk and one chunk for everything agree."""

        one_loss, one_grads, _ = _run(streamed=True, chunk_rows=1)
        all_loss, all_grads, _ = _run(streamed=True, chunk_rows=100000)
        torch.testing.assert_close(one_loss, all_loss, rtol=1e-5, atol=1e-6)
        for name in all_grads:
            torch.testing.assert_close(
                one_grads[name], all_grads[name], rtol=1e-4, atol=1e-5,
                msg=lambda m, _n=name: f"{_n}: {m}",
            )

    def test_head_receives_grads_via_loss(self) -> None:
        """The head weight AND bias get nonzero grads on the streamed path."""

        _, grads, _ = _run(streamed=True)
        head_grads = {n: g for n, g in grads.items() if n.startswith("head.")}
        assert any(n.endswith("weight") for n in head_grads), sorted(head_grads)
        assert any(n.endswith("bias") for n in head_grads), sorted(head_grads)
        assert all(g.abs().sum() > 0 for g in head_grads.values())

    def test_streamed_takes_precedence_over_checkpointed(self) -> None:
        """Both flags set: the streamed path runs."""

        _, _, node = _run(streamed=True, fused=True)
        assert "StreamedHeadCE" in node, node

    def test_mlp_head_falls_back_to_checkpointed_projection(self) -> None:
        """A head with a hidden layer is not one linear map: no streaming.

        It must take the checkpointed in-loss projection instead and still
        match the standard path.
        """

        assert streamable_projection(_build_model([16]).head) is None
        assert streamable_projection(_build_model().head) is not None
        ref_loss, ref_grads, _ = _run(streamed=False, hidden_dims=[16])
        got_loss, got_grads, node = _run(streamed=True, hidden_dims=[16])
        assert "StreamedHeadCE" not in node, node
        torch.testing.assert_close(got_loss, ref_loss, rtol=1e-5, atol=1e-6)
        assert got_grads.keys() == ref_grads.keys()
        for name in ref_grads:
            torch.testing.assert_close(
                got_grads[name], ref_grads[name], rtol=1e-4, atol=1e-5,
                msg=lambda m, _n=name: f"{_n}: {m}",
            )

    def test_eval_mode_bypasses_the_in_loss_head(self) -> None:
        """model.eval() must return REAL logits even with the flag set.

        Held-out evals read ``pred`` as logits; the in-loss path would hand
        them hidden states of the wrong width.
        """

        model = _build_model().eval()
        tokens = torch.randint(0, 64, (2, 17))
        with _FlagEnv(NESTOR_STREAMED_HEAD_CE="1"), torch.no_grad():
            out = model(tokens)
        assert out.pred.shape[-1] == 64, "eval pred is not vocab-width logits"
        assert "fused_head_module" not in out.aux


class FusedHeadModeTest:
    """``fused_head_mode`` reads the two flags with the documented precedence."""

    def test_mode_table(self) -> None:
        """Unset -> None; fused -> checkpoint; streamed (alone or both) -> stream."""

        table: list[tuple[str | None, str | None, str | None]] = [
            (None, None, None),
            ("0", "0", None),
            (None, "1", "checkpoint"),
            ("0", "1", "checkpoint"),
            ("1", None, "stream"),
            ("1", "0", "stream"),
            ("1", "1", "stream"),
        ]
        for streamed, fused, expected in table:
            with _FlagEnv(NESTOR_STREAMED_HEAD_CE=streamed, NESTOR_FUSED_HEAD_CE=fused):
                assert fused_head_mode() == expected, (streamed, fused)
