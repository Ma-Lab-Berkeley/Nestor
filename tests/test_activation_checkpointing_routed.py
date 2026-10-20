"""Activation checkpointing on ROUTED expert banks, in TRAIN mode.

``tests/test_activation_checkpointing.py`` pins checkpoint equivalence for
1-expert layers in eval mode -- router noise off, no expert dispatch. The
64k/32k long-context probes showed the routed MoE control
drifting progressively above its checkpoint-OFF twin while the dense control
(3 always-active experts) and the column arm stayed healthy, so the untested
combination is exactly the one that matters: a sparse expert bank with
train-mode routing noise and loss-free balancing under
``NESTOR_ACTIVATION_CHECKPOINT=1``.

These tests build a small routed bank (8 experts, top-2, noise MLP on,
loss-free balancing on) and require the checkpointed forward AND every
gradient to match the plain forward exactly, with the model in ``train()``
mode so the recompute has to reproduce the same noise draw and the same
expert selection. ``RoutedBankCheckpointCudaTest`` repeats it under the
production dispatch (``NESTOR_MOE_GROUPED_MM=1``) and under ``torch.compile``
when a GPU is present.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig

_ENV_KEYS = [
    "NESTOR_ACTIVATION_CHECKPOINT",
    "NESTOR_MOE_GROUPED_MM",
    "NESTOR_DENSE_NO_ROUTER",
    "NESTOR_DENSE_FULL_ATTN",
    "NESTOR_ATTN_PACKED_VARLEN",
    "NESTOR_GATHERED_CAUSAL_FLASH",
]


def _set_env(**values: str) -> dict[str, str | None]:
    """Set env flags, returning the previous values for :func:`_restore_env`."""

    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in values.items():
        os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    """Reinstate env flags saved by :func:`_set_env`."""

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _routed_bank_config() -> NestorConfig:
    """Two causal blocks with a routed 8-expert bank, top-2, noise + loss-free.

    Returns:
        NestorConfig: The config (1 attention group like the MoE baseline,
        ``attn_res_kimi`` reads like the baseline arms).
    """

    n = 2
    return NestorConfig(
        input_dim=16,
        num_blocks=n,
        feature_dims=[16] * n,
        pooling_strides=[1] * n,
        causal=True,
        block_num_layers=[2] * n,
        num_groups=[1] * n,
        num_groups_per_token=[1] * n,
        block_num_heads=[2] * n,
        block_head_dims=[8] * n,
        block_mlp_hidden_dims=[[16]] * n,
        router_num_experts_per_group=[8] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[2] * n,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        router_loss_free_balancing=True,
        router_bias_update_rate=0.001,
        residual_mode="convex",
        block_residual_mode="attn_res_kimi",
    )


def _run(
    model: Nestor,
    seed: int,
    device: str,
    compiled: bool = False,
    bf16: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One TRAIN-mode forward/backward; returns pred and every gradient.

    Args:
        model (Nestor): Backbone in ``train()`` mode.
        seed (int): Seed drawn before the input AND before the router noise,
            so both arms see one batch and one noise stream.
        device (str): ``"cpu"`` or ``"cuda"``.
        compiled (bool): Wrap the forward in ``torch.compile`` (GPU only).
        bf16 (bool): Run under bf16 autocast (the production regime; the
            grouped GEMM only traces in bf16).

    Returns:
        tuple[Tensor, dict[str, Tensor]]: Prediction and name -> grad map
            (including ``"input"``).
    """

    from contextlib import nullcontext

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = torch.randn(2, 24, 16, device=device, requires_grad=True)
    pos = torch.arange(24, device=device).expand(2, -1)
    weights = torch.randn(2, 24, 16, device=device)

    if compiled:
        # Two separately compiled graphs (plain vs checkpointed) would otherwise
        # draw the routers' pre-drawn noise from Inductor's per-graph philox
        # streams at different offsets; eager RNG semantics keep the draws equal.
        import torch._inductor.config as _inductor_config
        _inductor_config.fallback_random = True
    forward = torch.compile(model, fullgraph=False) if compiled else model
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    )
    with autocast:
        out = forward(x, pos, None)
    (out.pred.float() * weights).sum().backward()

    grads = {"input": x.grad.detach().clone()}
    grads.update(
        {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
    )
    return out.pred.detach(), grads


def _assert_checkpoint_matches(
    device: str, *, grouped_mm: bool, compiled: bool, bf16: bool = False
) -> None:
    """Checkpointed train-mode forward/backward == plain, on a routed bank.

    The failure this guards against is not a rounding one: with the CUDA RNG
    state not restored for the recompute, the backward routes tokens to
    different experts than the forward did and the input gradient differs by
    O(1) (measured: greatest abs diff 4.8 at rtol 2e-3). The tolerances
    below are therefore "kernel noise" wide, not "bug" wide.

    Args:
        device (str): Device to run on.
        grouped_mm (bool): ``NESTOR_MOE_GROUPED_MM=1`` (production dispatch).
        compiled (bool): Run under ``torch.compile``.
        bf16 (bool): Run under bf16 autocast (production; required for the
            compiled grouped GEMM).
    """

    env = {"NESTOR_MOE_GROUPED_MM": "1"} if grouped_mm else {}
    previous = _set_env(**env)
    try:
        torch.manual_seed(3)
        plain = Nestor(_routed_bank_config()).to(device).train()
        torch.manual_seed(3)
        checkpointed = Nestor(_routed_bank_config()).to(device).train()

        os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "0"
        plain_out, plain_grads = _run(plain, 11, device, compiled, bf16)
        floors: list[tuple[Tensor, dict[str, Tensor]]] = []
        if bf16:
            # Noise floor of the plain path itself: a same-seed repeat (kernel
            # nondeterminism: grouped-GEMM split-K, index_add atomics) and,
            # under compile, the eager twin of the same model (a differently
            # fused graph rounds differently -- exactly what separates the
            # plain and checkpointed compiled graphs).
            floors.append(_run(plain, 11, device, compiled, bf16))
            if compiled:
                floors.append(_run(plain, 11, device, False, bf16))
        os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "1"
        ckpt_out, ckpt_grads = _run(checkpointed, 11, device, compiled, bf16)
    finally:
        _restore_env(previous)

    assert plain_grads.keys() == ckpt_grads.keys()
    if bf16:
        # A re-routed token moves its whole 16-wide row by O(1) (4.8 before
        # the fix), so no element may move by more than 0.1. The mean is
        # judged against the plain path's OWN noise floor measured above --
        # gradients of reduced parameters (norm gains summed over every
        # token, with cancellation) amplify bf16 rounding to ~1e-2, which a
        # fixed absolute threshold cannot express.
        def _mean_floor(name: str, ref: Tensor) -> float:
            """Largest mean |diff| between the plain run and its repeats."""
            floor = 0.0
            for f_out, f_grads in floors:
                other = f_out if name == "pred" else f_grads[name]
                floor = max(floor, (other.float() - ref.float()).abs().mean().item())
            return floor

        def _assert_no_flip(name: str, got: Tensor, ref: Tensor) -> None:
            diff = (got.float() - ref.float()).abs()
            assert diff.max().item() < 0.1, (
                f"{name}: max |diff| {diff.max().item():.4f} -- a token was "
                "routed differently on the recompute"
            )
            floor = _mean_floor(name, ref)
            limit = max(4.0 * floor, 2e-3)
            assert diff.mean().item() <= limit, (
                f"{name}: mean |diff| {diff.mean().item():.5f} > {limit:.5f} "
                f"(4x the plain path's own noise floor {floor:.5f})"
            )

        _assert_no_flip("pred", ckpt_out, plain_out)
        for name, plain_grad in plain_grads.items():
            _assert_no_flip(name, ckpt_grads[name], plain_grad)
        return

    tol = {"rtol": 1e-4, "atol": 1e-5} if device == "cpu" else {"rtol": 2e-3, "atol": 2e-4}
    torch.testing.assert_close(ckpt_out.float(), plain_out.float(), **tol)
    for name, plain_grad in plain_grads.items():
        torch.testing.assert_close(
            ckpt_grads[name].float(),
            plain_grad.float(),
            msg=lambda text, name=name: f"{name}: {text}",
            **tol,
        )


class RoutedBankCheckpointTest:
    """CPU, eager: sparse dispatch path."""

    def test_train_mode_forward_and_gradients_match(self) -> None:
        _assert_checkpoint_matches("cpu", grouped_mm=False, compiled=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class RoutedBankCheckpointCudaTest:
    """GPU: the sparse path, then the production grouped-GEMM dispatch in
    fp32 eager, bf16 eager and bf16 compiled."""

    def test_sparse_eager_fp32_matches(self) -> None:
        _assert_checkpoint_matches("cuda", grouped_mm=False, compiled=False)

    def test_grouped_mm_eager_fp32_matches(self) -> None:
        _assert_checkpoint_matches("cuda", grouped_mm=True, compiled=False)

    def test_grouped_mm_eager_bf16_matches(self) -> None:
        _assert_checkpoint_matches(
            "cuda", grouped_mm=True, compiled=False, bf16=True
        )

    def test_grouped_mm_compiled_bf16_matches(self) -> None:
        torch._dynamo.reset()
        _assert_checkpoint_matches(
            "cuda", grouped_mm=True, compiled=True, bf16=True
        )
