"""Equivalence tests for the batched grouped-attention projections.

``NESTOR_GROUPED_PROJ_BMM=1`` replaces the per-group ``forward_gathered`` /
``forward_gathered_full`` loop in
:meth:`nestor.backbones.nestor.RoutedTransformerLayer._attn_update_general`
with :meth:`~nestor.backbones.nestor.RoutedTransformerLayer.
_gathered_proj_bmm_attention`: every group is capacity-padded to ONE shared
``cap`` and the ``4 * num_groups`` small ``ScaledLinear`` calls collapse into
two grouped GEMMs (fused q/k/v, then the output projection).

The per-group :class:`~nestor.scaling.blocks.ScaledLinear` modules stay the
parameter owners -- the batched weights are stacked per forward -- so the flag
must change **nothing**: not the outputs, not the gradient of any group's
``q/k/v/out_proj`` weight, not the parameter layout (both models share one
``state_dict``), and not the muP tagging applied by
:func:`~nestor.scaling.reference_mup.configure_reference_mup_model`.

The flag is read once per layer in ``__init__``, so every test builds two
models -- one with the flag, one without -- and copies the weights across, so
the attention path is the only difference. The flag-off per-group loop is the
reference implementation throughout.
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.nestor import RoutedTransformerLayer, RouterOutput
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.constants import PADDING_TOKEN_POS
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig

_BMM_FLAG = "NESTOR_GROUPED_PROJ_BMM"
_FLASH_FLAG = "NESTOR_GATHERED_CAUSAL_FLASH"
_FULL_ATTN_FLAG = "NESTOR_DENSE_FULL_ATTN"


class _FlagEnv:
    """Context manager that sets/clears the Nestor speed-stack env flags.

    Not a mock: it flips the real environment variables that the production
    modules read in ``__init__`` and restores the previous values on exit.

    Attributes:
        flags (dict[str, bool]): Flag name -> whether it is set to ``"1"``
            inside the block.
        previous (dict[str, str | None]): Values restored on exit.
    """

    flags: dict[str, bool]
    previous: dict[str, str | None]

    def __init__(self, **flags: bool) -> None:
        """Initialize the context manager.

        Args:
            **flags (bool): Keyword flag name -> enabled. Names are the env
                variables themselves (e.g. ``NESTOR_GROUPED_PROJ_BMM=True``).
        """

        self.flags = dict(flags)
        self.previous = {}

    def __enter__(self) -> Self:
        """Apply the flags and remember the previous values."""

        for name, enabled in self.flags.items():
            self.previous[name] = os.environ.get(name)
            if enabled:
                os.environ[name] = "1"
            else:
                os.environ.pop(name, None)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Restore the previous environment values."""

        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_layer(
    *,
    bmm: bool,
    causal: bool = True,
    flash: bool = False,
    full_attn: bool = False,
    num_groups: int = 4,
    dim: int = 32,
    seed: int = 5,
) -> RoutedTransformerLayer:
    """Build one token-dispatch routed layer under a given flag set.

    Args:
        bmm (bool): Whether ``NESTOR_GROUPED_PROJ_BMM`` is set.
        causal (bool): Whether the layer is causal.
        flash (bool): Whether ``NESTOR_GATHERED_CAUSAL_FLASH`` is set.
        full_attn (bool): Whether ``NESTOR_DENSE_FULL_ATTN`` is set.
        num_groups (int): Attention groups in the layer.
        dim (int): Token feature width.
        seed (int): Seed used for parameter initialization.

    Returns:
        RoutedTransformerLayer: The layer, in eval mode.
    """

    with _FlagEnv(
        **{
            _BMM_FLAG: bmm,
            _FLASH_FLAG: flash,
            _FULL_ATTN_FLAG: full_attn,
        }
    ):
        torch.manual_seed(seed)
        layer = RoutedTransformerLayer(
            dim=dim,
            num_experts_per_group=1,
            num_shared_experts=1,
            num_experts_per_group_per_token=1,
            num_groups=num_groups,
            num_heads=2,
            head_dim=16,
            mlp_hidden_dims=[24],
            mlp_activation="gelu",
            router_score_mlp_hidden_dims=[8],
            router_score_mlp_activation="gelu",
            router_noise_mlp_hidden_dims=[8],
            router_noise_mlp_activation="gelu",
            depth_scale=0.25,
            attention_token_dispatch=True,
            causal=causal,
            num_groups_per_token=2,
        )
    return layer.eval()


def _build_layer_pair(
    **kwargs: object,
) -> tuple[RoutedTransformerLayer, RoutedTransformerLayer]:
    """Build weight-identical (per-group loop, batched-GEMM) layers.

    Args:
        **kwargs (object): Forwarded to :func:`_build_layer` (minus ``bmm``).

    Returns:
        tuple[RoutedTransformerLayer, RoutedTransformerLayer]: Reference layer
        (flag off) and batched layer (flag on) sharing one ``state_dict``.
    """

    baseline = _build_layer(bmm=False, **kwargs)  # type: ignore[arg-type]
    batched = _build_layer(bmm=True, **kwargs)  # type: ignore[arg-type]
    # Identical parameter layout is part of the contract: the batched path
    # stacks the SAME per-group weights instead of re-registering them.
    batched.load_state_dict(baseline.state_dict())
    return baseline, batched


def _group_router_output(
    members: Tensor,  # [B, S, G] bool
    gates: Tensor,  # [B, S, G] float
    valid_token_mask: Tensor,  # [B, S] bool
) -> RouterOutput:
    """Assemble a group-router output from an explicit membership pattern.

    Args:
        members (Tensor): ``[B, S, G]`` per-token group membership.
        gates (Tensor): ``[B, S, G]`` gating coefficients (zero off-members).
        valid_token_mask (Tensor): ``[B, S]`` real (non-padding) tokens.

    Returns:
        RouterOutput: A router output the layer paths can consume. Only
        ``gates``, ``active_mask`` and ``valid_token_mask`` are read by the
        attention update; the rest are filled consistently.
    """

    batch_size, seq_len, num_groups = members.shape
    return RouterOutput(
        scores=torch.zeros(batch_size, seq_len, num_groups),
        softmax_probs=gates,
        gates=gates,
        topk_indices=torch.zeros(batch_size, seq_len, 1, dtype=torch.long),
        active_mask=members,
        valid_token_mask=valid_token_mask,
    )


def _dispatch_inputs(
    *,
    batch_size: int = 3,
    seq_len: int = 9,
    dim: int = 32,
    num_groups: int = 4,
    causal: bool = True,
    num_padded: int = 0,
    empty_group: int | None = None,
    seed: int = 11,
) -> tuple[Tensor, Tensor, Tensor, RouterOutput]:
    """Build a token-dispatch attention input set for one routed layer.

    Mirrors what :meth:`RoutedTransformerLayer._attn_update_general` receives
    from its block: a base mask carrying causality and padding, plus a group
    router output whose members never include padding tokens.

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        dim (int): Feature width.
        num_groups (int): Attention groups.
        causal (bool): Whether the base mask is causal.
        num_padded (int): Trailing padded positions in the LAST sequence.
        empty_group (int | None): Group forced to have no members anywhere.
        seed (int): Seed for the random draws.

    Returns:
        tuple[Tensor, Tensor, Tensor, RouterOutput]: ``x`` ``[B, S, dim]``,
        ``pos`` ``[B, S]``, the base mask ``[B, S, S]``, and the router output.
    """

    torch.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    if num_padded > 0:
        pos[-1, seq_len - num_padded :] = PADDING_TOKEN_POS
    valid = pos != PADDING_TOKEN_POS  # [B, S]

    pair = (
        torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        if causal
        else torch.ones(seq_len, seq_len, dtype=torch.bool)
    )
    base = pair.unsqueeze(0) & valid[:, :, None] & valid[:, None, :]

    members = (torch.rand(batch_size, seq_len, num_groups) > 0.45) & valid[
        ..., None
    ]
    members[:, :, 0] = valid  # keep group 0 non-empty on every sequence
    if empty_group is not None:
        members[:, :, empty_group] = False
    gates = torch.rand(batch_size, seq_len, num_groups) * members.to(
        torch.float32
    )
    return x, pos, base, _group_router_output(members, gates, valid)


def _attention_grads(
    layer: RoutedTransformerLayer,
) -> dict[str, Tensor]:
    """Collect every attention-group projection gradient of a layer.

    Args:
        layer (RoutedTransformerLayer): Layer with populated ``.grad`` fields.

    Returns:
        dict[str, Tensor]: ``"g{index}.{proj}"`` -> weight gradient, for the
        q/k/v/out projection of every attention group.
    """

    grads: dict[str, Tensor] = {}
    for index, group in enumerate(layer.attention_groups):
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weight = getattr(group, name).weight
            assert weight.grad is not None, f"missing grad for g{index}.{name}"
            grads[f"g{index}.{name}"] = weight.grad
    return grads


def _run_attention(
    layer: RoutedTransformerLayer,
    x: Tensor,
    base: Tensor,
    router_output: RouterOutput,
    pos: Tensor,
) -> Tensor:
    """Run one layer's grouped attention update.

    Args:
        layer (RoutedTransformerLayer): Layer under test.
        x (Tensor): ``[B, S, dim]`` pre-normalized sublayer input.
        base (Tensor): ``[B, S, S]`` base attention mask.
        router_output (RouterOutput): Group router output.
        pos (Tensor): ``[B, S]`` token positions.

    Returns:
        Tensor: ``[B, S, dim]`` attention update.
    """

    return layer._attn_update_general(x, base, router_output, pos)


def _dispatch_config(*, causal: bool = True) -> NestorConfig:
    """A tiny grouped-prefix backbone config with attention token dispatch.

    Block 0 has 4 attention groups (top-2 per token) routed through the
    gathered path; block 1 is dense. Strides are 1 so token positions (and
    therefore padding) survive to the end.

    Args:
        causal (bool): Whether the backbone is causal.

    Returns:
        NestorConfig: The tiny config.
    """

    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[32, 32],
        causal=causal,
        pooling_strides=[1, 1],
        block_num_layers=[2, 1],
        num_groups=[4, 1],
        num_groups_per_token=[2, 1],
        block_num_heads=2,
        block_head_dims=16,
        attention_token_dispatch=True,
        block_mlp_hidden_dims=[[24], [24]],
        block_mlp_activations="gelu",
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_score_mlp_hidden_dims=[[8], [8]],
        router_noise_mlp_hidden_dims=[[8], [8]],
    )


def _build_backbone_pair(
    config: NestorConfig, *, flash: bool = False
) -> tuple[Nestor, Nestor]:
    """Build weight-identical (per-group loop, batched-GEMM) backbones.

    Args:
        config (NestorConfig): Backbone config used for both models.
        flash (bool): Whether ``NESTOR_GATHERED_CAUSAL_FLASH`` is set for both.

    Returns:
        tuple[Nestor, Nestor]: Baseline (flag off) and batched (flag on)
        backbones with identical parameters, both in eval mode.
    """

    with _FlagEnv(**{_BMM_FLAG: False, _FLASH_FLAG: flash}):
        torch.manual_seed(7)
        baseline = Nestor(config).eval()
    with _FlagEnv(**{_BMM_FLAG: True, _FLASH_FLAG: flash}):
        torch.manual_seed(7)
        batched = Nestor(config).eval()
    batched.load_state_dict(baseline.state_dict())
    return baseline, batched


def _reference_mup_model(*, bmm: bool) -> SequenceModel:
    """Build a tiny grouped token-dispatch LM under the reference muP recipe.

    Args:
        bmm (bool): Whether ``NESTOR_GROUPED_PROJ_BMM`` is set.

    Returns:
        SequenceModel: The transformed model, in eval mode.
    """

    config = SequenceModelConfig(
        adapter_config=TextAdapterConfig(embed_dim=16, vocab_size=64),
        backbone_config=NestorConfig(
            input_dim=16,
            num_blocks=1,
            feature_dims=[16],
            pooling_strides=[1],
            causal=True,
            block_num_layers=2,
            num_groups=[4],
            num_groups_per_token=[2],
            block_num_heads=2,
            block_head_dims=8,
            attention_token_dispatch=True,
            block_mlp_hidden_dims=[[12]],
            block_mlp_activations="swiglu",
            router_num_experts_per_group=[1],
            router_num_shared_experts=[1],
            router_num_experts_per_group_per_token=1,
            router_score_mlp_hidden_dims=[[8]],
            router_noise_mlp_hidden_dims=[[8]],
        ),
        head_config=LanguageModelingHeadConfig(vocab_size=64),
        reference_mup=ReferenceMupConfig(num_layers=2),
    )
    with _FlagEnv(**{_BMM_FLAG: bmm}):
        torch.manual_seed(13)
        model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model.eval()


class GroupedProjBmmFlagTest:
    """Opt-in gating: the flag must be inert unless explicitly requested."""

    def test_disabled_without_the_flag(self) -> None:
        """No env flag -> the layer keeps the per-group loop."""

        layer = _build_layer(bmm=False)
        assert layer._grouped_proj_bmm is False

    def test_enabled_with_the_flag(self) -> None:
        """The flag turns the batched path on for a multi-group layer."""

        layer = _build_layer(bmm=True)
        assert layer._grouped_proj_bmm is True

    def test_single_group_layer_never_batches(self) -> None:
        """One group has nothing to batch, so the flag must stay inert."""

        layer = _build_layer(bmm=True, num_groups=1)
        assert layer._grouped_proj_bmm is False

    def test_parameter_layout_is_unchanged(self) -> None:
        """Flag-on and flag-off layers share one ``state_dict`` exactly."""

        baseline = _build_layer(bmm=False)
        batched = _build_layer(bmm=True)
        assert set(baseline.state_dict()) == set(batched.state_dict())
        # The per-group ScaledLinear modules remain the parameter owners.
        for group in batched.attention_groups:
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                assert getattr(group, name).weight.requires_grad


class GroupedProjBmmLayerTest:
    """Layer-level equivalence against the per-group gathered loop."""

    def test_matches_per_group_loop(self) -> None:
        """Random membership, causal mask: outputs match the loop."""

        baseline, batched = _build_layer_pair()
        x, pos, base, router = _dispatch_inputs()

        with torch.no_grad():
            expected = _run_attention(baseline, x, base, router, pos)
            actual = _run_attention(batched, x, base, router, pos)

        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_padding_matches_per_group_loop(self) -> None:
        """Padded (invalid) trailing tokens stay equivalent."""

        baseline, batched = _build_layer_pair()
        x, pos, base, router = _dispatch_inputs(num_padded=4, seed=21)
        assert not router.active_mask[-1, -1].any()

        with torch.no_grad():
            expected = _run_attention(baseline, x, base, router, pos)
            actual = _run_attention(batched, x, base, router, pos)

        assert torch.isfinite(expected).all()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_empty_group_matches_per_group_loop(self) -> None:
        """A group with no members anywhere contributes exactly zero.

        The per-group loop takes its ``cap == 0`` probe branch; the batched
        path pads that group to the shared capacity and masks it away. Both
        must yield the same output AND the same (zero) gradients for that
        group's projections.
        """

        baseline, batched = _build_layer_pair()
        x, pos, base, router = _dispatch_inputs(empty_group=2, seed=23)
        assert not router.active_mask[..., 2].any()

        expected = _run_attention(baseline, x, base, router, pos)
        actual = _run_attention(batched, x, base, router, pos)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        expected.square().sum().backward()
        actual.square().sum().backward()
        reference_grads = _attention_grads(baseline)
        batched_grads = _attention_grads(batched)
        for name, reference in reference_grads.items():
            torch.testing.assert_close(
                batched_grads[name], reference, rtol=1e-4, atol=1e-5
            )
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert torch.count_nonzero(batched_grads[f"g2.{name}"]) == 0

    def test_gradients_match_per_group_loop(self) -> None:
        """Every group's q/k/v/out weight gradient matches the loop's."""

        baseline, batched = _build_layer_pair()
        x, pos, base, router = _dispatch_inputs(seed=25)
        x_reference = x.clone().requires_grad_(True)
        x_batched = x.clone().requires_grad_(True)

        _run_attention(
            baseline, x_reference, base, router, pos
        ).square().sum().backward()
        _run_attention(
            batched, x_batched, base, router, pos
        ).square().sum().backward()

        reference_grads = _attention_grads(baseline)
        batched_grads = _attention_grads(batched)
        assert set(reference_grads) == set(batched_grads)
        for name, reference in reference_grads.items():
            assert torch.isfinite(batched_grads[name]).all()
            torch.testing.assert_close(
                batched_grads[name], reference, rtol=1e-4, atol=1e-5
            )
        assert x_reference.grad is not None and x_batched.grad is not None
        torch.testing.assert_close(
            x_batched.grad, x_reference.grad, rtol=1e-4, atol=1e-5
        )

    def test_composes_with_causal_flash(self) -> None:
        """``NESTOR_GATHERED_CAUSAL_FLASH`` verification survives batching.

        The first call verifies the compact-causal identity for all groups at
        once and switches to flash ``is_causal``; the second call runs on the
        verified (mask-free) branch. Both must match the per-group loop.
        """

        baseline, batched = _build_layer_pair(flash=True)
        x, pos, base, router = _dispatch_inputs(seed=27)

        with torch.no_grad():
            expected = _run_attention(baseline, x, base, router, pos)
            first = _run_attention(batched, x, base, router, pos)
            second = _run_attention(batched, x, base, router, pos)

        assert batched._grouped_proj_bmm_causal_ok is True
        # The verdict is mirrored onto the group modules so both paths agree.
        assert all(
            group.gathered_causal_flash_ok
            for group in batched.attention_groups
        )
        torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)

    def test_non_causal_mask_falls_back_permanently(self) -> None:
        """A non-causal base mask must keep the batched masked path."""

        baseline, batched = _build_layer_pair(causal=False, flash=True)
        x, pos, base, router = _dispatch_inputs(causal=False, seed=29)

        with torch.no_grad():
            expected = _run_attention(baseline, x, base, router, pos)
            first = _run_attention(batched, x, base, router, pos)
            second = _run_attention(batched, x, base, router, pos)

        assert batched._grouped_proj_bmm_causal_ok is False
        torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)

    def test_composes_with_dense_full_attn(self) -> None:
        """``NESTOR_DENSE_FULL_ATTN`` (no-padding contract) stays equivalent.

        With that flag the per-group loop runs ``forward_gathered_full``; the
        batched path must reproduce it under the same all-true base mask.
        """

        baseline, batched = _build_layer_pair(causal=False, full_attn=True)
        assert baseline._dispatch_full_attn and batched._dispatch_full_attn
        x, pos, base, router = _dispatch_inputs(causal=False, seed=31)

        with torch.no_grad():
            expected = _run_attention(baseline, x, base, router, pos)
            actual = _run_attention(batched, x, base, router, pos)

        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class GroupedProjBmmBackboneTest:
    """End-to-end equivalence inside the Nestor backbone."""

    def test_causal_backbone_matches_baseline(self) -> None:
        """A grouped causal backbone is unchanged by the flag."""

        config = _dispatch_config()
        baseline, batched = _build_backbone_pair(config)
        assert all(
            layer._grouped_proj_bmm for layer in batched.blocks[0].layers
        )
        torch.manual_seed(9)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            actual = batched(x, pos, attn_mask).pred

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_padded_backbone_matches_baseline(self) -> None:
        """Padded sequences stay equivalent end to end."""

        config = _dispatch_config()
        baseline, batched = _build_backbone_pair(config, flash=True)
        torch.manual_seed(10)
        x = torch.randn(3, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(3, 1)
        pos[1, 6:] = PADDING_TOKEN_POS
        pos[2, 4:] = PADDING_TOKEN_POS
        attn_mask = torch.ones(3, 8, 8, dtype=torch.bool)
        valid = (pos != PADDING_TOKEN_POS).unsqueeze(-1)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            batched(x, pos, attn_mask)  # verification forward
            actual = batched(x, pos, attn_mask).pred

        assert torch.isfinite(expected).all()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual * valid, expected * valid, rtol=1e-5, atol=1e-6
        )

    def test_backbone_gradients_match_baseline(self) -> None:
        """Every backbone parameter gradient survives the batching."""

        config = _dispatch_config()
        baseline, batched = _build_backbone_pair(config)
        torch.manual_seed(12)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        for model in (baseline, batched):
            model.zero_grad(set_to_none=True)
            model(x, pos, attn_mask).pred.square().sum().backward()

        batched_grads = dict(batched.named_parameters())
        for name, parameter in baseline.named_parameters():
            other = batched_grads[name]
            if parameter.grad is None:
                assert other.grad is None, name
                continue
            assert other.grad is not None, name
            torch.testing.assert_close(
                other.grad, parameter.grad, rtol=1e-4, atol=1e-5, msg=name
            )


class GroupedProjBmmReferenceMupTest:
    """The batched path must survive the reference muP transform."""

    def test_reference_mup_forward_matches_baseline(self) -> None:
        """A bias-free reference-muP LM is unchanged by the flag."""

        baseline = _reference_mup_model(bmm=False)
        batched = _reference_mup_model(bmm=True)
        batched.load_state_dict(baseline.state_dict())

        layer = batched.backbone.blocks[0].layers[0]
        assert layer._grouped_proj_bmm
        # reference_mup strips every ScaledLinear bias; the batched path must
        # take its bias-free branch.
        assert layer.attention_groups[0].q_proj.bias is None
        assert layer.attention_groups[0].out_proj.bias is None

        torch.manual_seed(17)
        tokens = torch.randint(0, 64, (2, 9))
        with torch.no_grad():
            expected = baseline(tokens).pred
            actual = batched(tokens).pred

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_reference_mup_tags_and_gradients_survive(self) -> None:
        """muP tags stay on the per-group weights and gradients still match."""

        from nestor.scaling.blocks import get_scaling

        baseline = _reference_mup_model(bmm=False)
        batched = _reference_mup_model(bmm=True)
        batched.load_state_dict(baseline.state_dict())

        for layer in batched.backbone.blocks[0].layers:
            for group in layer.attention_groups:
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    scaling = get_scaling(getattr(group, name).weight)
                    assert scaling is not None
                    assert scaling.optimizer == "adamw"
                    assert scaling.lr_mul == 1 / 8

        torch.manual_seed(19)
        tokens = torch.randint(0, 64, (2, 9))
        for model in (baseline, batched):
            model.zero_grad(set_to_none=True)
            model(tokens).pred.square().sum().backward()

        for layer_a, layer_b in zip(
            baseline.backbone.blocks[0].layers,
            batched.backbone.blocks[0].layers,
        ):
            reference_grads = _attention_grads(layer_a)
            batched_grads = _attention_grads(layer_b)
            for name, reference in reference_grads.items():
                torch.testing.assert_close(
                    batched_grads[name], reference, rtol=1e-4, atol=1e-5
                )


class GroupedProjBmmCompileTest:
    """The flag is meant for compiled training runs."""

    def test_compiled_layer_matches_eager_baseline(self) -> None:
        """``torch.compile(fullgraph=False)`` reproduces the loop's output.

        The batched region itself stays eager (``_maybe_disable_compile``,
        matching the other dispatch-attention helpers), so this checks that it
        composes with a compiled caller rather than that it traces.
        """

        try:
            probe = torch.compile(lambda t: t * 2.0)
            probe(torch.ones(2))
        except Exception as error:  # pragma: no cover - environment-specific
            pytest.skip(f"torch.compile unavailable: {error}")

        baseline, batched = _build_layer_pair(seed=33)
        x, pos, base, router = _dispatch_inputs(seed=35)
        compiled = torch.compile(
            batched._attn_update_general, fullgraph=False
        )

        with torch.no_grad():
            expected = _run_attention(baseline, x, base, router, pos)
            first = compiled(x, base, router, pos)
            second = compiled(x, base, router, pos)

        torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)

    def test_compiled_backbone_matches_eager_baseline(self) -> None:
        """A compiled backbone forward stays equivalent (flash on)."""

        try:
            probe = torch.compile(lambda t: t * 2.0)
            probe(torch.ones(2))
        except Exception as error:  # pragma: no cover - environment-specific
            pytest.skip(f"torch.compile unavailable: {error}")

        config = _dispatch_config()
        baseline, batched = _build_backbone_pair(config, flash=True)
        torch.manual_seed(37)
        x = torch.randn(2, 8, config.input_dim)
        pos = torch.arange(8).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)
        compiled = torch.compile(batched, fullgraph=False)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            first = compiled(x, pos, attn_mask).pred
            second = compiled(x, pos, attn_mask).pred

        torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)
