"""Shape contract and fallback tests for the batched grouped projections.

``NESTOR_GROUPED_ATTN_BMM=1`` stacks every attention group's
``q/k/v/out_proj`` weight into one batched GEMM, which is only possible when
the groups are structurally uniform (same ``dim``/``num_heads``/``head_dim``
and the same bias-ness). This file covers the two things the equivalence
suite in ``test_grouped_attn_bmm.py`` does not:

1. the uniformity guard -- :func:`~nestor.backbones.nestor.
   _attention_groups_batchable` and :meth:`~nestor.backbones.nestor.
   RoutedTransformerLayer.refresh_grouped_proj_bmm` must reject mixed-shape
   groups and silently keep the per-group loop rather than raise;
2. the shape contract of the arms this optimization exists for -- the g8t3
   decoder's ``attention_auto_head_split`` splits heads per BLOCK, so its 8
   groups really are uniform and the batched path really does apply -- plus a
   g8t3-shaped backbone (8 groups top-3, causal token dispatch, Kimi block
   AttnRes) running forward+backward twice.
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.nestor import (
    GroupRoutedAttention,
    RoutedTransformerLayer,
    RouterOutput,
    _attention_groups_batchable,
)

_BMM_FLAG = "NESTOR_GROUPED_ATTN_BMM"
_FLASH_FLAG = "NESTOR_GATHERED_CAUSAL_FLASH"


class _FlagEnv:
    """Context manager that sets/clears the Nestor speed-stack env flags.

    Not a mock: it flips the real environment variables the production modules
    read in ``__init__`` and restores the previous values on exit.

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
            **flags (bool): Env variable name -> enabled.
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


def _build_layer(*, bmm: bool, seed: int = 5) -> RoutedTransformerLayer:
    """Build a 4-group causal token-dispatch layer under a given flag set.

    Args:
        bmm (bool): Whether ``NESTOR_GROUPED_ATTN_BMM`` is set.
        seed (int): Seed used for parameter initialization.

    Returns:
        RoutedTransformerLayer: The layer, in eval mode.
    """

    with _FlagEnv(**{_BMM_FLAG: bmm}):
        torch.manual_seed(seed)
        layer = RoutedTransformerLayer(
            dim=32,
            num_experts_per_group=1,
            num_shared_experts=1,
            num_experts_per_group_per_token=1,
            num_groups=4,
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
            causal=True,
            num_groups_per_token=2,
        )
    return layer.eval()


def _replace_group(
    layer: RoutedTransformerLayer,
    group_idx: int,
    *,
    num_heads: int,
    head_dim: int,
    seed: int,
) -> None:
    """Swap one attention group for a real group of a different shape.

    Post-construction surgery of exactly the kind the uniformity guard exists
    for; the replacement is a real :class:`GroupRoutedAttention`, not a stub,
    seeded so two layers get identical replacements.

    Args:
        layer (RoutedTransformerLayer): Layer to mutate in place.
        group_idx (int): Index of the group to replace.
        num_heads (int): Head count of the replacement group.
        head_dim (int): Per-head width of the replacement group.
        seed (int): Seed used for the replacement's initialization.
    """

    torch.manual_seed(seed)
    layer.attention_groups[group_idx] = GroupRoutedAttention(
        dim=layer.dim, num_heads=num_heads, head_dim=head_dim
    ).eval()


def _dispatch_inputs(
    *,
    batch_size: int = 3,
    seq_len: int = 9,
    dim: int = 32,
    num_groups: int = 4,
    seed: int = 11,
) -> tuple[Tensor, Tensor, Tensor, RouterOutput]:
    """Build a causal token-dispatch attention input set for one layer.

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        dim (int): Feature width.
        num_groups (int): Attention groups.
        seed (int): Seed for the random draws.

    Returns:
        tuple[Tensor, Tensor, Tensor, RouterOutput]: ``x`` ``[B, S, dim]``,
        ``pos`` ``[B, S]``, the base mask ``[B, S, S]``, and the router output.
    """

    torch.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    valid = torch.ones(batch_size, seq_len, dtype=torch.bool)
    base = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    members = torch.rand(batch_size, seq_len, num_groups) > 0.45
    members[:, :, 0] = True  # keep group 0 non-empty on every sequence
    gates = torch.rand(batch_size, seq_len, num_groups) * members.to(
        torch.float32
    )
    router_output = RouterOutput(
        scores=torch.zeros(batch_size, seq_len, num_groups),
        softmax_probs=gates,
        gates=gates,
        topk_indices=torch.zeros(batch_size, seq_len, 1, dtype=torch.long),
        active_mask=members,
        valid_token_mask=valid,
    )
    return x, pos, base, router_output


def _g8t3_style_config() -> NestorConfig:
    """A tiny backbone with the g8t3 arms' topology.

    Block 0: 8 attention groups, top-3 per token, ``attention_auto_head_split``
    (6 full heads / 3 active groups -> 2 heads per group), causal token
    dispatch and loss-free balancing. Block 1: dense. Residual mode is the v3
    recipe's per-sublayer Kimi block AttnRes.

    Returns:
        NestorConfig: The tiny g8t3-shaped config.
    """

    return NestorConfig(
        input_dim=48,
        num_blocks=2,
        feature_dims=[48, 48],
        causal=True,
        pooling_strides=[1, 1],
        block_num_layers=[2, 1],
        num_groups=[8, 1],
        num_groups_per_token=[3, 1],
        block_num_heads=6,
        block_head_dims=8,
        attention_auto_head_split=True,
        attention_token_dispatch=True,
        block_mlp_hidden_dims=[[32], [32]],
        block_mlp_activations="swiglu",
        block_residual_mode="attn_res_kimi",
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_score_mlp_hidden_dims=[[8], [8]],
        router_noise_mlp_hidden_dims=[[8], [8]],
        router_loss_free_balancing=True,
        router_bias_update_rate=1e-3,
    )


def _build_backbone(*, bmm: bool, flash: bool = True) -> Nestor:
    """Build the g8t3-shaped backbone under a given flag set.

    Args:
        bmm (bool): Whether ``NESTOR_GROUPED_ATTN_BMM`` is set.
        flash (bool): Whether ``NESTOR_GATHERED_CAUSAL_FLASH`` is set.

    Returns:
        Nestor: The backbone, in eval mode.
    """

    with _FlagEnv(**{_BMM_FLAG: bmm, _FLASH_FLAG: flash}):
        torch.manual_seed(7)
        model = Nestor(_g8t3_style_config())
    return model.eval()


class GroupedAttnBmmUniformityTest:
    """The batched path must refuse (not crash on) non-uniform groups."""

    def test_layer_groups_are_batchable(self) -> None:
        """Groups built by the layer share one shape, so batching applies."""

        layer = _build_layer(bmm=True)
        assert _attention_groups_batchable(layer.attention_groups)
        assert layer._grouped_proj_bmm is True

    def test_mixed_head_counts_are_not_batchable(self) -> None:
        """A group with a different head count breaks the weight stack."""

        layer = _build_layer(bmm=True)
        _replace_group(layer, 1, num_heads=3, head_dim=16, seed=41)
        assert not _attention_groups_batchable(layer.attention_groups)

    def test_mixed_head_dims_are_not_batchable(self) -> None:
        """Same inner width, different head_dim: RoPE tables would differ."""

        layer = _build_layer(bmm=True)
        _replace_group(layer, 2, num_heads=4, head_dim=8, seed=43)
        assert not _attention_groups_batchable(layer.attention_groups)

    def test_mixed_rope_bases_are_not_batchable(self) -> None:
        """One shared ``inv_freq`` table cannot serve two RoPE bases."""

        layer = _build_layer(bmm=True)
        torch.manual_seed(45)
        layer.attention_groups[1] = GroupRoutedAttention(
            dim=layer.dim, num_heads=2, head_dim=16, rope_base=50000.0
        ).eval()
        assert not _attention_groups_batchable(layer.attention_groups)

    def test_mixed_bias_ness_is_not_batchable(self) -> None:
        """One bias-free group breaks the concatenated bias vector.

        ``reference_mup`` strips biases with ``linear.bias = None``; doing it
        to a single group (rather than all of them) must be rejected.
        """

        layer = _build_layer(bmm=True)
        layer.attention_groups[3].q_proj.bias = None
        assert not _attention_groups_batchable(layer.attention_groups)

    def test_non_uniform_layer_falls_back_to_the_loop(self) -> None:
        """The guard turns the flag off and the per-group loop still runs.

        Both layers get the same replacement group, so the fallback output
        must match the reference loop exactly, and the request itself is
        remembered (``_grouped_proj_bmm_requested``) even though the batched
        path is disabled.
        """

        baseline = _build_layer(bmm=False)
        batched = _build_layer(bmm=True)
        batched.load_state_dict(baseline.state_dict())
        _replace_group(baseline, 1, num_heads=3, head_dim=16, seed=47)
        _replace_group(batched, 1, num_heads=3, head_dim=16, seed=47)
        batched.refresh_grouped_proj_bmm()

        assert batched._grouped_proj_bmm_requested is True
        assert batched._grouped_proj_bmm is False

        x, pos, base, router = _dispatch_inputs(seed=49)
        with torch.no_grad():
            expected = baseline._attn_update_general(x, base, router, pos)
            actual = batched._attn_update_general(x, base, router, pos)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_guard_is_load_bearing(self) -> None:
        """Forcing the batched path onto mixed groups does fail.

        Documents WHY the guard exists: without it the weight stack raises, so
        the fallback is a correctness requirement rather than a nicety.
        """

        layer = _build_layer(bmm=True)
        _replace_group(layer, 1, num_heads=3, head_dim=16, seed=51)
        x, pos, base, router = _dispatch_inputs(seed=53)

        with pytest.raises(RuntimeError):
            layer._gathered_proj_bmm_attention(x, base, router, pos)


class G8t3GroupShapeTest:
    """A grouped-then-dense backbone satisfies the batched path's shape contract."""

    def test_g8t3_style_backbone_batches_every_grouped_layer(self) -> None:
        """Every block-0 layer takes the batched path; block 1 stays dense."""

        model = _build_backbone(bmm=True)
        assert all(layer._grouped_proj_bmm for layer in model.blocks[0].layers)
        assert not any(
            layer._grouped_proj_bmm for layer in model.blocks[1].layers
        )
        for layer in model.blocks[0].layers:
            assert len(layer.attention_groups) == 8
            assert all(
                group.num_heads == 2 and group.head_dim == 8
                for group in layer.attention_groups
            )


class G8t3BackboneBmmTest:
    """g8t3-shaped forward/backward under the flag (Kimi AttnRes, flash on)."""

    def test_matches_the_per_group_loop(self) -> None:
        """The flag leaves the g8t3-shaped backbone's output unchanged."""

        baseline = _build_backbone(bmm=False)
        batched = _build_backbone(bmm=True)
        batched.load_state_dict(baseline.state_dict())
        torch.manual_seed(61)
        x = torch.randn(2, 12, 48)
        pos = torch.arange(12).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 12, 12, dtype=torch.bool)

        with torch.no_grad():
            expected = baseline(x, pos, attn_mask).pred
            batched(x, pos, attn_mask)  # compact-causal verification forward
            actual = batched(x, pos, attn_mask).pred

        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_forward_backward_is_deterministic(self) -> None:
        """Two identical passes give bitwise identical outputs and grads.

        Run after a warm-up forward so both measured passes are on the
        verified compact-causal branch (the verification forward itself uses
        the gathered pair mask, which is only close, not bitwise equal).
        """

        model = _build_backbone(bmm=True)
        torch.manual_seed(63)
        x = torch.randn(2, 12, 48)
        pos = torch.arange(12).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 12, 12, dtype=torch.bool)

        with torch.no_grad():
            model(x, pos, attn_mask)  # verification forward

        outputs: list[Tensor] = []
        grads: list[dict[str, Tensor]] = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            pred = model(x, pos, attn_mask).pred
            pred.square().sum().backward()
            outputs.append(pred.detach().clone())
            grads.append(
                {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            )

        assert torch.equal(outputs[0], outputs[1])
        assert set(grads[0]) == set(grads[1])
        for name, first in grads[0].items():
            assert torch.equal(first, grads[1][name]), name

    def test_every_attention_projection_receives_grad(self) -> None:
        """No group's projection may drop out of the graph (DDP contract).

        All 8 groups take part in the two batched GEMMs on every rank, so
        every ``q/k/v/out_proj`` weight gets a gradient even when the router
        never selects that group -- DDP would error on an unused parameter.
        """

        model = _build_backbone(bmm=True)
        torch.manual_seed(65)
        x = torch.randn(2, 12, 48)
        pos = torch.arange(12).unsqueeze(0).repeat(2, 1)
        attn_mask = torch.ones(2, 12, 12, dtype=torch.bool)

        model.zero_grad(set_to_none=True)
        model(x, pos, attn_mask).pred.square().sum().backward()

        for layer_idx, layer in enumerate(model.blocks[0].layers):
            for group_idx, group in enumerate(layer.attention_groups):
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    weight = getattr(group, name).weight
                    assert weight.grad is not None, (
                        f"layer {layer_idx} group {group_idx} {name}"
                    )
                    assert torch.isfinite(weight.grad).all()
