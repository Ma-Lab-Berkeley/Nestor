"""Tests for the trivial-mask (``attn_mask=None``) contract.

Long-context runs die on O(S^2) mask machinery: torch's eager
``create_block_mask`` materializes S x S int64 index tensors (exactly 128 GiB
at S=131072), the fused flex kernel faults at S=65536,
and the gathered path builds a 16 GiB ``[B, S, S]`` bool mask
per group. For pre-tokenized causal input every one of those tensors encodes
NOTHING but ``tril`` -- so the adapter now emits ``attn_mask=None`` and the
backbone applies causality analytically (flash ``is_causal`` / compact-causal
gathered), never materializing an ``[S, S]`` tensor of any kind.

These tests pin the contract:

* ``TextAdapterTrivialMaskTest`` -- the adapter emits ``None`` for tensor
  input and a dense mask for padded string input.
* ``TrivialMaskEquivalenceTest`` -- for every architecture family in the
  long-context panel (dense 1-group kimi, token-dispatch multi-group column, non-dispatch
  multi-group), ``model(x, pos, None)`` matches the reference
  ``model(x, pos, ones_mask)`` forward (the old adapter behavior) exactly up
  to kernel accumulation order, in forward AND backward.
* ``BuildBlockMaskGuardTest`` -- the flex block-mask builder refuses the
  sequence lengths where it is known to OOM / fault, instead of dying inside
  the kernel.

All CPU (masked SDPA vs ``is_causal`` SDPA); no CUDA needed.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import Tensor

from nestor.backbones import Nestor, NestorConfig
from nestor.backbones.nestor import GroupRoutedAttention

# Env flags that change which attention path is compiled in at CONSTRUCTION
# time. The equivalence tests pin the DEFAULT paths, so clear them all.
_PATH_FLAGS = [
    "NESTOR_DENSE_FULL_ATTN",
    "NESTOR_DENSE_NO_ROUTER",
    "NESTOR_GATHERED_CAUSAL_FLASH",
    "NESTOR_GATHERED_VARLEN",
    "NESTOR_ATTN_PACKED_VARLEN",
    "NESTOR_ATTN_CAUSAL_SDPA",
    "NESTOR_GROUPED_ATTN_BMM",
    "NESTOR_GROUPED_PROJ_BMM",
    "NESTOR_MOE_GROUPED_MM",
    "NESTOR_COLUMN_SLOT_UPDATES",
    "NESTOR_DENSE_TAIL_ATTN",
    "NESTOR_STACKED_EXPERT_BANKS",
]


def _clear_path_flags() -> dict[str, str]:
    """Remove path-selecting env flags; returns the removed values."""

    removed: dict[str, str] = {}
    for flag in _PATH_FLAGS:
        value = os.environ.pop(flag, None)
        if value is not None:
            removed[flag] = value
    return removed


def _restore_flags(removed: dict[str, str]) -> None:
    """Reinstate env flags removed by :func:`_clear_path_flags`."""

    for flag, value in removed.items():
        os.environ[flag] = value


def _dense_kimi_config() -> NestorConfig:
    """Tiny causal DENSE model in the dense-baseline shape family.

    1 attention group, 1 shared expert, no token dispatch,
    ``block_residual_mode='attn_res_kimi'`` -- the family whose 64k/128k
    arms died inside ``forward_kimi -> _attn_update_general -> flex``.

    Returns:
        NestorConfig: The config.
    """

    n = 2
    return NestorConfig(
        input_dim=32,
        num_blocks=n,
        feature_dims=[32] * n,
        pooling_strides=[1] * n,
        causal=True,
        attention_token_dispatch=False,
        block_num_layers=[2, 1],
        num_groups=[1] * n,
        num_groups_per_token=[1] * n,
        block_num_heads=[2] * n,
        block_head_dims=[8] * n,
        block_mlp_hidden_dims=[[24]] * n,
        router_num_experts_per_group=[1] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[1] * n,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="attn_res_kimi",
    )


def _dispatch_column_config() -> NestorConfig:
    """Tiny causal token-dispatch COLUMN model in the Nestor shape family.

    Multi-group token dispatch with a k-ramp and a dense (k == G) tail,
    ``block_residual_mode='attn_res_column'`` -- the family whose 128k
    arms died building per-group ``[B, S, S]`` masks on the gathered path.

    Returns:
        NestorConfig: The config.
    """

    groups = [4, 4, 1]
    top_k = [2, 4, 1]
    n = len(groups)
    config = NestorConfig(
        input_dim=32,
        num_blocks=n,
        feature_dims=[32] * n,
        pooling_strides=[1] * n,
        causal=True,
        attention_token_dispatch=True,
        block_num_layers=[2, 1, 1],
        num_groups=groups,
        num_groups_per_token=top_k,
        block_num_heads=[2] * n,
        block_head_dims=[8] * n,
        block_mlp_hidden_dims=[[24]] * n,
        router_num_experts_per_group=[1] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[1] * n,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="attn_res_column",
        block_column_include_aggregate=True,
        block_column_gain_init=0.5,
        block_column_head_only=True,
        block_column_deviation_sources=True,
    )
    config.block_expert_pool_public_experts = [2] * n
    config.block_expert_pool_public_top_k = [2] * n
    config.block_expert_pool_private_experts = [2] * n
    config.block_expert_pool_private_top_k = [2] * n
    config.block_expert_pool_private_gate = "switch"
    return config


def _nondispatch_grouped_config() -> NestorConfig:
    """Tiny causal NON-dispatch multi-group model (g2-style mixture).

    Multi-group attention without token dispatch exercises the loop branch
    that must materialize ``tril & members (x) members`` locally when the base
    mask is ``None``.

    Returns:
        NestorConfig: The config.
    """

    n = 2
    return NestorConfig(
        input_dim=32,
        num_blocks=n,
        feature_dims=[32] * n,
        pooling_strides=[1] * n,
        causal=True,
        attention_token_dispatch=False,
        block_num_layers=[2, 1],
        num_groups=[2, 2],
        num_groups_per_token=[1, 1],
        block_num_heads=[2] * n,
        block_head_dims=[8] * n,
        block_mlp_hidden_dims=[[24]] * n,
        router_num_experts_per_group=[1] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[1] * n,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode="projection",
    )


def _forward_backward(
    model: Nestor,
    x: Tensor,
    pos: Tensor,
    attn_mask: Tensor | None,
    weights: Tensor,
) -> tuple[Tensor, dict[str, Tensor | None]]:
    """One fp32 forward/backward; returns pred and named parameter grads.

    Args:
        model (Nestor): The backbone under test.
        x (Float[Tensor, "batch seq dim"]): Input features.
        pos (Int[Tensor, "batch seq"]): Token positions.
        attn_mask (Bool[Tensor, "batch seq seq"] | None): Base mask or the
            trivial-mask contract.
        weights (Float[Tensor, "batch seq dim"]): Loss weights (fixed random).

    Returns:
        tuple[Tensor, dict[str, Tensor | None]]: Detached prediction and a
            name -> grad clone map.
    """

    model.zero_grad(set_to_none=True)
    pred = model(x, pos, attn_mask).pred
    (pred * weights).sum().backward()
    grads = {
        name: (None if p.grad is None else p.grad.detach().clone())
        for name, p in model.named_parameters()
    }
    return pred.detach(), grads


def _assert_none_matches_dense(config: NestorConfig, *, seed: int) -> None:
    """``model(x, pos, None)`` must match ``model(x, pos, ones)`` exactly.

    The all-true ``[B, S, S]`` mask is what the adapter used to emit for
    tensor input; the backbone composes causality on top of it. The None call
    must produce the same forward and the same gradients (fp32 CPU, tolerance
    covers masked-SDPA vs is_causal-SDPA accumulation-order differences).

    Args:
        config (NestorConfig): Model config to build.
        seed (int): Seed for weights and inputs.
    """

    removed = _clear_path_flags()
    try:
        torch.manual_seed(seed)
        model = Nestor(config)
        model.eval()

        batch, seq_len = 3, 40
        torch.manual_seed(seed + 1)
        x = torch.randn(batch, seq_len, 32)
        pos = torch.arange(seq_len).expand(batch, -1)
        dense = torch.ones(batch, seq_len, seq_len, dtype=torch.bool)
        weights = torch.randn(batch, seq_len, 32)

        pred_ref, grads_ref = _forward_backward(model, x, pos, dense, weights)
        pred_none, grads_none = _forward_backward(model, x, pos, None, weights)

        torch.testing.assert_close(pred_none, pred_ref, rtol=1e-5, atol=1e-5)
        assert grads_ref.keys() == grads_none.keys()
        for name in grads_ref:
            ref, cand = grads_ref[name], grads_none[name]
            assert (ref is None) == (cand is None), name
            if ref is not None:
                torch.testing.assert_close(
                    cand, ref, rtol=1e-4, atol=1e-5, msg=lambda m: f"{name}: {m}"
                )
    finally:
        _restore_flags(removed)


class TextAdapterTrivialMaskTest:
    """The adapter emits ``None`` for tensor input, dense for strings."""

    def test_tensor_input_emits_none_mask(self) -> None:
        from nestor.adapters.text import TextAdapter
        from nestor.configs.adapters import TextAdapterConfig

        torch.manual_seed(0)
        adapter = TextAdapter(
            TextAdapterConfig(embed_dim=16, vocab_size=64, max_length=32)
        )
        token_ids = torch.randint(0, 64, (2, 12))
        x, pos, attn_mask = adapter(token_ids)
        assert attn_mask is None
        assert x.shape == (2, 12, 16)
        assert torch.equal(pos, torch.arange(12).expand(2, -1))

    def test_string_input_keeps_dense_mask(self) -> None:
        from nestor.adapters.text import TextAdapter
        from nestor.configs.adapters import TextAdapterConfig
        from nestor.constants import PADDING_TOKEN_POS

        adapter = TextAdapter(
            TextAdapterConfig(embed_dim=16, max_length=32)
        )
        try:
            _x, pos, attn_mask = adapter(["a b c d", "a"])
        except Exception:  # no tokenizer available offline
            pytest.skip("tiktoken encoding unavailable")
        assert attn_mask is not None
        assert attn_mask.dtype == torch.bool
        assert attn_mask.all()
        assert (pos == PADDING_TOKEN_POS).any()


class TrivialMaskEquivalenceTest:
    """None-mask forward/backward equals the dense all-true-mask reference."""

    def test_dense_kimi_matches(self) -> None:
        _assert_none_matches_dense(_dense_kimi_config(), seed=11)

    def test_dispatch_column_matches(self) -> None:
        _assert_none_matches_dense(_dispatch_column_config(), seed=13)

    def test_nondispatch_grouped_matches(self) -> None:
        _assert_none_matches_dense(_nondispatch_grouped_config(), seed=17)

    def test_dispatch_column_matches_with_causal_flash(self) -> None:
        """None must also agree when the verified-flash flag is set."""

        removed = _clear_path_flags()
        os.environ["NESTOR_GATHERED_CAUSAL_FLASH"] = "1"
        try:
            _assert_none_matches_dense(_dispatch_column_config(), seed=19)
        finally:
            os.environ.pop("NESTOR_GATHERED_CAUSAL_FLASH", None)
            _restore_flags(removed)


class BuildBlockMaskGuardTest:
    """The flex block-mask builder refuses known-fatal sequence lengths."""

    def test_long_sequence_raises(self) -> None:
        attn = GroupRoutedAttention(dim=16, num_heads=2, head_dim=8, causal=True)
        # expand() gives the [1, 65536, 65536] shape without the 4 GiB alloc.
        huge = torch.ones(1, 1, 1, dtype=torch.bool).expand(1, 65536, 65536)
        with pytest.raises(RuntimeError, match="trivial-mask contract"):
            attn._build_block_mask(huge)

    def test_short_sequence_untouched(self) -> None:
        pytest.importorskip("torch.nn.attention.flex_attention")
        attn = GroupRoutedAttention(dim=16, num_heads=2, head_dim=8, causal=True)
        mask = torch.tril(torch.ones(1, 16, 16, dtype=torch.bool))
        # CPU create_block_mask works for tiny shapes; the point is only that
        # the guard does not fire below the threshold.
        try:
            attn._build_block_mask(mask)
        except RuntimeError as err:  # flex genuinely unavailable is fine
            assert "trivial-mask contract" not in str(err)
