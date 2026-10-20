"""Equivalence tests for fully packed token-dispatch attention.

``NESTOR_ATTN_PACKED_VARLEN=1`` routes
:meth:`~nestor.backbones.nestor.RoutedTransformerLayer._attn_update_general`
through :meth:`_attn_update_packed_varlen`: all groups' routed tokens packed
into one ``[<= B*S*k]`` buffer, per-group projections as one grouped GEMM, and
attention over per-(group, image) varlen segments. No capacity, no padding --
the per-group loop (flag off) is the reference implementation throughout, and
the flag must change nothing but speed and memory.

Every test builds the SAME model twice under a fixed seed -- once per flag
state -- so the attention path is the only difference (the pattern of
``test_grouped_proj_bmm``). Eval mode keeps the router noise off, so routing is
deterministic and the comparison is exact up to bf16 accumulation order.

The checkpointing test is the critical regression: the per-group varlen
integration crashed in real training because its dispatch depended on runtime
state that flips inside the forward being checkpointed, so the recompute took a
different branch ("recomputed values have different metadata"). The packed
path's gate is static configuration, which this test proves survives a
checkpointed backward.

CUDA only: ``varlen_attn`` and ``torch._grouped_mm`` have no CPU kernels.
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="varlen/grouped_mm are CUDA-only"
)

_FLAG = "NESTOR_ATTN_PACKED_VARLEN"
_CKPT = "NESTOR_ACTIVATION_CHECKPOINT"


class _FlagEnv:
    """Set/restore real env flags around a block (no monkeypatching).

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


def _build_model(*, packed: bool, head_dim: int = 16, num_heads: int = 2):
    """Build a tiny causal grouped token-dispatch LM on CUDA.

    Block 0 has 4 attention groups with top-2 routing, so the packed path's
    grouped GEMMs and varlen segments are actually exercised.

    Args:
        packed (bool): Whether ``NESTOR_ATTN_PACKED_VARLEN`` is set during
            construction (the flag is read in ``__init__``).
        head_dim (int): Per-head width; 60 exercises the %8 pad branch.

    Returns:
        torch.nn.Module: The model on CUDA, eval mode, fixed seed 13.
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
            block_num_heads=num_heads,
            block_head_dims=head_dim,
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
    with _FlagEnv(**{_FLAG: "1" if packed else "0"}):
        torch.manual_seed(13)
        model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model.cuda().eval()


def _flag_is_live(model: torch.nn.Module) -> bool:
    """Whether any layer actually enabled the packed path."""

    return any(
        getattr(m, "_attn_packed_varlen", False) for m in model.modules()
    )


class PackedVarlenLayerEquivalenceTest:
    """The packed layer path must reproduce the per-group loop."""

    @pytest.mark.parametrize(
        "num_heads, head_dim",
        [
            (2, 16),  # even width, %8-native
            (2, 60),  # head-dim %8 pad branch
            (3, 60),  # ODD heads x 60 -> inner width 180: the production 500m
                      # shape whose 360-byte strides the grouped-GEMM kernel
                      # rejects without the width-8 pad. The first A/B failed
                      # exactly here while even-head tests passed.
        ],
    )
    def test_forward_matches_per_group_loop(
        self, num_heads: int, head_dim: int
    ) -> None:
        """Same weights, same tokens: packed output == loop output."""

        ref = _build_model(packed=False, head_dim=head_dim, num_heads=num_heads)
        got = _build_model(packed=True, head_dim=head_dim, num_heads=num_heads)
        assert not _flag_is_live(ref) and _flag_is_live(got)

        torch.manual_seed(7)
        tokens = torch.randint(0, 64, (3, 40), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_ref = ref(tokens).pred
            out_got = got(tokens).pred

        assert torch.isfinite(out_got).all()
        torch.testing.assert_close(
            out_got.float(), out_ref.float(), rtol=3e-2, atol=3e-2
        )

    def test_gradients_match_per_group_loop(self) -> None:
        """Parameter gradients agree, group by group.

        The packed path stacks weights per forward; autograd must route every
        gradient back to the owning group's parameters, or training would
        silently diverge.
        """

        grads: list[dict[str, torch.Tensor]] = []
        for packed in (False, True):
            model = _build_model(packed=packed)
            model.train()  # grads; router noise is seeded identically below
            torch.manual_seed(7)
            tokens = torch.randint(0, 64, (3, 40), device="cuda")
            torch.manual_seed(99)  # router noise draw
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(tokens).pred.float().pow(2).mean().backward()
            grads.append(
                {
                    name: p.grad.detach().float().clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
            )

        assert grads[0].keys() == grads[1].keys(), "different params got grads"
        for name in grads[0]:
            torch.testing.assert_close(
                grads[1][name],
                grads[0][name],
                rtol=6e-2,
                atol=6e-2,
                msg=lambda m, _n=name: f"{_n}: {m}",
            )

    def test_survives_checkpointed_backward(self) -> None:
        """forward+backward under activation checkpointing must not raise.

        This is the failure mode that killed the per-group varlen integration
        in real training: a dispatch keyed on runtime-verified state diverges
        between a checkpointed forward and its recompute. The packed path's
        static gate cannot, which this asserts end to end.
        """

        with _FlagEnv(**{_CKPT: "1"}):
            model = _build_model(packed=True)
            model.train()
            torch.manual_seed(7)
            tokens = torch.randint(0, 64, (3, 40), device="cuda")
            torch.manual_seed(99)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens).pred.float().pow(2).mean()
            loss.backward()  # CheckpointError here == regression
        assert torch.isfinite(loss).all()

    def test_flag_off_is_inert(self) -> None:
        """Without the flag no layer enables the packed path."""

        model = _build_model(packed=False)
        assert not _flag_is_live(model)


class StackedAttnWeightsTest:
    """Stacked weight ownership must change nothing but the parameter layout.

    ``NESTOR_STACKED_ATTN_WEIGHTS=1`` moves the group projections into the
    stacked tensors the packed path consumes. Values are copied from the
    normally initialized groups, so a fixed seed gives bit-identical weights;
    forward outputs and the gradients (mapped through the stacking layout)
    must match the unstacked packed model exactly.
    """

    def test_forward_and_grads_match_unstacked(self) -> None:
        """Same seed, flag on vs off: same outputs, same mapped gradients."""

        ref = _build_model(packed=True, num_heads=3, head_dim=60)
        with _FlagEnv(NESTOR_STACKED_ATTN_WEIGHTS="1"):
            stk = _build_model(packed=True, num_heads=3, head_dim=60)
        assert any(
            getattr(m, "_stacked_attn", False) for m in stk.modules()
        ), "stacked mode did not engage"
        assert len(list(stk.parameters())) < len(list(ref.parameters()))

        torch.manual_seed(7)
        tokens = torch.randint(0, 64, (2, 24), device="cuda")
        for model in (ref, stk):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(tokens).pred.float()
            if model is ref:
                out_ref = out
            else:
                out_stk = out
            out.pow(2).mean().backward()
        torch.testing.assert_close(out_stk, out_ref, rtol=2e-2, atol=2e-2)

        layers_ref = [
            m for m in ref.modules() if hasattr(m, "attention_groups")
            and getattr(m, "_attn_packed_varlen", False)
        ]
        layers_stk = [
            m for m in stk.modules() if getattr(m, "_stacked_attn", False)
        ]
        assert layers_ref and len(layers_ref) == len(layers_stk)
        for lr, ls in zip(layers_ref, layers_stk):
            groups = list(lr.attention_groups)
            dim = groups[0].dim
            inner = groups[0].num_heads * groups[0].head_dim
            e8 = (inner + 7) // 8 * 8
            expected_wqkv = torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.cat(
                            [
                                g.q_proj.weight.grad,
                                g.k_proj.weight.grad,
                                g.v_proj.weight.grad,
                            ],
                            dim=0,
                        ).t().reshape(dim, 3, inner),
                        (0, e8 - inner),
                    ).reshape(dim, 3 * e8)
                    for g in groups
                ]
            )
            torch.testing.assert_close(
                ls.attn_wqkv_stacked.grad, expected_wqkv, rtol=5e-2, atol=5e-2
            )
            expected_wo = torch.stack(
                [
                    torch.nn.functional.pad(
                        g.out_proj.weight.grad.t(), (0, 0, 0, e8 - inner)
                    )
                    for g in groups
                ]
            )
            torch.testing.assert_close(
                ls.attn_wo_stacked.grad, expected_wo, rtol=5e-2, atol=5e-2
            )

    def test_views_track_device_moves(self) -> None:
        """After ``.cuda()`` the per-group views must alias the moved owners."""

        with _FlagEnv(NESTOR_STACKED_ATTN_WEIGHTS="1"):
            model = _build_model(packed=True)
        layer = next(
            m for m in model.modules() if getattr(m, "_stacked_attn", False)
        )
        g0 = next(iter(layer.attention_groups))
        assert g0.q_proj.weight.device == layer.attn_wqkv_stacked.device
        assert g0.q_proj.weight.device.type == "cuda"
        # The view must alias the stacked storage, not a stale copy.
        assert (
            g0.q_proj.weight._base is not None
            and g0.q_proj.weight._base.data_ptr()
            == layer.attn_wqkv_stacked.data_ptr()
        )
