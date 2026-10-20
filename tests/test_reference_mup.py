"""Tests for the opt-in variable-width-transformers reproduction path."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nestor.backbones import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.optim import AdamWConfig, MuonAdamWConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModel, SequenceModelConfig
from nestor.scaling.blocks import RMSNorm, ScaledLinear, get_scaling
from nestor.train.optim import build_optimizer


def _reference_model(matrix_optimizer: str = "adamw") -> SequenceModel:
    """Construct a tiny reference-parameterized MoE language model.

    Args:
        matrix_optimizer (str): Reference muP matrix target, ``"adamw"``
            (the reference recipe) or ``"muon"`` (spectral tags kept).

    Returns:
        SequenceModel: The transformed model.
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
            num_groups=[1],
            num_groups_per_token=[1],
            block_num_heads=2,
            block_head_dims=8,
            block_mlp_hidden_dims=[[12]],
            block_mlp_activations="swiglu",
            router_num_experts_per_group=[3],
            router_num_experts_per_group_per_token=2,
            router_score_mlp_hidden_dims=[[8]],
            router_noise_mlp_hidden_dims=[[8]],
        ),
        head_config=LanguageModelingHeadConfig(vocab_size=64),
        reference_mup=ReferenceMupConfig(
            num_layers=2, matrix_optimizer=matrix_optimizer
        ),
    )
    model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model


class ReferenceMupModelTest:
    """Correctness checks for the transformed model and optimizer tags."""

    def test_architecture_and_multipliers(self) -> None:
        """The transformed boundary, norms, routers, and scalars should match."""

        model = _reference_model()
        block = model.backbone.blocks[0]
        layer = block.layers[0]

        assert isinstance(block.output_proj, nn.Identity)
        assert isinstance(model.backbone.final_norm, RMSNorm)
        assert model.backbone.final_norm.eps == 1e-5
        assert layer.reference_residual_multiplier == 0.22
        assert layer.expert_router.noise_mlp is None
        assert len(layer.expert_router.score_mlp.layers) == 1
        assert model.adapter.embedding_multiplier == 12.0
        assert isinstance(model.adapter.norm, nn.Identity)
        assert model.head.logit_multiplier == 1 / 8
        assert all(
            module.bias is None
            for module in model.backbone.modules()
            if isinstance(module, (RMSNorm, ScaledLinear))
        )

    def test_initialization_and_optimizer_groups(self) -> None:
        """Marked matrices use lr/8 while embedding and LM head use base lr."""

        torch.manual_seed(0)
        model = _reference_model()
        layer = model.backbone.blocks[0].layers[0]
        q_proj = layer.attention_groups[0].q_proj
        out_proj = layer.attention_groups[0].out_proj

        assert math.isclose(q_proj.weight.std().item(), 0.1 / math.sqrt(8), rel_tol=0.2)
        assert math.isclose(
            out_proj.weight.std().item(),
            0.1 / math.sqrt(8 * 4),
            rel_tol=0.25,
        )
        scaling = get_scaling(q_proj.weight)
        assert scaling is not None and scaling.lr_mul == 1 / 8
        embedding_scaling = get_scaling(model.adapter.embed.weight)
        assert embedding_scaling is not None
        assert embedding_scaling.lr_mul == 1.0
        assert embedding_scaling.weight_decay is True

        optimizer = build_optimizer(
            AdamWConfig(lr=0.01, weight_decay=0.1),
            model,
        )
        assert {group["lr"] for group in optimizer.param_groups} == {0.00125, 0.01}

    def test_training_forward_is_deterministic(self) -> None:
        """Removing router noise should make repeated training forwards identical."""

        model = _reference_model().train()
        token_ids = torch.randint(0, 64, (2, 5))
        first = model(token_ids).pred
        second = model(token_ids).pred
        assert torch.equal(first, second)

    def test_deterministic_router_supports_bfloat16_autocast(self) -> None:
        """Normalized top-k gates must cast back before scatter under AMP."""

        model = _reference_model().train()
        token_ids = torch.randint(0, 64, (2, 5))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(token_ids)
        assert output.pred.dtype == torch.bfloat16

    def test_reference_residual_is_additive(self) -> None:
        """The reference rule should compute h + 0.22 * branch."""

        layer = _reference_model().backbone.blocks[0].layers[0]
        x = torch.randn(2, 3, 16)
        update = torch.randn_like(x)
        assert torch.allclose(layer._apply_residual(x, update), x + 0.22 * update)


class ReferenceMupMuonPairingTest:
    """The ``matrix_optimizer="muon"`` pairing of reference muP with MuonAdamW."""

    def test_matrices_keep_muon_tags_and_router_gets_base_rate(self) -> None:
        """Matrices stay Muon-owned; deterministic routers move to base AdamW."""

        model = _reference_model(matrix_optimizer="muon")
        layer = model.backbone.blocks[0].layers[0]

        for linear in (
            layer.attention_groups[0].q_proj,
            layer.attention_groups[0].out_proj,
            *layer.group_specific_experts[0][0].layers,
        ):
            scaling = get_scaling(linear.weight)
            assert scaling is not None and scaling.optimizer == "muon"
            d_out, d_in = linear.weight.shape
            assert math.isclose(scaling.muon_scale, math.sqrt(d_out / d_in))

        router_score = layer.expert_router.score_mlp.layers[-1]
        router_scaling = get_scaling(router_score.weight)
        assert router_scaling is not None
        assert router_scaling.optimizer == "adamw"
        assert router_scaling.lr_mul == 1.0

        # The reference embedding / LM-head groups are unchanged by the pairing.
        embedding_scaling = get_scaling(model.adapter.embed.weight)
        assert embedding_scaling is not None and embedding_scaling.lr_mul == 1.0
        head_scaling = get_scaling(model.head.projection.layers[0].weight)
        assert head_scaling is not None
        assert head_scaling.optimizer == "adamw" and head_scaling.lr_mul == 1.0

    def test_muon_adamw_partition_is_populated(self) -> None:
        """MuonAdamW on the paired model owns the matrices at spectral rates."""

        model = _reference_model(matrix_optimizer="muon")
        optimizer = build_optimizer(
            MuonAdamWConfig(lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.1),
            model,
        )
        muon_groups = [
            group for group in optimizer.param_groups if group["method"] == "muon"
        ]
        adamw_groups = [
            group for group in optimizer.param_groups if group["method"] == "adamw"
        ]
        assert muon_groups and adamw_groups
        muon_params = sum(
            param.numel() for group in muon_groups for param in group["params"]
        )
        # q/k/v, out_proj, and the SwiGLU expert matrices are Muon-owned.
        assert muon_params > 0
        for group in muon_groups:
            for param in group["params"]:
                scaling = get_scaling(param)
                assert scaling is not None
                # The builder groups on muon_scale rounded to 8 decimals.
                assert math.isclose(
                    group["lr"], 0.02 * scaling.muon_scale, rel_tol=1e-6
                )
        assert any(
            math.isclose(group["lr"], 3e-4) for group in adamw_groups
        )  # routers / embedding / LM head at the base AdamW rate

        # One real step must move a Muon-owned matrix (Newton-Schulz path).
        probe = muon_groups[0]["params"][0]
        before = probe.detach().clone()
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.grad = torch.randn_like(param) * 1e-3
        optimizer.step()
        assert not torch.equal(probe.detach(), before)

    def test_reference_adamw_target_warns_under_muon_config(self) -> None:
        """The default (AdamW-target) tags leave Muon empty and must warn."""

        model = _reference_model()
        with pytest.warns(UserWarning, match="Muon partition is EMPTY"):
            optimizer = build_optimizer(
                MuonAdamWConfig(lr_muon=0.02, lr_adamw=3e-4), model
            )
        assert all(group["method"] == "adamw" for group in optimizer.param_groups)

    def test_invalid_matrix_optimizer_rejected(self) -> None:
        """Unknown matrix_optimizer values fail fast at config time."""

        with pytest.raises(ValueError, match="matrix_optimizer"):
            ReferenceMupConfig(num_layers=2, matrix_optimizer="sgd")


def _column_reference_model(
    *, matrix_optimizer: str = "muon", group_router_lr: str = "native"
) -> SequenceModel:
    """Construct a tiny reference-parameterized model WITH attention columns.

    Unlike :func:`_reference_model` this has ``num_groups=[4]`` so the block
    group (column) router is live, which is what ``group_router_lr`` acts on.

    Args:
        matrix_optimizer (str): Reference muP matrix target.
        group_router_lr (str): ``"native"`` or ``"base"``.

    Returns:
        SequenceModel: The transformed model.
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
            block_num_heads=4,
            block_head_dims=4,
            block_mlp_hidden_dims=[[12]],
            block_mlp_activations="swiglu",
            router_num_experts_per_group=[3],
            router_num_experts_per_group_per_token=2,
            router_score_mlp_hidden_dims=[[8]],
            router_noise_mlp_hidden_dims=[[8]],
        ),
        head_config=LanguageModelingHeadConfig(vocab_size=64),
        reference_mup=ReferenceMupConfig(
            num_layers=2,
            matrix_optimizer=matrix_optimizer,
            group_router_lr=group_router_lr,
        ),
    )
    model = Model.from_config(config)
    assert isinstance(model, SequenceModel)
    return model


class ReferenceMupGroupRouterLrTest:
    """``ReferenceMupConfig.group_router_lr`` on the attention-column router."""

    def test_native_keeps_the_frozen_readout_multiplier(self) -> None:
        """Default: score head at ``M_l^-0.5 / d_in`` (the regime on record)."""

        model = _column_reference_model()
        block = model.backbone.blocks[0]
        score_head = block.group_router.score_mlp.layers[-1]
        scaling = get_scaling(score_head.weight)
        assert scaling is not None and scaling.optimizer == "adamw"
        # score_lr_mul = block_num_layers ** -0.5 = 2 ** -0.5, d_in = 8.
        assert math.isclose(scaling.lr_mul, (2**-0.5) / 8, rel_tol=1e-9)

    def test_base_moves_both_readouts_to_the_expert_router_rate(self) -> None:
        """Score AND noise read-outs at lr_mul 1; the hidden layer stays Muon."""

        model = _column_reference_model(group_router_lr="base")
        router = model.backbone.blocks[0].group_router
        for mlp in (router.score_mlp, router.noise_mlp):
            assert mlp is not None
            head = get_scaling(mlp.layers[-1].weight)
            assert head is not None and head.optimizer == "adamw"
            assert head.lr_mul == 1.0
            hidden = get_scaling(mlp.layers[0].weight)
            assert hidden is not None and hidden.optimizer == "muon"
        # Same treatment as the expert router.
        expert_router = model.backbone.blocks[0].layers[0].expert_router
        expert_scaling = get_scaling(expert_router.score_mlp.layers[-1].weight)
        assert expert_scaling is not None and expert_scaling.lr_mul == 1.0

    def test_base_readouts_land_in_the_base_rate_adamw_group(self) -> None:
        """Built optimizer puts the read-outs at exactly ``lr_adamw``."""

        model = _column_reference_model(group_router_lr="base")
        optimizer = build_optimizer(
            MuonAdamWConfig(lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.1),
            model,
        )
        head = model.backbone.blocks[0].group_router.score_mlp.layers[-1].weight
        owning = [
            group
            for group in optimizer.param_groups
            if any(param is head for param in group["params"])
        ]
        assert len(owning) == 1
        assert owning[0]["method"] == "adamw"
        assert math.isclose(owning[0]["lr"], 3e-4, rel_tol=1e-9)

    def test_invalid_group_router_lr_rejected(self) -> None:
        """Only the two documented treatments exist."""

        with pytest.raises(ValueError):
            ReferenceMupConfig(num_layers=2, group_router_lr="fast")
