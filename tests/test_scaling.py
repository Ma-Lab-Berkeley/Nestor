"""Tests for the scaling-aware building blocks and the Muon/AdamW optimizer."""

import math

import torch

from nestor.backbones import Nestor, NestorConfig
from nestor.configs.optim import MuonAdamWConfig
from nestor.scaling.blocks import (
    RMSNorm,
    RoPE,
    ScaledEmbedding,
    ScaledLinear,
    convex_residual,
    get_scaling,
)
from nestor.scaling.muon import MuonAdamW, zeropower_via_newtonschulz5
from nestor.train.optim import build_optimizer


def _tiny_config() -> NestorConfig:
    """A small two-block hierarchical config that exercises routing and pooling."""

    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[48, 64],
        pooling_strides=[2, 1],
        block_num_layers=2,
        num_groups=[2, 2],
        num_groups_per_token=[1, 1],
        block_num_heads=2,
        block_head_dims=8,
        block_mlp_hidden_dims=[64],
        block_mlp_activations="gelu",
        router_num_experts_per_group=[3, 3],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=2,
        router_score_mlp_hidden_dims=[],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[],
        router_noise_mlp_activations="gelu",
    )


class TestScaledLinear:
    def test_in_stream_init_and_tag(self) -> None:
        torch.manual_seed(0)
        d_in, d_out = 256, 512
        layer = ScaledLinear(d_in, d_out, role="in_stream")
        # Fan-in initialization: entry std ~ 1/sqrt(d_in).
        assert math.isclose(
            layer.weight.std().item(), 1.0 / math.sqrt(d_in), rel_tol=0.1
        )
        scaling = get_scaling(layer.weight)
        assert scaling is not None and scaling.optimizer == "muon"
        assert math.isclose(scaling.muon_scale, math.sqrt(d_out / d_in), rel_tol=1e-6)

    def test_score_init_and_tag(self) -> None:
        torch.manual_seed(0)
        d_in, d_out = 256, 10
        layer = ScaledLinear(d_in, d_out, role="score")
        # Score head: entry std ~ 1/d_in, a factor d_in below the in-stream rate.
        assert math.isclose(layer.weight.std().item(), 1.0 / d_in, rel_tol=0.15)
        scaling = get_scaling(layer.weight)
        assert scaling is not None and scaling.optimizer == "adamw"
        assert math.isclose(scaling.lr_mul, 1.0 / d_in, rel_tol=1e-6)

    def test_score_extra_lr_mul(self) -> None:
        d_in = 128
        layer = ScaledLinear(d_in, 4, role="score", lr_mul=0.5)
        scaling = get_scaling(layer.weight)
        assert scaling is not None
        assert math.isclose(scaling.lr_mul, 0.5 / d_in, rel_tol=1e-6)

    def test_projection_spectral_norm_and_tag(self) -> None:
        torch.manual_seed(0)
        d_in, d_out = 64, 192
        layer = ScaledLinear(d_in, d_out, role="projection")
        spectral = torch.linalg.matrix_norm(layer.weight, ord=2).item()
        assert math.isclose(spectral, math.sqrt(d_out / d_in), rel_tol=0.05)
        scaling = get_scaling(layer.weight)
        assert scaling is not None and scaling.optimizer == "muon"

    def test_bias_is_base_rate_adamw(self) -> None:
        layer = ScaledLinear(16, 16, role="in_stream", bias=True)
        scaling = get_scaling(layer.bias)
        assert scaling is not None and scaling.optimizer == "adamw"
        assert scaling.lr_mul == 1.0 and scaling.weight_decay is False


class TestRMSNorm:
    def test_unit_rms_output(self) -> None:
        torch.manual_seed(0)
        norm = RMSNorm(128)
        x = torch.randn(4, 7, 128) * 5.0
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_gain_and_bias_tagged_no_decay(self) -> None:
        norm = RMSNorm(32)
        for p in (norm.weight, norm.bias):
            scaling = get_scaling(p)
            assert scaling is not None and scaling.optimizer == "adamw"
            assert scaling.weight_decay is False


class TestRoPE:
    def test_shape_and_norm_preserved(self) -> None:
        torch.manual_seed(0)
        rope = RoPE(head_dim=16, base=10000.0)
        x = torch.randn(2, 4, 5, 16)
        pos = torch.arange(5).unsqueeze(0).expand(2, -1)
        out = rope(x, pos)
        assert out.shape == x.shape
        # A rotation preserves the per-vector norm.
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)

    def test_zero_position_is_identity(self) -> None:
        rope = RoPE(head_dim=8)
        x = torch.randn(1, 1, 3, 8)
        pos = torch.zeros(1, 3, dtype=torch.long)
        assert torch.allclose(rope(x, pos), x, atol=1e-5)


class TestConvexResidual:
    def test_convex_combination(self) -> None:
        h = torch.ones(4)
        update = torch.zeros(4)
        out = convex_residual(h, update, depth_scale=0.25)
        assert torch.allclose(out, torch.full((4,), 0.75))


class TestNewtonSchulz:
    def test_orthogonalization_drives_singular_values_to_one(self) -> None:
        torch.manual_seed(0)
        g = torch.randn(32, 48)
        o = zeropower_via_newtonschulz5(g, steps=8)
        sv = torch.linalg.svdvals(o.float())
        assert torch.allclose(sv, torch.ones_like(sv), atol=0.2)


class TestScaledEmbedding:
    def test_unit_scale_entries_and_tag(self) -> None:
        torch.manual_seed(0)
        emb = ScaledEmbedding(100, 64)
        assert math.isclose(emb.weight.std().item(), 1.0, rel_tol=0.1)
        scaling = get_scaling(emb.weight)
        assert scaling is not None and scaling.optimizer == "adamw"


class TestMuonAdamWBuild:
    def test_every_backbone_param_is_tagged(self) -> None:
        model = Nestor.from_config(_tiny_config())
        untagged = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and get_scaling(p) is None
        ]
        assert untagged == [], f"untagged params: {untagged}"

    def test_partition_into_muon_and_adamw(self) -> None:
        model = Nestor.from_config(_tiny_config())
        optim = build_optimizer(
            MuonAdamWConfig(lr_muon=0.02, lr_adamw=1e-3), model
        )
        assert isinstance(optim, MuonAdamW)
        methods = {g["method"] for g in optim.param_groups}
        assert methods == {"muon", "adamw"}
        # Muon owns only 2D matrices; AdamW owns the rest.
        for group in optim.param_groups:
            if group["method"] == "muon":
                assert all(p.ndim == 2 for p in group["params"])

    def test_step_updates_parameters(self) -> None:
        torch.manual_seed(0)
        model = Nestor.from_config(_tiny_config())
        optim = build_optimizer(
            MuonAdamWConfig(lr_muon=0.02, lr_adamw=1e-3), model
        )
        x = torch.randn(2, 8, 32)
        pos = torch.arange(8).unsqueeze(0).expand(2, -1).contiguous()
        attn_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        before = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        out = model(x, pos, attn_mask)
        loss = out.pred.pow(2).mean()
        loss.backward()
        optim.step()

        changed = [
            name
            for name, p in model.named_parameters()
            if not torch.allclose(p, before[name])
        ]
        assert len(changed) > 0
        assert torch.isfinite(loss)


class TestAdamWParity:
    def test_foreach_adamw_matches_torch(self) -> None:
        """The AdamW half should match torch.optim.AdamW step for step."""

        torch.manual_seed(0)
        p1 = torch.nn.Parameter(torch.randn(4, 5))
        p2 = torch.nn.Parameter(torch.randn(7))
        ref1 = p1.detach().clone().requires_grad_(True)
        ref2 = p2.detach().clone().requires_grad_(True)

        opt = MuonAdamW(
            [
                {
                    "params": [p1, p2],
                    "method": "adamw",
                    "lr": 1e-2,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                }
            ]
        )
        ref = torch.optim.AdamW(
            [ref1, ref2], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

        for _ in range(5):
            g1, g2 = torch.randn(4, 5), torch.randn(7)
            p1.grad, p2.grad = g1.clone(), g2.clone()
            ref1.grad, ref2.grad = g1.clone(), g2.clone()
            opt.step()
            ref.step()

        assert torch.allclose(p1, ref1, atol=1e-5)
        assert torch.allclose(p2, ref2, atol=1e-5)
