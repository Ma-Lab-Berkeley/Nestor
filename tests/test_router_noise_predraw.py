"""Router exploration noise pre-drawn OUTSIDE the activation checkpoint.

``Router.forward`` used to draw its ``randn_like`` inside the layer, i.e. inside
the per-layer activation checkpoint; torch 2.10's compiled checkpoint HOP does
not reproduce in-region draws on the backward recompute, so a recomputed layer
routed tokens to different experts than its forward. The block now draws every
layer's noise factor first (``RoutedTransformerLayer.draw_router_noise``) and
hands it in as a checkpoint INPUT. Pins: a provided factor reproduces the
router's own formula; the batched and sequential private-router paths agree in
TRAIN mode when fed the same factor; the pool's slabs have the documented
shapes; and a pooled column-mode backbone in TRAIN mode gives identical outputs
and gradients with checkpointing on and off (same seed -> same draws, and the
recompute reuses them).
"""

from __future__ import annotations

import copy
import os

import torch
from torch.nn import functional as F

from nestor.backbones import Nestor
from nestor.backbones.batched_router import route_private_batched
from nestor.backbones.expert_pools import ExpertPoolMLP
from nestor.backbones.nestor import NestorConfig, Router


def _router(seed: int = 0) -> Router:
    """A small top-2-of-6 switch router with a randomized choice bias."""

    torch.manual_seed(seed)
    router = Router(
        feature_dim=12, num_choices=6, num_choices_per_token=2,
        score_mlp_hidden_dims=[8], score_mlp_activation="gelu",
        noise_mlp_hidden_dims=[8], noise_mlp_activation="gelu",
        loss_free_balancing=True, bias_update_rate=1e-3, gate_mode="switch",
    )
    with torch.no_grad():
        router.choice_bias.normal_(std=0.1)
    return router.train()


def _pooled_column_config() -> NestorConfig:
    """Two causal blocks: a 4-column pooled block (public 2 + private 2 per column) then a dense one."""

    return NestorConfig(
        input_dim=32, num_blocks=2, feature_dims=[32, 32], causal=True,
        pooling_strides=[1, 1], block_num_layers=[2, 2],
        num_groups=[4, 1], num_groups_per_token=[2, 1],
        block_num_heads=2, block_head_dims=16, attention_token_dispatch=True,
        block_mlp_hidden_dims=[[24], [24]], block_mlp_activations="swiglu",
        block_residual_mode="attn_res_kimi",
        router_num_experts_per_group=[1, 1], router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_loss_free_balancing=True, router_bias_update_rate=1e-3,
        router_score_mlp_hidden_dims=[[8], [8]], router_noise_mlp_hidden_dims=[[8], [8]],
        block_expert_pool_public_experts=[2, 0], block_expert_pool_public_top_k=[1, 1],
        block_expert_pool_private_experts=[2, 0], block_expert_pool_private_top_k=[1, 1],
    )


class RouterNoiseArgumentTest:
    """A provided factor reproduces ``randn_like * softplus(noise_mlp(x))`` exactly."""

    def test_provided_noise_matches_formula(self) -> None:
        """``forward(noise=eps)`` equals the noiseless scores plus ``eps * softplus``."""

        router = _router()
        torch.manual_seed(1)
        x = torch.randn(2, 5, 12)
        mask = torch.ones(2, 5, dtype=torch.bool)
        eps = router.draw_noise(2, 5, x.device, x.dtype)
        assert eps.shape == (2, 5, 6)
        out = router(x, mask, noise=eps)
        with torch.no_grad():
            scores = router.score_mlp(x) + eps * F.softplus(router.noise_mlp(x))
        torch.testing.assert_close(out.scores, scores)
        # The draw itself is what the router would have drawn from the same seed.
        torch.manual_seed(7)
        drawn = router(x, mask)
        torch.manual_seed(7)
        eps2 = torch.randn_like(drawn.scores)
        torch.testing.assert_close(drawn.scores, router.score_mlp(x) + eps2 * F.softplus(router.noise_mlp(x)))


class BatchedVersusSequentialTrainModeTest:
    """With one shared factor, the batched and sequential private paths agree in train mode."""

    def test_same_factor_same_routing(self) -> None:
        """Scores, gates and choices match up to GEMM accumulation order."""

        torch.manual_seed(3)
        pool = ExpertPoolMLP(
            dim=24, num_groups=4, num_groups_per_token=2, num_public_experts=3,
            public_top_k=2, num_private_experts=5, private_top_k=2,
            router_score_mlp_hidden_dims=[16], router_score_mlp_activation="gelu",
            router_noise_mlp_hidden_dims=[16], router_noise_mlp_activation="gelu",
            loss_free_balancing=True, bias_update_rate=1e-3, private_gate_mode="switch",
        ).train()
        x = torch.randn(2, 7, 24)
        valid = torch.ones(2, 7, dtype=torch.bool)
        member = torch.zeros(2, 7, 4, dtype=torch.bool)
        member.scatter_(-1, torch.rand(2, 7, 4).topk(2, dim=-1).indices, True)
        noise = pool.draw_router_noise(2, 7, x.device, x.dtype)
        assert noise["private"].shape == (2, 7, 4, 5) and noise["public"].shape == (2, 7, 3)
        batched, _gates, _choices = route_private_batched(
            list(pool.private_routers), x, valid, member, training=True, noise=noise["private"]
        )
        sequential = [
            pool.private_routers[g](x, valid & member[:, :, g], noise=noise["private"][:, :, g])
            for g in range(4)
        ]
        for b, s_ in zip(batched, sequential):
            torch.testing.assert_close(b.scores, s_.scores, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(b.gates, s_.gates, rtol=1e-6, atol=1e-6)
            assert torch.equal(b.topk_indices, s_.topk_indices)
        # And the pool's own _route consumes the same slabs on either path.
        pool_b = copy.deepcopy(pool)
        pool_b.batched_routers = True
        _, seq_out, _ = pool._route(x, valid, member, router_noise=noise)
        _, bat_out, _ = pool_b._route(x, valid, member, router_noise=noise)
        for a, b in zip(seq_out, bat_out):
            assert torch.equal(a.topk_indices, b.topk_indices)


class CheckpointRecomputeTrainModeTest:
    """A pooled column backbone in TRAIN mode: checkpoint on == off, outputs and grads."""

    def _run(self, checkpoint: bool, seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        previous = os.environ.get("NESTOR_ACTIVATION_CHECKPOINT")
        os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "1" if checkpoint else "0"
        try:
            torch.manual_seed(seed)
            model = Nestor(_pooled_column_config()).train()
            torch.manual_seed(seed + 1)
            x = torch.randn(2, 6, 32)
            pos = torch.arange(6).unsqueeze(0).expand(2, 6)
            attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)
            torch.manual_seed(seed + 2)
            feats = model(x, pos, attn_mask).pred
            feats.float().pow(2).mean().backward()
            grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            return feats.detach().clone(), grads
        finally:
            if previous is None:
                os.environ.pop("NESTOR_ACTIVATION_CHECKPOINT", None)
            else:
                os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = previous

    def test_checkpoint_on_equals_off(self) -> None:
        """Same seed -> same pre-drawn noise -> identical routing, outputs and gradients."""

        plain, plain_grads = self._run(False, 21)
        ckpt, ckpt_grads = self._run(True, 21)
        torch.testing.assert_close(ckpt, plain, rtol=1e-6, atol=1e-6)
        assert len(plain_grads) == len(ckpt_grads) and len(plain_grads) > 0
        for a, b in zip(plain_grads, ckpt_grads):
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
