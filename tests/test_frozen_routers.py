"""Tests for the frozen-router recipe (``nestor.train.frozen_routers``).

Under ``NESTOR_FREEZE_ROUTERS=1`` the router score / noise networks and the
loss-free ``choice_bias`` buffers must survive an optimizer step bit-identically
while every other parameter moves, and the routing modules run without
exploration noise so a train-mode forward equals the eval-mode forward.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nestor.backbones import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModelConfig
from nestor.train.frozen_routers import (
    FREEZE_ROUTERS_ENV,
    freeze_routers,
    freeze_routers_enabled,
    hold_routing_in_eval_mode,
    router_modules,
    router_parameter_ids,
    routing_modules,
)


def _tiny_pool_model(seed: int = 0) -> Model:
    """A tiny causal Nestor LM with noisy routers of every kind.

    Block 0: four columns top-2 with token dispatch, a loss-free-balanced
    column router, a hybrid pool of three public experts top-2 and two
    switch-gated private experts per column (so the public, private and column
    routers all receive gradients); block 1: dense. Every router owns a noise
    MLP, so a train-mode forward differs from an eval-mode forward unless the
    routing is held in eval mode.

    Args:
        seed (int): Construction seed.

    Returns:
        Model: The model in eval mode.
    """

    torch.manual_seed(seed)
    backbone = NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[32, 32],
        causal=True,
        pooling_strides=[1, 1],
        block_num_layers=[2, 2],
        num_groups=[4, 1],
        num_groups_per_token=[2, 1],
        block_num_heads=2,
        block_head_dims=16,
        attention_token_dispatch=True,
        block_mlp_hidden_dims=[[24], [24]],
        block_mlp_activations="swiglu",
        block_residual_mode="attn_res_kimi",
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        router_num_experts_per_group_per_token=1,
        router_loss_free_balancing=True,
        router_bias_update_rate=1e-3,
        router_score_mlp_hidden_dims=[[8], [8]],
        router_noise_mlp_hidden_dims=[[8], [8]],
        block_expert_pool_public_experts=[3, 0],
        block_expert_pool_public_top_k=[2, 1],
        block_expert_pool_private_experts=[2, 0],
        block_expert_pool_private_top_k=[1, 1],
        block_expert_pool_private_gate="switch",
    )
    config = SequenceModelConfig(
        adapter_config=TextAdapterConfig(embed_dim=32, vocab_size=64),
        backbone_config=backbone,
        head_config=LanguageModelingHeadConfig(vocab_size=64),
    )
    return Model.from_config(config).eval()


def _tokens(seed: int = 1, length: int = 24) -> torch.Tensor:
    """Random token ids. Shape: (1, length)."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (1, length), generator=generator)


def _loss(model: Model, tokens: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy of a forward pass."""

    logits = model(tokens).pred[0]  # (L, vocab)
    return F.cross_entropy(logits[:-1].float(), tokens[0, 1:])


class FrozenRoutersSelectionTest:
    """Which parameters the freeze touches."""

    def test_router_parameters_are_exactly_those_under_a_router(self) -> None:
        """Structural detection == the parameters whose name says ``router``."""

        model = _tiny_pool_model()
        routers = router_modules(model)
        assert routers, "the tiny model must own routers"
        frozen = router_parameter_ids(model)
        assert frozen == {id(p) for r in routers for p in r.parameters()}
        for name, p in model.named_parameters():
            assert (id(p) in frozen) == ("router" in name), name

    def test_freeze_report_and_other_parameters_stay_trainable(self) -> None:
        """After the freeze only router parameters have ``requires_grad=False``."""

        model = _tiny_pool_model()
        report = freeze_routers(model)
        frozen = router_parameter_ids(model)
        names = dict(model.named_parameters())
        for name, p in names.items():
            assert p.requires_grad == (id(p) not in frozen), name
        assert report.num_frozen == len(frozen) == len(report.frozen_names)
        assert report.num_trainable == len(names) - report.num_frozen
        assert report.num_total == len(names)
        assert 0 < report.frozen_elements < report.total_elements
        assert all("router" in n for n in report.frozen_names)
        # Attention Q/K/V/O, the experts, the path gate, the norms, the
        # embedding and the head all train.
        for needle in ("q_proj", "k_proj", "v_proj", "out_proj", "experts", "path_gate", "norm", "head"):
            assert any(needle in n and p.requires_grad for n, p in names.items()), needle
        assert "frozen routers" in report.describe()

    def test_construction_frozen_parameter_stays_frozen(self) -> None:
        """A non-router parameter already frozen is not re-enabled."""

        model = _tiny_pool_model()
        name, param = next((n, p) for n, p in model.named_parameters() if "q_proj" in n)
        param.requires_grad_(False)
        report = freeze_routers(model)
        assert not param.requires_grad
        assert name not in report.frozen_names
        assert report.num_trainable == sum(p.requires_grad for p in model.parameters())

    def test_model_without_routers_is_refused(self) -> None:
        """The flag must not be a silent no-op."""

        with pytest.raises(ValueError):
            freeze_routers(nn.Linear(2, 2))


class FrozenRoutersTrainingTest:
    """Gradient flow, the optimizer step and the eval-mode routing."""

    def test_without_the_freeze_an_optimizer_step_moves_router_parameters(self) -> None:
        """The control: routers DO train in the ordinary recipe."""

        model = _tiny_pool_model()
        frozen = router_parameter_ids(model)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.train()
        torch.manual_seed(3)
        _loss(model, _tokens()).backward()
        torch.optim.SGD(model.parameters(), lr=0.5).step()
        moved = [n for n, p in model.named_parameters() if id(p) in frozen and not torch.equal(p, before[n])]
        assert moved

    def test_optimizer_step_moves_only_non_router_parameters(self) -> None:
        """Routers and ``choice_bias`` bit-identical after one SGD step, the rest moved."""

        model = _tiny_pool_model()
        freeze_routers(model)
        frozen = router_parameter_ids(model)
        before_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        before_buffers = {n: b.detach().clone() for n, b in model.named_buffers() if n.endswith("choice_bias")}
        assert before_buffers, "every router owns a choice_bias buffer"

        model.train()
        hold_routing_in_eval_mode(model)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
        _loss(model, _tokens()).backward()
        for n, p in model.named_parameters():
            if id(p) in frozen:
                assert p.grad is None, n
        optimizer.step()

        moved = 0
        for n, p in model.named_parameters():
            if id(p) in frozen:
                assert torch.equal(p.detach(), before_params[n]), n
            elif not torch.equal(p.detach(), before_params[n]):
                moved += 1
        assert moved > 0
        for n, b in model.named_buffers():
            if n in before_buffers:
                assert torch.equal(b, before_buffers[n]), n

    def test_held_routing_makes_the_train_forward_equal_to_the_eval_forward(self) -> None:
        """Noise off: the train-mode forward is the eval forward; unheld it is not."""

        model = _tiny_pool_model()
        tokens = _tokens()
        with torch.no_grad():
            model.eval()
            reference = model(tokens).pred.clone()
            model.train()
            torch.manual_seed(7)
            noisy = model(tokens).pred.clone()
            hold_routing_in_eval_mode(model)
            torch.manual_seed(7)
            held = model(tokens).pred.clone()
        assert torch.equal(held, reference)
        assert not torch.equal(noisy, reference)

    def test_routing_modules_are_switched_without_recursing(self) -> None:
        """Routers, pools and routed layers report ``training=False``; the rest stays in train mode."""

        model = _tiny_pool_model()
        model.train()
        switched = hold_routing_in_eval_mode(model)
        routing = routing_modules(model)
        assert switched == len(routing) > 0
        assert model.training
        for module in routing:
            assert not module.training
        routing_ids = set(map(id, routing))
        experts = [
            m for n, m in model.named_modules()
            if "experts" in n and any(True for _ in m.parameters(recurse=False)) and id(m) not in routing_ids
        ]
        assert experts
        for module in experts:
            assert module.training

    def test_held_routers_still_route_on_the_current_activations(self) -> None:
        """Frozen routers recompute their selections from the input: two inputs, two outputs."""

        model = _tiny_pool_model()
        freeze_routers(model)
        model.train()
        hold_routing_in_eval_mode(model)
        with torch.no_grad():
            reference = model(_tokens(seed=1)).pred.clone()
            again = model(_tokens(seed=1)).pred.clone()
            other = model(_tokens(seed=2)).pred.clone()
        assert torch.equal(reference, again)  # noise off: deterministic
        assert not torch.equal(reference, other)


class FrozenRoutersEnvTest:
    """The environment flag."""

    def test_env_flag_round_trip(self) -> None:
        previous = os.environ.get(FREEZE_ROUTERS_ENV)
        try:
            os.environ.pop(FREEZE_ROUTERS_ENV, None)
            assert not freeze_routers_enabled()
            os.environ[FREEZE_ROUTERS_ENV] = "1"
            assert freeze_routers_enabled()
            os.environ[FREEZE_ROUTERS_ENV] = "0"
            assert not freeze_routers_enabled()
        finally:
            if previous is None:
                os.environ.pop(FREEZE_ROUTERS_ENV, None)
            else:
                os.environ[FREEZE_ROUTERS_ENV] = previous
