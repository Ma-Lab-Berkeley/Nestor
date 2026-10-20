"""Loading trained checkpoints back into the registry's models.

The 200m dense and MoE checkpoints on record declare their expert bank as the
group-specific experts of their single attention group, where the registry now
declares the same bank as shared experts: same modules, same shapes, same
forward, different parameter names. Pins: the key remap is a bijection onto
the model's names, the two forms compute identical outputs after it, and the
loader rebuilds a registered experiment and loads a checkpoint strictly on
parameters.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import torch

from nestor.backbones import Nestor, NestorConfig
from nestor.eval.checkpoint import load_experiment_checkpoint, remap_legacy_bank_keys
from nestor.experiments.pretrain import Scale, _backbone, dense_100m
from nestor.models import Model


def _legacy_form(experts: int, top_k: int, width: int, d: int = 32, hd: int = 8) -> NestorConfig:
    """The 200m controls' original declaration: group-specific experts, no dispatch."""

    n = 2
    return NestorConfig(
        input_dim=d, num_blocks=n, feature_dims=[d] * n, causal=True, pooling_strides=[1] * n,
        block_num_layers=[2, 2], num_groups=[1] * n, num_groups_per_token=[1] * n,
        block_num_heads=d // hd, block_head_dims=hd,
        attention_auto_head_split=False, attention_token_dispatch=False,
        block_mlp_hidden_dims=[[width]] * n, block_mlp_activations="swiglu",
        block_residual_mode="attn_res_kimi",
        router_num_experts_per_group=[experts] * n, router_num_shared_experts=[0] * n,
        router_num_experts_per_group_per_token=top_k,
        router_loss_free_balancing=True, router_bias_update_rate=0.001,
        router_score_mlp_hidden_dims=[[128]] * n, router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[[128]] * n, router_noise_mlp_activations="gelu",
    )


def _registry_form(experts: int, top_k: int, width: int) -> NestorConfig:
    """The same bank through the registry's builder (shared experts, dispatch on)."""

    return _backbone(
        Scale("tiny", 32, 4, 8, width, 1, 1, 4096),
        groups=(1, 1), top_k=(1, 1), heads=(4, 4), layers=(2, 2),
        bank_experts=experts, bank_top_k=top_k, bank_width=width,
        block_residual_mode="attn_res_kimi",
    )


class LegacyBankRemapTest:
    """``group_specific_experts.0.`` keys map onto ``shared_experts.`` and nothing else moves."""

    def test_remap_is_a_bijection_onto_the_model_keys(self) -> None:
        for experts, top_k, width in ((24, 3, 16), (1, 1, 64)):
            legacy = Nestor.from_config(_legacy_form(experts, top_k, width))
            registry = Nestor.from_config(_registry_form(experts, top_k, width))
            model_keys = set(registry.state_dict())
            remapped = remap_legacy_bank_keys(legacy.state_dict(), model_keys)
            assert set(remapped) == model_keys
            assert len(remapped) == len(legacy.state_dict())
            moved = [k for k in legacy.state_dict() if k not in model_keys]
            assert moved and all("group_specific_experts.0." in k for k in moved)
            untouched = {k for k in legacy.state_dict() if k in model_keys}
            assert all(k in remapped for k in untouched)

    def test_keys_the_model_already_has_are_left_alone(self) -> None:
        state = {"a.group_specific_experts.0.w": 1, "b": 2}
        assert remap_legacy_bank_keys(state, {"a.group_specific_experts.0.w", "b"}) == state
        assert remap_legacy_bank_keys(state, {"a.shared_experts.w", "b"}) == {"a.shared_experts.w": 1, "b": 2}
        assert remap_legacy_bank_keys(state, {"b"}) == state  # no counterpart: untouched

    def test_forward_is_identical_after_the_remap(self) -> None:
        for experts, top_k, width in ((24, 3, 16), (1, 1, 64)):
            torch.manual_seed(0)
            legacy = Nestor.from_config(_legacy_form(experts, top_k, width)).eval()
            registry = Nestor.from_config(_registry_form(experts, top_k, width)).eval()
            state = remap_legacy_bank_keys(legacy.state_dict(), set(registry.state_dict()))
            missing, unexpected = registry.load_state_dict(state, strict=False)
            assert not missing and not unexpected
            x = torch.randn(2, 16, 32)
            pos = torch.arange(16).expand(2, 16)
            with torch.no_grad():
                a = legacy(x, pos, None).pred
                b = registry(x, pos, None).pred
            torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


class ExperimentCheckpointLoaderTest:
    """``load_experiment_checkpoint`` rebuilds the registered model and loads it strictly."""

    def test_round_trip_and_strictness(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            job = dense_100m().jobs[0]
        torch.manual_seed(1)
        model = Model.from_config(job.model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_final.pt"
            torch.save({"model": model.state_dict(), "step": 7}, path)
            loaded, info = load_experiment_checkpoint("dense_100m", path, device="cpu")
            assert info["step"] == 7 and info["num_params"] == sum(p.numel() for p in model.parameters())
            assert not [k for k in info["missing"] if k in dict(model.named_parameters())]
            assert not loaded.training
            name, reference = next(iter(model.named_parameters()))
            torch.testing.assert_close(dict(loaded.named_parameters())[name], reference)
            # A checkpoint missing parameters is refused unless explicitly allowed.
            partial = {k: v for k, v in model.state_dict().items() if "blocks.0" not in k}
            torch.save({"model": partial}, path)
            try:
                load_experiment_checkpoint("dense_100m", path, device="cpu")
            except SystemExit:
                pass
            else:
                raise AssertionError("missing parameters must be refused")
            _, info = load_experiment_checkpoint("dense_100m", path, device="cpu", allow_partial=True)
            assert info["missing"]


class InitStateLoadingTest:
    """The trainer's non-strict init path (``load_init_state``): remap, cover every parameter."""

    def test_legacy_checkpoint_initialises_the_registry_form_completely(self) -> None:
        from nestor.train.checkpoint import load_init_state

        torch.manual_seed(1)
        legacy = Nestor.from_config(_legacy_form(24, 3, 16)).eval()
        registry = Nestor.from_config(_registry_form(24, 3, 16)).eval()
        loaded, skipped = load_init_state(registry, legacy.state_dict())
        assert not skipped and loaded == len(legacy.state_dict())
        x = torch.randn(2, 16, 32)
        pos = torch.arange(16).expand(2, 16)
        with torch.no_grad():
            a = legacy(x, pos, None).pred
            b = registry(x, pos, None).pred
        assert torch.allclose(a, b, atol=1e-6)

    def test_unknown_keys_are_skipped_and_uncovered_parameters_refused(self) -> None:
        from nestor.train.checkpoint import load_init_state

        model = Nestor.from_config(_registry_form(1, 1, 64))
        state = dict(model.state_dict())
        state["not.a.key"] = torch.zeros(1)
        loaded, skipped = load_init_state(model, state)
        assert skipped == {"not.a.key"} and loaded == len(state) - 1

        first_param = next(iter(dict(model.named_parameters())))
        partial = {k: v for k, v in model.state_dict().items() if k != first_param}
        try:
            load_init_state(model, partial)
        except RuntimeError as exc:
            assert "uninitialised" in str(exc)
        else:
            raise AssertionError("an uncovered parameter must be refused")
        loaded, skipped = load_init_state(model, partial, allow_partial=True)
        assert not skipped and loaded == len(partial)
