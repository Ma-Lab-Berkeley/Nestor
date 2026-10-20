"""The twelve registered pretraining runs of the scaling figure.

Pins, per rung: the scale entry (d / L = 40, 16 heads of d / 16, MLP 4d, an
8 % warmup, the 2.5x-Chinchilla budget), the batch shapes reproducing the
rung's tokens per step; the two baselines' identical active geometry (dense
attention, Kimi read over four equal blocks, a top-3 bank of width
``round8(4d / 3)``) differing only in the expert count; Nestor's schedule,
hybrid pool, width-contract top-k, column read, and its causal 4k FLOP ratio
against the dense reference; the exact parameter count of every arm; and the
one shared recipe. Tiny-width forwards exercise both model forms on the CPU.
"""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Callable

import torch

from nestor.backbones import Nestor
from nestor.backbones.expert_pool_dispatch import (
    private_expert_hidden_dim,
    public_expert_hidden_dim,
)
from nestor.configs.losses import CausalLMLossConfig
from nestor.configs.optim import MuonAdamWConfig
from nestor.configs.sched import CosineAnnealingConfig
from nestor.experiments.pretrain import (
    BATCH_SHAPE,
    MOE_EXPERTS,
    NESTOR_HEADS,
    NESTOR_LAYERS,
    NESTOR_PRIVATE_EXPERTS,
    NESTOR_TOP_K,
    SCALES,
    TOKEN_REMAP,
    VOCAB_SIZE,
    Scale,
    _backbone,
    baseline_backbone,
    dense_1b,
    dense_100m,
    dense_200m,
    dense_500m,
    moe_1b,
    moe_100m,
    moe_200m,
    moe_500m,
    moe_expert_width,
    nestor_1b,
    nestor_100m,
    nestor_200m,
    nestor_500m,
    nestor_backbone,
    nestor_private_top_k,
)
from nestor.experiments.registry import get_experiment, list_experiments
from nestor.models import Model

S = 4096
ARMS: dict[str, dict[str, Callable]] = {
    "dense": {"100m": dense_100m, "200m": dense_200m, "500m": dense_500m, "1b": dense_1b},
    "moe": {"100m": moe_100m, "200m": moe_200m, "500m": moe_500m, "1b": moe_1b},
    "nestor": {"100m": nestor_100m, "200m": nestor_200m, "500m": nestor_500m, "1b": nestor_1b},
}
# Full parameter counts on record (the trained checkpoints' models).
NUM_PARAMS: dict[str, dict[str, int]] = {
    "dense": {"100m": 141_011_709, "200m": 233_903_269, "500m": 547_500_789, "1b": 1_097_053_509},
    "moe": {"100m": 373_387_389, "200m": 786_607_269, "500m": 2_405_976_309, "1b": 5_494_150_469},
    "nestor": {"100m": 377_505_845, "200m": 761_788_781, "500m": 2_442_638_813, "1b": 5_532_512_845},
}
PRIVATE_TOP_K: dict[str, tuple[int, ...]] = {
    "100m": (2, 1, 1, 1, 1, 1, 1),
    "200m": (2, 2, 1, 1, 1, 1, 1),
    "500m": (2, 2, 2, 2, 2, 1, 1),
    "1b": (2, 2, 2, 2, 2, 2, 2),
}
CAUSAL_4K_RATIO = {"100m": 0.986, "200m": 0.981, "500m": 0.979, "1b": 0.989}
BANK_WIDTH = {"100m": 640, "200m": 856, "500m": 1280, "1b": 1704}


def _job(fn: Callable):
    """First training job of a registered arm (the builder's stdout swallowed)."""

    with contextlib.redirect_stdout(io.StringIO()):
        return fn().jobs[0]


def _blocks(fn: Callable) -> list[dict]:
    """Expanded per-block backbone configs of a registered arm."""

    return _job(fn).model.backbone_config.to_dict()["block_configs"]


def _num_params(fn: Callable) -> int:
    """Full parameter count of a registered arm, built on the meta device."""

    job = _job(fn)
    with torch.device("meta"):
        model = Model.from_config(job.model)
    return sum(p.numel() for p in model.parameters())


def _causal_4k_ratio(scale: Scale, blocks: list[dict]) -> float:
    """Causal 4k body-FLOP ratio of a Nestor arm to its rung's dense reference.

    Reference implementation (slow, explicit): per block, attention
    projections cost ``8 S d w k`` (``w`` = heads x head_dim per column, ``k``
    columns per token), the causal core ``G x 2 x (S k / G)^2 x w``, and the
    hybrid MLP ``6 S d x (active public + active private width)``; the dense
    reference runs ``L`` layers of 16 heads and a ``4d`` MLP.
    """

    d, hd = scale.hidden_size, scale.head_dim
    total = 0.0
    for b in blocks:
        k = b["num_groups_per_token"]
        w = b["block_num_heads"] * hd
        proj = 8 * S * d * w * k
        members = S * k / b["num_groups"]
        core = b["num_groups"] * 2 * members**2 * w
        pw = public_expert_hidden_dim(d, b["expert_pool_public_top_k"])
        qw = private_expert_hidden_dim(
            d, k, b["expert_pool_private_top_k"], with_public_pool=True
        )
        mlp = 6 * S * d * (
            b["expert_pool_public_top_k"] * pw + k * b["expert_pool_private_top_k"] * qw
        )
        total += b["block_num_layers"] * (proj + core + mlp)
    dense = scale.num_layers * (
        8 * S * d * 16 * hd + 2 * S**2 * 16 * hd + 6 * S * d * scale.mlp_hidden
    )
    return total / dense


class RegistryTest:
    """Exactly the twelve cells are registered, named ``<family>_<scale>``."""

    def test_registered_names(self) -> None:
        expected = {f"{family}_{scale}" for family in ARMS for scale in ARMS[family]}
        assert set(list_experiments()) == expected

    def test_experiment_name_matches_function(self) -> None:
        for family, rungs in ARMS.items():
            for scale, fn in rungs.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    exp = fn()
                assert exp.name == f"{family}_{scale}" == fn.__name__
                assert get_experiment(exp.name) is fn


class ScaleTest:
    """Each rung is the proportional member of one family on a 2.5x-Chinchilla budget."""

    def test_geometry_ratios(self) -> None:
        for sc in SCALES.values():
            assert sc.hidden_size / sc.num_layers == 40, sc.name
            assert sc.hidden_size / sc.head_dim == 16, sc.name
            assert sc.mlp_hidden == 4 * sc.hidden_size, sc.name
            assert sc.num_layers % 4 == 0, sc.name

    def test_budgets(self) -> None:
        budgets = {"100m": 5.0e9, "200m": 10.0e9, "500m": 25.0e9, "1b": 50.0e9}
        for name, sc in SCALES.items():
            assert abs(sc.total_tokens / budgets[name] - 1.0) < 1e-3, name
            assert sc.warmup == round(0.08 * sc.max_steps), name

    def test_batch_shapes_reproduce_tokens_per_step(self) -> None:
        for (family, scale), (global_batch, grad_accum) in BATCH_SHAPE.items():
            assert global_batch * grad_accum * S == SCALES[scale].tokens_per_step, (family, scale)
            job = _job(ARMS[family][scale])
            assert job.train_dataloader.batch_size == global_batch, (family, scale)
            assert job.grad_accum_steps == grad_accum, (family, scale)
            assert job.max_steps == SCALES[scale].max_steps, (family, scale)


class BaselineGeometryTest:
    """Dense attention + a top-3 bank; the two baselines differ only in the expert count."""

    BANK_KEYS = (
        "router_num_experts_per_group",
        "router_num_shared_experts",
        "layer_num_experts",
        "block_mlp_hidden_dims",
        "router_num_experts_per_group_per_token",
    )

    def test_moe_differs_from_dense_only_in_the_bank(self) -> None:
        for scale in SCALES:
            dense = _job(ARMS["dense"][scale]).model.backbone_config.to_dict()
            moe = _job(ARMS["moe"][scale]).model.backbone_config.to_dict()
            assert set(dense) == set(moe)
            for key in dense:
                if key != "block_configs":
                    assert dense[key] == moe[key], (scale, key)
            assert len(dense["block_configs"]) == len(moe["block_configs"]) == 4
            for bd, bm in zip(dense["block_configs"], moe["block_configs"]):
                rest_d = {k: v for k, v in bd.items() if k not in self.BANK_KEYS}
                rest_m = {k: v for k, v in bm.items() if k not in self.BANK_KEYS}
                assert rest_d == rest_m, scale

    def test_dense_attention_kimi_read_four_blocks(self) -> None:
        for scale, sc in SCALES.items():
            for family in ("dense", "moe"):
                cfg = _job(ARMS[family][scale]).model.backbone_config.to_dict()
                assert cfg["block_residual_mode"] == "attn_res_kimi", (family, scale)
                blocks = cfg["block_configs"]
                assert [b["block_num_layers"] for b in blocks] == [sc.num_layers // 4] * 4
                for b in blocks:
                    assert b["num_groups"] == 1 and b["num_groups_per_token"] == 1
                    assert b["block_num_heads"] == 16 and b["block_head_dim"] == sc.head_dim
                    assert b["input_dim"] == sc.hidden_size
                    assert b["loss_free_balancing"] is True
                    assert b["expert_pool_private_experts"] == 0

    def test_bank_geometry(self) -> None:
        for scale, sc in SCALES.items():
            assert moe_expert_width(sc) == BANK_WIDTH[scale]
            assert abs(3 * BANK_WIDTH[scale] - sc.mlp_hidden) <= 8, scale
            for b in _blocks(ARMS["moe"][scale]):
                assert b["router_num_experts_per_group"] == MOE_EXPERTS == 24
                assert b["router_num_shared_experts"] == MOE_EXPERTS
                assert b["router_num_experts_per_group_per_token"] == 3
                assert b["block_mlp_hidden_dims"] == [BANK_WIDTH[scale]]
                assert b["block_mlp_activation"] == "swiglu"
            for b in _blocks(ARMS["dense"][scale]):
                assert b["router_num_experts_per_group"] == 1
                assert b["router_num_shared_experts"] == 1
                assert b["router_num_experts_per_group_per_token"] == 1
                assert b["block_mlp_hidden_dims"] == [sc.mlp_hidden]


class NestorGeometryTest:
    """The column schedule, the hybrid pool, the width contract and the column read."""

    def test_schedule_and_depth(self) -> None:
        for scale in SCALES:
            blocks = _blocks(ARMS["nestor"][scale])
            assert tuple(b["num_groups_per_token"] for b in blocks) == NESTOR_TOP_K
            assert tuple(b["block_num_heads"] for b in blocks) == NESTOR_HEADS
            assert tuple(b["block_num_layers"] for b in blocks) == NESTOR_LAYERS[scale]
            assert all(b["num_groups"] == 8 for b in blocks)
            assert blocks[-1]["num_groups_per_token"] == blocks[-1]["num_groups"]
            assert all(b["attention_token_dispatch"] is True for b in blocks)

    def test_heads_follow_round_16_over_k(self) -> None:
        assert NESTOR_HEADS == tuple(round(16 / k) for k in NESTOR_TOP_K)

    def test_hybrid_pool(self) -> None:
        for scale in SCALES:
            for b in _blocks(ARMS["nestor"][scale]):
                assert b["expert_pool_public_experts"] == 8 and b["expert_pool_public_top_k"] == 2
                assert b["expert_pool_private_experts"] == NESTOR_PRIVATE_EXPERTS[scale]
                assert b["expert_pool_private_gate"] == "switch"
                assert b["router_num_experts_per_group"] == 1  # the pool replaces the bank
                assert b["loss_free_balancing"] is True

    def test_private_top_k_from_the_width_contract(self) -> None:
        for scale, sc in SCALES.items():
            assert nestor_private_top_k(sc) == PRIVATE_TOP_K[scale], scale
            blocks = _blocks(ARMS["nestor"][scale])
            assert tuple(b["expert_pool_private_top_k"] for b in blocks) == PRIVATE_TOP_K[scale]
            for b in blocks:
                k, kq = b["num_groups_per_token"], b["expert_pool_private_top_k"]
                width = private_expert_hidden_dim(sc.hidden_size, k, kq, with_public_pool=True)
                top2 = private_expert_hidden_dim(sc.hidden_size, k, 2, with_public_pool=True)
                assert kq == (2 if top2 >= 160 else 1), (scale, k)
                assert width % 8 == 0 and width >= 120, (scale, k)

    def test_column_read(self) -> None:
        for scale in SCALES:
            cfg = _job(ARMS["nestor"][scale]).model.backbone_config
            assert cfg.block_residual_mode == "attn_res_column"
            assert cfg.block_column_include_aggregate is True
            assert cfg.block_column_gain_init == 0.0
            assert cfg.block_column_head_only is True
            assert cfg.block_column_deviation_sources is True
            expanded = cfg.to_dict()
            assert expanded["block_column_head_only"] is True
            assert expanded["block_column_deviation_sources"] is True

    def test_causal_4k_flop_ratio_as_documented_and_under_dense(self) -> None:
        for scale, sc in SCALES.items():
            ratio = _causal_4k_ratio(sc, _blocks(ARMS["nestor"][scale]))
            assert abs(ratio - CAUSAL_4K_RATIO[scale]) < 0.0015, (scale, ratio)
            assert ratio <= 1.0, (scale, ratio)


class ParameterCountTest:
    """Every arm builds to its recorded size; Nestor stores about what the MoE stores."""

    def test_exact_counts(self) -> None:
        for family, rungs in ARMS.items():
            for scale, fn in rungs.items():
                assert _num_params(fn) == NUM_PARAMS[family][scale], (family, scale)

    def test_nestor_stores_the_moe_control_to_within_a_few_percent(self) -> None:
        for scale in SCALES:
            n_moe, n_nestor = NUM_PARAMS["moe"][scale], NUM_PARAMS["nestor"][scale]
            assert abs(n_nestor - n_moe) / n_moe < 0.035, scale
            assert n_nestor > 2 * NUM_PARAMS["dense"][scale], scale


class RecipeTest:
    """One recipe for all twelve arms: reference muP, Muon 0.01 + AdamW 3e-4, cosine."""

    def test_optimizer_scheduler_loss(self) -> None:
        for family, rungs in ARMS.items():
            for scale, fn in rungs.items():
                job = _job(fn)
                sc = SCALES[scale]
                assert job.optimizer == MuonAdamWConfig(
                    lr_muon=0.01, lr_adamw=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
                ), (family, scale)
                assert job.scheduler == CosineAnnealingConfig(
                    T_max=sc.max_steps, warmup_steps=sc.warmup, eta_min=1e-5
                ), (family, scale)
                assert job.loss == CausalLMLossConfig(), (family, scale)
                assert job.max_grad_norm == 1.0 and job.use_amp is True
                assert job.seed == 42 and job.distributed.enabled is True

    def test_reference_mup_and_vocabulary(self) -> None:
        for rungs in ARMS.values():
            for fn in rungs.values():
                job = _job(fn)
                mup = job.model.reference_mup
                assert mup is not None and mup.matrix_optimizer == "muon"
                assert (mup.width_multiplier, mup.embedding_multiplier, mup.residual_multiplier) == (8.0, 12.0, 0.22)
                assert mup.num_layers == sum(b["block_num_layers"] for b in _blocks(fn))
                assert job.model.adapter_config.vocab_size == VOCAB_SIZE == 100_263
                assert job.model.head_config.vocab_size == VOCAB_SIZE
                assert job.model.adapter_config.max_length == job.train_dataset.block_size == 4096
                assert job.train_dataset.token_remap == TOKEN_REMAP == {100_257: 100_261}


class EnvironmentKnobTest:
    """The run-time knobs: RoPE base and weight initialisation from a checkpoint."""

    def test_rope_base_and_init_checkpoint(self) -> None:
        saved = {k: os.environ.get(k) for k in ("DCLM_ROPE_BASE", "DCLM_INIT_CHECKPOINT")}
        try:
            os.environ["DCLM_ROPE_BASE"] = "2000000"
            os.environ["DCLM_INIT_CHECKPOINT"] = "/tmp/some/model_final.pt"
            job = _job(nestor_100m)
            assert job.model.backbone_config.block_rope_bases == 2_000_000.0
            assert str(job.resume_model_checkpoint) == "/tmp/some/model_final.pt"
            assert job.resume_strict is False
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        job = _job(nestor_100m)
        assert job.model.backbone_config.block_rope_bases == 10_000.0
        assert job.resume_model_checkpoint is None and job.resume_strict is True


class TinyForwardTest:
    """Both model forms run forward and backward on the CPU at a toy width."""

    TINY = Scale("tiny", 32, 4, 8, 128, 10, 1, 4096)

    def test_baseline_forms(self) -> None:
        for experts in (1, 24):
            torch.manual_seed(0)
            cfg = baseline_backbone(self.TINY, experts=experts)
            cfg = type(cfg)(**{**cfg.__dict__, "block_mlp_hidden_dims": [[16]] * 4})
            model = Nestor.from_config(cfg)
            x = torch.randn(2, 8, 32)
            pos = torch.arange(8).expand(2, 8)
            out = model(x, pos, None).pred
            assert out.shape == (2, 8, 32) and torch.isfinite(out).all()
            out.square().mean().backward()

    def test_nestor_form(self) -> None:
        torch.manual_seed(0)
        cfg = _backbone(
            self.TINY,
            groups=(8,) * 7, top_k=NESTOR_TOP_K, heads=(1,) * 7, layers=(1,) * 7,
            bank_experts=1, bank_top_k=1, bank_width=128,
            public_experts=8, public_top_k=2, private_experts=2,
            private_top_k=(2, 2, 1, 1, 1, 1, 1),
            block_residual_mode="attn_res_column",
            block_column_include_aggregate=True,
            block_column_gain_init=0.0,
            block_column_head_only=True,
            block_column_deviation_sources=True,
        )
        model = Nestor.from_config(cfg)
        x = torch.randn(2, 16, 32)
        pos = torch.arange(16).expand(2, 16)
        out = model(x, pos, None).pred
        assert out.shape == (2, 16, 32) and torch.isfinite(out).all()
        out.square().mean().backward()
        # The nestor rungs themselves build on the meta device (no allocation).
        with torch.device("meta"):
            for scale in SCALES.values():
                Nestor.from_config(nestor_backbone(scale))
