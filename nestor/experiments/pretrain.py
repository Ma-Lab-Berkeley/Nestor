"""The paper's pretraining runs: Nestor against a dense transformer and a MoE at four scales.

Twelve registered experiments, one per cell of the pretraining scaling figure::

    family    100m          200m          500m          1b
    dense     dense_100m    dense_200m    dense_500m    dense_1b
    moe       moe_100m      moe_200m      moe_500m      moe_1b
    nestor    nestor_100m   nestor_200m   nestor_500m   nestor_1b

Every cell shares one recipe: DCLM-baseline tokens (cl100k_base), a 4096-token
context, a 2.5x-Chinchilla token budget per scale, reference muP with Muon
(lr 0.01, cosine to 1e-5, 8 % warmup) on the matrices and AdamW (3e-4) on the
embeddings, routers, and gains, weight decay 0.1, bf16 autocast, seed 42.
Within a scale the three families have identical active parameters and the
baselines share the dense transformer's forward FLOPs exactly; Nestor sits at
0.98-0.99x of them at 4k and falls to ~0.8x at 64k.

The two baselines are dense-attention decoders (one attention group of 16
heads) over four equal blocks joined by the Kimi depth read
(``attn_res_kimi``), whose MLP is a bank of ``E`` SwiGLU experts routed top-3
at width ``round8(4d / 3)``:

* ``dense``: ``E = 1`` (one SwiGLU of width ``4d`` per layer, unit gate).
* ``moe``: ``E = 24`` (8x the dense MLP stored, the same width active).

Nestor keeps the token width and head width of its rung and replaces the
dense attention by ``G = 8`` attention columns of which a token is routed to
``k`` per block along the schedule ``k = (3, 4, 5, 6, 6, 7, 8)`` (``heads per
column = round(16 / k)``), ending on a dense ``k = G`` tail. Its MLP is the
hybrid pool: ``P = 8`` public experts routed top-2 plus ``Q`` private experts
per column (switch-gated, top-2 while the width contract keeps them at least
160 wide, else top-1), a learned path gate mixing the two. Blocks are joined by
the column depth read (deviation sources, head-only, gains from zero). ``Q``
is 8 at 100m / 200m and 12 / 15 at 500m / 1b so stored parameters match the
MoE control to within 2 %.

Held-out cross-entropy on record (the paper's held-out DCLM split, seed 42)::

    family    100m     200m     500m     1b
    dense     3.5779   3.2931   2.9534   2.7427
    moe       3.3859   3.1077   2.7932   2.5921
    nestor    3.3562   3.0923   2.7578   2.5706

Everything that varies between a smoke run and the full run is read from the
environment so one definition serves both:

* ``DCLM_DATA_PATH``      path to the flat token ``.bin`` (see scripts/preprocess_dclm.py)
* ``DCLM_BLOCK_SIZE``     context length (default 4096)
* ``DCLM_ROPE_BASE``      RoPE theta for every block (default 10000)
* ``DCLM_INIT_CHECKPOINT`` initialise the weights from a ``model_*.pt`` (non-strict;
                          optimizer, schedule and step start fresh); auto-resume still
                          wins once the run has checkpoints of its own
* ``DCLM_GLOBAL_BATCH``   sequences per optimizer micro-step, summed across ranks
* ``DCLM_GRAD_ACCUM``     gradient-accumulation steps (multiplies the global batch)
* ``DCLM_MAX_STEPS``      optimizer steps (default = the scale's 2.5x Chinchilla budget)
* ``DCLM_WARMUP_STEPS``   linear warmup steps
* ``DCLM_LR_MUON`` / ``DCLM_LR_ADAMW`` / ``DCLM_WEIGHT_DECAY``
* ``DCLM_SEED`` / ``DCLM_RUN_SUFFIX`` / ``DCLM_CKPT_FREQ`` / ``DCLM_LOG_FREQ`` / ``DCLM_NUM_WORKERS``
* ``NESTOR_OUTPUT_ROOT`` / ``WANDB_PROJECT`` / ``WANDB_ENTITY`` / ``WANDB_TAGS``

The speed stack (bf16, DDP, torch.compile, the exact fast paths) is engaged
through the ``NESTOR_*`` env flags read by the trainer and the backbone; see
``scripts/slurm_pretrain.sh``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from nestor.backbones import NestorConfig
from nestor.backbones.expert_pool_dispatch import (
    private_expert_hidden_dim,
    round_to_multiple_of_8,
)
from nestor.configs.adapters import TextAdapterConfig
from nestor.configs.data import DataLoaderConfig, DclmDatasetConfig
from nestor.configs.distributed import DistributedConfig
from nestor.configs.experiment import ExperimentConfig
from nestor.configs.jobs import TrainJobConfig
from nestor.configs.losses import CausalLMLossConfig
from nestor.configs.optim import MuonAdamWConfig
from nestor.configs.reference_mup import ReferenceMupConfig
from nestor.configs.sched import CosineAnnealingConfig
from nestor.experiments.registry import register_experiment
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import SequenceModelConfig

# --- data ---------------------------------------------------------------------
TOKENIZER = "cl100k_base"
"""str: tiktoken encoding the token ``.bin`` was produced with."""
VOCAB_SIZE = 100_263
"""int: Embedding / unembedding rows. The reference recipe's vocabulary: cl100k's
ordinary tokens plus its end-of-text id remapped from 100257 to 100261."""
TOKEN_REMAP = {100_257: 100_261}
"""dict[int, int]: End-of-text id remap applied to every window read."""
DATA_PATH = Path(os.environ.get("DCLM_DATA_PATH", "data/dclm/dclm_train.bin"))
BLOCK_SIZE = int(os.environ.get("DCLM_BLOCK_SIZE", "4096"))
SERIES_BLOCK_SIZE = 4096
"""int: The pretraining context every budget below is stated at."""

# --- recipe -------------------------------------------------------------------
LR_MUON = 0.01
LR_ADAMW = 3e-4
WEIGHT_DECAY = 0.1
ROUTER_BIAS_UPDATE_RATE = 0.001
"""float: Loss-free balancing step of every router's detached choice bias."""
REFERENCE_MUP = dict(width_multiplier=8.0, embedding_multiplier=12.0, residual_multiplier=0.22)
"""dict: ``m_width``, ``m_emb``, ``m_residual`` of the reference muP parameterization."""


@dataclasses.dataclass(frozen=True)
class Scale:
    """One rung of the scaling ladder: its width, depth and token budget.

    Attributes:
        name (str): Rung name (``"100m"`` ... ``"1b"``).
        hidden_size (int): Token width ``d``; always ``40 x num_layers``.
        num_layers (int): Dense reference depth.
        head_dim (int): Attention head width; ``d / 16``.
        mlp_hidden (int): Dense SwiGLU intermediate width ``4d``.
        max_steps (int): Optimizer steps of the 2.5x-Chinchilla budget.
        warmup (int): Linear warmup steps (8 % of ``max_steps``).
        tokens_per_step (int): Tokens per optimizer step; every arm of the
            rung must reproduce it as ``global_batch * grad_accum * 4096``.
    """

    name: str
    hidden_size: int
    num_layers: int
    head_dim: int
    mlp_hidden: int
    max_steps: int
    warmup: int
    tokens_per_step: int

    @property
    def total_tokens(self) -> int:
        """Tokens seen over the whole run."""

        return self.max_steps * self.tokens_per_step


SCALES: dict[str, Scale] = {
    "100m": Scale("100m", 480, 12, 30, 1920, 19_080, 1526, 262_144),
    "200m": Scale("200m", 640, 16, 40, 2560, 19_080, 1526, 524_288),
    "500m": Scale("500m", 960, 24, 60, 3840, 23_844, 1908, 1_048_576),
    "1b": Scale("1b", 1280, 32, 80, 5120, 23_842, 1907, 2_097_152),
}
"""dict[str, Scale]: The four rungs: 5.0B / 10.0B / 25.0B / 50.0B tokens."""

BATCH_SHAPE: dict[tuple[str, str], tuple[int, int]] = {
    # (family, scale) -> (global_batch, grad_accum); the product x 4096 is the
    # rung's tokens per step. Launchers override the shape per world size via
    # DCLM_GLOBAL_BATCH / DCLM_GRAD_ACCUM; only the product is pinned.
    ("dense", "100m"): (16, 4), ("moe", "100m"): (16, 4), ("nestor", "100m"): (16, 4),
    ("dense", "200m"): (16, 8), ("moe", "200m"): (16, 8), ("nestor", "200m"): (16, 8),
    ("dense", "500m"): (16, 16), ("moe", "500m"): (16, 16), ("nestor", "500m"): (16, 16),
    ("dense", "1b"): (64, 8), ("moe", "1b"): (32, 16), ("nestor", "1b"): (64, 8),
}

# --- baseline geometry --------------------------------------------------------
BASELINE_BLOCKS = 4
"""int: Equal-depth blocks the baselines' Kimi depth read sees."""
BASELINE_HEADS = 16
MOE_EXPERTS = 24
"""int: Bank size of the MoE control (8x the dense MLP stored, top-3 active)."""
BANK_TOP_K = 3
"""int: Experts active per token in the MoE control, and in the dense arm's
``E = 1`` bank the same builder degenerates to a unit top-1 gate."""

# --- Nestor geometry ----------------------------------------------------------
NESTOR_COLUMNS = 8
"""int: Attention columns ``G`` in every block."""
NESTOR_TOP_K = (3, 4, 5, 6, 6, 7, 8)
"""tuple[int, ...]: Columns a token is routed to per block; the tail is dense (``k = G``)."""
NESTOR_HEADS = (5, 4, 3, 3, 3, 2, 2)
"""tuple[int, ...]: Heads per column per block, ``round(16 / k)``."""
NESTOR_LAYERS: dict[str, tuple[int, ...]] = {
    "100m": (1, 1, 2, 2, 2, 3, 2),
    "200m": (2, 2, 2, 3, 3, 3, 2),
    "500m": (3, 3, 3, 4, 4, 5, 3),
    "1b": (3, 4, 4, 6, 6, 6, 4),
}
"""dict[str, tuple[int, ...]]: Layers per block; the depth is filled against the
dense reference's causal 4k FLOPs (0.986 / 0.981 / 0.979 / 0.989x)."""
NESTOR_PUBLIC_EXPERTS = 8
NESTOR_PUBLIC_TOP_K = 2
NESTOR_PRIVATE_EXPERTS: dict[str, int] = {"100m": 8, "200m": 8, "500m": 12, "1b": 15}
"""dict[str, int]: Private experts per column ``Q``; stored-matched to the MoE control."""
NESTOR_MIN_PRIVATE_WIDTH = 160
"""int: A block routes top-2 private experts while they stay at least this wide."""
NESTOR_READ = dict(
    block_residual_mode="attn_res_column",
    block_column_include_aggregate=True,
    block_column_gain_init=0.0,
    block_column_head_only=True,
    block_column_deviation_sources=True,
)
"""dict: The column depth read: every multi-column block exports its aggregate
plus the per-column deviations, the sublayer reads see only the aggregates,
the head read sees everything, and the column gains start at zero."""


def moe_expert_width(scale: Scale) -> int:
    """Per-expert SwiGLU width of the baselines' bank, ``round8(4d / 3)``.

    Args:
        scale (Scale): The rung.

    Returns:
        int: 640 / 856 / 1280 / 1704 at 100m / 200m / 500m / 1b.
    """

    return round_to_multiple_of_8(scale.mlp_hidden / BANK_TOP_K)


def nestor_private_top_k(scale: Scale) -> tuple[int, ...]:
    """Private experts active per selected column, per block, from the width contract.

    Args:
        scale (Scale): The rung (its width sets the private expert widths).

    Returns:
        tuple[int, ...]: ``2`` where ``round8(2d / (2k)) >= 160``, else ``1``.
    """

    return tuple(
        2
        if private_expert_hidden_dim(scale.hidden_size, k, 2, with_public_pool=True)
        >= NESTOR_MIN_PRIVATE_WIDTH
        else 1
        for k in NESTOR_TOP_K
    )


def _backbone(
    scale: Scale,
    *,
    groups: tuple[int, ...],
    top_k: tuple[int, ...],
    heads: tuple[int, ...],
    layers: tuple[int, ...],
    bank_experts: int,
    bank_top_k: int,
    bank_width: int,
    public_experts: int = 0,
    public_top_k: int = 1,
    private_experts: int = 0,
    private_top_k: tuple[int, ...] | None = None,
    **read_kwargs: object,
) -> NestorConfig:
    """A constant-width causal decoder whose attention grouping varies by block.

    Block ``i`` runs ``groups[i]`` attention columns with ``top_k[i]`` active per
    token, each owning ``heads[i]`` heads of the scale's head width; token
    dispatch keeps every column's attention core over its member tokens only.
    The MLP is one bank of ``bank_experts`` SwiGLU experts of width
    ``bank_width`` routed top-``bank_top_k`` (``E = 1`` is the plain dense
    MLP), or, with ``private_experts > 0``, the hybrid public + private pool.
    Every router uses loss-free balancing.

    Args:
        scale (Scale): Width, head width and reference depth.
        groups (tuple[int, ...]): Columns per block.
        top_k (tuple[int, ...]): Active columns per token per block.
        heads (tuple[int, ...]): Heads per column per block.
        layers (tuple[int, ...]): Layers per block.
        bank_experts (int): Bank size (ignored under the hybrid pool, where
            it must be 1).
        bank_top_k (int): Experts active per token in the bank.
        bank_width (int): Per-expert intermediate width of the bank.
        public_experts (int): Public pool size ``P`` (0 = no pool).
        public_top_k (int): Public experts active per token.
        private_experts (int): Private experts per column ``Q`` (0 = no pool).
        private_top_k (tuple[int, ...] | None): Private experts active per
            selected column, per block.
        **read_kwargs: ``block_residual_mode`` and the column-read knobs.

    Returns:
        NestorConfig: The backbone configuration.
    """

    num_blocks = len(top_k)
    assert len(groups) == len(heads) == len(layers) == num_blocks
    assert all(k <= g for k, g in zip(top_k, groups)), "top_k cannot exceed the column count"
    if private_experts:
        assert bank_experts == 1, "the hybrid pool replaces the bank"
        assert private_top_k is not None and len(private_top_k) == num_blocks
    return NestorConfig(
        input_dim=scale.hidden_size,
        num_blocks=num_blocks,
        feature_dims=[scale.hidden_size] * num_blocks,
        causal=True,
        pooling_strides=[1] * num_blocks,
        block_num_layers=list(layers),
        num_groups=list(groups),
        num_groups_per_token=list(top_k),
        block_num_heads=list(heads),
        block_head_dims=scale.head_dim,
        attention_auto_head_split=False,
        attention_token_dispatch=True,
        block_mlp_hidden_dims=[[bank_width]] * num_blocks,
        block_mlp_activations="swiglu",
        # experts_per_group == shared experts: the bank is ONE set of experts
        # every attention column routes into, never a per-column replica.
        router_num_experts_per_group=[bank_experts] * num_blocks,
        router_num_shared_experts=[bank_experts] * num_blocks,
        router_num_experts_per_group_per_token=[bank_top_k] * num_blocks,
        router_loss_free_balancing=True,
        router_bias_update_rate=ROUTER_BIAS_UPDATE_RATE,
        router_score_mlp_hidden_dims=[[128]] * num_blocks,
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[[128]] * num_blocks,
        router_noise_mlp_activations="gelu",
        block_expert_pool_public_experts=[public_experts] * num_blocks,
        block_expert_pool_public_top_k=[public_top_k] * num_blocks,
        block_expert_pool_private_experts=[private_experts] * num_blocks,
        block_expert_pool_private_top_k=(
            list(private_top_k) if private_top_k is not None else [1] * num_blocks
        ),
        block_expert_pool_private_gate="switch" if private_experts else "renorm",
        **read_kwargs,  # type: ignore[arg-type]
    )


def baseline_backbone(scale: Scale, *, experts: int) -> NestorConfig:
    """Dense attention + a top-3 expert bank, Kimi depth read over four blocks.

    Args:
        scale (Scale): The rung.
        experts (int): Bank size: ``1`` is the dense transformer (one ``4d``
            SwiGLU per layer, unit gate), ``24`` the MoE control.

    Returns:
        NestorConfig: The baseline backbone.
    """

    assert scale.num_layers % BASELINE_BLOCKS == 0
    n = BASELINE_BLOCKS
    if experts == 1:
        bank_top_k, bank_width = 1, scale.mlp_hidden
    else:
        bank_top_k, bank_width = BANK_TOP_K, moe_expert_width(scale)
    return _backbone(
        scale,
        groups=(1,) * n, top_k=(1,) * n, heads=(BASELINE_HEADS,) * n,
        layers=(scale.num_layers // n,) * n,
        bank_experts=experts, bank_top_k=bank_top_k, bank_width=bank_width,
        block_residual_mode="attn_res_kimi",
    )


def nestor_backbone(scale: Scale) -> NestorConfig:
    """Nestor at one rung: the column schedule, the hybrid pool and the column read.

    Args:
        scale (Scale): The rung.

    Returns:
        NestorConfig: The Nestor backbone.
    """

    n = len(NESTOR_TOP_K)
    return _backbone(
        scale,
        groups=(NESTOR_COLUMNS,) * n, top_k=NESTOR_TOP_K, heads=NESTOR_HEADS,
        layers=NESTOR_LAYERS[scale.name],
        bank_experts=1, bank_top_k=1, bank_width=scale.mlp_hidden,
        public_experts=NESTOR_PUBLIC_EXPERTS, public_top_k=NESTOR_PUBLIC_TOP_K,
        private_experts=NESTOR_PRIVATE_EXPERTS[scale.name],
        private_top_k=nestor_private_top_k(scale),
        **NESTOR_READ,
    )


def _experiment(family: str, scale: Scale, backbone: NestorConfig) -> ExperimentConfig:
    """Assemble the training job of one cell of the scaling figure.

    Args:
        family (str): ``"dense"``, ``"moe"`` or ``"nestor"``.
        scale (Scale): The rung.
        backbone (NestorConfig): The cell's backbone.

    Returns:
        ExperimentConfig: One training job named ``<family>_<scale>``.

    Raises:
        AssertionError: At the series' context, when the batch shape does not
            reproduce the rung's tokens per step.
    """

    experiment_name = f"{family}_{scale.name}"
    default_global_batch, default_grad_accum = BATCH_SHAPE[(family, scale.name)]
    global_batch = int(os.environ.get("DCLM_GLOBAL_BATCH", default_global_batch))
    grad_accum = int(os.environ.get("DCLM_GRAD_ACCUM", default_grad_accum))
    if BLOCK_SIZE == SERIES_BLOCK_SIZE:
        assert global_batch * grad_accum * BLOCK_SIZE == scale.tokens_per_step, (
            f"{experiment_name}: {global_batch} x {grad_accum} x {BLOCK_SIZE} != "
            f"{scale.tokens_per_step:,} tokens/step -- every {scale.name} arm "
            "must match the budget"
        )
    max_steps = int(os.environ.get("DCLM_MAX_STEPS", scale.max_steps))
    warmup = int(os.environ.get("DCLM_WARMUP_STEPS", scale.warmup))
    run_suffix = os.environ.get("DCLM_RUN_SUFFIX", "").strip()
    run_name = f"{experiment_name}_{run_suffix}" if run_suffix else experiment_name
    init_checkpoint = os.environ.get("DCLM_INIT_CHECKPOINT", "").strip()
    rope_base = os.environ.get("DCLM_ROPE_BASE", "").strip()
    if rope_base:
        backbone = dataclasses.replace(backbone, block_rope_bases=float(rope_base))

    extra_tags = [t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()]
    train_job = TrainJobConfig(
        name=f"{run_name}_train",
        model=SequenceModelConfig(
            adapter_config=TextAdapterConfig(
                tokenizer=TOKENIZER,
                embed_dim=scale.hidden_size,
                max_length=BLOCK_SIZE,
                vocab_size=VOCAB_SIZE,
            ),
            backbone_config=backbone,
            head_config=LanguageModelingHeadConfig(vocab_size=VOCAB_SIZE),
            reference_mup=ReferenceMupConfig(
                matrix_optimizer="muon",
                num_layers=sum(backbone.block_num_layers),  # type: ignore[arg-type]
                **REFERENCE_MUP,
            ),
        ),
        train_dataset=DclmDatasetConfig(
            data_path=DATA_PATH,
            block_size=BLOCK_SIZE,
            tokenizer=TOKENIZER,
            split="train",
            token_remap=dict(TOKEN_REMAP),
        ),
        train_dataloader=DataLoaderConfig(
            batch_size=global_batch,
            num_workers=int(os.environ.get("DCLM_NUM_WORKERS", "8")),
            drop_last=True,
            shuffle=True,
        ),
        optimizer=MuonAdamWConfig(
            lr_muon=float(os.environ.get("DCLM_LR_MUON", LR_MUON)),
            lr_adamw=float(os.environ.get("DCLM_LR_ADAMW", LR_ADAMW)),
            weight_decay=float(os.environ.get("DCLM_WEIGHT_DECAY", WEIGHT_DECAY)),
            betas=(0.9, 0.95),
        ),
        scheduler=CosineAnnealingConfig(T_max=max_steps, warmup_steps=warmup, eta_min=1e-5),
        max_steps=max_steps,
        grad_accum_steps=grad_accum,
        max_grad_norm=1.0,
        use_amp=True,  # bf16 via NESTOR_BF16=1 in the launcher
        loss=CausalLMLossConfig(),
        eval_suite=None,
        eval_frequency=10_000_000,
        checkpoint_frequency=int(os.environ.get("DCLM_CKPT_FREQ", "2000")),
        log_frequency=int(os.environ.get("DCLM_LOG_FREQ", "20")),
        distributed=DistributedConfig(enabled=True),
        wandb_project=os.environ.get("WANDB_PROJECT") or "nestor",
        wandb_entity=os.environ.get("WANDB_ENTITY") or None,
        wandb_run_name=run_name,
        wandb_tags=["pretrain", family, scale.name, *extra_tags],
        seed=int(os.environ.get("DCLM_SEED", "42")),
        # Non-strict so a changed block size / RoPE base (non-persistent
        # buffers) loads; the pipeline's auto-resume takes over once this run
        # has written checkpoints of its own.
        resume_model_checkpoint=Path(init_checkpoint) if init_checkpoint else None,
        resume_strict=not init_checkpoint,
    )
    tokens_per_step = global_batch * grad_accum * BLOCK_SIZE
    print(
        f"[pretrain] {run_name}: d={scale.hidden_size} L={sum(backbone.block_num_layers)} "  # type: ignore[arg-type]
        f"ctx={BLOCK_SIZE} rope={backbone.block_rope_bases} init={init_checkpoint or '-'} | "
        f"{max_steps} steps x {tokens_per_step:,} tok/step = "
        f"{max_steps * tokens_per_step / 1e9:.2f}B tokens | data={DATA_PATH}"
    )
    return ExperimentConfig(
        name=run_name,
        jobs=[train_job],
        output_root=Path(os.environ.get("NESTOR_OUTPUT_ROOT", "./outputs")),
    )


def _dense(scale: str) -> ExperimentConfig:
    return _experiment("dense", SCALES[scale], baseline_backbone(SCALES[scale], experts=1))


def _moe(scale: str) -> ExperimentConfig:
    return _experiment("moe", SCALES[scale], baseline_backbone(SCALES[scale], experts=MOE_EXPERTS))


def _nestor(scale: str) -> ExperimentConfig:
    return _experiment("nestor", SCALES[scale], nestor_backbone(SCALES[scale]))


# --- the twelve cells -----------------------------------------------------------
@register_experiment
def dense_100m() -> ExperimentConfig:
    """Dense transformer, d=480, 12 L, 5B tokens (141.0M params; held-out 3.5779)."""
    return _dense("100m")


@register_experiment
def dense_200m() -> ExperimentConfig:
    """Dense transformer, d=640, 16 L, 10B tokens (233.9M params; held-out 3.2931)."""
    return _dense("200m")


@register_experiment
def dense_500m() -> ExperimentConfig:
    """Dense transformer, d=960, 24 L, 25B tokens (547.5M params; held-out 2.9534)."""
    return _dense("500m")


@register_experiment
def dense_1b() -> ExperimentConfig:
    """Dense transformer, d=1280, 32 L, 50B tokens (1.097B params; held-out 2.7427)."""
    return _dense("1b")


@register_experiment
def moe_100m() -> ExperimentConfig:
    """MoE control at 100m: 24 x w640 top-3 (373.4M params; held-out 3.3859)."""
    return _moe("100m")


@register_experiment
def moe_200m() -> ExperimentConfig:
    """MoE control at 200m: 24 x w856 top-3 (786.6M params; held-out 3.1077)."""
    return _moe("200m")


@register_experiment
def moe_500m() -> ExperimentConfig:
    """MoE control at 500m: 24 x w1280 top-3 (2.406B params; held-out 2.7932)."""
    return _moe("500m")


@register_experiment
def moe_1b() -> ExperimentConfig:
    """MoE control at 1b: 24 x w1704 top-3 (5.494B params; held-out 2.5921)."""
    return _moe("1b")


@register_experiment
def nestor_100m() -> ExperimentConfig:
    """Nestor at 100m: (1,1,2,2,2,3|2) = 13 L, Q=8 (377.5M params; held-out 3.3562)."""
    return _nestor("100m")


@register_experiment
def nestor_200m() -> ExperimentConfig:
    """Nestor at 200m: (2,2,2,3,3,3|2) = 17 L, Q=8 (761.8M params; held-out 3.0923)."""
    return _nestor("200m")


@register_experiment
def nestor_500m() -> ExperimentConfig:
    """Nestor at 500m: (3,3,3,4,4,5|3) = 25 L, Q=12 (2.443B params; held-out 2.7578)."""
    return _nestor("500m")


@register_experiment
def nestor_1b() -> ExperimentConfig:
    """Nestor at 1b: (3,4,4,6,6,6|4) = 33 L, Q=15 (5.533B params; held-out 2.5706)."""
    return _nestor("1b")
