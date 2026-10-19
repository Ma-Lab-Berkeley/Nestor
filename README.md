# Nestor

Nestor is a nested sparse transformer: every layer runs a set of attention
*columns* that each token is routed to a few of, a hybrid expert MLP (a small
public bank plus column-private experts), and a depth read that attends over
the completed blocks. This repository holds the model, the training stack, and
the twelve pretraining runs of the paper's scaling figure: Nestor against a
dense transformer and a mixture-of-experts control at 100M, 200M, 500M and 1B
active parameters.

## Install

```bash
uv sync                  # creates .venv with torch, tiktoken, wandb, ...
uv run nestor_cli --list-experiments
```

Triton is optional; the `NESTOR_TRITON_*` kernels are used only when it is
importable. Weights & Biases logging is enabled by `WANDB_PROJECT`; set
`WANDB_MODE=disabled` to turn it off.

## Data

Training reads one flat `.bin` of cl100k token ids. Build it from a local
copy of DCLM-baseline (`global-shard_XX_of_10/local-shard_Y_of_10/*.jsonl.zst`):

```bash
uv run python scripts/preprocess_dclm.py --dclm-root /data/dclm-baseline-1.0 \
    --out data/dclm/dclm_train.bin --target-tokens 11_000_000_000 --workers 48
uv run python scripts/preprocess_dclm.py --dclm-root /data/dclm-baseline-1.0 \
    --out data/dclm/dclm_val.bin --target-tokens 20_000_000 --global-shards 9
```

The held-out split must come from global shards the training corpus never
read: the training corpus consumes the sorted shard list from the first global
shard on, and the paper's held-out set was cut from global shard 9.

## The twelve runs

| experiment | model | d / layers | stored params | tokens | held-out CE |
|---|---|---|---|---|---|
| `dense_100m` | dense transformer | 480 / 12 | 141.0M | 5B | 3.5779 |
| `moe_100m` | 24-expert top-3 MoE | 480 / 12 | 373.4M | 5B | 3.3859 |
| `nestor_100m` | Nestor, Q = 8 | 480 / 13 | 377.5M | 5B | 3.3562 |
| `dense_200m` | dense transformer | 640 / 16 | 233.9M | 10B | 3.2931 |
| `moe_200m` | 24-expert top-3 MoE | 640 / 16 | 786.6M | 10B | 3.1077 |
| `nestor_200m` | Nestor, Q = 8 | 640 / 17 | 761.8M | 10B | 3.0923 |
| `dense_500m` | dense transformer | 960 / 24 | 547.5M | 25B | 2.9534 |
| `moe_500m` | 24-expert top-3 MoE | 960 / 24 | 2.406B | 25B | 2.7932 |
| `nestor_500m` | Nestor, Q = 12 | 960 / 25 | 2.443B | 25B | 2.7578 |
| `dense_1b` | dense transformer | 1280 / 32 | 1.097B | 50B | 2.7427 |
| `moe_1b` | 24-expert top-3 MoE | 1280 / 32 | 5.494B | 50B | 2.5921 |
| `nestor_1b` | Nestor, Q = 15 | 1280 / 33 | 5.533B | 50B | 2.5706 |

Every run shares one recipe (reference muP, Muon 0.01 on the matrices and
AdamW 3e-4 elsewhere, cosine to 1e-5 with an 8 % warmup, weight decay 0.1,
4096-token context, bf16). Within a scale the three models have identical
active parameters; the baselines share the dense transformer's forward FLOPs
exactly and Nestor runs at 0.98-0.99x of them at 4k. The full definitions,
including the column schedule and the width contract, are in
`nestor/experiments/pretrain.py`.

Train on one 8-GPU node:

```bash
DCLM_DATA_PATH=data/dclm/dclm_train.bin sbatch scripts/slurm_pretrain.sh nestor_200m
# or directly
DCLM_DATA_PATH=data/dclm/dclm_train.bin NESTOR_DDP=1 NESTOR_BF16=1 NESTOR_COMPILE=1 \
  uv run torchrun --standalone --nproc_per_node=8 -m nestor.cli --experiment nestor_200m
```

Tokens per optimizer step are pinned per scale; the launcher derives the
global batch and gradient accumulation from `PER_RANK` sequences per GPU.
Resubmitting an experiment resumes from its latest checkpoint under
`NESTOR_OUTPUT_ROOT` (default `outputs/`). The `NESTOR_*` flags in
`scripts/slurm_pretrain.sh` are the speed stack the paper runs trained with;
each is an exact reformulation of the eager forward and can be left off.

Evaluate a checkpoint on the held-out split:

```bash
uv run python scripts/eval_heldout.py --experiment nestor_200m \
    --checkpoint outputs/nestor_200m/<run>/checkpoints/model_final.pt \
    --val-bin data/dclm/dclm_val.bin
```

### Checkpoints trained under the research code

Checkpoints trained with the research version of this code load through the
same loader (`nestor.eval.checkpoint`), under the experiment of the same
family and scale. The 200m baselines there declared their expert bank under a
different parameter prefix; the loader renames those keys (the modules, shapes
and forward are identical).

## Layout

```
nestor/
  backbones/      Nestor (nestor.py), the depth reads, the hybrid expert pool, grouped-GEMM dispatch, Triton kernels
  scaling/        muP-tagged layers (RMSNorm, RoPE, scaled linear/MLP) and the Muon + AdamW optimizer
  adapters, heads, models   token embedding, language-modeling head, adapter-backbone-head model
  configs/        dataclass configs for models, data, optimization, evaluation, jobs
  data/           the memory-mapped token dataset and the tokenizer helpers
  train/          trainer (DDP, grad accumulation, checkpointing, MFU), losses, optimizer builder
  eval/           held-out evaluation and checkpoint loading
  experiments/    the registered pretraining runs
scripts/          preprocessing, evaluation, SLURM launchers
tests/            pytest suite (CPU; the CUDA-only tests skip)
```

## Tests

```bash
uv run pytest tests -q
```

`tests/test_pretrain_arms.py` pins the twelve runs (geometry, recipe, FLOP
ratios and exact parameter counts); the remaining files are correctness tests
of the model and the training stack against slow reference implementations.
