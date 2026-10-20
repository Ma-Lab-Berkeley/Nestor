# Nestor

Nestor is a nested sparse transformer: every layer runs a set of attention
*columns* that each token is routed to a few of, a hybrid expert MLP (a small
public bank plus column-private experts), and a depth read that attends over
the completed blocks. This repository holds the model, the training stack, the
twelve pretraining runs of the paper's scaling figure (Nestor against a dense
transformer and a mixture-of-experts control at 100M, 200M, 500M and 1B
active parameters), and the continued-training experiments on those runs:
long-context extension to 64k and domain adaptation with forgetting.

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
different parameter prefix; the loader, and the trainer's initialisation from a
checkpoint (`DCLM_INIT_CHECKPOINT`), rename those keys (the modules, shapes and
forward are identical).

## Continued training

Section 4 of the paper continues the pretrained models in two ways. Both start
from a trained checkpoint with a fresh optimizer and a restarted schedule, and
both are defined in `nestor/continual/`, whose modules hold the protocols,
price the models at any context length (`flops.py`) and print the environment
one run needs for the launchers in `scripts/`.

### Long context extension

Every model is extended from its 4k checkpoint to 8k, 16k, 32k and 64k, each
length in a separate run, on DCLM-baseline documents of at least the target
length taken from global shards the pretraining corpus never read. The
baselines train at equal tokens: 477 steps of 1,048,576 tokens (0.5B), the
rotary base raised to 5e5 / 1e6 / 2e6 / 5e6, Muon 0.0025 and AdamW 7.5e-5
with a 10 % warmup and a cosine, every parameter trained. Nestor costs fewer
FLOPs per token than the MoE control as the context grows (0.94-0.96x at 8k,
0.80-0.84x at 64k), so it also runs at equal FLOPs: `ceil(477 / r)` steps at
the same tokens per step with both peak learning rates scaled by `r`, the
extra steps spent either on new documents or on the documents of the
equal-token run repeated (one shuffled pass over a multiset in which no
sequence appears more than twice; `nestor/data/seen_subset.py`). The read is
the held-out loss over the first `S` tokens of 200 held-out documents of at
least `S` tokens (100 at 64k), with the run's rotary base, in full precision.

```bash
DCLM_ROOT=/data/dclm-baseline-1.0 sbatch scripts/slurm_prep_long_context_data.sh   # corpora, held-out sets, repeated-document bins
EXPERIMENT=nestor_200m LENGTH=32768 SETTING=equal_flops_new \
  INIT=outputs/nestor_200m/<run>/checkpoints/model_final.pt sbatch scripts/slurm_extend.sh
EXPERIMENT=nestor_200m LENGTH=32768 \
  CHECKPOINT=outputs/continual/nestor_200m_ext32768_equal_flops_new sbatch scripts/slurm_eval_long_context.sh
```

`python -m nestor.continual.extension --experiment nestor_200m --length 32768
--setting equal_flops_new` prints a run's budget. The equal-FLOP budgets of the
paper were measured against the MoE control at 100M, 500M and 1B and against
the dense transformer at 200M; the two baselines differ only by the rounding
of the MoE's expert widths, and `--reference dense` reproduces the 200M step
counts (570 and 592 at 32k and 64k, where the MoE reference gives 571 and
593). A run initialised from a checkpoint trained under the research code
must build the model in that file's expert-bank layout
(`NESTOR_STACKED_EXPERT_BANKS=0` for the 200m baselines); the trainer refuses
a layout mismatch.

Held-out loss at each length, before extension (4k) and after (200M Nestor:
seed 42 / seed 43):

| model | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| **100M** | | | | | |
| dense transformer | 3.578 | 3.511 | 3.495 | 3.501 | 3.490 |
| MoE transformer | 3.386 | 3.338 | 3.330 | 3.330 | 3.325 |
| Nestor, equal tokens | 3.356 | 3.311 | 3.302 | 3.302 | 3.296 |
| Nestor, equal FLOPs, repeated | | 3.309 | 3.298 | 3.297 | 3.287 |
| Nestor, equal FLOPs, new | | 3.307 | 3.295 | 3.293 | 3.277 |
| **200M** | | | | | |
| dense transformer | 3.293 | 3.256 | 3.248 | 3.244 | 3.237 |
| MoE transformer | 3.108 | 3.096 | 3.095 | 3.084 | 3.080 |
| Nestor, equal tokens | 3.092 / 3.088 | 3.081 / 3.078 | 3.084 / 3.083 | 3.070 / 3.068 | 3.066 / 3.063 |
| Nestor, equal FLOPs, repeated | | 3.079 / 3.076 | 3.075 / 3.073 | 3.061 / 3.058 | 3.056 / 3.051 |
| Nestor, equal FLOPs, new | | 3.077 / 3.073 | 3.074 / 3.071 | 3.056 / 3.054 | 3.043 / 3.042 |
| **500M** | | | | | |
| dense transformer | 2.953 | 2.971 | 2.970 | 2.952 | 2.948 |
| MoE transformer | 2.793 | 2.827 | 2.842 | 2.810 | 2.805 |
| Nestor, equal tokens | 2.758 | 2.792 | 2.811 | 2.779 | 2.773 |
| Nestor, equal FLOPs, repeated | | 2.789 | 2.802 | 2.766 | 2.757 |
| Nestor, equal FLOPs, new | | 2.787 | 2.800 | 2.758 | 2.744 |
| **1B** | | | | | |
| dense transformer | 2.743 | 2.797 | 2.810 | 2.779 | 2.780 |
| MoE transformer | 2.592 | 2.633 | 2.683 | 2.634 | 2.632 |
| Nestor, equal tokens | 2.571 | 2.616 | 2.661 | 2.612 | 2.607 |
| Nestor, equal FLOPs, repeated | | 2.616 | 2.655 | 2.603 | 2.599 |
| Nestor, equal FLOPs, new | | 2.610 | 2.650 | 2.591 | 2.574 |

### Domain adaptation and forgetting

The 500M Nestor, the 500M MoE transformer and the 1B dense transformer,
chosen for their similar pretraining losses, are first extended to 32k on
DCLM (the equal-token run above). From that checkpoint each model is trained
separately on four target domains: books (PG-19), mathematics (OpenWebMath),
code (AlgebraicStack) and German (FineWeb-2), for 477 steps of 32 x 32,768
tokens under the 32k extension recipe. The two sparse models train with
frozen routers (`NESTOR_FREEZE_ROUTERS=1`, `nestor/train/frozen_routers.py`:
the router score and noise networks and the load-balancing biases keep their
values and routing runs without exploration noise, while routes are still
recomputed from the current hidden states); the dense model is fully
fine-tuned. Each run is repeated with 25 % of the budget replaced by DCLM
sequences the model saw during context extension (`scripts/build_replay_mix.py`),
the total budget unchanged. Adaptation is the target-corpus loss before minus
after; forgetting is the DCLM loss after minus before, both at 32k (PG-19: the
first 32k tokens of each test book that reaches that length; the other
corpora: the first 200 non-overlapping 32k windows of the test stream; DCLM:
the ladder's 32k held-out set).

```bash
for c in pg19 owm algstack fw2de; do CORPUS=$c sbatch scripts/slurm_prep_domain_data.sh; done
EXPERIMENT=nestor_500m CORPUS=pg19 REPLAY=0 \
  INIT=outputs/continual/nestor_500m_ext32768_equal_tokens sbatch scripts/slurm_adapt.sh
EXPERIMENT=nestor_500m LENGTH=32768 DOC_BINS="dclm_longdoc_ge32768_val pg19_test" \
  CHECKPOINT=outputs/continual/nestor_500m_adapt_pg19 sbatch scripts/slurm_eval_long_context.sh
```

The starting checkpoint is read the same way (`PACKED_BINS="owm_test
algstack_test fw2de_test"` adds the page-length corpora), and
`scripts/check_frozen_routers.py` verifies that a frozen-router run left every
router tensor untouched and moved everything else.

Loss before and after adaptation (target corpus and DCLM, at 32k; the
average gives equal weight to the four domains):

| | | target before | after | gain | DCLM before | after | forgetting |
|---|---|---|---|---|---|---|---|
| **no replay** | | | | | | | |
| books | 500M Nestor | 2.732 | 2.548 | 0.184 | 2.779 | 2.996 | 0.217 |
| | 500M MoE | 2.765 | 2.582 | 0.182 | 2.810 | 3.039 | 0.229 |
| | 1B dense | 2.719 | 2.538 | 0.181 | 2.779 | 3.032 | 0.252 |
| math | 500M Nestor | 2.417 | 2.084 | 0.332 | 2.779 | 2.885 | 0.106 |
| | 500M MoE | 2.449 | 2.111 | 0.338 | 2.810 | 2.920 | 0.110 |
| | 1B dense | 2.412 | 2.075 | 0.338 | 2.779 | 2.914 | 0.134 |
| code | 500M Nestor | 1.789 | 1.084 | 0.705 | 2.779 | 3.073 | 0.294 |
| | 500M MoE | 1.813 | 1.102 | 0.712 | 2.810 | 3.106 | 0.296 |
| | 1B dense | 1.720 | 1.031 | 0.688 | 2.779 | 3.164 | 0.385 |
| German | 500M Nestor | 4.171 | 2.465 | 1.706 | 2.779 | 3.176 | 0.398 |
| | 500M MoE | 4.224 | 2.521 | 1.704 | 2.810 | 3.221 | 0.410 |
| | 1B dense | 4.164 | 2.431 | 1.733 | 2.779 | 3.273 | 0.494 |
| average | 500M Nestor | 2.777 | 2.045 | 0.732 | 2.779 | 3.033 | 0.254 |
| | 500M MoE | 2.813 | 2.079 | 0.734 | 2.810 | 3.072 | 0.261 |
| | 1B dense | 2.754 | 2.019 | 0.735 | 2.779 | 3.096 | 0.316 |
| **25 % replay** | | | | | | | |
| books | 500M Nestor | 2.732 | 2.573 | 0.158 | 2.779 | 2.828 | 0.049 |
| | 500M MoE | 2.765 | 2.607 | 0.158 | 2.810 | 2.863 | 0.053 |
| | 1B dense | 2.719 | 2.560 | 0.159 | 2.779 | 2.837 | 0.058 |
| math | 500M Nestor | 2.417 | 2.113 | 0.304 | 2.779 | 2.832 | 0.053 |
| | 500M MoE | 2.449 | 2.140 | 0.309 | 2.810 | 2.867 | 0.057 |
| | 1B dense | 2.412 | 2.101 | 0.311 | 2.779 | 2.842 | 0.063 |
| code | 500M Nestor | 1.789 | 1.131 | 0.658 | 2.779 | 2.867 | 0.088 |
| | 500M MoE | 1.813 | 1.146 | 0.668 | 2.810 | 2.901 | 0.091 |
| | 1B dense | 1.720 | 1.074 | 0.646 | 2.779 | 2.885 | 0.106 |
| German | 500M Nestor | 4.171 | 2.546 | 1.626 | 2.779 | 2.879 | 0.101 |
| | 500M MoE | 4.224 | 2.611 | 1.614 | 2.810 | 2.913 | 0.102 |
| | 1B dense | 4.164 | 2.497 | 1.667 | 2.779 | 2.899 | 0.120 |
| average | 500M Nestor | 2.777 | 2.091 | 0.687 | 2.779 | 2.851 | 0.073 |
| | 500M MoE | 2.813 | 2.126 | 0.687 | 2.810 | 2.886 | 0.076 |
| | 1B dense | 2.754 | 2.058 | 0.696 | 2.779 | 2.866 | 0.087 |

## Layout

```
nestor/
  backbones/      Nestor (nestor.py), the depth reads, the hybrid expert pool, grouped-GEMM dispatch, Triton kernels
  scaling/        muP-tagged layers (RMSNorm, RoPE, scaled linear/MLP) and the Muon + AdamW optimizer
  adapters, heads, models   token embedding, language-modeling head, adapter-backbone-head model
  configs/        dataclass configs for models, data, optimization, evaluation, jobs
  data/           the memory-mapped token dataset, the tokenizer helpers, seen-window subsets and replay mixtures
  train/          trainer (DDP, grad accumulation, checkpointing, MFU), losses, optimizer builder, frozen routers
  eval/           held-out evaluation, per-position loss at length, checkpoint loading
  experiments/    the registered pretraining runs
  continual/      the long-context extension and domain-adaptation protocols, FLOPs per token at any length
scripts/          preprocessing, evaluation, SLURM launchers (pretraining, extension, adaptation, data)
tests/            pytest suite (CPU; the CUDA-only tests skip)
```

## Tests

```bash
uv run pytest tests -q
```

`tests/test_pretrain_arms.py` pins the twelve runs (geometry, recipe, FLOP
ratios and exact parameter counts) and `tests/test_continual_protocol.py` the
continued-training protocols (the FLOP ratios and equal-FLOP budgets at every
scale and length, the run environments, the domain-adaptation runs); the
remaining files are correctness tests of the model and the training stack
against slow reference implementations.
