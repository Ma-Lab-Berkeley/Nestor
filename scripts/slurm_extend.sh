#!/usr/bin/env bash
# Extend one pretrained model to a longer context by continued pretraining (one node).
#
#   EXPERIMENT=nestor_200m LENGTH=32768 SETTING=equal_flops_new \
#   INIT=outputs/nestor_200m/<run>/checkpoints/model_final.pt sbatch scripts/slurm_extend.sh
#
# The protocol -- steps, warmup, learning rates, rotary base, corpus, tokens per
# step, the run's own output root -- comes from `python -m nestor.continual.extension
# --env` (nestor/continual/extension.py); this script adds the batch shape, the
# memory recipe and the speed stack, and launches torchrun. Resubmitting resumes
# the run from its latest checkpoint.
#
# Env: EXPERIMENT (a registered cell), LENGTH (8192 | 16384 | 32768 | 65536),
# SETTING (equal_tokens | equal_flops_new | equal_flops_repeated; the latter two
# for nestor_* only), REFERENCE (moe; the paper's 200M budgets used dense), INIT
# (the 4k model_final.pt, or its run root), DATA_DIR (data/longctx), OUTPUT_ROOT
# (outputs), NPROC_PER_NODE (8), PER_RANK (sequences per GPU per micro-step; 1),
# DCLM_SEED (42), NESTOR_STACKED_EXPERT_BANKS (1; must match the init file's
# layout, see scripts/lib/speed_stack.sh), and the memory recipe below.
#SBATCH --job-name=nestor-extend
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/extend_%x_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=<registered cell, e.g. nestor_200m>}"
LENGTH="${LENGTH:?set LENGTH=8192|16384|32768|65536}"
SETTING="${SETTING:-equal_tokens}"
REFERENCE="${REFERENCE:-moe}"
INIT="${INIT:?set INIT=path/to/the 4k model_final.pt (or its run root)}"
DATA_DIR="${DATA_DIR:-data/longctx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_RANK="${PER_RANK:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
mkdir -p "${TIKTOKEN_CACHE_DIR}"

# --- the protocol of the run (DCLM_* for the registry, CONTINUAL_* for this script)
eval "$(python -m nestor.continual.extension --experiment "${EXPERIMENT}" --length "${LENGTH}" \
    --setting "${SETTING}" --reference "${REFERENCE}" --data-dir "${DATA_DIR}" --init "${INIT}" \
    --output-root "${OUTPUT_ROOT}" --env)"
[ -f "${DCLM_DATA_PATH}" ] || { echo "corpus missing: ${DCLM_DATA_PATH} (scripts/slurm_prep_long_context_data.sh)" >&2; exit 2; }

# --- batch shape: world x per-rank x accum = the protocol's sequences per step
export DCLM_GLOBAL_BATCH=$(( NPROC_PER_NODE * PER_RANK ))
[ $(( CONTINUAL_SEQS_PER_STEP % DCLM_GLOBAL_BATCH )) -eq 0 ] \
    || { echo "${CONTINUAL_SEQS_PER_STEP} sequences/step is not a multiple of ${NPROC_PER_NODE} x ${PER_RANK}" >&2; exit 2; }
export DCLM_GRAD_ACCUM=$(( CONTINUAL_SEQS_PER_STEP / DCLM_GLOBAL_BATCH ))
export DCLM_CKPT_FREQ="${DCLM_CKPT_FREQ:-100}"
export DCLM_SEED="${DCLM_SEED:-42}"

# --- memory recipe (defaults = the paper's runs; every flag can be preset)
# Per-layer activation checkpointing for the larger models at the longer
# contexts, and eager execution whenever checkpointing is on (the paper's runs
# did not use the compiled checkpoint path); the fused head-CE at 1B from 32k,
# where the [S, vocab] logits no longer fit next to the activations.
scale="${EXPERIMENT##*_}"
ckpt_default=0; fused_default=0
case "${scale}:${LENGTH}" in
    1b:*|*:65536|500m:32768) ckpt_default=1 ;;
esac
case "${scale}:${LENGTH}" in 1b:32768|1b:65536) fused_default=1 ;; esac
export NESTOR_ACTIVATION_CHECKPOINT="${NESTOR_ACTIVATION_CHECKPOINT:-${ckpt_default}}"
export NESTOR_COMPILE="${NESTOR_COMPILE:-$(( 1 - NESTOR_ACTIVATION_CHECKPOINT ))}"
export NESTOR_FUSED_HEAD_CE="${NESTOR_FUSED_HEAD_CE:-${fused_default}}"
# The Triton kernels and batched routers were priced at 4k and cost peak
# memory at long contexts; the paper's extension runs ran without them.
export NESTOR_TRITON_PACKED_ATTN="${NESTOR_TRITON_PACKED_ATTN:-0}"
export NESTOR_TRITON_EXPERT_DISPATCH="${NESTOR_TRITON_EXPERT_DISPATCH:-0}"
export NESTOR_BATCHED_ROUTERS="${NESTOR_BATCHED_ROUTERS:-0}"
source scripts/lib/speed_stack.sh

export WANDB_PROJECT="${WANDB_PROJECT:-nestor}"
export WANDB_TAGS="${WANDB_TAGS:-extension,${SETTING},${LENGTH}}"

echo "[extend] ${EXPERIMENT} -> ${LENGTH} ${SETTING} (reference ${REFERENCE}) init=${DCLM_INIT_CHECKPOINT}"
echo "[extend] corpus=${DCLM_DATA_PATH} steps=${DCLM_MAX_STEPS} warmup=${DCLM_WARMUP_STEPS} lr_muon=${DCLM_LR_MUON} lr_adamw=${DCLM_LR_ADAMW} rope=${DCLM_ROPE_BASE}"
echo "[extend] batch=${DCLM_GLOBAL_BATCH}x${DCLM_GRAD_ACCUM} (${CONTINUAL_SEQS_PER_STEP} x ${LENGTH} tokens/step) gpus=${NPROC_PER_NODE} per_rank=${PER_RANK}"
echo "[extend] ckpt=${NESTOR_ACTIVATION_CHECKPOINT} compile=${NESTOR_COMPILE} fused_head=${NESTOR_FUSED_HEAD_CE} stacked_banks=${NESTOR_STACKED_EXPERT_BANKS} out=${NESTOR_OUTPUT_ROOT}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m nestor.cli \
    --experiment "${EXPERIMENT}"
