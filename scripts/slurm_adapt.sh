#!/usr/bin/env bash
# Adapt a 32k-extended model to a new domain by continued pretraining (one node).
#
#   EXPERIMENT=nestor_500m CORPUS=pg19 REPLAY=0 \
#   INIT=outputs/continual/nestor_500m_ext32768_equal_tokens sbatch scripts/slurm_adapt.sh
#
# The protocol -- the 32k extension recipe (477 steps of 32 x 32,768 tokens,
# rotary base 2e6, Muon 0.0025 / AdamW 7.5e-5), the corpus or its replay
# mixture, frozen routers for the sparse models, the run's own output root --
# comes from `python -m nestor.continual.adaptation --env`
# (nestor/continual/adaptation.py); this script adds the batch shape, the
# memory recipe and the speed stack, and launches torchrun.
#
# Env: EXPERIMENT (a registered cell; the paper: nestor_500m, moe_500m, dense_1b),
# CORPUS (pg19 | owm | algstack | fw2de), REPLAY (0 | 25: percent of the budget
# replayed from DCLM; the mixture comes from scripts/build_replay_mix.py), INIT
# (the 32k equal-token model_final.pt, or its run root), DATA_DIR (data/longctx),
# OUTPUT_ROOT (outputs), NPROC_PER_NODE (8), PER_RANK (1), DCLM_SEED (42),
# NESTOR_STACKED_EXPERT_BANKS (1; must match the init file's layout), and the
# memory recipe below.
#SBATCH --job-name=nestor-adapt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/adapt_%x_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=<registered cell, e.g. nestor_500m>}"
CORPUS="${CORPUS:?set CORPUS=pg19|owm|algstack|fw2de}"
REPLAY="${REPLAY:-0}"
INIT="${INIT:?set INIT=path/to/the 32k-extended model_final.pt (or its run root)}"
DATA_DIR="${DATA_DIR:-data/longctx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_RANK="${PER_RANK:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
mkdir -p "${TIKTOKEN_CACHE_DIR}"

# --- the protocol of the run (DCLM_* and NESTOR_FREEZE_ROUTERS for the trainer)
eval "$(python -m nestor.continual.adaptation --experiment "${EXPERIMENT}" --corpus "${CORPUS}" \
    --replay "${REPLAY}" --data-dir "${DATA_DIR}" --init "${INIT}" --output-root "${OUTPUT_ROOT}" --env)"
[ -f "${DCLM_DATA_PATH}" ] || { echo "corpus missing: ${DCLM_DATA_PATH} (scripts/slurm_prep_domain_data.sh)" >&2; exit 2; }

# --- batch shape
export DCLM_GLOBAL_BATCH=$(( NPROC_PER_NODE * PER_RANK ))
[ $(( CONTINUAL_SEQS_PER_STEP % DCLM_GLOBAL_BATCH )) -eq 0 ] \
    || { echo "${CONTINUAL_SEQS_PER_STEP} sequences/step is not a multiple of ${NPROC_PER_NODE} x ${PER_RANK}" >&2; exit 2; }
export DCLM_GRAD_ACCUM=$(( CONTINUAL_SEQS_PER_STEP / DCLM_GLOBAL_BATCH ))
export DCLM_CKPT_FREQ="${DCLM_CKPT_FREQ:-100}"
export DCLM_SEED="${DCLM_SEED:-42}"

# --- memory recipe: the 32k rung's (see scripts/slurm_extend.sh)
scale="${EXPERIMENT##*_}"
ckpt_default=0; fused_default=0
case "${scale}" in 500m|1b) ckpt_default=1 ;; esac
case "${scale}" in 1b) fused_default=1 ;; esac
export NESTOR_ACTIVATION_CHECKPOINT="${NESTOR_ACTIVATION_CHECKPOINT:-${ckpt_default}}"
export NESTOR_COMPILE="${NESTOR_COMPILE:-$(( 1 - NESTOR_ACTIVATION_CHECKPOINT ))}"
export NESTOR_FUSED_HEAD_CE="${NESTOR_FUSED_HEAD_CE:-${fused_default}}"
export NESTOR_TRITON_PACKED_ATTN="${NESTOR_TRITON_PACKED_ATTN:-0}"
export NESTOR_TRITON_EXPERT_DISPATCH="${NESTOR_TRITON_EXPERT_DISPATCH:-0}"
export NESTOR_BATCHED_ROUTERS="${NESTOR_BATCHED_ROUTERS:-0}"
source scripts/lib/speed_stack.sh

export WANDB_PROJECT="${WANDB_PROJECT:-nestor}"
export WANDB_TAGS="${WANDB_TAGS:-adaptation,${CORPUS},replay${REPLAY}}"

echo "[adapt] ${EXPERIMENT} on ${CORPUS}, replay ${REPLAY} %, frozen_routers=${NESTOR_FREEZE_ROUTERS:-0} init=${DCLM_INIT_CHECKPOINT}"
echo "[adapt] corpus=${DCLM_DATA_PATH} steps=${DCLM_MAX_STEPS} warmup=${DCLM_WARMUP_STEPS} lr_muon=${DCLM_LR_MUON} lr_adamw=${DCLM_LR_ADAMW} rope=${DCLM_ROPE_BASE}"
echo "[adapt] batch=${DCLM_GLOBAL_BATCH}x${DCLM_GRAD_ACCUM} (${CONTINUAL_SEQS_PER_STEP} x ${DCLM_BLOCK_SIZE} tokens/step) gpus=${NPROC_PER_NODE} per_rank=${PER_RANK}"
echo "[adapt] ckpt=${NESTOR_ACTIVATION_CHECKPOINT} compile=${NESTOR_COMPILE} fused_head=${NESTOR_FUSED_HEAD_CE} stacked_banks=${NESTOR_STACKED_EXPERT_BANKS} out=${NESTOR_OUTPUT_ROOT}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m nestor.cli \
    --experiment "${EXPERIMENT}"
