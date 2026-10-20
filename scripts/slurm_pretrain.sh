#!/usr/bin/env bash
# Pretrain one registered experiment on a single 8-GPU node.
#
#   DCLM_DATA_PATH=data/dclm/dclm_train.bin sbatch scripts/slurm_pretrain.sh nestor_200m
#   DCLM_MAX_STEPS=300 sbatch scripts/slurm_pretrain.sh dense_100m      # smoke
#
# Tokens per optimizer step are pinned per scale (see nestor/experiments/
# pretrain.py); PER_RANK sequences per GPU chooses how that budget is split
# between the global batch and gradient accumulation. Resubmitting resumes
# from the experiment's latest checkpoint under NESTOR_OUTPUT_ROOT.
#SBATCH --job-name=nestor-pretrain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/pretrain_%x_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

EXPERIMENT="${1:?usage: slurm_pretrain.sh <experiment>}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_RANK="${PER_RANK:-4}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# tiktoken caches the cl100k vocabulary here; pre-warm it on a node with internet.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
mkdir -p "${TIKTOKEN_CACHE_DIR}"

# --- batch shape: world x per-rank x accum x 4096 = the rung's tokens per step
case "${EXPERIMENT}" in
    *_100m) TOKENS_PER_STEP=262144 ;;
    *_200m) TOKENS_PER_STEP=524288 ;;
    *_500m) TOKENS_PER_STEP=1048576 ;;
    *_1b)   TOKENS_PER_STEP=2097152 ;;
    *) echo "unknown scale suffix in ${EXPERIMENT}" >&2; exit 1 ;;
esac
export DCLM_GLOBAL_BATCH="${DCLM_GLOBAL_BATCH:-$(( NPROC_PER_NODE * PER_RANK ))}"
export DCLM_GRAD_ACCUM="${DCLM_GRAD_ACCUM:-$(( TOKENS_PER_STEP / (DCLM_GLOBAL_BATCH * 4096) ))}"

# --- the speed stack every paper run trained with (all exact reformulations)
source scripts/lib/speed_stack.sh

export DCLM_DATA_PATH="${DCLM_DATA_PATH:?set DCLM_DATA_PATH=path/to/tokens.bin}"
export NESTOR_OUTPUT_ROOT="${NESTOR_OUTPUT_ROOT:-${WORK_DIR}/outputs}"
export WANDB_PROJECT="${WANDB_PROJECT:-nestor}"

echo "[pretrain] experiment=${EXPERIMENT} data=${DCLM_DATA_PATH} gpus=${NPROC_PER_NODE} " \
     "global_batch=${DCLM_GLOBAL_BATCH} grad_accum=${DCLM_GRAD_ACCUM}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m nestor.cli \
    --experiment "${EXPERIMENT}"
