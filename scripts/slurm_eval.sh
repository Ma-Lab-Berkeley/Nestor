#!/usr/bin/env bash
# Held-out cross-entropy / perplexity of a trained checkpoint (one GPU).
#
#   EXPERIMENT=nestor_200m CHECKPOINT=outputs/nestor_200m/<run>/checkpoints/model_final.pt \
#   VAL_BIN=data/dclm/dclm_val.bin sbatch scripts/slurm_eval.sh
#SBATCH --job-name=nestor-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
# Match the training forward topology (skipped gate-1.0 routers on dense
# blocks) and use the grouped-GEMM expert dispatch; the forward stays fp32.
export NESTOR_DENSE_NO_ROUTER="${NESTOR_DENSE_NO_ROUTER:-1}"
export NESTOR_MOE_GROUPED_MM="${NESTOR_MOE_GROUPED_MM:-1}"

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=<registered experiment>}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/model_final.pt}"
VAL_BIN="${VAL_BIN:?set VAL_BIN=path/to/heldout.bin}"
MAX_TOKENS="${MAX_TOKENS:-15000000}"

echo "[eval] experiment=${EXPERIMENT} ckpt=${CHECKPOINT} val=${VAL_BIN}"
python scripts/eval_heldout.py \
    --experiment "${EXPERIMENT}" \
    --checkpoint "${CHECKPOINT}" \
    --val-bin "${VAL_BIN}" \
    --max-tokens "${MAX_TOKENS}"
