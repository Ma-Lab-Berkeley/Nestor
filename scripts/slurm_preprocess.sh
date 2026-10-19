#!/usr/bin/env bash
# Tokenize DCLM-baseline jsonl.zst shards into a flat cl100k token .bin (CPU job).
#
#   DCLM_ROOT=/data/dclm-baseline-1.0 OUT=data/dclm/dclm_smoke.bin TARGET_TOKENS=300000000 \
#     sbatch scripts/slurm_preprocess.sh
#   DCLM_ROOT=... OUT=data/dclm/dclm_train.bin TARGET_TOKENS=11000000000 sbatch scripts/slurm_preprocess.sh
#   DCLM_ROOT=... OUT=data/dclm/dclm_val.bin TARGET_TOKENS=20000000 GLOBAL_SHARDS=9 sbatch scripts/slurm_preprocess.sh
#SBATCH --job-name=dclm-prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/prep_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
mkdir -p "${TIKTOKEN_CACHE_DIR}"

DCLM_ROOT="${DCLM_ROOT:?set DCLM_ROOT=path/to/dclm-baseline-1.0}"
OUT="${OUT:?set OUT=path/to/tokens.bin}"
TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
WORKERS="${WORKERS:-44}"
GLOBAL_SHARDS="${GLOBAL_SHARDS:-}"

ARGS=(--dclm-root "${DCLM_ROOT}" --out "${OUT}" --target-tokens "${TARGET_TOKENS}" --workers "${WORKERS}")
if [ -n "${GLOBAL_SHARDS}" ]; then ARGS+=(--global-shards "${GLOBAL_SHARDS}"); fi

python scripts/preprocess_dclm.py "${ARGS[@]}"
ls -lh "${OUT}" "${OUT}.meta.json"
