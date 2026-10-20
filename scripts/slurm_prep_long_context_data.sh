#!/usr/bin/env bash
# Build the long-context extension data from a local DCLM-baseline copy (CPU job).
#
#   DCLM_ROOT=/data/dclm-baseline-1.0 sbatch scripts/slurm_prep_long_context_data.sh
#
# For every target length S in LENGTHS:
#   * dclm_longdoc_ge<S>_train.bin -- the extension corpus: documents of at least
#     S tokens from global shards the pretraining corpus never read (TRAIN_SHARDS;
#     documents of >= 64k tokens are rare, so the 64k corpus scans two shards);
#   * dclm_longdoc_ge<S>_val.bin   -- the held-out set: documents of at least S
#     tokens from a third shard (HELDOUT_SHARD), disjoint from both;
#   * dclm_longdoc_ge<S>_train_repeated_T<steps>.bin -- for every Nestor cell, the
#     repeated-documents corpus of its equal-FLOP run (scripts/build_seen_subset.py;
#     REPEATED=0 skips them, REFERENCE=dense budgets them as the paper's 200M runs).
# The training corpus of the paper read the sorted shard list from the first
# global shard on and its held-out set came from shard 9; shards 7, 8 and 10
# were untouched, hence the defaults.
#
# Env: DCLM_ROOT (required), DATA_DIR (data/longctx), LENGTHS ("8192 16384 32768
# 65536"), HELDOUT_SHARD (10), TRAIN_SHARDS (8; "8,7" at 64k), HELDOUT_TOKENS
# (200M), TRAIN_TOKENS (1.2B), WORKERS, REPEATED (1), REFERENCE (moe), SCALES
# ("100m 200m 500m 1b").
#SBATCH --job-name=nestor-prep-longctx
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/prep_longctx_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
mkdir -p "${TIKTOKEN_CACHE_DIR}"

DCLM_ROOT="${DCLM_ROOT:?set DCLM_ROOT=path/to/dclm-baseline-1.0}"
DATA_DIR="${DATA_DIR:-data/longctx}"
LENGTHS="${LENGTHS:-8192 16384 32768 65536}"
HELDOUT_SHARD="${HELDOUT_SHARD:-10}"
HELDOUT_TOKENS="${HELDOUT_TOKENS:-200000000}"
TRAIN_TOKENS="${TRAIN_TOKENS:-1200000000}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
REPEATED="${REPEATED:-1}"
REFERENCE="${REFERENCE:-moe}"
SCALES="${SCALES:-100m 200m 500m 1b}"
mkdir -p "${DATA_DIR}"

build() {   # build <out.bin> <preprocess_dclm.py args...>: skips a bin whose sidecar exists
    local out="$1"; shift
    if [ -f "${out}.meta.json" ]; then echo "[prep] ${out} exists, skipping"; return 0; fi
    python scripts/preprocess_dclm.py --dclm-root "${DCLM_ROOT}" --out "${out}" --workers "${WORKERS}" "$@"
}

for S in ${LENGTHS}; do
    case "${S}" in 65536) shards="${TRAIN_SHARDS:-8,7}" ;; *) shards="${TRAIN_SHARDS:-8}" ;; esac
    echo "[prep] S=${S}: held-out from shard ${HELDOUT_SHARD}, extension corpus from shards ${shards}"
    build "${DATA_DIR}/dclm_longdoc_ge${S}_val.bin" --global-shards "${HELDOUT_SHARD}" \
        --min-doc-tokens "${S}" --target-tokens "${HELDOUT_TOKENS}"
    build "${DATA_DIR}/dclm_longdoc_ge${S}_train.bin" --global-shards "${shards}" \
        --min-doc-tokens "${S}" --target-tokens "${TRAIN_TOKENS}"
    if [ "${REPEATED}" = "1" ]; then
        for scale in ${SCALES}; do
            python scripts/build_seen_subset.py --experiment "nestor_${scale}" --length "${S}" \
                --reference "${REFERENCE}" --data-dir "${DATA_DIR}"
        done
    fi
done
echo "[prep] DONE"; ls -la "${DATA_DIR}"
