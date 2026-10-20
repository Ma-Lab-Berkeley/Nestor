#!/usr/bin/env bash
# The held-out reads of a continued-training checkpoint at its context length (one GPU).
#
#   EXPERIMENT=nestor_500m LENGTH=32768 \
#   CHECKPOINT=outputs/continual/nestor_500m_adapt_pg19 \
#   DOC_BINS="dclm_longdoc_ge32768_val pg19_test" sbatch scripts/slurm_eval_long_context.sh
#
# Rebuilds the model at LENGTH with the run's rotary base and scores, in full
# precision (scripts/eval_long_context.py):
#   1. every bin in DOC_BINS in documents mode -- the first LENGTH tokens of each
#      document of at least LENGTH tokens (the DCLM long-document held-out set,
#      the PG-19 test books), MAX_DOCS documents;
#   2. every bin in PACKED_BINS in packed mode -- consecutive LENGTH-token windows
#      of the test stream (OpenWebMath, AlgebraicStack, FineWeb-2), MAX_DOCS windows;
#   3. the 4k held-out set (scripts/eval_heldout.py) at the same rotary base --
#      the short-context read.
# A summary with one RESULT line per read lands in OUT_DIR/summary.txt. Evaluate
# the starting checkpoint the same way: adaptation and forgetting are differences
# of these reads.
#
# Env: EXPERIMENT, CHECKPOINT (model_*.pt or a run root, resolved to its newest
# model_final.pt), LENGTH (32768), ROPE_BASE (the protocol's for LENGTH), DATA_DIR
# (data/longctx), DOC_BINS ("dclm_longdoc_ge<LENGTH>_val"), PACKED_BINS (""),
# MAX_DOCS (200; 100 at 64k), VAL4K (data/dclm/dclm_val.bin), OUT_DIR.
#SBATCH --job-name=nestor-eval-longctx
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_longctx_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
# The eval forward topology of every number on record: skipped gate-1.0 routers
# on dense blocks, grouped-GEMM expert dispatch, fp32.
export NESTOR_DENSE_NO_ROUTER="${NESTOR_DENSE_NO_ROUTER:-1}"
export NESTOR_MOE_GROUPED_MM="${NESTOR_MOE_GROUPED_MM:-1}"

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=<registered cell>}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/model_final.pt (or a run root)}"
LENGTH="${LENGTH:-32768}"
DATA_DIR="${DATA_DIR:-data/longctx}"
VAL4K="${VAL4K:-data/dclm/dclm_val.bin}"
DOC_BINS="${DOC_BINS-dclm_longdoc_ge${LENGTH}_val}"
PACKED_BINS="${PACKED_BINS:-}"
case "${LENGTH}" in 65536) MAX_DOCS="${MAX_DOCS:-100}" ;; *) MAX_DOCS="${MAX_DOCS:-200}" ;; esac
case "${LENGTH}" in
    4096) default_rope=10000 ;; 8192) default_rope=500000 ;; 16384) default_rope=1000000 ;;
    32768) default_rope=2000000 ;; 65536) default_rope=5000000 ;;
    *) echo "unknown LENGTH ${LENGTH}" >&2; exit 2 ;;
esac
ROPE_BASE="${ROPE_BASE:-${default_rope}}"

CHECKPOINT="$(python -c 'import sys; from pathlib import Path; from nestor.continual.extension import resolve_checkpoint; print(resolve_checkpoint(Path(sys.argv[1])))' "${CHECKPOINT}")"
RUN_TAG="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/outputs/continual_eval/${EXPERIMENT}_${RUN_TAG}_S${LENGTH}}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"; : > "${SUMMARY}"
export NESTOR_GATHERED_CAP_BUCKET="${NESTOR_GATHERED_CAP_BUCKET:-$(( LENGTH / 32 ))}"

echo "[eval] ${EXPERIMENT} ckpt=${CHECKPOINT} length=${LENGTH} rope=${ROPE_BASE} out=${OUT_DIR}"

read_bins() {   # read_bins <mode> <bin names...>
    local mode="$1"; shift
    for bin in "$@"; do
        if [ ! -f "${DATA_DIR}/${bin}.bin" ]; then echo "[eval] missing ${DATA_DIR}/${bin}.bin, skipping"; continue; fi
        python scripts/eval_long_context.py --experiment "${EXPERIMENT}" --checkpoint "${CHECKPOINT}" \
            --bin "${DATA_DIR}/${bin}.bin" --mode "${mode}" --block-size "${LENGTH}" --rope-base "${ROPE_BASE}" \
            --max-docs "${MAX_DOCS}" --out "${OUT_DIR}/perpos_${bin}_${LENGTH}_${mode}.json" 2>&1 \
            | tee "${OUT_DIR}/perpos_${bin}_${LENGTH}_${mode}.log" \
            | grep -E "RESULT|binned CE|DONE" | sed "s/^/[${bin}@${LENGTH} ${mode}] /" >> "${SUMMARY}"
    done
}
# shellcheck disable=SC2086
read_bins documents ${DOC_BINS}
# shellcheck disable=SC2086
read_bins packed ${PACKED_BINS}

if [ -f "${VAL4K}" ]; then
    DCLM_BLOCK_SIZE=4096 DCLM_ROPE_BASE="${ROPE_BASE}" python scripts/eval_heldout.py \
        --experiment "${EXPERIMENT}" --checkpoint "${CHECKPOINT}" --val-bin "${VAL4K}" --block-size 4096 \
        --max-tokens "${MAX_TOKENS_4K:-15000000}" 2>&1 | tee "${OUT_DIR}/heldout4k.log" \
        | grep -E "RESULT|DONE" | sed "s/^/[dclm_val@4096] /" >> "${SUMMARY}"
else
    echo "[eval] missing ${VAL4K}, skipping the 4k read"
fi

echo "[eval] SUMMARY (${EXPERIMENT} ${RUN_TAG} S=${LENGTH} rope=${ROPE_BASE}):"; cat "${SUMMARY}"
