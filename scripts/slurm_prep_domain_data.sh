#!/usr/bin/env bash
# Build one target-domain corpus of the domain-adaptation experiments (CPU job).
#
#   for c in pg19 owm algstack fw2de; do CORPUS=$c sbatch scripts/slurm_prep_domain_data.sh; done
#
# Downloads the published release of the corpus (`hf download`, huggingface_hub;
# `uv sync --extra data` installs it with pyarrow for the parquet releases),
# tokenizes its training and test splits into <corpus>_{train,test}.bin
# (scripts/preprocess_text_bin.py, cl100k, end-of-text separated, no length
# filter), and builds the replay mixture(s) <corpus>_replay<pct>.bin from the
# 32k extension corpus (scripts/build_replay_mix.py; needs
# dclm_longdoc_ge32768_train.bin from scripts/slurm_prep_long_context_data.sh).
#
#   pg19      PG-19 books: the parquet re-upload emozilla/pg19 (train 28,602 books,
#             test 100 books, of which 77 reach 32k tokens)
#   owm       OpenWebMath: EleutherAI/proof-pile-2, split open-web-math, the first
#             OWM_SHARDS (12) of 63 training shards (~2.4B tokens) + the test split
#   algstack  AlgebraicStack: EleutherAI/proof-pile-2, split algebraic-stack, every
#             training file (one per language, ~8B tokens) + the test files
#   fw2de     FineWeb-2 German: HuggingFaceFW/fineweb-2, data/deu_Latn, the first
#             FW2DE_SHARDS (2) training parquets (~5.7B tokens) + the test parquet
#
# Env: CORPUS (required), DATA_DIR (data/longctx), RAW_DIR (data/raw), REPLAY_PCTS
# ("25"; "" skips the mixtures), WORKERS, OWM_SHARDS (12), FW2DE_SHARDS (2),
# HF_ENDPOINT (a mirror, if needed).
#SBATCH --job-name=nestor-prep-domain
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/prep_domain_%j.out

set -euo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${WORK_DIR}/data/tiktoken}"
mkdir -p "${TIKTOKEN_CACHE_DIR}"

CORPUS="${CORPUS:?set CORPUS=pg19|owm|algstack|fw2de}"
DATA_DIR="${DATA_DIR:-data/longctx}"
RAW_DIR="${RAW_DIR:-data/raw}"
REPLAY_PCTS="${REPLAY_PCTS-25}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
OWM_SHARDS="${OWM_SHARDS:-12}"
FW2DE_SHARDS="${FW2DE_SHARDS:-2}"
mkdir -p "${DATA_DIR}" "${RAW_DIR}"
command -v hf >/dev/null || { echo "the huggingface_hub CLI (hf) is missing: uv sync --extra data" >&2; exit 2; }

download() {   # download <repo> <local dir> <paths...>: exact files and/or `subdir/` folders
    local repo="$1" dir="$2"; shift 2
    hf download "${repo}" "$@" --repo-type dataset --local-dir "${dir}"
}
tokenize() {   # tokenize <out.bin> <inputs...>: skips a bin whose sidecar exists
    local out="$1"; shift
    if [ -f "${out}.meta.json" ]; then echo "[prep] ${out} exists, skipping"; return 0; fi
    python scripts/preprocess_text_bin.py --input "$@" --out "${out}" --workers "${WORKERS}"
}
numbered() {   # numbered <prefix> <n> <suffix>: prefix0000suffix ... prefix<n-1>suffix
    local prefix="$1" n="$2" suffix="$3" i
    for i in $(seq 0 $((n - 1))); do printf '%s%04d%s\n' "${prefix}" "${i}" "${suffix}"; done
}

case "${CORPUS}" in
    pg19)
        RAW="${RAW_DIR}/pg19"
        download emozilla/pg19 "${RAW}"
        tokenize "${DATA_DIR}/pg19_test.bin" "${RAW}/data/test-*.parquet"
        tokenize "${DATA_DIR}/pg19_train.bin" "${RAW}/data/train-*.parquet"
        ;;
    owm)
        RAW="${RAW_DIR}/proof-pile-2"
        mapfile -t TRAIN_FILES < <(numbered "open-web-math/train/shard-" "${OWM_SHARDS}" ".jsonl.zst")
        download EleutherAI/proof-pile-2 "${RAW}" "${TRAIN_FILES[@]}" "open-web-math/test/"
        tokenize "${DATA_DIR}/owm_test.bin" "${RAW}/open-web-math/test/*.jsonl.zst"
        tokenize "${DATA_DIR}/owm_train.bin" "${TRAIN_FILES[@]/#/${RAW}/}"
        ;;
    algstack)
        RAW="${RAW_DIR}/proof-pile-2"
        download EleutherAI/proof-pile-2 "${RAW}" "algebraic-stack/"
        tokenize "${DATA_DIR}/algstack_test.bin" "${RAW}/algebraic-stack/test/*.jsonl.zst"
        tokenize "${DATA_DIR}/algstack_train.bin" "${RAW}/algebraic-stack/train/*.jsonl.zst"
        ;;
    fw2de)
        RAW="${RAW_DIR}/fineweb-2"
        mapfile -t TRAIN_FILES < <(numbered "data/deu_Latn/train/000_0" "${FW2DE_SHARDS}" ".parquet")
        download HuggingFaceFW/fineweb-2 "${RAW}" "${TRAIN_FILES[@]}" "data/deu_Latn/test/"
        tokenize "${DATA_DIR}/fw2de_test.bin" "${RAW}/data/deu_Latn/test/*.parquet"
        tokenize "${DATA_DIR}/fw2de_train.bin" "${TRAIN_FILES[@]/#/${RAW}/}"
        ;;
    *) echo "unknown CORPUS=${CORPUS}" >&2; exit 2 ;;
esac

for pct in ${REPLAY_PCTS}; do
    out="${DATA_DIR}/${CORPUS}_replay${pct}.bin"
    if [ -f "${out}.meta.json" ]; then echo "[prep] ${out} exists, skipping"; continue; fi
    [ -f "${DATA_DIR}/dclm_longdoc_ge32768_train.bin" ] \
        || { echo "replay source missing: ${DATA_DIR}/dclm_longdoc_ge32768_train.bin (scripts/slurm_prep_long_context_data.sh)" >&2; exit 2; }
    python scripts/build_replay_mix.py --corpus "${CORPUS}" --replay "${pct}" --data-dir "${DATA_DIR}"
done
echo "[prep] DONE"; ls -la "${DATA_DIR}/${CORPUS}"_*
