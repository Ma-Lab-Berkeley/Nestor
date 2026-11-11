#!/usr/bin/env bash
# Price the trainer-side tunings on ONE GPU (or one node), in ONE job.
#
# Every leg is a short pretraining run of the same experiment from the same
# seed (scripts/slurm_pretrain.sh, the full speed stack; one rank unless
# NPROC_PER_NODE is set), and the legs differ only in the knobs under test.
# Tokens per optimizer step are the rung's in every leg, so `sec_per_step` is
# directly comparable:
#
#   base      4 sequences per micro-batch, single-rank DDP wrapper kept,
#             chunked full-logits head       (the stack before these knobs)
#   stream    base   + NESTOR_STREAMED_HEAD_CE=1
#   noddp     stream + bare model on one rank (NESTOR_DDP_SINGLE_RANK=0)
#   tuned     noddp  + 8 sequences per micro-batch (half the accumulation)
#             = what scripts/slurm_pretrain.sh launches on one GPU for a 100M model
#   gcfreeze  tuned  + NESTOR_GC_FREEZE=1    (opt-in; see the README)
#
# The legs run in ROUNDS rounds, in reverse order on every second round, and
# scripts/bench_summary.py reports each leg against the `base` leg of its own
# round (drift between rounds shows up as a spread, not as a speedup).
#
#   DCLM_DATA_PATH=data/dclm/dclm_train.bin sbatch scripts/bench_single_gpu.sh nestor_100m
#   STEPS=24 ROUNDS=2 LEGS="base tuned" bash scripts/bench_single_gpu.sh nestor_100m
#
# Under real DDP (`noddp` is then the same run as `stream`; PER_RANK is per GPU):
#   NPROC_PER_NODE=4 LEGS="base stream tuned" sbatch --gpus-per-node=4 \
#       --cpus-per-task=48 --mem=800G scripts/bench_single_gpu.sh nestor_200m
#SBATCH --job-name=nestor-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench_%x_%j.out

set -uo pipefail

WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${WORK_DIR}"
mkdir -p logs
source .venv/bin/activate   # for the summary; every leg activates it again itself

EXPERIMENT="${1:?usage: bench_single_gpu.sh <experiment>}"
STEPS="${STEPS:-24}"              # optimizer steps per leg
LOG_FREQ="${LOG_FREQ:-4}"         # optimizer steps per timing window
SKIP_WINDOWS="${SKIP_WINDOWS:-2}" # leading windows dropped from the timing (compile, then a recompile)
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ROUNDS="${ROUNDS:-2}"
LEGS="${LEGS:-base stream noddp tuned gcfreeze}"
BASELINE="${BASELINE:-base}"      # the leg every other leg is read against
BENCH_ROOT="${BENCH_ROOT:-${WORK_DIR}/outputs_bench/bench_${SLURM_JOB_ID:-$$}}"
mkdir -p "${BENCH_ROOT}/logs"

# Two more legs, not in the default LEGS:
#   minimal  the three-flag direct launch -- DDP, BF16 and torch.compile only,
#            every other speed-stack flag off -- for measuring what the speed
#            stack itself is worth on the same GPU;
#   chunk8   `tuned` with the chunked full-logits head instead of the streamed
#            one. LEGS="chunk8 tuned" BASELINE=chunk8 STEPS=200 is the paired
#            loss check of the streamed head: its two rounds are replicates,
#            so the spread between rounds of one leg is the run-to-run band
#            the difference between the legs has to sit inside.
STACK_OFF="NESTOR_DENSE_NO_ROUTER=0 NESTOR_GATHERED_CAUSAL_FLASH=0 NESTOR_GATHERED_VARLEN=0 \
NESTOR_MOE_GROUPED_MM=0 NESTOR_CE_CHUNK_ROWS=0 NESTOR_COLUMN_SLOT_UPDATES=0 NESTOR_DENSE_TAIL_ATTN=0 \
NESTOR_GROUPED_MM_FUSED_GATE=0 NESTOR_ATTN_PACKED_VARLEN=0 NESTOR_DDP_NO_SYNC=0 NESTOR_MUON_SHARD=0 \
NESTOR_STACKED_EXPERT_BANKS=0 NESTOR_BATCHED_ROUTERS=0 NESTOR_TRITON_PACKED_ATTN=0 \
NESTOR_TRITON_INDEX_BUILD=0 NESTOR_TRITON_EXPERT_DISPATCH=0"

leg_env() {   # leg_env <leg>: the knobs of one leg, as `env` arguments
    case "$1" in
        minimal)  echo "PER_RANK=4 NESTOR_DDP_SINGLE_RANK=1 NESTOR_STREAMED_HEAD_CE=0 NESTOR_GC_FREEZE=0 ${STACK_OFF}" ;;
        base)     echo "PER_RANK=4 NESTOR_DDP_SINGLE_RANK=1 NESTOR_STREAMED_HEAD_CE=0 NESTOR_GC_FREEZE=0" ;;
        stream)   echo "PER_RANK=4 NESTOR_DDP_SINGLE_RANK=1 NESTOR_STREAMED_HEAD_CE=1 NESTOR_GC_FREEZE=0" ;;
        noddp)    echo "PER_RANK=4 NESTOR_DDP_SINGLE_RANK=0 NESTOR_STREAMED_HEAD_CE=1 NESTOR_GC_FREEZE=0" ;;
        tuned)    echo "PER_RANK=8 NESTOR_DDP_SINGLE_RANK=0 NESTOR_STREAMED_HEAD_CE=1 NESTOR_GC_FREEZE=0" ;;
        gcfreeze) echo "PER_RANK=8 NESTOR_DDP_SINGLE_RANK=0 NESTOR_STREAMED_HEAD_CE=1 NESTOR_GC_FREEZE=1" ;;
        chunk8)   echo "PER_RANK=8 NESTOR_DDP_SINGLE_RANK=0 NESTOR_STREAMED_HEAD_CE=0 NESTOR_GC_FREEZE=0" ;;
        *) echo "unknown leg $1" >&2; return 2 ;;
    esac
}

echo "[bench] experiment=${EXPERIMENT} gpus=${NPROC_PER_NODE} steps=${STEPS} log_freq=${LOG_FREQ} rounds=${ROUNDS} legs=${LEGS}"
echo "[bench] commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?') node=$(hostname) root=${BENCH_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[bench] gpu: /'

FAILED=0
for round in $(seq 1 "${ROUNDS}"); do
    order="${LEGS}"
    if [ $(( round % 2 )) -eq 0 ]; then order="$(echo "${LEGS}" | tr ' ' '\n' | tac | tr '\n' ' ')"; fi
    for leg in ${order}; do
        knobs="$(leg_env "${leg}")" || exit 2
        log="${BENCH_ROOT}/logs/r${round}_${leg}.log"
        t0=$(date +%s)
        # shellcheck disable=SC2086
        env ${knobs} NPROC_PER_NODE="${NPROC_PER_NODE}" WORK_DIR="${WORK_DIR}" WANDB_MODE=disabled \
            DCLM_MAX_STEPS="${STEPS}" DCLM_LOG_FREQ="${LOG_FREQ}" DCLM_CKPT_FREQ=1000000 \
            DCLM_WARMUP_STEPS="${DCLM_WARMUP_STEPS:-${STEPS}}" \
            DCLM_RUN_SUFFIX="bench_r${round}_${leg}" \
            NESTOR_OUTPUT_ROOT="${BENCH_ROOT}/runs/r${round}_${leg}" \
            bash scripts/slurm_pretrain.sh "${EXPERIMENT}" > "${log}" 2>&1
        rc=$?
        echo "[bench] round ${round} leg ${leg} (${knobs}): rc=${rc} in $(( $(date +%s) - t0 )) s"
        if [ "${rc}" -ne 0 ]; then FAILED=1; tail -n 25 "${log}" | sed 's/^/[bench]   /'; fi
        grep -E "\[PERF\] (gc.freeze|one rank)" "${log}" | sed 's/^.*\[PERF\]/[bench]   [PERF]/' | sort -u
        # The legs only exist to be timed: drop their checkpoints right away.
        rm -rf "${BENCH_ROOT}/runs/r${round}_${leg}"
    done
done

echo "[bench] === summary (leading ${SKIP_WINDOWS} window(s) of each leg dropped)"
python scripts/bench_summary.py "${BENCH_ROOT}/logs" --baseline "${BASELINE}" --skip-windows "${SKIP_WINDOWS}"
echo "[bench] logs kept under ${BENCH_ROOT}/logs"
exit "${FAILED}"
