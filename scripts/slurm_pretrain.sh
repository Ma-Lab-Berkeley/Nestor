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
export NESTOR_DDP=1                    # DDP, not FSDP: every model fits on one GPU
export NESTOR_COMPILE=1                # torch.compile, fullgraph=False
export NESTOR_BF16=1                   # bf16 autocast, no GradScaler
export NESTOR_DENSE_NO_ROUTER=1        # skip + freeze the gate-1.0 routers of dense blocks
export NESTOR_RUNAHEAD=0
export NESTOR_DDP_BROADCAST_BUFFERS=1  # loss-free balancing biases are synced every step
export NESTOR_DDP_STATIC_GRAPH=0
export NESTOR_GATHERED_CAUSAL_FLASH=1
export NESTOR_GATHERED_VARLEN=1
export NESTOR_MOE_GROUPED_MM=1         # grouped-GEMM expert dispatch
export NESTOR_CE_CHUNK_ROWS="${NESTOR_CE_CHUNK_ROWS:-4096}"
export NESTOR_COLUMN_SLOT_UPDATES=1
export NESTOR_DENSE_TAIL_ATTN=1
export NESTOR_GROUPED_MM_FUSED_GATE=1
export NESTOR_ATTN_PACKED_VARLEN=1     # layer-packed varlen attention for the column blocks
export NESTOR_DDP_NO_SYNC=1            # one gradient all-reduce per optimizer step
export NESTOR_MUON_SHARD=1             # Muon momentum sharded across ranks
export NESTOR_STACKED_EXPERT_BANKS=1   # fresh runs only: changes the parameter layout
export NESTOR_BATCHED_ROUTERS=1
# Triton kernels for the packed column path (need triton; harmless on dense arms).
export NESTOR_TRITON_PACKED_ATTN="${NESTOR_TRITON_PACKED_ATTN:-1}"
export NESTOR_TRITON_INDEX_BUILD="${NESTOR_TRITON_INDEX_BUILD:-1}"
export NESTOR_TRITON_EXPERT_DISPATCH="${NESTOR_TRITON_EXPERT_DISPATCH:-1}"

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
