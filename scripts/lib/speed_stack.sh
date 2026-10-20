#!/usr/bin/env bash
# The speed stack every paper run trained with: exact reformulations of the
# eager forward, engaged through NESTOR_* flags read by the trainer and the
# backbone. Sourced by scripts/slurm_pretrain.sh and the continued-training
# launchers; every flag can be preset in the environment before sourcing.
#
# NESTOR_STACKED_EXPERT_BANKS changes the PARAMETER LAYOUT of the expert
# banks (one stacked tensor per bank instead of per-expert modules). A run
# that initialises from another run's checkpoint must build the model in the
# layout of that file: 1 for a checkpoint trained with this launcher, 0 for a
# research-code checkpoint that predates stacking (the trainer refuses a
# layout mismatch rather than training random experts).
export NESTOR_DDP="${NESTOR_DDP:-1}"                                  # DDP, not FSDP: every model fits on one GPU
export NESTOR_COMPILE="${NESTOR_COMPILE:-1}"                          # torch.compile, fullgraph=False
export NESTOR_BF16="${NESTOR_BF16:-1}"                                # bf16 autocast, no GradScaler
export NESTOR_DENSE_NO_ROUTER="${NESTOR_DENSE_NO_ROUTER:-1}"          # skip + freeze the gate-1.0 routers of dense blocks
export NESTOR_RUNAHEAD="${NESTOR_RUNAHEAD:-0}"
export NESTOR_DDP_BROADCAST_BUFFERS="${NESTOR_DDP_BROADCAST_BUFFERS:-1}"  # loss-free balancing biases are synced every step
export NESTOR_DDP_STATIC_GRAPH="${NESTOR_DDP_STATIC_GRAPH:-0}"
export NESTOR_GATHERED_CAUSAL_FLASH="${NESTOR_GATHERED_CAUSAL_FLASH:-1}"
export NESTOR_GATHERED_VARLEN="${NESTOR_GATHERED_VARLEN:-1}"
export NESTOR_MOE_GROUPED_MM="${NESTOR_MOE_GROUPED_MM:-1}"            # grouped-GEMM expert dispatch
export NESTOR_CE_CHUNK_ROWS="${NESTOR_CE_CHUNK_ROWS:-4096}"
export NESTOR_COLUMN_SLOT_UPDATES="${NESTOR_COLUMN_SLOT_UPDATES:-1}"
export NESTOR_DENSE_TAIL_ATTN="${NESTOR_DENSE_TAIL_ATTN:-1}"
export NESTOR_GROUPED_MM_FUSED_GATE="${NESTOR_GROUPED_MM_FUSED_GATE:-1}"
export NESTOR_ATTN_PACKED_VARLEN="${NESTOR_ATTN_PACKED_VARLEN:-1}"    # layer-packed varlen attention for the column blocks
export NESTOR_DDP_NO_SYNC="${NESTOR_DDP_NO_SYNC:-1}"                  # one gradient all-reduce per optimizer step
export NESTOR_MUON_SHARD="${NESTOR_MUON_SHARD:-1}"                    # Muon momentum sharded across ranks
export NESTOR_STACKED_EXPERT_BANKS="${NESTOR_STACKED_EXPERT_BANKS:-1}"  # see the note above
export NESTOR_BATCHED_ROUTERS="${NESTOR_BATCHED_ROUTERS:-1}"
# Triton kernels for the packed column path (need triton; harmless on dense arms).
export NESTOR_TRITON_PACKED_ATTN="${NESTOR_TRITON_PACKED_ATTN:-1}"
export NESTOR_TRITON_INDEX_BUILD="${NESTOR_TRITON_INDEX_BUILD:-1}"
export NESTOR_TRITON_EXPERT_DISPATCH="${NESTOR_TRITON_EXPERT_DISPATCH:-1}"
