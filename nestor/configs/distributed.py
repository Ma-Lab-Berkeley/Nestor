"""Distributed training configuration."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(kw_only=True)
class DistributedConfig:
    """
    Configuration for distributed training.

    Distributed training is always performed with FSDP2 (``fully_shard``).

    Attributes:
        enabled (bool): Whether distributed training is enabled.
        backend (Literal["nccl", "gloo"]): PyTorch distributed backend.
        init_method (str): URL for initializing the process group.
            Use "env://" for torchrun/environment variable based init.
        world_size (int | None): Total number of processes. If None, inferred from environment.
        rank (int | None): Global rank of this process. If None, inferred from environment.
        local_rank (int | None): Local rank on this node. If None, inferred from environment.
        fsdp2_reshard_after_forward (bool | int | None): FSDP2 reshard policy passed
            to ``fully_shard(...)``.
        fsdp2_param_dtype (Literal["float32", "float16", "bfloat16"] | None):
            Parameter compute dtype for FSDP2 mixed precision.
        fsdp2_reduce_dtype (Literal["float32", "float16", "bfloat16"] | None):
            Gradient reduction dtype for FSDP2 mixed precision.
        fsdp2_output_dtype (Literal["float32", "float16", "bfloat16"] | None):
            Output dtype for FSDP2 mixed precision.
        fsdp2_cpu_offload (bool): Whether FSDP2 should offload sharded state to CPU.
    """

    enabled: bool = False
    backend: Literal["nccl", "gloo"] = "nccl"
    init_method: str = "env://"
    world_size: int | None = None
    rank: int | None = None
    local_rank: int | None = None
    fsdp2_reshard_after_forward: bool | int | None = None
    fsdp2_param_dtype: Literal["float32", "float16", "bfloat16"] | None = None
    fsdp2_reduce_dtype: Literal["float32", "float16", "bfloat16"] | None = None
    fsdp2_output_dtype: Literal["float32", "float16", "bfloat16"] | None = None
    fsdp2_cpu_offload: bool = False
