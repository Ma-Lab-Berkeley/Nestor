"""Distributed training utilities."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from nestor.configs.distributed import DistributedConfig
from nestor.models import Model, iter_unique_modules

_INITIALIZED = False


def init_distributed(config: DistributedConfig) -> None:
    """Initialize distributed training.

    Args:
        config (DistributedConfig): Distributed training configuration.
    """

    global _INITIALIZED

    if not config.enabled:
        if _INITIALIZED and dist.is_initialized():
            assert hasattr(dist, "destroy_process_group")
            dist.destroy_process_group()
        _INITIALIZED = False
        return

    if _INITIALIZED:
        return

    backend = config.backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    world_size = config.world_size
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    rank = config.rank
    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    local_rank = config.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()
        if not 0 <= local_rank < visible_device_count:
            raise RuntimeError(
                "Invalid CUDA local rank for the visible devices. "
                f"Got LOCAL_RANK={local_rank} with "
                f"torch.cuda.device_count()={visible_device_count}. "
                "This usually means torchrun launched more processes than the "
                "number of visible GPUs. Set torchrun --nproc_per_node to match "
                "the visible CUDA devices (or fix CUDA_VISIBLE_DEVICES)."
            )
        torch.cuda.set_device(local_rank)

    assert hasattr(dist, "init_process_group")
    # Collective timeout. The default (~10 min for NCCL) is shorter than a cold
    # torch.compile of a large model, so on the FIRST optimizer step the gradient
    # all-reduce can fire before slower-compiling ranks arrive and trip the
    # watchdog (observed on the 32-GPU 3B-MoE run: first all-reduce timed out at
    # 620s on a cold cache; the warm-cache retry then succeeded). Default to 30
    # min so a one-time cold compile rides through; override via env.
    # 30 min was NOT enough for the 1b Kimi-read + per-layer-recompute arm on
    # a cold job-local cache (the first collective timed out at 33 min while
    # slower ranks were still compiling; two attempts died the same way).
    # 90 min matches a typical idle-GPU sweeper's zero-utilization kill
    # horizon, so a genuinely hung collective is reaped at the same time
    # either way.
    timeout_min = float(os.environ.get("NESTOR_DIST_TIMEOUT_MIN", "90"))
    dist.init_process_group(
        backend=backend,
        init_method=config.init_method,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=timeout_min),
    )

    _INITIALIZED = True


def is_distributed() -> bool:
    """Check if distributed training is initialized.

    Returns:
        bool: Whether a process group is initialized.
    """

    return _INITIALIZED and dist.is_initialized()


def is_main_process() -> bool:
    """Check if this is rank 0.

    Returns:
        bool: Whether this process is the main process.
    """

    if not is_distributed():
        return True
    assert hasattr(dist, "get_rank")
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the world size.

    Returns:
        int: Number of distributed ranks.
    """

    if not is_distributed():
        return 1
    assert hasattr(dist, "get_world_size")
    return dist.get_world_size()


def get_per_rank_batch_size(
    global_batch_size: int,
    *,
    world_size: int | None = None,
) -> int:
    """Convert a global batch size into the per-rank batch size.

    Args:
        global_batch_size (int): Batch size configured at the job level.
        world_size (int | None): Optional world size override.

    Returns:
        int: Batch size for one rank.

    Raises:
        ValueError: If the batch size or world size is invalid.
    """

    if global_batch_size <= 0:
        raise ValueError(
            f"Global batch size must be positive, got {global_batch_size}."
        )

    effective_world_size = get_world_size() if world_size is None else world_size
    if effective_world_size <= 0:
        raise ValueError(f"World size must be positive, got {effective_world_size}.")

    if global_batch_size % effective_world_size != 0:
        raise ValueError(
            "Global batch size must be divisible by world size. "
            f"Got global_batch_size={global_batch_size}, "
            f"world_size={effective_world_size}."
        )

    return global_batch_size // effective_world_size


def get_rank() -> int:
    """Get the global rank.

    Returns:
        int: Global rank, or ``0`` when not distributed.
    """

    if not is_distributed():
        return 0
    assert hasattr(dist, "get_rank")
    return dist.get_rank()


def get_local_rank() -> int:
    """Get the local rank.

    Returns:
        int: Local rank, or ``0`` when not distributed.
    """

    if not is_distributed():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def barrier() -> None:
    """Synchronize all ranks."""

    if is_distributed():
        assert hasattr(dist, "barrier")
        dist.barrier()


def get_padded_rank_indices(num_items: int) -> tuple[list[int], int]:
    """Return this rank's contiguous eval shard plus padded dummy indices.

    Unlike ``DistributedSampler(drop_last=False)``, this helper makes padding
    explicit: callers get both padded indices and the number of valid indices so
    duplicated dummy examples can be excluded from metrics.

    Args:
        num_items (int): Number of examples in the full dataset.

    Returns:
        tuple[list[int], int]: This rank's padded indices and count of valid
            non-padding examples.

    Raises:
        ValueError: If ``num_items`` is negative.
    """
    if num_items < 0:
        raise ValueError(f"num_items must be non-negative, got {num_items}.")
    if not is_distributed():
        return list(range(num_items)), num_items
    if num_items == 0:
        return [], 0

    world_size = get_world_size()
    rank = get_rank()
    per_rank = (num_items + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, num_items)
    indices = list(range(start, end))
    valid_count = len(indices)
    indices.extend([0] * (per_rank - valid_count))
    return indices, valid_count


def sum_tensor_across_ranks(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across distributed ranks.

    Args:
        tensor (torch.Tensor): Tensor to reduce in place.

    Returns:
        torch.Tensor: The same tensor after an all-reduce sum when distributed
            is initialized.
    """
    if is_distributed():
        assert hasattr(dist, "all_reduce")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def cleanup_distributed() -> None:
    """Clean up distributed training."""

    global _INITIALIZED
    if is_distributed():
        assert hasattr(dist, "destroy_process_group")
        dist.destroy_process_group()
    _INITIALIZED = False


def is_fsdp2_model(model: nn.Module) -> bool:
    """Check whether a model is wrapped by FSDP2.

    Args:
        model (nn.Module): Possibly wrapped model.

    Returns:
        bool: Whether the model is an FSDP2 module.
    """

    from torch.distributed.fsdp import FSDPModule

    return isinstance(model, FSDPModule)


def fused_head_param_names(model: nn.Module) -> list[str]:
    """Names of the parameters DDP must leave to :func:`reduce_ignored_grads`.

    Under ``NESTOR_FUSED_HEAD_CE=1`` the language-modelling head is applied
    inside the loss (row-chunked, checkpointed) rather than in the model's
    forward, so its parameters must not be tracked by DDP's reducer.

    Args:
        model (nn.Module): The unwrapped top-level model (a ``SequenceModel``
            whose head lives under the ``head`` attribute).

    Returns:
        list[str]: Fully qualified parameter names under ``model.head``;
            empty when the model has no ``head`` submodule.
    """

    head = getattr(model, "head", None)
    if not isinstance(head, nn.Module):
        return []
    return [f"head.{name}" for name, _ in head.named_parameters()]


def reduce_ignored_grads(model: nn.Module) -> None:
    """Average across ranks the gradients of parameters DDP ignores.

    DDP reduces only the parameters it tracks; the fused head's parameters
    (see :func:`fused_head_param_names`) are excluded because their grad
    hooks fire inside the loss, outside the reducer's bookkeeping. This
    all-reduces their ``.grad`` (sum, then divide by the world size) so the
    optimizer sees the same mean gradient DDP would have produced. Call once
    per optimizer step, on every rank, after the accumulation window's last
    backward and before clipping. A no-op for non-DDP models, single-process
    runs, and models with nothing ignored.

    Args:
        model (nn.Module): The training model, possibly ``torch.compile``d
            and/or DDP-wrapped.
    """

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Find the DDP wrapper itself (``_unwrap_compiled`` would strip it too):
    # peel ``torch.compile`` layers only, in either nesting order.
    ddp: Any = model
    while not isinstance(ddp, DDP) and hasattr(ddp, "_orig_mod"):
        ddp = ddp._orig_mod
    if not isinstance(ddp, DDP) or not dist.is_initialized():
        return
    ignored = getattr(ddp, "parameters_to_ignore", None)
    if not ignored:
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    params = dict(ddp.module.named_parameters())
    for name in sorted(ignored):
        param = params.get(name)
        if param is None:
            continue
        if param.grad is None:
            # Keep the collective count identical on every rank.
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def wrap_model_for_distributed(
    model: Model,
    config: DistributedConfig,
    *,
    device: torch.device,
) -> nn.Module:
    """Shard a model with FSDP2 according to the distributed configuration.

    Args:
        model (Model): Unwrapped top-level model.
        config (DistributedConfig): Distributed configuration.
        device (torch.device): Device used by the current rank.

    Returns:
        nn.Module: The original model. FSDP2 shards the parameters in-place
            via ``fully_shard(...)``.
    """

    if not config.enabled:
        return model

    # NESTOR_DDP=1: plain DistributedDataParallel instead of FSDP2. For models
    # that fit on one GPU (ViT-S ~23M, nestorbaseline ~250M << 141GB), parameter
    # sharding buys no memory but costs a per-layer all-gather (fwd) +
    # reduce-scatter (bwd); DDP just replicates and does ONE gradient all-reduce
    # per step, fully overlapped with backward. Params stay fp32 and the trainer's
    # autocast handles bf16 compute (compile fuses the casts). broadcast_buffers
    # off (RoPE inv_freq etc. are identical across ranks); static_graph=True so
    # torch.compile's DDPOptimizer buckets cleanly (the graph is static -- the
    # MoE expert-bank guard keeps every param used every step).
    if os.environ.get("NESTOR_DDP") == "1":
        from torch.nn.parallel import DistributedDataParallel as DDP

        # broadcast_buffers=True (default): the Nestor routers keep load-balancing
        # bias BUFFERS (choice_bias) that are UPDATED in forward; with sync off
        # they drift apart across ranks -> the ranks become different models and
        # eval top-1 drops ~15pt (verified: DDP 0.31 vs FSDP/original 0.47 @10k).
        # The expert-bank guard keeps every param used each step, so no
        # find_unused_parameters needed (DDP confirmed 0 unused for ViT).
        #
        # NESTOR_DDP_BROADCAST_BUFFERS=0 turns the per-forward buffer broadcast
        # off. Safe whenever no router uses loss-free balancing (choice_bias is
        # then never written, and the remaining buffers — RoPE inv_freq, cached
        # masks — are identical across ranks by construction); the broadcast is
        # then ~25 pointless tiny NCCL collectives per step.
        # Activation checkpointing moves when gradients are produced -- a
        # sublayer's grads now arrive during its recompute, not on the first
        # pass -- and DDP's reducer rejects that with "Expected to have finished
        # reduction in the prior iteration" (reproduced on gloo with a real
        # Nestor backbone in projection mode). static_graph tells the reducer
        # the ordering is fixed, which is the documented pairing for
        # checkpointing and is true here: the expert-bank guard keeps every
        # parameter used every step. NESTOR_DDP_STATIC_GRAPH forces it either
        # way for anyone who needs to A/B it.
        # static_graph is the FAST pairing, but it asserts the used-parameter
        # set never changes between iterations, and that does not always hold:
        # a 200m column arm at per-rank 16 x accum 2 died with
        # "one parameter is used in first iteration, but then got unused in the
        # second ... not compatible with static_graph", because at that batch
        # shape some routed group draws no tokens in one micro-batch. The same
        # arm at per-rank 8 x accum 4 is fine. Fall back to
        # find_unused_parameters (slower, but tolerates a varying set) with
        # NESTOR_DDP_STATIC_GRAPH=0; the two are mutually exclusive.
        checkpointing = os.environ.get("NESTOR_ACTIVATION_CHECKPOINT") == "1"
        static_graph_env = os.environ.get("NESTOR_DDP_STATIC_GRAPH")
        static_graph = (
            static_graph_env == "1"
            if static_graph_env is not None
            else checkpointing
        )
        find_unused = not static_graph and (
            checkpointing or os.environ.get("NESTOR_DDP_FIND_UNUSED") == "1"
        )
        if os.environ.get("NESTOR_FUSED_HEAD_CE") == "1":
            # The fused head+CE runs the head projection inside the LOSS,
            # outside DDP's forward, checkpointed once per row chunk -- so the
            # head weight's grad hooks fire outside the reducer's bookkeeping
            # and once per chunk. Both reducer modes reject that: find_unused
            # with "Expected to mark a variable ready only once", static_graph
            # with the reducer.cpp:1703 ``expect_autograd_hooks_`` assert. The
            # head therefore leaves the reducer entirely and its gradients are
            # all-reduced
            # by hand once per optimizer step (:func:`reduce_ignored_grads`,
            # called by the trainer before clipping) -- the mean of the
            # accumulated local sums, exactly what DDP computes.
            head_params = fused_head_param_names(model)
            DDP._set_params_and_buffers_to_ignore_for_model(model, head_params)
        # bf16-master parameters (NESTOR_BF16_MASTER=1) accumulate their own fp32
        # gradients and are reduced to their owners by the trainer; DDP must not
        # bucket their bf16 grads. Merge with the fused head's ignore list.
        bf16_names = list(getattr(model, "_bf16_master_param_names", []) or [])
        if bf16_names:
            already = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []) or [])
            DDP._set_params_and_buffers_to_ignore_for_model(
                model, sorted(set(already) | set(bf16_names))
            )
        return DDP(
            model,
            device_ids=[get_local_rank()] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            broadcast_buffers=(os.environ.get("NESTOR_DDP_BROADCAST_BUFFERS") != "0"),
            static_graph=static_graph,
            find_unused_parameters=find_unused,
        )

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )

    mesh = init_device_mesh(
        device_type="cuda" if device.type == "cuda" else device.type,
        mesh_shape=(get_world_size(),),
    )
    # NESTOR_BF16=1 stores/all-gathers params in bf16 so the forward runs in
    # bf16 with NO per-op fp32->fp16 autocast conversion copies (those copies are
    # ~30% of a real ViT-S step), and enables the SDPA flash backend. reduce in
    # fp32 keeps gradient accumulation accurate.
    _bf16 = os.environ.get("NESTOR_BF16") == "1"
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if _bf16 else _parse_dtype(config.fsdp2_param_dtype),
        reduce_dtype=torch.float32
        if _bf16
        else _parse_dtype(config.fsdp2_reduce_dtype),
        output_dtype=_parse_dtype(config.fsdp2_output_dtype),
    )
    offload_policy = CPUOffloadPolicy() if config.fsdp2_cpu_offload else OffloadPolicy()
    # Env override for diagnosis/tuning: NESTOR_FSDP_RESHARD=true|false|<int>.
    # reshard_after_forward=False keeps params gathered between fwd/bwd,
    # trading memory for fewer all-gathers (less comm) in the backward pass.
    reshard_after_forward = config.fsdp2_reshard_after_forward
    _env_reshard = os.environ.get("NESTOR_FSDP_RESHARD")
    if _env_reshard is not None:
        _val = _env_reshard.strip().lower()
        if _val in ("true", "false"):
            reshard_after_forward = _val == "true"
        else:
            reshard_after_forward = int(_val)
    for module in iter_unique_modules(model.fsdp2_wrap_order()):
        fully_shard(
            module,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
    return model


def _parse_dtype(dtype_name: str | None) -> torch.dtype | None:
    """Convert a config dtype string into a ``torch.dtype``.

    Args:
        dtype_name (str | None): Config dtype name.

    Returns:
        torch.dtype | None: Torch dtype, or ``None``.

    Raises:
        ValueError: If the dtype name is unsupported.
    """

    if dtype_name is None:
        return None

    dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(dtype_map))
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Allowed values: {allowed}."
        ) from exc
