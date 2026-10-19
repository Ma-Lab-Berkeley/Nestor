"""Training loop implementation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

from nestor.backbones import Backbone, ModelOutput
from nestor.configs.jobs import TrainJobConfig
from nestor.train.activation_offload import activation_offload
from nestor.train.checkpoint import load_checkpoint, save_checkpoint
from nestor.train.losses import Loss, LossOutput
from nestor.train.optim import build_optimizer, build_scheduler
from nestor.utils.batches import split_supervised_batch
from nestor.utils.distributed import is_main_process
from nestor.utils.logging import log_metrics

if TYPE_CHECKING:
    from nestor.eval.suite import EvalSuite
    from nestor.models import Model


def grad_sync_context(
    model: nn.Module, enabled: bool, last_in_window: bool
) -> Any:
    """Context for one micro-batch's forward/backward under DDP accumulation.

    With ``enabled`` (``NESTOR_DDP_NO_SYNC=1``) and a
    :class:`~torch.nn.parallel.DistributedDataParallel` model, every
    micro-batch but the last of an accumulation window runs inside
    ``model.no_sync()``: its gradients accumulate locally and the window's
    final backward all-reduces the sum once, instead of DDP reducing every
    micro-batch's gradient (A times the volume for an A-step window). The
    mean over ranks commutes with the sum over micro-batches, so the update
    is the same up to fp32 reassociation. A null context otherwise -- flag
    off, the window's last micro-batch, or a model that is not DDP (FSDP2
    manages its own reduction; a single-process model has nothing to sync).

    The DDP instance is looked up THROUGH ``torch.compile`` wrappers: the
    pipeline wraps ``DDP(model)`` first and compiles the result, so the
    trainer's ``self.model`` is an ``OptimizedModule`` whose ``_orig_mod`` is
    the DDP module. Testing ``isinstance(model, DDP)`` on the outer object
    made this flag a silent no-op under ``NESTOR_COMPILE=1``: DDP reduced
    every micro-batch, 4x the gradient volume at accum 4, which is what
    starved a 4-node 500m run (66 GB/step/node over one IB rail, 13.3 s/step).

    Args:
        model (nn.Module): The (possibly DDP-wrapped and/or compiled)
            training model.
        enabled (bool): Whether deferred synchronization is requested.
        last_in_window (bool): Whether this micro-batch completes the
            accumulation window (its backward must reduce).

    Returns:
        Any: A context manager wrapping the forward/backward.
    """

    import contextlib

    from torch.nn.parallel import DistributedDataParallel as _DDP

    ddp = model
    while not isinstance(ddp, _DDP) and hasattr(ddp, "_orig_mod"):
        ddp = ddp._orig_mod  # torch.compile's OptimizedModule -> wrapped module
    if last_in_window or not enabled or not isinstance(ddp, _DDP):
        return contextlib.nullcontext()
    return ddp.no_sync()


class Trainer:
    """Training loop manager."""

    def __init__(
        self,
        model: "Model | nn.Module",
        loss_fn: Loss,
        train_loader: DataLoader,
        eval_suite: "EvalSuite | None",
        config: TrainJobConfig,
        run_dir: Path,
        device: torch.device,
        logger: logging.Logger,
    ):
        """Initialize the trainer.

        Args:
            model (Model | nn.Module): Possibly wrapped top-level model.
            loss_fn (Loss): Loss function.
            train_loader (DataLoader): Training data loader.
            eval_suite (EvalSuite | None): Optional evaluation suite.
            config (TrainJobConfig): Training configuration.
            run_dir (Path): Run output directory.
            device (torch.device): Device for this rank.
            logger (logging.Logger): Logger instance.
        """

        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.eval_suite = eval_suite
        self.config = config
        self.run_dir = run_dir
        self.device = device
        self.logger = logger

        self.optimizer = build_optimizer(config.optimizer, model)
        self.scheduler = build_scheduler(
            config.scheduler,
            self.optimizer,
            config.max_steps,
        )
        # NESTOR_BF16: run autocast in bf16 (matches bf16 FSDP params -> no
        # fp32->fp16 conversion copies) and drop GradScaler (bf16 has fp32 range,
        # no loss scaling needed).
        import os as _os

        self._bf16 = _os.environ.get("NESTOR_BF16") == "1"
        self._amp_dtype = torch.bfloat16 if self._bf16 else torch.float16
        self.scaler = (
            GradScaler() if (config.use_amp and not self._bf16) else None
        )
        # Detached router-bias updates (loss-free balancing) need the aux merged
        # and walked every step; when no router balances, skip that work in the
        # hot loop entirely.
        self._needs_detached_updates = any(
            bool(getattr(module, "loss_free_balancing", False))
            for module in model.modules()
        )
        # Zero-filling missing grads exists for FSDP2 (its clip_grad_norm_
        # collectives must see the same parameter count on every rank). DDP
        # already produces a grad for every tracked parameter, and the
        # single-process path has no collectives — skip the per-step walk there.
        from torch.nn.parallel import DistributedDataParallel as _DDP

        from nestor.train.checkpoint import _unwrap_compiled
        from nestor.utils.distributed import is_distributed as _is_dist

        self._fill_missing_grads = _is_dist() and not isinstance(
            _unwrap_compiled(model), _DDP
        )
        # NESTOR_RUNAHEAD bounds how many optimizer steps the CPU may run
        # ahead of the GPU. Unbounded run-ahead (-1) regressed 4-GPU DDP ~24%
        # (0.222 -> 0.276 s/step): free-running ranks drain the loader prefetch
        # queues, so every step's collectives bind to the slowest rank's
        # just-in-time batch. 0 waits for the GPU after every step (the old
        # cadence, bisect-verified to restore full speed); K>0 waits on the
        # end event from K steps back, keeping a small bounded pipeline.
        import collections as _collections

        self._runahead = int(_os.environ.get("NESTOR_RUNAHEAD", "0"))
        self._step_end_events: _collections.deque = _collections.deque()
        # NESTOR_DDP_NO_SYNC=1: under gradient accumulation, all-reduce the
        # gradients ONCE per optimizer step instead of once per micro-batch.
        # DDP reduces on every backward by default, so an accumulation window
        # of A micro-batches moved A times the gradient volume (3.5 GB fp32
        # per micro-batch on the 200m column arms, ~10 GB at Q=32);
        # ``model.no_sync()`` on the first A-1 backward passes accumulates
        # locally and the final one reduces the sum. Same math (the mean over
        # ranks commutes with the sum over micro-batches, up to fp32
        # reassociation); DDP's buffer broadcast then also happens once per
        # window, right after the loss-free bias update it exists to
        # propagate. Off by default until measured (see _grad_sync_context).
        self._ddp_no_sync = _os.environ.get("NESTOR_DDP_NO_SYNC") == "1"
        # Model EMA (config.ema_decay): a fp32 shadow of the unwrapped model's
        # params, updated after every optimizer step. Eval swaps these in. Built
        # from the unwrapped (no compile/DDP) module so the keys are stable for
        # checkpoint save/resume. NESTOR_EMA_DECAY overrides the config.
        _ema_env = _os.environ.get("NESTOR_EMA_DECAY")
        self._ema_decay = (
            float(_ema_env) if _ema_env else config.ema_decay
        )
        self._ema: dict[str, torch.Tensor] | None = None
        self._ema_backup: dict[str, torch.Tensor] | None = None
        if self._ema_decay is not None:
            _base = _unwrap_compiled(model)
            self._ema = {
                name: p.detach().clone().float()
                for name, p in _base.named_parameters()
            }
        self.global_step = 0
        self.epoch = 0

        if config.resume_model_checkpoint is not None:
            self._resume_from_checkpoint(config.resume_model_checkpoint)

    def _resume_from_checkpoint(self, model_checkpoint_path: Path) -> None:
        """Resume training from a saved checkpoint.

        Args:
            model_checkpoint_path (Path): Path to the model checkpoint.
        """

        self.logger.info(f"Resuming model from: {model_checkpoint_path}")
        if self.config.resume_strict:
            checkpoint = load_checkpoint(
                model_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
                strict=True,
            )
            self.global_step = checkpoint.get("step", 0)
            self.epoch = checkpoint.get("epoch", 0)
            self.logger.info(
                f"Resumed at step {self.global_step}, epoch {self.epoch}"
            )
        else:
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            ckpt_state = checkpoint["model"]
            from nestor.train.checkpoint import _unwrap_compiled, filter_init_state

            base_model = _unwrap_compiled(self.model)
            # Refuses to leave any parameter at random init (a stacked-bank
            # file into an unstacked model, or vice versa); DCLM_INIT_ALLOW_PARTIAL=1
            # opts out for a deliberately new module.
            filtered, skipped = filter_init_state(
                ckpt_state,
                base_model.state_dict(),
                set(dict(base_model.named_parameters())),
                allow_partial=os.environ.get("DCLM_INIT_ALLOW_PARTIAL", "0") == "1",
            )
            base_model.load_state_dict(filtered, strict=False)
            if skipped:
                self.logger.info(
                    f"Skipped {len(skipped)} checkpoint keys "
                    f"(shape mismatch or missing): {skipped}"
                )
            self.logger.info(
                f"Loaded {len(filtered)}/{len(ckpt_state)} weights (non-strict); "
                "optimizer/scheduler/step reset to initial state"
            )

        # Restore (or re-seed) the EMA shadow. If the checkpoint carries an EMA
        # state, adopt it; otherwise re-seed from the just-loaded weights so the
        # shadow is not left at random init when resuming a pre-EMA checkpoint.
        if self._ema is not None:
            from nestor.train.checkpoint import _unwrap_compiled

            ckpt_ema = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
            if ckpt_ema:
                self._ema = {k: v.detach().clone().float() for k, v in ckpt_ema.items()}
                self.logger.info("Restored EMA shadow from checkpoint")
            else:
                base = _unwrap_compiled(self.model)
                self._ema = {
                    name: p.detach().clone().float()
                    for name, p in base.named_parameters()
                }
                self.logger.info("Re-seeded EMA shadow from loaded weights")

    def train(self) -> None:
        """Run the training loop."""

        # NESTOR_DIAG_OOM_SNAPSHOT=1: allocator snapshot dumped at the first
        # OOM (nestor/train/oom_snapshot.py); inert otherwise. The observer
        # fires inside the allocator; the except path below is the fallback
        # for a torch build without the observer hook.
        from nestor.train.oom_snapshot import install_oom_snapshot
        from nestor.utils.distributed import get_rank

        oom_dumper = install_oom_snapshot(self.run_dir, get_rank(), self.logger)
        self._oom_dumper = oom_dumper
        if oom_dumper is not None:
            try:
                self._train_body()
            except torch.OutOfMemoryError:
                oom_dumper()
                raise
            return
        self._train_body()

    def _train_body(self) -> None:
        """The training loop proper (``train`` minus the diagnostic wrappers)."""

        import os

        self._enter_train_mode()
        # Loss accumulators hold GPU tensors and are read with .item() only at
        # log boundaries: a per-microbatch .item() forces a full pipeline drain
        # every step (CPU can't run ahead to enqueue the next H2D/forward while
        # the GPU finishes backward+optimizer), which showed up as a ~30ms/step
        # bubble on the 4-GPU ViT-S runs.
        accum_loss: torch.Tensor | None = None
        accum_steps = 0
        accum_component_sums: dict[str, torch.Tensor] = {}
        accum_model_aux_batches: list[dict[str, Any]] = []

        self.logger.info(f"Starting training for {self.config.max_steps} steps")

        # --- MFU monitoring setup (guarded; never blocks training) ---
        import time as _time

        from nestor.utils.timing import PROCESS_START

        peak_flops = float(os.environ.get("NESTOR_MFU_PEAK_TFLOPS", "989")) * 1e12
        # The calibration's probe forward+backward may OOM by design (caught
        # below); it must not spend the OOM snapshot's one shot.
        oom_dumper = getattr(self, "_oom_dumper", None)
        if oom_dumper is not None:
            oom_dumper.suspended = True
        try:
            calibrated = self._calibrate_flops()
        finally:
            if oom_dumper is not None:
                oom_dumper.suspended = False
        flops_per_step = calibrated * max(
            1, self.config.grad_accum_steps
        )
        global_batch = self.config.train_dataloader.batch_size
        # Env override so a short measurement run surfaces MFU quickly.
        log_frequency = int(
            os.environ.get("NESTOR_LOG_FREQ") or self.config.log_frequency
        )
        # Throughput is measured over whole log windows (the .item() sync at a
        # log boundary makes the window wall time accurate); per-step timing
        # would only measure CPU launch time now that the loop never syncs
        # between logs.
        mfu_t_steady: float | None = None
        mfu_steps = 0
        mfu_win_t0 = _time.monotonic()
        mfu_win_steps = 0

        # Env-gated per-phase profiling of the REAL loop (data/fwdbwd/update).
        # Diagnostic harness and train() disagreed on s/step; this pins where
        # the real loop's time goes. Adds syncs, so only on when requested.
        _tprof = os.environ.get("NESTOR_TRAIN_PROFILE") == "1"
        _t_prev_end = _time.perf_counter()

        # Env-gated torch.profiler over the REAL loop: names the kernels/ops that
        # make a real train step ~3x the synthetic CUDA-event estimate. Profiles
        # a small window after warmup, dumps the top ops once, then disables.
        _torchprof = None
        if os.environ.get("NESTOR_TORCH_PROFILE") == "1":
            from torch.profiler import ProfilerActivity
            from torch.profiler import profile as _torch_profile

            _torchprof = _torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            )
        _tp_start = int(os.environ.get("NESTOR_TORCH_PROFILE_START", "12"))
        _tp_n = int(os.environ.get("NESTOR_TORCH_PROFILE_STEPS", "10"))
        _tp_active = False
        _tp_dumped = False

        # Fast-forward the dataloader after a mid-epoch resume. global_step is
        # restored from the checkpoint but `for batch in train_loader` restarts
        # at the FIRST batch of the epoch -- and a DCLM run covers a fraction
        # of one epoch, so a resumed run RE-READS the corpus from the start in
        # the same shuffled order: it repeats data it already trained on and
        # never reaches the untouched half of its budget (measured cost: the
        # resumed 500m arms landed 0.3-0.4 nats worse held-out than
        # scaling predicts). Skip the micro-batches already consumed this
        # epoch. Counted in optimizer steps x grad_accum, which is invariant
        # to per-rank batch-shape changes across legs (tokens/step is fixed).
        # Skipping pays one cheap mmap read per skipped batch (token bins have
        # no decode); NESTOR_RESUME_NO_FASTFORWARD=1 restores the old behavior.
        try:
            batches_per_epoch = len(self.train_loader)
        except TypeError:  # iterable-style loaders expose no length
            batches_per_epoch = 0
        _skip_batches = 0
        if (
            self.global_step > 0
            and batches_per_epoch > 0
            and os.environ.get("NESTOR_RESUME_NO_FASTFORWARD") != "1"
        ):
            consumed = self.global_step * self.config.grad_accum_steps
            _skip_batches = consumed % batches_per_epoch
            if _skip_batches:
                self.logger.info(
                    "[resume] fast-forwarding dataloader %d micro-batches "
                    "(step %d x accum %d, %d batches/epoch) so training "
                    "continues on unseen data",
                    _skip_batches,
                    self.global_step,
                    self.config.grad_accum_steps,
                    batches_per_epoch,
                )

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if _skip_batches > 0:
                    _skip_batches -= 1
                    continue
                if self.global_step >= self.config.max_steps:
                    break

                if _torchprof is not None and not _tp_dumped:
                    if mfu_steps == _tp_start and not _tp_active:
                        _torchprof.__enter__()
                        _tp_active = True
                    elif _tp_active and mfu_steps >= _tp_start + _tp_n:
                        _torchprof.__exit__(None, None, None)
                        _tp_active = False
                        _tp_dumped = True
                        if is_main_process():
                            self.logger.info(
                                "[TORCHPROF] top ops:\n%s",
                                _torchprof.key_averages().table(
                                    sort_by="cuda_time_total", row_limit=25
                                ),
                            )

                if _tprof:
                    _t_data = _time.perf_counter() - _t_prev_end
                    _t0 = _time.perf_counter()
                with self._grad_sync_context(
                    accum_steps + 1 >= self.config.grad_accum_steps
                ):
                    loss_output, model_aux = self._train_step(batch)
                if _tprof:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    _t_fb = _time.perf_counter() - _t0
                    _t1 = _time.perf_counter()
                # Track losses only for windows that end on a logged step; the
                # detached adds are tiny but pointless on the other 99%.
                _will_log = (self.global_step + 1) % log_frequency == 0
                if _will_log:
                    detached_total = loss_output.total.detach()
                    accum_loss = (
                        detached_total
                        if accum_loss is None
                        else accum_loss + detached_total
                    )
                    for name, value in loss_output.components.items():
                        prev = accum_component_sums.get(name)
                        detached = value.detach()
                        accum_component_sums[name] = (
                            detached if prev is None else prev + detached
                        )
                accum_steps += 1
                if self._needs_detached_updates:
                    accum_model_aux_batches.append(model_aux)

                if accum_steps >= self.config.grad_accum_steps:
                    if _tprof:
                        _t_item = _time.perf_counter() - _t1
                        _t2 = _time.perf_counter()
                    self._update_weights(accum_model_aux_batches)
                    if _tprof:
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        _t_upd = _time.perf_counter() - _t2
                        if is_main_process() and mfu_steps >= 5:
                            self.logger.info(
                                "[TPROF] data=%.3f fwd_bwd=%.3f item=%.3f "
                                "update=%.3f", _t_data, _t_fb, _t_item, _t_upd,
                            )
                    self.global_step += 1

                    # Bounded run-ahead (see __init__). K=0: full sync; K>0:
                    # wait for the step that finished K steps ago.
                    if self._runahead >= 0 and self.device.type == "cuda":
                        if self._runahead == 0:
                            torch.cuda.synchronize()
                        else:
                            end_event = torch.cuda.Event()
                            end_event.record()
                            self._step_end_events.append(end_event)
                            if len(self._step_end_events) > self._runahead:
                                self._step_end_events.popleft().synchronize()

                    mfu_steps += 1
                    if mfu_steps == 1:
                        # Steady clock starts after the first (compile-laden)
                        # step; sync once so it marks real GPU completion.
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        mfu_t_steady = _time.monotonic()
                        mfu_win_t0 = mfu_t_steady
                        mfu_win_steps = mfu_steps

                    if self.global_step % log_frequency == 0:
                        metrics = self._build_train_metrics(
                            accum_loss=(
                                accum_loss.item()
                                if accum_loss is not None
                                else 0.0
                            ),
                            accum_steps=accum_steps,
                            accum_component_sums={
                                name: value.item()
                                for name, value in accum_component_sums.items()
                            },
                        )
                        # The .item() reads above drained the pipeline, so the
                        # window wall time below is accurate.
                        mfu_now = _time.monotonic()
                        win_dt = (mfu_now - mfu_win_t0) / max(
                            mfu_steps - mfu_win_steps, 1
                        )
                        mfu_win_t0 = mfu_now
                        mfu_win_steps = mfu_steps
                        if self.scheduler is not None:
                            # Group 0 is the SMALLEST-scale Muon group under
                            # MuonAdamW (groups are sorted by spectral scale),
                            # so ``lr`` is not lr_muon and differs across
                            # architectures; ``lr_max`` is the largest group
                            # rate and follows the schedule shape directly.
                            last_lrs = self.scheduler.get_last_lr()
                            metrics["lr"] = last_lrs[0]
                            metrics["lr_max"] = max(last_lrs)
                        self._add_mfu_metrics(
                            metrics,
                            flops_per_step=flops_per_step,
                            peak_flops=peak_flops,
                            global_batch=global_batch,
                            now=mfu_now,
                            last_dt=win_dt,
                            steps=mfu_steps,
                            t_steady=mfu_t_steady,
                            process_start=PROCESS_START,
                        )
                        log_metrics(
                            self.logger,
                            metrics,
                            self.global_step,
                            prefix="train/",
                        )

                    accum_loss = None
                    accum_steps = 0
                    accum_component_sums = {}
                    accum_model_aux_batches = []

                    _eval_freq = int(
                        os.environ.get("NESTOR_EVAL_FREQ")
                        or self.config.eval_frequency
                    )
                    if (
                        self.eval_suite is not None
                        and self.global_step % _eval_freq == 0
                    ):
                        # Eval the LIVE model first (primary metric: the actual
                        # training trajectory, climbs from step 0). The EMA shadow
                        # with decay 0.9999 stays near-random until it warms (~tens
                        # of k steps), so evaluating only the EMA hides whether the
                        # live model is learning -- log it separately under *_ema.
                        self._run_eval_suite()
                        if self._ema is not None:
                            self._swap_in_ema()
                            try:
                                self._run_eval_suite(tag="_ema")
                            finally:
                                self._swap_out_ema()
                        self._enter_train_mode()

                    if self.global_step % self.config.checkpoint_frequency == 0:
                        self._save_checkpoint()
                        self._submit_eval_jobs()

                if _tprof:
                    _t_prev_end = _time.perf_counter()

            self.epoch += 1
            if hasattr(self.train_loader.sampler, "set_epoch"):
                cast(DistributedSampler, self.train_loader.sampler).set_epoch(
                    self.epoch
                )

        self._save_checkpoint(final=True)
        self.logger.info("Training complete!")

    def _calibrate_flops(self) -> float:
        """Measure FLOPs for one forward+backward microbatch (per rank).

        Used once at the start of training to drive MFU monitoring. Returns
        0.0 on any failure (MFU metrics are then simply omitted — training is
        never affected). ``NESTOR_MFU_FLOPS_PER_STEP`` pins the value and skips
        measurement.

        ``flex_attention`` is ``torch.compile``'d, so its score/context matmuls
        are invisible to ``FlopCounterMode``; we add them analytically from the
        per-layer (B, S, D) shapes captured via hooks, unless the counter is
        seen to already account for an SDPA op.
        """

        import os

        override = os.environ.get("NESTOR_MFU_FLOPS_PER_STEP")
        if override:
            try:
                value = float(override)
                self.logger.info("[MFU] flops/microbatch pinned via env: %.3e", value)
                return value
            except ValueError:
                pass

        try:
            from torch.utils.flop_counter import FlopCounterMode

            from nestor.utils.distributed import get_per_rank_batch_size

            adapter = getattr(self.config.model, "adapter_config", None)
            head = getattr(self.config.model, "head_config", None)
            per_rank = max(
                1, get_per_rank_batch_size(self.config.train_dataloader.batch_size)
            )

            # Calibrate on a SMALL microbatch and scale, rather than allocating
            # a second full-size one. The counter's forward+backward runs on top
            # of the already-resident model, gradients and optimizer state, so at
            # a large per-rank batch it OOMs and MFU is silently disabled for the
            # whole run -- exactly when MFU matters most (the 500m/1b MoE arms
            # died this way at per-rank 8-16). FLOPs are linear in batch for every
            # op here (GEMMs, and SDPA's B*S^2*D), so measuring at ``calib_bs``
            # and scaling by ``per_rank / calib_bs`` is exact, not an estimate.
            calib_bs = max(
                1,
                min(
                    per_rank,
                    int(os.environ.get("NESTOR_MFU_CALIB_BATCH", "2")),
                ),
            )
            calib_scale = per_rank / calib_bs
            # A synthetic microbatch of token ids at the adapter's context
            # length; the causal LM loss applies the next-token shift itself.
            vocab = int(
                getattr(adapter, "vocab_size", None)
                or getattr(head, "vocab_size", None)
                or 50257
            )
            tokens_per_seq = int(getattr(adapter, "max_length", 1024) or 1024)
            x = torch.randint(0, vocab, (calib_bs, tokens_per_seq), device=self.device)
            y = x

            attn_shapes: list[tuple[int, int, int]] = []

            def _attn_hook(_module: Any, inputs: Any, _output: Any) -> None:
                if inputs and hasattr(inputs[0], "shape") and inputs[0].dim() == 3:
                    b, s, d = inputs[0].shape
                    attn_shapes.append((int(b), int(s), int(d)))

            # Calibrate on the UNWRAPPED (eager) model. FlopCounterMode is a
            # TorchDispatchMode and forces eager dispatch anyway (it can't see
            # through inductor's fused triton kernels), so calling the compiled
            # wrapper here measured the same FLOPs BUT poisoned Dynamo: this is
            # the model's first invocation, and tracing under a dispatch mode +
            # forward hooks + flex-disabled made Dynamo cache an eager fallback,
            # so torch.compile never produced triton kernels during training
            # (the long-standing "compile gives ~0 benefit" bug). Using the
            # eager base model leaves the compiled wrapper's first real trace
            # for the training loop, in a clean context.
            from nestor.train.checkpoint import _unwrap_compiled

            calib_model = _unwrap_compiled(self.model)

            hooks = []
            for module in calib_model.modules():
                name = type(module).__name__.lower()
                if "attention" in name or name.endswith("attn"):
                    hooks.append(module.register_forward_hook(_attn_hook))

            # Force the SDPA attention path during measurement. flex_attention
            # is a higher-order op that FlopCounterMode cannot dispatch (it
            # raises); SDPA is mathematically identical and has a flop formula,
            # so the counter then captures the whole model (attention + experts).
            flex_module = None
            flex_saved = None
            try:
                import nestor.backbones.nestor as flex_module  # type: ignore

                flex_saved = flex_module._FLEX_ATTENTION_AVAILABLE
                flex_module._FLEX_ATTENTION_AVAILABLE = False
            except Exception:  # noqa: BLE001
                flex_module = None

            # Count the grouped-GEMM expert path AS IT RUNS, by registering a
            # flop formula for torch._grouped_mm instead of swapping in the
            # sparse fallback. The old workaround disabled the grouped path so
            # the same math would route through counted aten::mm calls, but the
            # two paths do not execute the same amount of work: on
            # a 1b column arm it reported 1.855e14 flops/microbatch
            # against a dense-6ND upper bound of 4.991e13 -- 3.7x more than an
            # all-parameters dense model, which a sparsely-activated MoE cannot
            # do. MFU was inflated by roughly that factor and moved with the
            # batch shape, which made it useless for comparing configs.
            from nestor.train.flop_formulas import register_extra_flop_formulas

            register_extra_flop_formulas()

            calib_model.train()
            self.optimizer.zero_grad(set_to_none=True)
            counter = FlopCounterMode(display=False)
            try:
                with counter:
                    with autocast("cuda", dtype=self._amp_dtype, enabled=self.config.use_amp):
                        output: ModelOutput = calib_model(x)
                        loss = self.loss_fn(output.pred, y, output.aux).total
                    loss.backward()
            finally:
                if flex_module is not None and flex_saved is not None:
                    flex_module._FLEX_ATTENTION_AVAILABLE = flex_saved
            gemm_flops = float(counter.get_total_flops())
            self.optimizer.zero_grad(set_to_none=True)
            for hook in hooks:
                hook.remove()
            # Drop any Dynamo state from calibration so the compiled model's
            # first trace happens fresh in the training loop.
            torch._dynamo.reset()

            attn_counted = False
            try:
                for per_mod in counter.get_flop_counts().values():
                    if any(
                        "scaled_dot_product" in str(op).lower()
                        or "attention" in str(op).lower()
                        for op in per_mod
                    ):
                        attn_counted = True
                        break
            except Exception:  # noqa: BLE001
                pass

            attn_flops = 0.0
            if not attn_counted:
                # fwd+bwd of QK^T + AV per attention layer ~= 12 * B * S^2 * D
                attn_flops = sum(
                    12.0 * b * s * s * d for (b, s, d) in attn_shapes
                )

            # Scale the small-batch measurement up to the real microbatch.
            total = (gemm_flops + attn_flops) * calib_scale
            nparams = sum(p.numel() for p in self.model.parameters())
            dense_ub = 6.0 * nparams * per_rank * tokens_per_seq
            self.logger.info(
                "[MFU] flops/microbatch: gemm=%.3e attn=%.3e (attn_layers=%d "
                "counted=%s) total=%.3e | dense 6ND upper-bound=%.3e | "
                "per_rank_bs=%d (measured at bs=%d, scaled x%.1f)",
                gemm_flops, attn_flops, len(attn_shapes), attn_counted,
                total, dense_ub, per_rank, calib_bs, calib_scale,
            )
            return total
        except Exception as exc:  # noqa: BLE001  — never break training
            self.logger.warning(
                "[MFU] flop calibration failed (%s); MFU metrics disabled", exc
            )
            return 0.0

    @staticmethod
    def _add_mfu_metrics(
        metrics: dict[str, float],
        *,
        flops_per_step: float,
        peak_flops: float,
        global_batch: int,
        now: float,
        last_dt: float,
        steps: int,
        t_steady: float | None,
        process_start: float,
    ) -> None:
        """Add throughput + MFU (instant, run-avg incl/excl init) to metrics.

        Guarded so a monitoring error can never interrupt training.
        """

        try:
            if last_dt > 0:
                metrics["sec_per_step"] = last_dt
                metrics["samples_per_sec"] = global_batch / last_dt
            if torch.cuda.is_available():
                # Peak since the last log window, so a run shows how much
                # headroom it actually has rather than only whether it OOMed.
                # Reserved is what the allocator holds from the driver;
                # allocated is what tensors are using -- a large gap is
                # fragmentation, a small one means the batch is genuinely
                # near the ceiling.
                metrics["mem_peak_alloc_gib"] = (
                    torch.cuda.max_memory_allocated() / 2 ** 30
                )
                metrics["mem_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved() / 2 ** 30
                )
                torch.cuda.reset_peak_memory_stats()
            if flops_per_step <= 0 or last_dt <= 0:
                return
            # per-rank achieved FLOP/s vs per-GPU peak
            metrics["tflops_per_gpu"] = flops_per_step / last_dt / 1e12
            # (1) instant: this step's achieved FLOP/s / peak
            metrics["mfu_instant"] = flops_per_step / last_dt / peak_flops
            # (2.1) running avg INCLUDING init (compile/load/prep): since proc start
            incl_dt = now - process_start
            if incl_dt > 0:
                metrics["mfu_runavg_incl_init"] = (
                    flops_per_step * steps / incl_dt / peak_flops
                )
            # (2.2) running avg EXCLUDING init: since after the first step
            if t_steady is not None and steps > 1:
                excl_dt = now - t_steady
                if excl_dt > 0:
                    metrics["mfu_runavg_excl_init"] = (
                        flops_per_step * (steps - 1) / excl_dt / peak_flops
                    )
        except Exception:  # noqa: BLE001
            pass

    def _grad_sync_context(self, last_in_window: bool) -> Any:
        """Context for one micro-batch's backward: sync grads or defer them.

        See :func:`grad_sync_context`; windows are counted in micro-batches
        across epoch boundaries (``accum_steps`` is only reset by an optimizer
        step), so the last micro-batch of every window is always synced.

        Args:
            last_in_window (bool): Whether this micro-batch completes the
                accumulation window (its backward must reduce).

        Returns:
            Any: A context manager wrapping the forward/backward.
        """

        return grad_sync_context(self.model, self._ddp_no_sync, last_in_window)

    def _train_step(self, batch: object) -> tuple[LossOutput, dict[str, Any]]:
        """Execute one training microbatch.

        Args:
            batch (object): Supervised batch object.

        Returns:
            tuple[LossOutput, dict[str, Any]]: Loss outputs and model aux data.
        """

        inputs, targets = split_supervised_batch(batch)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # The offload context wraps only the FORWARD: it installs the hook that
        # decides where each saved activation lives, and the matching unpack
        # fires during the backward regardless of scope. It is a no-op unless
        # NESTOR_ACTIVATION_OFFLOAD=1.
        with activation_offload(), autocast(
            "cuda", dtype=self._amp_dtype, enabled=self.config.use_amp
        ):
            output: ModelOutput = self.model(inputs)
            loss_output = self.loss_fn(output.pred, targets, output.aux)
            scaled_loss = loss_output.total / self.config.grad_accum_steps

        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss_output, output.aux

    def _update_weights(self, model_aux_batches: list[dict[str, Any]]) -> None:
        """Update model weights after a grad-accumulation window.

        Args:
            model_aux_batches (list[dict[str, Any]]): Aux outputs for the
                current accumulation window.
        """

        # Parameters DDP was told to ignore (the fused head, whose projection
        # runs inside the loss) never enter the reducer: average their
        # accumulated local gradients across ranks here, once per optimizer
        # step, before anything reads them.
        from nestor.utils.distributed import reduce_ignored_grads

        reduce_ignored_grads(self.model)
        # bf16-master mode (NESTOR_BF16_MASTER=1): the Muon matrices' fp32
        # gradient accumulators are reduced to their owner ranks here (the
        # reducer ignores those parameters) and clipped shard-aware below.
        from nestor.scaling.bf16_master import (
            bf16_master_params,
            clip_grad_norm_ as _bf16_clip,
            reduce_to_owners,
        )
        _bf16_mode = bool(bf16_master_params(self.model))
        if _bf16_mode:
            reduce_to_owners(self.optimizer)

        if self.config.max_grad_norm is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            # Ensure every parameter has a gradient so that FSDP2's AllReduce
            # inside clip_grad_norm_ sees the same parameter count on all ranks.
            # Sparse expert routing can leave unused experts with no gradient.
            # (DDP and single-process runs have no such collective — skip.)
            if self._fill_missing_grads:
                for p in self.model.parameters():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
            if _bf16_mode:
                _gn = _bf16_clip(self.model, self.optimizer, self.config.max_grad_norm)
            else:
                _gn = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            # NESTOR_DEBUG_GRAD=1: dump grad health for the first few steps to
            # diagnose frozen-loss runs (grad norm pre-clip, fraction of params
            # with non-finite / zero grad, and the actual per-group LRs).
            import os as _os
            if (
                _os.environ.get("NESTOR_DEBUG_GRAD") == "1"
                and is_main_process()
                and self.global_step < 5
            ):
                _ps = list(self.model.parameters())
                _nz = sum(
                    1 for p in _ps
                    if p.grad is not None and p.grad.abs().sum().item() == 0.0
                )
                _nan = sum(
                    1 for p in _ps
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                )
                _lrs = [g["lr"] for g in self.optimizer.param_groups]
                self.logger.info(
                    "[DEBUG_GRAD] step=%d grad_norm=%.4e finite=%s "
                    "zero-grad params=%d/%d nan-grad params=%d | "
                    "lrs(min/max)=%.3e/%.3e",
                    self.global_step, float(_gn),
                    bool(torch.isfinite(_gn)), _nz, len(_ps), _nan,
                    min(_lrs), max(_lrs),
                )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self._update_ema()

        # Advance the LR schedule and apply detached router-bias updates after
        # every optimizer step. (These were misplaced into _swap_out_ema by the
        # EMA commit b6967a9, behind its `_ema is None` early-return, which
        # froze the LR at 0 for every non-EMA run launched after it.)
        if self.scheduler is not None:
            self.scheduler.step()

        if self._needs_detached_updates:
            merged_aux = self._merge_model_aux_batches(model_aux_batches)
            self._update_detached_parameters(merged_aux)

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update the EMA shadow toward the current weights (post optimizer)."""

        if self._ema is None:
            return
        from nestor.train.checkpoint import _unwrap_compiled

        base = _unwrap_compiled(self.model)
        decay = cast(float, self._ema_decay)
        for name, param in base.named_parameters():
            self._ema[name].lerp_(param.detach().float(), 1.0 - decay)

    @torch.no_grad()
    def _swap_in_ema(self) -> None:
        """Load EMA weights into the live model, backing up the originals."""

        if self._ema is None:
            return
        from nestor.train.checkpoint import _unwrap_compiled

        base = _unwrap_compiled(self.model)
        self._ema_backup = {
            name: param.detach().clone()
            for name, param in base.named_parameters()
        }
        for name, param in base.named_parameters():
            param.data.copy_(self._ema[name].to(param.dtype))

    @torch.no_grad()
    def _swap_out_ema(self) -> None:
        """Restore the pre-EMA weights saved by :meth:`_swap_in_ema`."""

        if self._ema is None or self._ema_backup is None:
            return
        from nestor.train.checkpoint import _unwrap_compiled

        base = _unwrap_compiled(self.model)
        for name, param in base.named_parameters():
            param.data.copy_(self._ema_backup[name])
        self._ema_backup = None

    @staticmethod
    def _build_train_metrics(
        accum_loss: float,
        accum_steps: int,
        accum_component_sums: dict[str, float],
    ) -> dict[str, float]:
        """Build averaged training metrics.

        Args:
            accum_loss (float): Sum of total losses.
            accum_steps (int): Number of accumulated microbatches.
            accum_component_sums (dict[str, float]): Sum of component losses.

        Returns:
            dict[str, float]: Averaged metrics.
        """

        metrics = {"loss": accum_loss / accum_steps}
        metrics.update(
            {name: value / accum_steps for name, value in accum_component_sums.items()}
        )
        return metrics

    @staticmethod
    def _merge_model_aux_batches(
        model_aux_batches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge model auxiliary outputs across a grad-accumulation window.

        Args:
            model_aux_batches (list[dict[str, Any]]): Per-microbatch aux outputs.

        Returns:
            dict[str, Any]: Merged aux dictionary.
        """

        if not model_aux_batches:
            return {}
        if len(model_aux_batches) == 1:
            return model_aux_batches[0]

        def merge_router_aux(
            router_aux_batches: list[dict[str, Any]],
        ) -> dict[str, Any]:
            merged_router_aux = dict(router_aux_batches[-1])
            if all("topk_indices" in batch_aux for batch_aux in router_aux_batches):
                merged_router_aux["topk_indices"] = torch.cat(
                    [batch_aux["topk_indices"] for batch_aux in router_aux_batches],
                    dim=0,
                )
            if all("valid_token_mask" in batch_aux for batch_aux in router_aux_batches):
                merged_router_aux["valid_token_mask"] = torch.cat(
                    [batch_aux["valid_token_mask"] for batch_aux in router_aux_batches],
                    dim=0,
                )
            return merged_router_aux

        merged_aux = dict(model_aux_batches[-1])
        if "blocks" not in merged_aux:
            return merged_aux

        num_blocks = len(merged_aux["blocks"])
        merged_blocks: list[dict[str, Any]] = []
        for block_idx in range(num_blocks):
            block_aux_batches = [
                aux["blocks"][block_idx]
                for aux in model_aux_batches
                if "blocks" in aux and len(aux["blocks"]) > block_idx
            ]
            if not block_aux_batches:
                merged_blocks.append({})
                continue

            merged_block_aux = dict(block_aux_batches[-1])
            if all(
                isinstance(batch_aux.get("group_router"), dict)
                for batch_aux in block_aux_batches
            ):
                merged_block_aux["group_router"] = merge_router_aux(
                    [batch_aux["group_router"] for batch_aux in block_aux_batches]
                )

            layer_count = len(merged_block_aux.get("layers", []))
            merged_layers: list[dict[str, Any]] = []
            for layer_idx in range(layer_count):
                layer_aux_batches = [
                    batch_aux["layers"][layer_idx]
                    for batch_aux in block_aux_batches
                    if "layers" in batch_aux and len(batch_aux["layers"]) > layer_idx
                ]
                if not layer_aux_batches:
                    merged_layers.append({})
                    continue
                merged_layer_aux = dict(layer_aux_batches[-1])
                if all(
                    isinstance(layer_aux.get("expert_router"), dict)
                    for layer_aux in layer_aux_batches
                ):
                    merged_layer_aux["expert_router"] = merge_router_aux(
                        [layer_aux["expert_router"] for layer_aux in layer_aux_batches]
                    )
                # Expert-pool layers additionally carry one router aux per
                # private pool; merge each pool position across microbatches
                # so loss-free balancing sees the whole accumulation window.
                if all(
                    isinstance(layer_aux.get("expert_pool_private"), list)
                    for layer_aux in layer_aux_batches
                ):
                    private_lists = [
                        layer_aux["expert_pool_private"]
                        for layer_aux in layer_aux_batches
                    ]
                    num_private = min(len(aux) for aux in private_lists)
                    merged_layer_aux["expert_pool_private"] = [
                        merge_router_aux(
                            [aux[router_idx] for aux in private_lists]
                        )
                        for router_idx in range(num_private)
                    ]
                merged_layers.append(merged_layer_aux)

            if merged_layers:
                merged_block_aux["layers"] = merged_layers
            merged_blocks.append(merged_block_aux)

        merged_aux["blocks"] = merged_blocks
        return merged_aux

    def _enter_train_mode(self) -> None:
        """Switch the model to train mode (an eval-suite round trip calls this)."""

        self.model.train()

    def _update_detached_parameters(self, model_aux: dict[str, Any]) -> None:
        """Update detached model state after the optimizer step.

        Args:
            model_aux (dict[str, Any]): Merged auxiliary outputs.
        """

        for module in self.model.modules():
            if isinstance(module, Backbone):
                module.update_detached_parameters(model_aux)

    def _run_eval_suite(self, tag: str = "") -> dict[str, dict[str, Any]]:
        """Run the evaluation suite.

        Args:
            tag (str): Metric/file suffix (e.g. ``"_ema"``) so the live-model and
                EMA evals log under distinct keys without overwriting each other.

        Returns:
            dict[str, dict[str, Any]]: Results from all configured evaluations.
        """

        assert self.eval_suite is not None
        self.logger.info(f"Running eval suite{tag} at step {self.global_step}")
        results = self.eval_suite.run(step=self.global_step)
        for eval_name, eval_metrics in results.items():
            log_metrics(
                self.logger,
                eval_metrics,
                self.global_step,
                prefix=f"eval/{eval_name}{tag}/",
            )
        self._save_eval_results(results, tag=tag)
        return results

    def _save_eval_results(
        self, results: dict[str, dict[str, Any]], tag: str = ""
    ) -> None:
        """Save evaluation results to disk.

        Args:
            results (dict[str, dict[str, Any]]): Evaluation results.
            tag (str): Filename suffix to separate live/EMA eval dumps.
        """

        if not is_main_process():
            return

        eval_dir = self.run_dir / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"eval_step_{self.global_step:08d}{tag}.json"
        output = {
            "step": self.global_step,
            "epoch": self.epoch,
            "results": results,
        }
        with open(eval_path, "w") as file:
            json.dump(output, file, indent=2)
        self.logger.info(f"Eval results saved to {eval_path}")

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save a model checkpoint.

        Args:
            final (bool): Whether this is the final checkpoint.
        """

        checkpoint_dir = self.run_dir / (self.config.checkpoint_dir or "checkpoints")
        if is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        suffix = "final" if final else f"{self.global_step:08d}"
        model_path = checkpoint_dir / f"model_{suffix}.pt"
        save_checkpoint(
            model_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.epoch,
            extra={"ema": self._ema} if self._ema is not None else None,
        )
        if is_main_process():
            self.logger.info(f"Model checkpoint saved: {model_path}")

    def _submit_eval_jobs(self) -> None:
        """Submit SLURM eval jobs for the current checkpoint."""

        if not is_main_process():
            return
        if not self.config.eval_submit_scripts:
            return

        import shutil

        sbatch = shutil.which("sbatch")
        if sbatch is None:
            self.logger.warning("sbatch not found on PATH; skipping eval job submission")
            return

        checkpoint_dir = self.run_dir / (self.config.checkpoint_dir or "checkpoints")
        checkpoint_path = checkpoint_dir / f"model_{self.global_step:08d}.pt"

        for script_path, frequency in self.config.eval_submit_scripts:
            if self.global_step % frequency != 0:
                continue
            script = Path(script_path)
            if not script.exists():
                self.logger.warning(f"Eval script not found: {script}")
                continue
            try:
                result = subprocess.run(
                    [sbatch, str(script), str(checkpoint_path)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self.logger.info(
                        f"Submitted eval job: {script.name} "
                        f"(step {self.global_step}) -> {result.stdout.strip()}"
                    )
                else:
                    self.logger.warning(
                        f"Failed to submit {script.name}: {result.stderr.strip()}"
                    )
            except Exception as e:
                self.logger.warning(f"Error submitting {script.name}: {e}")

    @classmethod
    def from_config(
        cls,
        model: "Model | nn.Module",
        loss_fn: Loss,
        train_loader: DataLoader,
        eval_suite: "EvalSuite | None",
        job_config: TrainJobConfig,
        run_dir: Path,
        device: torch.device,
        logger: logging.Logger,
    ) -> "Trainer":
        """Create a trainer from configuration.

        Args:
            model (Model | nn.Module): Possibly wrapped top-level model.
            loss_fn (Loss): Loss function.
            train_loader (DataLoader): Training data loader.
            eval_suite (EvalSuite | None): Optional evaluation suite.
            job_config (TrainJobConfig): Training job configuration.
            run_dir (Path): Run output directory.
            device (torch.device): Device for this rank.
            logger (logging.Logger): Logger instance.

        Returns:
            Trainer: Configured trainer.
        """

        return cls(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            eval_suite=eval_suite,
            config=job_config,
            run_dir=run_dir,
            device=device,
            logger=logger,
        )
