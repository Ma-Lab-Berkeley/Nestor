"""Training pipeline."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from nestor.configs.data import DataLoaderConfig
from nestor.configs.experiment import ExperimentConfig
from nestor.configs.jobs import JobConfig, TrainJobConfig
from nestor.models import Model
from nestor.pipelines.base import BasePipeline
from nestor.utils.distributed import wrap_model_for_distributed

if TYPE_CHECKING:
    from nestor.eval.suite import EvalSuite
    from nestor.train.trainer import Trainer


def find_latest_checkpoint(
    experiment_dir: "Path",
    *,
    exclude_run_dir: "Path | None" = None,
    checkpoint_dir: "Path | str" = "checkpoints",
) -> "tuple[Path | None, int]":
    """Newest ``model_<step>.pt`` among the previous runs of an experiment.

    Args:
        experiment_dir (Path): ``<output_root>/<experiment name>``.
        exclude_run_dir (Path | None): The current run's directory, skipped.
        checkpoint_dir (Path | str): Checkpoint sub-directory of each run.

    Returns:
        tuple[Path | None, int]: The checkpoint path and its step, or
        ``(None, -1)`` when there is none.
    """

    from pathlib import Path

    best_step = -1
    best_path: Path | None = None
    if not Path(experiment_dir).exists():
        return None, best_step
    for run_path in Path(experiment_dir).iterdir():
        if not run_path.is_dir() or run_path == exclude_run_dir:
            continue
        ckpt_dir = run_path / checkpoint_dir
        if not ckpt_dir.exists():
            continue
        for ckpt in ckpt_dir.glob("model_[0-9]*.pt"):
            try:
                step = int(ckpt.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if step > best_step:
                best_step = step
                best_path = ckpt
    return best_path, best_step


def resolve_resume_checkpoint(
    *,
    explicit: "Path | None",
    explicit_strict: bool,
    latest_own: "Path | None",
) -> "tuple[Path | None, bool, str]":
    """Precedence between an explicit checkpoint and the experiment's own latest.

    Args:
        explicit (Path | None): ``TrainJobConfig.resume_model_checkpoint``.
        explicit_strict (bool): ``TrainJobConfig.resume_strict`` -- ``True`` is a
            real resume (optimizer / scheduler / step restored), ``False`` a
            weights-only initialisation from another run.
        latest_own (Path | None): Newest checkpoint written by this experiment.

    Returns:
        tuple[Path | None, bool, str]: ``(checkpoint, strict, mode)`` with mode
        ``"explicit"`` (strict explicit resume), ``"auto"`` (own latest,
        strict), ``"init"`` (non-strict initialisation, fresh run) or
        ``"none"``.
    """

    if explicit is not None and explicit_strict:
        return explicit, True, "explicit"
    if latest_own is not None:
        return latest_own, True, "auto"
    if explicit is not None:
        return explicit, False, "init"
    return None, True, "none"


class TrainPipeline(BasePipeline):
    """Pipeline for training jobs."""

    job_config: TrainJobConfig

    def __init__(
        self,
        job_config: TrainJobConfig,
        experiment_config: ExperimentConfig,
        device: torch.device,
    ):
        """Initialize the training pipeline.

        Args:
            job_config (TrainJobConfig): Training job configuration.
            experiment_config (ExperimentConfig): Experiment configuration.
            device (torch.device): Device for this rank.
        """

        super().__init__(job_config, experiment_config, device)
        self.model: Model | torch.nn.Module | None = None
        self.model_num_parameters: int | None = None
        self.train_loader: DataLoader | None = None
        self.eval_suite: "EvalSuite | None" = None
        self.trainer: "Trainer | None" = None
        self.loss_fn = None
        # wandb run id to continue when resuming from a checkpoint (Option A:
        # ties the wandb run to the checkpoint lineage). None -> fresh run.
        self._resume_wandb_id: str | None = None

    def setup(self) -> None:
        """Set up the training pipeline."""

        super().setup()

        self._auto_resume()
        self._init_wandb(
            project=self.job_config.wandb_project,
            entity=self.job_config.wandb_entity,
            run_name=self.job_config.wandb_run_name,
            tags=self.job_config.wandb_tags,
        )
        self._build_data()
        self._build_model()
        self._build_eval_suite()
        self._build_trainer()

        self.logger.info("Training pipeline setup complete")
        if self.model_num_parameters is not None:
            self.logger.info(
                f"  Model parameters: {self.model_num_parameters:,}"
            )
        if self.train_loader is not None:
            try:
                self.logger.info(f"  Training batches: {len(self.train_loader)}")
            except TypeError:
                pass
        if self.eval_suite is not None:
            self.logger.info(
                f"  Eval suite: {len(self.eval_suite.evals)} evaluations"
            )

    def _auto_resume(self) -> None:
        """Pick the checkpoint to start from: explicit strict resume, else the
        latest checkpoint of this experiment, else a non-strict initialisation."""

        experiment_dir = (
            self.experiment_config.output_root / self.experiment_config.name
        )
        latest, best_step = find_latest_checkpoint(
            experiment_dir,
            exclude_run_dir=self.run_dir,
            checkpoint_dir=self.job_config.checkpoint_dir or "checkpoints",
        )
        checkpoint, strict, mode = resolve_resume_checkpoint(
            explicit=self.job_config.resume_model_checkpoint,
            explicit_strict=self.job_config.resume_strict,
            latest_own=latest,
        )
        if checkpoint is None:
            return
        self.job_config.resume_model_checkpoint = checkpoint
        self.job_config.resume_strict = strict
        if mode == "init":
            # A NON-strict explicit checkpoint is an initialisation (the
            # long-context extension stage starting from the short-context
            # weights, DCLM_INIT_CHECKPOINT): a fresh run, no wandb run to
            # continue. Once this experiment has checkpoints of its own, a
            # retry resumes those instead (mode "auto").
            self.logger.info(
                f"Initialising weights from {checkpoint} (non-strict); optimizer, "
                "schedule and step start fresh"
            )
            return
        if mode == "auto":
            self.logger.info(
                f"Auto-resume: found checkpoint at step {best_step} from {checkpoint}"
            )
        # Continue the same wandb run that produced this checkpoint.
        self._resume_wandb_id = self._read_wandb_run_id(checkpoint)

    def _read_wandb_run_id(self, checkpoint_path: "Path | str") -> str | None:
        """Read the wandb run id persisted by the run that wrote a checkpoint.

        The id sidecar lives at ``<run_dir>/wandb_run_id.txt`` and a checkpoint
        is at ``<run_dir>/<checkpoint_dir>/model_*.pt``, so the run dir is the
        checkpoint's grandparent. Returns None if no sidecar is found.
        """

        from pathlib import Path

        try:
            run_dir = Path(checkpoint_path).resolve().parent.parent
            sidecar = run_dir / "wandb_run_id.txt"
            if sidecar.exists():
                run_id = sidecar.read_text().strip()
                if run_id:
                    self.logger.info(
                        f"Auto-resume: will continue wandb run {run_id}"
                    )
                    return run_id
        except Exception as exc:  # noqa: BLE001  — never block resume
            self.logger.warning(f"Could not read wandb run id sidecar: {exc}")
        return None

    def _build_data(self) -> None:
        """Build the training dataset and dataloader."""

        from nestor.data.datasets import Dataset

        train_dataset = Dataset.from_config(self.job_config.train_dataset)
        self.train_loader = self._build_dataloader(
            train_dataset,
            self.job_config.train_dataloader,
            is_train=True,
        )

    def _build_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        config: DataLoaderConfig,
        is_train: bool,
    ) -> DataLoader:
        """Build a dataloader from configuration.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to load.
            config: Dataloader configuration.
            is_train (bool): Whether the loader is used for training.

        Returns:
            DataLoader: Configured data loader.
        """

        from torch.utils.data import IterableDataset
        from torch.utils.data.distributed import DistributedSampler

        from nestor.utils.distributed import (
            get_per_rank_batch_size,
            get_rank,
            get_world_size,
            is_distributed,
        )

        num_workers = config.num_workers

        if isinstance(dataset, IterableDataset):
            return DataLoader(
                dataset,
                batch_size=get_per_rank_batch_size(config.batch_size),
                num_workers=num_workers,
                pin_memory=config.pin_memory,
                drop_last=config.drop_last,
                prefetch_factor=config.prefetch_factor
                if config.num_workers > 0
                else None,
            )

        sampler = None
        shuffle = config.shuffle if is_train else False

        if is_distributed():
            sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=get_per_rank_batch_size(config.batch_size),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            sampler=sampler,
            prefetch_factor=config.prefetch_factor
            if num_workers > 0
            else None,
            persistent_workers=config.persistent_workers
            and num_workers > 0,
        )

    def _build_model(self) -> None:
        """Build the top-level model and loss function."""

        from nestor.train.losses import Loss

        base_model = Model.from_config(self.job_config.model)
        self.model_num_parameters = sum(
            parameter.numel() for parameter in base_model.parameters()
        )
        base_model = base_model.to(self.device)
        # NESTOR_BF16_MASTER=1: Muon-tagged matrices become bf16
        # leaves with fp32 masters/accumulators BEFORE any wrapper (DDP's
        # reducer must ignore them) and before compile.
        from nestor.scaling.bf16_master import (
            bf16_master_enabled,
            convert_muon_params_to_bf16,
        )
        if bf16_master_enabled():
            converted = convert_muon_params_to_bf16(base_model)
            base_model._bf16_master_param_names = converted
            self.logger.info(
                "[PERF] bf16-master mode: %d Muon parameters stored as bf16 "
                "leaves (fp32 masters sharded by the optimizer)", len(converted),
            )
        # NESTOR_FREEZE_ROUTERS=1 (the domain-adaptation recipe of the sparse
        # models): freeze the router score networks BEFORE the wrap -- DDP
        # registers only the parameters that require gradients at wrap time.
        from nestor.train.frozen_routers import freeze_routers, freeze_routers_enabled

        if freeze_routers_enabled():
            report = freeze_routers(base_model)
            self.logger.info("[frozen routers] %s", report.describe())
            self.logger.info(
                "[frozen routers] first frozen: %s ... last: %s",
                report.frozen_names[0], report.frozen_names[-1],
            )
        # NESTOR_DISABLE_FSDP=1 skips every distributed wrapper (single-process
        # debugging; FSDP2 also blocks inductor CUDA graphs).
        if os.environ.get("NESTOR_DISABLE_FSDP") == "1":
            self.logger.info("[PERF] distributed wrapper disabled (NESTOR_DISABLE_FSDP=1)")
            self.model = base_model
        else:
            self.model = wrap_model_for_distributed(
                base_model,
                self.job_config.distributed,
                device=self.device,
            )
        # NESTOR_PREWARM_CAUSAL_SEQ=<S>: pre-materialize every block's lazy
        # causal-mask cache at sequence length S BEFORE any compiled forward
        # runs. Required for NESTOR_COMPILE_MODE=reduce-overhead: the lazy
        # fill inside _compose_attn_mask allocates the cached tril in CUDA-graph
        # pool memory, and the module retaining it across replays fails with
        # "accessing tensor output of CUDAGraphs that has been overwritten"
        # (observed in production). Prewarming here allocates it as ordinary
        # memory that replays only read. Harmless (16 MiB per block at S=4096)
        # under other compile modes.
        prewarm_seq = int(os.environ.get("NESTOR_PREWARM_CAUSAL_SEQ", "0"))
        if prewarm_seq > 0:
            prewarmed = 0
            for module in self.model.modules():
                cache = getattr(module, "_cached_causal_mask", None)
                if isinstance(cache, torch.Tensor) and getattr(
                    module, "causal", False
                ):
                    module._cached_causal_mask = torch.tril(
                        torch.ones(
                            (prewarm_seq, prewarm_seq),
                            dtype=torch.bool,
                            device=self.device,
                        )
                    )
                    prewarmed += 1
            self.logger.info(
                "[PERF] prewarmed %d causal-mask caches at seq %d",
                prewarmed, prewarm_seq,
            )
        # Optional torch.compile (env-gated; reversible). Fuses the many tiny
        # per-layer kernels (norms, activations, mask build, residual) that
        # make a small model launch-bound in eager mode. NESTOR_COMPILE_MODE
        # picks the inductor mode; fullgraph=False so data-dependent ops
        # (router nonzero / topk) graph-break instead of erroring.
        if os.environ.get("NESTOR_COMPILE") == "1":
            # NESTOR_NO_PERSISTENT_REDUCTIONS=1: force looped (non-persistent)
            # reductions. Persistent reductions hold the whole reduction in
            # shared memory; the AttnRes depth-attention kernel (with drop-path
            # fused in) overflows the H200 48KB/SM limit (243744 > 232448 B),
            # failing compile with "No valid triton configs / OutOfMemory". The
            # env TORCHINDUCTOR_PERSISTENT_REDUCTIONS is NOT honored by inductor;
            # setting the config object directly is. Looped reductions use
            # O(block) shared memory, so they fit. Only needed for residual_mode=
            # attn_res; harmless (small perf cost) otherwise.
            if os.environ.get("NESTOR_NO_PERSISTENT_REDUCTIONS") == "1":
                import torch._inductor.config as _inductor_config

                _inductor_config.triton.persistent_reductions = False
                self.logger.info(
                    "[PERF] inductor persistent_reductions disabled "
                    "(AttnRes shared-mem OOM workaround)"
                )
            # NESTOR_RECOMPILE_LIMIT: dynamo specializes each compiled frame
            # on its input structure, and the column-mode depth read is called
            # with a DIFFERENT number of sources in every block (a column-read
            # arm sees 4 -> 9 -> 18 -> 20). Each length is a separate
            # variant, so the stock limit of 8 is exhausted mid-model and the
            # remaining frames silently fall back to eager -- measured at 16.6
            # s/step vs 5.0 for the same arm without the aggregate. Raising the
            # limit costs compile time once; hitting it costs throughput for
            # the whole run.
            _recompile_limit = int(
                os.environ.get("NESTOR_RECOMPILE_LIMIT", "64")
            )
            import torch._dynamo.config as _dynamo_config

            for _attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(_dynamo_config, _attr):
                    setattr(_dynamo_config, _attr, _recompile_limit)
            self.logger.info(
                "[PERF] dynamo recompile limit = %d", _recompile_limit
            )
            mode = os.environ.get("NESTOR_COMPILE_MODE") or "default"
            self.logger.info("[PERF] torch.compile(mode=%s) enabled", mode)
            self.model = torch.compile(self.model, mode=mode)
        self.loss_fn = Loss.from_config(self.job_config.loss)

    def _build_eval_suite(self) -> None:
        """Build the evaluation suite if configured."""

        from nestor.eval.suite import EvalSuite

        if self.job_config.eval_suite is None:
            return

        assert self.model is not None
        # Run evals on the UNCOMPILED module. Through the compiled wrapper,
        # eval's different batch size + training=False force a recompile with
        # dynamic shapes, and torch 2.10's meta for aten._grouped_mm fails
        # under symbolic shapes — this killed every MoE sweep job at its first
        # eval (step 10000), with the surviving ranks hanging in allreduce
        # until the NCCL watchdog fired. Eval is rare, so eager speed is fine;
        # NESTOR_COMPILED_EVAL=1 restores the old behavior.
        eval_model = self.model
        if os.environ.get("NESTOR_COMPILED_EVAL") != "1":
            from nestor.train.checkpoint import _unwrap_compiled

            eval_model = _unwrap_compiled(self.model)
        self.eval_suite = EvalSuite.from_config(
            config=self.job_config.eval_suite,
            model=eval_model,
            device=self.device,
            logger=self.logger,
        )

    def _build_trainer(self) -> None:
        """Build the trainer."""

        from nestor.train.trainer import Trainer

        assert self.model is not None
        assert self.train_loader is not None
        assert self.loss_fn is not None

        self.trainer = Trainer.from_config(
            model=self.model,
            loss_fn=self.loss_fn,
            train_loader=self.train_loader,
            eval_suite=self.eval_suite,
            job_config=self.job_config,
            run_dir=self.run_dir,
            device=self.device,
            logger=self.logger,
        )

    def run(self) -> None:
        """Run the training pipeline."""

        self.setup()
        try:
            assert self.trainer is not None
            self.trainer.train()
        finally:
            self._finish_wandb()

        self.logger.info("Training complete!")

    @classmethod
    def from_config(
        cls,
        job_config: JobConfig,
        experiment_config: ExperimentConfig,
        device: str | None = None,
    ) -> "TrainPipeline":
        """Create a training pipeline from configuration.

        Args:
            job_config (JobConfig): Training job configuration.
            experiment_config (ExperimentConfig): Experiment configuration.
            device (str | None): Optional device string.

        Returns:
            TrainPipeline: Initialized pipeline.
        """

        assert isinstance(job_config, TrainJobConfig)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            job_config=job_config,
            experiment_config=experiment_config,
            device=torch.device(device),
        )
