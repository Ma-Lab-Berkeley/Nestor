"""The long-context extension ladder: 4k -> 8k / 16k / 32k / 64k by continued pretraining.

Every pretrained model of the scaling figure is extended from its 4k
checkpoint to each target length ``S`` in a separate run on DCLM-baseline
documents of at least ``S`` tokens (no run continues from a model extended
to a shorter length):

* **Equal tokens** (every model): 477 optimizer steps of 1,048,576 tokens
  (128 / 64 / 32 / 16 sequences per step at 8k / 16k / 32k / 64k), that is
  500,170,752 tokens, "0.5B"; the rotary base raised to 5e5 / 1e6 / 2e6 /
  5e6 in training and evaluation; all parameters trained, routers included;
  Muon at peak 0.0025 and AdamW at 7.5e-5 (a quarter of the pretraining
  peaks) with fresh optimizer states, 10 % linear warmup and cosine
  annealing over the remaining steps; weight decay 0.1 and clipping 1.0 as
  in pretraining.
* **Equal FLOPs** (Nestor): with ``r(S)`` its forward FLOPs per token
  relative to the MoE transformer of the same scale
  (:mod:`nestor.continual.flops`), the run trains for
  ``T = ceil(0.5B / r(S) / 1,048,576)`` steps at the same tokens per step,
  warmup 10 % of ``T``, the cosine spread over ``T``, and BOTH peak learning
  rates multiplied by ``r(S)`` so that steps x peak learning rate equals the
  equal-token run's. ``equal_flops_new`` draws from the full extension
  corpus, so the extra steps see documents the equal-token run did not;
  ``equal_flops_repeated`` restricts the corpus to the sequences the
  equal-token run consumed, extended by ``(T - 477)`` steps' worth of them
  drawn without replacement (``nestor.data.seen_subset``), and is one
  shuffled pass over that multiset.

Evaluation: the held-out loss at length ``S`` over the first ``S`` tokens of
200 held-out documents of at least ``S`` tokens (100 at 64k), with the run's
rotary base, in full precision (``scripts/eval_long_context.py``).

The module is also a command: it prints the ``DCLM_*`` environment one run
needs, which ``scripts/slurm_extend.sh`` evaluates before launching.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import shlex
from pathlib import Path

from nestor.continual.flops import flop_ratio, split_experiment_name

PRETRAIN_LENGTH: int = 4096
"""int: The context every model was pretrained at."""

LENGTHS: tuple[int, ...] = (8192, 16384, 32768, 65536)
"""tuple[int, ...]: The target context lengths of the ladder."""

ROPE_BASE: dict[int, int] = {
    PRETRAIN_LENGTH: 10_000,
    8192: 500_000,
    16384: 1_000_000,
    32768: 2_000_000,
    65536: 5_000_000,
}
"""dict[int, int]: Rotary base per context length, in training and evaluation."""

TOKENS_PER_STEP: int = 1_048_576
"""int: Tokens per optimizer step at every length."""

EXTENSION_TOKENS: int = 500_000_000
"""int: The equal-token budget (477 steps x 1,048,576 = 500,170,752 tokens)."""

EXTENSION_STEPS: int = math.ceil(EXTENSION_TOKENS / TOKENS_PER_STEP)
"""int: Optimizer steps of an equal-token run (477)."""

WARMUP_FRACTION: float = 0.1
"""float: Linear warmup as a fraction of the run's steps."""

LR_MUON: float = 0.0025
"""float: Peak Muon learning rate, a quarter of the pretraining peak."""

LR_ADAMW: float = 7.5e-5
"""float: Peak AdamW learning rate, a quarter of the pretraining peak."""

HELDOUT_DOCUMENTS: dict[int, int] = {8192: 200, 16384: 200, 32768: 200, 65536: 100}
"""dict[int, int]: Held-out documents scored at each length."""

SETTINGS: tuple[str, ...] = ("equal_tokens", "equal_flops_new", "equal_flops_repeated")
"""tuple[str, ...]: The three extension settings."""

REFERENCE_FAMILY: str = "moe"
"""str: The family ``r(S)`` is measured against."""


def extension_corpus_name(seq_len: int) -> str:
    """File name of the extension corpus at a length: DCLM documents of at least ``seq_len`` tokens."""

    return f"dclm_longdoc_ge{seq_len}_train.bin"


def heldout_corpus_name(seq_len: int) -> str:
    """File name of the held-out set at a length (documents of at least ``seq_len`` tokens)."""

    return f"dclm_longdoc_ge{seq_len}_val.bin"


def repeated_corpus_name(seq_len: int, steps: int) -> str:
    """File name of the repeated-documents multiset for a ``steps``-step equal-FLOP run."""

    return f"dclm_longdoc_ge{seq_len}_train_repeated_T{steps}.bin"


def extension_flop_ratio(experiment: str, seq_len: int, reference: str = REFERENCE_FAMILY) -> float:
    """``r(S)``: forward FLOPs per token of a cell relative to a baseline of its rung.

    The two baselines of a rung have the same body up to the rounding of the
    MoE's expert widths (identical at 100M and 500M, 0.3 % apart at 200M and
    0.2 % at 1B). The paper's budgets were taken against the MoE control,
    except at 200M, where the dense transformer was the reference; the 200M
    equal-FLOP runs therefore took 570 / 592 steps at 32k / 64k where the MoE
    reference gives 571 / 593.

    Args:
        experiment (str): A registered experiment name.
        seq_len (int): The context length.
        reference (str): ``"moe"`` or ``"dense"``, the baseline family to divide by.

    Returns:
        float: The causal body-FLOP ratio (exactly 1 for the reference itself).
    """

    _, scale = split_experiment_name(experiment)
    return flop_ratio(experiment, f"{reference}_{scale}", seq_len=seq_len)


def equal_flop_steps(ratio: float) -> int:
    """Steps of an equal-FLOP run: ``ceil(0.5B / r / tokens_per_step)``.

    Args:
        ratio (float): ``r(S)`` of the model.

    Returns:
        int: The step count (477 at ``r = 1``).
    """

    return math.ceil(EXTENSION_TOKENS / ratio / TOKENS_PER_STEP)


def resolve_checkpoint(path: Path) -> Path:
    """A ``model_final.pt`` path, or the newest one below a run root.

    Args:
        path (Path): A checkpoint file or a directory containing one.

    Returns:
        Path: The checkpoint file.

    Raises:
        FileNotFoundError: When no ``model_final.pt`` exists below ``path``.
    """

    if path.is_file():
        return path
    found = sorted(path.rglob("model_final.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no model_final.pt under {path}")
    return found[-1]


@dataclasses.dataclass(frozen=True)
class ExtensionRun:
    """One run of the ladder: a cell, a target length and a setting.

    Attributes:
        experiment (str): The registered cell (``nestor_200m``, ``dense_1b``, ...).
        seq_len (int): The target context length, one of :data:`LENGTHS`.
        setting (str): One of :data:`SETTINGS`; the equal-FLOP settings are
            Nestor's (the baselines define the budget).
        reference (str): The baseline family ``r(S)`` is measured against,
            ``"moe"`` (default) or ``"dense"``; see :func:`extension_flop_ratio`.
    """

    experiment: str
    seq_len: int
    setting: str = "equal_tokens"
    reference: str = REFERENCE_FAMILY

    def __post_init__(self) -> None:
        family, _ = split_experiment_name(self.experiment)
        if self.seq_len not in LENGTHS:
            raise ValueError(f"seq_len must be one of {LENGTHS}, got {self.seq_len}")
        if self.setting not in SETTINGS:
            raise ValueError(f"setting must be one of {SETTINGS}, got {self.setting!r}")
        if self.reference not in ("moe", "dense"):
            raise ValueError(f"reference must be 'moe' or 'dense', got {self.reference!r}")
        if self.setting != "equal_tokens" and family != "nestor":
            raise ValueError(
                f"{self.setting!r} is a Nestor setting; the baselines run at equal tokens "
                "(their FLOP ratio to the MoE control is 1)"
            )

    @property
    def family(self) -> str:
        """str: ``dense``, ``moe`` or ``nestor``."""

        return split_experiment_name(self.experiment)[0]

    @property
    def scale(self) -> str:
        """str: The rung (``100m`` ... ``1b``)."""

        return split_experiment_name(self.experiment)[1]

    @property
    def flop_ratio(self) -> float:
        """float: ``r(S)`` for the equal-FLOP settings, 1 for equal tokens."""

        if self.setting == "equal_tokens":
            return 1.0
        return extension_flop_ratio(self.experiment, self.seq_len, self.reference)

    @property
    def steps(self) -> int:
        """int: Optimizer steps: 477, or ``ceil(0.5B / r / tokens_per_step)`` at equal FLOPs."""

        return equal_flop_steps(self.flop_ratio)

    @property
    def warmup_steps(self) -> int:
        """int: Linear warmup steps, 10 % of the run."""

        return int(self.steps * WARMUP_FRACTION)

    @property
    def lr_muon(self) -> float:
        """float: Peak Muon learning rate (scaled by ``r`` at equal FLOPs)."""

        return LR_MUON * self.flop_ratio

    @property
    def lr_adamw(self) -> float:
        """float: Peak AdamW learning rate (scaled by ``r`` at equal FLOPs)."""

        return LR_ADAMW * self.flop_ratio

    @property
    def rope_base(self) -> int:
        """int: Rotary base of the run (training and evaluation)."""

        return ROPE_BASE[self.seq_len]

    @property
    def seqs_per_step(self) -> int:
        """int: Sequences per optimizer step (``1,048,576 / S``)."""

        return TOKENS_PER_STEP // self.seq_len

    @property
    def tokens(self) -> int:
        """int: Training tokens of the run."""

        return self.steps * TOKENS_PER_STEP

    @property
    def run_suffix(self) -> str:
        """str: Run-name suffix, ``ext<S>_<setting>``."""

        return f"ext{self.seq_len}_{self.setting}"

    def corpus(self, data_dir: Path) -> Path:
        """The training corpus of the run under ``data_dir``.

        Args:
            data_dir (Path): Directory of the long-context bins.

        Returns:
            Path: The full extension corpus, or the repeated-documents multiset.
        """

        if self.setting == "equal_flops_repeated":
            return data_dir / repeated_corpus_name(self.seq_len, self.steps)
        return data_dir / extension_corpus_name(self.seq_len)

    def heldout(self, data_dir: Path) -> Path:
        """The held-out set at the run's length under ``data_dir``."""

        return data_dir / heldout_corpus_name(self.seq_len)

    def env(
        self,
        *,
        data_dir: Path,
        init_checkpoint: Path | None,
        output_root: Path = Path("outputs"),
    ) -> dict[str, str]:
        """The environment of the run for ``nestor.cli`` and the launcher.

        ``DCLM_*`` keys are read by the experiment registry; ``NESTOR_OUTPUT_ROOT``
        gives the run its own root (the registry names a run after its cell, so
        a shared root would make auto-resume pick up the pretraining checkpoints);
        ``CONTINUAL_*`` keys are for the launcher (the batch shape).

        Args:
            data_dir (Path): Directory of the long-context bins.
            init_checkpoint (Path | None): The 4k ``model_final.pt`` to start from
                (a run root resolves to its newest one); ``None`` omits the key.
            output_root (Path): Root under which the run's own root is created.

        Returns:
            dict[str, str]: Environment variables, values as strings.
        """

        env = {
            "DCLM_BLOCK_SIZE": str(self.seq_len),
            "DCLM_ROPE_BASE": str(self.rope_base),
            "DCLM_DATA_PATH": str(self.corpus(data_dir)),
            "DCLM_MAX_STEPS": str(self.steps),
            "DCLM_WARMUP_STEPS": str(self.warmup_steps),
            "DCLM_LR_MUON": f"{self.lr_muon:.6g}",
            "DCLM_LR_ADAMW": f"{self.lr_adamw:.6g}",
            "DCLM_RUN_SUFFIX": self.run_suffix,
            "NESTOR_OUTPUT_ROOT": str(output_root / "continual" / f"{self.experiment}_{self.run_suffix}"),
            # The column path buckets its gathered rows; S / 32 keeps the number
            # of distinct shapes (and compiled variants) bounded at long contexts.
            "NESTOR_GATHERED_CAP_BUCKET": str(self.seq_len // 32),
            "CONTINUAL_SEQS_PER_STEP": str(self.seqs_per_step),
            "CONTINUAL_HELDOUT": str(self.heldout(data_dir)),
            "CONTINUAL_HELDOUT_DOCUMENTS": str(HELDOUT_DOCUMENTS[self.seq_len]),
        }
        if init_checkpoint is not None:
            env["DCLM_INIT_CHECKPOINT"] = str(resolve_checkpoint(init_checkpoint))
        return env

    def describe(self) -> str:
        """One line: cell, length, setting, steps, learning rates, ratio."""

        return (
            f"{self.experiment} -> {self.seq_len // 1024}k {self.setting}: r={self.flop_ratio:.4f} "
            f"steps={self.steps} warmup={self.warmup_steps} tokens={self.tokens:,} "
            f"lr_muon={self.lr_muon:.6g} lr_adamw={self.lr_adamw:.6g} rope={self.rope_base}"
        )


def paper_runs() -> list[ExtensionRun]:
    """The runs of the paper's ladder: every cell at every length, Nestor in all three settings.

    Returns:
        list[ExtensionRun]: 4 scales x 4 lengths x (1 + 1 + 3) = 80 runs.
    """

    runs: list[ExtensionRun] = []
    for scale in ("100m", "200m", "500m", "1b"):
        for seq_len in LENGTHS:
            runs.append(ExtensionRun(f"dense_{scale}", seq_len))
            runs.append(ExtensionRun(f"moe_{scale}", seq_len))
            runs.extend(ExtensionRun(f"nestor_{scale}", seq_len, setting) for setting in SETTINGS)
    return runs


def _format_env(env: dict[str, str]) -> str:
    """``export KEY=VALUE`` lines, shell-quoted, for ``eval`` in a launcher."""

    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())


def main() -> None:
    """Print one run's protocol, or its environment for the launcher."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="registered cell, e.g. nestor_200m")
    parser.add_argument("--length", type=int, required=True, choices=LENGTHS, help="target context length")
    parser.add_argument("--setting", default="equal_tokens", choices=SETTINGS)
    parser.add_argument(
        "--reference", default=REFERENCE_FAMILY, choices=("moe", "dense"),
        help="baseline family the equal-FLOP budget is measured against",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/longctx"), help="directory of the long-context bins")
    parser.add_argument("--init", type=Path, default=None, help="the 4k model_final.pt (or its run root)")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--env", action="store_true", help="print `export KEY=VALUE` lines instead of a summary")
    args = parser.parse_args()

    run = ExtensionRun(args.experiment, args.length, args.setting, args.reference)
    if args.env:
        print(_format_env(run.env(data_dir=args.data_dir, init_checkpoint=args.init, output_root=args.output_root)))
    else:
        print(run.describe())
        print(f"corpus:   {run.corpus(args.data_dir)}")
        print(f"held-out: {run.heldout(args.data_dir)} ({HELDOUT_DOCUMENTS[run.seq_len]} documents)")


if __name__ == "__main__":
    main()
