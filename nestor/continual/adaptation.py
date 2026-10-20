"""Domain adaptation and forgetting: continued pretraining on a new domain at 32k.

The 500M Nestor, the 500M MoE transformer and the 1B dense transformer,
chosen for their similar pretraining losses, are first extended to 32k on
DCLM (the equal-token run of :mod:`nestor.continual.extension`). From that
checkpoint each model is trained separately on four target domains:

* books -- PG-19,
* mathematics -- OpenWebMath (from Proof-Pile-2),
* code -- AlgebraicStack (from Proof-Pile-2),
* German -- the ``deu_Latn`` subset of FineWeb-2,

for 477 steps of 32 x 32,768 tokens (0.5B tokens) under the 32k extension
recipe (rotary base 2e6, Muon 0.0025 / AdamW 7.5e-5, 47 warmup steps, cosine,
weight decay 0.1, clipping 1.0, seed 42, sampler seed 0). The two sparse
models train with frozen routers (``NESTOR_FREEZE_ROUTERS=1``:
:mod:`nestor.train.frozen_routers`); the dense model is fully fine-tuned.
The corpora have no length filter: documents are end-of-text separated,
concatenated and cut into 32,768-token windows that may cross document
boundaries.

Each run is repeated with 25 % of the budget replaced by DCLM sequences the
model already saw during context extension (``nestor.data.seen_subset
.build_replay_mix``: the target part a prefix of the no-replay run's sequences,
the replay part a prefix of the extension run's), the total budget unchanged.

Evaluation, on the starting and the final checkpoint alike, at 32k and in
full precision: the target test split (PG-19: the first 32,768 tokens of each
test book that reaches that length; the others: the first 200 non-overlapping
32,768-token windows of the test stream), and, for forgetting, the first
32,768 tokens of 200 DCLM held-out documents (the ladder's 32k held-out set).
Adaptation is the target loss before minus after; forgetting is the DCLM loss
after minus before.

The module is also a command: it prints the environment one run needs, which
``scripts/slurm_adapt.sh`` evaluates before launching.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
from pathlib import Path

from nestor.continual.extension import (
    EXTENSION_STEPS,
    ExtensionRun,
    _format_env,
    extension_corpus_name,
    heldout_corpus_name,
    resolve_checkpoint,
)
from nestor.continual.flops import split_experiment_name

ADAPTATION_LENGTH: int = 32768
"""int: Every adaptation run trains and is evaluated at 32k."""

ADAPTATION_STEPS: int = EXTENSION_STEPS
"""int: 477 steps of 1,048,576 tokens, the equal-token budget."""

PAPER_EXPERIMENTS: tuple[str, ...] = ("nestor_500m", "moe_500m", "dense_1b")
"""tuple[str, ...]: The three models of the domain-adaptation experiments."""

PAPER_REPLAY_PCTS: tuple[int, ...] = (0, 25)
"""tuple[int, ...]: The two replay settings reported."""

MIX_TOKENS: int = 504_000_000
"""int: Nominal size of a replay mixture; the 477-step run consumes ~99 % of it once."""

EVAL_SEQUENCES: int = 200
"""int: Sequences scored per evaluation set (PG-19 has 77 test books of >= 32k tokens)."""

FROZEN_ROUTER_FAMILIES: tuple[str, ...] = ("moe", "nestor")
"""tuple[str, ...]: Families adapted with frozen routers (the dense model has none)."""


@dataclasses.dataclass(frozen=True)
class Corpus:
    """A target domain.

    Attributes:
        name (str): Short name, the key of :data:`CORPORA` and the run suffix.
        title (str): Human-readable name.
        train_bin (str): File name of the tokenized training split.
        test_bin (str): File name of the tokenized test split.
        eval_mode (str): ``"documents"`` (one sequence per test document of at
            least 32k tokens: PG-19's books) or ``"packed"`` (consecutive
            windows of the test stream: the page-length corpora).
    """

    name: str
    title: str
    train_bin: str
    test_bin: str
    eval_mode: str


CORPORA: dict[str, Corpus] = {
    "pg19": Corpus("pg19", "PG-19 (books)", "pg19_train.bin", "pg19_test.bin", "documents"),
    "owm": Corpus("owm", "OpenWebMath (mathematics)", "owm_train.bin", "owm_test.bin", "packed"),
    "algstack": Corpus("algstack", "AlgebraicStack (code)", "algstack_train.bin", "algstack_test.bin", "packed"),
    "fw2de": Corpus("fw2de", "FineWeb-2 German", "fw2de_train.bin", "fw2de_test.bin", "packed"),
}
"""dict[str, Corpus]: The four target domains."""

REPLAY_SOURCE: str = extension_corpus_name(ADAPTATION_LENGTH)
"""str: The replay source: the 32k extension corpus the starting checkpoints trained on."""

FORGETTING_HELDOUT: str = heldout_corpus_name(ADAPTATION_LENGTH)
"""str: The DCLM held-out set forgetting is read on."""


def replay_mix_name(corpus: str, replay_pct: int) -> str:
    """File name of the mixture of a target corpus with ``replay_pct`` % DCLM replay."""

    return f"{corpus}_replay{replay_pct}.bin"


def freezes_routers(experiment: str) -> bool:
    """Whether a cell is adapted with frozen routers (the sparse families).

    Args:
        experiment (str): A registered experiment name.

    Returns:
        bool: ``True`` for the MoE control and Nestor, ``False`` for the dense model.
    """

    return split_experiment_name(experiment)[0] in FROZEN_ROUTER_FAMILIES


@dataclasses.dataclass(frozen=True)
class AdaptationRun:
    """One domain-adaptation run: a cell, a target corpus and a replay share.

    Attributes:
        experiment (str): The registered cell.
        corpus (str): A key of :data:`CORPORA`.
        replay_pct (int): Percentage of the budget drawn from the DCLM replay
            source, ``0 <= replay_pct < 100``.
    """

    experiment: str
    corpus: str
    replay_pct: int = 0

    def __post_init__(self) -> None:
        split_experiment_name(self.experiment)
        if self.corpus not in CORPORA:
            raise ValueError(f"corpus must be one of {tuple(CORPORA)}, got {self.corpus!r}")
        if not 0 <= self.replay_pct < 100:
            raise ValueError(f"replay_pct must be in [0, 100), got {self.replay_pct}")

    @property
    def extension(self) -> ExtensionRun:
        """ExtensionRun: The 32k equal-token recipe the run follows."""

        return ExtensionRun(self.experiment, ADAPTATION_LENGTH, "equal_tokens")

    @property
    def freeze_routers(self) -> bool:
        """bool: Frozen routers for the sparse families."""

        return freezes_routers(self.experiment)

    @property
    def run_suffix(self) -> str:
        """str: Run-name suffix, ``adapt_<corpus>[_replay<pct>]``."""

        suffix = f"adapt_{self.corpus}"
        return f"{suffix}_replay{self.replay_pct}" if self.replay_pct else suffix

    def training_corpus(self, data_dir: Path) -> Path:
        """The training corpus under ``data_dir``: the target split, or its replay mixture."""

        if self.replay_pct:
            return data_dir / replay_mix_name(self.corpus, self.replay_pct)
        return data_dir / CORPORA[self.corpus].train_bin

    def target_test(self, data_dir: Path) -> Path:
        """The target corpus's test split under ``data_dir``."""

        return data_dir / CORPORA[self.corpus].test_bin

    def forgetting_heldout(self, data_dir: Path) -> Path:
        """The DCLM 32k held-out set under ``data_dir``."""

        return data_dir / FORGETTING_HELDOUT

    def env(
        self,
        *,
        data_dir: Path,
        init_checkpoint: Path | None,
        output_root: Path = Path("outputs"),
    ) -> dict[str, str]:
        """The environment of the run for ``nestor.cli`` and the launcher.

        The 32k equal-token extension environment with the corpus, the run
        suffix and the run's own output root replaced, plus
        ``NESTOR_FREEZE_ROUTERS=1`` for the sparse families.

        Args:
            data_dir (Path): Directory of the corpora.
            init_checkpoint (Path | None): The 32k-extended ``model_final.pt``
                (or its run root); ``None`` omits the key.
            output_root (Path): Root under which the run's own root is created.

        Returns:
            dict[str, str]: Environment variables, values as strings.
        """

        env = self.extension.env(data_dir=data_dir, init_checkpoint=None, output_root=output_root)
        env["DCLM_DATA_PATH"] = str(self.training_corpus(data_dir))
        env["DCLM_RUN_SUFFIX"] = self.run_suffix
        env["NESTOR_OUTPUT_ROOT"] = str(output_root / "continual" / f"{self.experiment}_{self.run_suffix}")
        env["CONTINUAL_TARGET_TEST"] = str(self.target_test(data_dir))
        env["CONTINUAL_TARGET_EVAL_MODE"] = CORPORA[self.corpus].eval_mode
        if self.freeze_routers:
            env["NESTOR_FREEZE_ROUTERS"] = "1"
        if init_checkpoint is not None:
            env["DCLM_INIT_CHECKPOINT"] = str(resolve_checkpoint(init_checkpoint))
        return env

    def describe(self) -> str:
        """One line: cell, corpus, replay, recipe."""

        routers = "frozen routers" if self.freeze_routers else "all parameters"
        return (
            f"{self.experiment} on {CORPORA[self.corpus].title}, {self.replay_pct} % replay, {routers}: "
            f"{ADAPTATION_STEPS} steps x {self.extension.seqs_per_step} x {ADAPTATION_LENGTH} tokens, "
            f"lr_muon={self.extension.lr_muon:.6g} lr_adamw={self.extension.lr_adamw:.6g} "
            f"rope={self.extension.rope_base}"
        )


def paper_runs() -> list[AdaptationRun]:
    """The runs of the paper: three models x four corpora x two replay shares.

    Returns:
        list[AdaptationRun]: 24 runs.
    """

    return [
        AdaptationRun(experiment, corpus, replay_pct)
        for replay_pct in PAPER_REPLAY_PCTS
        for corpus in CORPORA
        for experiment in PAPER_EXPERIMENTS
    ]


def main() -> None:
    """Print one run's protocol, or its environment for the launcher."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="registered cell, e.g. nestor_500m")
    parser.add_argument("--corpus", required=True, choices=tuple(CORPORA))
    parser.add_argument("--replay", type=int, default=0, help="percent of the budget replayed from DCLM (0 or 25)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/longctx"), help="directory of the corpora")
    parser.add_argument("--init", type=Path, default=None, help="the 32k-extended model_final.pt (or its run root)")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--env", action="store_true", help="print `export KEY=VALUE` lines instead of a summary")
    args = parser.parse_args()

    run = AdaptationRun(args.experiment, args.corpus, args.replay)
    if args.env:
        print(_format_env(run.env(data_dir=args.data_dir, init_checkpoint=args.init, output_root=args.output_root)))
    else:
        print(run.describe())
        print(f"corpus:     {run.training_corpus(args.data_dir)}")
        print(f"target:     {run.target_test(args.data_dir)} ({CORPORA[run.corpus].eval_mode})")
        print(f"forgetting: {run.forgetting_heldout(args.data_dir)}")
        print(f"env keys:   {' '.join(shlex.quote(k) for k in run.env(data_dir=args.data_dir, init_checkpoint=None))}")


if __name__ == "__main__":
    main()
