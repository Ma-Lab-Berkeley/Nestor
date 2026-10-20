"""Pins for the continued-training protocols (``nestor.continual``).

The FLOP accounting against the slow explicit formulas and the paper's
documented values; the length ladder's constants, equal-FLOP budgets (the
paper's table of ``r(S)`` and step counts at every scale and length) and the
environment one run needs; the domain-adaptation runs, their corpora and the
frozen-router flag; and the two command-line entry points.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from nestor.continual.adaptation import (
    ADAPTATION_LENGTH,
    CORPORA,
    FORGETTING_HELDOUT,
    PAPER_EXPERIMENTS,
    REPLAY_SOURCE,
    AdaptationRun,
    freezes_routers,
    paper_runs as adaptation_paper_runs,
    replay_mix_name,
)
from nestor.continual.extension import (
    EXTENSION_STEPS,
    LENGTHS,
    LR_ADAMW,
    LR_MUON,
    ROPE_BASE,
    SETTINGS,
    TOKENS_PER_STEP,
    ExtensionRun,
    equal_flop_steps,
    extension_flop_ratio,
    paper_runs as extension_paper_runs,
    resolve_checkpoint,
)
from nestor.continual.flops import body_flops, flop_ratio, split_experiment_name
from nestor.experiments.pretrain import SCALES

REPO = Path(__file__).resolve().parents[1]

# The paper's table of r(S) and equal-FLOP step counts, and the baseline each
# scale's budget was measured against (the two baselines differ only by the
# rounding of the MoE's expert widths; identical at 100M and 500M).
PAPER_BUDGETS: dict[str, dict[int, tuple[float, int]]] = {
    "100m": {8192: (0.940, 508), 16384: (0.895, 533), 32768: (0.860, 555), 65536: (0.837, 570)},
    "200m": {8192: (0.933, 512), 16384: (0.881, 542), 32768: (0.837, 570), 65536: (0.806, 592)},
    "500m": {8192: (0.938, 509), 16384: (0.888, 537), 32768: (0.840, 568), 65536: (0.802, 595)},
    "1b": {8192: (0.956, 499), 16384: (0.911, 524), 32768: (0.863, 553), 65536: (0.821, 581)},
}
PAPER_REFERENCE: dict[str, str] = {"100m": "moe", "200m": "dense", "500m": "moe", "1b": "moe"}
# Nestor's causal 4k ratio to the dense transformer (the pretraining section).
CAUSAL_4K_RATIO = {"100m": 0.986, "200m": 0.981, "500m": 0.979, "1b": 0.989}


def reference_dense_body_flops(scale: str, seq_len: int) -> float:
    """Explicit body FLOPs of a dense transformer of the rung over one sequence.

    ``L`` layers of 16 heads of width ``d / 16`` (projections ``8 S d^2``,
    causal core ``2 S^2 d``) and one SwiGLU of width ``4d`` (``6 S d x 4d``).
    """

    sc = SCALES[scale]
    d = sc.hidden_size
    return sc.num_layers * (8 * seq_len * d * d + 2 * seq_len**2 * d + 6 * seq_len * d * sc.mlp_hidden)


class FlopAccountingTest:
    """The FLOP module against the explicit formulas and the documented ratios."""

    def test_dense_body_matches_the_explicit_formula(self) -> None:
        for scale in SCALES:
            for seq_len in (4096, *LENGTHS):
                got = body_flops(f"dense_{scale}", seq_len=seq_len).total
                assert got == pytest.approx(reference_dense_body_flops(scale, seq_len), rel=1e-12), (scale, seq_len)

    def test_moe_and_dense_share_a_body_up_to_expert_width_rounding(self) -> None:
        # round8(4d / 3) x 3 against 4d: exact at 100M and 500M, 0.3 % at 200M, 0.2 % at 1B.
        for scale in SCALES:
            for seq_len in (4096, 32768):
                r = flop_ratio(f"moe_{scale}", f"dense_{scale}", seq_len=seq_len)
                assert abs(r - 1.0) < 0.0035, (scale, seq_len, r)
                if scale in ("100m", "500m"):
                    assert r == pytest.approx(1.0, abs=1e-12)

    def test_nestor_4k_ratio_to_dense_as_documented(self) -> None:
        for scale, expected in CAUSAL_4K_RATIO.items():
            r = flop_ratio(f"nestor_{scale}", f"dense_{scale}", seq_len=4096)
            assert abs(r - expected) < 0.0015, (scale, r)
            assert r < 1.0

    def test_nestor_ratio_falls_with_the_context_length(self) -> None:
        for scale in SCALES:
            ratios = [extension_flop_ratio(f"nestor_{scale}", s) for s in (4096, *LENGTHS)]
            assert ratios == sorted(ratios, reverse=True), (scale, ratios)
            # The attention share grows with S, so the ratio tends to the
            # attention ratio (0.75-0.81) and sits at 0.80-0.84 at 64k.
            assert 0.80 <= ratios[-1] <= 0.84, (scale, ratios[-1])

    def test_adaptation_compute_reference(self) -> None:
        # At 32k the 500M Nestor costs ~0.84x the 500M MoE and ~0.43x the 1B dense model.
        assert flop_ratio("nestor_500m", "moe_500m", seq_len=32768) == pytest.approx(0.840, abs=1e-3)
        assert flop_ratio("nestor_500m", "dense_1b", seq_len=32768) == pytest.approx(0.427, abs=2e-3)

    def test_experiment_names_are_parsed_and_checked(self) -> None:
        assert split_experiment_name("nestor_200m") == ("nestor", "200m")
        assert split_experiment_name("dense_1b") == ("dense", "1b")
        for bad in ("nestor", "vanilla_200m", "nestor_300m", ""):
            with pytest.raises(ValueError):
                split_experiment_name(bad)


class ExtensionProtocolTest:
    """The ladder's constants, budgets and run environment."""

    def test_constants(self) -> None:
        assert LENGTHS == (8192, 16384, 32768, 65536)
        assert TOKENS_PER_STEP == 1_048_576 and EXTENSION_STEPS == 477
        assert EXTENSION_STEPS * TOKENS_PER_STEP == 500_170_752
        assert LR_MUON == 0.0025 and LR_ADAMW == 7.5e-5
        assert ROPE_BASE == {4096: 10_000, 8192: 500_000, 16384: 1_000_000, 32768: 2_000_000, 65536: 5_000_000}
        assert SETTINGS == ("equal_tokens", "equal_flops_new", "equal_flops_repeated")

    def test_paper_budgets_ratio_and_steps(self) -> None:
        for scale, cells in PAPER_BUDGETS.items():
            for seq_len, (ratio, steps) in cells.items():
                run = ExtensionRun(f"nestor_{scale}", seq_len, "equal_flops_new", PAPER_REFERENCE[scale])
                assert abs(run.flop_ratio - ratio) < 6e-4, (scale, seq_len, run.flop_ratio)
                # The runs' budget is ceil(0.5B / r / tokens_per_step), which
                # the paper writes as ceil(477 / r); the two differ by one step
                # at a few cells, and the table's step counts are the runs'.
                assert run.steps == steps, (scale, seq_len, run.steps)
                assert run.warmup_steps == int(steps * 0.1)
                repeated = ExtensionRun(f"nestor_{scale}", seq_len, "equal_flops_repeated", PAPER_REFERENCE[scale])
                assert repeated.steps == steps and repeated.flop_ratio == run.flop_ratio

    def test_the_moe_reference_is_the_default_and_differs_at_200m_by_one_step(self) -> None:
        assert ExtensionRun("nestor_200m", 32768, "equal_flops_new").reference == "moe"
        assert ExtensionRun("nestor_200m", 32768, "equal_flops_new").steps == 571
        assert ExtensionRun("nestor_200m", 65536, "equal_flops_new").steps == 593
        assert ExtensionRun("nestor_200m", 8192, "equal_flops_new").steps == 512
        for scale in ("100m", "500m", "1b"):
            for seq_len, (_, steps) in PAPER_BUDGETS[scale].items():
                assert ExtensionRun(f"nestor_{scale}", seq_len, "equal_flops_new").steps == steps

    def test_equal_flop_steps_formula(self) -> None:
        assert equal_flop_steps(1.0) == EXTENSION_STEPS
        assert equal_flop_steps(0.5) == math.ceil(500_000_000 / 0.5 / TOKENS_PER_STEP) == 954

    def test_equal_tokens_run_and_its_environment(self, tmp_path: Path) -> None:
        run = ExtensionRun("moe_500m", 16384)
        assert run.family == "moe" and run.scale == "500m"
        assert run.flop_ratio == 1.0 and run.steps == 477 and run.warmup_steps == 47
        assert run.lr_muon == LR_MUON and run.lr_adamw == LR_ADAMW
        assert run.rope_base == 1_000_000 and run.seqs_per_step == 64
        assert run.tokens == 500_170_752 and run.run_suffix == "ext16384_equal_tokens"
        assert run.corpus(tmp_path) == tmp_path / "dclm_longdoc_ge16384_train.bin"
        assert run.heldout(tmp_path) == tmp_path / "dclm_longdoc_ge16384_val.bin"
        env = run.env(data_dir=tmp_path, init_checkpoint=None, output_root=Path("out"))
        assert env["DCLM_BLOCK_SIZE"] == "16384" and env["DCLM_ROPE_BASE"] == "1000000"
        assert env["DCLM_MAX_STEPS"] == "477" and env["DCLM_WARMUP_STEPS"] == "47"
        assert env["DCLM_LR_MUON"] == "0.0025" and env["DCLM_LR_ADAMW"] == "7.5e-05"
        assert env["DCLM_DATA_PATH"] == str(tmp_path / "dclm_longdoc_ge16384_train.bin")
        assert env["DCLM_RUN_SUFFIX"] == "ext16384_equal_tokens"
        assert env["NESTOR_OUTPUT_ROOT"] == str(Path("out") / "continual" / "moe_500m_ext16384_equal_tokens")
        assert env["NESTOR_GATHERED_CAP_BUCKET"] == "512"
        assert env["CONTINUAL_SEQS_PER_STEP"] == "64" and env["CONTINUAL_HELDOUT_DOCUMENTS"] == "200"
        assert "DCLM_INIT_CHECKPOINT" not in env
        assert {int(ExtensionRun("dense_100m", s).seqs_per_step) for s in LENGTHS} == {128, 64, 32, 16}
        assert ExtensionRun("dense_100m", 65536).env(data_dir=tmp_path, init_checkpoint=None)["CONTINUAL_HELDOUT_DOCUMENTS"] == "100"

    def test_equal_flops_runs_scale_the_learning_rates_and_pick_their_corpus(self, tmp_path: Path) -> None:
        new = ExtensionRun("nestor_1b", 65536, "equal_flops_new")
        repeated = ExtensionRun("nestor_1b", 65536, "equal_flops_repeated")
        r = new.flop_ratio
        assert 0.80 < r < 0.83
        for run in (new, repeated):
            assert run.steps == 581 and run.warmup_steps == 58
            assert run.lr_muon == pytest.approx(LR_MUON * r) and run.lr_adamw == pytest.approx(LR_ADAMW * r)
            assert run.tokens == 581 * TOKENS_PER_STEP
            # steps x peak learning rate equals the equal-token run's.
            assert run.steps * run.lr_muon == pytest.approx(477 * LR_MUON, rel=2e-3)
        assert new.corpus(tmp_path) == tmp_path / "dclm_longdoc_ge65536_train.bin"
        assert repeated.corpus(tmp_path) == tmp_path / "dclm_longdoc_ge65536_train_repeated_T581.bin"
        env = repeated.env(data_dir=tmp_path, init_checkpoint=None)
        assert env["DCLM_MAX_STEPS"] == "581" and env["DCLM_RUN_SUFFIX"] == "ext65536_equal_flops_repeated"
        assert float(env["DCLM_LR_MUON"]) == pytest.approx(LR_MUON * r, rel=1e-4)

    def test_invalid_runs_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            ExtensionRun("dense_200m", 32768, "equal_flops_new")  # a baseline at equal FLOPs
        with pytest.raises(ValueError):
            ExtensionRun("nestor_200m", 4096)  # not a target length
        with pytest.raises(ValueError):
            ExtensionRun("nestor_200m", 32768, "iso_flop")
        with pytest.raises(ValueError):
            ExtensionRun("nestor_200m", 32768, "equal_flops_new", "vanilla")
        with pytest.raises(ValueError):
            ExtensionRun("nestor_2b", 32768)

    def test_paper_runs_cover_the_ladder(self) -> None:
        runs = extension_paper_runs()
        assert len(runs) == 4 * 4 * 5
        assert len({(r.experiment, r.seq_len, r.setting) for r in runs}) == len(runs)
        assert all(r.setting == "equal_tokens" for r in runs if r.family != "nestor")
        assert len({r.run_suffix for r in runs}) == 4 * 3  # per length: the three settings; families share suffixes

    def test_resolve_checkpoint_takes_the_newest_final_checkpoint(self, tmp_path: Path) -> None:
        old = tmp_path / "20260101_000000__run" / "checkpoints" / "model_final.pt"
        new = tmp_path / "20260102_000000__run" / "checkpoints" / "model_final.pt"
        for p, stamp in ((old, 1_000), (new, 2_000)):
            p.parent.mkdir(parents=True)
            p.write_bytes(b"x")
            import os

            os.utime(p, (stamp, stamp))
        assert resolve_checkpoint(tmp_path) == new
        assert resolve_checkpoint(old) == old
        env = ExtensionRun("nestor_100m", 8192).env(data_dir=tmp_path, init_checkpoint=tmp_path)
        assert env["DCLM_INIT_CHECKPOINT"] == str(new)
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint(tmp_path / "nowhere")


class AdaptationProtocolTest:
    """The domain-adaptation runs, their corpora and the frozen-router flag."""

    def test_corpora_and_constants(self) -> None:
        assert tuple(CORPORA) == ("pg19", "owm", "algstack", "fw2de")
        assert CORPORA["pg19"].eval_mode == "documents"
        assert all(CORPORA[c].eval_mode == "packed" for c in ("owm", "algstack", "fw2de"))
        assert ADAPTATION_LENGTH == 32768
        assert REPLAY_SOURCE == "dclm_longdoc_ge32768_train.bin"
        assert FORGETTING_HELDOUT == "dclm_longdoc_ge32768_val.bin"
        assert replay_mix_name("owm", 25) == "owm_replay25.bin"
        assert PAPER_EXPERIMENTS == ("nestor_500m", "moe_500m", "dense_1b")

    def test_frozen_routers_for_the_sparse_families_only(self) -> None:
        assert freezes_routers("nestor_500m") and freezes_routers("moe_500m")
        assert not freezes_routers("dense_1b") and not freezes_routers("dense_100m")

    def test_run_environment(self, tmp_path: Path) -> None:
        run = AdaptationRun("nestor_500m", "algstack", 25)
        assert run.freeze_routers and run.run_suffix == "adapt_algstack_replay25"
        assert run.training_corpus(tmp_path) == tmp_path / "algstack_replay25.bin"
        assert run.target_test(tmp_path) == tmp_path / "algstack_test.bin"
        assert run.forgetting_heldout(tmp_path) == tmp_path / "dclm_longdoc_ge32768_val.bin"
        env = run.env(data_dir=tmp_path, init_checkpoint=None, output_root=Path("out"))
        assert env["DCLM_BLOCK_SIZE"] == "32768" and env["DCLM_ROPE_BASE"] == "2000000"
        assert env["DCLM_MAX_STEPS"] == "477" and env["DCLM_WARMUP_STEPS"] == "47"
        assert env["DCLM_LR_MUON"] == "0.0025" and env["DCLM_LR_ADAMW"] == "7.5e-05"
        assert env["DCLM_DATA_PATH"] == str(tmp_path / "algstack_replay25.bin")
        assert env["DCLM_RUN_SUFFIX"] == "adapt_algstack_replay25"
        assert env["NESTOR_OUTPUT_ROOT"] == str(Path("out") / "continual" / "nestor_500m_adapt_algstack_replay25")
        assert env["NESTOR_FREEZE_ROUTERS"] == "1"
        assert env["CONTINUAL_SEQS_PER_STEP"] == "32"
        assert env["CONTINUAL_TARGET_TEST"] == str(tmp_path / "algstack_test.bin")
        assert env["CONTINUAL_TARGET_EVAL_MODE"] == "packed"

        dense = AdaptationRun("dense_1b", "pg19")
        assert not dense.freeze_routers and dense.run_suffix == "adapt_pg19"
        env = dense.env(data_dir=tmp_path, init_checkpoint=None)
        assert "NESTOR_FREEZE_ROUTERS" not in env
        assert env["DCLM_DATA_PATH"] == str(tmp_path / "pg19_train.bin")
        assert env["CONTINUAL_TARGET_EVAL_MODE"] == "documents"

    def test_invalid_runs_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            AdaptationRun("nestor_500m", "books")
        with pytest.raises(ValueError):
            AdaptationRun("nestor_500m", "pg19", 100)
        with pytest.raises(ValueError):
            AdaptationRun("nestor_500m", "pg19", -1)
        with pytest.raises(ValueError):
            AdaptationRun("column_500m", "pg19")

    def test_paper_runs(self) -> None:
        runs = adaptation_paper_runs()
        assert len(runs) == 24
        assert len({(r.experiment, r.corpus, r.replay_pct) for r in runs}) == 24
        assert {r.replay_pct for r in runs} == {0, 25}
        assert sum(r.freeze_routers for r in runs) == 16


class CommandLineTest:
    """The two modules print the environment a launcher evaluates."""

    def _run(self, *args: str) -> str:
        out = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, cwd=REPO, check=False)
        assert out.returncode == 0, out.stderr
        return out.stdout

    def test_extension_env(self) -> None:
        out = self._run(
            "nestor.continual.extension", "--experiment", "nestor_100m", "--length", "8192",
            "--setting", "equal_flops_new", "--data-dir", "d", "--env",
        )
        lines = out.strip().splitlines()
        assert all(line.startswith("export ") for line in lines)
        assert "export DCLM_MAX_STEPS=508" in lines
        assert "export DCLM_BLOCK_SIZE=8192" in lines
        assert "export DCLM_DATA_PATH=d/dclm_longdoc_ge8192_train.bin" in lines
        summary = self._run("nestor.continual.extension", "--experiment", "nestor_100m", "--length", "8192")
        assert "steps=477" in summary

    def test_adaptation_env(self) -> None:
        out = self._run(
            "nestor.continual.adaptation", "--experiment", "moe_500m", "--corpus", "owm",
            "--replay", "25", "--data-dir", "d", "--env",
        )
        lines = out.strip().splitlines()
        assert "export NESTOR_FREEZE_ROUTERS=1" in lines
        assert "export DCLM_DATA_PATH=d/owm_replay25.bin" in lines
        assert "export DCLM_MAX_STEPS=477" in lines
