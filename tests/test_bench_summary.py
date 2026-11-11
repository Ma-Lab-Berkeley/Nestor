"""Tests for ``scripts/bench_summary.py`` (the single-GPU benchmark report).

The summary turns the trainer's ``Step N | train/...`` log lines of each
benchmark leg into steady-state seconds per optimizer step and pairs every leg
with the baseline leg of its own round. The tests feed it log text written out
by hand (the reference: the numbers are computed in the test from the same
lines) and check the parsing, the window skipping, and the paired ratio.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_summary() -> ModuleType:
    """Import ``scripts/bench_summary.py`` by path (``scripts`` is not a package).

    Returns:
        ModuleType: The loaded module.
    """

    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_summary.py"
    spec = importlib.util.spec_from_file_location("bench_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses resolves the module of a class
    # through sys.modules while the class body is being processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _log(sec_per_step: list[float], *, first_step: int = 4, stride: int = 4) -> str:
    """A leg's log: noise lines plus one trainer metric line per window.

    Args:
        sec_per_step (list[float]): ``sec_per_step`` of consecutive windows.
        first_step (int): Optimizer step of the first window.
        stride (int): Optimizer steps per window.

    Returns:
        str: Log text in the trainer's format.
    """

    lines = ["[pretrain] experiment=nestor_100m gpus=1", "Starting training for 12 steps"]
    for index, seconds in enumerate(sec_per_step):
        step = first_step + index * stride
        lines.append(
            f"2026-11-11 10:00:00 | INFO | Step {step} | train/loss: {11.0 - 0.1 * index:.4f} "
            f"| train/lr: 0.0010 | train/sec_per_step: {seconds:.4f} "
            f"| train/mem_peak_alloc_gib: {30.0 + index:.4f} | train/note: warm"
        )
    lines.append("Training complete!")
    return "\n".join(lines)


class BenchSummaryTest:
    """Parsing and pairing of the benchmark legs."""

    def test_parse_metrics_reads_only_trainer_lines(self) -> None:
        """Metric lines are parsed in order; non-numeric values are dropped."""

        summary = _load_summary()
        rows = summary.parse_metrics(_log([9.0, 2.0, 2.5]))
        assert [step for step, _ in rows] == [4, 8, 12]
        assert rows[1][1] == {
            "loss": 10.9,
            "lr": 0.001,
            "sec_per_step": 2.0,
            "mem_peak_alloc_gib": 31.0,
        }

    def test_leg_summary_skips_leading_windows(self) -> None:
        """The first window (compile-laden) is excluded from the timing only."""

        summary = _load_summary()
        leg = summary.summarize_leg(
            _log([9.0, 2.0, 2.5, 3.0]), round_index=1, name="base", skip_windows=1
        )
        assert leg.steps == [8, 12, 16]
        assert leg.sec_per_step == [2.0, 2.5, 3.0]
        assert leg.mean == (2.0 + 2.5 + 3.0) / 3
        assert leg.median == 2.5
        assert leg.losses == {4: 11.0, 8: 10.9, 12: 10.8, 16: 10.7}
        assert leg.mem_peak_alloc_gib == 33.0

    def test_empty_log_has_no_timing(self) -> None:
        """A leg that never logged a step reports no numbers instead of failing."""

        summary = _load_summary()
        leg = summary.summarize_leg(
            "Traceback (most recent call last):", round_index=1, name="x", skip_windows=1
        )
        assert leg.mean is None and leg.median is None
        assert leg.losses == {} and leg.mem_peak_alloc_gib is None
        assert "n/a" in summary.format_report([leg], baseline="base")

    def test_report_pairs_each_leg_with_its_rounds_baseline(self) -> None:
        """base/leg uses the baseline of the SAME round, then averages rounds."""

        summary = _load_summary()
        legs = [
            summary.summarize_leg(_log([9.0, 4.0, 4.0]), round_index=1, name="base", skip_windows=1),
            summary.summarize_leg(_log([9.0, 2.0, 2.0]), round_index=1, name="tuned", skip_windows=1),
            summary.summarize_leg(_log([9.0, 6.0, 6.0]), round_index=2, name="base", skip_windows=1),
            summary.summarize_leg(_log([9.0, 2.0, 2.0]), round_index=2, name="tuned", skip_windows=1),
        ]
        report = summary.format_report(legs, baseline="base")
        per_arm = {
            line.split()[0]: line.split()
            for line in report.split("\n\n")[1].splitlines()[1:]
        }
        # round 1: 4/2 = 2.0, round 2: 6/2 = 3.0 -> paired mean 2.5 (a pooled
        # ratio of the means would read 5/2 = 2.5 here too, so check the legs).
        assert float(per_arm["tuned"][3]) == 2.5
        assert float(per_arm["base"][3]) == 1.0
        assert float(per_arm["tuned"][2]) == 2.0
        leg_rows = [line.split() for line in report.split("\n\n")[0].splitlines()[1:]]
        ratios = {(row[0], row[1]): float(row[5]) for row in leg_rows}
        assert ratios[("1", "tuned")] == 2.0 and ratios[("2", "tuned")] == 3.0
