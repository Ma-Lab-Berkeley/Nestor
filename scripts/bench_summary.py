"""Summarize the legs of ``scripts/bench_single_gpu.sh``.

Each leg is one short training run whose stdout holds the trainer's log
lines::

    ... Step 8 | train/loss: 10.4127 | train/sec_per_step: 1.9312 | ...

``sec_per_step`` is the mean wall time per optimizer step over the log window
that ends at that step. The summary drops each leg's first windows (compile,
allocator growth), averages the rest, and prints every leg next to the
baseline leg of the same round, so the ratio is paired within a round.

    python scripts/bench_summary.py LOG_DIR --baseline base [--skip-windows 1]

``LOG_DIR`` holds ``r<round>_<leg>.log`` files.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import statistics
from pathlib import Path

_STEP_LINE = re.compile(r"Step (\d+) \| (.*)$")
_LEG_FILE = re.compile(r"r(\d+)_(.+)\.log$")


@dataclasses.dataclass(kw_only=True)
class LegSummary:
    """Steady-state numbers of one leg.

    Attributes:
        round_index (int): Round the leg ran in.
        name (str): Leg name.
        steps (list[int]): Optimizer step of every log window kept.
        sec_per_step (list[float]): Mean seconds per optimizer step of every
            log window kept.
        losses (dict[int, float]): Logged training loss by optimizer step
            (every window, the skipped ones included).
        mem_peak_alloc_gib (float | None): Largest logged peak allocation.
    """

    round_index: int
    name: str
    steps: list[int]
    sec_per_step: list[float]
    losses: dict[int, float]
    mem_peak_alloc_gib: float | None

    @property
    def mean(self) -> float | None:
        """Mean seconds per step over the kept windows (``None`` if empty)."""

        return statistics.fmean(self.sec_per_step) if self.sec_per_step else None

    @property
    def median(self) -> float | None:
        """Median seconds per step over the kept windows (``None`` if empty)."""

        return statistics.median(self.sec_per_step) if self.sec_per_step else None


def parse_metrics(text: str) -> list[tuple[int, dict[str, float]]]:
    """Extract the trainer's metric lines from a leg's log.

    Args:
        text (str): Captured stdout/stderr of one training run.

    Returns:
        list[tuple[int, dict[str, float]]]: ``(optimizer step, metrics)`` in
            log order; metric names have the ``train/`` prefix removed and
            non-numeric values are dropped.
    """

    rows: list[tuple[int, dict[str, float]]] = []
    for line in text.splitlines():
        match = _STEP_LINE.search(line)
        if match is None:
            continue
        metrics: dict[str, float] = {}
        for item in match.group(2).split(" | "):
            key, _, value = item.partition(": ")
            if not key.startswith("train/"):
                continue
            try:
                metrics[key.removeprefix("train/")] = float(value)
            except ValueError:
                continue
        if metrics:
            rows.append((int(match.group(1)), metrics))
    return rows


def summarize_leg(
    text: str, *, round_index: int, name: str, skip_windows: int
) -> LegSummary:
    """Reduce one leg's log to its steady-state numbers.

    Args:
        text (str): Captured stdout/stderr of the leg.
        round_index (int): Round the leg ran in.
        name (str): Leg name.
        skip_windows (int): Leading log windows excluded from the timing.

    Returns:
        LegSummary: The leg's kept windows, losses and peak memory.
    """

    rows = parse_metrics(text)
    timed = [(step, m) for step, m in rows if "sec_per_step" in m][skip_windows:]
    peaks = [m["mem_peak_alloc_gib"] for _, m in rows if "mem_peak_alloc_gib" in m]
    return LegSummary(
        round_index=round_index,
        name=name,
        steps=[step for step, _ in timed],
        sec_per_step=[m["sec_per_step"] for _, m in timed],
        losses={step: m["loss"] for step, m in rows if "loss" in m},
        mem_peak_alloc_gib=max(peaks) if peaks else None,
    )


def format_report(legs: list[LegSummary], *, baseline: str) -> str:
    """Render the per-leg table and the per-arm means.

    Args:
        legs (list[LegSummary]): Every leg of the benchmark.
        baseline (str): Name of the baseline leg; each leg's ratio is taken
            against the baseline leg of its own round.

    Returns:
        str: The report text.
    """

    base_by_round = {leg.round_index: leg for leg in legs if leg.name == baseline}
    lines = [
        f"{'round':>5} {'leg':<12} {'windows':>7} {'mean s/step':>12} "
        f"{'median':>8} {'base/leg':>9} {'peak GiB':>9} {'last loss':>10}"
    ]
    ratios: dict[str, list[float]] = {}
    means: dict[str, list[float]] = {}
    for leg in sorted(legs, key=lambda item: (item.round_index, item.name)):
        base = base_by_round.get(leg.round_index)
        ratio = (
            base.mean / leg.mean
            if base is not None and base.mean and leg.mean
            else None
        )
        if leg.mean is not None:
            means.setdefault(leg.name, []).append(leg.mean)
        if ratio is not None:
            ratios.setdefault(leg.name, []).append(ratio)
        last_loss = leg.losses[max(leg.losses)] if leg.losses else None
        lines.append(
            f"{leg.round_index:>5} {leg.name:<12} {len(leg.sec_per_step):>7} "
            f"{_fmt(leg.mean, '12.4f')} {_fmt(leg.median, '8.4f')} "
            f"{_fmt(ratio, '9.3f')} {_fmt(leg.mem_peak_alloc_gib, '9.2f')} "
            f"{_fmt(last_loss, '10.4f')}"
        )
    lines.append("")
    lines.append(f"{'leg':<12} {'rounds':>6} {'mean s/step':>12} {'base/leg (paired)':>18}")
    for name in sorted(means):
        paired = ratios.get(name, [])
        lines.append(
            f"{name:<12} {len(means[name]):>6} {statistics.fmean(means[name]):>12.4f} "
            f"{_fmt(statistics.fmean(paired) if paired else None, '18.3f')}"
        )
    return "\n".join(lines)


def _fmt(value: float | None, spec: str) -> str:
    """Format a number, or right-align ``n/a`` in the same width.

    Args:
        value (float | None): Value to format.
        spec (str): Format spec whose leading integer is the field width.

    Returns:
        str: The formatted field.
    """

    width = int(spec.split(".")[0])
    return format(value, spec) if value is not None else "n/a".rjust(width)


def main() -> None:
    """Read every ``r<round>_<leg>.log`` of a directory and print the report."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--baseline", default="base")
    parser.add_argument("--skip-windows", type=int, default=1)
    args = parser.parse_args()
    legs: list[LegSummary] = []
    for path in sorted(args.log_dir.glob("r*_*.log")):
        match = _LEG_FILE.search(path.name)
        if match is None:
            continue
        legs.append(
            summarize_leg(
                path.read_text(errors="replace"),
                round_index=int(match.group(1)),
                name=match.group(2),
                skip_windows=args.skip_windows,
            )
        )
    if not legs:
        raise SystemExit(f"no r<round>_<leg>.log files under {args.log_dir}")
    print(format_report(legs, baseline=args.baseline))
    print()
    print("loss by optimizer step:")
    for leg in sorted(legs, key=lambda item: (item.round_index, item.name)):
        trace = "  ".join(f"{step}:{loss:.4f}" for step, loss in sorted(leg.losses.items()))
        print(f"  r{leg.round_index} {leg.name:<12} {trace}")


if __name__ == "__main__":
    main()
