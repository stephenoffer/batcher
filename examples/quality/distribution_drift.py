"""Detecting that today's data does not look like yesterday's.

Drift is a comparison of two distributions, and the cheapest useful version is a comparison
of their summary statistics. It catches the failures that a null check does not: a unit
change, a truncated upstream feed, a new default silently applied.

    python examples/quality/distribution_drift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # Two "days" of data from the same distribution.
    baseline = lineitem.head(100_000)
    today = lineitem.slice(100_000, 100_000)

    def profile(dataset: bt.Dataset) -> dict[str, float]:
        row = dataset.agg(
            mean=col("l_quantity").mean(),
            std=bt.std(col("l_quantity")),
            median=bt.median(col("l_quantity")),
            null_rate=bt.null_rate(col("l_quantity")),
        ).to_pydict()
        return {name: value[0] for name, value in row.items()}

    before = profile(baseline)
    after = profile(today)
    print("baseline:", {k: round(v, 4) for k, v in before.items()})
    print("today:   ", {k: round(v, 4) for k, v in after.items()})

    # Same distribution: the means agree closely.
    assert abs(before["mean"] - after["mean"]) / before["mean"] < 0.05
    assert before["null_rate"] == after["null_rate"] == 0.0

    # Now a genuinely drifted batch: the units changed.
    drifted = profile(today.with_columns(l_quantity=col("l_quantity") * 10))
    shift = abs(before["mean"] - drifted["mean"]) / before["mean"]
    print(f"drifted mean shift: {shift:.2%}")
    assert shift > 1.0

    # A gate, phrased as a threshold on the shift.
    def within_tolerance(reference: dict[str, float], candidate: dict[str, float]) -> bool:
        return abs(reference["mean"] - candidate["mean"]) / reference["mean"] < 0.1

    assert within_tolerance(before, after)
    assert not within_tolerance(before, drifted)


if __name__ == "__main__":
    main()
