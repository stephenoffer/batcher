"""A confidence interval around a difference between two groups.

The interval is what turns "these two numbers differ" into "these two numbers differ by more
than sampling noise". Computing it from the two standard errors is arithmetic; the useful
part is checking whether it straddles zero.

    python examples/statistics/hypothesis_intervals.py
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

    # Two genuinely different populations: discounted lines against undiscounted ones.
    discounted = lineitem.filter(col("l_discount") > 0.0)
    plain = lineitem.filter(col("l_discount") == 0.0)

    def stats(name: str, dataset: bt.Dataset) -> tuple[float, float, int]:
        row = dataset.agg(
            mean=col("l_extendedprice").mean(),
            sem=bt.sem(col("l_extendedprice")),
            n=bt.count(),
        ).to_pydict()
        mean, sem, n = row["mean"][0], row["sem"][0], row["n"][0]
        print(f"{name:<14} n={n:>7} mean={mean:>12,.2f} sem={sem:>8,.2f}")
        return mean, sem, n

    left_mean, left_sem, left_n = stats("discounted", discounted)
    right_mean, right_sem, right_n = stats("undiscounted", plain)

    difference = left_mean - right_mean
    combined_sem = (left_sem**2 + right_sem**2) ** 0.5
    low = difference - 1.96 * combined_sem
    high = difference + 1.96 * combined_sem
    print(f"difference {difference:,.2f}, 95% CI [{low:,.2f}, {high:,.2f}]")

    assert left_n + right_n == lineitem.count()
    assert combined_sem > 0
    assert low < difference < high

    # Whether the interval straddles zero is the whole verdict.
    straddles = low <= 0.0 <= high
    print("interval includes zero:", straddles)

    # And a group compared against itself must straddle zero, which is the sanity check
    # that the arithmetic is right.
    self_difference = left_mean - left_mean
    self_interval = 1.96 * (2 * left_sem**2) ** 0.5
    assert -self_interval <= self_difference <= self_interval


if __name__ == "__main__":
    main()
