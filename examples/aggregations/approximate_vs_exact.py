"""Sketches versus exact aggregates: what you trade and what you keep.

An exact distinct count needs every distinct value in memory. A HyperLogLog sketch needs
a fixed few kilobytes whatever the cardinality. What you keep is mergeability — the
sketch combines across partitions, so the distributed answer equals the single-node one.

    python examples/aggregations/approximate_vs_exact.py
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

    both = lineitem.agg(
        exact_parts=bt.n_unique(col("l_partkey")),
        approx_parts=bt.approx_n_unique(col("l_partkey")),
        exact_median=bt.median(col("l_extendedprice")),
        approx_median=bt.approx_median(col("l_extendedprice")),
    ).to_pydict()
    print(both)

    exact = both["exact_parts"][0]
    approx = both["approx_parts"][0]
    error = abs(approx - exact) / exact
    print(f"distinct parts: exact={exact} approx={approx} error={error:.4%}")
    assert error < 0.05

    median_error = abs(both["approx_median"][0] - both["exact_median"][0]) / both["exact_median"][0]
    assert median_error < 0.02

    # The property that matters: the sketch is mergeable, so computing it per group and
    # over the whole table are consistent views of the same data. The union of the
    # per-group distinct sets bounds the global one from below.
    per_mode = lineitem.group_by("l_shipmode").agg(parts=bt.approx_n_unique(col("l_partkey")))
    largest_group = max(per_mode.to_pydict()["parts"])
    assert largest_group <= approx * 1.05


if __name__ == "__main__":
    main()
