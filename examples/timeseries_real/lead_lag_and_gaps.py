"""Detecting gaps in an event series.

The gap between consecutive events is a lag away. Once you have it, "which customers went
quiet" is a filter, not a procedure — which is the whole reason to keep this in the engine
rather than iterating dates in Python.

    python examples/timeseries_real/lead_lag_and_gaps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderdate", "o_orderkey")

    spaced = orders.with_columns(
        previous_order=col("o_orderdate")
        .shift(1)
        .over(partition_by=["o_custkey"], order_by=["o_orderdate"]),
    ).with_columns(gap_days=col("o_orderdate") - col("previous_order"))

    # The first order for each customer has no gap.
    firsts = spaced.filter(col("previous_order").is_null()).count()
    print("customers with a first order:", firsts)
    assert firsts == orders.n_unique("o_custkey")

    gaps = spaced.filter(col("gap_days").is_not_null())
    stats = gaps.agg(
        shortest=col("gap_days").min(),
        typical=bt.median(col("gap_days")),
        longest=col("gap_days").max(),
    ).to_pydict()
    print(stats)
    assert stats["shortest"][0] >= 0
    assert stats["shortest"][0] <= stats["typical"][0] <= stats["longest"][0]

    # Customers who went quiet for more than a year between orders.
    quiet = gaps.filter(col("gap_days") > 365).select("o_custkey").distinct()
    print("customers with a year-long gap:", quiet.count())
    assert quiet.count() < orders.n_unique("o_custkey")

    # Every gap is non-negative, because the window is ordered by the same column it
    # measures — which is the invariant that catches an unordered window.
    assert gaps.filter(col("gap_days") < 0).count() == 0


if __name__ == "__main__":
    main()
