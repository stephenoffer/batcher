"""Aggregating a windowed column: two-stage summaries.

A window adds a column; aggregating that column is a second stage. The order matters — the
window runs over every row and the aggregate collapses them, so putting the window after the
group-by computes something different and usually meaningless.

    python examples/aggregations/aggregate_over_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_linenumber", "l_extendedprice")

    # Stage one: each line's share of its order.
    with_share = lineitem.with_columns(
        order_total=col("l_extendedprice").sum().over(partition_by=["l_orderkey"])
    ).with_columns(share=col("l_extendedprice") / col("order_total"))

    # Stage two: summarize those shares.
    summary = with_share.agg(
        mean_share=col("share").mean(),
        max_share=col("share").max(),
        lines=bt.count(),
    ).to_pydict()
    print({name: round(value[0], 6) for name, value in summary.items()})

    # A share is a proportion, and the mean share is one over the mean lines per order.
    assert 0.0 < summary["mean_share"][0] <= 1.0
    orders = lineitem.n_unique("l_orderkey")
    expected = orders / summary["lines"][0]
    print(f"mean share {summary['mean_share'][0]:.6f}, 1/(lines per order) {expected:.6f}")
    assert abs(summary["mean_share"][0] - expected) < 0.05

    # The dominant line per order: how concentrated is an order's value.
    concentration = (
        with_share.group_by("l_orderkey")
        .agg(top_share=col("share").max(), lines=bt.count())
        .sort("top_share", descending=True)
    )
    result = concentration.head(5).to_pydict()
    print(result["top_share"])
    assert all(0.0 < value <= 1.0 + 1e-9 for value in result["top_share"])

    # A single-line order is entirely one line.
    singles = concentration.filter(col("lines") == 1).to_pydict()
    assert all(abs(value - 1.0) < 1e-9 for value in singles["top_share"])

    # And a multi-line order is not.
    multi = concentration.filter(col("lines") > 1).to_pydict()
    assert all(value < 1.0 for value in multi["top_share"])


if __name__ == "__main__":
    main()
