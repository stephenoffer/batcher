"""Differences and cumulative sums as expressions.

A first difference is a lag subtracted from the value; a cumulative sum is a framed window.
Both are the building blocks of every rate-of-change calculation, and both keep the row
count, so the detail stays available beside the derived column.

    python examples/expr_numeric/cumulative_and_differences.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    daily = (
        tpch("orders")
        .group_by("o_orderdate")
        .agg(revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(40)
    )

    derived = (
        daily.with_columns(
            previous=col("revenue").shift(1).over(order_by=["o_orderdate"]),
            cumulative=col("revenue").sum().over(order_by=["o_orderdate"], frame=(None, 0)),
        )
        .with_columns(
            change=col("revenue") - col("previous"),
            pct=(col("revenue") - col("previous")) / col("previous"),
        )
        .sort("o_orderdate")
    )

    result = derived.to_pydict()
    print([None if v is None else round(v) for v in result["change"][:5]])

    # The first row has no predecessor.
    assert result["change"][0] is None

    # Differences telescope: they sum to the total change over the span.
    changes = [value for value in result["change"] if value is not None]
    telescoped = sum(changes)
    span = result["revenue"][-1] - result["revenue"][0]
    assert abs(telescoped - span) < 1e-3

    # The cumulative sum ends at the total and is non-decreasing for positive values.
    assert abs(result["cumulative"][-1] - sum(result["revenue"])) < 1e-3
    assert result["cumulative"] == sorted(result["cumulative"])

    # The percentage change reconciles with the absolute one.
    for index in range(1, len(result["revenue"])):
        expected = result["change"][index] / result["previous"][index]
        assert abs(result["pct"][index] - expected) < 1e-9
    assert bt is not None


if __name__ == "__main__":
    main()
