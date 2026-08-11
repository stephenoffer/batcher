"""Turning a report into a tidy table, and back.

A report is wide: one column per metric. A tidy table is long: one row per metric. Long is
what a group-by wants, so unpivoting before aggregating is usually cheaper than writing one
aggregate per column.

    python examples/relational/wide_to_long_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    wide = (
        tpch("lineitem")
        .group_by("l_shipmode")
        .agg(
            quantity=col("l_quantity").sum(),
            revenue=col("l_extendedprice").sum(),
            discount=col("l_discount").sum(),
        )
        .sort("l_shipmode")
    )
    print(wide.to_pydict()["l_shipmode"])

    long = wide.unpivot(index=["l_shipmode"], variable_name="metric", value_name="value").sort(
        "l_shipmode", "metric"
    )

    result = long.to_pydict()
    print(result["metric"][:6])

    # Three metrics per ship mode.
    assert long.count() == wide.count() * 3
    assert set(result["metric"]) == {"quantity", "revenue", "discount"}

    # One aggregate over the long form replaces three over the wide one.
    totals = long.group_by("metric").agg(total=col("value").sum()).sort("metric").to_pydict()
    print(totals)
    wide_totals = wide.agg(
        quantity=col("quantity").sum(),
        revenue=col("revenue").sum(),
        discount=col("discount").sum(),
    ).to_pydict()
    for index, metric in enumerate(totals["metric"]):
        assert abs(totals["total"][index] - wide_totals[metric][0]) < 1e-3

    # And pivoting back returns the original shape.
    back = long.pivot(index=["l_shipmode"], on="metric", values="value", aggregate="sum").sort(
        "l_shipmode"
    )
    assert back.count() == wide.count()
    assert set(back.columns) == set(wide.columns)
    assert bt is not None


if __name__ == "__main__":
    main()
