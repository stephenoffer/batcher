"""A cross-tab report built from conditional aggregates rather than a pivot.

A pivot's output schema depends on the data, which makes it a pipeline breaker and makes the
downstream schema unpredictable. When the categories are known, conditional sums give the
same table with a schema you wrote down.

    python examples/aggregations/pivot_style_reports.py
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
    revenue = col("l_extendedprice") * (1 - col("l_discount"))

    # Known categories, so the schema is fixed and written down.
    report = (
        lineitem.group_by("l_shipmode")
        .agg(
            returned=bt.when(col("l_returnflag") == "R").then(revenue).otherwise(0.0).sum(),
            accepted=bt.when(col("l_returnflag") == "A").then(revenue).otherwise(0.0).sum(),
            none=bt.when(col("l_returnflag") == "N").then(revenue).otherwise(0.0).sum(),
            total=revenue.sum(),
        )
        .sort("l_shipmode")
    )
    result = report.to_pydict()
    print(report.columns)
    for index, mode in enumerate(result["l_shipmode"]):
        print(
            f"  {mode:<9} R={result['returned'][index]:>14,.0f} "
            f"A={result['accepted'][index]:>14,.0f} N={result['none'][index]:>14,.0f}"
        )

    # The schema is fixed regardless of what the data contains.
    assert report.columns == ["l_shipmode", "returned", "accepted", "none", "total"]

    # The three categories partition the total exactly.
    for index in range(len(result["l_shipmode"])):
        parts = result["returned"][index] + result["accepted"][index] + result["none"][index]
        assert abs(parts - result["total"][index]) < 1e-3

    # The pivot version gives the same numbers with a data-dependent schema.
    pivoted = (
        lineitem.with_columns(revenue=revenue)
        .pivot(index=["l_shipmode"], on="l_returnflag", values="revenue", aggregate="sum")
        .sort("l_shipmode")
    )
    pivot_result = pivoted.to_pydict()
    print("pivot columns:", pivoted.columns)
    assert set(pivoted.columns) - {"l_shipmode"} <= {"A", "N", "R"}
    assert all(
        abs(a - b) < 1e-3 for a, b in zip(result["returned"], pivot_result["R"], strict=True)
    )


if __name__ == "__main__":
    main()
