"""TPC-H Q15 — the supplier with the highest quarterly revenue, via a reused subquery.

The SQL defines a view and references it twice: once to find the maximum, once to find
who achieved it. Here the intermediate Dataset is `cache()`d, so the second reference
reads the computed result instead of recomputing the aggregation.

    python examples/tpch/q15_top_supplier.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    supplier = tpch("supplier")

    start = dt.date(1996, 1, 1)
    end = dt.date(1996, 4, 1)

    revenue = (
        lineitem.filter((col("l_shipdate") >= bt.lit(start)) & (col("l_shipdate") < bt.lit(end)))
        .group_by("l_suppkey")
        .agg(total_revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        .cache()
    )

    best = revenue.agg(peak=col("total_revenue").max()).to_pydict()["peak"][0]

    result = (
        revenue.filter(col("total_revenue") == best)
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        # `left_on`/`right_on` keeps the left key and drops the right one, so the
        # supplier key is still spelled `l_suppkey` after the join.
        .select("l_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("l_suppkey")
        .to_pydict()
    )

    print(result["s_name"], [round(value, 2) for value in result["total_revenue"]])

    assert len(result["l_suppkey"]) >= 1
    # Ties are legal and must all be returned, so check the value rather than the count.
    assert all(value == best for value in result["total_revenue"])
    assert best == max(revenue.to_pydict()["total_revenue"])


if __name__ == "__main__":
    main()
