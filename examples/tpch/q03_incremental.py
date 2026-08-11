"""Q3 recomputed incrementally as new orders arrive.

A top-N over a sum is not incrementally maintainable in general — a new row can promote any
order into the top ten. What *is* maintainable is the per-order revenue underneath it, so the
incremental part is the aggregate and the top-N is recomputed over a much smaller relation.

    python examples/tpch/q03_incremental.py
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
    customer = tpch("customer")
    orders = tpch("orders")
    lineitem = tpch("lineitem")
    cutoff = dt.date(1995, 3, 15)

    base = (
        customer.filter(col("c_mktsegment") == "BUILDING")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .filter(col("o_orderdate") < bt.lit(cutoff))
    )

    def per_order(lines: bt.Dataset) -> bt.Dataset:
        return (
            base.join(lines, left_on="o_orderkey", right_on="l_orderkey")
            .filter(col("l_shipdate") > bt.lit(cutoff))
            .group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        )

    # Two arrivals of line items.
    first_half = lineitem.head(100_000)
    second_half = lineitem.slice(100_000, 100_000)

    partial_one = per_order(first_half)
    partial_two = per_order(second_half)

    # The mergeable step: union the partials and re-aggregate.
    incremental = (
        partial_one.union(partial_two)
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(revenue=col("revenue").sum())
        .sort("revenue", descending=True)
        .limit(10)
        .to_pydict()
    )

    # The one-shot answer over everything.
    one_shot = per_order(lineitem).sort("revenue", descending=True).limit(10).to_pydict()

    print("incremental top keys:", incremental["o_orderkey"][:5])
    print("one-shot top keys:   ", one_shot["o_orderkey"][:5])

    assert incremental["o_orderkey"] == one_shot["o_orderkey"]
    assert all(
        abs(a - b) < 1e-6 for a, b in zip(incremental["revenue"], one_shot["revenue"], strict=True)
    )
    assert incremental["revenue"] == sorted(incremental["revenue"], reverse=True)


if __name__ == "__main__":
    main()
