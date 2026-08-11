"""TPC-H Q3 — unshipped orders with the highest revenue.

Three tables, two date predicates pointing in opposite directions, and a top-N. The
interesting property is that the filters are highly selective on *both* sides of the
join, so the order the engine picks matters more here than the aggregation does.

    python examples/tpch/q03_shipping_priority.py
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

    revenue = (col("l_extendedprice") * (1 - col("l_discount"))).alias("revenue")

    result = (
        customer.filter(col("c_mktsegment") == "BUILDING")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .filter(col("o_orderdate") < bt.lit(cutoff))
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .filter(col("l_shipdate") > bt.lit(cutoff))
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(revenue=revenue.sum())
        .sort("revenue", descending=True)
        .limit(10)
        .to_pydict()
    )

    print(f"{len(result['o_orderkey'])} orders")
    for key, value in list(zip(result["o_orderkey"], result["revenue"], strict=True))[:5]:
        print(f"  order {key:>8} {value:>14,.2f}")

    assert result["revenue"] == sorted(result["revenue"], reverse=True)
    assert all(value > 0 for value in result["revenue"])
    # Every order predates the cutoff, which is the half of the predicate that is easy
    # to get backwards.
    assert all(date < cutoff for date in result["o_orderdate"])


if __name__ == "__main__":
    main()
