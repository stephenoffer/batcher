"""TPC-H Q10 — the customers costing you the most in returns.

A four-table join reduced to a top-20. Note the group key: it carries every customer
attribute the report prints, because they are functionally dependent on the key but the
grouping still has to name them. That is the usual reason a TPC-H group key is eight
columns wide rather than one.

    python examples/tpch/q10_returned_item_reporting.py
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
    nation = tpch("nation")

    start = dt.date(1993, 10, 1)
    end = dt.date(1994, 1, 1)

    result = (
        customer.join(orders, left_on="c_custkey", right_on="o_custkey")
        .filter((col("o_orderdate") >= bt.lit(start)) & (col("o_orderdate") < bt.lit(end)))
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .filter(col("l_returnflag") == "R")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .group_by("c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        .sort("revenue", descending=True)
        .limit(20)
        .to_pydict()
    )

    print(f"{len(result['c_custkey'])} customers")
    for name, revenue in list(zip(result["c_name"], result["revenue"], strict=True))[:5]:
        print(f"  {name} {revenue:>14,.2f}")

    assert result["revenue"] == sorted(result["revenue"], reverse=True)
    # The group key includes the customer key, so no customer can appear twice.
    assert len(set(result["c_custkey"])) == len(result["c_custkey"])


if __name__ == "__main__":
    main()
