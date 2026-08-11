"""Shrinking a wrong result down to the operator that caused it.

The procedure is bisection: run the pipeline one stage at a time and check the row count and
a control total after each. The stage where the number stops making sense is the one to look
at, and finding it is mechanical rather than clever.

    python examples/operations/debugging_a_wrong_result.py
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
    orders = tpch("orders")

    # A pipeline with a real bug in it: the order total is summed after a fan-out join, so
    # it is counted once per line rather than once per order.
    buggy = (
        orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_orderpriority")
        .agg(order_value=col("o_totalprice").sum())
    )

    # Bisect it: check the count and a control total after each stage.
    stages = [
        ("orders", orders),
        ("joined", orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")),
    ]
    for name, stage in stages:
        rows = stage.count()
        total = stage.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        print(f"  {name:<8} {rows:>8} rows, o_totalprice sums to {total:>18,.0f}")

    orders_total = orders.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
    joined_total = (
        orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .agg(t=col("o_totalprice").sum())
        .to_pydict()["t"][0]
    )

    # The join is where the control total stops making sense: it grew.
    assert joined_total > orders_total
    print("the join inflated the order total: that is the bug")

    # The fix: aggregate the many-side first, so the join does not fan out.
    per_order = lineitem.group_by("l_orderkey").agg(lines=bt.count())
    fixed = (
        orders.join(per_order, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_orderpriority")
        .agg(order_value=col("o_totalprice").sum())
    )

    matched_total = (
        orders.join(
            lineitem.select("l_orderkey"),
            left_on="o_orderkey",
            right_on="l_orderkey",
            how="semi",
        )
        .agg(t=col("o_totalprice").sum())
        .to_pydict()["t"][0]
    )
    fixed_total = sum(fixed.to_pydict()["order_value"])
    buggy_total = sum(buggy.to_pydict()["order_value"])
    print(f"buggy {buggy_total:,.0f} vs fixed {fixed_total:,.0f}")
    assert abs(fixed_total - matched_total) < 1e-2
    assert buggy_total > fixed_total


if __name__ == "__main__":
    main()
