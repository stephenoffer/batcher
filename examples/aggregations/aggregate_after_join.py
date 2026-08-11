"""Aggregating across a join, and the fan-out that silently doubles your totals.

Joining a one-row-per-order table to a many-rows-per-order table repeats the order's
columns once per line. Summing an order-level column after that join counts it several
times. The fix is to aggregate before joining, or to sum a line-level column instead.

    python examples/aggregations/aggregate_after_join.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    joined = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")

    # The trap: `o_totalprice` is repeated once per line, so this sum is inflated.
    inflated = joined.agg(total=col("o_totalprice").sum()).to_pydict()["total"][0]

    # The truth, over the orders that actually have lines here.
    matched_orders = orders.join(
        lineitem.select("l_orderkey"), left_on="o_orderkey", right_on="l_orderkey", how="semi"
    )
    honest = matched_orders.agg(total=col("o_totalprice").sum()).to_pydict()["total"][0]

    print(f"inflated {inflated:,.0f} vs honest {honest:,.0f}")
    assert inflated > honest

    # Aggregate first, then join: one row per order on both sides, so nothing repeats.
    per_order = lineitem.group_by("l_orderkey").agg(line_revenue=col("l_extendedprice").sum())
    safe = orders.join(per_order, left_on="o_orderkey", right_on="l_orderkey").agg(
        total=col("o_totalprice").sum(),
        lines=bt.count(),
    )
    result = safe.to_pydict()
    print(result)
    assert abs(result["total"][0] - honest) < 1e-3
    assert result["lines"][0] == matched_orders.count()


if __name__ == "__main__":
    main()
