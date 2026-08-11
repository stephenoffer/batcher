"""TPC-H Q18 — the orders whose total quantity crosses a threshold.

The `IN (SELECT orderkey ... GROUP BY ... HAVING ...)` becomes a semi join against a
pre-aggregated relation. The important part is that the aggregate is computed once and
probed once, rather than re-evaluated per candidate order.

    python examples/tpch/q18_large_volume_customer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")
    customer = tpch("customer")

    threshold = 250

    big_orders = (
        lineitem.group_by("l_orderkey")
        .agg(order_qty=col("l_quantity").sum())
        .filter(col("order_qty") > threshold)
    )

    result = (
        orders.join(big_orders, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(total_qty=col("l_quantity").sum())
        .sort("o_totalprice", "o_orderdate", descending=[True, False])
        .limit(10)
        .to_pydict()
    )

    print(f"{len(result['o_orderkey'])} large orders")
    for name, qty in list(zip(result["c_name"], result["total_qty"], strict=True))[:5]:
        print(f"  {name} qty={qty}")

    # Re-deriving the quantity after the join must agree with the pre-aggregate that
    # selected the order, which is the join-fan-out bug this query invites.
    assert all(qty > threshold for qty in result["total_qty"])
    assert result["o_totalprice"] == sorted(result["o_totalprice"], reverse=True)


if __name__ == "__main__":
    main()
