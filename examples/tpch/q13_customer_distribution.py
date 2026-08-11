"""TPC-H Q13 — how many customers have how many orders, including the zeros.

The left join is the whole query. Customers with no orders must survive it and land in
the "0 orders" bucket; an inner join drops exactly the population the report is about.
The count is over the order key, not the row, so a null-extended row counts as zero.

    python examples/tpch/q13_customer_distribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer")
    orders = tpch("orders")

    # The excluded orders are the ones whose comment marks them as a special request.
    real_orders = orders.filter(
        ~(col("o_comment").str.contains("special") & col("o_comment").str.contains("requests"))
    )

    per_customer = (
        customer.join(real_orders, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey")
        .agg(order_count=col("o_orderkey").count())
    )

    result = (
        per_customer.group_by("order_count")
        .agg(custdist=bt.count())
        .sort("custdist", "order_count", descending=True)
        .to_pydict()
    )

    print(result["order_count"][:8], result["custdist"][:8])

    # Every customer lands in exactly one bucket.
    assert sum(result["custdist"]) == customer.count()
    # Customers with no matching order are counted, and they are counted as zero.
    assert 0 in result["order_count"]


if __name__ == "__main__":
    main()
