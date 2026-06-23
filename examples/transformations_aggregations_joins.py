"""Transformations, aggregations, and joins — the DataFrame core.

Builds two small in-memory tables and walks the operations a typical pipeline uses:
derive columns, filter, group/aggregate, and join. Every step returns a new lazy
Dataset; ``to_pydict`` triggers execution.

    python examples/transformations_aggregations_joins.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col, count


def main() -> None:
    orders = bt.from_pydict(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [10, 10, 20, 30, 20],
            "amount": [100.0, 50.0, 200.0, 75.0, 25.0],
            "status": ["paid", "paid", "paid", "refunded", "paid"],
        }
    )
    customers = bt.from_pydict(
        {
            "customer_id": [10, 20, 30],
            "region": ["us", "eu", "us"],
        }
    )

    # Derive a column, then keep only paid orders.
    paid = orders.with_columns(net=col("amount") * 0.97).filter(col("status") == "paid")

    # Per-customer revenue.
    by_customer = paid.group_by("customer_id").agg(revenue=col("net").sum(), n_orders=count())

    # Join customer attributes, then roll up to region.
    by_region = (
        by_customer.join(customers, on="customer_id", how="inner")
        .group_by("region")
        .agg(revenue=col("revenue").sum(), customers=count())
        .sort("revenue", descending=True)
    )

    result = by_region.to_pydict()
    print(result)

    # us = customer 10 (100+50)*0.97 + customer 30 (refunded → 0 paid) = 145.5
    # eu = customer 20 (200+25)*0.97 = 218.25
    assert result["region"] == ["eu", "us"]
    assert result["revenue"][0] == 218.25


if __name__ == "__main__":
    main()
