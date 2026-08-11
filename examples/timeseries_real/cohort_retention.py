"""A cohort table: grouping customers by when they first appeared.

The first-order date per customer is an aggregate joined back, not a window: `min` over a
window is not defined for a Date32 column, so the grouped form is both the portable
spelling and the clearer one. From there the cohort table is one more group-by.

    python examples/timeseries_real/cohort_retention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_custkey", "o_orderkey", "o_orderdate")

    # Each customer's first order date, computed once and joined back.
    first_orders = orders.group_by("o_custkey").agg(first_order=col("o_orderdate").min())
    with_cohort = orders.join(first_orders, on="o_custkey").with_columns(
        cohort=col("first_order").dt.truncate("year"),
        period=col("o_orderdate").dt.truncate("year"),
    )

    table = (
        with_cohort.group_by("cohort", "period")
        .agg(customers=col("o_custkey").n_unique(), orders=bt.count())
        .sort("cohort", "period")
    )
    result = table.to_pydict()
    for cohort, period, customers in list(
        zip(result["cohort"], result["period"], result["customers"], strict=True)
    )[:8]:
        print(f"  cohort {cohort.year} in {period.year}: {customers} customers")

    # A cohort cannot be active before it exists.
    assert all(
        period >= cohort for cohort, period in zip(result["cohort"], result["period"], strict=True)
    )

    # Every order lands in exactly one cell.
    assert sum(result["orders"]) == orders.count()

    # The cohort's own first period holds all of its customers.
    first_period = [
        customers
        for cohort, period, customers in zip(
            result["cohort"], result["period"], result["customers"], strict=True
        )
        if cohort == period
    ]
    assert sum(first_period) == orders.n_unique("o_custkey")


if __name__ == "__main__":
    main()
