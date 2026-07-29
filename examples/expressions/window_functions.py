"""Window functions: per-row values computed from a window of related rows.

The difference from ``group_by`` is that the row count is preserved. That is what you want
for a running total, a rank within a partition, or a comparison against the previous row.

    python examples/expressions/window_functions.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    sales = bt.from_pydict(
        {
            "region": ["us", "us", "us", "eu", "eu"],
            "day": [1, 2, 3, 1, 2],
            "amount": [10, 30, 20, 50, 40],
        }
    )

    windowed = sales.with_columns(
        # An aggregate over the partition, broadcast back to every row.
        region_total=col("amount").sum().over(partition_by=["region"]),
        region_max=col("amount").max().over(partition_by=["region"]),
        # Ranking within the partition, highest first.
        rank=bt.row_number().over(partition_by=["region"], order_by=[("amount", True)]),
        dense=bt.dense_rank().over(partition_by=["region"], order_by=[("amount", True)]),
        # A running total in day order.
        running=col("amount").cum_sum(partition_by=["region"], order_by=["day"]),
        # The previous day's value, and the change from it.
        prev=col("amount").shift(1).over(partition_by=["region"], order_by=["day"]),
        delta=col("amount").diff(1, partition_by=["region"], order_by=["day"]),
    ).sort("region", "day")

    result = windowed.to_pydict()
    print(result)

    # The row count is unchanged: five in, five out.
    assert len(result["amount"]) == 5

    # Every us row carries the same partition total.
    pairs = list(zip(result["region"], result["region_total"], strict=True))
    assert {t for r, t in pairs if r == "us"} == {60}
    assert {t for r, t in pairs if r == "eu"} == {90}

    # Rank 1 is the biggest amount in each region.
    top = {
        r: a
        for r, a, k in zip(result["region"], result["amount"], result["rank"], strict=True)
        if k == 1
    }
    assert top == {"us": 30, "eu": 50}

    # The running total accumulates in day order within a region.
    us_running = [v for r, v in zip(result["region"], result["running"], strict=True) if r == "us"]
    assert us_running == [10, 40, 60]

    # The first row of each partition has no previous row.
    us_prev = [v for r, v in zip(result["region"], result["prev"], strict=True) if r == "us"]
    assert us_prev[0] is None
    assert us_prev[1] == 10

    # A share-of-total column, the classic use for a broadcast aggregate.
    shares = sales.select(
        region=col("region"),
        share=col("amount") / col("amount").sum().over(partition_by=["region"]),
    ).to_pydict()
    print(shares)
    us_shares = [s for r, s in zip(shares["region"], shares["share"], strict=True) if r == "us"]
    assert abs(sum(us_shares) - 1.0) < 1e-9


if __name__ == "__main__":
    main()
