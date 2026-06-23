"""Window functions: per-partition aggregates and ranking.

Window functions compute a value for each row from a window of related rows without
collapsing them (unlike group_by). An aggregate gets an ``.over(partition_by=...)``;
ranking functions like ``bt.row_number()`` take ``order_by`` too.

    python examples/window_functions.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    sales = bt.from_pydict(
        {
            "region": ["us", "us", "us", "eu", "eu"],
            "rep": ["a", "b", "c", "d", "e"],
            "amount": [100, 300, 200, 50, 75],
        }
    )

    ranked = sales.with_columns(
        # Each rep's share of their region's total (aggregate over a partition).
        region_total=col("amount").sum().over(partition_by=["region"]),
        # Rank within region, highest amount first (order_by takes (col, descending)).
        rank_in_region=bt.row_number().over(partition_by=["region"], order_by=[("amount", True)]),
    ).sort("region", "rank_in_region")

    result = ranked.to_pydict()
    print(result)

    rows = [dict(zip(result, vals, strict=True)) for vals in zip(*result.values(), strict=True)]
    # Every us row carries the same region total (600); eu's is 125.
    us_totals = {r["region_total"] for r in rows if r["region"] == "us"}
    assert us_totals == {600}
    # The top-ranked rep in each region is the highest earner.
    top = {r["region"]: r["rep"] for r in rows if r["rank_in_region"] == 1}
    assert top == {"us": "b", "eu": "e"}


if __name__ == "__main__":
    main()
