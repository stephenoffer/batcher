"""Splitting an ordered partition into equal buckets with ntile.

`ntile(4)` labels each row with its quartile. Unlike a quantile *value*, this is a label
per row, so it composes with a group-by afterwards — which is how you get "average
revenue per quartile" in one pass.

    python examples/windows/ntile_and_quartiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_nationkey", "c_acctbal")

    bucketed = customer.with_columns(
        quartile=bt.ntile(4).over(order_by=[("c_acctbal", False)]),
        decile=bt.ntile(10).over(order_by=[("c_acctbal", False)]),
    )

    per_quartile = (
        bucketed.group_by("quartile")
        .agg(customers=bt.count(), low=col("c_acctbal").min(), high=col("c_acctbal").max())
        .sort("quartile")
        .to_pydict()
    )
    print(per_quartile)

    assert per_quartile["quartile"] == [1, 2, 3, 4]
    # Buckets differ in size by at most one row.
    assert max(per_quartile["customers"]) - min(per_quartile["customers"]) <= 1
    # And they are ordered: every value in bucket n is below every value in bucket n+1.
    assert all(
        high <= next_low
        for high, next_low in zip(per_quartile["high"], per_quartile["low"][1:], strict=False)
    )

    # Ten deciles over the same order.
    assert bucketed.n_unique("decile") == 10


if __name__ == "__main__":
    main()
