"""Joining a table to itself, and keeping the two sides apart.

A self-join needs every column disambiguated or the result is unreadable. Renaming one
side before the join is clearer than relying on a suffix, and it makes the join condition
say what it means.

    python examples/joins/self_joins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_custkey", "o_orderdate", "o_totalprice")

    # Pairs of orders placed by the same customer on the same day.
    right_side = orders.select(
        col("o_custkey").alias("other_cust"),
        col("o_orderdate").alias("other_date"),
        col("o_orderkey").alias("other_key"),
        col("o_totalprice").alias("other_price"),
    )

    pairs = (
        orders.join(
            right_side,
            left_on=["o_custkey", "o_orderdate"],
            right_on=["other_cust", "other_date"],
        )
        # Drop the self-match, and keep each unordered pair once.
        .filter(col("o_orderkey") < col("other_key"))
    )

    print("same-day order pairs:", pairs.count())
    result = pairs.head(5).to_pydict()
    print(result["o_orderkey"], result["other_key"])

    # No row is paired with itself, and each pair appears in one direction only.
    assert all(
        left < right for left, right in zip(result["o_orderkey"], result["other_key"], strict=True)
    )

    # Cross-check: a customer with n orders on one day contributes n*(n-1)/2 pairs.
    per_day = orders.group_by("o_custkey", "o_orderdate").agg(n=bt.count())
    expected = sum(n * (n - 1) // 2 for n in per_day.to_pydict()["n"])
    assert pairs.count() == expected


if __name__ == "__main__":
    main()
