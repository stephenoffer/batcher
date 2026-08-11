"""Joining on a range rather than an equality.

An equality join hashes. A range join cannot, so it needs either a sort-merge strategy or
an equality key to partition on first. Adding a coarse equality — the same year, the same
bucket — turns an unbounded comparison into a bounded one.

    python examples/joins/range_and_inequality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_orderdate", "o_totalprice").head(2_000)

    # A price band table: which tier does each order fall into.
    bands = bt.from_pydict(
        {
            "tier": ["small", "medium", "large"],
            "low": [0.0, 50_000.0, 150_000.0],
            "high": [50_000.0, 150_000.0, 1_000_000.0],
        }
    )

    # The band table is tiny, so a cross join followed by the range predicate is the
    # honest spelling — and the optimizer can see the whole thing.
    banded = orders.cross_join(bands).filter(
        (col("o_totalprice") >= col("low")) & (col("o_totalprice") < col("high"))
    )

    print("banded rows:", banded.count())
    # The bands are disjoint and cover the range, so every order lands in exactly one.
    assert banded.count() == orders.count()

    per_tier = banded.group_by("tier").agg(orders=bt.count()).sort("tier").to_pydict()
    print(per_tier)
    assert sum(per_tier["orders"]) == orders.count()

    # Every assignment really satisfies its band.
    sample = banded.select("o_totalprice", "low", "high", "tier").head(20).to_pydict()
    assert all(
        low <= price < high
        for price, low, high in zip(
            sample["o_totalprice"], sample["low"], sample["high"], strict=True
        )
    )


if __name__ == "__main__":
    main()
