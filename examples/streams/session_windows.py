"""Session windows: grouping events by a gap rather than by a clock.

A session ends when nothing happens for long enough. That makes the window boundaries
data-dependent, which is exactly the gaps-and-islands pattern: mark the rows that start a
new session, then take a running sum of those marks as the session id.

    python examples/streams/session_windows.py
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

    gap_days = 365

    sessions = (
        orders.with_columns(
            previous=col("o_orderdate")
            .shift(1)
            .over(partition_by=["o_custkey"], order_by=["o_orderdate", "o_orderkey"])
        )
        .with_columns(
            # A new session starts at the first order and after every long gap.
            starts=bt.when(col("previous").is_null())
            .then(1)
            .when((col("o_orderdate") - col("previous")) > gap_days)
            .then(1)
            .otherwise(0)
        )
        .with_columns(
            session=col("starts")
            .sum()
            .over(
                partition_by=["o_custkey"],
                order_by=["o_orderdate", "o_orderkey"],
                frame=(None, 0),
            )
        )
    )

    summary = (
        sessions.group_by("o_custkey", "session")
        .agg(
            orders=bt.count(),
            first=col("o_orderdate").min(),
            last=col("o_orderdate").max(),
        )
        .sort("o_custkey", "session")
    )
    result = summary.to_pydict()
    print(f"{summary.count()} sessions across {orders.n_unique('o_custkey')} customers")

    # Every order belongs to exactly one session.
    assert sum(result["orders"]) == orders.count()

    # Sessions are numbered from 1 within each customer.
    assert min(result["session"]) == 1

    # A session never spans backwards.
    assert all(first <= last for first, last in zip(result["first"], result["last"], strict=True))

    # There are at least as many sessions as customers, because everyone has one.
    assert summary.count() >= orders.n_unique("o_custkey")


if __name__ == "__main__":
    main()
