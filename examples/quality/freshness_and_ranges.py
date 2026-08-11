"""Checking that data is recent and in range.

A freshness check is a range check on a date column. Both are the assertions that catch a
broken upstream job before its output reaches a dashboard, and both are cheap enough to run
on every load.

    python examples/quality/freshness_and_ranges.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    span = orders.agg(
        earliest=col("o_orderdate").min(),
        latest=col("o_orderdate").max(),
    ).to_pydict()
    print("date span:", span["earliest"][0], "to", span["latest"][0])
    assert span["earliest"][0] < span["latest"][0]

    # A range check on the business domain: TPC-H orders live in the 1990s.
    in_window = orders.dq.in_range(
        "o_orderdate", dt.date(1990, 1, 1), dt.date(2000, 1, 1)
    ).validate()
    print(in_window)
    assert in_window.ok

    # A freshness check phrased the same way. This data is historical, so a "last 30 days"
    # rule fails — which is the correct answer and shows the check has teeth.
    cutoff = span["latest"][0] - dt.timedelta(days=30)
    recent = orders.filter(col("o_orderdate") >= bt.lit(cutoff)).count()
    print(f"orders in the last 30 days of the data: {recent}")
    assert 0 < recent < orders.count()

    # Numeric ranges, on a column where the bound is a business rule rather than a type.
    prices = orders.dq.in_range("o_totalprice", 0.0, 1_000_000.0).validate()
    assert prices.ok

    negative = orders.filter(col("o_totalprice") < 0).count()
    print("negative prices:", negative)
    assert negative == 0


if __name__ == "__main__":
    main()
