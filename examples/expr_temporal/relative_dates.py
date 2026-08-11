"""Windows relative to a reference date rather than to today.

`current_date()` makes a query's result depend on when it ran, which is right for a dashboard
and wrong for a test. Taking the reference from the data instead makes the same query
reproducible, and is usually what a backfill needs.

    python examples/expr_temporal/relative_dates.py
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
    orders = tpch("orders").select("o_orderdate", "o_totalprice")

    # The reference taken from the data, not from the clock.
    latest = orders.agg(m=col("o_orderdate").max()).to_pydict()["m"][0]
    print("latest order:", latest)

    windows = {
        "last 30 days": latest - dt.timedelta(days=30),
        "last 90 days": latest - dt.timedelta(days=90),
        "last year": latest - dt.timedelta(days=365),
    }
    counts: dict[str, int] = {}
    for name, start in windows.items():
        counts[name] = orders.filter(col("o_orderdate") > bt.lit(start)).count()
        print(f"  {name:<14} {counts[name]:>7} orders since {start}")

    # The windows nest, so their counts do too.
    assert counts["last 30 days"] <= counts["last 90 days"] <= counts["last year"]
    assert counts["last year"] < orders.count()

    # And the query is reproducible: running it again gives the same numbers.
    again = orders.filter(col("o_orderdate") > bt.lit(latest - dt.timedelta(days=30))).count()
    assert again == counts["last 30 days"]

    # `current_date` is the clock-relative form. Over historical data every row is in the
    # past, so it selects everything — which is the correct answer and shows why a report
    # over old data needs the reference from the data.
    all_past = orders.filter(col("o_orderdate") < bt.current_date()).count()
    print("orders before today:", all_past)
    assert all_past == orders.count()

    # A rolling window anchored on the data, as a report would build it.
    report = (
        orders.filter(col("o_orderdate") > bt.lit(latest - dt.timedelta(days=365)))
        .with_columns(month=col("o_orderdate").dt.truncate("month"))
        .group_by("month")
        .agg(orders=bt.count(), revenue=col("o_totalprice").sum())
        .sort("month")
    )
    assert report.count() <= 13
    assert sum(report.to_pydict()["orders"]) == counts["last year"]


if __name__ == "__main__":
    main()
