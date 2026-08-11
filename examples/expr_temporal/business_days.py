"""Weekdays, weekends, and business-day predicates.

Reporting on "working days" only means anything if you say what a working day is. These
predicates cover the calendar half of that; the holiday half is data you have to supply,
which is why there is no built-in for it.

    python examples/expr_temporal/business_days.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_orderdate", "o_totalprice")

    classified = orders.select(
        "o_orderdate",
        "o_totalprice",
        weekday=col("o_orderdate").dt.weekday(),
        day_name=col("o_orderdate").dt.day_name(),
        is_weekend=col("o_orderdate").dt.is_weekend(),
        is_weekday=col("o_orderdate").dt.is_weekday(),
        is_business=col("o_orderdate").dt.is_business_day(),
    )

    sample = classified.head(3).to_pydict()
    print(sample)

    # Weekend and weekday partition the calendar.
    counts = classified.agg(
        weekend=bt.count_if(col("is_weekend")),
        weekday=bt.count_if(col("is_weekday")),
        total=bt.count(),
    ).to_pydict()
    print(counts)
    assert counts["weekend"][0] + counts["weekday"][0] == counts["total"][0]

    # Roughly two sevenths of dates fall at a weekend.
    share = counts["weekend"][0] / counts["total"][0]
    print(f"weekend share {share:.3f}")
    assert 0.25 < share < 0.32

    # Business days are weekdays here, since no holiday calendar was supplied.
    mismatch = classified.filter(col("is_business") != col("is_weekday")).count()
    assert mismatch == 0

    # Revenue by day name, which is the report this was all for.
    by_day = (
        classified.group_by("day_name")
        .agg(revenue=col("o_totalprice").sum(), orders=bt.count())
        .sort("revenue", descending=True)
        .to_pydict()
    )
    print(by_day["day_name"], by_day["orders"])
    assert len(by_day["day_name"]) == 7


if __name__ == "__main__":
    main()
