"""Time-series patterns: extract date parts, resample, and compute period change.

A small per-region sales series shows the moves a time-series pipeline makes:
pull calendar fields out of a date column with the ``.dt`` accessor, resample to a
monthly bucket with ``.dt.truncate`` + ``group_by``, and compute period-over-period
change with ``lag`` over a time-ordered window. Ordering is by time; the windowed
work runs in the engine.

    python examples/timeseries.py
"""

from __future__ import annotations

import datetime as dt

import batcher as bt
from batcher import col, lag


def main() -> None:
    sales = bt.from_pydict(
        {
            "region": ["us", "us", "us", "eu", "eu", "eu"],
            "ts": [
                dt.date(2024, 1, 5),
                dt.date(2024, 2, 3),
                dt.date(2024, 3, 2),
                dt.date(2024, 1, 9),
                dt.date(2024, 2, 8),
                dt.date(2024, 3, 4),
            ],
            "revenue": [100.0, 150.0, 120.0, 80.0, 90.0, 130.0],
        }
    )

    # Calendar-field extraction with the .dt accessor.
    parts = sales.select(
        "region",
        year=col("ts").dt.year(),
        month=col("ts").dt.month(),
        quarter=col("ts").dt.quarter(),
    )
    parts_result = parts.to_pydict()
    print(parts_result)
    assert parts_result["year"] == [2024] * 6
    assert parts_result["month"] == [1, 2, 3, 1, 2, 3]

    # Resample: truncate each timestamp to the start of its month, then aggregate.
    monthly = (
        sales.with_columns(month=col("ts").dt.truncate("month"))
        .group_by("region", "month")
        .agg(total=col("revenue").sum())
        .sort("region", "month")
    )
    monthly_result = monthly.to_pydict()
    print(monthly_result)
    assert monthly_result["total"] == [80.0, 90.0, 130.0, 100.0, 150.0, 120.0]

    # Period-over-period change: lag the previous in-region value over a time order.
    change = (
        sales.sort("region", "ts")
        .with_columns(
            prev_revenue=lag(col("revenue")).over(partition_by=["region"], order_by=["ts"])
        )
        .with_columns(mom_change=col("revenue") - col("prev_revenue"))
        .select("region", "ts", "revenue", "prev_revenue", "mom_change")
    )
    change_result = change.to_pydict()
    print(change_result)

    # The first row of each region has no prior period, so the change is null.
    assert change_result["prev_revenue"][0] is None  # eu, first month
    assert change_result["prev_revenue"][3] is None  # us, first month
    # eu: 90 - 80 = 10, then 130 - 90 = 40.
    assert change_result["mom_change"][1] == 10.0
    assert change_result["mom_change"][2] == 40.0
    # us: 150 - 100 = 50, then 120 - 150 = -30.
    assert change_result["mom_change"][4] == 50.0
    assert change_result["mom_change"][5] == -30.0


if __name__ == "__main__":
    main()
