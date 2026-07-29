"""Bucketing timestamps: truncate to a period, or snap to a period boundary.

``truncate``/``floor`` round a timestamp down to a unit, which is how you build an hourly
or daily rollup key. The ``*_start``/``*_end`` pairs snap to calendar boundaries, which is
what a month-over-month report needs.

    python examples/expressions/temporal_truncation.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import col


def main() -> None:
    hits = bt.from_pydict(
        {
            "ts": [
                datetime(2024, 2, 15, 10, 45, 30),
                datetime(2024, 2, 15, 10, 50, 5),
                datetime(2024, 3, 1, 0, 0, 0),
            ],
            "n": [1, 2, 4],
        }
    )

    bucketed = hits.with_columns(
        hour_bucket=col("ts").dt.truncate("hour"),
        day_bucket=col("ts").dt.truncate("day"),
        # `floor` is the same operation under the Polars name.
        floored=col("ts").dt.floor("hour"),
        # `normalize` drops the time of day (midnight of the same date).
        midnight=col("ts").dt.normalize(),
        # Calendar boundaries.
        month_start=col("ts").dt.month_start(),
        month_end=col("ts").dt.month_end(),
        quarter_start=col("ts").dt.quarter_start(),
        quarter_end=col("ts").dt.quarter_end(),
        year_start=col("ts").dt.year_start(),
        year_end=col("ts").dt.year_end(),
        # Boundary predicates.
        is_month_start=col("ts").dt.is_month_start(),
        is_quarter_start=col("ts").dt.is_quarter_start(),
    )

    result = bucketed.to_pydict()
    print(result)

    assert result["hour_bucket"][0] == datetime(2024, 2, 15, 10, 0, 0)
    assert result["hour_bucket"][0] == result["floored"][0]
    assert result["day_bucket"][0] == datetime(2024, 2, 15, 0, 0, 0)
    assert result["midnight"][0] == datetime(2024, 2, 15, 0, 0, 0)
    assert result["month_start"][0] == datetime(2024, 2, 1, 0, 0, 0)
    # 2024 is a leap year, so February ends on the 29th.
    assert result["month_end"][0].day == 29
    assert result["quarter_start"][0] == datetime(2024, 1, 1, 0, 0, 0)
    assert result["quarter_end"][0].month == 3
    assert result["year_start"][0] == datetime(2024, 1, 1, 0, 0, 0)
    assert result["year_end"][0].month == 12
    assert result["is_month_start"] == [False, False, True]
    assert result["is_quarter_start"] == [False, False, False]

    # The rollup this exists for: two hits in one hour, one in another.
    hourly = (
        hits.group_by(hour=col("ts").dt.truncate("hour"))
        .agg(total=col("n").sum())
        .sort("hour")
        .to_pydict()
    )
    print(hourly)
    assert hourly["total"] == [3, 4]


if __name__ == "__main__":
    main()
