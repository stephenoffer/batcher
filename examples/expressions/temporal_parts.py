"""Pulling calendar parts out of a timestamp column.

Every accessor here is a projection, so extracting a year to group by costs one pass and
no Python. The SQL spellings (``dayofweek``, ``weekofyear``, ``monthname``) and the
Polars spellings (``weekday``, ``week``, ``month_name``) both exist and agree.

    python examples/expressions/temporal_parts.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import col


def main() -> None:
    events = bt.from_pydict(
        {
            "ts": [
                datetime(2024, 1, 1, 9, 30, 15),
                datetime(2024, 7, 4, 18, 5, 0),
                datetime(2024, 12, 31, 23, 59, 59),
            ],
        }
    )

    parts = events.with_columns(
        year=col("ts").dt.year(),
        month=col("ts").dt.month(),
        day=col("ts").dt.day(),
        hour=col("ts").dt.hour(),
        minute=col("ts").dt.minute(),
        second=col("ts").dt.second(),
        quarter=col("ts").dt.quarter(),
        # Day-of-week and day-of-year, plus their readable names.
        dow=col("ts").dt.dayofweek(),
        doy=col("ts").dt.dayofyear(),
        day_name=col("ts").dt.day_name(),
        month_name=col("ts").dt.month_name(),
        # ISO calendar.
        iso_year=col("ts").dt.iso_year(),
        week=col("ts").dt.week(),
        # Calendar facts about the date.
        leap=col("ts").dt.is_leap_year(),
        month_len=col("ts").dt.days_in_month(),
        year_len=col("ts").dt.days_in_year(),
        # Just the date part.
        date=col("ts").dt.date(),
    )

    result = parts.to_pydict()
    print(result)

    assert result["year"] == [2024, 2024, 2024]
    assert result["month"] == [1, 7, 12]
    assert result["day"] == [1, 4, 31]
    assert result["hour"] == [9, 18, 23]
    assert result["minute"] == [30, 5, 59]
    assert result["second"] == [15, 0, 59]
    assert result["quarter"] == [1, 3, 4]
    assert result["doy"] == [1, 186, 366]
    assert result["day_name"][0] == "Monday"
    assert result["month_name"] == ["January", "July", "December"]
    # 2024 is a leap year, so February has 29 days and the year has 366.
    assert result["leap"] == [True, True, True]
    assert result["year_len"] == [366, 366, 366]
    assert result["month_len"] == [31, 31, 31]

    # The point: group by a derived part without a Python loop.
    by_quarter = events.group_by(q=col("ts").dt.quarter()).agg(n=bt.count()).sort("q").to_pydict()
    assert by_quarter["q"] == [1, 3, 4]
    assert by_quarter["n"] == [1, 1, 1]


if __name__ == "__main__":
    main()
