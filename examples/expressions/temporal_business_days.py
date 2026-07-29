"""Weekend and business-day predicates, and formatting a timestamp for output.

Reports almost always want weekdays only, and almost always want a string at the end.
Both are expressions, so the filter pushes down toward the scan and the formatting happens
in Rust rather than in a Python ``strftime`` loop.

    python examples/expressions/temporal_business_days.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import col


def main() -> None:
    # 2024-03-01 is a Friday, so the 2nd/3rd are the weekend.
    week = bt.from_pydict(
        {
            "ts": [
                datetime(2024, 3, 1, 9, 0),
                datetime(2024, 3, 2, 9, 0),
                datetime(2024, 3, 3, 9, 0),
                datetime(2024, 3, 4, 9, 0),
            ],
            "hits": [10, 2, 1, 12],
        }
    )

    marked = week.with_columns(
        day_name=col("ts").dt.day_name(),
        weekend=col("ts").dt.is_weekend(),
        weekday=col("ts").dt.is_weekday(),
        business=col("ts").dt.is_business_day(),
        # Formatting: `strftime` takes a format string, `to_string` defaults to ISO 8601.
        stamp=col("ts").dt.strftime("%Y-%m-%d"),
        pretty=col("ts").dt.strftime("%a %d %b %Y"),
        iso=col("ts").dt.to_string(),
    )

    result = marked.to_pydict()
    print(result)

    assert result["day_name"] == ["Friday", "Saturday", "Sunday", "Monday"]
    assert result["weekend"] == [False, True, True, False]
    assert result["weekday"] == [True, False, False, True]
    # No holiday calendar is applied, so a business day is simply a weekday.
    assert result["business"] == result["weekday"]
    assert result["stamp"] == ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"]
    assert result["pretty"][0] == "Fri 01 Mar 2024"
    assert result["iso"][0].startswith("2024-03-01T09:00")

    # The report this exists for: weekday traffic only.
    weekdays = (
        week.filter(col("ts").dt.is_weekday())
        .select(day=col("ts").dt.day_name(), hits=col("hits"))
        .to_pydict()
    )
    print(weekdays)
    assert weekdays["day"] == ["Friday", "Monday"]
    assert weekdays["hits"] == [10, 12]


if __name__ == "__main__":
    main()
