"""Time zones: converting between them, and the reporting-boundary trap.

Store UTC, convert at the edge. A daily rollup computed in UTC and labelled as local time
is wrong by up to a day at the boundary, and it is wrong quietly -- the numbers look
plausible, they are just attributed to the wrong day.

    python examples/expressions/temporal_timezones.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import col


def main() -> None:
    # Late-evening UTC timestamps: in New York these are still the *previous* day.
    events = bt.from_pydict(
        {
            "ts_utc": [
                datetime(2024, 3, 1, 2, 30),
                datetime(2024, 3, 1, 23, 30),
                datetime(2024, 3, 2, 1, 15),
            ],
            "n": [1, 2, 4],
        }
    )

    converted = events.with_columns(
        ny=col("ts_utc").dt.convert_timezone("UTC", "America/New_York"),
        tokyo=col("ts_utc").dt.convert_timezone("UTC", "Asia/Tokyo"),
    ).to_pydict()

    print(converted)

    # New York is behind UTC, so the wall clock moves back.
    assert converted["ny"][0] < converted["ts_utc"][0]
    # Tokyo is ahead.
    assert converted["tokyo"][0] > converted["ts_utc"][0]

    # The trap: the same rows bucket into different days depending on the zone.
    utc_days = (
        events.group_by(day=col("ts_utc").dt.truncate("day"))
        .agg(total=col("n").sum())
        .sort("day")
        .to_pydict()
    )
    ny_days = (
        events.group_by(
            day=col("ts_utc").dt.convert_timezone("UTC", "America/New_York").dt.truncate("day")
        )
        .agg(total=col("n").sum())
        .sort("day")
        .to_pydict()
    )
    print("UTC days:", utc_days)
    print("NY days :", ny_days)

    # In UTC: 1 Mar has 1+2=3, 2 Mar has 4.
    assert utc_days["total"] == [3, 4]
    # In New York the first row falls on 29 Feb and the last on 1 Mar, so the buckets
    # differ -- same data, different attribution.
    assert ny_days["total"] != utc_days["total"]

    # Round-tripping returns the original instant.
    back = events.select(
        r=col("ts_utc")
        .dt.convert_timezone("UTC", "Asia/Tokyo")
        .dt.convert_timezone("Asia/Tokyo", "UTC")
    ).to_pydict()
    assert back["r"] == events.to_pydict()["ts_utc"]

    # Formatting for a report happens after the conversion, never before.
    labelled = events.select(
        local=col("ts_utc")
        .dt.convert_timezone("UTC", "America/New_York")
        .dt.strftime("%Y-%m-%d %H:%M")
    ).to_pydict()
    print(labelled)
    assert labelled["local"][0].startswith("2024-02-29")


if __name__ == "__main__":
    main()
