"""Durations between two timestamp columns, and shifting a timestamp.

``*_between`` gives a whole-unit difference between two columns, which is how you compute
an age, a lead time, or a session length. ``offset_by`` shifts by a duration string, which
is how you build a "30 days ago" cutoff without leaving the expression API.

Read the direction carefully: ``later.days_between(earlier)`` counts *from* the argument
*to* the receiver, so the later timestamp is the one you call it on. Reversing the two
gives a negative answer.

    python examples/expressions/temporal_differences.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import col


def main() -> None:
    orders = bt.from_pydict(
        {
            "placed": [
                datetime(2024, 1, 1, 8, 0, 0),
                datetime(2024, 1, 10, 12, 0, 0),
            ],
            "shipped": [
                datetime(2024, 1, 3, 20, 30, 0),
                datetime(2024, 1, 10, 15, 45, 0),
            ],
        }
    )

    timed = orders.with_columns(
        # Read as: from `placed` up to `shipped`. Call it on the later column.
        seconds=col("shipped").dt.seconds_between(col("placed")),
        minutes=col("shipped").dt.minutes_between(col("placed")),
        hours=col("shipped").dt.hours_between(col("placed")),
        days=col("shipped").dt.days_between(col("placed")),
        weeks=col("shipped").dt.weeks_between(col("placed")),
        # Shift a timestamp by a duration string.
        due=col("placed").dt.offset_by("3d"),
        reminder=col("placed").dt.offset_by("-1d"),
        # Epoch conversions, for interop with systems that speak integers.
        epoch_s=col("placed").dt.epoch(),
        epoch_ms=col("placed").dt.epoch_ms(),
    )

    result = timed.to_pydict()
    print(result)

    # Order 1: 1 Jan 08:00 -> 3 Jan 20:30 is 2 days 12.5 hours.
    assert result["days"][0] == 2
    assert result["hours"][0] == 60
    assert result["minutes"][0] == 60 * 60 + 30
    assert result["seconds"][0] == result["minutes"][0] * 60
    assert result["weeks"][0] == 0
    # Order 2 shipped the same day.
    assert result["days"][1] == 0
    assert result["hours"][1] == 3

    assert result["due"][0] == datetime(2024, 1, 4, 8, 0, 0)
    assert result["reminder"][0] == datetime(2023, 12, 31, 8, 0, 0)
    # Milliseconds are seconds scaled by 1000.
    assert result["epoch_ms"][0] == result["epoch_s"][0] * 1000

    # The SLA check this exists for: flag anything that took more than two whole days.
    late = orders.filter(col("shipped").dt.days_between(col("placed")) > 2).to_pydict()
    assert len(late["placed"]) == 0


if __name__ == "__main__":
    main()
