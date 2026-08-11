"""Is this date the start of something? The boundary predicates.

These are the filters behind "month-end close" and "quarterly snapshot". Each is exact:
`is_month_end` knows February has 28 or 29 days, which is the part a hand-written
`day == 30` check gets wrong twice a year.

    python examples/expr_temporal/boundaries_and_flags.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate")

    flagged = orders.select(
        "o_orderdate",
        month_start=col("o_orderdate").dt.is_month_start(),
        month_end=col("o_orderdate").dt.is_month_end(),
        quarter_start=col("o_orderdate").dt.is_quarter_start(),
        quarter_end=col("o_orderdate").dt.is_quarter_end(),
        year_start=col("o_orderdate").dt.is_year_start(),
        year_end=col("o_orderdate").dt.is_year_end(),
        leap=col("o_orderdate").dt.is_leap_year(),
        days_in_month=col("o_orderdate").dt.days_in_month(),
        days_in_year=col("o_orderdate").dt.days_in_year(),
    )

    counts = flagged.agg(
        month_starts=bt.count_if(col("month_start")),
        quarter_starts=bt.count_if(col("quarter_start")),
        year_starts=bt.count_if(col("year_start")),
    ).to_pydict()
    print(counts)

    # A year start is also a quarter start is also a month start, so the counts nest.
    assert counts["year_starts"][0] <= counts["quarter_starts"][0] <= counts["month_starts"][0]

    # Month length is exact, and leap years have 366 days.
    lengths = flagged.select("days_in_month", "days_in_year", "leap").distinct().to_pydict()
    print(sorted(set(lengths["days_in_month"])))
    assert set(lengths["days_in_month"]) <= {28, 29, 30, 31}
    assert set(lengths["days_in_year"]) <= {365, 366}
    assert all(
        (days == 366) == leap
        for days, leap in zip(lengths["days_in_year"], lengths["leap"], strict=True)
    )

    # 29 February only occurs in a leap year.
    february_29 = flagged.filter(col("days_in_month") == 29)
    assert february_29.count() == 0 or all(february_29.to_pydict()["leap"])


if __name__ == "__main__":
    main()
