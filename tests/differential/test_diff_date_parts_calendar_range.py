"""Date-part extraction across the calendar, vs DuckDB.

`test_diff_dt_fields.py` spreads sixty timestamps over 2021, which is enough to show the
fields are wired and nothing about whether the arithmetic behind them is right: no value
before the epoch, no century, no leap day, no null. `year`/`month`/`day`/`quarter`/
`dayofyear`/`dayofweek`/`hour`/`minute`/`second` over a `Date32` or a timezone-naive timestamp
are computed by integer civil-calendar arithmetic (`bc-expr/src/eval/temporal/civil.rs`)
rather than Arrow's chrono-per-value kernel, and every mistake that arithmetic can make lives
exactly where that fixture never goes: floor division before 1970, the 100- and 400-year
leap rules, and the day a leap year's ordinal shifts.

So this sweeps every day from 1582 to 2400, which crosses all of those, with a null in every
thirteenth slot, at a size that spans many morsels so the parallel path does the work.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

_FIRST = dt.date(1582, 10, 15)
_LAST = dt.date(2400, 12, 31)


@pytest.fixture(scope="module")
def calendar():
    days = (_LAST - _FIRST).days + 1
    dates = [None if i % 13 == 5 else _FIRST + dt.timedelta(days=i) for i in range(days)]
    # A time of day that walks every hour, minute and second, and lands on both sides of
    # midnight — so a flooring mistake before 1970 moves the *day*, not just the hour.
    stamps = [
        None
        if d is None
        else dt.datetime(d.year, d.month, d.day) + dt.timedelta(seconds=(i * 7_919) % 86_400)
        for i, d in enumerate(dates)
    ]
    assert days > 250_000, "must span many morsels, or the parallel path is not exercised"
    return pa.table({"d": pa.array(dates, pa.date32()), "ts": pa.array(stamps, pa.timestamp("us"))})


_FIELDS = ("year", "month", "day", "quarter", "dayofyear", "dayofweek")
_TIME_FIELDS = ("hour", "minute", "second")


@pytest.mark.parametrize("column", ["d", "ts"])
def test_calendar_fields_across_centuries(duck, calendar, column):
    duck.register("cal", calendar)
    fields = _FIELDS + (_TIME_FIELDS if column == "ts" else ())
    out = (
        bt.from_arrow(calendar)
        .select(**{f: getattr(col(column).dt, f)() for f in fields})
        .collect()
    )
    expected = duck.sql(f"SELECT {', '.join(f'{f}({column}) AS {f}' for f in fields)} FROM cal")
    assert_same(out, expected)


@pytest.mark.parametrize("column", ["d", "ts"])
def test_a_year_group_key_across_centuries(duck, calendar, column):
    """The shape the kernel exists for: the extracted field is the grouping key."""
    duck.register("cal", calendar)
    out = (
        bt.from_arrow(calendar)
        .group_by(y=col(column).dt.year(), doy=col(column).dt.dayofyear())
        .agg(n=col(column).count())
        .collect()
    )
    expected = duck.sql(
        f"SELECT year({column}) AS y, dayofyear({column}) AS doy, COUNT({column}) AS n "
        f"FROM cal GROUP BY ALL"
    )
    assert out.num_rows > 300_000 // 2, "one group per (year, day) — a near-row-count grouping"
    assert_same(out, expected)
