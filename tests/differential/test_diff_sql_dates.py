"""Date arithmetic: INTERVAL (DAY/WEEK), date_add/date_diff, CAST AS DATE/TIMESTAMP."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "id": [1, 2, 3],
            "d": pa.array(
                [dt.date(2021, 1, 1), dt.date(2021, 6, 15), dt.date(2020, 12, 25)], pa.date32()
            ),
            "d2": pa.array(
                [dt.date(2021, 1, 10), dt.date(2021, 7, 1), dt.date(2021, 1, 1)], pa.date32()
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        # date ± INTERVAL returns a DATE here (DuckDB promotes to timestamp); a CAST
        # normalizes both engines to DATE so the comparison is value-for-value.
        "SELECT id, CAST(d + INTERVAL 5 DAY AS DATE) r FROM t",
        "SELECT id, CAST(d - INTERVAL 10 DAY AS DATE) r FROM t",
        "SELECT id, CAST(d + INTERVAL 2 WEEK AS DATE) r FROM t",
        "SELECT id, CAST(date_add(d, INTERVAL 3 DAY) AS DATE) r FROM t",
    ],
)
def test_date_interval(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, date_diff('day', d, d2) n FROM t",
        "SELECT id FROM t WHERE d + INTERVAL 7 DAY > DATE '2021-01-05'",
        "SELECT id, CAST(d AS DATE) r FROM t",
        "SELECT CAST('2021-03-15' AS DATE) r",
        "SELECT CAST('2021-03-15 10:30:00' AS TIMESTAMP) r",
    ],
)
def test_date_functions(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.fixture
def ts(duck):
    tbl = pa.table(
        {
            "id": [1, 2, 3],
            "ev": pa.array(
                [
                    dt.datetime(2013, 7, 15, 12, 40, 37),
                    dt.datetime(2013, 7, 15, 12, 41, 5),
                    dt.datetime(2020, 12, 25, 23, 59, 59),
                ],
                pa.timestamp("us"),
            ),
        }
    )
    duck.register("ts", tbl)
    return tbl


@pytest.mark.parametrize(
    "unit",
    ["minute", "hour", "day", "month", "year", "second"],
)
def test_date_trunc(duck, ts, unit):
    """DATE_TRUNC('<unit>', ts) — the ClickBench Q42 shape — matches DuckDB."""
    q = f"SELECT id, DATE_TRUNC('{unit}', ev) m FROM ts ORDER BY id"
    assert_same(bt.sql(q, ts=ts).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        # TIMESTAMP ± INTERVAL DAY/WEEK must add exact days to the *microsecond*
        # instant (keeping the time-of-day), not route the value through a Date32
        # epoch-day cast. Regression: the DAY/WEEK branch cast the operand to int64
        # and back to DATE, so a timestamp (µs since epoch) either crashed the
        # Date32 cast or produced a garbage date. DuckDB keeps the time component.
        "SELECT id, ev + INTERVAL 5 DAY r FROM ts ORDER BY id",
        "SELECT id, ev - INTERVAL 10 DAY r FROM ts ORDER BY id",
        "SELECT id, ev + INTERVAL 2 WEEK r FROM ts ORDER BY id",
        "SELECT id, ev + INTERVAL 1 MONTH r FROM ts ORDER BY id",
        "SELECT id, ev + INTERVAL 1 YEAR r FROM ts ORDER BY id",
        "SELECT id, date_add(ev, INTERVAL 3 DAY) r FROM ts ORDER BY id",
    ],
)
def test_timestamp_interval(duck, ts, q):
    assert_same(bt.sql(q, ts=ts).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        # MONTH/YEAR calendar arithmetic now works. Batcher keeps the DATE type
        # (consistent with day intervals); DuckDB promotes date+month/year to
        # TIMESTAMP, so the queries CAST back to DATE to compare the *values*
        # (incl. end-of-month clamping: Jan 31 + 1 month → Feb 28).
        "SELECT CAST(d + INTERVAL 1 MONTH AS DATE) m FROM t",
        "SELECT CAST(d + INTERVAL 1 YEAR AS DATE) y FROM t",
        "SELECT CAST(d - INTERVAL 2 MONTH AS DATE) m FROM t",
        "SELECT CAST(CAST('2021-01-31' AS DATE) + INTERVAL 1 MONTH AS DATE) m",
    ],
)
def test_month_year_interval(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
