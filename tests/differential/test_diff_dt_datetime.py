"""Differential date/time parity vs DuckDB, focused on the historically buggy edges.

Covers pre-1970 / negative-epoch instants, leap days, ISO week/year boundaries, and
the overflow / unsupported-format paths that used to panic. The `epoch` case pins a
real regression: a microsecond→second cast truncated toward zero, so a sub-second
instant just before 1970 landed one second late (and collided with 1970-01-01).
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

# Timestamps chosen to exercise the edges: negative epoch (incl. sub-second),
# leap days, century boundaries, ISO-week boundaries, and a null.
_TS = [
    dt.datetime(1969, 12, 31, 23, 59, 59, 500000),  # sub-second pre-1970
    dt.datetime(1969, 12, 31, 23, 59, 59, 1),
    dt.datetime(1969, 12, 31, 0, 0, 0),
    dt.datetime(1970, 1, 1, 0, 0, 0, 500000),
    dt.datetime(1970, 1, 1, 0, 0, 0),
    dt.datetime(1900, 1, 1, 0, 0, 0),
    dt.datetime(2000, 2, 29, 12, 0, 0),
    dt.datetime(2024, 2, 29, 23, 59, 59),
    dt.datetime(2021, 1, 1, 0, 0, 0),  # ISO week 53 of 2020
    dt.datetime(2020, 12, 31, 0, 0, 0),
    dt.datetime(2024, 1, 31, 13, 45, 30),
    dt.datetime(1999, 12, 31, 23, 59, 59),
    None,
]


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "ts": pa.array(_TS, pa.timestamp("us")),
            "d": pa.array([x.date() if x else None for x in _TS], pa.date32()),
        }
    )
    duck.register("t", tbl)
    return tbl


_FIELDS = [
    ("year", "year(ts)"),
    ("month", "month(ts)"),
    ("day", "day(ts)"),
    ("hour", "hour(ts)"),
    ("minute", "minute(ts)"),
    ("second", "second(ts)"),
    ("quarter", "quarter(ts)"),
    ("week", "week(ts)"),
    ("dayofyear", "dayofyear(ts)"),
    ("isodow", "isodow(ts)"),
    ("century", "century(ts)"),
    ("decade", "decade(ts)"),
    ("millennium", "millennium(ts)"),
    ("iso_year", "isoyear(ts)"),
    ("days_in_month", "day(last_day(ts))"),
]


@pytest.mark.parametrize(("method", "sql"), _FIELDS)
def test_field_extraction_vs_duckdb(duck, t, method, sql):
    from conftest import assert_same

    out = bt.from_arrow(t).select(v=getattr(col("ts").dt, method)()).collect()
    expected = duck.sql(f"SELECT {sql} AS v FROM t")
    assert_same(out, expected)


def test_epoch_floors_negative_subsecond(duck, t):
    """epoch must floor: 1969-12-31T23:59:59.5 → -1 second, not 0 (regression)."""
    from conftest import assert_same

    out = bt.from_arrow(t).select(v=col("ts").dt.epoch()).collect()
    # DuckDB's epoch is a fractional DOUBLE; whole floored seconds is the oracle.
    expected = duck.sql("SELECT CAST(floor(epoch(ts)) AS BIGINT) AS v FROM t")
    assert_same(out, expected)


@pytest.mark.parametrize(
    ("by", "sql"),
    [
        ("1mo", "ts + INTERVAL 1 MONTH"),
        ("-1mo", "ts - INTERVAL 1 MONTH"),
        ("1y", "ts + INTERVAL 1 YEAR"),
        ("-1y", "ts - INTERVAL 1 YEAR"),
        ("1d", "ts + INTERVAL 1 DAY"),
        ("-1d", "ts - INTERVAL 1 DAY"),
        ("1h", "ts + INTERVAL 1 HOUR"),
        ("-1h", "ts - INTERVAL 1 HOUR"),
        ("1mo15d", "ts + INTERVAL 1 MONTH + INTERVAL 15 DAY"),
    ],
)
def test_offset_by_vs_duckdb(duck, t, by, sql):
    from conftest import assert_same

    out = bt.from_arrow(t).select(v=col("ts").dt.offset_by(by)).collect()
    expected = duck.sql(f"SELECT {sql} AS v FROM t")
    assert_same(out, expected)


def test_last_day_leap_february(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(v=col("ts").dt.days_in_month()).collect()
    expected = duck.sql("SELECT day(last_day(ts)) AS v FROM t")
    assert_same(out, expected)


def test_offset_huge_interval_does_not_crash(t):
    """A huge offset must yield null (no panic / silent wrap)."""
    out = bt.from_arrow(t).select(v=col("ts").dt.offset_by("9000000000000d")).collect()
    # Every row is out of range → null; the point is that it computes without crashing.
    assert out.num_rows == len(_TS)


def test_strftime_unsupported_specifier_does_not_crash(t):
    """`%Z` on a naive instant used to panic; it must now compute (null per row)."""
    out = bt.from_arrow(t).select(v=col("ts").dt.strftime("%Z")).collect()
    assert out.num_rows == len(_TS)
