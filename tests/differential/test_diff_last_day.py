"""Differential tests for `.dt.last_day()` (and `.dt.millennium()`) vs DuckDB.

`last_day(ts)` in DuckDB returns a DATE at the last day of the instant's month.
Batcher returns that day at 00:00:00 as a Timestamp(Microsecond) (mirroring how
`date_trunc` builds its result). To compare exactly, the DuckDB expression casts
its DATE result to TIMESTAMP (`last_day(ts)::TIMESTAMP`), so both sides surface a
midnight ``datetime.datetime`` through ``to_pylist()``.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    # A fixture spanning multiple months: Feb in a leap year (2020) and a
    # non-leap year (2021), 30- and 31-day months, and December (month rollover).
    instants = [
        dt.datetime(2020, 1, 15, 8, 30, 0),  # Jan (31 days)
        dt.datetime(2020, 2, 1, 0, 0, 0),  # Feb leap year (29 days)
        dt.datetime(2020, 2, 29, 23, 59, 59),  # Feb leap, last day already
        dt.datetime(2021, 2, 10, 12, 0, 0),  # Feb non-leap (28 days)
        dt.datetime(2021, 4, 5, 6, 7, 8),  # Apr (30 days)
        dt.datetime(2021, 6, 30, 0, 0, 0),  # Jun (30 days), last day already
        dt.datetime(2021, 7, 1, 1, 2, 3),  # Jul (31 days)
        dt.datetime(2021, 11, 20, 9, 9, 9),  # Nov (30 days)
        dt.datetime(2021, 12, 25, 18, 0, 0),  # Dec (31 days), year rollover
        dt.datetime(1999, 12, 31, 23, 0, 0),  # Dec 1999, millennium boundary
        dt.datetime(2000, 1, 1, 0, 0, 0),  # Jan 2000
        None,  # null propagation
    ]
    tbl = pa.table({"ts": pa.array(instants, type=pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


def test_last_day_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(l=col("ts").dt.last_day()).collect()
    # Cast DuckDB's DATE result to TIMESTAMP so it aligns with Batcher's
    # Timestamp(us) at midnight.
    expected = duck.sql("SELECT last_day(ts)::TIMESTAMP AS l FROM t")
    assert_same(out, expected)


def test_last_day_and_millennium_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            l=col("ts").dt.last_day(),
            mil=col("ts").dt.millennium(),
        )
        .collect()
    )
    expected = duck.sql("SELECT last_day(ts)::TIMESTAMP AS l, millennium(ts) AS mil FROM t")
    assert_same(out, expected)
