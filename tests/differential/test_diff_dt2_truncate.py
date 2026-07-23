"""Differential `date_trunc` parity vs DuckDB across every truncation unit.

Wave-1 shipped `date_trunc` with only year/month/day/hour/minute/second; the
calendar units DuckDB also supports (quarter, week, decade, century, millennium)
and the sub-second units (millisecond, microsecond) errored instead of computing.
These cases pin the full unit vocabulary, including pre-1970 instants where the
truncation must floor toward −∞ (not toward zero).
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

# Instants chosen to exercise the edges: pre-1970 (negative epoch, incl. a
# sub-second fraction that flooring vs truncating tells apart), ISO-week
# boundaries where the Monday floor crosses a year, leap day, century boundary.
_TS = [
    dt.datetime(2024, 2, 15, 13, 45, 30, 123456),
    dt.datetime(1969, 6, 15, 13, 45, 30, 500000),  # pre-1970 sub-second
    dt.datetime(1955, 3, 7, 8, 20, 55, 0),
    dt.datetime(2021, 1, 1, 0, 0, 0, 0),  # ISO week floors into 2020
    dt.datetime(2020, 12, 31, 23, 59, 59, 999999),
    dt.datetime(2000, 2, 29, 12, 0, 0, 0),  # leap day
    dt.datetime(1901, 1, 1, 0, 0, 0, 0),
    None,
]


@pytest.fixture
def t(duck):
    tbl = pa.table({"ts": pa.array(_TS, pa.timestamp("us"))})
    duck.register("t", tbl)
    return tbl


_UNITS = [
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "decade",
    "century",
    "millennium",
    "millisecond",
    "microsecond",
]


@pytest.mark.parametrize("unit", _UNITS)
def test_date_trunc_all_units_vs_duckdb(duck, t, unit):
    out = bt.from_arrow(t).select(v=col("ts").dt.truncate(unit)).collect()
    expected = duck.sql(f"SELECT date_trunc('{unit}', ts) AS v FROM t")
    assert_same(out, expected)


def test_date_trunc_unknown_unit_errors(t):
    """A typo'd unit must raise cleanly, not silently null the whole column."""
    with pytest.raises(Exception):  # noqa: B017 -- engine RuntimeError surfaces here
        bt.from_arrow(t).select(v=col("ts").dt.truncate("fortnight")).collect()
