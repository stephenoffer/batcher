"""`year(col)` / `decade(col)` filters match DuckDB after the sargable rewrite.

The `temporal_sargable` NORMALIZE rules turn a monotonic temporal-extraction
comparison into a half-open range on the raw column; the result MUST stay identical to
DuckDB evaluating the extraction directly — for every comparison operator, for both a
DATE and a TIMESTAMP column, across boundary years, with NULLs and on empty input.

Importing the rule module registers the 10 rules into `DEFAULT_REGISTRY` so the full
`Optimizer` (used by `.collect()`) applies them.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.kyber.rules.extra import temporal_sargable as _temporal_sargable  # noqa: F401

# Comparison operators, mapped to the Batcher expression builder and the SQL spelling.
_OPS = {
    "eq": (lambda e, y: e == y, "="),
    "lt": (lambda e, y: e < y, "<"),
    "le": (lambda e, y: e <= y, "<="),
    "gt": (lambda e, y: e > y, ">"),
    "ge": (lambda e, y: e >= y, ">="),
}


@pytest.fixture
def date_tbl(duck):
    # Boundary-heavy dates: year starts/ends and a null, spanning 2019–2023.
    dates = [
        dt.date(2019, 12, 31),
        dt.date(2020, 1, 1),
        dt.date(2020, 6, 15),
        dt.date(2020, 12, 31),
        dt.date(2021, 1, 1),
        dt.date(2021, 7, 4),
        dt.date(2022, 12, 31),
        dt.date(2023, 1, 1),
        None,
    ]
    tbl = pa.table({"d": pa.array(dates, type=pa.date32()), "v": list(range(len(dates)))})
    duck.register("date_tbl", tbl)
    return tbl


@pytest.fixture
def ts_tbl(duck):
    times = [
        dt.datetime(2019, 12, 31, 23, 59, 59),
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 6, 15, 12, 30),
        dt.datetime(2021, 1, 1, 0, 0, 0),
        dt.datetime(2021, 12, 31, 23, 59, 59),
        dt.datetime(2022, 3, 3, 3, 3, 3),
        None,
    ]
    tbl = pa.table({"t": pa.array(times, type=pa.timestamp("us")), "v": list(range(len(times)))})
    duck.register("ts_tbl", tbl)
    return tbl


# --- year over a DATE column ---------------------------------------------------


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("year", [2019, 2020, 2021, 2023])
def test_year_filter_date_column_vs_duckdb(duck, date_tbl, op, year):
    build, sql_op = _OPS[op]
    out = bt.from_arrow(date_tbl).filter(build(col("d").dt.year(), year)).collect()
    expected = duck.sql(f"SELECT * FROM date_tbl WHERE year(d) {sql_op} {year}")
    assert_same(out, expected)


# --- year over a TIMESTAMP column ----------------------------------------------


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("year", [2019, 2020, 2021, 2022])
def test_year_filter_timestamp_column_vs_duckdb(duck, ts_tbl, op, year):
    build, sql_op = _OPS[op]
    out = bt.from_arrow(ts_tbl).filter(build(col("t").dt.year(), year)).collect()
    expected = duck.sql(f"SELECT * FROM ts_tbl WHERE year(t) {sql_op} {year}")
    assert_same(out, expected)


# --- decade over a DATE column -------------------------------------------------


@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("decade", [201, 202, 203])
def test_decade_filter_date_column_vs_duckdb(duck, date_tbl, op, decade):
    build, sql_op = _OPS[op]
    out = bt.from_arrow(date_tbl).filter(build(col("d").dt.decade(), decade)).collect()
    expected = duck.sql(f"SELECT * FROM date_tbl WHERE extract(decade FROM d) {sql_op} {decade}")
    assert_same(out, expected)


# --- decade over a TIMESTAMP column --------------------------------------------


@pytest.mark.parametrize("op", list(_OPS))
def test_decade_filter_timestamp_column_vs_duckdb(duck, ts_tbl, op):
    build, sql_op = _OPS[op]
    out = bt.from_arrow(ts_tbl).filter(build(col("t").dt.decade(), 202)).collect()
    expected = duck.sql(f"SELECT * FROM ts_tbl WHERE extract(decade FROM t) {sql_op} 202")
    assert_same(out, expected)


# --- literal-on-the-left (mirrored operator) -----------------------------------


def test_literal_on_left_vs_duckdb(duck, date_tbl):
    # `2021 < year(d)` exercises the operator-mirror path.
    out = bt.from_arrow(date_tbl).filter(bt.lit(2021) < col("d").dt.year()).collect()
    expected = duck.sql("SELECT * FROM date_tbl WHERE 2021 < year(d)")
    assert_same(out, expected)


# --- empty input ---------------------------------------------------------------


def test_empty_input_vs_duckdb(duck):
    tbl = pa.table({"d": pa.array([], type=pa.date32()), "v": pa.array([], type=pa.int64())})
    duck.register("empty_tbl", tbl)
    out = bt.from_arrow(tbl).filter(col("d").dt.year() == 2021).collect()
    expected = duck.sql("SELECT * FROM empty_tbl WHERE year(d) = 2021")
    assert_same(out, expected)


# --- all-null column -----------------------------------------------------------


def test_all_null_column_vs_duckdb(duck):
    tbl = pa.table({"d": pa.array([None, None], type=pa.date32()), "v": [1, 2]})
    duck.register("null_tbl", tbl)
    for build, sql_op in _OPS.values():
        out = bt.from_arrow(tbl).filter(build(col("d").dt.year(), 2021)).collect()
        expected = duck.sql(f"SELECT * FROM null_tbl WHERE year(d) {sql_op} 2021")
        assert_same(out, expected)


# --- non-sargable extraction still matches (rule must not fire) -----------------


def test_month_extraction_unchanged_vs_duckdb(duck, date_tbl):
    # `month` recurs yearly → the rule leaves it; the engine still evaluates it, and
    # the result must equal DuckDB's.
    out = bt.from_arrow(date_tbl).filter(col("d").dt.month() == 12).collect()
    expected = duck.sql("SELECT * FROM date_tbl WHERE month(d) = 12")
    assert_same(out, expected)
