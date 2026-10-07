"""Nanosecond precision, duration components and totals, multi-format parsing, and
per-row calendar offsets, held to DuckDB.

* ``epoch_ns``/``nanosecond`` used to compose over a microsecond cast, so a
  ``timestamp[ns]`` column lost its last three digits with no error.
* ``hour()``/``minute()``/``second()`` on a duration returned the *total* (49 for 49h05m)
  where DuckDB returns the component (1). That is a behaviour change to a wrong answer;
  ``dt.total(unit)`` is the spelling for the total.
* ``to_datetime`` took one format; DuckDB's ``strptime(s, [f1, f2])`` tries a list in
  order, which is the oracle for the widened form.
* ``date_add``/``add_months`` took only a constant count.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

_NS = [1_000_000_001, -1, 1_709_251_200_123_456_789, 0, None]


def _ns_table():
    return pa.table({"t": pa.array(_NS, pa.timestamp("ns"))})


def test_epoch_ns_and_nanosecond_keep_every_digit(duck):
    got = bt.from_arrow(_ns_table()).select(e=col("t").dt.epoch_ns(), n=col("t").dt.nanosecond())
    duck.register("nst", _ns_table())
    # DuckDB's `nanosecond` counts the seconds too; Batcher's is Polars' nanosecond-of-second.
    expected = duck.sql(
        "SELECT epoch_ns(t) AS e, "
        "CAST(((nanosecond(t) % 1000000000) + 1000000000) % 1000000000 AS BIGINT) AS n "
        "FROM nst"
    )
    assert_same(got.collect(), expected)


def test_epoch_ns_on_a_microsecond_column_and_a_date_is_scaled_exactly(duck):
    ds = bt.from_pydict(
        {"t": [dt.datetime(2024, 1, 1, 0, 0, 0, 123456), None], "d": [dt.date(2024, 1, 1), None]}
    )
    got = ds.select(
        e=col("t").dt.epoch_ns(), n=col("t").dt.nanosecond(), de=col("d").dt.epoch_ns()
    ).collect()
    expected = duck.sql(
        "SELECT epoch_ns(t) AS e, CAST(nanosecond(t) % 1000000000 AS BIGINT) AS n, "
        "epoch_ns(CAST(d AS TIMESTAMP)) AS de FROM (VALUES "
        "(TIMESTAMP '2024-01-01 00:00:00.123456', DATE '2024-01-01'), (NULL, NULL)) v(t, d)"
    )
    assert_same(got, expected)


_PAIRS = [
    (dt.datetime(2024, 1, 3, 1, 5), dt.datetime(2024, 1, 1)),  # 2 days 01:05:00
    (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3, 1, 5, 7, 250000)),  # negative
    (dt.datetime(2024, 1, 1, 0, 0, 59), dt.datetime(2024, 1, 1)),  # under a minute
    (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1)),  # zero
    (None, dt.datetime(2024, 1, 1)),
]


def _spans(duck):
    ds = bt.from_pydict({"a": [p[0] for p in _PAIRS], "b": [p[1] for p in _PAIRS]})
    rows = ", ".join(
        "(" + ", ".join("NULL" if v is None else f"TIMESTAMP '{v.isoformat()}'" for v in pair) + ")"
        for pair in _PAIRS
    )
    duck.sql(f"CREATE OR REPLACE TABLE spans AS SELECT a - b AS d FROM (VALUES {rows}) v(a, b)")
    return ds.select(d=col("a") - col("b"))


@pytest.mark.parametrize("part", ["day", "hour", "minute", "second"])
def test_duration_components_match_duckdb_intervals(duck, part):
    got = _spans(duck).select(r=getattr(col("d").dt, part)()).collect()
    assert_same(got, duck.sql(f"SELECT {part}(d) AS r FROM spans"))


@pytest.mark.parametrize(("unit", "seconds"), [("s", 1), ("m", 60), ("h", 3600), ("d", 86400)])
def test_duration_total_truncates_the_duckdb_epoch(duck, unit, seconds):
    got = _spans(duck).select(r=col("d").dt.total(unit)).collect()
    expected = duck.sql(f"SELECT CAST(trunc(epoch(d) / {seconds}) AS BIGINT) AS r FROM spans")
    assert_same(got, expected)


def test_duration_total_in_sub_second_units(duck):
    got = _spans(duck).select(ms=col("d").dt.total("ms"), us=col("d").dt.total("us")).collect()
    expected = duck.sql(
        "SELECT CAST(trunc(epoch_us(d) / 1000) AS BIGINT) AS ms, epoch_us(d) AS us FROM spans"
    )
    assert_same(got, expected)


def test_a_calendar_field_of_a_duration_is_refused():
    span = bt.from_pydict({"a": [dt.datetime(2024, 1, 2)], "b": [dt.datetime(2024, 1, 1)]})
    with pytest.raises(Exception, match="total"):
        span.select(r=(col("a") - col("b")).dt.year()).collect()
    with pytest.raises(bt.PlanError, match="month"):
        col("d").dt.total("mo")


_MIXED = ["2024-01-15", "15/01/2024", "2024-01-15 10:30:00", "junk", None, ""]


def test_strptime_tries_formats_in_order_like_duckdb(duck):
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"]
    ds = bt.from_pydict({"s": _MIXED})
    got = ds.select(t=col("s").str.to_datetime(formats)).collect()
    rows = ", ".join("(NULL)" if s is None else f"('{s}')" for s in _MIXED)
    expected = duck.sql(
        f"SELECT try_strptime(s, {formats!r}) AS t FROM (VALUES {rows}) v(s)".replace('"', "'")
    )
    assert_same(got, expected)


def test_strict_multi_format_raises_only_when_every_format_fails():
    ok = bt.from_pydict({"s": ["2024-01-15", "15/01/2024"]})
    out = ok.select(t=col("s").str.to_datetime(["%Y-%m-%d", "%d/%m/%Y"], strict=True))
    assert out.to_pydict()["t"] == [dt.datetime(2024, 1, 15)] * 2
    bad = bt.from_pydict({"s": ["2024-01-15", "junk"]})
    with pytest.raises(Exception, match="any of the formats"):
        bad.select(t=col("s").str.to_datetime(["%Y-%m-%d", "%d/%m/%Y"], strict=True)).collect()


def test_multi_format_sql_strptime(duck):
    query = "SELECT strptime(s, ['%Y-%m-%d', '%d/%m/%Y']) AS t FROM v"
    table = pa.table({"s": ["2024-01-15", "15/01/2024", None]})
    duck.register("v", table)
    got = bt.sql(query, v=bt.from_arrow(table)).collect()
    assert_same(got, duck.sql(query))


def test_a_single_format_list_serializes_as_before():
    one = col("s").str.to_datetime(["%Y"]).to_ir()
    assert one == col("s").str.to_datetime("%Y").to_ir()
    with pytest.raises(bt.PlanError, match="non-empty"):
        col("s").str.to_datetime([])


_DATES = [
    dt.date(2024, 1, 31),
    dt.date(2024, 1, 31),
    dt.date(2023, 2, 28),
    None,
    dt.date(2024, 3, 1),
]
_COUNTS = [1, -2, 12, 3, None]


def _per_row(duck):
    ds = bt.from_pydict({"d": _DATES, "n": _COUNTS})
    rows = ", ".join(
        f"({'NULL' if d is None else repr(d.isoformat())}::DATE, {'NULL' if n is None else n})"
        for d, n in zip(_DATES, _COUNTS, strict=True)
    )
    duck.sql(f"CREATE OR REPLACE TABLE pr AS SELECT * FROM (VALUES {rows}) v(d, n)")
    return ds


def test_per_row_add_months_clamps_like_duckdb(duck):
    got = _per_row(duck).select(r=col("d").dt.add_months(col("n"))).collect()
    expected = duck.sql("SELECT CAST(d + to_months(n) AS DATE) AS r FROM pr")
    assert_same(got, expected)


def test_per_row_date_add_and_sub_match_duckdb(duck):
    ds = _per_row(duck)
    got = ds.select(a=bt.date_add(col("d"), col("n")), s=bt.date_sub(col("d"), col("n"))).collect()
    expected = duck.sql("SELECT d + n AS a, d - n AS s FROM pr")
    assert_same(got, expected)


def test_sql_add_months_accepts_a_column(duck):
    table = pa.table({"d": pa.array(_DATES, pa.date32()), "n": pa.array(_COUNTS, pa.int64())})
    got = bt.sql("SELECT add_months(d, n) AS m FROM v", v=bt.from_arrow(table)).collect()
    duck.register("v", table)
    assert_same(got, duck.sql("SELECT CAST(d + to_months(n) AS DATE) AS m FROM v"))


def test_spark_date_add_accepts_a_column(duck):
    table = pa.table({"d": pa.array(_DATES, pa.date32()), "n": pa.array(_COUNTS, pa.int64())})
    got = bt.sql(
        "SELECT date_add(d, n) AS a, date_sub(d, n) AS s FROM v",
        v=bt.from_arrow(table),
        dialect="spark",
    ).collect()
    duck.register("v", table)
    expected = duck.sql("SELECT d + CAST(n AS INTEGER) AS a, d - CAST(n AS INTEGER) AS s FROM v")
    assert_same(got, expected)


def test_per_row_shift_on_a_timestamp_keeps_the_clock():
    ds = bt.from_pydict({"t": [dt.datetime(2024, 1, 31, 13, 45)], "n": [1]})
    out = ds.select(
        m=col("t").dt.add_months(col("n")), d=bt.date_add(col("t"), col("n"))
    ).to_pydict()
    assert out == {
        "m": [dt.datetime(2024, 2, 29, 13, 45)],
        "d": [dt.datetime(2024, 2, 1, 13, 45)],
    }
