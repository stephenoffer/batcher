"""The temporal parameters that restore another engine's meaning, and the DATE bugfixes, vs DuckDB.

Every default here is DuckDB's, and each test pins that default beside the parameter, so a
parameter can never drift the default. The parameterised forms are checked against the
DuckDB expression that spells the same meaning (``dayofweek(d) + 1`` for Spark's
numbering, ``strftime(d, '%a')`` for an abbreviated name). The fixture carries a leap day,
pre-1970 instants with a sub-second part, a naive wall-clock inside a US DST gap, a
year boundary, and nulls in both columns.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

_TS = [
    dt.datetime(2024, 2, 29, 13, 45, 30, 123456),  # leap day
    dt.datetime(1969, 12, 31, 23, 59, 59, 500000),  # before the epoch, sub-second
    dt.datetime(2024, 3, 10, 2, 30),  # a wall-clock that does not exist in US DST
    dt.datetime(2023, 12, 31, 23, 59, 59),  # year boundary
    dt.datetime(2024, 2, 18),  # a Sunday at midnight
    None,
]
_D = [
    dt.date(2024, 2, 29),
    dt.date(1969, 6, 1),
    dt.date(2024, 3, 10),
    None,
    dt.date(2024, 2, 18),
    dt.date(2000, 1, 1),
]
_D0 = [
    dt.date(2024, 1, 1),
    dt.date(1970, 1, 1),
    None,
    dt.date(2024, 1, 1),
    dt.date(2024, 2, 25),
    dt.date(1999, 12, 31),
]


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "id": list(range(len(_TS))),
            "ts": pa.array(_TS, pa.timestamp("us")),
            "d": pa.array(_D, pa.date32()),
            "d0": pa.array(_D0, pa.date32()),
            "e": pa.array([0, -1, 1_700_000_000, None, 86_400, -86_401], pa.int64()),
            "dd": pa.array([0, -1, 19_675, None, 1, -214], pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def _types(out) -> dict[str, pa.DataType]:
    table = out if isinstance(out, pa.Table) else out.collect()
    return {f.name: f.type for f in table.schema}


# --- bug: *_between on DATE read the day count as microseconds ---------------------------


def test_days_between_on_dates_matches_date_diff(duck, t):
    out = bt.from_arrow(t).select(
        "id",
        fwd=col("d").dt.days_between(col("d0")),
        back=col("d0").dt.days_between("d"),
        weeks=col("d").dt.weeks_between(col("d0")),
        secs=col("d").dt.seconds_between(col("d0")),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, date_diff('day', d0, d) AS fwd, date_diff('day', d, d0) AS back, "
            "trunc(date_diff('day', d0, d) / 7)::BIGINT AS weeks, "
            "date_diff('second', d0, d) AS secs FROM t"
        ),
    )


def test_days_between_mixed_date_and_timestamp(duck, t):
    out = bt.from_arrow(t).select("id", r=col("ts").dt.days_between(col("d")))
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, CASE WHEN epoch_us(ts) >= epoch_us(d::TIMESTAMP) "
            "THEN (epoch_us(ts) - epoch_us(d::TIMESTAMP)) // 86400000000 "
            "ELSE -((epoch_us(d::TIMESTAMP) - epoch_us(ts)) // 86400000000) END AS r FROM t"
        ),
    )


# --- dayofweek(start, base) ---------------------------------------------------------------


@pytest.mark.parametrize("source", ["d", "ts"])
def test_dayofweek_numberings(duck, t, source):
    c = col(source)
    out = bt.from_arrow(t).select(
        "id",
        duckdb=c.dt.dayofweek(),
        spark=c.dt.dayofweek(start="sunday", base=1),
        daft=c.dt.dayofweek(start="monday", base=0),
        iso=c.dt.dayofweek(start="monday", base=1),
    )
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, dayofweek({source}) AS duckdb, dayofweek({source}) + 1 AS spark, "
            f"isodow({source}) - 1 AS daft, isodow({source}) AS iso FROM t"
        ),
    )
    assert set(_types(out).values()) == {pa.int64()}


@pytest.mark.parametrize(("start", "base"), [("tuesday", 0), ("sunday", 2), ("monday", True)])
def test_dayofweek_rejects_unknown_conventions(start, base):
    with pytest.raises(bt.PlanError):
        col("d").dt.dayofweek(start=start, base=base)


def test_dayofweek_default_keeps_the_duckdb_wire_shape():
    assert col("d").dt.dayofweek().to_ir() == {
        "e": "date",
        "fn": "day_of_week",
        "input": col("d").to_ir(),
    }


# --- dayname / monthname(abbreviated) ------------------------------------------------------


@pytest.mark.parametrize("source", ["d", "ts"])
def test_abbreviated_names(duck, t, source):
    c = col(source)
    out = bt.from_arrow(t).select(
        "id",
        day=c.dt.dayname(),
        day3=c.dt.dayname(abbreviated=True),
        mon=c.dt.monthname(),
        mon3=c.dt.monthname(abbreviated=True),
    )
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, dayname({source}) AS day, strftime({source}, '%a') AS day3, "
            f"monthname({source}) AS mon, strftime({source}, '%b') AS mon3 FROM t"
        ),
    )


# --- truncate(preserve_type) / month_start(keep_time) / last_day(keep_time) --------------


def test_truncate_default_is_a_timestamp_for_a_date_as_in_duckdb(duck, t):
    out = bt.from_arrow(t).select("id", r=col("d").dt.truncate("month"))
    assert _types(out)["r"] == pa.timestamp("us")
    assert_same(out.collect(), duck.sql("SELECT id, date_trunc('month', d) AS r FROM t"))
    assert (
        duck.sql("SELECT typeof(date_trunc('month', d)) FROM t LIMIT 1").fetchone()[0]
        == "TIMESTAMP"
    )


@pytest.mark.parametrize("unit", ["year", "quarter", "month", "week", "day"])
def test_truncate_preserve_type_keeps_a_date(duck, t, unit):
    out = bt.from_arrow(t).select(
        "id",
        r=col("d").dt.truncate(unit, preserve_type=True),
        ts=col("ts").dt.truncate(unit, preserve_type=True),
    )
    assert _types(out) == {"id": pa.int64(), "r": pa.date32(), "ts": pa.timestamp("us")}
    assert_same(
        out.collect(),
        duck.sql(
            f"SELECT id, date_trunc('{unit}', d)::DATE AS r, date_trunc('{unit}', ts) AS ts FROM t"
        ),
    )


def test_month_start_keep_time(duck, t):
    out = bt.from_arrow(t).select(
        "id",
        plain=col("ts").dt.month_start(),
        ts=col("ts").dt.month_start(keep_time=True),
        d=col("d").dt.month_start(keep_time=True),
    )
    assert _types(out) == {
        "id": pa.int64(),
        "plain": pa.timestamp("us"),
        "ts": pa.timestamp("us"),
        "d": pa.date32(),
    }
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, date_trunc('month', ts) AS plain, "
            "date_trunc('month', ts) + (ts - date_trunc('day', ts)) AS ts, "
            "date_trunc('month', d)::DATE AS d FROM t"
        ),
    )


def test_last_day_keep_time(duck, t):
    out = bt.from_arrow(t).select(
        "id",
        plain=col("ts").dt.last_day(),
        ts=col("ts").dt.last_day(keep_time=True),
        d=col("d").dt.last_day(keep_time=True),
    )
    assert _types(out) == {
        "id": pa.int64(),
        "plain": pa.date32(),
        "ts": pa.timestamp("us"),
        "d": pa.date32(),
    }
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, last_day(ts) AS plain, "
            "last_day(ts)::TIMESTAMP + (ts - date_trunc('day', ts)) AS ts, last_day(d) AS d FROM t"
        ),
    )


def test_keep_time_is_not_rewritten_into_a_range(duck, t):
    # `month_start(keep_time=True) = <first of month>` is not a contiguous range on the
    # column, so the date_trunc range rewrite must leave it alone.
    lit = dt.datetime(2024, 2, 1, 13, 45, 30, 123456)
    out = (
        bt.from_arrow(t)
        .filter(col("ts").dt.month_start(keep_time=True) == bt.lit(lit))
        .select("id")
    )
    assert out.collect().column("id").to_pylist() == [0]


def test_truncate_flags_default_off_on_the_wire():
    assert "preserve_type" not in col("d").dt.truncate("month").to_ir()
    assert "keep_time" not in col("d").dt.month_start().to_ir()
    ir = col("d").dt.month_start(keep_time=True).to_ir()
    assert ir["preserve_type"] is True and ir["keep_time"] is True


# --- epoch units, partition_days(as_date), from_epoch("name") -----------------------------


def test_timestamp_units(duck, t):
    out = bt.from_arrow(t).select(
        "id",
        s=col("ts").dt.timestamp("s"),
        us=col("ts").dt.timestamp("us"),
        ns=col("ts").dt.timestamp("ns"),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, floor(epoch(ts))::BIGINT AS s, epoch_us(ts) AS us, "
            "epoch_ns(ts) AS ns FROM t"
        ),
    )


def test_partition_days_as_date(duck, t):
    out = bt.from_arrow(t).select(
        "id", n=bt.partition_days("ts"), day=bt.partition_days("ts", as_date=True)
    )
    assert _types(out)["day"] == pa.date32()
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, date_diff('day', DATE '1970-01-01', ts::DATE) AS n, ts::DATE AS day FROM t"
        ),
    )


def test_from_epoch_reads_a_bare_string_as_a_column(duck, t):
    out = bt.from_arrow(t).select("id", r=bt.from_epoch("e"), d=bt.from_unix_date("dd"))
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT id, make_timestamp(e * 1000000) AS r, "
            "DATE '1970-01-01' + dd::INTEGER AS d FROM t"
        ),
    )
