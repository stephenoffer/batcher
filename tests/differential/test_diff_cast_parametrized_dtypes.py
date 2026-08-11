"""Parametrized cast targets — `decimal(p,s)`, `timestamp(unit[, tz])`, `time`, `duration`.

The cast vocabulary used to be a fixed list of eight names, which meant there was no
spelling at all for the types financial and event data arrive as. A Parquet money column
could be read, summed and compared, but ``cast("decimal(12,4)")`` raised ``unknown cast
dtype`` and nothing else worked either. The same held for every timestamp that was not
microseconds, and for every time-of-day and duration type.

These cases pin the new grammar against DuckDB, which spells the same targets
``DECIMAL(12,4)`` / ``TIMESTAMP_NS`` / ``TIME``. Where DuckDB has no comparable type
(Arrow durations), the case asserts the resulting Arrow type instead, which is the part
the grammar is responsible for.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.errors import PlanError


@pytest.fixture
def source(duck):
    """Values in the shapes a cast target has to accept them from."""
    table = pa.table(
        {
            "s": pa.array(["1.50", "-2.25", None], pa.string()),
            "f": pa.array([1.5, -2.25, None], pa.float64()),
            "i": pa.array([3, -4, None], pa.int64()),
            "ts": pa.array([1_000_000, 2_000_000, None], pa.timestamp("us")),
        }
    )
    duck.register("t", table)
    return bt.from_arrow(table)


def test_cast_a_float_to_a_decimal(duck, source):
    ds = source.select(amt=col("f").cast("decimal(12,4)"))
    assert ds.schema.field("amt").type == pa.decimal128(12, 4)
    assert_same(ds.collect(), duck.sql("SELECT CAST(f AS DECIMAL(12,4)) AS amt FROM t"))


def test_cast_a_string_to_a_decimal(duck, source):
    ds = source.select(amt=col("s").cast("decimal(10,2)"))
    assert_same(ds.collect(), duck.sql("SELECT CAST(s AS DECIMAL(10,2)) AS amt FROM t"))


def test_cast_an_integer_to_a_decimal_is_exact(duck, source):
    """An integer widened into a decimal keeps every digit — the point of having decimals."""
    ds = source.select(amt=col("i").cast("decimal(20,2)"))
    assert_same(ds.collect(), duck.sql("SELECT CAST(i AS DECIMAL(20,2)) AS amt FROM t"))


def test_decimal_scale_defaults_to_zero(duck, source):
    ds = source.select(amt=col("i").cast("decimal(9)"))
    assert ds.schema.field("amt").type == pa.decimal128(9, 0)
    assert_same(ds.collect(), duck.sql("SELECT CAST(i AS DECIMAL(9,0)) AS amt FROM t"))


def test_the_sql_aliases_name_the_same_decimal(duck, source):
    for spelling in ("decimal(12,4)", "decimal128(12,4)", "numeric(12,4)", "DECIMAL(12, 4)"):
        ds = source.select(amt=col("f").cast(spelling))
        assert ds.schema.field("amt").type == pa.decimal128(12, 4), spelling


def test_cast_to_a_nanosecond_timestamp(duck, source):
    ds = source.select(t=col("ts").cast("timestamp(ns)"))
    assert ds.schema.field("t").type == pa.timestamp("ns")
    assert_same(ds.collect(), duck.sql("SELECT CAST(ts AS TIMESTAMP_NS) AS t FROM t"))


def test_cast_to_a_second_resolution_timestamp_truncates_like_duckdb(duck, source):
    ds = source.select(t=col("ts").cast("timestamp(s)"))
    assert ds.schema.field("t").type == pa.timestamp("s")
    assert_same(ds.collect(), duck.sql("SELECT CAST(ts AS TIMESTAMP_S) AS t FROM t"))


def test_cast_to_a_zoned_timestamp_carries_the_zone(source):
    """Arrow compares a zone byte-wise, so the exact identifier must survive the cast."""
    ds = source.select(t=col("ts").cast("timestamp(us, UTC)"))
    assert ds.schema.field("t").type == pa.timestamp("us", "UTC")
    assert ds.collect().schema.field("t").type == pa.timestamp("us", "UTC")


def test_a_zone_is_not_case_folded(source):
    lower = source.select(t=col("ts").cast("timestamp(us, utc)")).schema.field("t").type
    upper = source.select(t=col("ts").cast("timestamp(us, UTC)")).schema.field("t").type
    assert lower != upper


def test_cast_to_a_time_of_day_picks_its_width(source):
    micros = source.select(t=col("ts").cast("time(us)"))
    assert micros.schema.field("t").type == pa.time64("us")
    assert micros.collect().schema.field("t").type == pa.time64("us")
    millis = source.select(t=col("ts").cast("time(ms)"))
    assert millis.schema.field("t").type == pa.time32("ms")


def test_cast_to_a_duration(source):
    ds = source.select(d=col("i").cast("duration(s)"))
    assert ds.schema.field("d").type == pa.duration("s")
    assert ds.collect().schema.field("d").type == pa.duration("s")


def test_cast_to_the_narrow_integer_widths(duck, source):
    ds = source.select(a=col("i").cast("int16"), b=col("i").cast("int8"))
    assert ds.schema.field("a").type == pa.int16()
    assert_same(
        ds.collect(),
        duck.sql("SELECT CAST(i AS SMALLINT) AS a, CAST(i AS TINYINT) AS b FROM t"),
    )


def test_try_cast_to_a_decimal_nulls_what_will_not_fit(duck):
    """`try_cast` keeps its null-on-failure contract for a parametrized target too."""
    table = pa.table({"s": pa.array(["1.50", "not a number", "3.25"], pa.string())})
    duck.register("u", table)
    ds = bt.from_arrow(table).select(amt=col("s").try_cast("decimal(10,2)"))
    assert_same(ds.collect(), duck.sql("SELECT TRY_CAST(s AS DECIMAL(10,2)) AS amt FROM u"))


def test_a_parametrized_cast_survives_a_group_by(duck, source):
    """The target has to reach the engine intact through an aggregation, not just a select."""
    ds = (
        source.with_columns(amt=col("f").cast("decimal(12,4)"))
        .group_by("i")
        .agg(total=col("amt").sum())
    )
    assert_same(
        ds.collect(),
        duck.sql("SELECT i, SUM(CAST(f AS DECIMAL(12,4))) AS total FROM t GROUP BY i"),
    )


@pytest.mark.parametrize(
    "spelling",
    [
        "decimal(39,2)",  # past what a decimal128 carries
        "decimal(4,6)",  # more fractional digits than total digits
        "decimal",  # a decimal with no precision is not a type
        "time32(us)",  # a 32-bit time cannot carry microseconds
        "duration(fortnight)",
        "not_a_type",
    ],
)
def test_an_impossible_target_is_rejected_at_plan_time(source, spelling):
    """Rejecting beats clamping: the clamped type would overflow the very values asked for."""
    with pytest.raises(PlanError, match="unknown cast dtype"):
        source.select(x=col("f").cast(spelling))


def test_the_error_names_the_parametrized_forms(source):
    """A user who reaches for a decimal must be told the spelling exists."""
    with pytest.raises(PlanError, match=r"decimal\(12,4\)"):
        source.select(x=col("f").cast("decimal_thing"))
