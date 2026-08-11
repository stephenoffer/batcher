"""`date_diff` over every unit, against DuckDB.

Two defects met in this function, and the second is the one that mattered.

The visible one was a gap: only DAY and WEEK were implemented, so `date_diff('minute', ...)`
-- the ordinary way to measure a gap between two events -- raised. The silent one was that
DAY and WEEK were computed as `CAST(b AS BIGINT) - CAST(a AS BIGINT)`, which is a *day*
count for a DATE and a *microsecond* count for a TIMESTAMP. On timestamps the function
returned 120000000 where DuckDB returns 1, with no error: a wrong answer off by a factor of
86.4 billion, in the direction that looks like a plausible large number.

The semantics matter as much as the arithmetic. `date_diff` counts **boundary crossings**,
not elapsed time: 00:59 to 01:00 is one hour apart (one boundary between them) and 00:00 to
00:59 is zero, even though the second span is 59 times longer. Reading it as elapsed-time
divided by the unit gets both cases backwards, so the unit is treated here as a grid to snap
both endpoints onto.

WEEK is deliberately not on that grid: DuckDB reports whole 7-day spans truncated toward
zero, which is a different rule, and it is asserted separately below so a future
"consistency" cleanup cannot quietly fold it in with the rest.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_FIXED_UNITS = ["microsecond", "millisecond", "second", "minute", "hour", "day"]
_CALENDAR_UNITS = ["month", "quarter", "year"]


def _timestamps() -> pa.Table:
    """Endpoints chosen for the boundaries they straddle, not for spread."""
    points = [
        dt.datetime(2024, 1, 1, 0, 0),
        dt.datetime(2024, 1, 1, 0, 0, 0, 999999),  # sub-microsecond edge
        dt.datetime(2024, 1, 1, 23, 59, 59),
        dt.datetime(2024, 1, 2, 0, 0),  # one second later, over a day boundary
        dt.datetime(2024, 2, 29, 12, 0),  # leap day
        dt.datetime(2024, 3, 1, 0, 0),
        dt.datetime(2023, 12, 31, 23, 59),  # over a year boundary
        dt.datetime(2025, 1, 1, 0, 0),
        dt.datetime(1969, 12, 31, 23, 30),  # negative epoch: floor, not truncate
        dt.datetime(1970, 1, 1, 0, 0),
        dt.datetime(2024, 3, 10, 1, 59),  # US DST transition
        dt.datetime(2024, 3, 10, 3, 1),
    ]
    return pa.table(
        {
            "a": pa.array([p for p in points for _ in points]),
            "b": pa.array([q for _ in points for q in points]),
        }
    )


def _dates() -> pa.Table:
    """The input shape that already worked, so the fix has to keep it working."""
    points = [
        dt.date(2024, 1, 1),
        dt.date(2024, 3, 1),
        dt.date(2023, 12, 31),
        dt.date(1969, 12, 31),
        dt.date(1970, 1, 1),
        dt.date(2024, 2, 29),
        dt.date(2025, 7, 4),
    ]
    return pa.table(
        {
            "a": pa.array([p for p in points for _ in points]),
            "b": pa.array([q for _ in points for q in points]),
        }
    )


@pytest.mark.parametrize("unit", _FIXED_UNITS + _CALENDAR_UNITS + ["week"])
def test_every_unit_matches_duckdb_on_timestamps(duck, unit):
    """144 endpoint pairs per unit, over leap, DST, epoch-sign and boundary edges."""
    table = _timestamps()
    sql = f"SELECT date_diff('{unit}', a, b) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize("unit", _FIXED_UNITS + _CALENDAR_UNITS + ["week"])
def test_every_unit_matches_duckdb_on_dates(duck, unit):
    """DATE was the only input shape that worked before; it must still."""
    table = _dates()
    sql = f"SELECT date_diff('{unit}', a, b) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_timestamp_day_difference_is_not_a_microsecond_count(duck):
    """The regression stated as a value, because the fuzz above states it only as a match.

    `date_diff('day', ...)` over two timestamps two minutes apart used to return
    120000000. Nothing about that is obviously wrong at a glance, which is why it lasted.
    """
    table = pa.table(
        {
            "a": pa.array([dt.datetime(2024, 1, 1, 23, 59)]),
            "b": pa.array([dt.datetime(2024, 1, 2, 0, 1)]),
        }
    )
    sql = "SELECT date_diff('day', a, b) AS r FROM t"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["r"] == [1]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    ("unit", "a", "b", "expected"),
    [
        # Boundary crossings, not elapsed time: one minute apart but one hour between.
        ("hour", dt.datetime(2024, 1, 1, 0, 59), dt.datetime(2024, 1, 1, 1, 0), 1),
        ("hour", dt.datetime(2024, 1, 1, 0, 0), dt.datetime(2024, 1, 1, 0, 59), 0),
        ("minute", dt.datetime(2024, 1, 1, 0, 0, 59), dt.datetime(2024, 1, 1, 0, 1), 1),
        ("month", dt.datetime(2024, 1, 31), dt.datetime(2024, 2, 1), 1),
        ("month", dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 31), 0),
        ("year", dt.datetime(2023, 12, 31), dt.datetime(2024, 1, 1), 1),
        # Negative directions keep their sign.
        ("hour", dt.datetime(2024, 1, 1, 5, 0), dt.datetime(2024, 1, 1, 1, 0), -4),
        ("minute", dt.datetime(2024, 1, 1, 0, 1), dt.datetime(2024, 1, 1, 0, 0, 59), -1),
    ],
)
def test_the_boundary_crossing_rule_stated_case_by_case(duck, unit, a, b, expected):
    """Spelled out rather than fuzzed: an elapsed-time reading passes neither pair."""
    table = pa.table({"a": pa.array([a]), "b": pa.array([b])})
    sql = f"SELECT date_diff('{unit}', a, b) AS r FROM t"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["r"] == [expected]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Thursday to the following Monday: a week *boundary* is crossed, but only 4 days
        # elapsed, and DuckDB answers 0. This is the case that proves WEEK is not on the
        # boundary grid the other units use.
        (dt.date(2024, 1, 4), dt.date(2024, 1, 8), 0),
        (dt.date(2024, 1, 4), dt.date(2024, 1, 10), 0),
        (dt.date(2024, 1, 1), dt.date(2024, 1, 8), 1),
        # Truncation toward zero, so -6 days is 0 rather than -1.
        (dt.date(2024, 1, 8), dt.date(2024, 1, 2), 0),
        (dt.date(2024, 1, 15), dt.date(2024, 1, 1), -2),
    ],
)
def test_week_counts_whole_seven_day_spans_not_calendar_weeks(duck, a, b, expected):
    """WEEK follows a different rule from every other unit, on purpose."""
    table = pa.table({"a": pa.array([a]), "b": pa.array([b])})
    sql = "SELECT date_diff('week', a, b) AS r FROM t"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["r"] == [expected]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_null_endpoint_yields_null(duck):
    """Nulls propagate rather than becoming an epoch-zero difference."""
    table = pa.table(
        {
            "a": pa.array([dt.datetime(2024, 1, 1), None, None]),
            "b": pa.array([None, dt.datetime(2024, 1, 1), None]),
        }
    )
    sql = "SELECT date_diff('hour', a, b) AS r FROM t"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["r"] == [None, None, None]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_an_unsupported_unit_names_the_ones_that_work():
    """The refusal has to be actionable, since six units used to be missing entirely."""
    table = _dates()
    with pytest.raises(Exception, match=r"(?i)date_diff unit"):
        bt.sql("SELECT date_diff('fortnight', a, b) AS r FROM t", t=table).collect()


def test_the_result_is_an_integer_column(duck):
    """A float difference would compare equal under `assert_same` and still be wrong."""
    table = _timestamps()
    got = bt.sql("SELECT date_diff('second', a, b) AS r FROM t", t=table)
    assert pa.types.is_integer(got.schema.field("r").type)


def test_it_survives_a_partitioned_collect(duck):
    """Same expression, many partitions: no per-partition epoch anchoring."""
    table = _timestamps()
    sql = "SELECT date_diff('minute', a, b) AS r FROM t"
    duck.register("t", table)
    ds = bt.sql(sql, t=table)
    assert_same(ds.repartition(4).collect(), duck.sql(sql))
