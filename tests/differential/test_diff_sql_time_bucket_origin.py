"""``time_bucket`` boundaries must land where DuckDB's do, or the call must be refused.

`WindowStart` anchors buckets at the Unix epoch. DuckDB anchors them at **2000-01-03**,
10,959 days later. The two grids coincide only when the bucket width divides that gap —
which is why the widths already under test looked right: 1 DAY, 2 HOUR and 5 MINUTE all
do. A width that does not put every boundary on a different instant, silently:

    time_bucket(INTERVAL 2 DAY, DATE '2021-01-01')   -- DuckDB 2020-12-31, Batcher 2021-01-01
    time_bucket(INTERVAL 7 DAY, DATE '2021-01-04')   -- DuckDB 2021-01-04, Batcher 2020-12-31

A whole week of rows lands in the neighbouring bucket, so a time-series aggregate reports
the wrong totals against the wrong periods and nothing raises. Rather than answer on a
shifted grid, a misaligned width is refused.

A *calendar* width is a different problem and no longer shares that answer. A month is not
a number of microseconds at all, so it never had a `WindowStart` width to misalign; it is
bucketed on the **month index** instead, which reproduces DuckDB's 2000-01 origin exactly
and needs nothing to divide evenly.

The first group pins that every *aligned* width still agrees with DuckDB exactly, so the
guard cannot be satisfied by simply refusing more.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_TS = "TIMESTAMP '2024-03-05 06:07:08'"


@pytest.mark.parametrize(
    "width",
    [
        "1 DAY",
        # 10,959 is divisible by 3, so a 3-day grid coincides with DuckDB's even though a
        # 2-day one does not. The rule is arithmetic, not "multi-day widths are wrong".
        "3 DAY",
        "1 HOUR",
        "2 HOUR",
        "3 HOUR",
        "4 HOUR",
        "6 HOUR",
        "12 HOUR",
        "1 MINUTE",
        "5 MINUTE",
        "15 MINUTE",
        "30 MINUTE",
        "30 SECOND",
        "1 SECOND",
    ],
)
def test_aligned_widths_match_duckdb(duck, width):
    query = f"SELECT time_bucket(INTERVAL {width}, {_TS}) AS r"
    assert_same(bt.sql(query).collect(), duck.sql(query))


@pytest.mark.parametrize(
    "width", ["2 DAY", "4 DAY", "7 DAY", "9 DAY", "5 HOUR", "7 HOUR", "7 MINUTE"]
)
def test_misaligned_widths_are_refused_rather_than_shifted(width):
    with pytest.raises(NotImplementedError, match="2000-01-03"):
        bt.sql(f"SELECT time_bucket(INTERVAL {width}, {_TS}) AS r").collect()


def test_the_refusal_names_a_width_that_works():
    """The error has to be actionable, and the width it suggests has to be accepted."""
    with pytest.raises(NotImplementedError, match="6 HOUR"):
        bt.sql(f"SELECT time_bucket(INTERVAL 7 DAY, {_TS}) AS r").collect()
    bt.sql(f"SELECT time_bucket(INTERVAL 6 HOUR, {_TS}) AS r").collect()


@pytest.mark.parametrize(
    "width",
    ["1 MONTH", "2 MONTH", "3 MONTH", "6 MONTH", "1 QUARTER", "1 YEAR", "2 YEAR"],
)
@pytest.mark.parametrize(
    "value",
    [
        _TS,
        "DATE '2024-01-31'",
        # Before the 2000-01 origin, where a truncating month index rounds toward zero and
        # puts the row in the *next* bucket up. Only a floored one lands where DuckDB does.
        "DATE '1999-07-04'",
        "DATE '1970-01-01'",
        "TIMESTAMP '1987-11-30 23:59:59'",
    ],
)
def test_calendar_widths_match_duckdb(duck, width, value):
    """Calendar widths used to be refused; they are answered on the month index now.

    A month is not a number of microseconds, so `WindowStart` genuinely cannot express one
    — the refusal was right for the mechanism it had. Bucketing the *month index* instead
    needs no width that divides anything, and it reproduces DuckDB's 2000-01 origin exactly,
    including on the far side of it.
    """
    query = f"SELECT time_bucket(INTERVAL {width}, {value}) AS r"
    assert_same(bt.sql(query).collect(), duck.sql(query))


@pytest.mark.parametrize("value", ["DATE '2024-01-31'", _TS])
@pytest.mark.parametrize("width", ["1 DAY", "6 HOUR", "1 MONTH", "1 YEAR"])
def test_a_bucket_keeps_the_type_it_was_given(duck, width, value):
    """Bucketing a DATE yields a DATE in DuckDB; it used to widen to a timestamp here.

    The value was right and the *type* was not, which is the failure mode that survives an
    order-independent comparison: a `GROUP BY time_bucket(...)` key silently stopped being
    joinable against the date column it came from.
    """
    query = f"SELECT time_bucket(INTERVAL {width}, {value}) AS r"
    expected_type = duck.execute(query).arrow().read_all().schema.field("r").type
    actual = bt.sql(query).collect()
    assert actual.schema.field("r").type == expected_type
    assert_same(actual, duck.sql(query))


@pytest.mark.parametrize("width", ["0 MONTH", "-1 MONTH", "0 DAY"])
def test_a_non_positive_width_is_refused_with_a_reason(width):
    with pytest.raises(NotImplementedError, match="positive"):
        bt.sql(f"SELECT time_bucket(INTERVAL {width}, {_TS}) AS r").collect()
