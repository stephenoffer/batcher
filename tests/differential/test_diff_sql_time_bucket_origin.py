"""``time_bucket`` boundaries must land where DuckDB's do, or the call must be refused.

`WindowStart` anchors buckets at the Unix epoch. DuckDB anchors them at **2000-01-03**,
10,959 days later. The two grids coincide only when the bucket width divides that gap —
which is why the widths already under test looked right: 1 DAY, 2 HOUR and 5 MINUTE all
do. A width that does not put every boundary on a different instant, silently:

    time_bucket(INTERVAL 2 DAY, DATE '2021-01-01')   -- DuckDB 2020-12-31, Batcher 2021-01-01
    time_bucket(INTERVAL 7 DAY, DATE '2021-01-04')   -- DuckDB 2021-01-04, Batcher 2020-12-31

A whole week of rows lands in the neighbouring bucket, so a time-series aggregate reports
the wrong totals against the wrong periods and nothing raises. Rather than answer on a
shifted grid, a misaligned width is now refused — the same choice already made for
``INTERVAL 1 MONTH``, and for the same reason.

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


def test_calendar_widths_are_still_refused():
    with pytest.raises(NotImplementedError):
        bt.sql(f"SELECT time_bucket(INTERVAL 1 MONTH, {_TS}) AS r").collect()
