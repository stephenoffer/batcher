"""Lakehouse partition transforms, against DuckDB and against the Iceberg specification.

`bt.partition_years`/`_months`/`_days`/`_hours`/`_truncate` compute the value a
partitioned table stores beside a row. Getting one wrong does not raise: it writes the
row into a partition the table's own writer would not have chosen, and the table stays
readable and silently mis-clustered. So each transform is pinned twice — against DuckDB
computing the same definition out of its own primitives, and against the literal values
the Iceberg specification states.

Both halves matter and neither is redundant. The DuckDB half catches an arithmetic slip;
the literal half catches a *definitional* one, which DuckDB cannot see because the query
would then be wrong in both engines. The three definitional traps are pre-1970 values
(negative, not zero and not an error), the epoch offset (`months` counts from 1970-01,
so January 1970 is 0 and not 1), and the floored remainder in `truncate`, which is the
one place the engine's own `%` gives the wrong answer.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _tbl() -> pa.Table:
    """Timestamps either side of the epoch, plus the integers `truncate` is defined on."""
    return pa.table(
        {
            "ts": pa.array(
                [
                    dt.datetime(2024, 3, 5, 13, 45, 30),
                    dt.datetime(1970, 1, 1, 0, 0, 0),
                    dt.datetime(1969, 12, 31, 23, 0, 0),
                    dt.datetime(1969, 6, 1, 12, 0, 0),
                    None,
                ],
                pa.timestamp("us"),
            ),
            "n": pa.array([17, 0, -1, -7, None], pa.int64()),
        }
    )


def test_time_transforms_match_duckdb_computing_the_same_definition(duck):
    """Years/months/days/hours against DuckDB's own date arithmetic."""
    duck.register("t", _tbl())
    out = bt.from_arrow(_tbl()).select(
        y=bt.partition_years("ts"),
        m=bt.partition_months("ts"),
        d=bt.partition_days("ts"),
        h=bt.partition_hours("ts"),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT year(ts) - 1970 AS y,"
            "       (year(ts) - 1970) * 12 + month(ts) - 1 AS m,"
            "       datediff('day', DATE '1970-01-01', ts) AS d,"
            "       CAST(floor(epoch(ts) / 3600) AS BIGINT) AS h "
            "FROM t"
        ),
    )


def test_truncate_matches_duckdbs_floored_remainder(duck):
    """The floored remainder, written out in DuckDB rather than assumed."""
    duck.register("t", _tbl())
    out = bt.from_arrow(_tbl()).select(p=bt.partition_truncate("n", 5))
    assert_same(out.collect(), duck.sql("SELECT n - ((n % 5) + 5) % 5 AS p FROM t"))


def test_the_specifications_own_numbers_including_before_the_epoch():
    """The literal values Iceberg states, which a DuckDB cross-check cannot supply.

    A wrong *definition* is wrong in both engines at once, so this half is the one that
    holds the epoch offset and the sign convention.
    """
    out = (
        bt.from_arrow(_tbl())
        .select(
            y=bt.partition_years("ts"),
            m=bt.partition_months("ts"),
            d=bt.partition_days("ts"),
            h=bt.partition_hours("ts"),
        )
        .to_pydict()
    )
    assert out["y"] == [54, 0, -1, -1, None]
    # 1970-01 is month 0, so December 1969 is -1 and June 1969 is -7.
    assert out["m"] == [650, 0, -1, -7, None]
    assert out["d"] == [19787, 0, -1, -214, None]
    # The hour before the epoch is -1, not 0: the transform floors, it does not truncate.
    assert out["h"] == [474901, 0, -1, -5124, None]


def test_truncate_floors_toward_negative_infinity():
    """`v - (v % W)` gives -5 here; the specification says -10."""
    out = bt.from_arrow(_tbl()).select(p=bt.partition_truncate("n", 5)).to_pydict()
    assert out["p"] == [15, 0, -5, -10, None]


def test_truncate_rejects_a_non_positive_width():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="positive integer"):
        bt.partition_truncate("n", 0)
    with pytest.raises(PlanError, match="positive integer"):
        bt.partition_truncate("n", -3)


def test_the_sql_spelling_is_the_same_function():
    """SQL and the DataFrame surface must not be two implementations."""
    src = bt.from_arrow(_tbl())
    frame = src.select(
        y=bt.partition_years("ts"),
        m=bt.partition_months("ts"),
        d=bt.partition_days("ts"),
        h=bt.partition_hours("ts"),
        p=bt.partition_truncate("n", 5),
    ).to_pydict()
    sql = bt.sql(
        "SELECT partition_years(ts) y, partition_months(ts) m, partition_days(ts) d,"
        "       partition_hours(ts) h, partition_truncate(n, 5) p FROM t",
        t=src,
    ).to_pydict()
    assert frame == sql


def test_sql_truncate_takes_the_text_reading_for_a_string_column():
    """Iceberg's `truncate` on text is a prefix; only SQL knows the column's type.

    The DataFrame builder has no type in scope and so serves the numeric reading alone
    (`.str.substr` is the text spelling there). The SQL translator does, so the one name
    reaches both — and this pins that it dispatches on the *type* rather than on arity.
    """
    src = bt.from_pydict({"s": ["abcdef", "xy", None], "n": [17, -7, None]})
    out = bt.sql("SELECT partition_truncate(s, 3) s, partition_truncate(n, 5) n FROM t", t=src)
    assert out.to_pydict() == {"s": ["abc", "xy", None], "n": [15, -10, None]}


def test_a_partition_value_is_stable_across_the_distributed_path():
    """A transform that disagreed with itself across partitions would mis-place rows."""
    rows = {"ts": [dt.datetime(2024, 1, 1) + dt.timedelta(hours=i) for i in range(200)]}
    ds = bt.from_pydict(rows).select(h=bt.partition_hours("ts"), d=bt.partition_days("ts"))
    assert ds.collect().to_pydict() == ds.collect(distributed=True, num_workers=3).to_pydict()
