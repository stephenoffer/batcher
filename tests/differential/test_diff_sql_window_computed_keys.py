"""Computed `PARTITION BY` / window `ORDER BY` keys in SQL, checked against DuckDB.

`rank() OVER (PARTITION BY date_trunc('month', ts) ORDER BY amount)` is how a monthly
ranking is written, and the translator used to refuse it with *"window PARTITION BY
supports plain columns only"*. The refusal was a translator gap rather than an engine one:
the `Window` operator has always taken an `Expr` key, so the fix is to hoist the computed
key into a hidden column exactly as `hoist_window_args` already did for a computed argument
(`sum(price * qty) OVER (...)`).

Two things beyond "it runs" are asserted here, because both are how the fix could be wrong
while still returning rows: the hidden columns must not reach the output, and two windows
partitioned on the *same* expression must share one hoisted column so they still collapse
into a single `Window` operator instead of partitioning the data twice.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table():
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "b", "a"]),
            "t": pa.array(
                [
                    dt.datetime(2024, 1, 5),
                    dt.datetime(2024, 1, 20),
                    dt.datetime(2024, 2, 3),
                    dt.datetime(2024, 2, 20),
                    dt.datetime(2024, 3, 1),
                ]
            ),
            "x": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "n": pa.array([3, 1, 4, 1, 5], type=pa.int64()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        # A computed PARTITION BY: the monthly bucket.
        "SELECT x, sum(x) OVER (PARTITION BY date_trunc('month', t)) AS s FROM ts",
        # A computed ORDER BY, alone and beside a plain one.
        "SELECT x, row_number() OVER (ORDER BY x * -1) AS r FROM ts",
        "SELECT x, rank() OVER (PARTITION BY g ORDER BY date_trunc('month', t), x) AS r FROM ts",
        # Both computed at once.
        "SELECT x, dense_rank() OVER (PARTITION BY g || '-' || g ORDER BY n % 3) AS r FROM ts",
        # A computed key beside a computed *argument*, which hoists through the other path.
        "SELECT x, sum(x * n) OVER (PARTITION BY date_trunc('month', t)) AS s FROM ts",
        # A computed key with an explicit frame.
        "SELECT x, sum(x) OVER ("
        "  PARTITION BY date_trunc('month', t) ORDER BY x"
        "  ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM ts",
        # A value function over a computed partition.
        "SELECT x, lag(x) OVER (PARTITION BY date_trunc('month', t) ORDER BY x) AS p FROM ts",
    ],
    ids=["partition", "order", "order_mixed", "both", "with_arg", "framed", "value_fn"],
)
def test_a_computed_window_key_matches_duckdb(duck, sql):
    table = _table()
    duck.register("ts", table)
    assert_same(bt.sql(sql, ts=table).collect(), duck.sql(sql))


def test_the_hoisted_key_does_not_reach_the_output(duck):
    """A hidden column that leaks is a schema change, which no value comparison catches."""
    table = _table()
    sql = "SELECT *, sum(x) OVER (PARTITION BY date_trunc('month', t)) AS s FROM ts"
    duck.register("ts", table)
    got = bt.sql(sql, ts=table)
    assert got.columns == ["g", "t", "x", "n", "s"]
    assert not any(c.startswith("__bc_") for c in got.columns)
    assert_same(got.collect(), duck.sql(sql))


def test_two_windows_on_one_computed_key_share_a_single_operator(duck):
    """The dedup: a key hoisted twice would split one `Window` into two doing the same work.

    Correctness would survive that, which is exactly why it needs its own assertion — the
    cost would not show up in any result comparison.
    """
    table = _table()
    sql = (
        "SELECT x,"
        " sum(x) OVER (PARTITION BY date_trunc('month', t)) AS s,"
        " avg(x) OVER (PARTITION BY date_trunc('month', t)) AS a"
        " FROM ts"
    )
    duck.register("ts", table)
    ds = bt.sql(sql, ts=table)
    assert_same(ds.collect(), duck.sql(sql))
    assert ds.explain().count("window") == 1, ds.explain()


def test_a_computed_key_still_works_through_a_named_window(duck):
    table = _table()
    sql = (
        "SELECT x, rank() OVER w AS r FROM ts"
        " WINDOW w AS (PARTITION BY date_trunc('month', t) ORDER BY x)"
    )
    duck.register("ts", table)
    assert_same(bt.sql(sql, ts=table).collect(), duck.sql(sql))


def test_a_key_the_hoist_cannot_lower_still_refuses_by_name():
    """A window nested inside a partition key is not lowerable (nor legal SQL elsewhere).

    What matters is that it refuses and *names the construct* rather than partitioning on
    something it silently mis-lowered.
    """
    table = _table()
    with pytest.raises(Exception, match=r"(?i)window"):
        bt.sql("SELECT sum(x) OVER (PARTITION BY sum(x) OVER ()) AS s FROM ts", ts=table).collect()
