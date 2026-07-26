"""Window aggregates beyond sum/avg/min/max/count — vs DuckDB.

DuckDB, Spark and Polars all let *any* aggregate be a window function; this engine had
five, and `col("x").std().over("g")` raised `unknown window function 'stddev'`. Nine more
are supported: `var`, `stddev`, `product`, `bool_and`, `bool_or`, `bit_and`, `bit_or`,
`bit_xor` and `count_distinct` — the ones whose running form costs O(1) per row.

Both window shapes are exercised, because they are different kernels computing different
relations from the same operator:

* **whole-partition** (no ORDER BY): one value per partition, broadcast to its rows;
* **running** (with ORDER BY): the accumulation up to and including each row's peer
  group, which is SQL's default `RANGE` frame.

Comparisons are ordered (`assert_same_ordered` over a sorted result), not the usual
multiset: a running aggregate is a *sequence*, and the order-independent comparison the
rest of the suite uses cannot see a running kernel that emits the right values in the
wrong order.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

# (label, the Batcher aggregate, the DuckDB aggregate, the column it reads)
AGGREGATES = [
    ("var", lambda c: c.var(), "var_samp", "x"),
    ("stddev", lambda c: c.std(), "stddev", "x"),
    ("product", lambda c: c.product(), "product", "x"),
    ("bool_and", lambda c: c.bool_and(), "bool_and", "flag"),
    ("bool_or", lambda c: c.bool_or(), "bool_or", "flag"),
    ("bit_and", lambda c: c.bit_and(), "bit_and", "i"),
    ("bit_or", lambda c: c.bit_or(), "bit_or", "i"),
    ("bit_xor", lambda c: c.bit_xor(), "bit_xor", "i"),
    ("count_distinct", lambda c: c.n_unique(), "count(DISTINCT %s)", "i"),
]


@pytest.fixture
def table(duck):
    """Groups of different sizes, with a singleton (where sample variance is NULL), a
    null in every value column, and repeated values so `count_distinct` is not the
    row count."""
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "b", "c", "d"],
            "o": [1, 2, 3, 1, 2, 3, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 5.0, None],
            "i": [6, 4, 6, 12, 8, 3, 7, None],
            "flag": [True, False, True, True, True, False, True, None],
        }
    )
    duck.register("t", t)
    return t


def _duck_call(agg: str, column: str) -> str:
    return agg % column if "%s" in agg else f"{agg}({column})"


@pytest.mark.differential
@pytest.mark.parametrize(("label", "batcher_agg", "duck_agg", "column"), AGGREGATES)
def test_whole_partition_matches_duckdb(duck, table, label, batcher_agg, duck_agg, column):
    out = (
        bt.from_arrow(table)
        .select(g=col("g"), o=col("o"), r=batcher_agg(col(column)).over("g"))
        .sort("g", "o")
        .collect()
    )
    expected = duck.sql(
        f"SELECT g, o, {_duck_call(duck_agg, column)} OVER (PARTITION BY g) r FROM t ORDER BY g, o"
    )
    assert_same_ordered(out, expected)


@pytest.mark.differential
@pytest.mark.parametrize(("label", "batcher_agg", "duck_agg", "column"), AGGREGATES)
def test_running_matches_duckdb(duck, table, label, batcher_agg, duck_agg, column):
    out = (
        bt.from_arrow(table)
        .select(g=col("g"), o=col("o"), r=batcher_agg(col(column)).over("g", order_by="o"))
        .sort("g", "o")
        .collect()
    )
    expected = duck.sql(
        f"SELECT g, o, {_duck_call(duck_agg, column)} OVER (PARTITION BY g ORDER BY o) r "
        "FROM t ORDER BY g, o"
    )
    assert_same_ordered(out, expected)


@pytest.mark.differential
@pytest.mark.parametrize(("label", "batcher_agg", "duck_agg", "column"), AGGREGATES)
def test_unpartitioned_matches_duckdb(duck, table, label, batcher_agg, duck_agg, column):
    """No PARTITION BY at all — one partition over every row."""
    out = (
        bt.from_arrow(table)
        .select(g=col("g"), o=col("o"), r=batcher_agg(col(column)).over(order_by="o"))
        .sort("g", "o")
        .collect()
    )
    expected = duck.sql(
        f"SELECT g, o, {_duck_call(duck_agg, column)} OVER (ORDER BY o) r FROM t ORDER BY g, o"
    )
    assert_same_ordered(out, expected)


@pytest.mark.differential
def test_peer_rows_share_the_running_value(duck, table):
    """Tied ORDER BY rows share their peer group's value — SQL's RANGE semantics.

    This is the property a running kernel gets wrong if it emits per row instead of per
    peer group, and the one an order-independent comparison cannot see.
    """
    ties = pa.table({"o": [1, 1, 2, 2, 3], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    duck.register("ties", ties)
    out = (
        bt.from_arrow(ties)
        .select(o=col("o"), r=col("x").product().over(order_by="o"))
        .sort("o", "r")
        .collect()
    )
    expected = duck.sql("SELECT o, product(x) OVER (ORDER BY o) r FROM ties ORDER BY o, r")
    assert_same_ordered(out, expected)
    assert out.column("r").to_pylist() == [2.0, 2.0, 24.0, 24.0, 120.0]


@pytest.mark.differential
def test_a_window_aggregate_agrees_with_the_group_by_aggregate(duck, table):
    """The whole-partition window value must equal the `GROUP BY` value for that group.

    `var`/`stddev` keep the same Welford recurrence in both places on purpose, so this
    holds by construction; the test is what makes "by construction" checkable. See
    `test_variance_survives_a_large_mean_with_a_small_spread` for what happened when the
    two states differed.
    """
    windowed = (
        bt.from_arrow(table)
        .select(g=col("g"), v=col("x").var().over("g"), s=col("x").std().over("g"))
        .unique()
        .sort("g")
        .collect()
    )
    grouped = duck.sql("SELECT g, var_samp(x) v, stddev(x) s FROM t GROUP BY g ORDER BY g")
    assert_same_ordered(windowed, grouped)


@pytest.mark.differential
def test_variance_survives_a_large_mean_with_a_small_spread(duck):
    """The cancellation case that decides which recurrence the kernel may use.

    With `(n, Sx, Sx^2)` the variance is recovered by subtracting two nearly equal large
    numbers, and over `[1e9+1, 1e9+2, 1e9+3]` that returns exactly `0` where the answer is
    `1`. `agg/var.rs` moved to Welford for this; the first version of the *window* kernel
    did not, and its fixture (values 1 through 8) could not see the difference — the
    window aggregate disagreed with the `GROUP BY` aggregate by a factor of infinity.
    """
    big = pa.table({"g": ["a", "a", "a"], "x": [1e9 + 1, 1e9 + 2, 1e9 + 3]})
    duck.register("big", big)
    out = bt.from_arrow(big).select(v=col("x").var().over("g"), s=col("x").std().over("g"))
    assert out.to_pydict()["v"] == [1.0, 1.0, 1.0]
    assert out.to_pydict()["s"] == [1.0, 1.0, 1.0]
    expected = duck.sql("SELECT var_samp(x) v, stddev(x) s FROM big")
    grouped = bt.from_arrow(big).agg(v=col("x").var(), s=col("x").std()).collect()
    assert_same_ordered(grouped, expected)


@pytest.mark.differential
def test_an_explicit_frame_is_refused_for_the_extended_aggregates():
    """The framed path has a hand-written sliding kernel per function and has none for
    these, so a frame is rejected at plan time rather than silently ignored."""
    ds = bt.from_pydict({"g": ["a", "a"], "o": [1, 2], "x": [1.0, 2.0]})
    with pytest.raises(Exception, match="frame"):
        ds.select(r=col("x").product().over("g", order_by="o", frame=(-1, 0))).collect()
