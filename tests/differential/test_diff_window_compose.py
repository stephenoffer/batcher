"""Window expressions composed with ordinary arithmetic match DuckDB exactly.

A `WindowExpr` has no scalar IR — it is hoisted out of the surrounding expression
into a `Window` operator, leaving a synthetic column behind
(`plan.expr_rewrite.hoist_windows`). These cases pin that the desugaring computes
what SQL computes for the same query: a window inside arithmetic, two windows in one
expression, a window nested in a window's argument, a window in a `WHERE`-style
predicate, and the running-total / share-of-total shapes an analyst actually writes.

Also pinned here: ``a / b`` is *true* division. Batcher lowers ``/`` to
``div(cast(a, float64), b)`` so integer operands divide to a float, as in DuckDB,
Python and Polars — rather than truncating.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b", "b"], type=pa.string()),
            "x": pa.array([10, 20, 30, 7, None, 21], type=pa.int64()),
        }
    )


@pytest.fixture
def ds(duck):
    t = _t()
    duck.register("t", t)
    return bt.from_arrow(t)


def test_window_inside_arithmetic_matches_duckdb(duck, ds):
    """``x - lag(x) OVER (ORDER BY id)`` — the first-difference shape."""
    got = ds.select("id", d=col("x") - col("x").shift(1).over(order_by=["id"])).to_arrow()
    assert_same(got, duck.sql("SELECT id, x - lag(x) OVER (ORDER BY id) AS d FROM t"))


def test_two_windows_in_one_expression_matches_duckdb(duck, ds):
    """Two independent windows combined — each hoists to its own `Window` node."""
    got = ds.select(
        "id",
        spread=col("x").shift(-1).over(order_by=["id"]) - col("x").shift(1).over(order_by=["id"]),
    ).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, lead(x) OVER (ORDER BY id) - lag(x) OVER (ORDER BY id) AS spread FROM t"
        ),
    )


def test_share_of_group_total_matches_duckdb(duck, ds):
    """``x / sum(x) OVER (PARTITION BY g)`` — true division of an int by an int window."""
    got = ds.select("id", share=col("x") / col("x").sum().over(partition_by=["g"])).to_arrow()
    assert_same(got, duck.sql("SELECT id, x / sum(x) OVER (PARTITION BY g) AS share FROM t"))


def test_running_total_over_partition_matches_duckdb(duck, ds):
    """A running total composed with arithmetic (``+ 0`` forces the hoist path)."""
    got = ds.select("id", rt=col("x").cum_sum(partition_by=["g"], order_by=["id"]) + 0).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, sum(x) OVER (PARTITION BY g ORDER BY id "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) + 0 AS rt FROM t"
        ),
    )


def test_window_nested_in_window_argument_matches_duckdb(duck, ds):
    """``sum(lag(x)) OVER (...)`` — the inner window materializes first."""
    got = ds.select(
        "id", z=col("x").shift(1).over(order_by=["id"]).cum_sum(order_by=["id"])
    ).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, sum(prev) OVER (ORDER BY id "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS z "
            "FROM (SELECT id, lag(x) OVER (ORDER BY id) AS prev FROM t)"
        ),
    )


def test_filter_on_window_matches_duckdb_subquery(duck, ds):
    """A window in a predicate: keep rows above their group mean.

    SQL forbids a window in `WHERE`, so the oracle is the subquery Batcher desugars
    to — the window sees every input row, before the filter.
    """
    got = ds.filter(col("x") > col("x").mean().over(partition_by=["g"])).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, g, x FROM ("
            "  SELECT *, avg(x) OVER (PARTITION BY g) AS m FROM t"
            ") WHERE x > m"
        ),
    )


def test_filter_on_window_preserves_input_schema(ds):
    """The synthetic hoist column never reaches the output schema."""
    out = ds.filter(col("x") > col("x").mean().over(partition_by=["g"]))
    assert out.columns == ["id", "g", "x"]


def test_mixed_predicate_survives_pushdown_through_the_window(duck, ds):
    """One conjunct is on the partition key, so `push_filter_through_window` moves it
    *below* the `Window` this hoist created. That is sound only because such a
    predicate drops whole partitions — pin the result, because a wrong guard here
    would silently change what the window sees."""
    pred = (col("g") == "a") & (col("x") > col("x").mean().over(partition_by=["g"]))
    assert_same(
        ds.filter(pred).to_arrow(),
        duck.sql(
            "SELECT id, g, x FROM (SELECT *, avg(x) OVER (PARTITION BY g) AS m FROM t) "
            "WHERE g = 'a' AND x > m"
        ),
    )


def test_pushable_conjunct_beside_is_unique_matches_duckdb(duck, ds):
    """`is_unique()` partitions by `x`, so a predicate on `x` is pushable — and must
    not change which values count as unique."""
    got = ds.filter(col("x").is_unique() & (col("x") > 15)).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT id, g, x FROM (SELECT *, count(1) OVER (PARTITION BY x) AS c FROM t) "
            "WHERE c = 1 AND x > 15"
        ),
    )


def test_rank_composed_with_arithmetic_matches_duckdb(duck, ds):
    """A value window function used as an operand."""
    got = ds.select("id", r0=bt.rank().over(order_by=["id"]) - 1).to_arrow()
    assert_same(got, duck.sql("SELECT id, rank() OVER (ORDER BY id) - 1 AS r0 FROM t"))


def test_window_mixed_with_plain_columns_in_with_columns(duck, ds):
    """Window and non-window columns in one `with_columns` call."""
    got = ds.with_columns(y=col("x") + 1, total=col("x").sum().over(partition_by=["g"])).to_arrow()
    assert_same(
        got,
        duck.sql("SELECT id, g, x, x + 1 AS y, sum(x) OVER (PARTITION BY g) AS total FROM t"),
    )


@pytest.mark.parametrize(
    "expr_sql",
    ["a / b", "a / 4", "7 / b", "a / 2.0"],
)
def test_true_division_matches_duckdb(duck, expr_sql):
    """``/`` is true division: integer operands yield a float, as in DuckDB."""
    t = pa.table(
        {
            "a": pa.array([1, 7, -9, 0, None], type=pa.int64()),
            "b": pa.array([2, 3, 4, 5, 2], type=pa.int64()),
        }
    )
    duck.register("d", t)
    got = bt.from_arrow(t).sql(f"SELECT {expr_sql} AS r FROM self").to_arrow()
    assert_same(got, duck.sql(f"SELECT {expr_sql} AS r FROM d"))


def test_true_division_python_operator_matches_duckdb(duck):
    """The Python ``/`` operator agrees with SQL ``/`` and with DuckDB."""
    t = pa.table({"a": pa.array([1, 7], type=pa.int64()), "b": pa.array([2, 3], type=pa.int64())})
    duck.register("d", t)
    got = bt.from_arrow(t).select(r=col("a") / col("b")).to_arrow()
    assert_same(got, duck.sql("SELECT a / b AS r FROM d"))
