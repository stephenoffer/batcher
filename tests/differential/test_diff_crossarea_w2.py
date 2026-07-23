"""Cross-area wave-2 defects — expression/SQL surface bugs pinned against DuckDB.

Each test states an input once, runs it through Batcher and through DuckDB, and asserts
the results agree. These are the literal-vs-column and SQL-frame divergences from the
wave-2 cross-area hunt; every test here fails on the pre-fix engine.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


# --- BUG 1: is_in with a NULL member (SQL three-valued IN) -----------------------------
def test_is_in_with_null_member_is_three_valued(duck):
    """``x IN (1, NULL)`` is True for a match, NULL otherwise — never raises, never False."""
    table = pa.table({"x": pa.array([1, 2, None], pa.int64())})
    duck.register("t", table)
    got = bt.from_arrow(table).select(r=bt.col("x").is_in([1, None])).collect()
    assert_same(got, duck.sql("SELECT (x IN (1, NULL)) AS r FROM t"))


def test_is_in_only_null_member(duck):
    """``x IN (NULL)`` is NULL for every row (no non-null member can ever match)."""
    table = pa.table({"x": pa.array([1, 2, None], pa.int64())})
    duck.register("t", table)
    got = bt.from_arrow(table).select(r=bt.col("x").is_in([None])).collect()
    assert_same(got, duck.sql("SELECT (x IN (NULL)) AS r FROM t"))


def test_is_in_mixed_null_and_values(duck):
    """A NULL member turns would-be-False rows into NULL, leaving matches True."""
    table = pa.table({"x": pa.array([1, 2, 3, 4, 5, None], pa.int64())})
    duck.register("t", table)
    got = bt.from_arrow(table).select(r=bt.col("x").is_in([1, 3, 5, None])).collect()
    assert_same(got, duck.sql("SELECT (x IN (1, 3, 5, NULL)) AS r FROM t"))


def test_is_in_no_null_still_two_valued(duck):
    """A NULL-free IN list stays ordinary two-valued membership (regression guard)."""
    table = pa.table({"x": pa.array([1, 2, 3], pa.int64())})
    duck.register("t", table)
    got = bt.from_arrow(table).select(r=bt.col("x").is_in([1, 3])).collect()
    assert_same(got, duck.sql("SELECT (x IN (1, 3)) AS r FROM t"))


# --- BUG 2: arg_max/arg_min/first/last take a column, not a string literal --------------
def test_arg_max_string_names_a_column(duck):
    """``arg_max(v, "k")`` orders by column ``k`` — not the constant string ``'k'``."""
    table = pa.table({"g": ["a", "a", "a"], "v": [10, 20, 30], "k": [1, 5, 2]})
    duck.register("t", table)
    got = bt.from_arrow(table).group_by("g").agg(r=bt.col("v").arg_max("k")).collect()
    assert_same(got, duck.sql("SELECT g, arg_max(v, k) AS r FROM t GROUP BY g"))


def test_arg_min_string_names_a_column(duck):
    """``arg_min(v, "k")`` orders by column ``k``."""
    table = pa.table({"g": ["a", "a", "a"], "v": [10, 20, 30], "k": [1, 5, 2]})
    duck.register("t", table)
    got = bt.from_arrow(table).group_by("g").agg(r=bt.col("v").arg_min("k")).collect()
    assert_same(got, duck.sql("SELECT g, arg_min(v, k) AS r FROM t GROUP BY g"))


def test_arg_max_string_equals_col_form():
    """The string spelling must equal the explicit ``col`` spelling."""
    table = pa.table({"g": ["a", "a", "a"], "v": [10, 20, 30], "k": [1, 5, 2]})
    ds = bt.from_arrow(table)
    by_str = ds.group_by("g").agg(r=bt.col("v").arg_max("k")).collect().to_pydict()
    by_col = ds.group_by("g").agg(r=bt.col("v").arg_max(bt.col("k"))).collect().to_pydict()
    assert by_str == by_col


def test_first_last_string_names_a_column(duck):
    """``first(v, "k")`` / ``last(v, "k")`` order by column ``k`` (arg_min/arg_max)."""
    table = pa.table({"g": ["a", "a", "a"], "v": [10, 20, 30], "k": [1, 5, 2]})
    duck.register("t", table)
    got = (
        bt.from_arrow(table)
        .group_by("g")
        .agg(lo=bt.col("v").first("k"), hi=bt.col("v").last("k"))
        .collect()
    )
    assert_same(
        got, duck.sql("SELECT g, arg_min(v, k) AS lo, arg_max(v, k) AS hi FROM t GROUP BY g")
    )


# --- BUG 3: str.left with negative n drops the trailing |n| chars (DuckDB) --------------
@pytest.mark.parametrize("n", [3, 0, -2, -10, 6, -6, 1, -1])
def test_str_left_negative(duck, n):
    """``left(s, n)`` matches DuckDB across n>0, n=0, n<0, and |n|>len."""
    table = pa.table({"s": ["abcdef", "", "x", None]})
    duck.register("t", table)
    got = bt.from_arrow(table).select(r=bt.col("s").str.left(n)).collect()
    assert_same(got, duck.sql(f"SELECT left(s, {n}) AS r FROM t"))


# --- BUG 4: SQL last_value / nth_value use the default running frame (cross-area) -------
# The engine's value_window (bc-runtime/src/window.rs) selects the whole-partition
# position for last_value / nth_value and ignores the frame, so SQL last_value(v)
# OVER (ORDER BY i) returns the running value per the SQL default frame (ledger B90/B170,
# fixed by the frame-aware value_window in bc-runtime).
def test_sql_last_value_running_frame(duck):
    """``last_value(v) OVER (ORDER BY i)`` is the running value [10, 20, 30], not [30, 30, 30]."""
    table = pa.table({"i": [1, 2, 3], "v": [10, 20, 30]})
    duck.register("t", table)
    got = bt.sql(
        "SELECT i, last_value(v) OVER (ORDER BY i) AS v2 FROM t ORDER BY i",
        t=bt.from_arrow(table),
    ).collect()
    assert_same_ordered(
        got, duck.sql("SELECT i, last_value(v) OVER (ORDER BY i) AS v2 FROM t ORDER BY i")
    )
