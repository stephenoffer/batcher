"""The compat spellings compute what DuckDB computes.

`tests/unit/test_expr_ergonomics.py` proves each alias builds the *same IR* as its
primary. That is the structural half. This is the semantic half: it runs the alias
through the engine and checks the answer against the oracle, so an alias pointed at
the wrong primary is caught by a wrong *result* rather than only by an IR diff.

Nulls are present in every fixture, because the null path is where an alias aimed at
a near-miss primary (``fill_null`` vs ``fill_nan``, ``is_null`` vs ``is_nan``) would
diverge.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def nums():
    return pa.table({"x": [1, None, 3, -7, 0], "y": [2, 5, None, 3, 4]})


def test_isna_and_notna_match_is_null_semantics(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(a=col("x").isna(), b=col("x").notna()).collect()
    assert_same(got, duck.execute("SELECT x IS NULL AS a, x IS NOT NULL AS b FROM t"))


def test_fillna_matches_coalesce(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").fillna(0)).collect()
    assert_same(got, duck.execute("SELECT coalesce(x, 0) AS r FROM t"))


def test_isin_matches_sql_in(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").isin([1, 3])).collect()
    assert_same(got, duck.execute("SELECT x IN (1, 3) AS r FROM t"))


def test_astype_matches_cast(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").astype("float64")).collect()
    assert_same(got, duck.execute("SELECT CAST(x AS DOUBLE) AS r FROM t"))


def test_astype_is_case_insensitive(duck, nums):
    """pandas spells this ``"Int64"``; the result must be the lowercase cast."""
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").astype("Float64")).collect()
    assert_same(got, duck.execute("SELECT CAST(x AS DOUBLE) AS r FROM t"))


@pytest.mark.parametrize(
    ("method", "sql"),
    [
        ("add", "x + y"),
        ("sub", "x - y"),
        ("mul", "x * y"),
        ("mod", "x % y"),
        ("eq", "x = y"),
        ("ne", "x <> y"),
        ("lt", "x < y"),
        ("le", "x <= y"),
        ("gt", "x > y"),
        ("ge", "x >= y"),
    ],
)
def test_operator_methods_match_sql(duck, nums, method, sql):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=getattr(col("x"), method)(col("y"))).collect()
    assert_same(got, duck.execute(f"SELECT {sql} AS r FROM t"))


def test_truediv_matches_sql_float_division(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").truediv(col("y"))).collect()
    assert_same(got, duck.execute("SELECT CAST(x AS DOUBLE) / y AS r FROM t"))


def test_boolean_methods_match_sql(duck):
    t = pa.table({"a": [True, True, False, None], "b": [True, False, False, True]})
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(u=col("a").and_(col("b")), v=col("a").or_(col("b")), w=col("a").not_())
        .collect()
    )
    assert_same(got, duck.execute("SELECT a AND b AS u, a OR b AS v, NOT a AS w FROM t"))


def test_nunique_matches_count_distinct(duck):
    t = pa.table({"g": ["a", "a", "b", "b"], "x": [1, 1, 2, None]})
    duck.register("t", t)
    got = bt.from_arrow(t).group_by("g").agg(r=col("x").nunique()).collect()
    assert_same(got, duck.execute("SELECT g, count(DISTINCT x) AS r FROM t GROUP BY g"))


# --- namespace aliases ---------------------------------------------------------------
@pytest.fixture
def strs():
    return pa.table({"s": ["123", "abc", "a1", " ", None]})


@pytest.mark.parametrize(
    ("method", "sql"),
    [
        ("isdigit", "s ~ '^[0-9]+$'"),
        ("isalpha", "s ~ '^[A-Za-z]+$'"),
        ("isalnum", "s ~ '^[A-Za-z0-9]+$'"),
    ],
)
def test_str_predicate_aliases_match_duckdb(duck, strs, method, sql):
    duck.register("t", strs)
    got = bt.from_arrow(strs).select(r=getattr(col("s").str, method)()).collect()
    assert_same(got, duck.execute(f"SELECT {sql} AS r FROM t"))


def test_strip_prefix_and_suffix_match_duckdb(duck):
    t = pa.table({"s": ["abcd", "xcd", None]})
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(a=col("s").str.strip_prefix("ab"), b=col("s").str.strip_suffix("cd"))
        .collect()
    )
    assert_same(
        got,
        duck.execute(
            "SELECT CASE WHEN starts_with(s, 'ab') THEN substr(s, 3) ELSE s END AS a, "
            "CASE WHEN ends_with(s, 'cd') THEN substr(s, 1, length(s) - 2) ELSE s END AS b "
            "FROM t"
        ),
    )


def test_dt_snake_case_aliases_match_duckdb(duck):
    t = pa.table({"d": pa.array([datetime.date(2024, 2, 15), None], type=pa.date32())})
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(
            a=col("d").dt.day_of_week(), b=col("d").dt.day_of_year(), c=col("d").dt.week_of_year()
        )
        .collect()
    )
    assert_same(
        got,
        duck.execute("SELECT isodow(d) AS a, dayofyear(d) AS b, weekofyear(d) AS c FROM t"),
    )


def test_list_aliases_match_their_primaries_through_the_engine():
    """DuckDB's list functions are 1-based, so the oracle here is Batcher's own primary.

    The IR equality is already covered in the unit test; this proves the pair also
    *executes* to the same values, which is what a mis-bound alias would break.
    """
    t = pa.table({"l": [[3, 1, 2], [], [5]]})
    ds = bt.from_arrow(t)
    alias = ds.select(a=col("l").list.lengths(), b=col("l").list.argmin()).to_pydict()
    primary = ds.select(a=col("l").list.len(), b=col("l").list.arg_min()).to_pydict()
    assert alias == primary
