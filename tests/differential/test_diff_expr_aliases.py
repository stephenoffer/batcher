"""The spellings users reach for from pandas/Polars/SQL compute what DuckDB computes.

Each spelling here once had a second, compat name; those were removed so each capability
has one name. The differential checks stay on the kept spelling, so a kept name bound to
the wrong implementation is caught by a wrong *result*.

Nulls are present in every fixture, because the null path is where a function aimed at
a near-miss implementation (``fill_null`` vs ``fill_nan``, ``is_null`` vs ``is_nan``)
would diverge.
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
    got = bt.from_arrow(nums).select(a=col("x").is_null(), b=col("x").is_not_null()).collect()
    assert_same(got, duck.execute("SELECT x IS NULL AS a, x IS NOT NULL AS b FROM t"))


def test_fillna_matches_coalesce(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").fill_null(0)).collect()
    assert_same(got, duck.execute("SELECT coalesce(x, 0) AS r FROM t"))


def test_isin_matches_sql_in(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").is_in([1, 3])).collect()
    assert_same(got, duck.execute("SELECT x IN (1, 3) AS r FROM t"))


def test_astype_matches_cast(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").cast("float64")).collect()
    assert_same(got, duck.execute("SELECT CAST(x AS DOUBLE) AS r FROM t"))


def test_astype_is_case_insensitive(duck, nums):
    """pandas spells this ``"Int64"``; the result must be the lowercase cast."""
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=col("x").cast("Float64")).collect()
    assert_same(got, duck.execute("SELECT CAST(x AS DOUBLE) AS r FROM t"))


@pytest.mark.parametrize(
    ("op", "sql"),
    [
        (lambda a, b: a + b, "x + y"),
        (lambda a, b: a - b, "x - y"),
        (lambda a, b: a * b, "x * y"),
        (lambda a, b: a % b, "x % y"),
        (lambda a, b: a == b, "x = y"),
        (lambda a, b: a != b, "x <> y"),
        (lambda a, b: a < b, "x < y"),
        (lambda a, b: a <= b, "x <= y"),
        (lambda a, b: a > b, "x > y"),
        (lambda a, b: a >= b, "x >= y"),
    ],
    ids=["add", "sub", "mul", "mod", "eq", "ne", "lt", "le", "gt", "ge"],
)
def test_operators_match_sql(duck, nums, op, sql):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=op(col("x"), col("y"))).collect()
    assert_same(got, duck.execute(f"SELECT {sql} AS r FROM t"))


def test_truediv_matches_sql_float_division(duck, nums):
    duck.register("t", nums)
    got = bt.from_arrow(nums).select(r=(col("x") / col("y"))).collect()
    assert_same(got, duck.execute("SELECT CAST(x AS DOUBLE) / y AS r FROM t"))


def test_boolean_methods_match_sql(duck):
    t = pa.table({"a": [True, True, False, None], "b": [True, False, False, True]})
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(u=(col("a") & col("b")), v=(col("a") | col("b")), w=(~col("a")))
        .collect()
    )
    assert_same(got, duck.execute("SELECT a AND b AS u, a OR b AS v, NOT a AS w FROM t"))


def test_nunique_matches_count_distinct(duck):
    t = pa.table({"g": ["a", "a", "b", "b"], "x": [1, 1, 2, None]})
    duck.register("t", t)
    got = bt.from_arrow(t).group_by("g").agg(r=col("x").count_distinct()).collect()
    assert_same(got, duck.execute("SELECT g, count(DISTINCT x) AS r FROM t GROUP BY g"))


# --- namespace methods ---------------------------------------------------------------
@pytest.fixture
def strs():
    return pa.table({"s": ["123", "abc", "a1", " ", None]})


@pytest.mark.parametrize(
    ("method", "sql"),
    [
        ("is_numeric", "s ~ '^[0-9]+$'"),
        ("is_alpha", "s ~ '^[A-Za-z]+$'"),
        ("is_alnum", "s ~ '^[A-Za-z0-9]+$'"),
    ],
)
def test_str_predicates_match_duckdb(duck, strs, method, sql):
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


def test_dt_day_numbering_matches_duckdb(duck):
    """The `.dt` day spellings resolve to the DuckDB function they are named for.

    The fixture spans a full week including a Sunday, and `dayofweek` is checked against
    DuckDB's `dayofweek`. Both details are load-bearing: `dayofweek` is Sunday=0, not
    `isodow` (Sunday=7), and the two numberings agree on every day *except* Sunday. A
    single-Thursday fixture compared against `isodow` passes whichever function the name
    is bound to, which is how a removed alias came to be documented as ISO while behaving
    as Sunday=0.
    """
    week = [datetime.date(2024, 2, 12) + datetime.timedelta(days=i) for i in range(7)]
    t = pa.table({"d": pa.array([*week, None], type=pa.date32())})
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(a=col("d").dt.dayofweek(), b=col("d").dt.dayofyear(), c=col("d").dt.week())
        .collect()
    )
    assert_same(
        got,
        duck.execute("SELECT dayofweek(d) AS a, dayofyear(d) AS b, weekofyear(d) AS c FROM t"),
    )
    # ...and the ISO spellings really are the other convention, on the day that separates
    # them. Without this the two families could both be bound to `dayofweek` undetected.
    iso = bt.from_arrow(t).select(a=col("d").dt.weekday()).collect()
    assert_same(iso, duck.execute("SELECT isodow(d) AS a FROM t"))
