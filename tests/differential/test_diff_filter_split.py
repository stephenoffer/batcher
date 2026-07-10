"""Cost-based filter splitting preserves results vs DuckDB.

`split_expensive_filter` rewrites `Filter(cheap AND expensive)` into stacked `Filter`s so
the expensive predicate only sees rows the cheap one kept. Splitting an `AND` is exact
under Kleene logic — `filter` keeps only TRUE, and `and_kleene(a, b)` is TRUE exactly
when both are — but that is precisely the kind of claim that must be proven against the
oracle rather than argued.

The cases below cover what makes the rewrite non-obvious: NULLs in the cheap column, the
expensive column, and both; a predicate that keeps everything and one that keeps nothing;
an empty input; multi-conjunct predicates where the rule must choose a split point; and
the `OR` / single-predicate shapes where it must not fire at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

# Registers the @rule on import so the full optimizer applies it under `.collect()`.
import batcher.kyber.rules.extra.filter_split
from batcher import col
from conftest import assert_same


def _table() -> pa.Table:
    return pa.table(
        {
            "x": pa.array([1, 2, None, 60, 70, None, 95, 80, 55, 99], pa.int64()),
            "y": pa.array([9, 8, 7, 6, None, 4, 3, 2, 1, None], pa.int64()),
            "s": pa.array(
                ["abc1", "abc9", None, "abc9", "xyz", None, "abc9", None, "abz", "abc9"],
                pa.string(),
            ),
        }
    )


@pytest.fixture
def data(duck):
    t = _table()
    duck.register("t", t)
    return bt.from_arrow(t)


@pytest.mark.differential
def test_cheap_and_expensive_conjunction(data, duck):
    got = data.filter((col("x") > 50) & col("s").str.regexp_matches("^abc9")).collect()
    assert_same(got, duck.sql("select * from t where x > 50 and regexp_matches(s, '^abc9')"))


@pytest.mark.differential
def test_expensive_conjunct_written_first(data, duck):
    # Author order must not matter: the rule reorders by rank, `AND` is commutative.
    got = data.filter(col("s").str.regexp_matches("^abc9") & (col("x") > 50)).collect()
    assert_same(got, duck.sql("select * from t where regexp_matches(s, '^abc9') and x > 50"))


@pytest.mark.differential
def test_three_conjuncts_with_nulls_everywhere(data, duck):
    got = data.filter(
        (col("x") > 50) & col("s").str.regexp_matches("^abc") & (col("y") < 7)
    ).collect()
    assert_same(
        got,
        duck.sql("select * from t where x > 50 and regexp_matches(s, '^abc') and y < 7"),
    )


@pytest.mark.differential
def test_two_expensive_conjuncts(data, duck):
    got = data.filter(
        col("s").str.regexp_matches("^abc") & col("s").str.regexp_matches("9$")
    ).collect()
    assert_same(
        got,
        duck.sql("select * from t where regexp_matches(s,'^abc') and regexp_matches(s,'9$')"),
    )


@pytest.mark.differential
def test_predicate_keeping_every_row(data, duck):
    got = data.filter((col("x") > -1000) & col("s").str.regexp_matches("")).collect()
    assert_same(got, duck.sql("select * from t where x > -1000 and regexp_matches(s, '')"))


@pytest.mark.differential
def test_predicate_keeping_no_row(data, duck):
    got = data.filter((col("x") > 1000) & col("s").str.regexp_matches("^abc9")).collect()
    assert_same(got, duck.sql("select * from t where x > 1000 and regexp_matches(s, '^abc9')"))


@pytest.mark.differential
def test_empty_input(duck):
    t = pa.table(
        {
            "x": pa.array([], pa.int64()),
            "s": pa.array([], pa.string()),
        }
    )
    duck.register("e", t)
    got = bt.from_arrow(t).filter((col("x") > 5) & col("s").str.contains("a")).collect()
    assert_same(got, duck.sql("select * from e where x > 5 and contains(s, 'a')"))


@pytest.mark.differential
def test_disjunction_is_untouched(data, duck):
    # The rule must not fire; the result must still match.
    got = data.filter((col("x") > 90) | col("s").str.regexp_matches("^xyz")).collect()
    assert_same(got, duck.sql("select * from t where x > 90 or regexp_matches(s, '^xyz')"))


@pytest.mark.differential
def test_split_below_an_aggregate(data, duck):
    got = (
        data.filter((col("x") > 50) & col("s").str.regexp_matches("^abc9"))
        .group_by("s")
        .agg(n=col("x").count())
        .collect()
    )
    assert_same(
        got,
        duck.sql(
            "select s, count(x) as n from t where x > 50 and regexp_matches(s, '^abc9') group by s"
        ),
    )


@pytest.mark.differential
def test_split_feeds_a_join(duck):
    left = _table()
    right = pa.table({"x": pa.array([60, 95, 99, 12], pa.int64()), "tag": ["a", "b", "c", "d"]})
    duck.register("l", left)
    duck.register("r", right)
    got = (
        bt.from_arrow(left)
        .filter((col("x") > 50) & col("s").str.regexp_matches("^abc9"))
        .join(bt.from_arrow(right), on="x")
        .collect()
    )
    assert_same(
        got,
        duck.sql(
            "select * from l join r using (x) where l.x > 50 and regexp_matches(l.s, '^abc9')"
        ),
    )
