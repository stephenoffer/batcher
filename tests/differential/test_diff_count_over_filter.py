"""`COUNT(*) ... WHERE p` matches DuckDB — including when nothing reaches the filter.

Kyber rewrites a keyless `COUNT(*)` over a `Filter` into a count of the predicate mask,
so the filtered batch is never built. The rewrite must preserve the aggregate's
**identity**: `COUNT` over zero rows is 0, but `SUM` over zero rows is NULL. Desugaring
the mask to `sum(iff(p, 1, 0))` therefore turned `SELECT COUNT(*) FROM empty WHERE p`
into NULL — a wrong answer that only appears once the filter's input is empty (an empty
join, a fully-pruned scan), which is precisely when a count is being used as a guard.

The empty cases below are the regression; the non-empty ones pin that the rewrite still
agrees with DuckDB on the ordinary path, nulls included (a null predicate drops the row).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    table = pa.table({"k": pa.array([1, 2, 3, None, 5], type=pa.int64())})
    duck.register("t", table)
    return bt.from_arrow(table)


@pytest.mark.parametrize(
    ("predicate", "sql"),
    [
        (lambda: col("k") > 2, "k > 2"),
        (lambda: col("k") > 99, "k > 99"),  # matches nothing
        (lambda: col("k") < 0, "k < 0"),
        (lambda: col("k").is_null(), "k IS NULL"),
        (lambda: col("k").is_not_null(), "k IS NOT NULL"),
    ],
)
def test_count_star_over_a_filter_matches_duckdb(duck, t, predicate, sql):
    got = t.filter(predicate()).agg(n=count()).to_pydict()["n"][0]
    assert got == duck.sql(f"SELECT count(*) FROM t WHERE {sql}").fetchone()[0]


def test_count_star_over_an_empty_scan_is_zero(duck, t):
    """Every row filtered out: the count is 0, never null."""
    assert t.filter(col("k") > 99).count() == 0
    assert t.filter(col("k") > 99).agg(n=count()).to_pydict()["n"] == [0]


def test_count_star_over_an_empty_join_is_zero(duck):
    """The shape that exposed the bug: the filter's *input* is empty, not just its output."""
    left = pa.table({"k": pa.array([1], type=pa.int64())})
    right = pa.table({"k": pa.array([2], type=pa.int64())})
    duck.register("l", left)
    duck.register("r", right)

    empty = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k")
        .select(x=col("k") + 1)
        .filter(col("x") > 0)
    )
    expected = duck.sql(
        "SELECT count(*) FROM (SELECT l.k + 1 AS x FROM l JOIN r USING (k)) WHERE x > 0"
    ).fetchone()[0]
    assert expected == 0
    assert empty.agg(n=count()).to_pydict()["n"] == [0]
    assert empty.count() == 0


def test_count_if_still_matches_duckdb_including_null_on_empty(duck, t):
    """`count_if` is a *sum* of a mask by definition, and DuckDB returns NULL for it over
    zero rows. Fixing the `COUNT(*)` rewrite must not have changed that."""
    got = t.filter(col("k") > 99).agg(n=bt.count_if(col("k") > 0)).to_pydict()["n"][0]
    expected = duck.sql("SELECT count_if(k > 0) FROM t WHERE k > 99").fetchone()[0]
    assert got == expected is None


def test_grouped_count_over_a_filter_matches_duckdb(duck, t):
    """The rule must not fire with GROUP BY: a filter also drops all-fail groups."""
    got = t.filter(col("k") > 2).group_by("k").agg(n=count()).to_pydict()
    expected = duck.sql("SELECT k, count(*) AS n FROM t WHERE k > 2 GROUP BY k").to_arrow_table()
    assert sorted(got["n"]) == sorted(expected.column("n").to_pylist())
