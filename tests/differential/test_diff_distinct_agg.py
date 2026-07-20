"""`SUM(DISTINCT x)` / `AVG(DISTINCT x)` and friends vs DuckDB.

A distinct aggregate is the plain aggregate over rows deduped on the group keys plus the
aggregated expression, so the dedup happens once up front and the aggregate itself is an
ordinary one. Previously every query here raised ``NotImplementedError``.

NULL handling is the case worth pinning: `DISTINCT` treats all NULLs as one value, and
the aggregates then skip it — so a group that is entirely NULL aggregates to NULL rather
than disappearing.

A plain aggregate alongside a DISTINCT one uses a two-level aggregate instead: level 1
groups by `(keys, x)` — deduping `x` implicitly — while pre-aggregating the plain one into
a mergeable partial, and level 2 combines those partials. It is deliberately not two
aggregates joined on the group keys, because a join drops the NULL-keyed group.

What still has no correct single-pass form (two different DISTINCT expressions, or a plain
aggregate with no single-column mergeable partial such as `avg`) must reject cleanly rather
than return a plausible wrong number.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    # `v` has within-group duplicates (a: 1,1), a NULL, and an all-NULL group (c).
    table = pa.table(
        {
            "k": ["a", "a", "b", "b", "b", "c"],
            "v": [1, 1, 2, 3, None, None],
            "w": [10, 20, 30, 40, 50, 60],
        }
    )
    duck.register("t", table)
    return table


@pytest.mark.differential
@pytest.mark.parametrize("fn", ["sum", "avg", "count", "min", "max"])
def test_distinct_aggregate_grouped(duck, t, fn):
    """Each distinct aggregate must match DuckDB per group, NULLs included."""
    query = f"SELECT k, {fn}(DISTINCT v) AS a FROM t GROUP BY k"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize("fn", ["sum", "avg", "count"])
def test_distinct_aggregate_global(duck, t, fn):
    """No GROUP BY — the dedup is over the whole relation."""
    query = f"SELECT {fn}(DISTINCT v) AS a FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_distinct_aggregate_over_an_expression(duck, t):
    """The DISTINCT argument may be an expression, not just a column."""
    query = "SELECT k, sum(DISTINCT v * 2) AS a FROM t GROUP BY k"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_several_distinct_aggregates_over_one_expression(duck, t):
    """Several distinct aggregates sharing one expression need only a single dedup."""
    query = "SELECT k, sum(DISTINCT v) AS s, avg(DISTINCT v) AS m FROM t GROUP BY k"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_distinct_aggregate_with_having(duck, t):
    """HAVING filters the deduped aggregate, not the raw rows."""
    query = "SELECT k, sum(DISTINCT v) AS s FROM t GROUP BY k HAVING sum(DISTINCT v) > 2"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_distinct_aggregate_over_a_computed_group_key(duck, t):
    """A computed group key must survive the dedup and still group correctly."""
    query = "SELECT upper(k) AS uk, sum(DISTINCT v) AS s FROM t GROUP BY upper(k)"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize(
    "plain",
    ["count(w) AS c", "count(*) AS c", "sum(w) AS c", "min(w) AS c", "max(w) AS c"],
)
def test_distinct_mixed_with_a_plain_aggregate(duck, t, plain):
    """A plain aggregate alongside a DISTINCT one, via the two-level rewrite.

    A single dedup would make the plain aggregate undercount, so instead level 1 groups by
    `(keys, v)` — which dedups `v` implicitly — while pre-aggregating the plain aggregate,
    and level 2 combines those partials. `count` is the case worth watching: a group's
    total is the SUM of its sub-counts, not their count.
    """
    query = f"SELECT k, sum(DISTINCT v) AS s, {plain} FROM t GROUP BY k"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_distinct_mixed_keeps_the_null_group(duck):
    """The NULL group must survive — the reason this is not implemented as a join.

    Joining two per-phase aggregates on the group keys would drop the NULL-keyed group,
    because a join does not match NULL to NULL while GROUP BY treats it as a group. The
    two-level aggregate never joins, so the group is preserved.
    """
    table = pa.table({"k": ["a", "a", None, None], "v": [1, 1, 2, 2], "w": [5, 6, 7, 8]})
    duck.register("t2", table)
    query = "SELECT k, sum(DISTINCT v) AS s, count(w) AS c FROM t2 GROUP BY k"
    out = bt.sql(query, t2=table).collect()
    assert out.num_rows == 2, "the NULL group was dropped"
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_distinct_mixed_global_no_group_by(duck, t):
    """The mixed rewrite with no GROUP BY — one row out."""
    query = "SELECT sum(DISTINCT v) AS s, count(w) AS c FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_distinct_mixed_with_a_non_decomposable_aggregate_rejects(t):
    """`avg` has no single-column mergeable partial, so it cannot be pre-aggregated."""
    query = "SELECT k, sum(DISTINCT v) AS s, avg(w) AS a FROM t GROUP BY k"
    with pytest.raises(NotImplementedError, match="mergeable partial"):
        bt.sql(query, t=t).collect()


def test_two_different_distinct_expressions_reject(t):
    """Two DISTINCT expressions need two different dedups, so one pass cannot serve both."""
    query = "SELECT k, sum(DISTINCT v) AS s, sum(DISTINCT w) AS c FROM t GROUP BY k"
    with pytest.raises(NotImplementedError, match="DISTINCT expressions"):
        bt.sql(query, t=t).collect()
