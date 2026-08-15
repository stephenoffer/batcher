"""A small uncorrelated `IN (subquery)` under an `OR`, inlined as a literal set.

`_sql.parser.subquery.in_set.inline_small_set` collects a subquery of at most
`_INLINE_SET_MAX` distinct values and hands them to `Expr.is_in` instead of building a mark
join. `IN` is three-valued, so what these cases pin is every way a NULL can enter: in the
probe, in the set, in both, and in neither — plus `NOT IN`, where SQL's answer surprises
people and DuckDB is the arbiter.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def tables(duck):
    left = pa.table({"x": [1, 2, 3, None, 5], "g": ["a", "b", "c", "d", "e"]})
    small = pa.table({"k": [2, 5], "n": [10, 20]})
    with_null = pa.table({"k": [2, None], "n": [10, 20]})
    empty = pa.table({"k": pa.array([], pa.int64()), "n": pa.array([], pa.int64())})
    for name, table in (("l", left), ("r", small), ("rn", with_null), ("re", empty)):
        duck.register(name, table)
    return left, small, with_null, empty


def _session(tables):
    left, small, with_null, empty = tables
    sess = bt.Session()
    for name, table in (("l", left), ("r", small), ("rn", with_null), ("re", empty)):
        sess.register(name, table)
    return sess


@pytest.mark.parametrize(
    "sql",
    [
        # The shape the rewrite exists for: an `IN (subquery)` the optimizer cannot turn into
        # a semi-join because it is one side of an OR.
        "SELECT g FROM l WHERE x IN (SELECT k FROM r) OR g = 'c'",
        "SELECT g FROM l WHERE x NOT IN (SELECT k FROM r) OR g = 'c'",
        # A NULL in the set: an unmatched row is NULL, not FALSE, which changes `NOT IN`
        # from "the rest" to "nothing".
        "SELECT g FROM l WHERE x IN (SELECT k FROM rn) OR g = 'c'",
        "SELECT g FROM l WHERE x NOT IN (SELECT k FROM rn) OR g = 'c'",
        # An empty set: `IN` is FALSE for every row including the NULL probe, `NOT IN` TRUE.
        "SELECT g FROM l WHERE x IN (SELECT k FROM re) OR g = 'c'",
        "SELECT g FROM l WHERE x NOT IN (SELECT k FROM re) OR g = 'c'",
        # A subquery that selects nothing at run time, from a non-empty relation.
        "SELECT g FROM l WHERE x IN (SELECT k FROM r WHERE k > 100) OR g = 'c'",
        # The membership answer feeding something other than a WHERE-clause disjunct.
        "SELECT count(*) AS c FROM l WHERE x IN (SELECT k FROM r) OR x > 4",
        "SELECT g, (x IN (SELECT k FROM r)) AS member FROM l",
        "SELECT g, (x NOT IN (SELECT k FROM rn)) AS member FROM l",
        # Both sides of the OR are subqueries.
        "SELECT g FROM l WHERE x IN (SELECT k FROM r) OR x IN (SELECT k FROM rn)",
        # A computed left-hand side, which is what q45 has (`SUBSTRING(ca_zip, 1, 5)`).
        "SELECT g FROM l WHERE (x + 0) IN (SELECT k FROM r) OR g = 'c'",
    ],
)
def test_in_subquery_under_or_matches_duckdb(duck, tables, sql):
    assert_same(_session(tables).sql(sql).collect(), duck.sql(sql))


def test_a_set_over_the_inline_cap_still_matches_duckdb(duck):
    """Past `_INLINE_SET_MAX` the mark join takes over, and must answer identically."""
    from batcher._sql.parser.subquery.in_set import _INLINE_SET_MAX

    n = _INLINE_SET_MAX * 2
    left = pa.table({"x": list(range(n)), "g": [f"g{i}" for i in range(n)]})
    right = pa.table({"k": list(range(0, n, 2))})
    duck.register("l", left)
    duck.register("r", right)
    sess = bt.Session()
    sess.register("l", left)
    sess.register("r", right)
    sql = "SELECT count(*) AS c FROM l WHERE x IN (SELECT k FROM r) OR g = 'g3'"
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_a_string_keyed_set_matches_duckdb(duck):
    """q45's own types: the set is strings, and the probe may be absent from it."""
    left = pa.table({"item": ["a", "b", "c", None], "v": [1, 2, 3, 4]})
    right = pa.table({"item": ["b", "c"], "tag": ["x", "y"]})
    duck.register("l", left)
    duck.register("r", right)
    sess = bt.Session()
    sess.register("l", left)
    sess.register("r", right)
    sql = "SELECT v FROM l WHERE item IN (SELECT item FROM r) OR v = 1"
    assert_same(sess.sql(sql).collect(), duck.sql(sql))
