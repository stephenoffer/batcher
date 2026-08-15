"""`<expr> IN (subquery)` matches DuckDB, and `NOT IN` keeps its three-valued answer.

`WHERE v + 1 IN (SELECT k FROM u)` raised before ``subquery.in_expr``: `_apply_in_subquery`
reads the left side as a *column name* to hand a semi join, and an expression has no name.
Everything past that point was already general, so the fix names the value rather than adding a
second `IN`.

That distinction is what these cases are really pinning. `x NOT IN (S)` is **not** an anti join
when `S` can yield NULL, and the correct three-valued answer lives in `_not_in_antijoin`; a
separate implementation for expressions could have restated that rule wrongly and would have
looked right on every input without a NULL. So the NULL-bearing subquery is here for both the
expression and the plain-column spelling, and the plain-column cases are regression guards that
the shared path is still the one doing the work.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_T = pa.table({"g": ["a", "b", "a", "b", "c", None], "v": [1, 2, 3, None, 5, 6]})
_U = pa.table({"g": ["a", "c", "d"], "k": [1, 2, 3]})
# A NULL in the probed column: the reason `NOT IN` cannot be a plain anti join.
_N = pa.table({"k": [1, None, 2]})


@pytest.fixture
def sess(duck):
    duck.register("t", _T)
    duck.register("u", _U)
    duck.register("n", _N)
    s = bt.Session()
    s.register("t", bt.from_arrow(_T))
    s.register("u", bt.from_arrow(_U))
    s.register("n", bt.from_arrow(_N))
    return s


def _check(sess, duck, sql: str) -> None:
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


@pytest.mark.parametrize(
    "expr",
    ["v + 0", "v * 2", "v - 1", "abs(v)", "CAST(v AS BIGINT)"],
    ids=["add", "mul", "sub", "func", "cast"],
)
def test_expression_in_subquery(sess, duck, expr):
    _check(sess, duck, f"SELECT g, v FROM t WHERE {expr} IN (SELECT k FROM u)")


def test_expression_not_in_subquery(sess, duck):
    _check(sess, duck, "SELECT g, v FROM t WHERE v + 0 NOT IN (SELECT k FROM u)")


def test_expression_not_in_a_subquery_yielding_null(sess, duck):
    """The three-valued case: every row is UNKNOWN, so `NOT IN` keeps none of them."""
    _check(sess, duck, "SELECT g, v FROM t WHERE v + 0 NOT IN (SELECT k FROM n)")


def test_expression_in_a_subquery_yielding_null(sess, duck):
    _check(sess, duck, "SELECT g, v FROM t WHERE v + 0 IN (SELECT k FROM n)")


def test_a_string_expression(sess, duck):
    _check(sess, duck, "SELECT g FROM t WHERE upper(g) IN (SELECT upper(g) FROM u)")


def test_the_synthetic_column_does_not_reach_the_output(sess, duck):
    """The rewrite adds a column to name the expression; the caller must never see it."""
    out = sess.sql("SELECT * FROM t WHERE v + 0 IN (SELECT k FROM u)").collect()
    assert out.column_names == ["g", "v"]
    assert_same(out, duck.sql("SELECT * FROM t WHERE v + 0 IN (SELECT k FROM u)"))


def test_a_column_named_like_the_synthetic_one_is_not_clobbered(sess, duck):
    """The synthetic name is chosen against the outer columns, so a collision cannot happen."""
    t2 = pa.table({"__bx_in": [1, 2, 3], "v": [1, 2, 3]})
    duck.register("t2", t2)
    sess.register("t2", bt.from_arrow(t2))
    sql = "SELECT * FROM t2 WHERE v + 0 IN (SELECT k FROM u)"
    out = sess.sql(sql).collect()
    assert out.column_names == ["__bx_in", "v"]
    assert_same(out, duck.sql(sql))


def test_plain_column_in_still_takes_the_semi_join(sess, duck):
    _check(sess, duck, "SELECT g, v FROM t WHERE v IN (SELECT k FROM u)")


def test_plain_column_not_in_keeps_three_valued_logic(sess, duck):
    _check(sess, duck, "SELECT g, v FROM t WHERE v NOT IN (SELECT k FROM n)")


def test_correlated_in_still_decorrelates(sess, duck):
    _check(sess, duck, "SELECT g, v FROM t WHERE v IN (SELECT k FROM u x WHERE x.g = t.g)")


def test_expression_in_a_correlated_subquery(sess, duck):
    _check(sess, duck, "SELECT g, v FROM t WHERE v + 0 IN (SELECT k FROM u x WHERE x.g = t.g)")


def test_expression_in_subquery_matches_distributed(sess, duck):
    sql = "SELECT g, v FROM t WHERE v + 0 IN (SELECT k FROM u)"
    expected = duck.sql(sql)
    assert_same(sess.sql(sql).collect(), expected)
    assert_same(sess.sql(sql).collect(distributed=True), expected)
