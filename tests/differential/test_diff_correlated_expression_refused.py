"""A correlation the rewrites cannot decorrelate must be refused, never answered wrongly.

Every subquery rewrite in `_sql/parser/subquery/` works the same way: pull the correlations
it understands out of the inner `WHERE`, hand the remainder to the inner relation as *local*
predicates, and let `_reject_correlated` refuse anything still reaching outward. Only an
equality between two **plain columns** is understood, so a correlation through an expression
(`outer.c + 1`, a function call) falls into the remainder — and the refusal is the only thing
standing between it and a wrong answer.

That refusal had a hole, and it was exactly the shape most correlated subqueries take.
`_reject_correlated` treated an aliased table's *base name* as in-scope, so with the inner
query aliasing the same table the outer query reads:

    SELECT k FROM a WHERE EXISTS (SELECT 1 FROM a u WHERE u.v = a.v + 1)

`a.v` was classified local, the predicate became `u.v = u.v + 1`, and the query returned
**no rows** where DuckDB returns five. `NOT EXISTS` returned **every** row instead of two.
No error, nothing in the plan to look at, and invisible to anyone who aliases the outer
query too — `FROM a x ... WHERE x.v` was correctly refused the whole time. SQL scoping
shadows the base name inside `FROM a u`, which is the rule `_local_tables` twenty lines
above already applied.

So this file pins both halves, and the second half is the load-bearing one:

* the correlations that *do* decorrelate still agree with DuckDB, spelled both with and
  without an outer alias — a fix that made the guard stricter must not start refusing them;
* the correlations that do not are **refused**, in every spelling, including the two that
  used to answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


def _tables(duck):
    a = pa.table({"k": ["a", "b", "c", "d"], "v": pa.array([1, 2, 3, None], pa.int64())})
    b = pa.table({"k": ["a", "b", "c"], "w": pa.array([2, 3, 9], pa.int64())})
    duck.register("a", a)
    duck.register("b", b)
    return bt.from_arrow(a), bt.from_arrow(b)


# --- what must keep working -----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # Plain-column correlation, outer unaliased -- the spelling that used to be wrong
        # when the *expression* form was used, so it is worth pinning that it still works.
        "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.w = a.v)",
        "SELECT k FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.w = a.v)",
        # Same, with the outer aliased.
        "SELECT x.k FROM a x WHERE EXISTS (SELECT 1 FROM b y WHERE y.w = x.v)",
        # Self-correlation on a plain column, inner aliasing the outer's own table: this is
        # the shape whose base name the guard used to admit, and it must stay *answerable*
        # rather than becoming a casualty of tightening the rule.
        "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM a u WHERE u.v = a.v)",
        "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM a u WHERE u.v = a.v AND u.k <> a.k)",
        # Correlated scalar subquery, and an IN whose expression is on the *inner* side --
        # which is fine, because nothing reaches outward through it.
        "SELECT k, (SELECT max(w) FROM b WHERE b.k = a.k) m FROM a",
        "SELECT k FROM a WHERE v IN (SELECT w - 1 FROM b)",
    ],
)
def test_a_decorrelatable_correlation_still_matches_duckdb(duck, sql) -> None:
    a, b = _tables(duck)
    assert_same(bt.sql(sql, a=a, b=b).collect(), duck.sql(sql))


# --- what must be refused rather than answered ----------------------------------------


@pytest.mark.parametrize(
    ("sql", "why"),
    [
        (
            "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM a u WHERE u.v = a.v + 1)",
            "the outer side is an expression, and the inner aliases the outer's own table "
            "-- this returned zero rows instead of two",
        ),
        (
            "SELECT k FROM a WHERE NOT EXISTS (SELECT 1 FROM a u WHERE u.v = a.v + 1)",
            "the same shape negated -- this returned every row instead of two",
        ),
        (
            "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.w = a.v + 1)",
            "the outer side is an expression, distinct tables",
        ),
        (
            "SELECT k FROM a x WHERE EXISTS (SELECT 1 FROM a y WHERE y.v = x.v + 1)",
            "the outer side is an expression, both aliased",
        ),
        (
            "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.w = abs(a.v))",
            "the outer side is a function call",
        ),
        (
            "SELECT k, (SELECT max(w) FROM b WHERE b.w > a.v + 1) m FROM a",
            "a scalar subquery correlated through an expression",
        ),
    ],
)
def test_a_correlation_through_an_expression_is_refused(duck, sql, why) -> None:
    """Refusing is the contract. Answering it wrongly is what this test exists to prevent."""
    a, b = _tables(duck)
    duck.sql(sql)  # DuckDB can do it; that is precisely why silence would be dangerous
    with pytest.raises(NotImplementedError, match=r"correlated subquery"):
        bt.sql(sql, a=a, b=b).collect()


def test_the_refusal_names_the_column_and_what_is_in_scope(duck) -> None:
    """A generic "not supported" leaves the reader to bisect their own query."""
    a, b = _tables(duck)
    sql = "SELECT k FROM a WHERE EXISTS (SELECT 1 FROM a u WHERE u.v = a.v + 1)"
    with pytest.raises(NotImplementedError) as excinfo:
        bt.sql(sql, a=a, b=b).collect()
    message = str(excinfo.value)
    assert "a.v" in message, "the offending column is not named"
    assert "'u'" in message, "the tables actually in scope are not named"
