"""Quantified comparison subqueries (`<op> ANY` / `<op> ALL`) and parenthesized predicates.

Two gaps in the subquery folder, both of which refused a query DuckDB answers:

* **`x = ANY (SELECT ...)` raised `unsupported SQL expression: Any`.** It is standard SQL
  and it is exactly `x IN (SELECT ...)`, so it now rewrites to one before anything else
  reads the tree, and inherits `IN`'s decorrelation and three-valued logic wholesale.
* **Parentheses changed the answer.** `NOT x IN (SELECT ...)` folded to an anti-join,
  while the identical `NOT (x IN (SELECT ...))` was refused — every shape test was an
  `isinstance` on the node, and the `Paren` wrapper matched none of them.

The NULL cases carry the weight here. `NOT IN` against a set containing a NULL is UNKNOWN
for *every* row, so the correct answer is no rows at all — the classic NOT-IN trap, and the
one an anti-join gets wrong if it is applied naively. Each case is therefore run against
three right-hand sides: ordinary, NULL-bearing, and empty.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_LEFT = pa.table(
    {
        "i": pa.array([1, 2, 3, None], pa.int64()),
        "g": pa.array(["a", "b", "c", "d"], pa.string()),
    }
)

#: The three shapes of right-hand side that decide a quantified predicate's answer.
_RIGHTS = {
    "ordinary": pa.table({"v": pa.array([2, 3], pa.int64())}),
    "with_null": pa.table({"v": pa.array([2, None], pa.int64())}),
    "empty": pa.table({"v": pa.array([], pa.int64())}),
}

_EQUIVALENT = [
    "SELECT g FROM t WHERE i = ANY (SELECT v FROM u)",
    "SELECT g FROM t WHERE i = SOME (SELECT v FROM u)",
    "SELECT g FROM t WHERE i IN (SELECT v FROM u)",
    "SELECT g FROM t WHERE NOT (i IN (SELECT v FROM u))",
    "SELECT g FROM t WHERE NOT (i = ANY (SELECT v FROM u))",
    "SELECT g FROM t WHERE i <> ALL (SELECT v FROM u)",
    "SELECT g FROM t WHERE i NOT IN (SELECT v FROM u)",
    "SELECT g FROM t WHERE ((i IN (SELECT v FROM u)))",
    "SELECT g FROM t WHERE NOT (EXISTS (SELECT 1 FROM u WHERE u.v = t.i))",
    "SELECT g FROM t WHERE i = ANY (SELECT v FROM u WHERE v > 1)",
]


@pytest.mark.parametrize("right", list(_RIGHTS), ids=list(_RIGHTS))
@pytest.mark.parametrize("query", _EQUIVALENT)
def test_quantified_and_parenthesized_subqueries_match_duckdb(duck, query, right):
    u = _RIGHTS[right]
    duck.register("t", _LEFT)
    duck.register("u", u)
    assert_same(bt.sql(query, t=_LEFT, u=u).collect(), duck.sql(query))


def test_any_over_a_multi_column_row_value(duck):
    """`(a, b) = ANY (SELECT x, y ...)` is the multi-key form of `IN`, not a scalar one."""
    left = pa.table({"a": pa.array([1, 1, 2], pa.int64()), "b": pa.array([1, 2, 2], pa.int64())})
    right = pa.table({"x": pa.array([1, 2], pa.int64()), "y": pa.array([2, 2], pa.int64())})
    query = "SELECT a, b FROM t WHERE (a, b) = ANY (SELECT x, y FROM u)"
    duck.register("t", left)
    duck.register("u", right)
    assert_same(bt.sql(query, t=left, u=right).collect(), duck.sql(query))


#: Every quantified operator the `IN` rewrite does not cover. Each is lowered to a CASE over
#: the subquery's row count, non-null count and min/max, which is the full three-valued truth
#: table rather than the `x > (SELECT max(c) ...)` shortcut.
_INEQUALITY_FORMS = [
    f"i {op} {q} (SELECT v FROM u)" for op in (">", ">=", "<", "<=") for q in ("ANY", "ALL")
] + ["i = ALL (SELECT v FROM u)", "i <> ANY (SELECT v FROM u)"]

#: Right-hand sides that decide a non-equality quantified predicate. `all_null` is the case
#: where no non-null element exists to refute or witness, and `ties` repeats the extreme.
_ORDERED_RIGHTS = {
    **_RIGHTS,
    "all_null": pa.table({"v": pa.array([None], pa.int64())}),
    "ties": pa.table({"v": pa.array([3, 3], pa.int64())}),
}


def _rows(table: pa.Table) -> list[tuple]:
    return [tuple(r.values()) for r in table.to_pylist()]


@pytest.mark.parametrize("right", list(_ORDERED_RIGHTS), ids=list(_ORDERED_RIGHTS))
@pytest.mark.parametrize("form", _INEQUALITY_FORMS)
def test_an_inequality_quantified_subquery_keeps_its_three_valued_answer(duck, form, right):
    """The projected TRUE / FALSE / NULL value matches DuckDB, and so does its negation.

    Read as a value, not only as a filter: an empty set makes ``x > ALL (S)`` TRUE and a NULL
    in S makes it UNKNOWN, and a rewrite that is right under WHERE but wrong under NOT is
    exactly what comparing the projected column (and `NOT` of it) catches. Ordered by the
    unique `g`, so the comparison is positional and a boolean read back as an integer fails.
    """
    u = _ORDERED_RIGHTS[right]
    query = f"SELECT g, {form} AS b, NOT ({form}) AS nb FROM t ORDER BY g"
    duck.register("t", _LEFT)
    duck.register("u", u)
    got = bt.sql(query, t=_LEFT, u=u).collect()
    assert got.schema.field("b").type == pa.bool_()
    assert _rows(got) == duck.sql(query).fetchall()


@pytest.mark.parametrize("right", list(_ORDERED_RIGHTS), ids=list(_ORDERED_RIGHTS))
@pytest.mark.parametrize("form", _INEQUALITY_FORMS)
def test_an_inequality_quantified_subquery_filters_like_duckdb(duck, form, right):
    """Under WHERE and WHERE NOT, over a UNION subquery (the derived-table lowering)."""
    u = _ORDERED_RIGHTS[right]
    union_form = form.replace("SELECT v FROM u", "SELECT v FROM u UNION ALL SELECT v FROM u")
    duck.register("t", _LEFT)
    duck.register("u", u)
    for query in (
        f"SELECT g FROM t WHERE {form}",
        f"SELECT g FROM t WHERE NOT ({union_form})",
    ):
        assert_same(bt.sql(query, t=_LEFT, u=u).collect(), duck.sql(query))


@pytest.mark.parametrize("form", _INEQUALITY_FORMS)
def test_a_correlated_inequality_quantified_subquery_matches_duckdb(duck, form):
    """A correlated S decorrelates through its scalar aggregates, empty groups included."""
    right = pa.table(
        {
            "k": pa.array(["a", "a", "b", "c"], pa.string()),
            "v": pa.array([1, None, 2, 9], pa.int64()),
        }
    )
    correlated = form.replace("FROM u)", "FROM u WHERE u.k = t.g)")
    query = f"SELECT g, {correlated} AS b FROM t ORDER BY g"
    duck.register("t", _LEFT)
    duck.register("u", right)
    assert _rows(bt.sql(query, t=_LEFT, u=right).collect()) == duck.sql(query).fetchall()


def test_a_row_valued_inequality_quantified_subquery_is_refused():
    """`(a, b) > ALL (...)` has no scalar extreme to compare against, so it raises."""
    left = pa.table({"a": pa.array([1], pa.int64()), "b": pa.array([1], pa.int64())})
    right = pa.table({"x": pa.array([1], pa.int64()), "y": pa.array([2], pa.int64())})
    with pytest.raises(NotImplementedError, match="row-valued"):
        bt.sql("SELECT a FROM t WHERE (a, b) > ALL (SELECT x, y FROM u)", t=left, u=right)


@pytest.mark.parametrize("right", sorted(_RIGHTS))
def test_a_quantified_subquery_under_or_matches_duckdb(duck, right):
    """`= ANY` normalizes to `IN`, so it inherits `IN`'s semantics under `OR` as well.

    This was a refusal, on the grounds that an `IN` subquery under `OR` cannot become a
    semi-join (the join drops the rows the `OR` keeps) and the `EXISTS` rewrite that looks
    equivalent is not. The engine now answers it, so what is worth pinning is the answer.

    Parameterized over all three right-hand sides because they are what decide a quantified
    predicate: an ordinary list, a list holding a NULL (which makes the predicate NULL rather
    than false for a non-member, so only the `OR`'s own rows survive), and an empty one
    (where `= ANY` is false for every row and the `OR` is the entire result).
    """
    u = _RIGHTS[right]
    query = "SELECT g FROM t WHERE i = ANY (SELECT v FROM u) OR g = 'd'"
    duck.register("t", _LEFT)
    duck.register("u", u)
    assert_same(bt.sql(query, t=_LEFT, u=u).collect(), duck.sql(query))


@pytest.mark.parametrize(
    "query",
    [
        "SELECT ALL i FROM t",
        "SELECT i FROM t UNION ALL SELECT v FROM u",
        "SELECT COUNT(ALL i) AS n FROM t",
    ],
)
def test_the_other_uses_of_the_all_keyword_are_untouched(duck, query):
    """`ALL` is also a quantifier on SELECT, UNION and an aggregate — none is a subquery."""
    u = _RIGHTS["ordinary"]
    duck.register("t", _LEFT)
    duck.register("u", u)
    assert_same(bt.sql(query, t=_LEFT, u=u).collect(), duck.sql(query))
