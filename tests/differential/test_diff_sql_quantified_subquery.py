"""Quantified comparison subqueries (`= ANY` / `<> ALL`) and parenthesized predicates.

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


@pytest.mark.parametrize(
    "query",
    [
        "SELECT g FROM t WHERE i > ANY (SELECT v FROM u)",
        "SELECT g FROM t WHERE i <= ANY (SELECT v FROM u)",
        "SELECT g FROM t WHERE i >= ALL (SELECT v FROM u)",
        "SELECT g FROM t WHERE i = ALL (SELECT v FROM u)",
        "SELECT g FROM t WHERE i <> ANY (SELECT v FROM u)",
    ],
)
def test_a_quantified_form_with_no_exact_rewrite_is_refused_not_guessed(query):
    """An inequality `ALL` over a NULL-bearing set is UNKNOWN, which min/max cannot express.

    Refusing is the contract: a wrong row in the result costs more than an error, and the
    message carries the rewrite that does work.
    """
    with pytest.raises(NotImplementedError, match=r"(ANY|ALL)"):
        bt.sql(query, t=_LEFT, u=_RIGHTS["ordinary"]).to_arrow()


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
