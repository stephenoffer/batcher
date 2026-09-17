"""What each aggregate returns over a relation with no rows, held against DuckDB.

An empty input is not an edge case in production, it is Tuesday: a filter matches nothing, a
partition is late, an upstream stage fails open. SQL's answers here are specific and not
uniform -- `sum` over no rows is NULL while `count` is 0, and `bool_and` over no rows is NULL
rather than the vacuous truth a reader might expect -- so "returns something sensible" is not
a specification and the oracle is the only way to check it.

All thirteen agree with DuckDB. The non-empty control beside each is what makes that mean
something: an engine whose aggregates returned NULL unconditionally would pass every
empty-relation assertion in this file.

`first` is the one that differs, and in Batcher's favour. `Expr.first` *requires* an
`order_by`, so `first(v)` does not compile at all, where DuckDB's `first(v)` returns whichever
row it happened to see. Refusing an aggregate whose answer the query does not determine is
the same discipline `.claude/rules/python-control-plane.md` applies to `row_number` ties and
to `LIMIT` over an unordered relation. Given the ordering it asks for, it agrees.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

_EMPTY_SQL = "CREATE TABLE e (v DOUBLE, a BIGINT)"
#: The `::DOUBLE` is load-bearing. DuckDB types a bare `8.0` in a `VALUES` list as
#: `DECIMAL(2,1)`, not `DOUBLE`, so without it the oracle computes decimal aggregates while
#: Batcher computes binary-float ones and the comparison is between two different types that
#: happen to agree on this data. The same typing difference is why a `CAST(2.5 AS BIGINT)`
#: probe against a `VALUES` fixture reads as half-away-from-zero rounding while the real
#: DOUBLE cast is half-to-even in both engines -- an hour was lost to that, on this file.
_FULL_SQL = (
    "CREATE TABLE t AS SELECT * FROM "
    "(VALUES (8.0::DOUBLE,1),(7.0::DOUBLE,2),(9.0::DOUBLE,2)) x(v,a)"
)


@pytest.fixture
def empty(duck):
    duck.execute(_EMPTY_SQL)
    return bt.from_arrow(pa.table({"v": pa.array([], pa.float64()), "a": pa.array([], pa.int64())}))


@pytest.fixture
def full(duck):
    duck.execute(_FULL_SQL)
    return bt.from_pydict({"v": [8.0, 7.0, 9.0], "a": [1, 2, 2]})


#: (label, batcher aggregate, SQL aggregate). Chosen so the answers are *not* uniform:
#: some are NULL over no rows and some are 0, which is the whole point.
_AGGREGATES = [
    ("sum", lambda c: c("v").sum(), "sum(v)"),
    ("count", lambda c: c("v").count(), "count(v)"),
    ("mean", lambda c: c("v").mean(), "avg(v)"),
    ("min", lambda c: c("v").min(), "min(v)"),
    ("max", lambda c: c("v").max(), "max(v)"),
    ("median", lambda c: c("v").median(), "median(v)"),
    ("std", lambda c: c("v").std(), "stddev(v)"),
    ("var", lambda c: c("v").var(), "var_samp(v)"),
    ("count_distinct", lambda c: c("a").count_distinct(), "count(DISTINCT a)"),
    ("quantile", lambda c: c("v").quantile(0.5), "quantile_cont(v, 0.5)"),
    ("bool_or", lambda c: (c("v") > 0).bool_or(), "bool_or(v>0)"),
    ("bool_and", lambda c: (c("v") > 0).bool_and(), "bool_and(v>0)"),
]

_IDS = [case[0] for case in _AGGREGATES]


@pytest.mark.parametrize(("label", "build", "sql"), _AGGREGATES, ids=_IDS)
def test_an_aggregate_over_no_rows_matches_duckdb(label, build, sql, empty, duck):
    got = empty.agg(r=build(bt.col)).to_arrow()
    assert_same_ordered(got, duck.sql(f"SELECT {sql} AS r FROM e"))


@pytest.mark.parametrize(("label", "build", "sql"), _AGGREGATES, ids=_IDS)
def test_the_same_aggregate_over_rows_matches_duckdb(label, build, sql, full, duck):
    """The control. Every assertion above would hold for an engine that returned NULL from
    every aggregate, so each one needs its non-empty twin to mean anything."""
    got = full.agg(r=build(bt.col)).to_arrow()
    assert_same_ordered(got, duck.sql(f"SELECT {sql} AS r FROM t"))


def test_the_answers_are_not_all_the_same(empty):
    """The second control, on the fixture rather than the engine. If every aggregate returned
    NULL over no rows, the file would be checking one behaviour thirteen times; SQL says
    `count` and `count(DISTINCT ...)` are 0 while the rest are NULL, and that split is the
    reason to test them individually."""
    answers = {
        label: empty.agg(r=build(bt.col)).to_pydict()["r"][0] for label, build, _ in _AGGREGATES
    }
    assert answers["count"] == 0
    assert answers["count_distinct"] == 0
    assert answers["sum"] is None
    assert answers["bool_and"] is None, "bool_and over no rows is NULL, not vacuous truth"


def test_the_oracle_column_is_a_double_not_a_decimal(full, duck):
    """Pins the fixture's typing, since the bug it prevents is invisible in the results.

    A `DECIMAL` oracle agrees with a `DOUBLE` engine on well-behaved data and stops agreeing
    exactly where these tests are pointed -- rounding, quantiles, and anything with a
    representation boundary. Asserting the type is the only way this stays true, because
    asserting the values does not notice."""
    assert duck.execute("SELECT typeof(v) FROM t LIMIT 1").fetchone()[0] == "DOUBLE"
    assert full.collect_schema()["v"] == "float64"


class TestFirstRequiresAnOrdering:
    """Batcher is stricter than DuckDB here, deliberately."""

    def test_first_without_an_ordering_does_not_compile(self):
        with pytest.raises(TypeError, match="order_by"):
            bt.col("v").first()

    @pytest.mark.parametrize("relation", ["empty", "full"])
    def test_ordered_first_matches_duckdb(self, relation, request, duck):
        dataset = request.getfixturevalue(relation)
        table = "e" if relation == "empty" else "t"
        got = dataset.agg(r=bt.col("v").first(order_by=bt.col("v"))).to_arrow()
        assert_same_ordered(got, duck.sql(f"SELECT first(v ORDER BY v) AS r FROM {table}"))
