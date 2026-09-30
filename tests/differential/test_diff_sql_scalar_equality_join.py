"""`col = (SELECT agg(...) FROM t)` over a relation the outer query reads, planned as a join.

TPC-H q15 keeps the supplier whose revenue equals `(SELECT max(total_revenue) FROM revenue)`,
where `revenue` is a CTE the outer query also joins. Evaluated eagerly as a literal, the CTE
ran twice. `scalar_sub.equality_join` plans it as an equi-join onto the aggregate's one row
instead, so the plan holds the CTE twice and subplan reuse computes it once. These cases
hold that join to DuckDB where a join and `=` could part ways: ties, NULLs, an empty input,
a HAVING that removes the one row, and float keys.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_TABLES = {
    "ties": pa.table({"k": [1, 1, 2, 2, 3, 3], "v": [5.0, 5.0, 2.0, 8.0, 3.0, 5.0]}),
    "nulls": pa.table(
        {"k": pa.array([1, 2, None, 3], pa.int64()), "v": pa.array([1.0, None, 9.0, 4.0])}
    ),
    "empty": pa.table({"k": pa.array([], pa.int64()), "v": pa.array([], pa.float64())}),
    "floats": pa.table({"k": [1, 2, 3, 4], "v": [-0.0, 0.0, 1.5, -2.5]}),
    "one_row": pa.table({"k": [7], "v": [1.25]}),
}

_Q15 = (
    "WITH r AS (SELECT k, sum(v) AS s FROM t GROUP BY k) "
    "SELECT k, s FROM r WHERE s = (SELECT max(s) FROM r) ORDER BY k"
)

_QUERIES = [
    _Q15,
    # The subquery on the left of `=`.
    "WITH r AS (SELECT k, sum(v) AS s FROM t GROUP BY k) "
    "SELECT k, s FROM r WHERE (SELECT min(s) FROM r) = s",
    # A HAVING that removes the aggregate's one row: nothing matches.
    "WITH r AS (SELECT k, sum(v) AS s FROM t GROUP BY k) "
    "SELECT k, s FROM r WHERE s = (SELECT max(s) FROM r HAVING count(*) > 100)",
    # An aggregate over no rows is NULL, which matches nothing.
    "SELECT k, v FROM t WHERE v = (SELECT max(v) FROM t WHERE v > 1000)",
    # Integer key against an integer aggregate, joined to a second relation.
    "SELECT a.k, b.v FROM t a JOIN t b ON a.k = b.k WHERE a.k = (SELECT max(k) FROM t)",
    # Types differ (integer column, double aggregate): the literal path keeps it.
    "SELECT k, v FROM t WHERE k = (SELECT avg(k) FROM t)",
]


@pytest.mark.parametrize("table", sorted(_TABLES))
@pytest.mark.parametrize("query", _QUERIES)
def test_equality_to_an_aggregate_matches_duckdb(duck, table, query):
    tbl = _TABLES[table]
    got = bt.sql(query, t=bt.from_arrow(tbl)).collect()
    duck.register("t", tbl)
    assert_same(got, duck.sql(query))
    if "ORDER BY" in query:
        expected = duck.sql(query).fetch_arrow_table()
        assert got.column("k").to_pylist() == expected.column("k").to_pylist()


def test_the_shared_cte_is_planned_as_one_join_not_a_literal():
    """Positive control: the rewrite applies to q15's shape, and only to a shared relation."""
    from batcher.plan.logical import Join
    from batcher.plan.visitor import walk

    t = bt.from_arrow(_TABLES["ties"])
    shared = bt.sql(_Q15, t=t)
    joins = [n for n in walk(shared._plan) if isinstance(n, Join)]
    assert len(joins) == 1, "q15's shape was expected to become one equi-join"
    # A subquery over a relation the outer query does not read stays a literal filter.
    other = bt.from_arrow(_TABLES["one_row"])
    alone = bt.sql("SELECT k, v FROM t WHERE v = (SELECT max(v) FROM u)", t=t, u=other)
    assert not [n for n in walk(alone._plan) if isinstance(n, Join)]


def test_a_subquery_correlated_through_an_unqualified_column_is_left_to_decorrelation(duck):
    """TPC-H q2's shape: `ps_supplycost = (SELECT min(ps_supplycost) ... WHERE p_partkey = ...)`.

    `p_partkey` is the outer relation's, unqualified, so the subquery cannot be planned on its
    own. The equality-join path must recognise it as correlated and leave it alone.
    """
    part = pa.table({"p_partkey": [1, 2, 3], "p_name": ["a", "b", "c"]})
    ps = pa.table(
        {
            "ps_partkey": [1, 1, 2, 2, 3],
            "ps_cost": [5.0, 3.0, 7.0, 7.0, 1.0],
            "ps_s": [1, 2, 3, 4, 5],
        }
    )
    query = (
        "SELECT p_partkey, ps_s FROM part, ps WHERE p_partkey = ps_partkey "
        "AND ps_cost = (SELECT min(ps_cost) FROM ps WHERE p_partkey = ps_partkey)"
    )
    got = bt.sql(query, part=part, ps=ps).collect()
    duck.register("part", part)
    duck.register("ps", ps)
    assert_same(got, duck.sql(query))
