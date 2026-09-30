"""`pre_aggregate_facts`: a star aggregate's facts, pre-aggregated beneath its dimensions.

`Agg((A JOIN B) JOIN C)` becomes `Agg'(A JOIN Agg_partial(B JOIN C))`, or `Agg(F JOIN D)`
becomes `Agg'(Agg_partial(F) JOIN D)`. The aligned planner applies it (the optimizer does not),
so these tests apply it directly and run the result. It claims correctness for any fan-out on
inner joins, with no uniqueness proof on
`A`'s key: every row of `A` matches the same rows of `B JOIN C` before and after, so merging
per-key partials reproduces each group. These cases hold it to DuckDB where that claim is
tested hardest -- a dimension key that repeats, NULL keys on every side, groups whose string
keys collide across different dimension keys, and each decomposable aggregate.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# `dim` repeats key 2 (with different names) and holds a NULL key; two keys share a name.
_DIM = pa.table(
    {
        "d_k": pa.array([1, 2, 2, 3, None, 4], pa.int64()),
        "d_name": ["ann", "bob", "bea", "ann", "nul", "dee"],
        "d_region": pa.array([10, 20, 20, 10, 30, 40], pa.int64()),
    }
)
_ORD = pa.table(
    {
        "o_id": pa.array([100, 101, 102, 103, 104, 105, 106], pa.int64()),
        "o_k": pa.array([1, 2, 2, 3, None, 9, 4], pa.int64()),
        "o_pri": ["H", "L", "H", "L", "H", "L", "H"],
    }
)
_ITEM = pa.table(
    {
        "i_id": pa.array([100, 100, 101, 102, 102, 102, 103, 104, None, 106], pa.int64()),
        "i_amt": pa.array([1.5, 2.0, 3.25, 4.0, None, 5.5, 6.0, 7.0, 8.0, 9.75]),
        "i_qty": pa.array([1, 2, 3, 4, 5, None, 7, 8, 9, 10], pa.int64()),
    }
)

_QUERIES = [
    # q10's shape: group by dimension strings, sum a fact measure.
    "SELECT d_k, d_name, sum(i_amt) AS s FROM dim, ord, item "
    "WHERE d_k = o_k AND o_id = i_id GROUP BY d_k, d_name",
    # Group by a string that two dimension keys share: partials of both merge into one group.
    "SELECT d_name, sum(i_amt) AS s, count(i_qty) AS c, count(*) AS n FROM dim, ord, item "
    "WHERE d_k = o_k AND o_id = i_id GROUP BY d_name",
    # min / max, and a group key the fact side supplies alongside the dimension's.
    "SELECT d_name, o_pri, min(i_amt) AS lo, max(i_qty) AS hi FROM dim, ord, item "
    "WHERE d_k = o_k AND o_id = i_id GROUP BY d_name, o_pri",
    # A measure expression over both fact tables.
    "SELECT d_name, sum(i_amt * i_qty) AS s FROM dim JOIN ord ON d_k = o_k "
    "JOIN item ON o_id = i_id GROUP BY d_name",
    # The same query, joined in the other order.
    "SELECT d_name, sum(i_amt) AS s FROM item JOIN ord ON o_id = i_id "
    "JOIN dim ON d_k = o_k GROUP BY d_name",
]


def _rewritten(query: str):
    from batcher import kyber
    from batcher.kyber.rules.agg_pushdown import pre_aggregate_facts

    ds = bt.sql(query, dim=_DIM, ord=_ORD, item=_ITEM)
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources)
    return opt, pre_aggregate_facts(opt), ds._sources


@pytest.mark.parametrize("query", _QUERIES)
def test_pre_aggregated_star_matches_duckdb(duck, query):
    from batcher.dist.executors.ray_runtime import _single_node

    opt, rewritten, sources = _rewritten(query)
    # Positive control: every query here has the shape, so the rewrite must change the plan.
    assert rewritten is not opt
    got = _single_node(rewritten, sources)
    for name, table in (("dim", _DIM), ("ord", _ORD), ("item", _ITEM)):
        duck.register(name, table)
    assert_same(got, duck.sql(query))


def test_the_rewrite_pre_aggregates_the_facts_exactly_once():
    """Applied twice, the rewrite adds nothing: one partial aggregate below the join."""
    from batcher.kyber.rules.agg_pushdown import pre_aggregate_facts
    from batcher.plan.logical import Aggregate
    from batcher.plan.visitor import walk

    _, rewritten, _ = _rewritten(_QUERIES[0])
    twice = pre_aggregate_facts(rewritten)
    partial = [
        n
        for n in walk(twice)
        if isinstance(n, Aggregate) and any(a.alias.startswith("__rp") for a in n.aggregates)
    ]
    assert len(partial) == 1
