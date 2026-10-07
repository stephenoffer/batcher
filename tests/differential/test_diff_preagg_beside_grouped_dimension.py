"""Pre-aggregating the facts when the *dimension* side already holds an aggregate, vs DuckDB.

`pre_aggregate_beneath_dimension` refuses to group a side that already holds one of its own
partials. It used to ask that of the whole join, so a dimension semi-joined to a grouped
subquery -- TPC-DS q70's `store` filtered to the top states by a ranked aggregate -- refused
the push for the facts on the other side. The shape here is that one: facts joined to a store
dimension restricted by an `IN` over a grouped, ranked subquery, then grouped by two store
strings, plain and under ROLLUP. NULL measures and a store with no facts are included.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher.io.source import source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Aggregate
from batcher.plan.visitor import walk

_N = 6_000


@pytest.fixture
def sess(duck):
    facts = pa.table(
        {
            "f_store": [i % 9 for i in range(_N)],
            "f_profit": [None if i % 97 == 0 else float(i % 13) - 4.0 for i in range(_N)],
        }
    )
    store = pa.table(
        {
            "s_store": list(range(10)),  # store 9 has no facts
            "s_state": ["CA", "CA", "TX", "TX", "NY", "NY", "WA", "WA", "OR", "OR"],
            "s_county": [f"c{i % 4}" for i in range(10)],
        }
    )
    duck.register("facts", facts)
    duck.register("store", store)
    s = bt.Session()
    s.register("facts", facts)
    s.register("store", store)
    return s


_TOP_STATES = """
    s_state IN (SELECT s_state FROM (
        SELECT s_state, rank() OVER (PARTITION BY s_state ORDER BY sum(f_profit) DESC) AS r
        FROM facts, store WHERE f_store = s_store GROUP BY s_state) t
      WHERE r <= 5)
"""


def _partials(ds) -> list[Aggregate]:
    stats = [source_statistics(s) for s in ds._sources]
    plan = Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)
    return [
        n
        for n in walk(plan)
        if isinstance(n, Aggregate) and any(a.alias.startswith("__rp") for a in n.aggregates)
    ]


def test_grouped_by_dimension_strings(duck, sess):
    sql = f"""
        SELECT s_state, s_county, sum(f_profit) AS p, count(f_profit) AS n
        FROM facts, store WHERE f_store = s_store AND {_TOP_STATES}
        GROUP BY s_state, s_county
    """
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_rollup_ordered(duck, sess):
    sql = f"""
        SELECT s_state, s_county, sum(f_profit) AS p
        FROM facts, store WHERE f_store = s_store AND {_TOP_STATES}
        GROUP BY ROLLUP (s_state, s_county)
        ORDER BY s_state NULLS FIRST, s_county NULLS FIRST
    """
    assert_same_ordered(sess.sql(sql).collect(), duck.sql(sql))


def test_the_facts_are_pre_aggregated_beside_a_grouped_dimension(sess):
    sql = f"""
        SELECT s_state, s_county, sum(f_profit) AS p
        FROM facts, store WHERE f_store = s_store AND {_TOP_STATES}
        GROUP BY s_state, s_county
    """
    partials = _partials(sess.sql(sql))
    assert len(partials) >= 2, "expected the subquery's partial and the outer query's"
