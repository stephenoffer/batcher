"""Multi-level GROUP BY computed from one shared finest aggregate matches DuckDB.

`api.multi_group` builds a `ROLLUP`/`CUBE`/`GROUPING SETS` as one aggregate at the finest
grouping, holding each aggregate's partial state, with every level merging those partials.
The edges that merging gets wrong are the ones covered here: a count over an empty input
(a sum of no partial counts is NULL, the count is 0), NULL measures (a `count(x)` must not
count them, a `mean` must not divide by them), a `mean` over integers as well as floats, a
decimal `mean` (declined, so each level aggregates on its own) and the placeholder `max(k)`
the grand total groups a key by.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FACT = pa.table(
    {
        "k": pa.array([1, 1, 2, 2, 3, 3, 4, None], pa.int64()),
        "v": pa.array([10, None, 30, 40, None, 60, 70, 80], pa.int64()),
        "f": pa.array([1.5, 2.5, None, 4.0, 5.0, 6.5, 7.0, 8.0], pa.float64()),
        "m": pa.array(
            [Decimal("1.10"), Decimal("2.20"), None, Decimal("4.40")] * 2, pa.decimal128(9, 2)
        ),
    }
)
_DIM = pa.table(
    {
        "k": pa.array([1, 2, 3, 4], pa.int64()),
        "region": pa.array(["e", "e", "w", None]),
        "city": pa.array(["x", "y", "z", "z"]),
    }
)
_EMPTY_WHERE = "f.v > 1000"

_JOIN = "FROM fact f JOIN dim d ON f.k = d.k"
_AGGS = (
    "count(*) AS n, count(f.v) AS nv, sum(f.v) AS s, min(f.f) AS lo, max(f.v) AS hi, "
    "avg(f.v) AS av, avg(f.f) AS af"
)

_QUERIES = [
    f"SELECT d.region, d.city, {_AGGS} {_JOIN} GROUP BY ROLLUP(d.region, d.city)",
    f"SELECT d.region, d.city, {_AGGS} {_JOIN} GROUP BY CUBE(d.region, d.city)",
    f"SELECT d.region, d.city, {_AGGS} {_JOIN} "
    "GROUP BY GROUPING SETS ((d.region, d.city), (d.city), ())",
    # Every level empty but the grand total, which still has one row: counts must say 0.
    f"SELECT d.region, d.city, {_AGGS} {_JOIN} WHERE {_EMPTY_WHERE} "
    "GROUP BY ROLLUP(d.region, d.city)",
    # A decimal mean has no partial form here: the levels aggregate on their own.
    f"SELECT d.region, avg(f.m) AS am, sum(f.m) AS sm {_JOIN} GROUP BY ROLLUP(d.region)",
]


def _session() -> bt.Session:
    s = bt.Session()
    s.register("fact", _FACT)
    s.register("dim", _DIM)
    return s


@pytest.mark.parametrize("query", _QUERIES)
def test_shared_levels_match_duckdb(duck, query):
    duck.register("fact", _FACT)
    duck.register("dim", _DIM)
    got = _session().sql(query).collect()
    assert_same_for_query(got, duck.sql(query), query)


def test_the_levels_really_share_one_aggregate():
    """The positive control: without it every case above could pass on the old lowering."""
    from batcher.plan.logical import Aggregate
    from batcher.plan.visitor import walk

    plan = _session().sql(_QUERIES[0])._plan
    shared = [
        n
        for n in walk(plan)
        if isinstance(n, Aggregate)
        and any(re.fullmatch(r"__lvl_\d+", s.alias) for s in n.aggregates)
    ]
    assert len(shared) == 3, "ROLLUP(a, b) has three levels, each reading the shared aggregate"
    assert len({id(n) for n in shared}) == 1, "the levels must read one and the same aggregate"
