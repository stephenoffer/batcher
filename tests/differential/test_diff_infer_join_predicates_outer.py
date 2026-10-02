"""Key constraints mirrored through outer/semi/anti joins and group keys match DuckDB.

`infer_join_predicates` now filters a left join's null-extended side, and a semi/anti join's
right side, by a constraint proven on the left key -- including one proven under an
aggregate's group key (TPC-DS q78). What could go wrong is losing a preserved row or a
null-extended row's NULLs, so each shape keeps unmatched left rows, NULL keys on both sides,
duplicate keys, and right rows outside the constraint that must vanish without changing an
answer. 120,000 fact rows, so the parallel executor shards them.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_for_query

pytestmark = pytest.mark.differential

_N = 120_000


def _fact(rng, prefix: str) -> pa.Table:
    item = rng.integers(0, 60, _N).astype("float64")
    item[rng.random(_N) < 0.02] = np.nan
    return pa.table(
        {
            f"{prefix}_date": pa.array(rng.integers(0, 1_460, _N), pa.int64()),
            f"{prefix}_item": pa.array(item, from_pandas=True).cast(pa.int64()),
            f"{prefix}_cust": pa.array(rng.integers(0, 50, _N), pa.int64()),
            f"{prefix}_q": pa.array(rng.integers(0, 10, _N), pa.int64()),
        }
    )


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(78)
    dates = pa.table(
        {
            "d_sk": pa.array(range(1_460), pa.int64()),
            "d_year": pa.array([2000 + i // 365 for i in range(1_460)], pa.int64()),
        }
    )
    return {"ss": _fact(rng, "ss"), "ws": _fact(rng, "ws"), "cs": _fact(rng, "cs"), "dates": dates}


def _run(duck, tables, sql):
    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
        duck.register(name, t)
    return s.sql(sql).collect(), duck.sql(sql)


_PER_YEAR = """
     {p} AS (SELECT d_year y, {f}_item i, {f}_cust c, sum({f}_q) q, count(*) n
             FROM {f}, dates WHERE {f}_date = d_sk GROUP BY d_year, {f}_item, {f}_cust)"""

_CTES = (
    "WITH"
    + _PER_YEAR.format(p="a", f="ss")
    + ","
    + _PER_YEAR.format(p="b", f="ws")
    + ","
    + _PER_YEAR.format(p="e", f="cs")
)

_QUERIES = [
    # TPC-DS q78: two left joins under an OR that only a matched side can satisfy.
    _CTES
    + """
    SELECT a.y, a.i, a.c, a.q, b.q bq, e.q eq FROM a
    LEFT JOIN b ON b.y = a.y AND b.i = a.i AND b.c = a.c
    LEFT JOIN e ON e.y = a.y AND e.i = a.i AND e.c = a.c
    WHERE (coalesce(b.q, 0) > 0 OR coalesce(e.q, 0) > 0) AND a.y = 2001
    ORDER BY a.i NULLS FIRST, a.c, a.q LIMIT 200""",
    # Every unmatched left row must survive with NULLs on the right.
    _CTES
    + """
    SELECT a.y, a.i, a.c, a.q, b.q bq, b.n bn FROM a
    LEFT JOIN b ON b.y = a.y AND b.i = a.i AND b.c = a.c
    WHERE a.y = 2002 AND a.c < 3""",
    # A range constraint, and a semi and an anti join on the same keys.
    _CTES
    + """
    SELECT a.y, a.i, a.c, a.q FROM a
    WHERE a.y >= 2002 AND EXISTS (SELECT 1 FROM b WHERE b.y = a.y AND b.i = a.i AND b.c = a.c)""",
    _CTES
    + """
    SELECT a.y, a.i, a.c, a.q FROM a
    WHERE a.y >= 2002
      AND NOT EXISTS (SELECT 1 FROM b WHERE b.y = a.y AND b.i = a.i AND b.c = a.c AND b.q > 30)""",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_matches_duckdb(duck, tables, query):
    got, want = _run(duck, tables, query)
    assert got.num_rows > 0
    assert_same_for_query(got, want, query)


def test_the_null_extended_side_really_is_filtered(tables):
    """The control: the right side's aggregate input carries the left side's year."""
    import json

    from batcher import kyber
    from batcher.core import default_hub
    from batcher.plan.logical import Aggregate, Filter
    from batcher.plan.visitor import walk

    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
    ds = s.sql(_QUERIES[1])
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources, hub=default_hub())
    aggregates = [n for n in walk(opt) if isinstance(n, Aggregate)]
    assert len(aggregates) == 2
    for agg in aggregates:
        filters = [json.dumps(n.predicate.to_ir()) for n in walk(agg) if isinstance(n, Filter)]
        assert any("2002" in f for f in filters), "a per-year aggregate was not scoped to 2002"


def test_a_full_join_keeps_every_row(duck, tables):
    """No mirroring through a full join: both sides are preserved."""
    sql = (
        _CTES
        + """
    SELECT a.y, a.i, b.y by2, b.i bi FROM (SELECT * FROM a WHERE y = 2001 AND c = 1) a
    FULL JOIN (SELECT * FROM b WHERE c = 1) b ON a.y = b.y AND a.i = b.i AND a.c = b.c"""
    )
    got, want = _run(duck, tables, sql)
    assert_same(got, want)
