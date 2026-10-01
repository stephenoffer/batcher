"""Pre-aggregating the facts beneath a string-keyed dimension matches DuckDB.

`pre_aggregate_beneath_dimension` turns TPC-DS q4/q11's `year_total` -- facts joined to
`customer` and `date_dim`, grouped by `customer` strings and `d_year` -- into `customer JOIN
Agg_partial(facts JOIN date_dim)`. What it must get right is the merge: a `count` of partial
counts, `min`/`max` of partials, sums over NULL measures, NULL join keys that match nothing,
duplicate dimension keys that fan a partial out, and an input that a filter empties. The
fixture is 120,000 rows, past `MIN_ROWS_TO_SHARD`, so the parallel executor shards it.

Each answer is also compared with the rule switched off, and a control shows the rule fired:
without it every comparison here would pass on a plan the rule never touched.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_for_query
from batcher import kyber
from batcher.config import active_config, config_context
from batcher.core import default_hub
from batcher.kyber.rules.agg_pushdown import reassociate
from batcher.plan.logical import Aggregate
from batcher.plan.visitor import walk

pytestmark = pytest.mark.differential

_N = 120_000


def _tables() -> dict[str, pa.Table]:
    rng = np.random.default_rng(5)
    cust = rng.integers(0, 3_000, _N).astype("float64")
    cust[rng.random(_N) < 0.03] = np.nan  # a NULL customer matches nothing
    v = rng.integers(-100, 1_000, _N).astype("float64")
    v[rng.random(_N) < 0.05] = np.nan  # NULL measures
    facts = pa.table(
        {
            "f_cust": pa.array(cust, from_pandas=True).cast(pa.int64()),
            "f_date": pa.array(rng.integers(0, 1_460, _N), pa.int64()),
            "f_v": pa.array(v, from_pandas=True),
        }
    )
    # Customer key 7 appears twice, so its partials fan out across two string groups.
    sk = [*range(3_000), 7]
    customer = pa.table(
        {
            "c_sk": pa.array(sk, pa.int64()),
            "c_id": pa.array([f"AAAA{i:08d}" for i in sk[:-1]] + ["DUPLICATE"]),
            "c_first": pa.array([None if i % 97 == 0 else f"f{i % 400}" for i in sk]),
            "c_last": pa.array([f"l{i % 900}" for i in sk]),
        }
    )
    dates = pa.table(
        {
            "d_sk": pa.array(range(1_460), pa.int64()),
            "d_year": pa.array([2000 + i // 365 for i in range(1_460)], pa.int64()),
        }
    )
    return {"facts": facts, "customer": customer, "dates": dates}


@pytest.fixture(scope="module")
def tables():
    return _tables()


def _session(tables) -> bt.Session:
    s = bt.Session()
    for name, t in tables.items():
        s.register(name, t)
    return s


_YEAR_TOTAL = """
SELECT c_id, c_first, c_last, d_year,
       sum(f_v) tot, count(*) n, count(f_v) nv, min(f_v) lo, max(f_v) hi
FROM facts, customer, dates
WHERE f_cust = c_sk AND f_date = d_sk {where}
GROUP BY c_id, c_first, c_last, d_year
"""

_QUERIES = [
    _YEAR_TOTAL.format(where=""),
    _YEAR_TOTAL.format(where="AND d_year IN (2001, 2002)"),
    _YEAR_TOTAL.format(where="AND f_v > 100000"),  # empty
    # TPC-DS q11's outer shape: the year totals compared year over year, ordered.
    """
    WITH t AS (SELECT c_id, d_year, sum(f_v) tot FROM facts, customer, dates
               WHERE f_cust = c_sk AND f_date = d_sk GROUP BY c_id, c_last, d_year)
    SELECT a.c_id, a.tot, b.tot AS tot2 FROM t a, t b
    WHERE a.c_id = b.c_id AND a.d_year = 2001 AND b.d_year = 2002 AND b.tot > a.tot
    ORDER BY a.c_id LIMIT 50
    """,
]


@pytest.fixture
def no_plan_cache():
    """Plan every query afresh, so a memoized plan cannot answer for the one under test."""
    cfg = active_config()
    with config_context(
        cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, plan_cache_entries=0))
    ):
        yield


@pytest.mark.parametrize("query", _QUERIES)
def test_matches_duckdb(duck, tables, query):
    for name, t in tables.items():
        duck.register(name, t)
    s = _session(tables)
    s.sql(query).collect()  # seed the distinct counts the rule's gate reads
    assert_same_for_query(s.sql(query).collect(), duck.sql(query), query)


def test_the_rule_fires_on_the_executed_plan(tables, no_plan_cache):
    s = _session(tables)
    ds = s.sql(_QUERIES[0])
    ds.collect()
    opt = kyber.optimize_logical(ds._plan, sources=ds._sources, hub=default_hub())
    partials = [
        n
        for n in walk(opt)
        if isinstance(n, Aggregate) and any(a.alias.startswith("__rp") for a in n.aggregates)
    ]
    assert partials, "the facts were not pre-aggregated beneath the customer join"


def test_the_answer_does_not_depend_on_the_rule(tables, monkeypatch, no_plan_cache):
    """A/B on the engine alone. The measures are whole numbers, so the sums are exact."""
    s = _session(tables)
    s.sql(_QUERIES[0]).collect()
    with_rule = s.sql(_QUERIES[0]).collect()
    monkeypatch.setattr(reassociate, "_partial_reduces", lambda *_args: False)
    without = s.sql(_QUERIES[0]).collect()

    def rows(t):
        return sorted(t.to_pylist(), key=repr)

    assert rows(with_rule) == rows(without)
