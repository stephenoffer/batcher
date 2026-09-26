"""A join's build-side aggregate restricted to the probe side's keys returns what DuckDB returns.

Kyber sends `prefer_sideways` when a join reads an aggregate far larger than its probe side, and
the engine then runs the plan on the executor that restricts the aggregate's input to the probe
keys first (`bc_interp::join_par::sideways`). These queries are that shape at a size that clears
every gate — a 400,000-row fact table grouped on the join key under a small outer side, with NULL
keys on both sides, duplicate outer keys and outer keys with no group — under each join type the
restriction admits, including TPC-H q21's `EXISTS`/`NOT EXISTS` pair.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_FACT_ROWS = 400_000
_FACT = pa.table(
    {
        "k": pa.array(
            [None if i % 1_013 == 0 else i % 60_000 for i in range(_FACT_ROWS)], pa.int64()
        ),
        "s": pa.array([i % 17 for i in range(_FACT_ROWS)], pa.int64()),
        "v": pa.array([float(i % 91) for i in range(_FACT_ROWS)]),
    }
)
_OUTER = pa.table(
    {
        "ok": pa.array(
            [None if i % 53 == 0 else (i * 37) % 70_000 for i in range(2_500)], pa.int64()
        ),
        "os": pa.array([i % 17 for i in range(2_500)], pa.int64()),
    }
)

_QUERIES = [
    "SELECT ok, n, t FROM outer_t LEFT JOIN (SELECT k, count(*) AS n, sum(v) AS t FROM fact "
    "GROUP BY k) g ON ok = g.k",
    "SELECT ok, n FROM outer_t JOIN (SELECT k, count(*) AS n FROM fact GROUP BY k) g ON ok = g.k",
    "SELECT count(*) AS c FROM outer_t WHERE EXISTS (SELECT 1 FROM fact WHERE k = ok AND s <> os)",
    "SELECT count(*) AS c FROM outer_t WHERE NOT EXISTS "
    "(SELECT 1 FROM fact WHERE k = ok AND s <> os)",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_restricted_build_side_matches_duckdb(duck, query):
    duck.register("fact", _FACT)
    duck.register("outer_t", _OUTER)
    s = bt.Session()
    s.register("fact", _FACT)
    s.register("outer_t", _OUTER)
    got = s.sql(query).collect()
    assert got.num_rows > 0
    assert_same_for_query(got, duck.sql(query), query)
