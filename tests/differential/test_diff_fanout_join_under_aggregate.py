"""A high fan-out join reduced by an aggregate stays in memory under a tight budget, vs DuckDB.

The spill gate charged the plan's widest operator output as resident, and a self-join with
800 rows a side per key outputs 800 x its row count. Under any budget that output exceeds,
the gate routed the query out of core -- and the out-of-core path aggregates the spilled
join's *complete* result, so it holds exactly the rows the gate was afraid of: a 4M-row
self-join was OOM-killed on a 64 GB node, where the streaming executor, which folds the join
into the aggregate morsel by morsel, peaks at 0.6 GB. The gate now leaves out an output an
aggregate consumes (`carbonite.policies.spill_advice._folded_by_an_aggregate`).

Here a memory cap far below the join's output makes the old routing certain. The aggregate
query must answer like DuckDB and must not go out of core; the control is the same join with
no aggregate over it, whose output *is* the result, and which still does.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.config import option_context

pytestmark = pytest.mark.differential

_ROWS = 20_000
_KEYS = 25  # 800 rows a side per key: 16M joined rows
_CAP = 8 << 20
_OUT_OF_CORE = "out-of-core"


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pa.table(
        {
            "a": pa.array([i % _KEYS for i in range(_ROWS)], pa.int64()),
            "b": pa.array([i * 0.5 for i in range(_ROWS)], pa.float64()),
        }
    )


def test_a_high_fan_out_join_under_an_aggregate_matches_duckdb_in_memory(duck, table) -> None:
    ds = bt.from_arrow(table)
    query = ds.join(ds, on="a").agg(
        n=bt.col("a").count(), m=(bt.col("b") - bt.col("b_right")).abs().max()
    )
    duck.register("t", table)
    with option_context("memory.max_memory_bytes", _CAP):
        got = query.collect()
        plan = query.explain(analyze=True)
    assert_same(
        got,
        duck.sql("SELECT count(l.a) AS n, max(abs(l.b - r.b)) AS m FROM t l JOIN t r ON l.a = r.a"),
    )
    assert got.column("n")[0].as_py() == _ROWS * (_ROWS // _KEYS)
    assert _OUT_OF_CORE not in plan, plan


def test_the_same_join_with_nothing_reducing_it_still_goes_out_of_core(table) -> None:
    """The positive control: the gate still sees a join output that is the result."""
    ds = bt.from_arrow(table)
    query = ds.join(ds, on="a").select("a", "b", "b_right")
    with option_context("memory.max_memory_bytes", _CAP):
        plan = query.explain(analyze=True)
    assert _OUT_OF_CORE in plan, plan
