"""A UDF operand whose staged output is empty must still run its breaker, distributed.

`map_batches(...).join(other)` runs the UDF branch as its own distributed stage onto shared
scratch, then joins a scan of that scratch. When every shard kept no rows, no file was
written and the stage reported no schema, so the operand was *declined* -- and on splittable
data a declined shape raises `PlanError` (audit finding F100). A filter that happens to match
nothing turned a correct query into an error.

It is declined rather than folded to "empty" for a reason that still holds: a breaker is not
uniformly empty-preserving, and a left join with an empty right side emits every left row.
So the fix keeps the breaker and gives it an empty input of the right *type*: each writing
shard now reports the UDF's output schema even when it wrote nothing
(`dist.executors.map._empty_write`), and the empty operand is staged as a zero-row input of
that schema. Each join type then applies its own empty-input semantics, checked here against
single-node on rows, column names and column types.

The non-empty identity UDF is the control: the same shape distributed before the fix, so a
pass there shows the route, and a failure only in the empty column shows the defect.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_tables_equal
from _ray_cluster import ray_session_fixture

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_N = 3000
_WORKERS = 2

_ray_session = ray_session_fixture(4)


@pytest.fixture(scope="module")
def tables(cluster_scratch) -> tuple[str, str]:
    left, right = cluster_scratch("empty_udf_left"), cluster_scratch("empty_udf_right")
    for part in range(3):
        keys = list(range(part, _N, 3))
        pq.write_table(pa.table({"k": keys, "v": [k % 7 for k in keys]}), left / f"p{part}.parquet")
        pq.write_table(
            pa.table({"k": keys, "w": [float(k % 5) for k in keys]}), right / f"p{part}.parquet"
        )
    return str(left), str(right)


#: Lambdas rather than module functions: a worker unpickles a module-level function by
#: importing its module, and a test module is not importable on a Ray worker.
_UDFS = {
    "control": lambda batch: batch,
    "empty": lambda batch: batch.filter(pa.array([False] * batch.num_rows)),
}


@pytest.mark.parametrize("udf", sorted(_UDFS))
@pytest.mark.parametrize("how", ["inner", "left", "right", "semi", "anti"])
def test_a_join_over_a_udf_operand_matches_single_node(tables, udf, how):
    left, right = (bt.read.parquet(p) for p in tables)
    ds = left.join(right.map_batches(_UDFS[udf]), on="k", how=how)
    want = ds.collect(distributed=False)
    got = ds.collect(distributed=True, num_workers=_WORKERS)
    assert got.schema == want.schema
    order = [("k", "ascending")]
    assert_tables_equal(got.sort_by(order), want.sort_by(order), ordered=True)
    if how in ("left", "anti") and udf == "empty":
        assert got.num_rows == _N, "the control needs the left side to survive"
