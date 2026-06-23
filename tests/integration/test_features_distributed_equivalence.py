"""Every feature added this cycle runs in distributed mode and produces results
identical to single-node (the mergeable-algebra / single-node-fallback invariant).

Distributed dispatch handles aggregate/join/sort/map directly; other shapes (e.g.
window) fall back to the single-node engine — either way the result must match."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture(autouse=True)
def _need_ray():
    pytest.importorskip("ray")


def _both(ds, key: str):
    single = ds.collect().sort_by(key).to_pydict()
    dist = ds.collect(distributed=True, num_workers=2).sort_by(key).to_pydict()
    return single, dist


def test_window_distributed_equals_single_node():
    t = pa.table({"k": [i % 4 for i in range(400)], "v": list(range(400))})
    w = bt.from_arrow(t).window(
        partition_by=["k"], order_by=[("v", False)], functions={"rn": "row_number"}
    )
    single, dist = _both(w, "v")
    assert single == dist


def test_udf_join_distributed_equals_single_node():
    a = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    b = pa.table({"id": [1, 2, 3], "w": [100, 200, 300]})

    def add1(batch: pa.RecordBatch) -> pa.RecordBatch:
        arr = pa.array([x + 1 for x in batch.column("v").to_pylist()], type=pa.int64())
        return pa.RecordBatch.from_arrays([batch.column("id"), arr], names=["id", "v"])

    j = bt.from_arrow(a).map_batches(add1).join(bt.from_arrow(b), on="id")
    single, dist = _both(j, "id")
    assert single == dist


def test_parallel_map_batches_distributed_equals_single_node():
    t = pa.table({"k": [i % 4 for i in range(400)], "v": list(range(400))})
    m = bt.from_arrow(t).map_batches(lambda x: x, batch_size=50, num_workers=4)
    single, dist = _both(m, "v")
    assert single == dist


def test_filter_aggregate_distributed_equals_single_node():
    t = pa.table({"k": [i % 8 for i in range(800)], "v": list(range(800))})
    from batcher import count

    g = bt.from_arrow(t).filter(col("v") > 200).group_by("k").agg(n=count(), s=col("v").sum())
    single, dist = _both(g, "k")
    assert single == dist
