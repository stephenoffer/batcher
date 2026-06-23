"""map_batches composed with joins/unions — non-linear UDF pipelines."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _add_one(col_name: str):
    def fn(batch: pa.RecordBatch) -> pa.RecordBatch:
        bumped = pa.array([x + 1 for x in batch.column(col_name).to_pylist()], type=pa.int64())
        arrays = [bumped if name == col_name else batch.column(name) for name in batch.schema.names]
        return pa.RecordBatch.from_arrays(arrays, names=list(batch.schema.names))

    return fn


def _sorted(table: pa.Table, key: str) -> dict:
    return table.sort_by(key).to_pydict()


def test_map_batches_then_join():
    a = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    b = pa.table({"id": [1, 2, 3], "w": [100, 200, 300]})
    da = bt.from_arrow(a).map_batches(_add_one("v"))  # v -> v+1
    out = da.join(bt.from_arrow(b), on="id").collect()
    got = _sorted(out, "id")
    assert got["id"] == [1, 2, 3]
    assert got["v"] == [11, 21, 31]
    assert got["w"] == [100, 200, 300]


def test_join_with_udf_on_both_sides():
    a = pa.table({"id": [1, 2], "v": [10, 20]})
    b = pa.table({"id": [1, 2], "w": [5, 6]})
    da = bt.from_arrow(a).map_batches(_add_one("v"))  # 11, 21
    db = bt.from_arrow(b).map_batches(_add_one("w"))  # 6, 7
    got = _sorted(da.join(db, on="id").collect(), "id")
    assert got["v"] == [11, 21]
    assert got["w"] == [6, 7]


def test_union_of_udf_branches():
    a = pa.table({"id": [1, 2], "v": [10, 20]})
    b = pa.table({"id": [3, 4], "v": [30, 40]})
    da = bt.from_arrow(a).map_batches(_add_one("v"))  # 11, 21
    db = bt.from_arrow(b).map_batches(_add_one("v"))  # 31, 41
    got = _sorted(da.union(db).collect(), "v")
    assert got["v"] == [11, 21, 31, 41]


def test_empty_side_join_does_not_crash():
    # Filter the UDF side to empty, then join — the tracked schema lets the engine
    # scan the empty input without a batch to read a schema from.
    from batcher import col

    a = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    b = pa.table({"id": [1, 2, 3], "w": [100, 200, 300]})
    da = bt.from_arrow(a).filter(col("v") > 1000).map_batches(_add_one("v"))  # empty
    out = da.join(bt.from_arrow(b), on="id").collect()
    assert out.num_rows == 0  # inner join with an empty side → empty, no crash
