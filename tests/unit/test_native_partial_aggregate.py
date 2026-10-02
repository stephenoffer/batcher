"""The engine-read map side of a distributed aggregate equals the Python-driven fold.

`native_partial_aggregate` hands a Parquet partition's row groups to the engine, whose
workers read, map and fold them in one call; `streaming_partial_aggregate` decodes the
partition in Python and folds it chunk by chunk. The reducers cannot tell the two apart, so
the partial each produces must finalize to the same rows. Pinned per partition, over a
source whose group key carries NULLs, with a predicate that keeps rows and one that keeps
none (the typed empty partial), and with the fallback contract: anything that is not a
manifest of plain row groups returns ``None`` so the caller folds it the old way.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.native import engine
from batcher.dist.executors.partition_io import (
    iter_partition_descriptor,
    native_partial_aggregate,
    partition_descriptors,
    streaming_partial_aggregate,
)
from batcher.dist.flight_aggregate import _relabel_single_source
from batcher.plan.ir_specs import agg_spec_json

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def parquet_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("native_fold")
    for i in range(3):
        n = 40_000
        k = [(j * 7 + i) % 97 if j % 41 else None for j in range(n)]
        v = [float(j % 13) + 0.25 for j in range(n)]
        pq.write_table(pa.table({"k": k, "v": v}), root / f"p{i}.parquet", row_group_size=8_000)
    return str(root)


def _spec(ds):
    agg = ds._plan
    gk, aj = agg_spec_json(agg)
    map_plan, sid = _relabel_single_source(agg.input)
    return json.dumps(map_plan.to_ir()), gk, aj, ds._sources[sid]


def _rows(nat, gk, aj, partial):
    return nat.combine_finalize(gk, aj, [partial]).sort_by("k").to_pylist()


def _grouped(src, threshold):
    return (
        src.filter(bt.col("v") > threshold)
        .group_by("k")
        .agg(s=bt.col("v").sum(), n=bt.col("v").count(), lo=bt.col("v").min())
    )


@pytest.mark.parametrize("threshold", [1.0, 99.0])
def test_each_partition_folds_to_the_streaming_partial(parquet_dir, threshold):
    nat = engine()
    map_ir, gk, aj, source = _spec(_grouped(bt.read.parquet(parquet_dir), threshold))
    parts = partition_descriptors(source, 2)
    assert len(parts) == 2 and all(p.get("splits") for p in parts)
    for part in parts:
        metrics: list[str] = []
        native = native_partial_aggregate(nat, map_ir, gk, aj, part, "", on_metrics=metrics.append)
        assert native is not None, "a row-group manifest must take the engine path"
        assert metrics, "the engine path must still meter the map prefix"
        chunked = streaming_partial_aggregate(
            nat, map_ir, gk, aj, iter_partition_descriptor(part), ""
        )
        assert native.schema == chunked.schema
        assert _rows(nat, gk, aj, native) == _rows(nat, gk, aj, chunked)


def test_the_partitions_combine_to_the_single_node_answer(parquet_dir):
    nat = engine()
    ds = _grouped(bt.read.parquet(parquet_dir), 1.0)
    map_ir, gk, aj, source = _spec(ds)
    partials = [
        native_partial_aggregate(nat, map_ir, gk, aj, p, "")
        for p in partition_descriptors(source, 3)
    ]
    got = nat.combine_finalize(gk, aj, partials).sort_by("k").to_pylist()
    want = ds.collect(distributed=False).sort_by("k").to_pylist()
    assert got == want


def test_a_partition_that_is_not_a_row_group_manifest_falls_back(parquet_dir):
    nat = engine()
    table = pq.read_table(parquet_dir)
    map_ir, gk, aj, source = _spec(_grouped(bt.from_arrow(table), 1.0))
    part = partition_descriptors(source, 2)[0]
    assert native_partial_aggregate(nat, map_ir, gk, aj, part, "") is None
