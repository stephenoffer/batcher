"""Section-O guard regression tests — lock in the structural guarantees that
distinguish Batcher from documented Ray Data failure modes.

The plan's rule is "a guard without a test is not done." These assert the
*structural* guards (no GPU/cluster needed): the Arrow-only contract, immutable
dataflow, schema-known-after-limit, streamable map pipelines, and metadata-driven
count — each closing a named Ray Data pain point so a future change can't silently
regress it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher import col
from batcher.plan.logical import is_streamable


def test_o4_3_udf_receives_arrow_recordbatch_one_columnar_contract():
    # Ray pays an Arrow→pandas→Arrow conversion tax per stage; Batcher's UDF boundary
    # is Arrow end to end. The callback must see a pyarrow.RecordBatch, not pandas.
    seen = {}

    def grab(batch):
        seen["type"] = type(batch).__module__ + "." + type(batch).__name__
        return batch

    bt.from_pydict({"a": [1, 2, 3]}).ml.map_batches(grab, output_columns=["a"]).collect()
    assert seen["type"] == "pyarrow.lib.RecordBatch"


def test_o1_6_immutable_dataflow_new_column_udf_works():
    # No read-only-zero-copy mutation trap (Ray's ValueError): UDFs return NEW batches.
    def add_flag(batch):
        cols = [*batch.columns, pa.array([1] * batch.num_rows, type=pa.int64())]
        return pa.RecordBatch.from_arrays(cols, names=[*batch.schema.names, "flag"])

    ds = bt.from_pydict({"x": [1, 2, 3]})
    out = ds.ml.map_batches(add_flag, output_columns=["x", "flag"]).collect()
    assert out.column_names == ["x", "flag"]
    assert out.column("flag").to_pylist() == [1, 1, 1]


def test_o14_5_limit_and_head_keep_schema_known():
    # Ray's .limit() can return an Unknown schema; Batcher knows it from the plan.
    ds = bt.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert ds.head(2).columns == ["a", "b"]
    assert ds.limit(1).columns == ["a", "b"]
    assert ds.head(2).collect().column_names == ["a", "b"]


def test_o8_1_map_filter_project_pipeline_is_streamable():
    # A breaker-free pipeline streams (bounded memory, no all-to-all materialize),
    # unlike Ray where careless ops break streaming execution.
    d = bt.from_pydict({"x": list(range(100))})
    assert is_streamable(d.filter(col("x") > 10).select("x")._plan)
    # A group-by is a genuine breaker — correctly NOT a whole-plan streamable map.
    assert not is_streamable(d.group_by("x").agg(n=bt.count())._plan)


def test_o8_6_head_streams_without_materializing_whole_source():
    # head(n) over a streamable pipeline yields n rows; correctness of the short-circuit
    # is covered in test_limit_shortcircuit — here we assert the contract holds via API.
    out = bt.from_pydict({"x": list(range(1000))}).head(5).collect()
    assert out.column("x").to_pylist() == [0, 1, 2, 3, 4]


def test_o4_3_explode_keeps_columns_known():
    e = bt.from_pydict({"id": [1], "xs": [[1, 2, 3]]}).explode("xs", alias="x")
    assert e.columns == ["id", "x"]
    assert e.collect().num_rows == 3


def test_count_is_exact_on_in_memory_source():
    assert bt.from_pydict({"x": list(range(42))}).count() == 42


def test_count_and_aggregate_over_map_batches_do_not_crash_metadata_fastpath():
    # The metadata fast-path (count/aggregate from stats) must degrade to normal
    # execution for a map_batches/ML pipeline (opaque to the IR), never crash.
    ds = bt.from_pydict({"x": [1, 2, 3]}).ml.map_batches(lambda b: b, output_columns=["x"])
    assert ds.count() == 3
    agg = ds.group_by().agg(n=bt.count()).collect()
    assert agg.column("n").to_pylist() == [3]
