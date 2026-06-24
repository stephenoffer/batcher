"""Workstream G — Delta Lake as an incremental streaming source + Change Data Feed.

`bt.read.delta(uri, stream=True)` reads new commits as an unbounded stream (the
medallion layer-chaining mechanism); `bt.read.read_change_feed(uri)` yields the
row-level CDC stream. Both are Checkpointable by Delta version, so a streaming query
resumes exactly-once after a restart.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

deltalake = pytest.importorskip("deltalake")
pytestmark = pytest.mark.integration

_CFG = {"delta.enableChangeDataFeed": "true"}


def _write(uri, ids, mode="append"):
    deltalake.write_deltalake(
        uri, pa.table({"id": ids, "v": [i * 10 for i in ids]}), mode=mode, configuration=_CFG
    )


def test_delta_stream_reads_appended_commits(tmp_path):
    uri = str(tmp_path / "t")
    _write(uri, [1, 2], mode="overwrite")
    _write(uri, [3])
    ds = bt.read.delta(uri, stream=True, starting_version=0)
    rows = pa.Table.from_batches(list(ds.iter_batches()))
    assert sorted(rows.column("id").to_pylist()) == [1, 2, 3]
    assert rows.column_names == ["id", "v"]  # append mode drops CDF metadata cols


def test_change_data_feed_yields_change_types(tmp_path):
    uri = str(tmp_path / "t")
    _write(uri, [1, 2], mode="overwrite")
    feed = bt.read.read_change_feed(uri, starting_version=0)
    cdf = pa.Table.from_batches(list(feed.iter_batches()))
    assert "_change_type" in cdf.column_names
    assert set(cdf.column("_change_type").to_pylist()) == {"insert"}


def test_delta_stream_checkpoint_resumes_by_version(tmp_path):
    uri = str(tmp_path / "t")
    out = str(tmp_path / "out")
    ckpt = str(tmp_path / "ck")
    _write(uri, [1, 2], mode="overwrite")

    # Run 1: stream the first commits to a Parquet sink, checkpointed.
    bt.read.delta(uri, stream=True, starting_version=0).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    ).await_termination()
    assert sorted(bt.read(out, format="parquet").to_pydict()["id"]) == [1, 2]

    # A new commit arrives, then the query restarts against the same checkpoint:
    # it resumes at the recorded version and reads ONLY the new commit.
    _write(uri, [3, 4])
    bt.read.delta(uri, stream=True, starting_version=0).write(
        out, format="parquet", trigger=bt.Trigger.available_now(), checkpoint=ckpt
    ).await_termination()
    assert sorted(bt.read(out, format="parquet").to_pydict()["id"]) == [1, 2, 3, 4]


def test_delta_stream_is_checkpointable():
    from batcher.io.formats.lakehouse.delta import DeltaStreamSource

    src = DeltaStreamSource("/tmp/nonexistent", starting_version=5)
    assert src.bounded is False
    assert src.snapshot_position() == {"version": 4}  # next read starts at 5
    src.seek({"version": 10})
    assert src.snapshot_position() == {"version": 10}


def test_change_feed_captures_deletes(tmp_path):
    import pyarrow.compute as pc

    uri = str(tmp_path / "t")
    _write(uri, [1, 2], mode="overwrite")
    deltalake.DeltaTable(uri).delete("id = 1")  # a delete commit
    feed = bt.read.read_change_feed(uri, starting_version=0)
    cdf = pa.Table.from_batches(list(feed.iter_batches()))
    types = set(pc.cast(cdf.column("_change_type"), pa.string()).to_pylist())
    assert "delete" in types and "insert" in types


def test_delta_streaming_sink_appends_per_microbatch(tmp_path):
    out = str(tmp_path / "sink")
    sch = pa.schema([("id", pa.int64())])

    def feed():
        yield pa.record_batch({"id": [1, 2]}, schema=sch)
        yield pa.record_batch({"id": [3]}, schema=sch)

    q = bt.from_batches(feed, sch, bounded=False).write.delta(
        out, trigger=bt.Trigger.available_now()
    )
    q.await_termination()
    assert sorted(bt.read.delta(out).to_pydict()["id"]) == [1, 2, 3]
