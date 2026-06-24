"""Workstream A — unified `ds.write` streaming orchestration + sinks.

The same `ds.write(...)` is batch or streaming: a bounded source with no trigger
writes once and returns a `WriteManifest`; a trigger (or an unbounded source) runs a
`StreamingQuery` against a streaming sink. These tests cover the triggers, output
modes, and the console/memory/foreachBatch sinks, plus that the batch path is
unchanged.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("k", pa.string()), ("v", pa.int64())])


def _batches():
    yield pa.RecordBatch.from_pydict({"k": ["a", "b"], "v": [1, 2]}, schema=_SCHEMA)
    yield pa.RecordBatch.from_pydict({"k": ["a", "c"], "v": [3, 4]}, schema=_SCHEMA)
    yield pa.RecordBatch.from_pydict({"k": ["b", "a"], "v": [5, 6]}, schema=_SCHEMA)


def _stream():
    return bt.from_batches(_batches, _SCHEMA, bounded=False)


@pytest.mark.integration
def test_append_stateless_to_memory():
    q = (
        _stream()
        .filter(bt.col("v") > 2)
        .write.memory("m_append", trigger=bt.Trigger.available_now())
    )
    q.await_termination()
    got = sorted(bt.read_memory("m_append").to_pydict()["v"])
    assert got == [3, 4, 5, 6]
    assert q.is_active is False


@pytest.mark.integration
def test_complete_aggregate_to_memory_matches_batch():
    q = (
        _stream()
        .group_by("k")
        .agg(total=bt.col("v").sum())
        .write.memory("m_complete", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    q.await_termination()
    streamed = dict(
        zip(*[bt.read_memory("m_complete").to_pydict()[c] for c in ("k", "total")], strict=False)
    )
    # Batch oracle over the same data.
    batch = (
        bt.from_arrow(pa.Table.from_batches(list(_batches())))
        .group_by("k")
        .agg(total=bt.col("v").sum())
    )
    oracle = dict(zip(*[batch.to_pydict()[c] for c in ("k", "total")], strict=False))
    assert streamed == oracle == {"a": 10, "b": 7, "c": 4}


@pytest.mark.integration
def test_foreach_batch_receives_whole_tables():
    seen: list[tuple[int, int]] = []

    def sink(table: pa.Table, batch_id: int) -> None:
        seen.append((batch_id, table.num_rows))

    q = _stream().write.for_each_batch(sink, trigger=bt.Trigger.available_now())
    q.await_termination()
    assert [n for _, n in seen] == [2, 2, 2]
    assert [bid for bid, _ in seen] == [0, 1, 2]


@pytest.mark.integration
def test_trigger_once_processes_one_microbatch():
    q = _stream().write.memory("m_once", trigger=bt.Trigger.once())
    q.await_termination()
    assert bt.read_memory("m_once").count() == 2  # only the first micro-batch
    assert q.recent_progress()[-1].batch_id == 0


@pytest.mark.integration
def test_append_on_aggregate_without_watermark_raises():
    with pytest.raises(PlanError, match="watermark"):
        _stream().group_by("k").agg(total=bt.col("v").sum()).write.memory(
            "m_bad", trigger=bt.Trigger.available_now(), output_mode="append"
        )


@pytest.mark.integration
def test_bounded_write_no_trigger_is_batch(tmp_path):
    out = str(tmp_path / "batch.parquet")
    manifest = bt.from_pydict({"k": ["a"], "v": [1]}).write(out)
    # Returns a WriteManifest (not a StreamingQuery) — the batch path is unchanged.
    assert manifest.total_rows == 1
    assert bt.read(out).count() == 1


@pytest.mark.integration
def test_continuous_trigger_processes_stateless_stream():
    # Continuous mode runs micro-batches back-to-back over a stateless pipeline.
    q = (
        bt.read.rate(5, num_rows=20, pace=False)
        .filter(bt.col("value") >= 5)
        .write.memory("m_cont", trigger=bt.Trigger.continuous("1s"))
    )
    q.await_termination()
    assert sorted(bt.read_memory("m_cont").to_pydict()["value"]) == list(range(5, 20))


@pytest.mark.integration
def test_continuous_trigger_rejects_aggregation():
    with pytest.raises(PlanError, match="continuous"):
        _stream().group_by("k").agg(total=bt.col("v").sum()).write.memory(
            "m_cont_bad", trigger=bt.Trigger.continuous("1s"), output_mode="complete"
        )


@pytest.mark.integration
def test_streaming_file_sink_round_trips(tmp_path):
    out = str(tmp_path / "stream_out")
    q = (
        _stream()
        .filter(bt.col("v") >= 3)
        .write(out, format="parquet", trigger=bt.Trigger.available_now())
    )
    q.await_termination()
    assert sorted(bt.read(out, format="parquet").to_pydict()["v"]) == [3, 4, 5, 6]
