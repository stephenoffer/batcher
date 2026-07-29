"""An in-memory streaming sink must read back as a relation, even with no rows.

A stream whose filter matched nothing, and a `complete`-mode query whose running result
went to zero rows, are ordinary outcomes. `bt.read_memory` answered both with a table that
had *no columns at all*, which then fails to concatenate against the same query's non-empty
run and reads as "no such relation" rather than "no such rows".
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.formats.streaming.sinks import _MEMORY, _MEMORY_SCHEMA, MemoryStreamSink


@pytest.fixture(autouse=True)
def _clean():
    yield
    for name in ("empty-stream", "typed-empty", "learned", "unschooled"):
        _MEMORY.pop(name, None)
        _MEMORY_SCHEMA.pop(name, None)


def test_a_stream_that_matches_nothing_still_reads_back_typed():
    ds = bt.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]}).filter(bt.col("a") > 100)
    q = ds.write.memory("empty-stream", trigger=bt.Trigger.available_now())
    assert q.await_termination(30) is True

    out = bt.read_memory("empty-stream").collect()
    assert out.num_rows == 0
    assert out.schema.names == ["a", "b"]
    assert out.schema.field("a").type == pa.int64()


def test_the_empty_result_concatenates_with_a_non_empty_run():
    """The property that makes the schema matter: an empty run and a full one are the same
    relation, so they must be combinable."""
    full = bt.from_pydict({"a": [1], "b": ["x"]})
    q = full.write.memory("typed-empty", trigger=bt.Trigger.available_now())
    assert q.await_termination(30) is True
    non_empty = bt.read_memory("typed-empty").collect()

    empty = bt.from_pydict({"a": [1], "b": ["x"]}).filter(bt.col("a") > 100)
    q2 = empty.write.memory("empty-stream", trigger=bt.Trigger.available_now())
    assert q2.await_termination(30) is True

    combined = pa.concat_tables([non_empty, bt.read_memory("empty-stream").collect()])
    assert combined.num_rows == 1


def test_the_sink_learns_a_schema_from_data_when_none_was_supplied():
    """A sink built directly (no conductor to hand it the plan's schema) still reads back
    typed once it has seen a batch — the `complete`-mode-emptied case."""
    sink = MemoryStreamSink("learned", output_mode="complete")
    sink.open()
    sink.write_batch(0, pa.table({"g": [1], "n": [2]}))
    sink.write_batch(1, pa.table({"g": pa.array([], pa.int64()), "n": pa.array([], pa.int64())}))

    from batcher.io.formats.streaming.sinks import memory_table

    out = memory_table("learned")
    assert out.num_rows == 0
    assert out.schema.names == ["g", "n"]


def test_a_never_written_sink_with_no_schema_is_still_readable():
    """No conductor schema and no data: there is nothing to type it from, so the historical
    column-less table is the honest answer rather than a crash."""
    from batcher.io.formats.streaming.sinks import memory_table

    sink = MemoryStreamSink("unschooled")
    sink.open()
    assert memory_table("unschooled").num_rows == 0


def test_reopening_replaces_a_stale_schema():
    sink = MemoryStreamSink("learned", schema=pa.schema([("a", pa.int64())]))
    sink.open()
    assert _MEMORY_SCHEMA["learned"].names == ["a"]
    MemoryStreamSink("learned", schema=pa.schema([("b", pa.string())])).open()
    assert _MEMORY_SCHEMA["learned"].names == ["b"]
