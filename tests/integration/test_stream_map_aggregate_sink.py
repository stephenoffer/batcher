"""Inference then a rollup, written somewhere -- the shape an ML streaming job actually is.

`map_batches(model).group_by(...).agg(...)` streamed correctly through `iter_batches` and
answered every streaming write with a bare `NotImplementedError: map_batches is executed in
Python, not lowered to the engine IR`. That is true of `MapBatches.to_ir()` and says
nothing to someone who wrote a scoring UDF and a rollup, which is the most common thing
anyone does with a stream and a model.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("k", pa.string()), ("v", pa.int64())])
_BATCHES = [[("a", 1), ("b", 2)], [("a", 3)]]


def _stream():
    def feed():
        for rows in _BATCHES:
            yield pa.record_batch(
                {"k": [k for k, _ in rows], "v": [v for _, v in rows]}, schema=_SCHEMA
            )

    return bt.from_batches(feed, _SCHEMA, bounded=False)


def _score(batch: pa.RecordBatch) -> dict:
    """A stand-in for a model: per-batch, whole-Arrow, and opaque to the engine."""
    return {
        "k": batch.column("k").to_pylist(),
        "v": [value * 10 for value in batch.column("v").to_pylist()],
    }


def _scored(dataset):
    return dataset.map_batches(_score, output_columns=["k", "v"])


@pytest.mark.integration
def test_a_mapped_aggregate_reaches_a_sink():
    query = (
        _scored(_stream())
        .group_by("k")
        .agg(total=col("v").sum())
        .write.memory("map_agg_sink", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    assert query.await_termination(timeout=60) is True
    written = bt.read_memory("map_agg_sink").to_pydict()
    assert dict(zip(written["k"], written["total"], strict=True)) == {"a": 40, "b": 20}


@pytest.mark.integration
def test_the_sink_agrees_with_iter_batches():
    """One aggregate, two terminals. The fold is built against the *mapped* schema in both,
    which is the substitution that makes them the same computation rather than two."""
    streamed = [
        row
        for batch in _scored(_stream()).group_by("k").agg(total=col("v").sum()).iter_batches()
        for row in batch.to_pylist()
    ]
    query = (
        _scored(_stream())
        .group_by("k")
        .agg(total=col("v").sum())
        .write.memory("map_agg_parity", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    assert query.await_termination(timeout=60) is True
    written = bt.read_memory("map_agg_parity").to_pylist()
    assert sorted((r["k"], r["total"]) for r in streamed) == sorted(
        (r["k"], r["total"]) for r in written
    )


@pytest.mark.integration
def test_a_keyless_mapped_aggregate_reaches_a_sink_too():
    query = (
        _scored(_stream())
        .agg(total=col("v").sum())
        .write.memory("map_agg_global", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("map_agg_global").to_pydict()["total"][-1] == 60


@pytest.mark.integration
def test_update_mode_narrows_a_mapped_aggregate_to_the_group_that_moved():
    """The map does not change what `update` means: `b` gets no row in the second
    micro-batch, so its unchanged total is not re-sent."""
    seen: list[pa.Table] = []
    query = (
        _scored(_stream())
        .group_by("k")
        .agg(total=col("v").sum())
        .write.for_each_batch(
            lambda table, _id: seen.append(table),
            trigger=bt.Trigger.available_now(),
            output_mode="update",
        )
    )
    assert query.await_termination(timeout=60) is True
    emitted = [
        dict(zip(t.to_pydict()["k"], t.to_pydict()["total"], strict=True))
        for t in seen
        if t.num_rows
    ]
    assert emitted == [{"a": 10, "b": 20}, {"a": 40}]


@pytest.mark.integration
def test_a_checkpoint_is_refused_rather_than_resuming_from_an_empty_aggregate():
    """The fold is built against whatever schema the UDF returns, which is unknowable
    before the UDF has run -- so there is nothing to restore into. Accepting the argument
    would advance the offset log while the running total silently restarted at zero."""
    with pytest.raises(PlanError, match="aggregate over map_batches"):
        _scored(_stream()).group_by("k").agg(total=col("v").sum()).write.memory(
            "map_agg_ckpt",
            trigger=bt.Trigger.available_now(),
            output_mode="complete",
            checkpoint="/tmp/nope",
        )


@pytest.mark.integration
def test_a_udf_that_returns_nothing_leaves_the_aggregate_empty_rather_than_failing():
    """A filtering UDF is an ordinary thing to write, and the fold has no schema to build
    against until something comes back from it."""

    def drop_everything(batch: pa.RecordBatch) -> dict:
        return {"k": [], "v": []}

    query = (
        _stream()
        .map_batches(drop_everything, output_columns=["k", "v"])
        .group_by("k")
        .agg(total=col("v").sum())
        .write.memory("map_agg_empty", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("map_agg_empty").count() == 0
