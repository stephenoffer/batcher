"""Invariant #7 at the streaming gate: distributed must never mean *different*.

`single-node == distributed` is a hard invariant, and the distributed streaming gate is where
it is enforced for streams: a plan the cluster cannot fold with single-node semantics is
**refused**, not run with different ones.

One shape slipped through. A watermarked windowed aggregation is an `Aggregate` over a
streamable input, so the old gate (`isinstance(plan, Aggregate) and is_streamable(plan.input)`)
waved it past — while `python/batcher/dist/` contains no watermark at all: no window eviction,
no late-row drop, no append output mode. The query silently degraded to an unbounded
complete-mode aggregate that re-emits the whole running result every epoch and grows state
without bound. Same query, `distributed=True` vs `False`, different answers, no error.

These tests pin the gate's decision directly. Reaching it end to end needs a partitionable
source, but the decision *is* the contract.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.api.io_namespace.writer import _undistributable_stream_reason

pytestmark = pytest.mark.unit


def _stream():
    """An unbounded source — the only kind a streaming write folds."""
    import datetime as dt

    import pyarrow as pa

    schema = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])
    base = dt.datetime(2024, 1, 1)

    def batches():
        yield pa.RecordBatch.from_pylist([{"ts": base, "v": 1}], schema=schema)

    return bt.from_batches(batches, schema, bounded=False)


def test_plain_streaming_aggregate_is_distributable():
    """The shape the mergeable algebra genuinely covers: partial → combine → finalize."""
    plan = _stream().group_by("v").agg(total=col("v").sum())._plan
    assert _undistributable_stream_reason(plan) is None


def test_watermarked_aggregate_is_refused_not_silently_degraded():
    plan = (
        _stream()
        .with_watermark("ts", "5m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("v").sum())
        ._plan
    )

    reason = _undistributable_stream_reason(plan)

    assert reason is not None, (
        "a watermarked windowed aggregate must be refused: the distributed runner has no "
        "watermark, so running it would return a different result than single-node"
    )
    assert "watermark" in reason
    # The message must point at the way out, not just say no.
    assert "distributed=False" in reason


def test_a_second_pipeline_breaker_is_still_refused():
    """The case the gate always caught — it must keep catching it."""
    plan = _stream().sort("v").group_by("v").agg(total=col("v").sum())._plan
    reason = _undistributable_stream_reason(plan)
    assert reason is not None
    assert "pipeline breaker" in reason
