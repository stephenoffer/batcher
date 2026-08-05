"""Continuous processing has no micro-batch boundary, so a stateful plan cannot use it.

The single-source launcher refused this from the start. The driver path -- the stream-
stream join and the stream union, and now the stream-static join, the session window, the
watermark dedup and a limit -- never checked, so a continuous trigger was accepted and then
quietly run as micro-batches. The answer was right; the latency the caller asked for was
not what they got, and nothing said so.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])


def _stream():
    def feed():
        yield pa.record_batch({"k": ["a"], "ts": [_BASE], "v": [1]}, schema=_SCHEMA)

    return bt.from_batches(feed, _SCHEMA, bounded=False)


def _dimension():
    return bt.from_pydict({"k": ["a"], "lab": ["A"]})


_STATEFUL = {
    "session_window": lambda: _stream().session_window(
        "ts", "5m", partition_by=["k"], total=col("v").sum()
    ),
    "stream_static_join": lambda: _stream().join(_dimension(), on="k", how="inner"),
    "watermark_dedup": lambda: (
        _stream()
        .with_watermark("ts", "1h")
        .drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h")
    ),
    "limit": lambda: _stream().head(1),
    "interval_join": lambda: _stream().join_stream(
        _stream(), on="k", left_time="ts", right_time="ts", within="5m"
    ),
    "union": lambda: _stream().union(_stream()),
}


@pytest.mark.integration
@pytest.mark.parametrize("shape", sorted(_STATEFUL))
def test_a_continuous_trigger_is_refused_for_a_stateful_plan(shape):
    with pytest.raises(PlanError, match="only stateless pipelines"):
        _STATEFUL[shape]().write.memory(f"cont_{shape}", trigger=bt.Trigger.continuous("1 second"))


@pytest.mark.integration
def test_a_stateless_pipeline_still_takes_one():
    """The refusal is about retained state, not about the driver path."""
    query = (
        _stream()
        .filter(col("v") > 0)
        .write.memory("cont_stateless", trigger=bt.Trigger.continuous("1 second"))
    )
    query.stop()
    assert query.is_active is False
