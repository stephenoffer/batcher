"""A limited stream ends, so it should be able to land somewhere before it does.

`bt.read.kafka(...).head(1000).write.parquet(...)` -- take the first thousand events to a
table and stop -- is how anyone looks at an unfamiliar topic. It raised "this plan cannot
be streamed to a sink (it has a pipeline breaker other than a top-level aggregation)",
which is true of the plan's shape and useless as advice, while the identical pipeline
consumed with `iter_batches` worked.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("v", pa.int64())])


def _stream(n: int = 6):
    def feed():
        for i in range(n):
            yield pa.record_batch({"v": [i]}, schema=_SCHEMA)

    return bt.from_batches(feed, _SCHEMA, bounded=False)


@pytest.mark.integration
def test_the_first_n_rows_reach_a_sink_and_the_query_stops():
    query = _stream().head(3).write.memory("limit_sink", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("limit_sink").to_pydict() == {"v": [0, 1, 2]}


@pytest.mark.integration
def test_the_sink_gets_exactly_what_iter_batches_yields():
    streamed = [row for batch in _stream().head(3).iter_batches() for row in batch.to_pylist()]
    query = _stream().head(3).write.memory("limit_parity", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("limit_parity").to_pylist() == streamed


@pytest.mark.integration
def test_an_offset_is_respected_too():
    """`head` is a `Limit` with offset 0; `slice` is the same node with one set."""
    query = _stream().slice(2, 2).write.memory("limit_offset", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("limit_offset").to_pydict() == {"v": [2, 3]}


@pytest.mark.integration
def test_a_limit_larger_than_the_stream_takes_what_there_is():
    query = _stream(2).head(100).write.memory("limit_short", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("limit_short").to_pydict() == {"v": [0, 1]}


@pytest.mark.integration
def test_a_checkpoint_is_refused_rather_than_accepted_and_ignored():
    """How many rows have already gone out is not a source offset, so a restart would
    resume with the count at zero while the offset log claimed the rows were done."""
    with pytest.raises(PlanError, match="how many rows have gone out"):
        _stream().head(3).write.memory(
            "limit_ckpt", trigger=bt.Trigger.available_now(), checkpoint="/tmp/nope"
        )
