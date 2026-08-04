"""A deduplicated stream has to be able to reach a sink.

`drop_duplicates_within_watermark` streamed fine through `iter_batches` and was refused by
every streaming write with "this plan cannot be streamed to a sink". That is the one thing
anyone deduplicates a stream *for*: the dedup is a stage in a pipeline that lands
somewhere, not a thing you print. The two terminals disagreeing about which plans exist is
also the shape of gap nothing catches, because each is tested on its own.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])
_ROWS = [[("a", 0, 1), ("b", 5, 2)], [("a", 20, 3)], [("a", 200, 4), ("b", 205, 5)]]


def _stream():
    def feed():
        for rows in _ROWS:
            yield pa.record_batch(
                {
                    "k": [k for k, _, _ in rows],
                    "ts": [_BASE + dt.timedelta(minutes=m) for _, m, _ in rows],
                    "v": [v for _, _, v in rows],
                },
                schema=_SCHEMA,
            )

    return bt.from_batches(feed, _SCHEMA, bounded=False)


def _deduped(dataset):
    return dataset.with_watermark("ts", "1h").drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="1h"
    )


def _normalized(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(sorted((k, str(v)) for k, v in row.items())) for row in rows)


@pytest.mark.integration
def test_a_deduplicated_stream_writes_to_a_sink():
    query = _deduped(_stream()).write.memory("dedup_to_sink", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert sorted(bt.read_memory("dedup_to_sink").to_pydict()["k"]) == ["a", "b"]


@pytest.mark.integration
def test_the_sink_gets_exactly_what_iter_batches_yields():
    """The two terminals are two consumers of one operator, so any disagreement between
    them is a defect in whichever is wrong -- and until this route existed, one of them
    simply refused to run."""
    streamed: list[dict] = []
    for batch in _deduped(_stream()).iter_batches():
        streamed.extend(batch.to_pylist())

    query = _deduped(_stream()).write.memory("dedup_parity", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert _normalized(streamed) == _normalized(bt.read_memory("dedup_parity").to_pylist())


@pytest.mark.integration
def test_a_checkpoint_is_refused_rather_than_accepted_and_ignored():
    """The seen-key set is not a source offset, so a restart would resume with an empty
    one while the offset log said otherwise -- which looks exactly like exactly-once
    recovery and is not."""
    with pytest.raises(PlanError, match="seen-key set"):
        _deduped(_stream()).write.memory(
            "dedup_ckpt", trigger=bt.Trigger.available_now(), checkpoint="/tmp/nope"
        )


@pytest.mark.integration
def test_a_non_append_output_mode_is_refused():
    """A dedup emits each surviving row once; there is no result table to restate."""
    with pytest.raises(PlanError, match="needs an aggregation"):
        _deduped(_stream()).write.memory(
            "dedup_mode", trigger=bt.Trigger.available_now(), output_mode="complete"
        )
