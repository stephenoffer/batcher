"""A filter on top of a streaming operator must not decide whether it can reach a sink.

`impressions.join_stream(clicks, ...).filter(col("clicked").is_null())` -- the cookbook's
own recipe for "impressions with no click" -- could be printed and not written. So could a
`select` after a stream-static join, and a filter after a session window or a dedup. The
router peels row-wise operators off a breaker and re-applies them per batch, which is why
`iter_batches` was fine; the sink launcher tested only the top node, so a single `filter`
on top made every one of these plans unrecognizable.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])
_ROWS = [
    [("a", 0, 1), ("a", 2, 2)],
    [("a", 240, 3), ("b", 245, 4)],
]


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


def _dimension():
    return bt.from_pydict({"k": ["a"], "lab": ["A"]})


_SHAPES = {
    "filter_after_session": lambda: (
        _stream()
        .session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
        .filter(col("total") > 1)
    ),
    "select_after_session": lambda: (
        _stream()
        .session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
        .select("k", "total")
    ),
    "select_after_static_join": lambda: (
        _stream().join(_dimension(), on="k", how="inner").select("k", "v")
    ),
    "filter_after_dedup": lambda: (
        _stream()
        .with_watermark("ts", "1h")
        .drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h")
        .filter(col("v") > 0)
    ),
    "filter_after_interval_join": lambda: (
        _stream()
        .join_stream(_stream(), on="k", left_time="ts", right_time="ts", within="30m")
        .filter(col("v") > 0)
    ),
    "filter_after_union": lambda: _stream().union(_stream()).filter(col("v") > 1),
}


def _rows_from(dataset) -> list[dict]:
    return [row for batch in dataset.iter_batches() for row in batch.to_pylist()]


def _normalized(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(sorted((k, str(v)) for k, v in row.items())) for row in rows)


@pytest.mark.integration
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_a_row_wise_operator_on_top_still_reaches_a_sink(shape):
    query = _SHAPES[shape]().write.memory(f"peeled_{shape}", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    written = bt.read_memory(f"peeled_{shape}").to_pylist()
    assert _normalized(written) == _normalized(_rows_from(_SHAPES[shape]()))


@pytest.mark.integration
def test_two_stacked_operators_are_peeled_together():
    """`select` over `filter` over a session window -- nothing about the depth matters."""
    plan = (
        _stream()
        .session_window("ts", "30m", partition_by=["k"], total=col("v").sum())
        .filter(col("total") > 3)
        .select("k")
    )
    query = plan.write.memory("peeled_two", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    # Only `b`'s single session totals more than 3; `a`'s two total 3 each.
    assert bt.read_memory("peeled_two").to_pydict()["k"] == ["b"]
