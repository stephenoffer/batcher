"""Score a stream, then enrich or deduplicate it -- the UDF underneath, not on top.

`map_batches` is opaque to the engine by design: the UDF is Python and `MapBatches.to_ir()`
raises. Two streaming drivers reached for `kyber.optimize`, which builds a `PhysicalPlan`
and so lowers the whole subtree, and answered a plan with a UDF beneath them with a bare
`NotImplementedError: map_batches is executed in Python, not lowered to the engine IR`.

Scoring a stream and then joining a dimension to it, or deduplicating the result, are both
obvious pipelines. Both now go through the breaker-free router the stateless streaming path
uses, which runs the UDF in Python and pushes the projection and predicate down exactly as
the private loops did.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])
_ROWS = [[("a", 0, 1), ("b", 5, 2)], [("a", 20, 3)], [("a", 200, 4)]]


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


def _score(batch: pa.RecordBatch) -> dict:
    """A stand-in for a model: whole-Arrow, per batch, and opaque to the engine."""
    return {
        "k": batch.column("k").to_pylist(),
        "ts": batch.column("ts").to_pylist(),
        "v": [value * 2 for value in batch.column("v").to_pylist()],
    }


def _scored():
    return _stream().map_batches(_score, output_columns=["k", "ts", "v"])


def _normalized(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(sorted((k, str(v)) for k, v in row.items())) for row in rows)


def _rows_from(dataset) -> list[dict]:
    return [row for batch in dataset.iter_batches() for row in batch.to_pylist()]


@pytest.mark.integration
def test_a_scored_stream_joins_a_static_dimension():
    dimension = bt.from_pydict({"k": ["a"], "lab": ["A"]})
    rows = _rows_from(_scored().join(dimension, on="k", how="left"))
    assert sorted((r["k"], r["v"], r["lab"]) for r in rows) == [
        ("a", 2, "A"),
        ("a", 6, "A"),
        ("a", 8, "A"),
        ("b", 4, None),
    ]


@pytest.mark.integration
def test_a_scored_stream_deduplicates():
    rows = _rows_from(
        _scored()
        .with_watermark("ts", "1h")
        .drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h")
    )
    assert sorted((r["k"], r["v"]) for r in rows) == [("a", 2), ("b", 4)]


@pytest.mark.integration
@pytest.mark.parametrize("shape", ["join", "dedup"])
def test_the_sink_gets_what_iter_batches_yields(shape):
    """The UDF runs once per micro-batch on either terminal, so the two must agree about
    what a scored-then-enriched stream is."""
    dimension = bt.from_pydict({"k": ["a"], "lab": ["A"]})

    def build():
        if shape == "join":
            return _scored().join(dimension, on="k", how="left")
        return (
            _scored()
            .with_watermark("ts", "1h")
            .drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h")
        )

    streamed = _rows_from(build())
    query = build().write.memory(f"map_under_{shape}", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert _normalized(bt.read_memory(f"map_under_{shape}").to_pylist()) == _normalized(streamed)


@pytest.mark.integration
def test_the_first_n_scored_rows_can_be_taken_and_written():
    """Scoring the first hundred events off an unfamiliar topic is how anyone smoke-tests a
    model against live data. `core.streaming.stream_limit` cannot take a UDF pipeline -- it
    asks Kyber to lower the plan -- so the router answered "the plan must materialize",
    which a limit does not, naming a breaker the caller had not written."""
    rows = _rows_from(_scored().head(2))
    assert [r["v"] for r in rows] == [2, 4]

    query = _scored().head(2).write.memory("map_limit", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    assert bt.read_memory("map_limit").to_pydict()["v"] == [2, 4]


@pytest.mark.integration
def test_an_offset_and_an_over_long_limit_behave_on_a_scored_stream():
    """`slice` is the same node with an offset set, and a limit past the end of a finite
    feed takes what there is rather than waiting for rows that never come."""
    assert [r["v"] for r in _rows_from(_scored().slice(1, 2))] == [4, 6]
    assert [r["v"] for r in _rows_from(_scored().head(100))] == [2, 4, 6, 8]
