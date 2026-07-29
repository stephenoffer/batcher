"""Batch as the bounded case of streaming: the same operators, incrementally.

Batcher runs batch and streaming on one operator set, so the transformation you tested on
a file is the one that runs on the stream. This example uses the bounded `rate` source so
it terminates, but the pipeline shape is the same for Kafka.

    python examples/operations/streaming_basics.py
"""

from __future__ import annotations

from datetime import datetime

import batcher as bt
from batcher import col


def main() -> None:
    # A bounded stand-in for an unbounded source, so the script finishes.
    events = bt.from_pydict(
        {
            "ts": list(range(20)),
            "user": [f"u{i % 4}" for i in range(20)],
            "amount": [float(i) for i in range(20)],
        }
    )

    # The transformation. Nothing here is streaming-specific.
    pipeline = (
        events.filter(col("amount") > 2)
        .with_columns(bucket=col("ts").floordiv(5))
        .group_by("bucket")
        .agg(total=col("amount").sum(), n=bt.count())
        .sort("bucket")
    )

    batch_result = pipeline.to_pydict()
    print("batch:", batch_result)
    assert sum(batch_result["n"]) == 17  # 20 rows minus the three filtered out

    # The same plan, consumed incrementally. Memory is bounded by the batch size rather
    # than the table size, and the answer is identical.
    seen = 0
    for chunk in pipeline.iter_batches(batch_size=2):
        seen += chunk.num_rows
    assert seen == len(batch_result["bucket"])

    # A windowed aggregate over event time, which is the streaming shape.
    # Use `floordiv` for the bucket, not `/` plus a cast: casting a float to an integer
    # *rounds to nearest*, so `ts=5` would land in window 1 rather than window 0.
    windowed = (
        events.group_by(window=col("ts").floordiv(10))
        .agg(total=col("amount").sum())
        .sort("window")
        .to_pydict()
    )
    print("windows:", windowed)
    assert windowed["window"] == [0, 1]
    assert windowed["total"] == [sum(range(10)), sum(range(10, 20))]

    # The streaming-specific verbs work on a real timestamp column, and take their
    # durations as strings ("5s", "10m"), not numbers.
    timed = bt.from_pydict(
        {
            "ts": [datetime(2024, 1, 1, 0, 0, s) for s in (0, 1, 2, 30, 31)],
            "user": ["u1", "u1", "u2", "u1", "u2"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # Watermarks bound how long the engine waits for late data. Declaring one is what
    # lets a windowed aggregate emit and release state instead of growing forever.
    marked = timed.with_watermark("ts", "10s")
    assert marked.count() == 5

    # Session windows group activity separated by a gap, per key.
    sessions = timed.session_window("ts", "5s", partition_by=["user"], total=col("amount").sum())
    session_rows = sessions.to_pydict()
    print("sessions:", session_rows)
    # u1 has a burst at 0-1s and another at 30s, so it yields two sessions.
    assert len(session_rows[next(iter(session_rows))]) >= 3

    # Deduplication inside the watermark, for an at-least-once source that repeats rows.
    deduped = timed.drop_duplicates_within_watermark(["user"], event_time="ts", lateness="10s")
    assert deduped.count() <= 5

    # The guarantee this rests on: the batch and streaming paths compute the same thing.
    streamed_total = sum(
        b.column("total").to_pylist()[0] for b in pipeline.iter_batches(batch_size=1)
    )
    assert abs(streamed_total - sum(batch_result["total"])) < 1e-9


if __name__ == "__main__":
    main()
