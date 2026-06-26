# examples: skip  # noqa: ERA001  (harness marker: needs a Kafka broker + Delta sink)
"""Streaming micro-batch pipeline: Kafka in, windowed aggregate, Delta out.

The same ``Dataset`` API serves batch and streaming — you change the *source* (and
add a ``trigger``), not the transformation. This script reads an unbounded Kafka
topic, declares a watermark so windowed state stays bounded, aggregates click
counts into event-time hourly windows, and writes each micro-batch to a Delta sink
with exactly-once checkpointing.

It is marked ``# examples: skip`` because a Kafka broker and a Delta sink are
external infrastructure; the harness collects it for API-shape coverage but does not
execute it. Run it against a real cluster with::

    pip install 'batcher-engine[kafka,delta]'
    python examples/streaming_pipeline.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    # Unbounded source: a Kafka topic of click events.
    clicks = bt.read.kafka(topic="clicks", bootstrap_servers="localhost:9092")

    # The transformation is ordinary Dataset code — identical to a batch query.
    # with_watermark bounds the windowed state: a window is emitted and evicted once
    # the watermark (max event time minus the lateness bound) passes its end, and
    # rows later than that are dropped.
    hourly = (
        clicks.with_watermark("ts", "10 minutes")
        .group_by(window=bt.window(col("ts"), "1h"))
        .agg(hits=col("n").sum())
    )

    # A trigger turns the write into a streaming query: each micro-batch is appended
    # to the Delta sink, and checkpoint= records source offsets and sink commits so a
    # restart resumes exactly where it left off (exactly-once).
    query = hourly.write.delta(
        "lake/gold/hourly_clicks",
        trigger=bt.Trigger.processing_time("1 minute"),
        output_mode="append",
        checkpoint="lake/gold/hourly_clicks/_checkpoint",
    )

    # The handle drives the running query.
    try:
        query.await_termination()
    finally:
        query.stop()


if __name__ == "__main__":
    main()
