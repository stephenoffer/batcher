"""Live smoke test: publish a short stream to a real Pub/Sub topic.

Skipped unless ``BATCHER_LIVE_PUBSUB_TOPIC`` is set. See tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_PUBSUB_TOPIC"),
    reason="set BATCHER_LIVE_PUBSUB_TOPIC to run against a live service",
)


def test_write_pubsub_publishes():
    topic = os.environ["BATCHER_LIVE_PUBSUB_TOPIC"]  # projects/<p>/topics/<t>
    rows = bt.read.rate_micro_batch(10, num_rows=20).select(value=bt.col("value").cast("string"))
    query = rows.write.pubsub(topic, dedup_ids="smoke", trigger=bt.Trigger.available_now())
    assert query.await_termination()
