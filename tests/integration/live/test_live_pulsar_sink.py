"""Live smoke test: publish a short stream to a real Pulsar topic and read it back.

Skipped unless ``BATCHER_LIVE_PULSAR_URL`` is set. See tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_PULSAR_URL"),
    reason="set BATCHER_LIVE_PULSAR_URL to run against a live service",
)


def test_write_pulsar_round_trips_through_a_real_broker():
    url = os.environ["BATCHER_LIVE_PULSAR_URL"]
    topic = os.environ.get("BATCHER_LIVE_PULSAR_TOPIC", "persistent://public/default/batcher-smoke")
    rows = bt.read.rate_micro_batch(10, num_rows=20).select(value=bt.col("value").cast("string"))
    query = rows.write.pulsar(
        topic, service_url=url, producer_name="batcher-smoke", trigger=bt.Trigger.available_now()
    )
    assert query.await_termination()
    back = bt.read.pulsar(topic, service_url=url, subscription="batcher-smoke-check")
    seen = next(back.iter_batches())
    assert seen.num_rows > 0
