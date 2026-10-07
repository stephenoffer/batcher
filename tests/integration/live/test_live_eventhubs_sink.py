"""Live smoke test: send a short stream to a real Event Hub.

Skipped unless ``BATCHER_LIVE_EVENTHUBS_CONN`` is set. See tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_EVENTHUBS_CONN"),
    reason="set BATCHER_LIVE_EVENTHUBS_CONN to run against a live service",
)


def test_write_eventhubs_sends():
    hub = os.environ.get("BATCHER_LIVE_EVENTHUBS_NAME", "batcher-smoke")
    rows = bt.read.rate_micro_batch(10, num_rows=20).select(
        value=bt.col("value").cast("string"), key=bt.col("value").cast("string")
    )
    query = rows.write.eventhubs(
        hub, connection_str="env:BATCHER_LIVE_EVENTHUBS_CONN", trigger=bt.Trigger.available_now()
    )
    assert query.await_termination()
