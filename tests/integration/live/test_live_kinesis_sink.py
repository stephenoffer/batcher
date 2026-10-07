"""Live smoke test: PutRecords a short stream into a real Kinesis stream.

Skipped unless ``BATCHER_LIVE_KINESIS_STREAM`` is set. See tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_KINESIS_STREAM"),
    reason="set BATCHER_LIVE_KINESIS_STREAM to run against a live service",
)


def test_write_kinesis_puts_records():
    stream = os.environ["BATCHER_LIVE_KINESIS_STREAM"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    rows = bt.read.rate_micro_batch(10, num_rows=20).select(
        value=bt.col("value").cast("string"), key=bt.col("value").cast("string")
    )
    query = rows.write.kinesis(stream, region=region, trigger=bt.Trigger.available_now())
    assert query.await_termination()
