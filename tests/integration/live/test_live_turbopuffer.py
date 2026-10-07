"""Live smoke test for the Turbopuffer connector.

Skipped unless ``BATCHER_LIVE_TURBOPUFFER_API_KEY`` is set.

Set ``BATCHER_LIVE_TURBOPUFFER_API_KEY`` and ``BATCHER_LIVE_TURBOPUFFER_REGION`` (such as
``gcp-us-central1``). A fresh namespace is created by the write.

It writes a few points, reads them back, deletes them, and checks the read sees the delete.
Nothing here runs in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import uuid

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_TURBOPUFFER_API_KEY"),
    reason="set BATCHER_LIVE_TURBOPUFFER_API_KEY to run the live Turbopuffer smoke test",
)


def test_turbopuffer_round_trip() -> None:
    conn = {
        "api_key": "env:BATCHER_LIVE_TURBOPUFFER_API_KEY",
        "region": os.environ["BATCHER_LIVE_TURBOPUFFER_REGION"],
    }
    namespace = f"batcher-smoke-{uuid.uuid4().hex[:12]}"
    frame = bt.from_pydict(
        {"id": [1, 2, 3], "embedding": [[0.1, 0.2, 0.3, 0.4]] * 3, "n": [1, 2, 3]}
    )
    assert frame.write.turbopuffer(namespace, **conn).total_rows == 3
    got = bt.read.turbopuffer(namespace, **conn).sort("id").to_pydict()
    assert got["id"] == [1, 2, 3]
    assert got["embedding"][0] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    bt.from_pydict({"id": [2]}).write.turbopuffer(namespace, mode="delete", **conn)
    assert bt.read.turbopuffer(namespace, **conn).sort("id").to_pydict()["id"] == [1, 3]
