"""Live smoke test for the Qdrant connector.

Skipped unless ``BATCHER_LIVE_QDRANT_URL`` is set.

Set ``BATCHER_LIVE_QDRANT_URL`` (and ``BATCHER_LIVE_QDRANT_API_KEY`` for a cloud cluster) and
``BATCHER_LIVE_QDRANT_COLLECTION``, an existing collection of 4-dimensional cosine vectors.

It writes a few points, reads them back, deletes them, and checks the read sees the delete.
Nothing here runs in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import uuid

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_QDRANT_URL"),
    reason="set BATCHER_LIVE_QDRANT_URL to run the live Qdrant smoke test",
)


def test_qdrant_round_trip() -> None:
    url = os.environ["BATCHER_LIVE_QDRANT_URL"]
    collection = os.environ["BATCHER_LIVE_QDRANT_COLLECTION"]
    conn = {"url": url, "api_key": os.environ.get("BATCHER_LIVE_QDRANT_API_KEY")}
    tag = uuid.uuid4().hex
    ids = [f"{tag}-{i}" for i in range(3)]
    frame = bt.from_pydict({"id": ids, "embedding": [[0.1, 0.2, 0.3, 0.4]] * 3, "tag": [tag] * 3})
    assert frame.write.qdrant(collection, metric="cosine", **conn).total_rows == 3
    got = bt.read.qdrant(collection, **conn).filter(bt.col("tag") == tag).to_pydict()
    assert sorted(got["id"]) == sorted(ids)
    bt.from_pydict({"id": ids}).write.qdrant(collection, mode="delete", **conn)
    assert bt.read.qdrant(collection, **conn).filter(bt.col("tag") == tag).count() == 0
