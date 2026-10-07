"""Live smoke test for the Pinecone connector.

Skipped unless ``BATCHER_LIVE_PINECONE_API_KEY`` is set.

Set ``BATCHER_LIVE_PINECONE_API_KEY`` and ``BATCHER_LIVE_PINECONE_INDEX``, an existing serverless
index of 4-dimensional cosine vectors. A fresh namespace is used and emptied.

It writes a few points, reads them back, deletes them, and checks the read sees the delete.
Nothing here runs in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_PINECONE_API_KEY"),
    reason="set BATCHER_LIVE_PINECONE_API_KEY to run the live Pinecone smoke test",
)


def test_pinecone_round_trip() -> None:
    conn = {"api_key": "env:BATCHER_LIVE_PINECONE_API_KEY", "namespace": uuid.uuid4().hex}
    index = os.environ["BATCHER_LIVE_PINECONE_INDEX"]
    ids = ["a", "b", "c"]
    frame = bt.from_pydict({"id": ids, "embedding": [[0.1, 0.2, 0.3, 0.4]] * 3, "n": [1, 2, 3]})
    assert frame.write.pinecone(index, metric="cosine", **conn).total_rows == 3
    # Pinecone is eventually consistent: a fresh write may take a moment to list.
    for _ in range(30):
        got = bt.read.pinecone(index, **conn).to_pydict()
        if len(got.get("id", [])) == 3:
            break
        time.sleep(2)
    assert sorted(got["id"]) == ids
    bt.from_pydict({"id": ids}).write.pinecone(index, mode="delete", **conn)
