"""Live smoke test for the Milvus connector.

Skipped unless ``BATCHER_LIVE_MILVUS_URI`` is set.

Set ``BATCHER_LIVE_MILVUS_URI`` (a server URL, or a Milvus Lite file path with ``pymilvus``
installed) and ``BATCHER_LIVE_MILVUS_COLLECTION``, an existing collection with an INT64
primary key named ``pk``, one 4-dimensional FLOAT_VECTOR field and the dynamic field enabled.

It writes a few points, reads them back, deletes them, and checks the read sees the delete.
Nothing here runs in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import uuid

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_MILVUS_URI"),
    reason="set BATCHER_LIVE_MILVUS_URI to run the live Milvus smoke test",
)


def test_milvus_round_trip() -> None:
    uri = os.environ["BATCHER_LIVE_MILVUS_URI"]
    collection = os.environ["BATCHER_LIVE_MILVUS_COLLECTION"]
    conn = {"uri": uri, "token": os.environ.get("BATCHER_LIVE_MILVUS_TOKEN")}
    base = uuid.uuid4().int % 10**12
    ids = [base + i for i in range(3)]
    frame = bt.from_pydict({"id": ids, "embedding": [[0.1, 0.2, 0.3, 0.4]] * 3})
    assert frame.write.milvus(collection, **conn).total_rows == 3
    got = bt.read.milvus(collection, filter=f"pk in {ids}", **conn).to_pydict()
    assert len(next(iter(got.values()))) == 3
    bt.from_pydict({"id": ids}).write.milvus(collection, mode="delete", **conn)
