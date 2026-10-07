"""Live smoke test: Oracle through the generic URI routes.

Skipped unless ``BATCHER_LIVE_ORACLE_URI`` holds a connection URI for a reachable
server; ``BATCHER_LIVE_PASSWORD`` supplies the password, as a literal or an ``env:``
reference. Not run in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import uuid

import pytest

import batcher as bt

URI = os.environ.get("BATCHER_LIVE_ORACLE_URI", "")
PASSWORD = os.environ.get("BATCHER_LIVE_PASSWORD")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URI, reason="set BATCHER_LIVE_ORACLE_URI to run against a live Oracle"),
]


def test_query_round_trips():
    out = bt.read.sql("SELECT 1 AS one FROM dual", uri=URI, password=PASSWORD)
    # Values only: Oracle folds the alias to ONE.
    assert list(out.to_pydict().values()) == [[1]]


def test_upsert_replaces_by_key():
    table = f"batcher_live_{uuid.uuid4().hex[:8]}"
    first = bt.from_pydict({"id": [1, 2], "v": ["a", "b"]})
    # The first upsert creates the table with `id` as its primary key.
    first.write.sql(table, uri=URI, password=PASSWORD, mode="upsert", key_columns="id")
    bt.from_pydict({"id": [2, 3], "v": ["B", "c"]}).write.sql(
        table, uri=URI, password=PASSWORD, mode="upsert", key_columns="id"
    )
    out = bt.read.sql(f"SELECT id, v FROM {table}", uri=URI, password=PASSWORD).to_pydict()
    assert sorted(zip(*out.values(), strict=True)) == [(1, "a"), (2, "B"), (3, "c")]
