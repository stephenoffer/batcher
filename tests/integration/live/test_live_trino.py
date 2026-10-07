"""Live smoke test: Trino through the generic URI routes.

Skipped unless ``BATCHER_LIVE_TRINO_URI`` holds a connection URI for a reachable
server; ``BATCHER_LIVE_PASSWORD`` supplies the password, as a literal or an ``env:``
reference. Not run in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

URI = os.environ.get("BATCHER_LIVE_TRINO_URI", "")
PASSWORD = os.environ.get("BATCHER_LIVE_PASSWORD")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URI, reason="set BATCHER_LIVE_TRINO_URI to run against a live Trino"),
]


def test_query_round_trips():
    out = bt.read.sql("SELECT 1 AS one", uri=URI, password=PASSWORD)
    # Values only: Oracle folds the alias to ONE.
    assert list(out.to_pydict().values()) == [[1]]


def test_append_lands_rows():
    table = os.environ.get("BATCHER_LIVE_TRINO_TABLE")
    if not table:
        pytest.skip("set BATCHER_LIVE_TRINO_TABLE to a writable (id bigint) table")
    before = bt.read.sql(f"SELECT count(*) AS n FROM {table}", uri=URI, password=PASSWORD)
    bt.from_pydict({"id": [1]}).write.sql(table, uri=URI, password=PASSWORD, mode="append")
    after = bt.read.sql(f"SELECT count(*) AS n FROM {table}", uri=URI, password=PASSWORD)
    assert after.to_pydict()["n"][0] == before.to_pydict()["n"][0] + 1
