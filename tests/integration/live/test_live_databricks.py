"""Live smoke test: Databricks warehouse session options, query ids, and the volume write.

Skipped unless ``BATCHER_LIVE_DATABRICKS_HOST``, ``BATCHER_LIVE_DATABRICKS_HTTP_PATH`` and
``BATCHER_LIVE_DATABRICKS_TOKEN`` (a value or an ``env:`` reference) are set. The write also
needs ``BATCHER_LIVE_DATABRICKS_VOLUME`` (a ``/Volumes/...`` directory) and
``BATCHER_LIVE_DATABRICKS_TABLE`` (an existing ``(id BIGINT)`` table). Not run in CI; see
tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

HOST = os.environ.get("BATCHER_LIVE_DATABRICKS_HOST", "")
HTTP_PATH = os.environ.get("BATCHER_LIVE_DATABRICKS_HTTP_PATH", "")
TOKEN = os.environ.get("BATCHER_LIVE_DATABRICKS_TOKEN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (HOST and HTTP_PATH and TOKEN),
        reason="set the BATCHER_LIVE_DATABRICKS_* variables to run against a live workspace",
    ),
]


def _warehouse() -> dict[str, object]:
    return {"server_hostname": HOST, "http_path": HTTP_PATH, "access_token": TOKEN}


def test_warehouse_read_with_session_options():
    ds = bt.read.databricks(query="SELECT 1 AS one", statement_timeout_s=120, **_warehouse())
    assert ds.to_pydict() == {"one": [1]}


def test_volume_write_returns_the_copy_query_id():
    volume = os.environ.get("BATCHER_LIVE_DATABRICKS_VOLUME")
    table = os.environ.get("BATCHER_LIVE_DATABRICKS_TABLE")
    if not (volume and table):
        pytest.skip("set BATCHER_LIVE_DATABRICKS_VOLUME and BATCHER_LIVE_DATABRICKS_TABLE")
    manifest = bt.from_pydict({"id": [1, 2]}).write.databricks(
        table, volume_path=volume, **_warehouse()
    )
    assert manifest.files[0].job["query_id"]
    assert manifest.total_rows == 2
