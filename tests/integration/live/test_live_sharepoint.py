"""Live smoke test for `bt.read.sharepoint` (Microsoft Graph delta), skipped without credentials.

Run: ``BATCHER_LIVE_GRAPH_TENANT=<tenant id> BATCHER_LIVE_GRAPH_CLIENT_ID=<app id>
BATCHER_LIVE_GRAPH_SECRET=<client secret> BATCHER_LIVE_GRAPH_DRIVE_ID=<drive id>
pytest tests/integration/live/test_live_sharepoint.py``.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

_VARS = (
    "BATCHER_LIVE_GRAPH_TENANT",
    "BATCHER_LIVE_GRAPH_CLIENT_ID",
    "BATCHER_LIVE_GRAPH_SECRET",
    "BATCHER_LIVE_GRAPH_DRIVE_ID",
)

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(v) for v in _VARS), reason="Microsoft Graph credentials not set"
)


def test_live_sharepoint_delta_then_incremental(tmp_path):
    auth = bt.io.OAuth2ClientCredentials(
        token_url=(
            f"https://login.microsoftonline.com/{os.environ['BATCHER_LIVE_GRAPH_TENANT']}"
            "/oauth2/v2.0/token"
        ),
        client_id=os.environ["BATCHER_LIVE_GRAPH_CLIENT_ID"],
        client_secret="env:BATCHER_LIVE_GRAPH_SECRET",
        scope="https://graph.microsoft.com/.default",
    )
    state = str(tmp_path / "delta.json")
    drive = os.environ["BATCHER_LIVE_GRAPH_DRIVE_ID"]
    first = bt.read.sharepoint(drive_id=drive, auth=auth, state=state).to_pydict()
    assert len(first["id"]) > 0
    assert bt.io.Incremental(state=state).load()["cursor"]
    second = bt.read.sharepoint(drive_id=drive, auth=auth, state=state).to_pydict()
    assert len(second["id"]) <= len(first["id"])
