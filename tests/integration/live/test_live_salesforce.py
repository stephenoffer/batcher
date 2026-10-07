"""Live smoke test for `bt.read.salesforce` (Bulk API 2.0), skipped without credentials.

Run: ``BATCHER_LIVE_SALESFORCE_INSTANCE_URL=https://<domain>.my.salesforce.com
BATCHER_LIVE_SALESFORCE_TOKEN=<access token>
pytest tests/integration/live/test_live_salesforce.py``.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("BATCHER_LIVE_SALESFORCE_INSTANCE_URL")
        and os.environ.get("BATCHER_LIVE_SALESFORCE_TOKEN")
    ),
    reason="BATCHER_LIVE_SALESFORCE_INSTANCE_URL / BATCHER_LIVE_SALESFORCE_TOKEN not set",
)


def test_live_salesforce_query_all_and_resume(tmp_path):
    schema = pa.schema(
        [
            ("Id", pa.string()),
            ("Name", pa.string()),
            ("SystemModstamp", pa.timestamp("ms", tz="UTC")),
        ]
    )
    inc = bt.io.Incremental(state=str(tmp_path / "sf.json"))
    ds = bt.read.salesforce(
        "Account",
        instance_url=os.environ["BATCHER_LIVE_SALESFORCE_INSTANCE_URL"],
        schema=schema,
        auth=bt.io.BearerToken("env:BATCHER_LIVE_SALESFORCE_TOKEN"),
        include_deleted=True,
        incremental=inc,
    )
    rows = ds.to_pydict()
    assert "IsDeleted" in rows
    assert inc.load() is not None or not rows["Id"]
