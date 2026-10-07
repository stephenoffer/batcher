"""Live round trip for Google Sheets, skipped unless a scratch spreadsheet is provided.

Run: ``BATCHER_LIVE_GSHEETS_SPREADSHEET_ID=<id of a scratch sheet>
BATCHER_LIVE_GSHEETS_TOKEN=<access token, or omit to use Application Default Credentials>
pytest tests/integration/live/test_live_google_sheets.py``. The test overwrites ``Sheet1!A1:C``.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_GSHEETS_SPREADSHEET_ID"),
    reason="BATCHER_LIVE_GSHEETS_SPREADSHEET_ID not set",
)


def test_live_google_sheets_round_trip():
    sheet = os.environ["BATCHER_LIVE_GSHEETS_SPREADSHEET_ID"]
    auth = (
        bt.io.BearerToken("env:BATCHER_LIVE_GSHEETS_TOKEN")
        if os.environ.get("BATCHER_LIVE_GSHEETS_TOKEN")
        else None
    )
    data = {"name": ["a", "b", "c"], "n": [1, 2, 3], "ok": [True, False, True]}
    bt.from_pydict(data).write.google_sheets(
        sheet, "Sheet1!A1:C", auth=auth, batch_rows=2, distributed=False
    )
    back = bt.read.google_sheets(sheet, "Sheet1!A1:C", auth=auth).to_pydict()
    assert back == data
