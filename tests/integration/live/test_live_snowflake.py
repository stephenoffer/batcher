"""Live smoke test: Snowflake key-pair auth, read, and the staged bulk write.

Skipped unless ``BATCHER_LIVE_SNOWFLAKE_ACCOUNT``, ``BATCHER_LIVE_SNOWFLAKE_USER``,
``BATCHER_LIVE_SNOWFLAKE_KEY_FILE`` and ``BATCHER_LIVE_SNOWFLAKE_WAREHOUSE`` are set
(``BATCHER_LIVE_SNOWFLAKE_DATABASE``/``_SCHEMA`` name where the scratch table goes). Not
run in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import uuid

import pytest

import batcher as bt

_ENV = {
    key: os.environ.get(f"BATCHER_LIVE_SNOWFLAKE_{key.upper()}", "")
    for key in ("account", "user", "key_file", "warehouse", "database", "schema")
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not all(_ENV[k] for k in ("account", "user", "key_file", "warehouse")),
        reason="set the BATCHER_LIVE_SNOWFLAKE_* variables to run against a live Snowflake",
    ),
]


def _auth() -> dict[str, str]:
    opts = {
        "account": _ENV["account"],
        "user": _ENV["user"],
        "auth": "key_pair",
        "private_key_file": _ENV["key_file"],
        "warehouse": _ENV["warehouse"],
    }
    opts.update({k: _ENV[k] for k in ("database", "schema") if _ENV[k]})
    return opts


def test_key_pair_read_and_staged_write():
    assert list(bt.read.snowflake("SELECT 1 AS ONE", **_auth()).to_pydict().values()) == [[1]]
    table = f"BATCHER_LIVE_{uuid.uuid4().hex[:8].upper()}"
    manifest = bt.from_pydict({"ID": [1, 2]}).write.snowflake(table, mode="append", **_auth())
    assert manifest.files[0].job["rows_loaded"] == 2
