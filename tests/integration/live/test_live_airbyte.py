"""Live smoke test for `bt.read.airbyte` with the public source-faker image.

Run: ``BATCHER_LIVE_AIRBYTE_IMAGE=airbyte/source-faker:6 pytest
tests/integration/live/test_live_airbyte.py`` on a host with Docker.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_AIRBYTE_IMAGE"), reason="BATCHER_LIVE_AIRBYTE_IMAGE not set"
)


def test_live_airbyte_faker_users_with_state(tmp_path):
    state = str(tmp_path / "ab.json")
    ds = bt.read.airbyte(
        "users",
        image=os.environ["BATCHER_LIVE_AIRBYTE_IMAGE"],
        config={"count": 50, "seed": 1},
        state=state,
    )
    assert len(ds.to_pydict()["id"]) == 50
    assert bt.io.Incremental(state=state).load() is not None
