"""Live smoke test for `bt.read.github`, skipped unless a token is provided.

Run: ``BATCHER_LIVE_GITHUB_TOKEN=<token> pytest tests/integration/live/test_live_github.py``.
Optionally ``BATCHER_LIVE_GITHUB_REPO=owner/name`` (default ``apache/arrow``).
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_GITHUB_TOKEN"), reason="BATCHER_LIVE_GITHUB_TOKEN not set"
)


@pytest.mark.parametrize("resource", ["issues", "pulls", "releases"])
def test_live_github_reads_two_pages(resource):
    repo = os.environ.get("BATCHER_LIVE_GITHUB_REPO", "apache/arrow")
    ds = bt.read.github(repo, resource, token="env:BATCHER_LIVE_GITHUB_TOKEN", max_pages=2)
    rows = ds.to_pydict()
    assert len(rows["id"]) > 0
    assert len(set(rows["id"])) == len(rows["id"])
