"""Live smoke test for `bt.read.graphql` against GitHub's GraphQL API.

Run: ``BATCHER_LIVE_GITHUB_TOKEN=<token> pytest tests/integration/live/test_live_graphql.py``.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_GITHUB_TOKEN"), reason="BATCHER_LIVE_GITHUB_TOKEN not set"
)

QUERY = (
    'query($after: String) { repository(owner: "apache", name: "arrow") '
    "{ issues(first: 25, after: $after) { nodes { number title } "
    "pageInfo { endCursor hasNextPage } } } }"
)


def test_live_graphql_pages_a_relay_connection():
    ds = bt.read.graphql(
        "https://api.github.com/graphql",
        QUERY,
        records_path="repository.issues.nodes",
        page_info_path="repository.issues.pageInfo",
        auth=bt.io.BearerToken("env:BATCHER_LIVE_GITHUB_TOKEN"),
        max_pages=2,
    )
    assert len(ds.to_pydict()["number"]) == 50
