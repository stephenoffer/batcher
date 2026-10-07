"""Live smoke test: one `ProviderLimit` held across the workers of a real Ray cluster.

Skipped unless ``BATCHER_LIVE_RAY_ADDRESS`` names a multi-node cluster whose workers import
this tree's Batcher. It runs the same check as `tests/integration/test_shared_provider_limit.py`
against that cluster, where the workers sit on other nodes than the driver.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_RAY_ADDRESS"),
    reason="set BATCHER_LIVE_RAY_ADDRESS to a multi-node Ray cluster to run",
)


def test_workers_on_several_nodes_obey_one_quota(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", os.environ["BATCHER_LIVE_RAY_ADDRESS"])
    from tests.integration.test_shared_provider_limit import (
        test_two_workers_obey_one_concurrency_quota,
    )

    test_two_workers_obey_one_concurrency_quota()
