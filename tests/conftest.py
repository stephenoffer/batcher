"""Suite-wide fixtures.

The one cross-cutting concern every test shares: the process-global `MetadataHub`.
It accumulates learned statistics (cardinalities, selectivities, GPU utilization)
across executions so plans improve with use — but in a test process that makes
outcomes *order-dependent*: a test asserting on cardinality- or cost-driven plan
shape (join build-side choice, adaptive cardinalities, approximate quantiles) can be
perturbed by stats an earlier test recorded. Resetting the hub before each test makes
the suite deterministic regardless of collection order, without changing production
behavior (the reset only drops the cached in-process handle).

Learning *within* a single test (multiple `collect()`s in one function) is preserved
— the reset happens only between tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_metadata_hub():
    """Reset the process-wide MetadataHub around every test for deterministic order.

    No-op (yields cleanly) in a pure-Python environment where Core can't be imported,
    so tests that don't touch the engine still run.
    """
    try:
        from batcher.core import reset_default_hub
    except Exception:
        yield
        return
    reset_default_hub()
    yield
    reset_default_hub()
