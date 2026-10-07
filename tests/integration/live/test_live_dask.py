"""Live smoke test: every to_dask policy against the real dask.dataframe.

Skipped unless ``BATCHER_LIVE_DASK`` is set. See tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_DASK"),
    reason="set BATCHER_LIVE_DASK to run against a live service",
)


@pytest.mark.parametrize("materialize", ["arrow", "deferred", "parquet"])
def test_to_dask_matches_the_engine(materialize, tmp_path):
    pytest.importorskip("dask.dataframe")
    ds = bt.from_pydict({"i": list(range(1000)), "s": [f"r{i % 7}" for i in range(1000)]})
    frame = ds.to_dask(materialize=materialize, npartitions=4, staging_path=str(tmp_path))
    assert int(frame["i"].sum().compute()) == sum(range(1000))
    assert len(frame) == 1000
