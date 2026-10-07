"""Live smoke test: BigQuery read and the Parquet load-job write.

Skipped unless ``BATCHER_LIVE_BIGQUERY_PROJECT`` names a project and
``BATCHER_LIVE_BIGQUERY_DATASET`` a dataset the ambient ``google.auth`` credentials can
write. Not run in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os
import uuid

import pytest

import batcher as bt

PROJECT = os.environ.get("BATCHER_LIVE_BIGQUERY_PROJECT", "")
DATASET = os.environ.get("BATCHER_LIVE_BIGQUERY_DATASET", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (PROJECT and DATASET),
        reason="set BATCHER_LIVE_BIGQUERY_PROJECT and BATCHER_LIVE_BIGQUERY_DATASET",
    ),
]


def test_write_returns_a_job_and_keeps_repeated_fields():
    table = f"{PROJECT}.{DATASET}.batcher_live_{uuid.uuid4().hex[:8]}"
    ds = bt.from_pydict({"id": [1, 2], "tags": [["a"], ["b", "c"]]})
    manifest = ds.write.bigquery(table, project=PROJECT)
    assert manifest.files[0].job["job_id"]
    out = bt.read.bigquery(table=table, project=PROJECT).to_pydict()
    assert sorted(zip(out["id"], out["tags"], strict=True)) == [(1, ["a"]), (2, ["b", "c"])]
