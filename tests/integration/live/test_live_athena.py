"""Live smoke test: the Athena profile over PyAthena.

Skipped unless ``BATCHER_LIVE_ATHENA_REGION`` and ``BATCHER_LIVE_ATHENA_WORKGROUP`` are set
(``BATCHER_LIVE_ATHENA_OUTPUT`` adds an ``s3://`` result location). Credentials come from
the ambient AWS chain. Not run in CI; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

REGION = os.environ.get("BATCHER_LIVE_ATHENA_REGION", "")
WORKGROUP = os.environ.get("BATCHER_LIVE_ATHENA_WORKGROUP", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (REGION and WORKGROUP),
        reason="set BATCHER_LIVE_ATHENA_REGION and BATCHER_LIVE_ATHENA_WORKGROUP",
    ),
]


def test_query_round_trips():
    ds = bt.read.athena(
        "SELECT 1 AS one",
        region=REGION,
        workgroup=WORKGROUP,
        output_location=os.environ.get("BATCHER_LIVE_ATHENA_OUTPUT"),
    )
    assert ds.to_pydict() == {"one": [1]}
