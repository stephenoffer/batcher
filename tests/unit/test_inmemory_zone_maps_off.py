"""`zone_maps=False` means "report no bounds" on every statistics path, not just `statistics()`.

An engine-produced relation (a materialized shared subplan, an adaptive stage boundary) is
built with zone maps off because its bounds would be rebuilt and discarded every run. The
narrowed per-column form the conductor actually calls ignored the flag, so each intermediate
paid the O(rows) pass anyway, and a float `SUM` whose last bit moves with parallel summation
order reached the plan-cache key: TPC-DS q80's final stage re-planned on alternate runs.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io import InMemorySource

pytestmark = pytest.mark.unit


def _batches() -> list[pa.RecordBatch]:
    return [pa.record_batch({"x": [1.5, 2.5, None], "k": [1, 2, 3]})]


def test_with_zone_maps_off_no_column_reports_bounds():
    src = InMemorySource(_batches(), zone_maps=False)
    for name in ("x", "k"):
        stat = src.column_bounds(name)
        assert stat is None or (stat.min is None and stat.max is None)
        assert src.column_ascending(name) is False
    # The O(1) facts are still there: they are exact and cost nothing.
    assert src.column_bounds("x").null_count == 1


def test_with_zone_maps_on_the_same_columns_do():
    # Positive control: the flag, not the data, is what withholds the bounds.
    src = InMemorySource(_batches())
    assert src.column_bounds("x").min == 1.5
    assert src.column_bounds("k").max == 3
    assert src.column_ascending("k") is True
