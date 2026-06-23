"""Zero-config sizing helpers: `autoconfig` turns "unset" into a data-sized value."""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.api.orchestration import _MAX_PARTITIONS, DEFAULT_PARTITIONS, auto_num_partitions
from batcher.config import Config, OptimizerConfig, config_context
from batcher.io.source import InMemorySource
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit


def _scan_over(n_rows: int):
    batch = pa.record_batch({"x": list(range(n_rows))})
    src = InMemorySource([batch])
    plan = Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema()))
    return plan, [src]


def test_auto_num_partitions_scales_with_data():
    # A small target_rows_per_task forces many partitions for a modest input.
    plan, sources = _scan_over(1000)
    with config_context(Config().replace(optimizer=OptimizerConfig(target_rows_per_task=100))):
        n = auto_num_partitions(plan, sources, hub=None)
    assert n == 10  # ceil(1000 / 100), within [4, 4096]


def test_auto_num_partitions_clamps_and_has_floor():
    plan, sources = _scan_over(10)
    with config_context(
        Config().replace(optimizer=OptimizerConfig(target_rows_per_task=1_000_000))
    ):
        n = auto_num_partitions(plan, sources, hub=None)
    assert n >= 4  # floor — never degenerate to 1 bucket
    assert n <= _MAX_PARTITIONS


def test_auto_num_partitions_falls_back_when_unknown(monkeypatch):
    # If estimation raises, fall back to the historical default rather than erroring.

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no estimate")

    monkeypatch.setattr("batcher.kyber.cardinality.CardinalityEstimator", _Boom)
    plan, sources = _scan_over(100)
    assert auto_num_partitions(plan, sources, hub=None) == DEFAULT_PARTITIONS
