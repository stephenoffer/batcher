"""Learned column statistics belong to a *source*, not to a bare column name.

Two tables both have an `id`. A statistics store keyed by column name alone merges
them, so whichever table was measured last silently answers for every other table with
a column of that name — process-wide, on every join and group-by estimate that reads it.
These tests pin the qualification that prevents it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher import kyber
from batcher.io.source.inmemory import InMemorySource
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import source_stats_key

pytestmark = pytest.mark.unit


def _source(n_rows: int) -> InMemorySource:
    table = pa.table({"id": list(range(n_rows))})
    return InMemorySource(table.to_batches())


def _scan(source: InMemorySource) -> Scan:
    return Scan(source_id=0, schema=SchemaRef.from_arrow(source.schema()))


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_ndv_measured_on_one_source_does_not_answer_for_another() -> None:
    """A high-ndv `id` measured on a big table must not become a small table's `id` ndv."""
    hub = _hub()
    big, small = _source(1_000_000), _source(10)

    kyber.record_column_stats(hub, {"id": 1_000_000.0}, {}, source_key=source_stats_key(big))
    kyber.record_column_stats(hub, {"id": 10.0}, {}, source_key=source_stats_key(small))

    learned = kyber.load_learned_stats(hub)

    est_big = StatsEstimator([big], learned)
    est_small = StatsEstimator([small], learned)

    assert est_big.estimate(_scan(big)).column("id").ndv == 1_000_000.0
    # Before qualification this read back 10.0 (or 1e6) depending purely on write order.
    assert est_small.estimate(_scan(small)).column("id").ndv == 10.0


def test_each_source_keeps_its_own_quantiles_and_mcv() -> None:
    """The quantile grid and top values are per-source too, and reach `RelStats.columns`."""
    hub = _hub()
    a, b = _source(100), _source(100)

    kyber.record_column_stats(
        hub,
        {"id": 100.0},
        {"id": {"probs": [0.0, 1.0], "values": [0.0, 99.0]}},
        {"id": 8.0},
        {"id": {"7": 0.5}},
        source_key=source_stats_key(a),
    )
    kyber.record_column_stats(
        hub,
        {"id": 100.0},
        {"id": {"probs": [0.0, 1.0], "values": [1000.0, 2000.0]}},
        {"id": 64.0},
        {"id": {"1500": 0.9}},
        source_key=source_stats_key(b),
    )

    learned = kyber.load_learned_stats(hub)
    stat_a = StatsEstimator([a], learned).estimate(_scan(a)).column("id")
    stat_b = StatsEstimator([b], learned).estimate(_scan(b)).column("id")

    assert stat_a.quantiles == {"probs": [0.0, 1.0], "values": [0.0, 99.0]}
    assert stat_b.quantiles == {"probs": [0.0, 1.0], "values": [1000.0, 2000.0]}
    assert stat_a.mcv == {"7": 0.5}
    assert stat_b.mcv == {"1500": 0.9}
    assert (stat_a.avg_bytes, stat_b.avg_bytes) == (8.0, 64.0)


def test_unqualified_entries_still_apply_as_a_global_fallback() -> None:
    """A hub persisted by an older build wrote unqualified keys; they must still be read."""
    hub = _hub()
    src = _source(50)
    kyber.record_column_stats(hub, {"id": 42.0}, {})  # legacy shape: no source_key

    learned = kyber.load_learned_stats(hub)
    assert StatsEstimator([src], learned).estimate(_scan(src)).column("id").ndv == 42.0


def test_a_source_qualified_stat_beats_the_legacy_global_one() -> None:
    hub = _hub()
    src = _source(50)
    kyber.record_column_stats(hub, {"id": 42.0}, {})  # legacy global
    kyber.record_column_stats(hub, {"id": 7.0}, {}, source_key=source_stats_key(src))

    learned = kyber.load_learned_stats(hub)
    assert StatsEstimator([src], learned).estimate(_scan(src)).column("id").ndv == 7.0
