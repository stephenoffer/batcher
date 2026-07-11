"""Persisted source statistics round-trip (the write-once, read-next contract).

A `SourceStatistics` written for a source `identity` must reload byte-for-byte in the
fields that drive re-read planning — row/byte counts, physical ordering, partition keys,
and per-column stats — so a footerless source (CSV/JSON) keeps what the write measured.
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.metadata.source_stats_store import load_source_stats, save_source_stats
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_roundtrip_preserves_counts_and_columns():
    hub = _hub()
    stats = SourceStatistics(
        row_count=1000,
        byte_size=64_000,
        columns={
            "x": ColumnStat(min=0, max=99, null_count=0, ndv=100, provenance=Provenance.EXACT),
        },
        exact_rows=True,
    )
    save_source_stats(hub, "src://a", stats)
    got = load_source_stats(hub, "src://a")
    assert got is not None
    assert got.row_count == 1000
    assert got.byte_size == 64_000
    assert got.columns["x"].ndv == 100
    assert got.columns["x"].provenance is Provenance.EXACT


def test_roundtrip_preserves_sorted_by_and_partition_keys():
    # Physical ordering + partition keys enable redundant-sort removal and partition
    # pruning on a reloaded footerless source — they must survive persistence.
    hub = _hub()
    stats = SourceStatistics(
        row_count=10,
        sorted_by=("ts", "id"),
        partition_keys=("region",),
    )
    save_source_stats(hub, "src://b", stats)
    got = load_source_stats(hub, "src://b")
    assert got is not None
    assert got.sorted_by == ("ts", "id")
    assert got.partition_keys == ("region",)


def test_missing_identity_returns_none():
    assert load_source_stats(_hub(), "src://absent") is None
