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
from batcher.plan.stats import ColumnStat, Provenance, SortOrder

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
        sorted_by=(SortOrder("ts", descending=True), SortOrder("id")),
        partition_keys=("region",),
    )
    save_source_stats(hub, "src://b", stats)
    got = load_source_stats(hub, "src://b")
    assert got is not None
    # The direction survives: a descending key round-trips as descending, and the plain
    # ascending one still persists in the compact bare-name form an older store wrote.
    assert got.sorted_by == (SortOrder("ts", descending=True), SortOrder("id"))
    assert got.partition_keys == ("region",)


def test_a_bare_column_name_from_an_older_store_reads_as_ascending():
    """The compact encoding is what earlier stores wrote, and it must keep its meaning."""
    hub = _hub()
    save_source_stats(hub, "src://c", SourceStatistics(row_count=1, sorted_by=("ts",)))
    got = load_source_stats(hub, "src://c")
    assert got is not None
    assert got.sorted_by == (SortOrder("ts"),)


def test_missing_identity_returns_none():
    assert load_source_stats(_hub(), "src://absent") is None


def test_roundtrip_preserves_the_distributional_stats():
    """The quantile grid, the MCV table and the measured width survive a save/load.

    These are what answer a *cold* range or equality predicate. Dropping them on persist left
    a first-ever read of a written path falling back to the Selinger range constant and
    `1/ndv`, with the measured statistics sitting unused in the same record.
    """
    hub = _hub()
    grid = {"probs": [0.0, 0.5, 1.0], "values": [10.0, 55.0, 99.0]}
    stats = SourceStatistics(
        row_count=1000,
        columns={
            "x": ColumnStat(
                min=10,
                max=99,
                ndv=100,
                quantiles=grid,
                mcv={"55": 0.4},
                avg_bytes=8.0,
                provenance=Provenance.SKETCH,
            )
        },
    )
    save_source_stats(hub, "src://dist", stats)
    got = load_source_stats(hub, "src://dist")
    assert got is not None
    col = got.columns["x"]
    assert col.quantiles == grid
    assert col.mcv == {"55": 0.4}
    assert col.avg_bytes == 8.0


def test_a_column_carrying_only_a_quantile_grid_is_not_dropped():
    """A grid alone is a usable statistic, so it must survive the bare-provenance check."""
    hub = _hub()
    grid = {"probs": [0.0, 1.0], "values": [1.0, 2.0]}
    save_source_stats(
        hub, "src://gridonly", SourceStatistics(columns={"x": ColumnStat(quantiles=grid)})
    )
    got = load_source_stats(hub, "src://gridonly")
    assert got is not None
    assert got.columns["x"].quantiles == grid


@pytest.mark.parametrize(
    "bad",
    [
        {"probs": [0.0, 1.0], "values": [1.0]},  # length mismatch
        {"probs": [], "values": []},  # empty
        {"probs": [0.0, 1.0]},  # no values
        {"probs": [0.0, 1.0], "values": ["a", "b"]},  # non-numeric
        {"probs": [True, False], "values": [1.0, 2.0]},  # bools are not positions
    ],
)
def test_a_malformed_grid_is_dropped_rather_than_half_decoded(bad):
    """A half-decoded CDF gives a confident wrong selectivity; a missing one falls back."""
    hub = _hub()
    save_source_stats(
        hub, "src://bad", SourceStatistics(columns={"x": ColumnStat(ndv=5, quantiles=bad)})
    )
    got = load_source_stats(hub, "src://bad")
    assert got is not None
    assert got.columns["x"].quantiles is None
    assert got.columns["x"].ndv == 5  # the rest of the column still round-trips


def test_an_oversized_grid_is_dropped_whole_and_an_oversized_mcv_is_truncated():
    """The asymmetry is deliberate: truncating a CDF misdescribes its tail, an MCV's does not."""
    hub = _hub()
    n = 2000
    stats = SourceStatistics(
        columns={
            "x": ColumnStat(
                quantiles={
                    "probs": [i / n for i in range(n)],
                    "values": [float(i) for i in range(n)],
                },
                mcv={str(i): float(i) for i in range(n)},
            )
        }
    )
    save_source_stats(hub, "src://big", stats)
    col = load_source_stats(hub, "src://big").columns["x"]
    assert col.quantiles is None
    assert len(col.mcv) == 256
    assert col.mcv["1999"] == 1999.0  # the most frequent survived the truncation


def test_the_reloaded_grid_actually_moves_a_range_estimate():
    """A persisted statistic nothing consumes is a no-op, so prove the round-trip is live.

    The grid says the column's values are packed into the bottom of its `[0, 1000]` bounds, so
    `x < 100` keeps most rows. Without a grid the estimator falls back to interpolating on the
    bounds alone and reads the same predicate as keeping about a tenth. The two must differ,
    or persisting the grid bought nothing.
    """
    from batcher.kyber.stats.selectivity import predicate_selectivity
    from batcher.plan.expr_ir import Binary, Col, Lit

    hub = _hub()
    packed = {"probs": [0.0, 0.25, 0.5, 0.75, 1.0], "values": [0.0, 10.0, 25.0, 60.0, 1000.0]}
    save_source_stats(
        hub,
        "src://live",
        SourceStatistics(
            row_count=1000,
            columns={"x": ColumnStat(min=0, max=1000, ndv=500, quantiles=packed)},
        ),
    )
    col = load_source_stats(hub, "src://live").columns["x"]
    assert col.quantiles == packed, "precondition: the grid survived the round-trip"

    pred = Binary("lt", Col("x"), Lit(100))
    bounds = {"x": (0, 1000)}
    with_grid = predicate_selectivity(pred, {"x": 500.0}, None, {"x": col.quantiles}, None, bounds)
    without_grid = predicate_selectivity(pred, {"x": 500.0}, None, {}, None, bounds)

    assert with_grid > without_grid, (
        f"the reloaded grid did not reach the estimator: {with_grid} vs {without_grid}"
    )
    assert with_grid > 0.7, f"the grid says most rows are below 100, got {with_grid}"
