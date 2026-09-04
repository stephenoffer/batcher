"""Iceberg tables that publish Puffin statistics plan with them, and trust them correctly.

An Iceberg table can carry a statistics file per snapshot: a Puffin blob per column whose
`ndv` property is the distinct-count estimate whichever engine last ran an ANALYZE wrote.
Batcher discarded them, so the first query against someone else's analyzed table planned
its joins on a Selinger guess.

The interesting half of this is not that the number arrives, it is *how much it is
trusted*. An Iceberg column carries exact manifest bounds and an approximate distinct
count at the same time, so the two must not share one trust tag. Both directions of that
are asserted, because getting it wrong in either direction is a correctness bug rather
than a worse plan.
"""

from __future__ import annotations

import pytest

import batcher as bt

# Imported unconditionally: `puffin` defers its own pyiceberg import, so the module loads
# without the optional dependency and the guard below is what gates execution.
from batcher.io.formats.lakehouse.iceberg.puffin import (
    APACHE_DATASKETCHES_THETA_V1,
    statistics_ndv,
    with_statistics_ndv,
)
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

pytest.importorskip("pyiceberg", reason="Iceberg support needs pyiceberg")

pytestmark = pytest.mark.io


@pytest.fixture()
def table(tmp_path):
    """A real Iceberg table with three columns and one snapshot."""
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "t",
        **{"uri": f"sqlite:///{tmp_path}/cat.db", "warehouse": f"file://{warehouse}"},
    )
    catalog.create_namespace("ns")
    data = pa.table(
        {
            "id": pa.array(range(200), pa.int64()),
            "grp": pa.array([f"g{i % 7}" for i in range(200)]),
            "val": pa.array([float(i) for i in range(200)], pa.float64()),
        }
    )
    tbl = catalog.create_table("ns.t", schema=data.schema)
    tbl.append(data)
    return tbl


def _publish_ndv(tbl, ndv_by_column: dict[str, int], *, snapshot_id=None) -> None:
    """Attach a statistics file declaring `ndv` per column, as an ANALYZE would."""
    from pyiceberg.table.statistics import BlobMetadata, StatisticsFile

    snapshot = tbl.current_snapshot()
    sid = snapshot.snapshot_id if snapshot_id is None else snapshot_id
    schema = tbl.schema()
    blobs = [
        BlobMetadata(
            type=APACHE_DATASKETCHES_THETA_V1,
            snapshot_id=sid,
            sequence_number=snapshot.sequence_number,
            fields=[schema.find_field(name).field_id],
            properties={"ndv": str(count)},
        )
        for name, count in ndv_by_column.items()
    ]
    stats = StatisticsFile(
        snapshot_id=sid,
        statistics_path=f"{tbl.location()}/metadata/stats.puffin",
        file_size_in_bytes=1,
        file_footer_size_in_bytes=1,
        blob_metadata=blobs,
    )
    with tbl.update_statistics() as update:
        update.set_statistics(stats)


def test_published_distinct_counts_are_read(table):
    """The `ndv` property of each theta blob reaches the planner, keyed by column name."""
    _publish_ndv(table, {"grp": 7, "id": 200})
    table.refresh()

    assert statistics_ndv(table, None) == {"grp": 7.0, "id": 200.0}


def test_a_table_with_no_statistics_yields_nothing(table):
    """The common case is a table nobody has analyzed, and it is not an error."""
    assert statistics_ndv(table, None) == {}


def test_statistics_from_an_ancestor_snapshot_are_used(table):
    """An ANALYZE is almost always older than the newest append; it still informs a plan."""
    import pyarrow as pa

    analyzed_at = table.current_snapshot().snapshot_id
    _publish_ndv(table, {"grp": 7}, snapshot_id=analyzed_at)
    table.refresh()
    table.append(
        pa.table(
            {
                "id": pa.array([999], pa.int64()),
                "grp": pa.array(["g0"]),
                "val": pa.array([1.0], pa.float64()),
            }
        )
    )
    table.refresh()

    assert table.current_snapshot().snapshot_id != analyzed_at
    assert statistics_ndv(table, None) == {"grp": 7.0}


def test_a_non_ancestor_snapshot_is_ignored(table):
    """Statistics for a snapshot this read cannot reach describe rows it will not see."""
    _publish_ndv(table, {"grp": 7}, snapshot_id=123456789)
    table.refresh()

    assert statistics_ndv(table, None) == {}


def test_a_sketch_never_downgrades_exact_bounds():
    """The trust question, in the direction that would lose information.

    Manifest bounds are exact and answer `min()`/`max()` without a scan. Attaching an
    approximate distinct count must not touch that.
    """
    stats = SourceStatistics(
        row_count=200,
        columns={"val": ColumnStat(min=0.0, max=199.0, provenance=Provenance.EXACT)},
    )
    enriched = with_statistics_ndv(stats, {"val": 200.0})
    column = enriched.columns["val"]

    assert column.min == 0.0 and column.max == 199.0
    assert column.provenance.is_exact, "exact bounds must survive a sketch being attached"
    assert column.ndv == 200.0


def test_a_sketch_is_never_trusted_as_an_exact_distinct_count():
    """The same question in the direction that would give a wrong answer.

    `count(distinct)` may be answered from metadata only when the count is exact. A theta
    sketch is an estimate, so `ndv_is_exact` must refuse it however precise it looks.
    """
    stats = SourceStatistics(row_count=200, columns={})
    column = with_statistics_ndv(stats, {"grp": 7.0}).columns["grp"]

    assert column.ndv == 7.0
    assert column.ndv_provenance is Provenance.SKETCH
    assert not column.ndv_is_exact


def test_an_exact_distinct_count_outranks_a_published_sketch():
    """Enrichment must not overwrite better information with worse."""
    stats = SourceStatistics(
        row_count=200,
        columns={"grp": ColumnStat(ndv=7.0, ndv_provenance=Provenance.EXACT)},
    )
    column = with_statistics_ndv(stats, {"grp": 99.0}).columns["grp"]

    assert column.ndv == 7.0
    assert column.ndv_is_exact


def test_the_source_surfaces_published_statistics_end_to_end(table, tmp_path):
    """The whole path: a published sketch reaches `SourceStatistics` through the reader."""
    _publish_ndv(table, {"grp": 7})
    table.refresh()

    # `name` matters: SqlCatalog scopes its rows by catalog name, so a spec that omits it
    # builds a catalog which cannot see tables the fixture created.
    ds = bt.read.iceberg(
        "ns.t",
        catalog={
            "type": "sql",
            "name": "t",
            "uri": f"sqlite:///{tmp_path}/cat.db",
            "warehouse": f"file://{tmp_path}/wh",
        },
    )
    stats = ds._sources[0].statistics()

    assert stats is not None
    assert stats.columns["grp"].ndv == 7.0
    assert not stats.columns["grp"].ndv_is_exact
    # And the row count the manifest gives is untouched by the enrichment.
    assert stats.row_count == 200
