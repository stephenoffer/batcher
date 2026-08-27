"""The distinct-value counts a table's Puffin statistics publish, read for planning.

An Iceberg table can carry *statistics files* — Puffin blobs holding a sketch per column,
written by whichever engine last ran an ANALYZE. The standard NDV blob
(``apache-datasketches-theta-v1``) records its cardinality estimate in the blob's own
``properties`` under ``ndv``, which is how Trino and Spark consume it: they read the
property rather than deserializing the sketch, because a number is all a cost model wants.

Batcher does the same here. Reading them means the *first* query against someone else's
table plans with a real distinct count instead of a Selinger guess — for join ordering, for
build-side choice, and for equality selectivity, all of which are badly served by a default.

# What this deliberately does not do

It does not write them. The spec requires the blob payload to be a conformant Apache
DataSketches theta sketch, and Batcher's own mergeable sketches (`bc-sketches`) are HLL.
Writing an HLL under a theta blob type would produce a file every other engine misreads,
which is worse than publishing nothing. Publishing requires a theta sketch in `bc-sketches`
so that the sketch is built in the data plane where the rows already are; that is a Rust
change and is not attempted from here.

# Trust

An NDV read here is `Provenance.SKETCH`, and it is attached to `ColumnStat.ndv_provenance`
rather than to the bundle. That distinction is load-bearing: an Iceberg column carries
**exact** min/max bounds from the manifest *and* an approximate distinct count, and
collapsing both into one tag would either downgrade the bounds — losing `min()`/`max()`
answered from metadata — or promote the sketch, which would let `count(distinct)` be
answered from an estimate. Neither is acceptable, and the per-field tag is why neither
happens.

# Snapshot ancestry

A statistics file names the snapshot it was computed for, and a table almost always has
appends on top of its last ANALYZE. Statistics from an **ancestor** snapshot are therefore
used when the current one has none, which is what Trino and Spark do: a cardinality
estimate that is a few commits stale is still incomparably better than a default, and it
can never answer a query, only rank a plan. Statistics from a snapshot that is *not* an
ancestor are ignored, because those describe a branch this read is not on.

Layer: `io`, neutral.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyiceberg.table import Table

__all__ = ["APACHE_DATASKETCHES_THETA_V1", "statistics_ndv", "with_statistics_ndv"]

#: The blob type the Iceberg spec defines for a distinct-value sketch. Other blob types
#: exist and are skipped rather than guessed at.
APACHE_DATASKETCHES_THETA_V1 = "apache-datasketches-theta-v1"

#: The property every writer of that blob records its cardinality estimate under.
_NDV_PROPERTY = "ndv"


def statistics_ndv(table: Table, snapshot_id: int | None) -> dict[str, float]:
    """Per-column distinct-value estimates from `table`'s Puffin statistics.

    Args:
        table: The pyiceberg table being read.
        snapshot_id: The snapshot this read is pinned to, or None for the current one.

    Returns:
        A mapping of column name to estimated distinct count. Empty when the table
        publishes no statistics, which is the common case and not an error.
    """
    try:
        return _statistics_ndv(table, snapshot_id)
    except Exception:
        # A malformed or unreadable statistics file must cost a worse plan, never a failed
        # query: everything here is an estimate that only ranks alternatives.
        from batcher._internal.logging import get_logger

        get_logger("io").debug("could not read Iceberg statistics", exc_info=True)
        return {}


def _statistics_ndv(table: Table, snapshot_id: int | None) -> dict[str, float]:
    """The unguarded body of `statistics_ndv`."""
    files = list(getattr(table.metadata, "statistics", ()) or ())
    if not files:
        return {}
    target = snapshot_id if snapshot_id is not None else _current_snapshot_id(table)
    if target is None:
        return {}
    chosen = _closest(files, table, target)
    if chosen is None:
        return {}
    schema = table.schema()
    out: dict[str, float] = {}
    for blob in getattr(chosen, "blob_metadata", ()) or ():
        if getattr(blob, "type", None) != APACHE_DATASKETCHES_THETA_V1:
            continue
        raw = (getattr(blob, "properties", None) or {}).get(_NDV_PROPERTY)
        if raw is None:
            continue
        try:
            ndv = float(raw)
        except (TypeError, ValueError):
            continue
        if ndv < 0:
            continue
        # A theta blob names exactly one source field. A blob naming several is a
        # multi-column sketch this does not interpret, rather than one to attribute to an
        # arbitrary member of the set.
        fields = list(getattr(blob, "fields", ()) or ())
        if len(fields) != 1:
            continue
        name = _column_name(schema, fields[0])
        if name is not None:
            out[name] = ndv
    return out


def _current_snapshot_id(table: Table) -> int | None:
    """The table's current snapshot id, or None for an empty table."""
    snapshot = table.current_snapshot()
    return None if snapshot is None else int(snapshot.snapshot_id)


def _closest(files: list[Any], table: Table, target: int) -> Any | None:
    """The statistics file for `target`, else the nearest ancestor's, else None.

    Walking the parent chain rather than taking the newest file is what keeps this correct
    on a table with branches: a statistics file on a sibling branch describes rows this
    read will not see, and using it would misestimate in a way nothing later corrects.
    """
    by_snapshot = {int(f.snapshot_id): f for f in files if f.snapshot_id is not None}
    if not by_snapshot:
        return None
    parents = {
        int(s.snapshot_id): (None if s.parent_snapshot_id is None else int(s.parent_snapshot_id))
        for s in (table.metadata.snapshots or ())
    }
    seen: set[int] = set()
    current: int | None = target
    while current is not None and current not in seen:
        found = by_snapshot.get(current)
        if found is not None:
            return found
        seen.add(current)
        current = parents.get(current)
    return None


def _column_name(schema: Any, field_id: Any) -> str | None:
    """The top-level column name for an Iceberg field id.

    A nested field resolves to a dotted path, which is not a column of the Arrow schema the
    scan produces, so it is dropped rather than recorded under a name no plan can match.
    """
    try:
        name = schema.find_column_name(int(field_id))
    except Exception:
        return None
    if not name or "." in name:
        return None
    return str(name)


def with_statistics_ndv(stats: Any, ndv_by_column: dict[str, float]) -> Any:
    """Return `stats` with each column's published distinct count attached.

    The count goes on `ColumnStat.ndv_provenance`, never on the bundle's `provenance`. An
    Iceberg column carries exact manifest bounds *and* an approximate distinct count;
    collapsing the two into one tag would either lose `min()`/`max()` answered from
    metadata or let `count(distinct)` be answered from an estimate.

    Args:
        stats: The `SourceStatistics` assembled from the manifest.
        ndv_by_column: Column name to estimated distinct count.

    Returns:
        A new `SourceStatistics`, or `stats` unchanged when there is nothing to add.
    """
    if not ndv_by_column:
        return stats
    import dataclasses

    from batcher.plan.stats import ColumnStat, Provenance

    columns = dict(stats.columns)
    for name, ndv in ndv_by_column.items():
        existing = columns.get(name)
        if existing is None:
            columns[name] = ColumnStat(ndv=ndv, ndv_provenance=Provenance.SKETCH)
            continue
        # A measurement the connector already has outranks a published sketch: an exact
        # ndv is strictly better information, and overwriting it with an estimate would
        # be a downgrade dressed up as an enrichment.
        if existing.ndv is not None and existing.ndv_is_exact:
            continue
        columns[name] = dataclasses.replace(existing, ndv=ndv, ndv_provenance=Provenance.SKETCH)
    return dataclasses.replace(stats, columns=columns)
