"""Iceberg's per-file metrics, normalized into the add-action layout the engine prunes with.

Iceberg records, for every data file, its record count and per-column lower/upper bounds and
null counts — the same zone map Delta's add actions carry. Batcher already has one
implementation that aggregates that shape into `SourceStatistics` and one that prunes files
with it (`io.stats.lakehouse_manifest`, `io.stats.file_skipping`), so the only thing missing
was the translation. This module is it: Iceberg's metrics in, the neutral manifest out.

The connector's own docstring used to decline this, on the grounds that a data file's bounds
are *"field-id keyed, byte-encoded"* and decoding them across pyiceberg versions is fragile.
That is true of `lower_bounds`/`upper_bounds` — and it is not what we read.
`table.inspect.data_files()` also exposes **`readable_metrics`**, a struct keyed by **column
name** with values already decoded to their Arrow type. There is no field-id mapping and no
byte decoding to get wrong.

What this unlocks is not a marginal gain. Without column bounds an Iceberg scan has no zone
map at all: Kyber's pruning rules cannot fire, no predicate can be proven empty from
metadata, and `min()`/`max()` cannot be answered without reading the table. With them,
Iceberg gets exactly what Delta has.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher.io.stats.file_skipping import MAX_PREFIX, MIN_PREFIX, NULL_PREFIX

__all__ = ["file_manifest"]


def file_manifest(table: Any, snapshot_id: int | None = None) -> pa.Table | None:
    """The snapshot's data files as a manifest in the add-action layout.

    Columns: ``path``, ``num_records``, and ``min.<col>`` / ``max.<col>`` /
    ``null_count.<col>`` for every column whose metrics the table recorded.

    Args:
        table: The pyiceberg `Table`.
        snapshot_id: Optional snapshot to inspect; the current one when omitted.

    Returns:
        The manifest, or None if the metrics cannot be read (the caller then falls back to
        a plain row count, exactly as it did before).
    """
    try:
        files = table.inspect.data_files(snapshot_id=snapshot_id)
    except Exception:
        return None
    if files.num_rows == 0:
        return None
    try:
        return _normalize(files)
    except Exception:
        return None


def _normalize(files: pa.Table) -> pa.Table | None:
    """Rewrite ``inspect.data_files()`` into the neutral per-file manifest."""
    names = files.column_names
    if "file_path" not in names or "record_count" not in names:
        return None

    columns: dict[str, Any] = {
        "path": files.column("file_path"),
        "num_records": files.column("record_count"),
    }
    if "readable_metrics" in names:
        columns.update(_metric_columns(files.column("readable_metrics")))
    return pa.table(columns)


def _metric_columns(metrics: Any) -> dict[str, Any]:
    """One ``min.``/``max.``/``null_count.`` column per column the metrics describe.

    `readable_metrics` is a struct of per-column structs, so each field is flattened out to
    the prefixed name the manifest layout uses. A column whose bound the writer did not
    record comes through as null — which the pruning layer reads as "unknown, keep the
    file", never as "no match".
    """
    combined = metrics.combine_chunks() if hasattr(metrics, "combine_chunks") else metrics
    struct = combined.chunk(0) if isinstance(combined, pa.ChunkedArray) else combined

    out: dict[str, Any] = {}
    for field in struct.type:
        column = struct.field(field.name)
        for source, prefix in (
            ("lower_bound", MIN_PREFIX),
            ("upper_bound", MAX_PREFIX),
            ("null_value_count", NULL_PREFIX),
        ):
            try:
                out[f"{prefix}{field.name}"] = column.field(source)
            except (KeyError, pa.ArrowInvalid):
                continue  # this metric was not recorded for this column
    return out
