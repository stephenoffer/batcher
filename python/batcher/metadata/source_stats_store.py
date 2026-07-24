"""Persisted source statistics — remember what Batcher wrote, for the next read.

When Batcher writes a dataset it knows the result exactly: the row count and byte
size, and (cheaply, from the result already in memory) per-column distinct counts.
A later read of that path can then be answered from metadata even for formats with
no footer (CSV, JSON), instead of re-scanning. This module persists a
`SourceStatistics` into the `MetadataHub`, keyed by the source's stable
`identity()`, and loads it back on read.

It stores only JSON-safe scalars and tags row counts `exact_rows=True` (the write
counted every row) while distinct counts stay `SKETCH` (HLL-derived). Best-effort
throughout: persistence never breaks a write, and a missing/garbled entry simply
falls through to a normal read.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.metadata.hub import MetadataHub
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["load_source_stats", "save_source_stats"]

_NAMESPACE = "io.source_stats"
_JSON_SCALARS = (int, float, str, bool)


def save_source_stats(hub: MetadataHub, identity: str, stats: SourceStatistics) -> None:
    """Persist `stats` for a source `identity`. Best-effort; never raises."""
    import contextlib

    with contextlib.suppress(Exception):  # persistence must never break a write
        hub.save_params(f"{_NAMESPACE}:{identity}", _encode(stats))


def load_source_stats(hub: MetadataHub, identity: str) -> SourceStatistics | None:
    """Load persisted statistics for a source `identity`, or None if absent."""
    try:
        blob = hub.load_params(f"{_NAMESPACE}:{identity}")
    except Exception as exc:
        note_suppressed("metadata", "load source stats", exc)
        return None
    return _decode(blob) if blob else None


def _encode(stats: SourceStatistics) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name, col in stats.columns.items():
        encoded = _encode_column(col)
        if encoded:
            columns[name] = encoded
    return {
        "row_count": stats.row_count,
        "byte_size": stats.byte_size,
        "exact_rows": stats.exact_rows,
        # Physical properties drive redundant-sort removal and partition pruning; without
        # them a Batcher-written footerless source (CSV/JSON) loses its ordering on reload.
        "sorted_by": list(stats.sorted_by),
        "partition_keys": list(stats.partition_keys),
        "columns": columns,
    }


def _encode_column(col: ColumnStat) -> dict[str, Any]:
    out: dict[str, Any] = {"provenance": col.provenance.name}
    for field in ("min", "max", "null_count", "ndv", "total_sum", "mean"):
        value = getattr(col, field)
        if isinstance(value, _JSON_SCALARS):
            out[field] = value
    if col.bloom is not None:  # the data-skipping membership index, base64 for JSON
        import base64

        out["bloom"] = base64.b64encode(col.bloom).decode("ascii")
    # Drop a bare provenance with no usable values.
    if len(out) == 1:
        return {}
    # Per-field provenance sub-tags. `ndv` and `null_count` each carry their OWN trust
    # tag, which the bundle's cannot express: a sketch distinct count rides beside exact
    # bounds (`ndv_provenance` weaker), and an exact null count rides beside byte-truncated
    # bounds (`null_count_provenance` stronger). Dropping them makes `ndv_is_exact` /
    # `null_count_is_exact` fall back to the *bundle* tag on reload — silently promoting a
    # sketch to EXACT (a wrong `count_distinct`) or demoting an exact count to a rescan. So
    # they must round-trip alongside the bundle provenance the values were stored with.
    for field in ("ndv_provenance", "null_count_provenance"):
        sub = getattr(col, field)
        if sub is not None:
            out[field] = sub.name
    return out


def _decode(blob: dict[str, Any]) -> SourceStatistics | None:
    try:
        columns = {name: _decode_column(c) for name, c in blob.get("columns", {}).items()}
        return SourceStatistics(
            row_count=blob.get("row_count"),
            byte_size=blob.get("byte_size"),
            columns=columns,
            sorted_by=tuple(blob.get("sorted_by", ())),
            partition_keys=tuple(blob.get("partition_keys", ())),
            exact_rows=bool(blob.get("exact_rows", True)),
        )
    except Exception as exc:
        note_suppressed("metadata", "decode source stats", exc)
        return None


def _decode_column(blob: dict[str, Any]) -> ColumnStat:
    prov = Provenance[blob.get("provenance", "DEFAULT")]
    bloom_b64 = blob.get("bloom")
    bloom = None
    if isinstance(bloom_b64, str):
        import base64

        bloom = base64.b64decode(bloom_b64)
    ndv_prov = blob.get("ndv_provenance")
    null_prov = blob.get("null_count_provenance")
    return ColumnStat(
        min=blob.get("min"),
        max=blob.get("max"),
        null_count=blob.get("null_count"),
        ndv=blob.get("ndv"),
        total_sum=blob.get("total_sum"),
        mean=blob.get("mean"),
        provenance=prov,
        bloom=bloom,
        ndv_provenance=Provenance[ndv_prov] if isinstance(ndv_prov, str) else None,
        null_count_provenance=Provenance[null_prov] if isinstance(null_prov, str) else None,
    )
