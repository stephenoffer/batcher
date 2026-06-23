"""Manifest-derived statistics for lakehouse tables (Delta, Iceberg).

A lakehouse table's transaction log / manifest already records, per data file,
the exact record count and per-column min/max/null-count — the IO layer lists
these during split planning. This module aggregates them into a
`SourceStatistics` with no data scan.

Row counts from a manifest are authoritative → `exact_rows=True`. Column
min/max are aggregated across files and tagged `DEFAULT` (usable for zone-map
pruning and selectivity, but not as an exact `min()`/`max()` answer): a file's
recorded bounds may be writer-truncated for strings, and aggregation across
files only widens the range. Null counts are summed and stay non-exact for the
same reason.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["delta_statistics"]


def delta_statistics(add_actions: Any) -> SourceStatistics | None:
    """Aggregate a Delta table's flattened add-actions into `SourceStatistics`.

    `add_actions` is the Arrow table returned by
    `DeltaTable.get_add_actions(flatten=True)` — one row per data file with a
    `num_records` column and, when the table collects stats, `min.<col>` /
    `max.<col>` / `null_count.<col>` columns. Returns the exact total row count
    always, and per-column bounds when present. Best-effort: any failure yields
    None so the caller falls back to a plain row count.
    """
    try:
        import pyarrow.compute as pc
    except Exception:
        return None
    try:
        names = add_actions.column_names
        if "num_records" not in names:
            return None
        total = int(pc.sum(add_actions.column("num_records")).as_py() or 0)
        columns = _delta_columns(add_actions, names, pc)
        return SourceStatistics(row_count=total, columns=columns, exact_rows=True)
    except Exception:
        return None


def _delta_columns(add_actions: Any, names: list[str], pc: Any) -> dict[str, ColumnStat]:
    """Per-column min/max/null_count aggregated across files (pruning-grade)."""
    cols: dict[str, ColumnStat] = {}
    min_cols = {n[len("min.") :]: n for n in names if n.startswith("min.")}
    max_cols = {n[len("max.") :]: n for n in names if n.startswith("max.")}
    null_cols = {n[len("null_count.") :]: n for n in names if n.startswith("null_count.")}
    for col in set(min_cols) | set(max_cols) | set(null_cols):
        cmin = _agg(add_actions, min_cols.get(col), pc, "min")
        cmax = _agg(add_actions, max_cols.get(col), pc, "max")
        cnull = _agg(add_actions, null_cols.get(col), pc, "sum")
        if cmin is None and cmax is None and cnull is None:
            continue
        cols[col] = ColumnStat(
            min=cmin,
            max=cmax,
            null_count=float(cnull) if cnull is not None else None,
            provenance=Provenance.DEFAULT,  # bounds only — file stats may be truncated
        )
    return cols


def _agg(table: Any, column: str | None, pc: Any, how: str):
    if column is None:
        return None
    try:
        col = table.column(column)
        if how == "min":
            return pc.min(col).as_py()
        if how == "max":
            return pc.max(col).as_py()
        return pc.sum(col).as_py()
    except Exception:
        return None
