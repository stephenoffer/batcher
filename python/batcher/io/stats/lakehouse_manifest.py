"""Manifest-derived statistics for lakehouse tables (Delta, Iceberg).

A lakehouse table's transaction log / manifest already records, per data file,
the exact record count and per-column min/max/null-count — the IO layer lists
these during split planning. This module aggregates them into a
`SourceStatistics` with no data scan.

Row counts from a manifest are authoritative → `exact_rows=True`. Column
statistics carry provenance by kind:

  - **Partition columns** (``partition.<col>``): the value is the literal
    partition key, constant within each file and never truncated, so its
    aggregated min/max is `EXACT` — a Hive-style partition column that takes one
    value across the table has ``min == max`` exactly, letting a filter on it be
    answered from the manifest.
  - **Numeric/temporal/bool data columns**: Delta records the *true* per-file
    min/max/null-count for these (only string/binary stats are truncated), so
    when **every** file recorded the stat the cross-file aggregate (min of mins,
    max of maxs, sum of null-counts) is still `EXACT`. A NaN float bound is
    dropped as unordered.
  - **String/binary data columns**, or any column some file left unrecorded:
    `DEFAULT` — a valid pruning/zone-map bound, but never an exact `min()`/`max()`.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.io.stats.columnar_footer import is_exact_minmax_type
from batcher.io.stats.file_skipping import (
    MAX_PREFIX as _MAX_PREFIX,
)
from batcher.io.stats.file_skipping import (
    MIN_PREFIX as _MIN_PREFIX,
)
from batcher.io.stats.file_skipping import (
    NULL_PREFIX as _NULL_PREFIX,
)
from batcher.io.stats.file_skipping import (
    PARTITION_PREFIX as _PARTITION_PREFIX,
)
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["manifest_statistics"]


def manifest_statistics(add_actions: Any) -> SourceStatistics | None:
    """Aggregate a lakehouse table's per-file manifest into `SourceStatistics`.

    `add_actions` is the manifest in the add-action layout — one row per data file with a
    `num_records` column, optional `partition.<col>` partition values, and, when the table
    collects stats, `min.<col>` / `max.<col>` / `null_count.<col>` columns. Delta produces
    it directly (`get_add_actions(flatten=True)`); Iceberg's connector normalizes its
    `readable_metrics` into the same shape. Nothing here is format-specific, which is the
    point — the aggregation, the provenance rules, and the file skipping that consumes it
    are one implementation for both.

    Returns the exact total row count always, exact bounds for partition and numeric
    columns where every file recorded them, and pruning-grade bounds otherwise.
    Best-effort: any failure yields None so the caller falls back to a plain row count.
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except Exception:
        return None
    try:
        names = add_actions.column_names
        if "num_records" not in names:
            return None
        total = int(pc.sum(add_actions.column("num_records")).as_py() or 0)
        columns = _delta_columns(add_actions, names, pa, pc)
        return SourceStatistics(row_count=total, columns=columns, exact_rows=True)
    except Exception as exc:
        note_suppressed("io", "extract manifest stats", exc)
        return None


def _delta_columns(add_actions: Any, names: list[str], pa: Any, pc: Any) -> dict[str, ColumnStat]:
    """Per-column stats aggregated across files, with kind-specific provenance."""
    cols: dict[str, ColumnStat] = {}
    # Partition columns first — their literal value is exact and outranks any
    # data-column stat of the same name.
    partition_cols = {
        n[len(_PARTITION_PREFIX) :]: n for n in names if n.startswith(_PARTITION_PREFIX)
    }
    for col, src in partition_cols.items():
        stat = _partition_stat(add_actions, src, pc)
        if stat is not None:
            cols[col] = stat
    min_cols = {n[len(_MIN_PREFIX) :]: n for n in names if n.startswith(_MIN_PREFIX)}
    max_cols = {n[len(_MAX_PREFIX) :]: n for n in names if n.startswith(_MAX_PREFIX)}
    null_cols = {n[len(_NULL_PREFIX) :]: n for n in names if n.startswith(_NULL_PREFIX)}
    for col in set(min_cols) | set(max_cols) | set(null_cols):
        if col in cols:
            continue  # a partition column already carries an exact value
        stat = _data_column_stat(
            add_actions, min_cols.get(col), max_cols.get(col), null_cols.get(col), pa, pc
        )
        if stat is not None:
            cols[col] = stat
    return cols


def _partition_stat(add_actions: Any, src: str, pc: Any) -> ColumnStat | None:
    """Exact min/max for a partition column (its value is the literal key)."""
    try:
        col = add_actions.column(src)
        cmin = pc.min(col).as_py()
        cmax = pc.max(col).as_py()
    except Exception:
        return None
    if cmin is None and cmax is None:
        return None
    # The partition value is constant within a file and untruncated across the
    # manifest → the aggregate is the true extreme.
    return ColumnStat(min=cmin, max=cmax, provenance=Provenance.EXACT)


def _data_column_stat(
    add_actions: Any,
    min_name: str | None,
    max_name: str | None,
    null_name: str | None,
    pa: Any,
    pc: Any,
) -> ColumnStat | None:
    """Aggregate a data column's per-file min/max/null-count across the manifest.

    `EXACT` only when the column is a non-truncatable type *and* every file
    recorded the bound (a missing per-file stat would only widen the aggregate,
    breaking exactness). A NaN float bound is dropped as unordered.
    """
    cmin, min_complete = _agg_bound(add_actions, min_name, pc, "min")
    cmax, max_complete = _agg_bound(add_actions, max_name, pc, "max")
    cnull, null_complete = _agg_bound(add_actions, null_name, pc, "sum")
    if cmin is None and cmax is None and cnull is None:
        return None
    exact_type = _exact_data_type(add_actions, min_name or max_name, pa)
    nan = _is_nan(cmin) or _is_nan(cmax)
    if nan:
        cmin, cmax = None, None
    bounds_exact = exact_type and min_complete and max_complete and not nan
    return ColumnStat(
        min=cmin,
        max=cmax,
        # Exact null count only when the type is exact and every file recorded it.
        null_count=float(cnull) if (cnull is not None and null_complete and exact_type) else None,
        provenance=Provenance.EXACT if bounds_exact else Provenance.DEFAULT,
    )


def _agg_bound(add_actions: Any, name: str | None, pc: Any, how: str) -> tuple[Any, bool]:
    """(`aggregated value`, `complete?`) for one add-action column.

    `complete` is True when no file left the stat null — the precondition for the
    aggregate to be an exact extreme/sum rather than a mere bound.
    """
    if name is None:
        return None, False
    try:
        col = add_actions.column(name)
        complete = col.null_count == 0
        if how == "min":
            return pc.min(col).as_py(), complete
        if how == "max":
            return pc.max(col).as_py(), complete
        return pc.sum(col).as_py(), complete
    except Exception as exc:
        note_suppressed("io", "aggregate manifest bound", exc)
        return None, False


def _exact_data_type(add_actions: Any, name: str | None, pa: Any) -> bool:
    """Whether the add-action stat column `name` has a non-truncatable value type."""
    if name is None:
        return False
    try:
        field = add_actions.schema.field(name)
    except Exception:
        return False
    return is_exact_minmax_type(field.type) and not pa.types.is_null(field.type)


def _is_nan(value: Any) -> bool:
    import math

    return isinstance(value, float) and math.isnan(value)
