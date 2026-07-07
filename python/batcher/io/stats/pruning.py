"""Row-group / file pruning metadata mined from Parquet footers.

Dataset-wide min/max (in `columnar_footer`) prune a predicate against the whole
source; this module exposes the *finer* per-row-group bounds a range predicate
uses to eliminate whole row groups (and whole files) without reading a row. It is
the zone-map surface: each `RowGroupBounds` carries one row group's row count and
per-column min/max/null-count, and `surviving_rows_for_range` reports how many
rows could still match a ``lower <= col <= upper`` bound after pruning the groups
that provably cannot.

The count is an *upper bound* on matches (a surviving group may still contain
non-matching rows), with one exact corner: when it is **zero**, the predicate is
provably empty over the source — the sound basis for an exact `is_empty()` on a
range predicate. Bounds for a truncated string column are still valid for
pruning (they only widen the range), so they are surfaced too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RowGroupBounds", "parquet_row_group_bounds", "surviving_rows_for_range"]


@dataclass(frozen=True, slots=True)
class RowGroupBounds:
    """One Parquet row group's pruning metadata: rows + per-column min/max/nulls.

    `mins`/`maxs` hold a column's chunk extremes (present only when the footer
    recorded them); `null_counts` holds the chunk null count. A column absent from
    a map had no recorded statistic in that row group and cannot be pruned on.
    """

    file_path: str
    row_group: int
    num_rows: int
    mins: Mapping[str, Any] = field(default_factory=dict)
    maxs: Mapping[str, Any] = field(default_factory=dict)
    null_counts: Mapping[str, int] = field(default_factory=dict)


def parquet_row_group_bounds(
    fs: Any, files: list[str], columns: list[str] | None = None
) -> list[RowGroupBounds]:
    """Per-row-group min/max/null-count across `files`, for zone-map pruning.

    `columns` restricts extraction to the predicate's columns (cheaper); None
    mines every column. Reads only footers — never a data page. Best-effort: a
    file whose footer cannot be read is skipped rather than raised.
    """
    import pyarrow.parquet as pq

    want = set(columns) if columns is not None else None
    out: list[RowGroupBounds] = []
    for path in files:
        try:
            with fs.open(path) as fh:
                meta = pq.ParquetFile(fh).metadata
        except Exception:
            continue
        names = meta.schema.names
        for rg in range(meta.num_row_groups):
            rgroup = meta.row_group(rg)
            mins: dict[str, Any] = {}
            maxs: dict[str, Any] = {}
            nulls: dict[str, int] = {}
            for ci in range(rgroup.num_columns):
                col = rgroup.column(ci)
                name = names[ci] if ci < len(names) else col.path_in_schema
                if want is not None and name not in want:
                    continue
                _collect_bounds(col, name, mins, maxs, nulls)
            out.append(RowGroupBounds(path, rg, rgroup.num_rows, mins, maxs, nulls))
    return out


def _collect_bounds(
    col: Any, name: str, mins: dict[str, Any], maxs: dict[str, Any], nulls: dict[str, int]
) -> None:
    stats = getattr(col, "statistics", None)
    if stats is None:
        return
    if getattr(stats, "has_min_max", False):
        mins[name] = stats.min
        maxs[name] = stats.max
    if getattr(stats, "has_null_count", False) and stats.null_count is not None:
        nulls[name] = stats.null_count


def surviving_rows_for_range(
    bounds: list[RowGroupBounds],
    column: str,
    lower: Any | None = None,
    upper: Any | None = None,
) -> int:
    """Upper-bound row count that survives ``lower <= column <= upper`` pruning.

    A row group is pruned when its recorded ``[min, max]`` is entirely below
    `lower` or entirely above `upper`; its rows then cannot match. A group with no
    recorded bound for `column` is conservatively kept (cannot be pruned). The
    returned sum over surviving groups is an upper bound on matching rows — and is
    exactly zero iff every group was pruned, which proves the predicate empty.
    """
    total = 0
    for rg in bounds:
        cmin = rg.mins.get(column)
        cmax = rg.maxs.get(column)
        if lower is not None and cmax is not None and cmax < lower:
            continue  # whole group below the lower bound
        if upper is not None and cmin is not None and cmin > upper:
            continue  # whole group above the upper bound
        total += rg.num_rows
    return total
