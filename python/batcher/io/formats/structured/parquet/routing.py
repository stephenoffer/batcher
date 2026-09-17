"""How a predicated Parquet read spends its work: skip row groups on the footers, decode the rest.

A pushed predicate has two jobs a reader can do for it. Skipping the row groups whose footer
bounds prove they hold no match saves I/O and decode outright. Removing the non-matching rows
from the groups that remain saves nothing the engine's own `Filter` would not also do, because
that `Filter` stays above the scan and runs in parallel Rust over the morsels. The Parquet source
used to do both whenever it could: the native reader pruned and then re-filtered in Python, and
a predicate it could not translate (any date) went to pyarrow's filtered scan, which decodes and
filters in one pass.

Measured on TPC-H sf10 read from Parquet, doing only the first job is faster almost everywhere:
decode the surviving row groups whole, natively, and leave the rows to the engine. Two builds
alternating over two rounds, per query, took q12 to **0.60x**, q3 and q6 to **0.66x**, q1 to
**0.76x**, and the suite from **1.526x DuckDB to 1.356x**. The reason is CPU rather than
parallelism: pyarrow's filtered read of q1's `lineitem` columns spends 10.6 CPU-seconds where
the native decode of the same columns spends 5.2, and the engine's filter is a SIMD comparison.

What it costs is memory. The filtered read holds only matching rows; this one holds every row
of every surviving group until the engine filters them. So it is taken only when those groups'
estimated width fits in a quarter of the query's memory envelope, and the selective, filtered
read keeps every case that does not fit. The row groups are still pruned first, so a clustered
table, where the predicate's matches sit in a few groups, reads just those groups either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from batcher.io.stats import RowGroupBounds
from batcher.io.stats.file_identity import FileMetaCache, file_identity

__all__ = [
    "decoded_bytes",
    "predicate_columns",
    "row_group_bounds_cached",
    "surviving_row_groups",
    "survivors_worth_pruning",
]

#: The share of the memory envelope an unfiltered read of the surviving row groups may occupy.
#: Kept well below the spill limit because it is a decoded *input*, and the operators above
#: it need their own room.
MEMORY_FRACTION = 0.25

# Per (file content, columns) row-group bounds. The footer walk is one metadata read per file,
# and a query shape run repeatedly should not pay it every time.
_BOUNDS_CACHE = FileMetaCache(4_096)


def predicate_columns(ir: Any) -> list[str]:
    """Every column a predicate IR names, in first-seen order."""
    seen: dict[str, None] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("e") == "col" and isinstance(node.get("name"), str):
                seen[node["name"]] = None
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(ir)
    return list(seen)


def row_group_bounds_cached(
    fs: Any, files: Sequence[str], columns: list[str]
) -> list[RowGroupBounds]:
    """`parquet_row_group_bounds` for `files`, memoized per file on its content identity."""
    from batcher.io.stats import parquet_row_group_bounds

    out: list[RowGroupBounds] = []
    for path in files:
        identity = file_identity(path, fs)
        key = None if identity is None else (identity, tuple(columns))
        hit = _BOUNDS_CACHE.get(key) if key is not None else None
        if hit is None:
            hit = parquet_row_group_bounds(fs, [path], columns)
            if key is not None:
                _BOUNDS_CACHE.put(key, hit, weight=max(1, len(hit)))
        out.extend(hit)
    return out


def surviving_row_groups(
    bounds: Sequence[RowGroupBounds], predicate: dict[str, Any], columns: list[str]
) -> list[RowGroupBounds]:
    """The row groups whose footer bounds do not prove `predicate` matches none of their rows.

    The bounds are laid out as the add-action manifest `io.stats.file_skipping` prunes, one
    row per row group, so a row group is skipped by exactly the rules that skip a lakehouse
    file: dates aligned to their column, a float max never trusted for what a NaN satisfies, an
    absent statistic always kept.
    """
    from batcher.io.stats.file_skipping import file_prune_mask

    if not bounds:
        return []
    table: dict[str, list[Any]] = {
        "path": [str(i) for i in range(len(bounds))],
        "num_records": [rg.num_rows for rg in bounds],
    }
    for column in columns:
        table[f"min.{column}"] = [rg.mins.get(column) for rg in bounds]
        table[f"max.{column}"] = [rg.maxs.get(column) for rg in bounds]
        table[f"null_count.{column}"] = [rg.null_counts.get(column) for rg in bounds]
    try:
        manifest = pa.table(table)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
        return list(bounds)  # bounds pyarrow cannot type prune nothing
    mask = file_prune_mask(predicate, manifest)
    if mask is None:
        return list(bounds)
    return [rg for rg, keep in zip(bounds, mask.to_pylist(), strict=True) if keep]


def survivors_worth_pruning(
    bounds: Sequence[RowGroupBounds],
    predicate: dict[str, Any],
    columns: list[str],
    morsel_rows: int,
) -> list[RowGroupBounds]:
    """`surviving_row_groups`, unless every row group together is no more than one morsel.

    Pruning evaluates the predicate over a pyarrow manifest, and building its literals imports
    pandas through pyarrow's own shim, measured at about 250 ms once per process. A read no
    larger than one morsel decodes in a small fraction of that, and the engine's `Filter`
    removes the same rows either way, so on such a read pruning costs more than it could save.
    """
    if sum(rg.num_rows for rg in bounds) <= morsel_rows:
        return list(bounds)
    return surviving_row_groups(bounds, predicate, columns)


def decoded_bytes(schema: pa.Schema, projection: list[str] | None, rows: int) -> float:
    """The estimated in-memory size of `rows` rows of `projection` decoded from `schema`."""
    from batcher.plan.types.widths import column_bytes

    names = projection if projection is not None else schema.names
    width = sum(column_bytes(schema.field(name).type) for name in names if name in schema.names)
    return width * rows
