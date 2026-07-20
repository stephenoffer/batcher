"""Prove — never assume — that a Parquet dataset is globally sorted by a key.

`SourceStatistics.sorted_by` is consumed by Kyber to **delete** a redundant `Sort`
(`kyber/rules/ordering.py`). No structured source ever populated it, so a dataset written
sorted by Spark, Hive or DuckDB — which record `sorting_columns` in the footer — was
re-sorted on every read.

This is the most dangerous statistic in the bundle, and the reason it is a module rather
than three lines: every other statistic is a *bound*, so being wrong makes a plan slower.
Claiming sortedness wrongly makes the optimizer remove a sort that was doing real work,
and the query returns rows in the wrong order — silently, since a wrong order is not an
error and an order-independent test cannot see it.

So the claim is only made when all three conditions are *proved* from the footers:

1. **Every row group of every file** declares the same leading sorting column, ascending
   and nulls-last — the canonical form `RelStats.sorted_by` means. Parquet's flags carry
   descending/nulls-first variants, and those are different orderings, so they are refused
   rather than reinterpreted.
2. **Row groups are ordered within each file**: row group *i*'s max ≤ row group *i+1*'s
   min. A file may declare each row group internally sorted while the groups themselves
   are shuffled.
3. **Files are ordered across the dataset**: file *i*'s max ≤ file *i+1*'s min, in the
   order the scan reads them. A directory of individually-sorted files is not a sorted
   relation.

Any missing statistic, any null in the key, any unordered pair — the claim is dropped.
The cost of declining is a sort that was going to happen anyway.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

__all__ = ["proved_sorted_by"]


def proved_sorted_by(metadatas: list[Any]) -> tuple[str, ...]:
    """The columns this dataset is provably sorted by, ascending and nulls-last.

    Args:
        metadatas: Each file's Parquet `FileMetaData`, **in the order the scan reads
            them**, with unreadable files represented by None. Order matters: the proof
            is about the concatenation of the files as the scan produces it.

    Returns:
        The leading sorted columns, or `()` when sortedness cannot be proved. Never a
        guess — a caller may delete a `Sort` on the strength of this.
    """
    present = [m for m in metadatas if m is not None]
    if not present or len(present) != len(metadatas):
        # An unreadable footer leaves a gap in the proof: the file it stands for could
        # hold anything, so the concatenation cannot be shown to be ordered.
        return ()
    key = _declared_key(present)
    if key is None:
        return ()
    name, index = key
    file_bounds: list[tuple[Any, Any]] = []
    for meta in present:
        bounds = _ordered_row_group_bounds(meta, index)
        if bounds is None:
            return ()
        file_bounds.append(bounds)
    return (name,) if _ascending(file_bounds) else ()


def _declared_key(metadatas: list[Any]) -> tuple[str, int] | None:
    """The `(name, column_index)` every row group agrees it is sorted by, else None."""
    agreed: int | None = None
    for meta in metadatas:
        for rg in range(meta.num_row_groups):
            columns = meta.row_group(rg).sorting_columns
            if not columns:
                return None
            first = columns[0]
            # Descending and nulls-first are *different* orderings than the canonical one
            # `sorted_by` denotes; reinterpreting them would be the wrong-order bug.
            if getattr(first, "descending", False) or getattr(first, "nulls_first", False):
                return None
            if agreed is None:
                agreed = first.column_index
            elif agreed != first.column_index:
                return None
    if agreed is None:
        return None
    names = metadatas[0].schema.names
    if agreed >= len(names):
        return None
    return names[agreed], agreed


def _ordered_row_group_bounds(meta: Any, index: int) -> tuple[Any, Any] | None:
    """`(min, max)` for the key across `meta`, or None if its row groups are not ordered.

    A file may mark each row group sorted while the groups are themselves out of order —
    that file is not sorted, and only comparing adjacent groups reveals it.
    """
    bounds: list[tuple[Any, Any]] = []
    for rg in range(meta.num_row_groups):
        column = meta.row_group(rg).column(index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            return None
        # A null in the key makes nulls-last ordering unverifiable from bounds alone.
        if getattr(stats, "null_count", None):
            return None
        bounds.append((stats.min, stats.max))
    if not bounds:
        return None
    return (bounds[0][0], bounds[-1][1]) if _ascending(bounds) else None


def _ascending(bounds: list[tuple[Any, Any]]) -> bool:
    """Whether consecutive `(min, max)` ranges are non-overlapping and increasing."""
    try:
        for earlier, later in pairwise(bounds):
            if earlier[1] > later[0]:
                return False
    except TypeError:  # values that do not compare (mixed types) prove nothing
        return False
    return True
