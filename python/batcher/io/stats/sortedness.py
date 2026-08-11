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

1. **Every row group of every file** declares the same leading sorting column, in the same
   direction. Both directions are provable and both are claimed: a descending sort key is
   as common as an ascending one (a recent-first event table is written exactly that way),
   and `SortOrder` records which, so there is nothing to reinterpret.
2. **Row groups are ordered within each file**: for an ascending key, row group *i*'s max ≤
   row group *i+1*'s min, and the mirrored comparison for a descending one. A file may
   declare each row group internally sorted while the groups themselves are shuffled.
3. **Files are ordered across the dataset**: file *i* precedes file *i+1* under the same
   comparison, in the order the scan reads them. A directory of individually-sorted files
   is not a sorted relation.

Any missing statistic, any null in the key, any unordered pair — the claim is dropped.
The cost of declining is a sort that was going to happen anyway.

Parquet's `nulls_first` flag is deliberately *not* a reason to refuse, and the reason is
worth stating because it looks like a hole. Condition 2 rejects any row group whose key
holds a null, so by the time a claim is made the key is proved null-free — and with no
null row to place, the two null placements describe the same row order. The claim is
recorded in the canonical nulls-last spelling, and `orderings_satisfy` matches a
nulls-first request against it through the same proof. The order of the two checks is
load-bearing: the null flag may only be ignored *because* nulls are refused downstream.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from batcher.plan.stats import SortOrder

__all__ = ["proved_sorted_by"]


def proved_sorted_by(metadatas: list[Any]) -> tuple[SortOrder, ...]:
    """The ordering this dataset is provably stored in.

    Args:
        metadatas: Each file's Parquet `FileMetaData`, **in the order the scan reads
            them**, with unreadable files represented by None. Order matters: the proof
            is about the concatenation of the files as the scan produces it.

    Returns:
        The leading proved ordering key, or `()` when sortedness cannot be proved. Never a
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
    name, index, descending = key
    file_bounds: list[tuple[Any, Any]] = []
    for meta in present:
        bounds = _ordered_row_group_bounds(meta, index, descending)
        if bounds is None:
            return ()
        file_bounds.append(bounds)
    return (SortOrder(name, descending),) if _monotonic(file_bounds, descending) else ()


def _declared_key(metadatas: list[Any]) -> tuple[str, int, bool] | None:
    """The `(name, column_index, descending)` every row group agrees on, else None.

    Every row group must name the same column *and* the same direction — a dataset whose
    files disagree about which way the key runs is not sorted at all.
    """
    agreed: tuple[int, bool] | None = None
    for meta in metadatas:
        for rg in range(meta.num_row_groups):
            columns = meta.row_group(rg).sorting_columns
            if not columns:
                return None
            first = columns[0]
            # `nulls_first` is not read here on purpose: `_ordered_row_group_bounds`
            # refuses any row group whose key holds a null, so the flag describes the
            # placement of rows that are proved not to exist. See the module docstring.
            declared = (first.column_index, bool(getattr(first, "descending", False)))
            if agreed is None:
                agreed = declared
            elif agreed != declared:
                return None
    if agreed is None:
        return None
    index, descending = agreed
    names = metadatas[0].schema.names
    if index >= len(names):
        return None
    return names[index], index, descending


def _ordered_row_group_bounds(meta: Any, index: int, descending: bool) -> tuple[Any, Any] | None:
    """The file's `(min, max)` key bounds, or None if its row groups are not ordered.

    A file may mark each row group sorted while the groups are themselves out of order —
    that file is not sorted, and only comparing adjacent groups reveals it.

    The pair is always `(min, max)`, whichever direction the key runs, so the across-file
    check is the *same* comparison over the same shape of value — the file the scan reads
    first simply holds the largest keys when the order is descending.
    """
    bounds: list[tuple[Any, Any]] = []
    for rg in range(meta.num_row_groups):
        column = meta.row_group(rg).column(index)
        stats = column.statistics
        if stats is None or not stats.has_min_max:
            return None
        # A null in the key makes the ordering unverifiable from bounds alone — and it is
        # what lets `_declared_key` ignore the `nulls_first` flag.
        if getattr(stats, "null_count", None):
            return None
        bounds.append((stats.min, stats.max))
    if not bounds or not _monotonic(bounds, descending):
        return None
    # Row groups run low-to-high ascending and high-to-low descending, so the file's
    # extremes sit at opposite ends of the list in the two cases.
    return (bounds[-1][0], bounds[0][1]) if descending else (bounds[0][0], bounds[-1][1])


def _monotonic(bounds: list[tuple[Any, Any]], descending: bool) -> bool:
    """Whether consecutive `(min, max)` ranges run monotonically in the declared direction.

    Ascending requires each range to end at or below where the next begins; descending
    requires each range to *begin* at or above where the next *ends*, which is the same
    statement read through the reversed pair.
    """
    try:
        for earlier, later in pairwise(bounds):
            if descending:
                if earlier[0] < later[1]:
                    return False
            elif earlier[1] > later[0]:
                return False
    except TypeError:  # values that do not compare (mixed types) prove nothing
        return False
    return True
