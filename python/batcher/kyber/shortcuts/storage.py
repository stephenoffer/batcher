"""Storage shortcuts — what a scan *would* have read, without reading it.

These read the connectors' declared `SourceStatistics` directly rather than the estimated
`RelStats`, because they are questions about the bytes on disk, not about the rows a plan
produces: how many files, how many row groups, how many bytes, what the table is partitioned
and clustered by. A footer already told us all of it.

The answers are what makes a cost decision explicable — "this query reads 340 files and 12 GB"
is a sentence a person can act on, and it costs one metadata round trip to say. All-or-nothing
where the sum has to be complete: a byte total that silently omits the one source that could
not describe itself is worse than no total at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from batcher.plan.stats import SortOrder, as_sort_orders

#: One element of a prefix `_common_prefix` intersects — a partition key name or an
#: ordering key. Both are compared by equality and nothing else, so one function serves.
_Key = TypeVar("_Key", str, SortOrder)

__all__ = [
    "bytes_per_row",
    "has_exact_row_count",
    "is_partitioned",
    "partition_keys",
    "row_count",
    "row_group_count",
    "sorted_by",
    "source_count",
    "total_bytes",
]

# A `SourceStatistics | None` per bound source, index-aligned with the sources themselves.
SourceStats = Sequence[object | None]


def source_count(stats: SourceStats) -> int:
    """How many sources the query is bound to (one per scanned table/dataset)."""
    return len(stats)


def row_count(stats: SourceStats) -> int | None:
    """The exact total row count across every source, or None unless all are exact.

    This is the *scanned* row count — what the files hold — not the query's result count
    (a filter or a join changes that). `rows.count` answers the latter.
    """
    total = 0
    for stat in stats:
        if stat is None or not getattr(stat, "exact_rows", False):
            return None
        rows = getattr(stat, "row_count", None)
        if rows is None:
            return None
        total += int(rows)
    return total


def has_exact_row_count(stats: SourceStats) -> bool:
    """Whether every source can state its row count exactly, without a scan."""
    return row_count(stats) is not None


def total_bytes(stats: SourceStats) -> int | None:
    """The total on-disk size across every source, in bytes, or None unless all report one."""
    total = 0
    for stat in stats:
        size = None if stat is None else getattr(stat, "byte_size", None)
        if size is None:
            return None
        total += int(size)
    return total


def row_group_count(stats: SourceStats) -> int | None:
    """The total number of physical blocks (Parquet row groups, ORC stripes), or None.

    The granularity a zone-map prune actually skips at: pruning that removes 90% of the rows
    but leaves every row group touched saves nothing, and this is the number that says so.
    """
    total = 0
    for stat in stats:
        groups = None if stat is None else getattr(stat, "row_group_count", None)
        if groups is None:
            return None
        total += int(groups)
    return total


def bytes_per_row(stats: SourceStats) -> float | None:
    """The average on-disk bytes per row across every source, or None when either total isn't known.

    On-disk, so it is *compressed* width — the number that predicts scan time. An in-memory
    row is wider; `approx.approx_row_bytes` estimates that one.
    """
    size = total_bytes(stats)
    rows = row_count(stats)
    if size is None or not rows:
        return None
    return size / rows


def partition_keys(stats: SourceStats) -> tuple[str, ...]:
    """The partition keys shared by **every** source, in order — empty if they disagree.

    A key only one side is partitioned by cannot prune the query, so the intersection (as a
    prefix, since partition order is what directory layout encodes) is the honest answer.
    """
    declared = [
        tuple(getattr(stat, "partition_keys", ()) or ()) for stat in stats if stat is not None
    ]
    if not declared or len(declared) != len(stats):
        return ()
    return _common_prefix(declared)


def is_partitioned(stats: SourceStats) -> bool:
    """Whether the data is physically partitioned on at least one column."""
    return bool(partition_keys(stats))


def sorted_by(stats: SourceStats) -> tuple[SortOrder, ...]:
    """The ordering shared by **every** source, in order, direction included.

    The same prefix-intersection rule as `partition_keys`: an ordering only one input
    maintains is not an ordering of the scan. Two sources sorted by the same column in
    opposite directions agree on nothing, and the differing key ends the prefix.
    """
    declared = [
        as_sort_orders(getattr(stat, "sorted_by", ()) or ()) for stat in stats if stat is not None
    ]
    if not declared or len(declared) != len(stats):
        return ()
    return _common_prefix(declared)


def _common_prefix(sequences: list[tuple[_Key, ...]]) -> tuple[_Key, ...]:
    """The longest leading run of keys every sequence agrees on."""
    if not sequences:
        return ()
    prefix: list[_Key] = []
    for position in zip(*sequences, strict=False):
        first = position[0]
        if any(name != first for name in position):
            break
        prefix.append(first)
    return tuple(prefix)
