"""What a split set guarantees about *where equal values live* — the clustering protocol.

A `Split` is the unit of distributed read parallelism, and a split set is handed to workers
one whole split at a time (`dist.executors.partition_io.assignment.assign_splits` never cuts
one in half). So when a source's splits happen to line up with a column's *values* — a
Hive-partitioned table emitting one split per ``day=`` directory — the read has already done
the work a shuffle would do: every row with a given ``day`` is inside exactly one split,
therefore on exactly one worker.

That is worth a great deal. A ``GROUP BY day`` over such a table needs **no exchange at all**:
each worker folds its own directories to final groups and the driver concatenates. Today the
same query hash-shuffles every row to discover a partitioning the storage layout already had.

This module is how a split says so. Two optional attributes make the claim:

  * ``clustering_columns`` -- the columns whose value is constant across every row this split
    reads;
  * ``clustering_value`` -- that constant value, one entry per column.

A split declaring them says only "my rows all share this value". The guarantee a consumer
actually needs is stronger: that *no other split* holds the same value, so the value's rows
are on one worker. Two designs reach it, and the reason both exist is that the two families
of reader split differently:

  * a **directory** reader (`ParquetDatasetSource`) emits one split per ``day=`` directory, so
    distinctness holds by construction;
  * a **file** reader (Delta, Iceberg) emits one split per data file, and a partition holds
    many files, so distinctness does *not* hold and cannot be made to hold without collapsing
    the read to one task per partition.

So distinctness is not required of the split set. It is *established by the assignment*:
`group_by_clustering` puts every split sharing a value into one group, and the scheduler
assigns whole groups (`dist.executors.partition_io` `cluster_by`). A value then lands on one
worker whatever the reader's split granularity, and the fine per-file splits survive inside
the group.

`declared_clustering` is the half that must still be checked -- that every split in the set
claims the same columns. A set where one split declares ``day`` and the next declares nothing
has no column every row can be located by, and guarantees nothing.

**Why any of this is checked rather than declared.** Getting the condition wrong does not make
a query slow, it makes it *wrong*: a group split across two workers is reported twice, as two
partial groups, each labelled final. That failure is invisible on one node, invisible in every
unit test that does not run a cluster, and silent at PB scale. Verifying against the split set
the read will actually use costs one pass over a list of strings on the driver and removes the
whole class.

The attributes are duck-typed rather than required by `Split`: a split that says nothing
declares no clustering and is treated as unclustered, which is always safe.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["clustering_of", "declared_clustering", "group_by_clustering"]


def clustering_of(split: object) -> tuple[tuple[str, ...], tuple[object, ...]] | None:
    """The columns `split` holds constant and the values it holds them at, or None.

    Args:
        split: A split, which may or may not declare a clustering.

    Returns:
        A ``(columns, values)`` pair of equal, non-zero length, or None when the split
        declares no clustering or declares a malformed one.
    """
    cols = getattr(split, "clustering_columns", None)
    vals = getattr(split, "clustering_value", None)
    if not cols or vals is None:
        return None
    cols_t = tuple(cols)
    vals_t = tuple(vals)
    if not cols_t or len(cols_t) != len(vals_t):
        return None
    return cols_t, vals_t


def declared_clustering(splits: Sequence[object]) -> tuple[str, ...]:
    """The columns *every* split in this set holds constant, or ``()``.

    Half of the co-location guarantee. The other half -- that a value occurs on only one
    worker -- is established by assigning whole `group_by_clustering` groups, not by the split
    set. A set is only usable if all of it agrees: one split declaring ``day`` beside one
    declaring ``hour``, or one declaring nothing, leaves no column every row can be located by.

    Args:
        splits: The split set the read will actually use, in any order.

    Returns:
        The clustering columns common to every split, or an empty tuple.
    """
    if not splits:
        return ()
    first = clustering_of(splits[0])
    if first is None:
        return ()
    cols = first[0]
    for split in splits[1:]:
        got = clustering_of(split)
        if got is None or got[0] != cols:
            return ()
    return cols


def group_by_clustering(splits: Sequence[object]) -> list[list[object]] | None:
    """`splits` partitioned into one group per distinct clustering value.

    Assigning whole groups is what turns "these rows share a value" into "this value is on one
    worker", which is the property that lets a consumer grouping on those columns skip its
    exchange. Group *count* is then the parallelism the layout can supply, so a caller weighing
    the trade reads it from here rather than from the split count.

    Insertion order is preserved, both across groups and within one, so an order-sensitive
    caller sees the splits in the order the source planned them.

    Args:
        splits: The split set the read will actually use.

    Returns:
        The groups, or None when the set declares no common clustering.
    """
    if not declared_clustering(splits):
        return None
    groups: dict[tuple[object, ...], list[object]] = {}
    for split in splits:
        got = clustering_of(split)
        assert got is not None  # `declared_clustering` above proved every split declares one
        groups.setdefault(got[1], []).append(split)
    return list(groups.values())
