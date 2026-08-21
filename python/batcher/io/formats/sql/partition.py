"""Range partitioning — turning one big table read into N parallel queries.

A single SQL query is a single stream: one connection, one cursor, one core, however
large the table. That is the difference between a warehouse extract that finishes in
minutes and one that finishes in hours, and it is why every bulk-extract tool has some
form of this feature.

The shape here is deliberately **Spark's JDBC reader**, because that is the spelling
users already know: pick an indexed numeric column, give its approximate bounds, say how
many partitions you want, and the reader issues that many queries in parallel, each
covering a disjoint slice.

    partition_on="id", lower_bound=1, upper_bound=1_000_000, num_partitions=8

The one thing worth understanding before using it:

**The bounds do not filter.** `lower_bound`/`upper_bound` describe where to *cut*, not
what to *keep*. The first partition is unbounded below and the last is unbounded above,
so a row outside the stated range is still read — it simply lands in an edge partition.
If bounds filtered, a stale `upper_bound` would silently drop every row inserted since
you wrote it, and the read would look perfectly successful. Getting the bounds wrong
here costs *skew*, never rows. Same reasoning puts `NULL` keys explicitly in the first
partition rather than letting them fall through every range test and vanish.

Together those two properties give the invariant this module is built around, and which
its tests assert directly: the partitions are **disjoint and exhaustive** — concatenating
them reproduces the unpartitioned read exactly, for any bounds, including wrong ones.

Choosing a column: it must be numeric and should be indexed and reasonably uniform. A
partitioned read on an unindexed column makes the database do N full scans instead of
one, which is slower than not partitioning at all.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

from batcher._internal.errors import BackendError

__all__ = ["range_predicates"]


def _stride_bounds(lower: float, upper: float, num_partitions: int) -> list[float]:
    """The `num_partitions - 1` interior cut points between `lower` and `upper`.

    Computed as ``lower + i * (upper - lower) / n`` in floating point rather than by
    accumulating a stride, so rounding error cannot drift across partitions and leave a
    gap between one partition's upper edge and the next one's lower edge.
    """
    span = upper - lower
    return [lower + (i * span) / num_partitions for i in range(1, num_partitions)]


def range_predicates(
    column: str,
    lower_bound: float,
    upper_bound: float,
    num_partitions: int,
    quote: Callable[[str], str] = lambda name: name,
) -> list[str | None]:
    """Disjoint, exhaustive SQL ``WHERE`` fragments splitting `column` into slices.

    Every row of the source relation matches exactly one returned fragment, including
    rows whose key is NULL or lies outside ``[lower_bound, upper_bound]`` — see the
    module docstring for why that is a correctness requirement and not a nicety.

    Args:
        column: The partition key. Must be numeric, and should be indexed.
        lower_bound: Approximate minimum of `column`. A cut point, not a filter.
        upper_bound: Approximate maximum of `column`. A cut point, not a filter.
        num_partitions: How many parallel queries to produce. 1 means "do not partition"
            and yields a single `None` fragment.
        quote: How to delimit `column` for the target dialect (`uri.quote_identifier`).
            Defaults to leaving it verbatim, which is what a backend that cannot name its
            dialect must do.

    Returns:
        One fragment per partition, in key order. A `None` entry means "no filter" —
        the whole relation — which is what a single partition is.

    Raises:
        BackendError: If `num_partitions` is below 1, or the bounds are inverted.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.partition import range_predicates
            >>> for fragment in range_predicates("id", 0, 100, 4):
            ...     print(fragment)
            id < 25.0 OR id IS NULL
            id >= 25.0 AND id < 50.0
            id >= 50.0 AND id < 75.0
            id >= 75.0

            >>> from batcher.io.formats.sql.uri import quote_identifier
            >>> range_predicates(
            ...     "order", 0, 100, 2, quote=lambda n: quote_identifier(n, "postgresql")
            ... )[1]
            '"order" >= 50.0'
    """
    if num_partitions < 1:
        raise BackendError(f"num_partitions must be >= 1, got {num_partitions}")
    if upper_bound < lower_bound:
        raise BackendError(
            f"upper_bound ({upper_bound}) must be >= lower_bound ({lower_bound}) "
            f"for partition column {column!r}"
        )
    if num_partitions == 1:
        return [None]
    if upper_bound == lower_bound:
        # Every cut point would coincide, so N-1 of the partitions would be empty and the
        # read would be N queries doing one query's work. One partition is the honest
        # answer for a key with no spread.
        return [None]

    cuts = _stride_bounds(lower_bound, upper_bound, num_partitions)
    # NULL keys ride in the first partition: `col < x` and `col >= x` are both UNKNOWN
    # for NULL, so without this every NULL-keyed row would match no partition at all and
    # disappear from a read that reported success.
    # Delimited for the dialect where one is known. A partition column is chosen for being
    # indexed and numeric, which says nothing about its *name*: `order`, `key` and `end`
    # are all reserved words and all plausible, and an unquoted one is a syntax error from
    # the server on every split at once rather than a slow read.
    name = quote(column)
    fragments: list[str | None] = [f"{name} < {cuts[0]} OR {name} IS NULL"]
    fragments += [f"{name} >= {lo} AND {name} < {hi}" for lo, hi in pairwise(cuts)]
    fragments.append(f"{name} >= {cuts[-1]}")
    return fragments
