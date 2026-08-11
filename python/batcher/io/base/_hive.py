"""Hive partitioning: what a ``col=value`` path segment is, and where the runs begin.

Split out of `sink` because it is a different responsibility from "the template method a
format writer subclasses", and because two other places need it: the distributed write
cuts its shards on the same partition-key runs, and any reader reasoning about the layout
has to agree with the writer on what a segment means. A second copy of these rules would
drift, and the drift would be silent — a value that encodes one way and decodes another is
a wrong answer, not a formatting difference.
"""

from __future__ import annotations

import os
from typing import Any

import pyarrow as pa

__all__ = [
    "HIVE_NULL",
    "hive_partition_run_starts",
    "hive_path_segment",
    "warn_high_cardinality_partitioning",
]

#: How a Hive directory name spells NULL. Every engine reading this layout special-cases
#: this exact string, which is why a real value equal to it cannot be written.
HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"

# Partition-directory count in one shard past which the layout is called out as a mistake.
# Hive partitioning trades a directory per distinct value for the ability to skip whole
# directories, and that trade inverts once the values are many and small: the write becomes
# a per-directory `mkdirs` + PUT storm, and the *next* query pays a listing over all of them
# to read a few rows each. Partitioning by a high-cardinality column (an id, a timestamp to
# the second) is the way people arrive here, and the symptom — a write that takes hours and
# a table that is slow forever after — never names its cause. 10,000 in a single shard is
# already far past any layout chosen on purpose.
_HIGH_CARDINALITY_PARTITIONS = max(
    1, int(os.environ.get("BATCHER_PARTITION_WARN_THRESHOLD", "10000"))
)


def hive_partition_run_starts(ordered: pa.Table, cols: list[str], pc: Any) -> list[int]:
    """Row offsets where a new partition-key run begins in a key-sorted table.

    A row starts a run when any key column differs from the previous row, where "differs"
    treats NULL as equal to NULL and NaN as equal to NaN — the grouping `group_by` gives
    them, and the opposite of what `equal` gives.

    Every step is a column operation, including the last one. Reading the boundaries out
    with ``to_pylist()`` and a Python ``enumerate`` was a per-row loop in the control plane
    — the one thing `.claude/rules/architecture.md` forbids outright — and it dominated the
    partitioned write rather than merely showing up in it: at 8M rows over 97 partitions it
    cost **5,546 ms**, against 1,214 ms for the sort and 1,318 ms for the gather it was
    there to interpret. `indices_nonzero` answers the same question in Arrow and returns
    one index per *partition*, so the Python list that comes back is the size of the
    result rather than the size of the input.
    """
    n = ordered.num_rows
    if n == 0:
        return []
    if n == 1:
        # One row is one run, and answering it here is not only a shortcut. The comparison
        # below would slice both operands to length 0, and a zero-length slice of a
        # `ChunkedArray` has **no chunks** — which `pc.indices_nonzero` segfaults on
        # (pyarrow 19.0.1), taking the interpreter with it rather than raising. A
        # single-row partitioned write is an ordinary thing to do, so this is the shape
        # that would have found it.
        return [0]
    changed = None
    for name in cols:
        column = ordered.column(name)
        previous, current = column.slice(0, n - 1), column.slice(1, n - 1)
        same = pc.fill_null(pc.equal(previous, current), False)
        same = pc.or_(same, pc.and_(pc.is_null(previous), pc.is_null(current)))
        if pa.types.is_floating(column.type):
            both_nan = pc.and_(
                pc.fill_null(pc.is_nan(previous), False),
                pc.fill_null(pc.is_nan(current), False),
            )
            same = pc.or_(same, both_nan)
        differs = pc.invert(pc.fill_null(same, False))
        changed = differs if changed is None else pc.or_(changed, differs)
    # `changed[i]` compares row i+1 against row i, so a True at i starts a run at i+1.
    return [0, *pc.add(pc.indices_nonzero(changed), 1).to_pylist()]


def warn_high_cardinality_partitioning(partitions: int, partition_by: list[str], path: str) -> None:
    """Say so when a write is about to create an unreasonable number of directories.

    Correctness-neutral, and deliberately loud: the cost lands on the *next* reader as much
    as on this write, so by the time anyone notices, the table exists and re-partitioning it
    means rewriting it. On a distributed write each shard reports its own share, which is
    the conservative direction — a shard that alone exceeds the threshold says it, and the
    whole write is necessarily worse.

    Args:
        partitions: Distinct partition-key combinations this shard is about to write.
        partition_by: The columns being partitioned on, named in the message.
        path: The destination, named in the message.
    """
    if partitions <= _HIGH_CARDINALITY_PARTITIONS:
        return
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        f"writing {path!r} partitioned by {partition_by} creates {partitions:,} directories "
        "from this shard alone. Hive partitioning pays for itself by letting a reader skip "
        "whole directories, and that trade inverts at this cardinality: the write becomes a "
        "PUT per directory and every later query pays a listing over all of them. Partition "
        "on a coarser column (a date rather than a timestamp, a bucket rather than an id), "
        "or drop partition_by and use sort_by to get file-level skipping instead.",
        PerformanceWarning,
        stacklevel=3,
    )


def hive_path_segment(value: Any) -> str:
    """The Hive path segment for a partition `value`, URL-encoded like Spark/Hive.

    A raw value containing ``/`` would spawn a spurious subdirectory (``c=x/y`` reads
    back as ``c=x``), and other reserved characters break directory discovery. pyarrow's
    Hive partitioning URI-decodes segment values on read, so the write must URI-encode
    them (``x/y`` → ``x%2Fy``) for the value to survive the round trip. NULL keeps its
    sentinel unencoded — the reader special-cases that exact string.

    A *real* value equal to that sentinel is refused rather than written. It cannot be
    represented: every Hive reader, pyarrow's included, decodes the segment before
    comparing it to the sentinel, so no escaping survives — ``%5F_HIVE_DEFAULT_PARTITION__``
    decodes back to the sentinel and reads as NULL just the same. Hive and Spark have the
    identical hole and fill it silently; writing the row and reading it back as NULL is
    data corruption that nothing downstream can detect, so it is a refusal here.
    """
    if value is None:
        return HIVE_NULL
    text = str(value)
    if text == HIVE_NULL:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"a partition value of {HIVE_NULL!r} cannot be written: that exact string is "
            "how a Hive layout spells NULL in a directory name, so the row would read back "
            "with a null partition key. Every escaping is decoded before the comparison, so "
            "there is no spelling that survives. Map the value to something else, or "
            "partition on a different column."
        )
    from urllib.parse import quote

    return quote(text, safe="")
