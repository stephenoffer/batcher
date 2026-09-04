"""Where a cached result is allowed to live — the `StorageLevel` contract.

`Dataset.cache()` marks a result for reuse; this says *what medium* that reuse may
draw on. It is a resource contract in the same sense as `ResourceBounds`: `api`
states it, Carbonite's `CacheStore` honors it, and neither imports the other, so it
lives here in the neutral `plan` layer alongside the rest of that vocabulary.

The three levels are Spark's, by the same names, because they are the names a user
arriving from Spark or Ray Data already types:

- `MEMORY_ONLY` — the result occupies the storage-memory envelope; when it is
  evicted it is gone and the next terminal recomputes it.
- `MEMORY_AND_DISK` — eviction *demotes* to the local spill tier instead of
  dropping, so a result that no longer fits RAM is still cheaper to recall than to
  recompute. The default, and the one that makes a bounded cache useful.
- `DISK_ONLY` — never charged against the memory envelope at all. For a result far
  larger than the storage budget that is still much cheaper to read than to
  recompute (a wide join, a re-scanned remote table).

Deliberately **not** offered: Spark's `_SER` (serialized) and `_2` (replicated)
variants. Batcher's in-memory form is already Arrow, which is the serialized form —
there is no second representation to choose between — and replication is a property
of the shared cache backend, not of a level.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["StorageLevel"]


class StorageLevel(Enum):
    """Which storage media a cached result may occupy.

    Passed to :meth:`batcher.Dataset.cache` and :meth:`batcher.Dataset.persist`. The
    names match Spark's so a ported script reads unchanged; see the module docstring
    for why the serialized and replicated variants have no counterpart here.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.StorageLevel.MEMORY_AND_DISK.uses_disk
            True
            >>> bt.StorageLevel.parse("disk_only")
            <StorageLevel.DISK_ONLY: 'disk_only'>
    """

    #: Memory only; an evicted result is recomputed on the next terminal op.
    MEMORY_ONLY = "memory_only"
    #: Memory, demoting to the local spill tier on eviction rather than dropping.
    MEMORY_AND_DISK = "memory_and_disk"
    #: Disk only; never charged against the storage-memory envelope.
    DISK_ONLY = "disk_only"

    @property
    def uses_memory(self) -> bool:
        """Whether a result at this level may occupy the storage-memory envelope.

        Returns:
            `False` only for `DISK_ONLY`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.StorageLevel.DISK_ONLY.uses_memory
                False
        """
        return self is not StorageLevel.DISK_ONLY

    @property
    def uses_disk(self) -> bool:
        """Whether a result at this level may be written to the cache's disk tier.

        Returns:
            `False` only for `MEMORY_ONLY`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.StorageLevel.MEMORY_ONLY.uses_disk
                False
        """
        return self is not StorageLevel.MEMORY_ONLY

    @classmethod
    def parse(cls, value: StorageLevel | str | None) -> StorageLevel:
        """Coerce a user-supplied level to a `StorageLevel`.

        Accepts the enum itself, or its name in any case with either spelling of the
        separator (``"MEMORY_AND_DISK"``, ``"memory and disk"``). `None` resolves to
        `MEMORY_AND_DISK`, the default: a cache whose only response to a full budget
        is to forget is the one shape of cache that cannot help a workload larger
        than RAM, which is the workload that asked for a cache.

        Args:
            value: The level, its name, or `None` for the default.

        Returns:
            The resolved `StorageLevel`.

        Raises:
            PlanError: If `value` names no level. The message lists every accepted
                name, because the mistake this catches is almost always a Spark
                spelling (``MEMORY_ONLY_SER``) with no counterpart here.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.StorageLevel.parse(None)
                <StorageLevel.MEMORY_AND_DISK: 'memory_and_disk'>
                >>> bt.StorageLevel.parse("MEMORY ONLY")
                <StorageLevel.MEMORY_ONLY: 'memory_only'>
        """
        from batcher._internal.errors import PlanError

        if value is None:
            return cls.MEMORY_AND_DISK
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            for level in cls:
                if level.value == normalized:
                    return level
        names = ", ".join(repr(level.value) for level in cls)
        raise PlanError(
            f"unknown storage level {value!r}",
            hint=(
                f"pass one of {names} (or the bt.StorageLevel member). Spark's "
                "serialized (_SER) and replicated (_2) variants have no counterpart: "
                "the in-memory form is already Arrow, and replication is a property "
                "of the shared cache backend rather than of a level."
            ),
        )
