"""The `Source` protocol — the contract every connector satisfies.

A `Source` knows its schema without reading data, and reads (optionally a column subset)
only when asked. This is the neutral interface the optimizer and executor program against;
the concrete connectors (in-memory, iterator, materialized, and the file formats under
`io/formats/`) live in `_impl` and the format modules. Split out of `_impl` so the contract
and its implementations each stay within the module-size budget.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import pyarrow as pa

from batcher.io.splits import Split

__all__ = ["Checkpointable", "Source", "is_checkpointable"]


@runtime_checkable
class Source(Protocol):
    """A lazily-readable relation.

    `bounded` (default ``True``) marks whether the source is finite. Unbounded
    sources (Kafka and other brokers, incremental file discovery) set it ``False``
    so terminal operations choose a streaming path and `collect()` refuses to
    materialize an infinite stream. Read it via `is_bounded` to honor the default.
    """

    bounded: bool

    def schema(self) -> pa.Schema:
        """The full schema of the source, without reading the data."""
        ...

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read the source, optionally only `projection` columns."""
        ...

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Yield record batches lazily (the streaming read path)."""
        ...

    def row_count(self) -> int | None:
        """The number of rows, if known cheaply without reading data (else None)."""
        ...

    def identity(self) -> str:
        """A stable identifier for this source (for keyed metadata/learning)."""
        ...

    # Optional (duck-typed via `source_statistics`): a connector may also expose
    #   def statistics(self) -> SourceStatistics | None
    # returning footer/manifest/catalog row counts and per-column min/max/null/ndv
    # known without scanning. Sources that don't implement it fall back to
    # `row_count()`. Not a required Protocol method so `runtime_checkable` still
    # accepts the many sources that predate it.

    def splits(self, target_size: int | None = None) -> list[Split]:
        """Independently-readable slices for distributed/parallel reads.

        A source that cannot subdivide returns a single `WholeSourceSplit`.
        """
        ...

    # Optional (duck-typed via `is_checkpointable`): a *replayable* streaming source
    # may also expose
    #   def snapshot_position(self) -> dict        # what it has read through
    #   def seek(self, position: dict) -> None     # resume from a recorded position
    # so a streaming query can checkpoint offsets and resume exactly-once after a
    # restart (Kafka offsets, Kinesis sequence numbers, a rate cursor). Not required
    # Protocol methods, so non-replayable sources are simply at-least-once.


@runtime_checkable
class Checkpointable(Protocol):
    """A streaming source whose read position can be snapshotted and resumed."""

    def snapshot_position(self) -> dict: ...

    def seek(self, position: dict) -> None: ...


def is_checkpointable(source: Source) -> bool:
    """Whether `source` supports offset snapshot/seek for exactly-once recovery."""
    return callable(getattr(source, "snapshot_position", None)) and callable(
        getattr(source, "seek", None)
    )
