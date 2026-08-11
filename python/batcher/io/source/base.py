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

__all__ = ["Checkpointable", "RateLimited", "Source", "is_checkpointable", "is_rate_limited"]


@runtime_checkable
class Source(Protocol):
    """A lazily-readable relation.

    `bounded` (default ``True``) marks whether the source is finite. Unbounded
    sources (Kafka and other brokers, incremental file discovery) set it ``False``
    so terminal operations choose a streaming path and `collect()` refuses to
    materialize an infinite stream. Read it via `is_bounded` to honor the default.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io import InMemorySource, Source
            >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
            >>> isinstance(src, Source)
            True
    """

    bounded: bool

    def schema(self) -> pa.Schema:
        """The full schema of the source, without reading the data.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1]})]).schema().names
                ['x']

        Returns:
            The Arrow schema every batch this source produces conforms to.
        """
        ...

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read the source, optionally only `projection` columns.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2], "y": [3, 4]})])
                >>> src.read(["x"])[0].num_columns
                1

        Args:
            projection: Columns the scan must produce. All columns when omitted;
                a columnar source reads only these.

        Returns:
            Every batch of the source, materialized.
        """
        ...

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Yield record batches lazily (the streaming read path).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> next(src.iter_batches()).num_rows
                3

        Args:
            projection: Columns the scan must produce. All columns when omitted.

        Returns:
            An iterator over the source's batches, read one at a time.
        """
        ...

    def row_count(self) -> int | None:
        """The number of rows, if known cheaply without reading data (else None).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1, 2, 3]})]).row_count()
                3

        Returns:
            The exact row count, or None when counting would cost a data scan.
        """
        ...

    def identity(self) -> str:
        """A stable identifier for this source (for keyed metadata/learning).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1]})]).identity()[:4]
                'mem:'

        Returns:
            A key the metadata hub stores learned statistics under.
        """
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

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> len(src.splits())
                1

        Args:
            target_size: Rough size (bytes) to aim for per split. The source
                chooses its own granularity when omitted.

        Returns:
            The splits covering the source exactly once.
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


@runtime_checkable
class RateLimited(Protocol):
    """A streaming source whose per-trigger admission can be narrowed while it runs.

    The seam a streaming rate controller acts through. A source that implements it can be
    told, before each micro-batch, how many rows that trigger may read; one that does not is
    simply never throttled, and its configured static cap continues to govern.

    An admission cap changes how much of a stream a trigger reads, never what the query
    computes from the rows it read, so honouring one can never change a result.
    """

    def set_admission_limit(self, max_rows: int | None) -> None: ...


def is_rate_limited(source: Source) -> bool:
    """Whether `source` accepts a per-trigger admission cap (`RateLimited`).

    Read through a callable check rather than `isinstance`, matching `is_checkpointable`
    beside it, so a duck-typed connector participates without inheriting anything.
    """
    return callable(getattr(source, "set_admission_limit", None))
