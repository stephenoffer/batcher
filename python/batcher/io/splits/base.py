"""The `Split` protocol and the whole-source fallback.

A `Split` is the unit of distributed read parallelism: it carries only *locators*
(a format name + path, a set of row-group ids, …) — never data — so it serializes
cheaply to a remote worker that then reads just its slice directly from storage.
Splits intentionally mirror the `Source` read surface (`schema`/`read`/
`iter_batches`/`row_count`/`identity`) so a worker treats a split exactly like a
source. A source that cannot subdivide advertises a single `WholeSourceSplit`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = ["Split", "WholeSourceSplit"]


@runtime_checkable
class Split(Protocol):
    """An independently-readable, picklable slice of a source.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io import InMemorySource, Split
            >>> split = InMemorySource([pa.record_batch({"x": [1, 2]})]).splits()[0]
            >>> isinstance(split, Split)
            True
    """

    def schema(self) -> pa.Schema:
        """The schema of the rows this split reads, without reading them.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> split = InMemorySource([pa.record_batch({"x": [1]})]).splits()[0]
                >>> split.schema().names
                ['x']

        Returns:
            The Arrow schema every batch this split produces conforms to.
        """
        ...

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read this slice, optionally only `projection` columns.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2], "y": [3, 4]})])
                >>> src.splits()[0].read(["x"])[0].num_columns
                1

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            The slice's batches, materialized.
        """
        ...

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream this slice batch by batch (the bounded-memory read path).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> split = InMemorySource([pa.record_batch({"x": [1, 2, 3]})]).splits()[0]
                >>> next(split.iter_batches()).num_rows
                3

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            An iterator over the slice's batches.
        """
        ...

    def row_count(self) -> int | None:
        """The rows in this slice, if known without reading data (else None).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1, 2]})]).splits()[0].row_count()
                2

        Returns:
            The row count, or None when counting would cost a data scan.
        """
        ...

    def identity(self) -> str:
        """A stable identifier for this slice (for keyed metadata/learning).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1]})]).splits()[0].identity()[:4]
                'mem:'

        Returns:
            A key that distinguishes this slice from its siblings.
        """
        ...


@dataclass(frozen=True, slots=True)
class WholeSourceSplit:
    """A non-subdividable source read as a single split.

    Holds the source object itself, so it is only as picklable as that source
    (fine for in-memory / iterator sources, which carry their own data/closure).
    File and table sources never use this — they emit locator-only splits.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io import InMemorySource, WholeSourceSplit
            >>> src = InMemorySource([pa.record_batch({"x": [1, 2]})])
            >>> isinstance(WholeSourceSplit(src), WholeSourceSplit)
            True
    """

    source: Source

    def schema(self) -> pa.Schema:
        """The wrapped source's schema.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource, WholeSourceSplit
                >>> src = InMemorySource([pa.record_batch({"x": [1]})])
                >>> WholeSourceSplit(src).schema().names
                ['x']

        Returns:
            The source's Arrow schema.
        """
        return self.source.schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read the whole source, pushing `predicate` down where it is supported.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource, WholeSourceSplit
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> WholeSourceSplit(src).read()[0].num_rows
                3

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: A filter the source may apply during the read. The engine
                re-checks it regardless, so ignoring it is still correct.

        Returns:
            The source's batches, materialized.
        """
        from batcher.io.source import read_source

        return read_source(self.source, projection, predicate)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the whole source batch by batch.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource, WholeSourceSplit
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> next(WholeSourceSplit(src).iter_batches()).num_rows
                3

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            An iterator over the source's batches.
        """
        return self.source.iter_batches(projection)

    def row_count(self) -> int | None:
        """The wrapped source's row count, if it knows one cheaply.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource, WholeSourceSplit
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2]})])
                >>> WholeSourceSplit(src).row_count()
                2

        Returns:
            The source's row count, or None when it is unknown.
        """
        return self.source.row_count()

    def identity(self) -> str:
        """The wrapped source's identity — the split covers all of it.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource, WholeSourceSplit
                >>> src = InMemorySource([pa.record_batch({"x": [1]})])
                >>> WholeSourceSplit(src).identity()[:4]
                'mem:'

        Returns:
            The source's stable identifier.
        """
        return self.source.identity()
