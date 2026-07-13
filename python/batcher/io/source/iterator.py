"""`IteratorSource` — a streaming relation backed by a re-iterable batch factory.

The entry point for unbounded / larger-than-memory inputs: nothing is materialized,
so the schema is supplied up front and the factory is re-invoked for each read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pyarrow as pa

from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["IteratorSource"]


class IteratorSource:
    """A streaming relation backed by a re-iterable factory of record batches.

    `factory` is a zero-argument callable returning a *fresh* iterator of
    `pyarrow.RecordBatch` each time it is called (so the source can be read more
    than once, e.g. plan-build validation then execution). The schema must be
    supplied up front since the data is not materialized. This is the entry point
    for unbounded / larger-than-memory streaming inputs.

    `bounded` defaults to ``True`` (a finite generator); pass ``bounded=False`` for
    a genuinely unbounded stream so `collect()` refuses to materialize it.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io import IteratorSource
            >>> schema = pa.schema([("x", pa.int64())])
            >>> src = IteratorSource(lambda: iter([pa.record_batch({"x": [1, 2]})]), schema)
            >>> len(src.read())
            1
    """

    __slots__ = ("_bounded", "_factory", "_schema")

    def __init__(
        self,
        factory: Callable[[], Iterator[pa.RecordBatch]],
        schema: pa.Schema,
        *,
        bounded: bool = True,
    ) -> None:
        self._factory = factory
        self._schema = schema
        self._bounded = bounded

    @property
    def bounded(self) -> bool:
        """Whether the stream is finite, so a `collect()` would terminate.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64())])
                >>> IteratorSource(lambda: iter([]), schema, bounded=False).bounded
                False

        Returns:
            True for a finite generator (the default), False for a live stream.
        """
        return self._bounded

    def schema(self) -> pa.Schema:
        """The schema declared up front — the data is never scanned to infer it.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64())])
                >>> IteratorSource(lambda: iter([]), schema).schema().names
                ['x']

        Returns:
            The Arrow schema every batch the factory yields must conform to.
        """
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Drain a fresh iterator into a list of batches.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64()), ("y", pa.int64())])
                >>> batch = pa.record_batch({"x": [1], "y": [2]})
                >>> src = IteratorSource(lambda: iter([batch]), schema)
                >>> src.read(["x"])[0].num_columns
                1

        Args:
            projection: Columns to produce. All columns when omitted.

        Returns:
            Every batch the factory yields. Only meaningful for a bounded stream.
        """
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream a fresh iterator from the factory, batch by batch.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64())])
                >>> batch = pa.record_batch({"x": [1, 2, 3]})
                >>> src = IteratorSource(lambda: iter([batch]), schema)
                >>> next(src.iter_batches()).num_rows
                3

        Args:
            projection: Columns to produce. All columns when omitted.

        Returns:
            An iterator over the factory's batches.
        """
        for b in self._factory():
            yield b.select(projection) if projection is not None else b

    def row_count(self) -> int | None:
        """Always None — a stream's length is unknown, and possibly unbounded.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64())])
                >>> IteratorSource(lambda: iter([]), schema).row_count() is None
                True

        Returns:
            None. Counting would mean draining the stream.
        """
        return None  # streaming sources have unknown (possibly unbounded) length.

    def identity(self) -> str:
        """A schema-based key — a stream has no stable cross-run identity.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64())])
                >>> IteratorSource(lambda: iter([]), schema).identity()[:7]
                'stream:'

        Returns:
            A key the optimizer's shape-keyed metadata is stored under.
        """
        return f"stream:{self._schema}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One `WholeSourceSplit` — a generator cannot be sliced for parallel reads.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import IteratorSource
                >>> schema = pa.schema([("x", pa.int64())])
                >>> len(IteratorSource(lambda: iter([]), schema).splits())
                1

        Args:
            target_size: Ignored — a stream has no addressable slices.

        Returns:
            A single split wrapping this source.
        """
        return [WholeSourceSplit(self)]
