"""`InMemorySource` — a relation already materialized as Arrow record batches.

The source `from_pydict`/`from_arrow` produce. Because the batches are immutable and
already resident, its statistics are EXACT and computed at most once: the learned-metadata
moat starts here, since a repeat `MIN`/`MAX`/`COUNT(*) WHERE …` can be answered from
metadata without touching the data again. The heavy column passes live in `inmemory_stats`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["InMemorySource"]

# Overflow-safe narrow → wide numeric widenings the engine normalizes to at the FFI
# boundary anyway (Int8/16/32, UInt8/16/32 → Int64; Float16/32 → Float64). Doing it once
# here — instead of the Rust boundary re-casting on *every* query — turns a per-query
# O(rows) `cast` (arrow's int16→int64 cast is ~47 ms / 10 M rows) into a one-time cost.
# UInt64 (can overflow Int64), dictionaries, and everything else are left to the Rust
# `normalize_batch`, which stays the correctness backstop (a no-op on already-wide cols).
_WIDEN_NARROW: dict[pa.DataType, pa.DataType] = {
    pa.int8(): pa.int64(),
    pa.int16(): pa.int64(),
    pa.int32(): pa.int64(),
    pa.uint8(): pa.int64(),
    pa.uint16(): pa.int64(),
    pa.uint32(): pa.int64(),
    pa.float16(): pa.float64(),
    pa.float32(): pa.float64(),
}


def _widen_schema(schema: pa.Schema, targets: dict[str, pa.DataType]) -> pa.Schema:
    """`schema` with each `targets` column retyped to its widened type (metadata only)."""
    return pa.schema([f.with_type(targets[f.name]) if f.name in targets else f for f in schema])


class InMemorySource:
    """A relation already materialized as Arrow record batches.

    Narrow numeric columns (Int8/16/32, UInt8/16/32, Float16/32) are widened to
    Int64/Float64 — the types the engine normalizes to at the FFI boundary anyway — but
    **lazily and per column, with caching**: only the columns a query actually reads are
    cast, and the cast happens once (arrow's int16→int64 cast is ~47 ms / 10 M rows, and
    the Rust boundary would otherwise redo it on *every* query). This is skipped when
    ``shrink_output_dtypes`` is on — that opt-in path re-narrows pass-through outputs from
    the source widths, which pre-widening would erase — so that path keeps the Rust
    boundary as the (correctness-equivalent) fallback.

    `zone_maps=False` drops the O(rows) column-bounds pass in `statistics()`, leaving the
    exact row count. Pass it for a relation the engine produced and consumes exactly once (an
    adaptive stage boundary), whose bounds would be rebuilt and discarded every run.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io import InMemorySource
            >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
            >>> src.row_count()
            3
    """

    __slots__ = (
        "_batches",
        "_bounds_cache",
        "_cache",
        "_mean_cache",
        "_ndv_cache",
        "_schema",
        "_stats",
        "_sum_cache",
        "_targets",
        "_valuecount_cache",
        "_zone_maps",
    )
    bounded = True
    # The batches are already in RAM, so a statistics pass over them costs no I/O. The
    # conductor reads this before sketching cold-start distinct counts on the query path
    # (`api.terminal._metadata.seed_column_ndv`); a file-backed source leaves it False and
    # learns its ndv from the batches a run already scanned, rather than re-reading them.
    resident = True
    # `identity()` is shape-based for in-memory data (see `identity`), so two *different*
    # in-memory relations sharing a schema+size collide on it. Fine for the optimizer's shape
    # keys, but NOT for the cross-query source-stats cache, whose entries (row count, column
    # min/max) depend on the actual data — so in-memory stats are never stored under the
    # shared identity; they are memoized per instance in `statistics()`.
    stable_stats_identity = False

    def __init__(self, batches: list[pa.RecordBatch], *, zone_maps: bool = True) -> None:
        if not batches:
            raise ValueError("InMemorySource requires at least one record batch")
        from batcher.config import active_config

        self._batches = batches
        self._zone_maps = zone_maps
        src_schema = batches[0].schema
        if active_config().execution.shrink_output_dtypes:
            self._targets: dict[str, pa.DataType] = {}
            self._schema = src_schema
        else:
            self._targets = {f.name: t for f in src_schema if (t := _WIDEN_NARROW.get(f.type))}
            self._schema = _widen_schema(src_schema, self._targets) if self._targets else src_schema
        self._cache: dict[tuple[int, str], pa.Array] = {}
        self._stats: object | None = None
        self._ndv_cache: dict[str, int | None] = {}
        self._mean_cache: dict[str, float | None] = {}
        self._sum_cache: dict[str, float | int | None] = {}
        self._valuecount_cache: dict[tuple[str, str, object], int | None] = {}
        self._bounds_cache: dict[str, object] = {}

    def schema(self) -> pa.Schema:
        """The batches' schema, with narrow numeric columns widened.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1]})]).schema().names
                ['x']

        Returns:
            The Arrow schema every batch this source produces conforms to.
        """
        return self._schema

    def _build_column(self, name: str) -> pa.ChunkedArray | pa.Array:
        """Column `name` across all batches as one (narrow-int widened) Arrow column."""
        chunks = [self._widened(bi, name, b.column(name)) for bi, b in enumerate(self._batches)]
        return pa.chunked_array(chunks) if len(chunks) > 1 else chunks[0]

    def column_ndv(self, name: str) -> int | None:
        """EXACT distinct count of `name`'s non-null values, computed once and cached.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 1, 2]})])
                >>> src.column_ndv("x")
                2

        Args:
            name: The column to count distinct values of.

        Returns:
            The exact distinct count, or None for a type it cannot count.
        """
        if name not in self._ndv_cache:
            from batcher.io.source import inmemory_stats

            self._ndv_cache[name] = inmemory_stats.column_ndv(self._build_column, name)
        return self._ndv_cache[name]

    def column_mean(self, name: str) -> float | None:
        """EXACT average of `name`'s non-null values, computed once and cached.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> src.column_mean("x")
                2.0

        Args:
            name: The column to average.

        Returns:
            The exact mean, or None for a non-numeric or all-null column.
        """
        if name not in self._mean_cache:
            from batcher.io.source import inmemory_stats

            self._mean_cache[name] = inmemory_stats.column_mean(self._build_column, name)
        return self._mean_cache[name]

    def column_sum(self, name: str) -> float | int | None:
        """EXACT total of `name`'s non-null values (exactly representable), else None.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> src.column_sum("x")
                6

        Args:
            name: The column to total.

        Returns:
            The exact sum, or None when it is not exactly representable.
        """
        if name not in self._sum_cache:
            from batcher.io.source import inmemory_stats

            self._sum_cache[name] = inmemory_stats.column_sum(self._build_column, name)
        return self._sum_cache[name]

    def column_bounds(self, name: str):
        """EXACT `ColumnStat` (min/max/null-count) of `name`, computed once and cached.

        The single-column form of `statistics()`: the conductor requests bounds only for
        the columns a query's predicate references, so a filter over one column of a wide
        relation scans that column alone instead of every column (see
        `api.source_stats.collect_source_stats`).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [3, 1, 2]})])
                >>> src.column_bounds("x").min
                1

        Args:
            name: The column to bound.

        Returns:
            The column's exact min/max/null-count, or None for a non-ordered /
            all-null column (exactly as `statistics()` skips it).
        """
        if name not in self._bounds_cache:
            from batcher.io.source import inmemory_stats

            field = self._schema.field(name)
            self._bounds_cache[name] = inmemory_stats.column_bounds(
                self._build_column, field.type, name
            )
        return self._bounds_cache[name]

    def column_predicate_count(self, op: str, name: str, value: object) -> int | None:
        """EXACT surviving count of ``name <op> value`` (nulls excluded), cached per key.

        Lets a repeat ``COUNT(*) WHERE col <op> v`` be answered from metadata — the
        learned-metadata moat for any single-column comparison filter.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> src.column_predicate_count("gt", "x", 1)
                2

        Args:
            op: The comparison — one of ``eq``/``ne``/``lt``/``le``/``gt``/``ge``.
            name: The column being compared.
            value: The literal it is compared against.

        Returns:
            The exact number of surviving rows, or None when the comparison
            cannot be evaluated on this column's type.
        """
        key = (op, name, value)
        if key not in self._valuecount_cache:
            from batcher.io.source import inmemory_stats

            self._valuecount_cache[key] = inmemory_stats.column_predicate_count(
                self._build_column, op, name, value
            )
        return self._valuecount_cache[key]

    def statistics(self):
        """EXACT per-column min/max/null-count over the ordered columns, computed once.

        An in-memory relation is immutable, so its column bounds are exact and constant; the
        one vectorized pass (memoized per instance) lets Kyber answer an unfiltered
        ``MIN``/``MAX`` from metadata on every subsequent run — the learned-metadata moat a
        static optimizer can't match. See `io.inmemory_stats` for the computation. The pass
        is O(rows), so it pays for itself only on a relation queried more than once;
        `zone_maps=False` reports the row count alone (see the constructor).

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> src.statistics().row_count
                3

        Returns:
            The relation's exact row count and per-column bounds.
        """
        if self._stats is None:
            rows = sum(b.num_rows for b in self._batches)
            if not self._zone_maps:
                return SourceStatistics(row_count=rows)
            from batcher.io.source import inmemory_stats

            self._stats = inmemory_stats.statistics(self._build_column, self._schema, rows)
        return self._stats

    def _widened(self, bi: int, name: str, col: pa.Array) -> pa.Array:
        """The widened (and cached) form of column `name` in batch `bi`, or `col` as-is."""
        target = self._targets.get(name)
        if target is None:
            return col
        key = (bi, name)
        arr = self._cache.get(key)
        if arr is None:
            import pyarrow.compute as pc

            arr = pc.cast(col, target)
            self._cache[key] = arr
        return arr

    def _project(self, bi: int, b: pa.RecordBatch, projection: list[str] | None) -> pa.RecordBatch:
        if projection is None and not self._targets:
            return b
        names = projection if projection is not None else b.schema.names
        selected = b.select(names)
        if not self._targets:
            return selected
        arrays = [self._widened(bi, n, selected.column(j)) for j, n in enumerate(names)]
        return pa.RecordBatch.from_arrays(
            arrays, schema=_widen_schema(selected.schema, self._targets)
        )

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Return the resident batches, projected (and widened) as asked.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2], "y": [3, 4]})])
                >>> src.read(["x"])[0].num_columns
                1

        Args:
            projection: Columns to produce. All columns when omitted.

        Returns:
            The source's batches. No copy is made beyond the projection itself.
        """
        return [self._project(i, b, projection) for i, b in enumerate(self._batches)]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Yield the resident batches one at a time.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> src = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
                >>> next(src.iter_batches()).num_rows
                3

        Args:
            projection: Columns to produce. All columns when omitted.

        Returns:
            An iterator over the source's batches.
        """
        for i, b in enumerate(self._batches):
            yield self._project(i, b, projection)

    def row_count(self) -> int | None:
        """The exact number of resident rows.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1, 2, 3]})]).row_count()
                3

        Returns:
            The summed row count of every batch.
        """
        return sum(b.num_rows for b in self._batches)

    def identity(self) -> str:
        """A shape-based key (schema + size) — in-memory data has no cross-run identity.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> InMemorySource([pa.record_batch({"x": [1]})]).identity()[:4]
                'mem:'

        Returns:
            A key the optimizer's shape-keyed metadata is stored under. Two
            different relations of the same schema and size collide on it, which
            is why `stable_stats_identity` is False.
        """
        # In-memory data has no stable cross-run identity; key by schema + size.
        return f"mem:{self._schema}:{self.row_count()}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One `WholeSourceSplit` — resident batches are not re-partitioned for reading.

        Examples:
            .. doctest::

                >>> import pyarrow as pa
                >>> from batcher.io import InMemorySource
                >>> len(InMemorySource([pa.record_batch({"x": [1]})]).splits())
                1

        Args:
            target_size: Ignored — the batches are already in memory, so there is
                nothing to size a read against.

        Returns:
            A single split wrapping this source.
        """
        return [WholeSourceSplit(self)]
