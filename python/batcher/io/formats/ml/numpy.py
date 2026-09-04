"""NumPy ``.npy`` / ``.npz`` source — arrays as Arrow columns.

A 1-D array becomes a single ``data`` column; an ``(n, dim)`` array becomes a
``FixedSizeList`` column (the Ray Data ``read_numpy`` convention); a higher-rank
``(n, *shape)`` array becomes a fixed-shape-tensor column that preserves the full
per-row shape. ``.npz`` archives expose one column per stored array.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES
from batcher.plan.source_stats import SourceStatistics

__all__ = ["NumpySource"]

# Bytes of array data a streamed chunk may hold. Bounds the read window; the batches
# handed on are re-cut to the configured morsel by `FileSource._normalize`, so this is
# about resident memory rather than about batch shape.
_CHUNK_BYTES = max(1 << 20, int(os.environ.get("BATCHER_NUMPY_CHUNK_BYTES", str(64 << 20))))


def _np() -> Any:
    """The numpy module, deferred so `import batcher` never pulls it.

    Deferred rather than module-scope because importing `batcher` eagerly imports every IO
    format so each can self-register, and numpy is not a core dependency — a module-scope
    import here made the whole package, and the autodoc build that imports it, fail without
    numpy installed.
    """
    return require("numpy", feature="Reading NumPy files", provides="numpy", extra="numpy")


def _array_to_arrow(arr: Any) -> pa.Array:
    """One `.npy` array as an Arrow column, by the engine's single set of rank rules.

    This used to restate those rules — 1-D scalar, ``(n, dim)`` fixed-size list, deeper a
    tensor column — which `io.interop.numpy_to_column` already owned for `bt.from_numpy`,
    `from_torch` and `from_tf`. The two were byte-identical on every shape either handled,
    checked before the swap rather than assumed, and had already drifted on the shapes only
    one of them did: a complex or 0-d array produced a pyarrow dtype *number* here where the
    same array in memory named the conversion that fixes it.

    Delegating removes the second copy rather than syncing it, which is the only way two
    statements of the same rule stay equal.

    The mask handling `numpy_to_column` also brings is inert on this path, and deliberately
    left unmentioned elsewhere: NumPy's own format cannot carry a mask. ``np.save`` refuses a
    masked array outright, and ``np.savez`` accepts one but stores only its data, so a
    ``.npy``/``.npz`` file never holds a mask for this reader to keep.
    """
    from batcher.io.interop import numpy_to_column

    return numpy_to_column(arr)


def _header_schema(fh: IO[Any]) -> pa.Schema | None:
    """The Arrow schema of a ``.npy`` file from its header alone, or None if unreadable.

    The type comes from one synthetic row pushed through `_columns_for_array` — the *same*
    mapping the read uses — so the advertised schema cannot drift from the batches. A
    dtype the mapping cannot handle simply returns None and the caller loads the file.

    Going through the whole mapping rather than its single-column tail is what keeps that
    promise for a compound dtype: a structured file's rows are its *fields*, so a schema
    built here as one ``data`` column described a table the read would never produce, and
    the mismatch surfaced as ``KeyError: 'Field "data" does not exist in schema'`` rather
    than as anything about the file.
    """
    from batcher.io.stats.free_counts import npy_header_shape_dtype

    header = npy_header_shape_dtype(fh)
    if header is None:
        return None
    shape, dtype = header
    np = _np()
    try:
        probe = np.zeros((1, *shape[1:]), dtype=dtype)
        columns = _columns_for_array(probe)
        return pa.schema([pa.field(name, col.type) for name, col in columns.items()])
    except Exception:
        return None


def _table_from_npy_handle(fh: IO[Any]) -> pa.Table:
    np = _np()
    loaded = np.load(fh, allow_pickle=False)
    if hasattr(loaded, "files"):  # .npz archive: each member is one named column
        return pa.table({k: _array_to_arrow(loaded[k]) for k in loaded.files})
    return pa.table(_columns_for_array(loaded))


def _columns_for_array(arr: Any) -> dict[str, pa.Array]:
    """The columns a single stored array becomes: its fields if compound, else ``data``.

    A **structured** ``.npy`` is a saved record array — what ``np.rec.array`` and
    ``np.genfromtxt`` produce, and what most scientific binary dumps are — so it is a table,
    and becomes one column per field: the same reading `bt.from_numpy` gives it. It used to
    reach Arrow whole and fail the file with ``Unsupported numpy type 20``, a message naming
    neither the fields it held nor the fact that they were the columns being asked for.

    Only the single-array file expands. An ``.npz`` member keeps its archive key as its
    column name, because two structured members could name the same field and expanding them
    would silently drop one; a structured member there raises through `_array_to_arrow`
    instead, which now names the type and the fix.
    """
    from batcher.io.interop import structured_to_columns

    fields = structured_to_columns(arr)
    return fields if fields is not None else {"data": _array_to_arrow(arr)}


@SOURCES.register("numpy")
class NumpySource(FileSource):
    """One or more ``.npy``/``.npz`` files (single file, directory, or glob)."""

    suffix = ".npy"
    format_name = "numpy"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        """Type the file from its header when it has one, rather than by loading it.

        `schema()` is called eagerly at `bt.read.numpy(...)` — before an operator is
        declared, let alone executed — and this used to answer it by loading the whole
        array and converting it to Arrow, then keeping the column names and discarding
        everything else. Constructing a reader over a 200 GB `.npy` therefore had to hold
        200 GB (twice, across the conversion) to learn that its one column is called
        ``data``. The header already records `shape` and `dtype`, which is the entire
        input to the type mapping.

        An ``.npz`` archive has no single header here, and a header that will not parse is
        a file this cannot type, so both fall back to the full load — slow and right,
        never fast and wrong.
        """
        pos = fh.tell()
        schema = _header_schema(fh)
        if schema is not None:
            return schema
        fh.seek(pos)
        return _table_from_npy_handle(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        table = _table_from_npy_handle(fh)
        if projection is not None:
            table = table.select(projection)
        # One array is one Arrow chunk, so this is a single RecordBatch of however many
        # rows the file holds. `FileSource._normalize` cuts it to the configured morsel on
        # the way out, for every format at once — capping it a second time here would be
        # the same rule in two places, which is how the two drift.
        return table.to_batches()

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream a ``.npy`` in row chunks off a memory map, rather than loading it whole.

        Without this the base falls back to `_read_file`, which calls `np.load` — so the
        entire array is resident before the first row reaches the consumer, and resident
        *again* as Arrow during the conversion. Measured on a 1.57 GB array: **1,545 MB**
        peak and **1.58 s** before the first batch, for a read that then emitted 12,000
        morsels. `_read_schema` already documents this hazard for the schema path ("a 200 GB
        `.npy` therefore had to hold 200 GB"); the read itself still paid it.

        A memory map turns that into a bounded window: each chunk is copied out of the map,
        converted, and released. The map only helps where the bytes are addressable as a
        local file, so a remote path — and an ``.npz`` archive, which is a zip and has no
        array to map — falls back to the whole-file load, exactly as before.

        Args:
            path: The array file to stream.
            projection: Columns the scan must produce. All columns when omitted.

        Yields:
            One `RecordBatch` per chunk of rows, in file order.
        """
        array = self._mapped(path)
        if array is None:
            yield from super()._iter_file(path, projection)
            return
        np = _np()
        row_bytes = max(1, int(array.dtype.itemsize) * int(np.prod(array.shape[1:], dtype=int)))
        rows = max(1, _CHUNK_BYTES // row_bytes)
        for start in range(0, len(array), rows):
            # `ascontiguousarray` copies the window out of the map, so the batch does not
            # keep the mapping alive and the next chunk's pages can be reclaimed.
            table = pa.table(
                {"data": _array_to_arrow(np.ascontiguousarray(array[start : start + rows]))}
            )
            if projection is not None:
                table = table.select(projection)
            yield from table.to_batches()

    def _mapped(self, path: str) -> Any | None:
        """`path` as a memory-mapped array, or None when it cannot be mapped.

        Best-effort by design: anything that is not a plain local ``.npy`` — an ``.npz``
        archive, an object-store path, a header this numpy will not map — returns None and
        the caller reads the file the original way rather than failing.
        """
        from batcher.io._concurrent import is_local_path

        if not is_local_path(path) or not path.endswith(".npy"):
            return None
        try:
            local = self._fs._p(path) if hasattr(self._fs, "_p") else path
            return _np().load(local, mmap_mode="r", allow_pickle=False)
        except Exception:
            return None

    def _file_row_count(self, path: str) -> int | None:
        from batcher.io.stats.free_counts import npy_header_rows

        try:
            with self._fs.open(path) as fh:
                return npy_header_rows(fh)
        except Exception:
            return None

    def statistics(self) -> SourceStatistics | None:
        """Exact row count from ``.npy`` headers (leading axis), no array load."""
        from batcher.io.stats import numpy_statistics

        try:
            return numpy_statistics(self._fs, self._files())
        except Exception:
            return None
