"""NumPy ``.npy`` / ``.npz`` source — arrays as Arrow columns.

A 1-D array becomes a single ``data`` column; an ``(n, dim)`` array becomes a
``FixedSizeList`` column (the Ray Data ``read_numpy`` convention); a higher-rank
``(n, *shape)`` array becomes a fixed-shape-tensor column that preserves the full
per-row shape. ``.npz`` archives expose one column per stored array.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES
from batcher.plan.source_stats import SourceStatistics

__all__ = ["NumpySource"]


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # numpy is near-ubiquitous but kept optional/deferred
        raise BackendError("reading NumPy files needs numpy: pip install numpy") from exc
    return np


def _array_to_arrow(arr: Any) -> pa.Array:
    if arr.ndim == 1:
        return pa.array(arr)
    if arr.ndim == 2:
        width = int(arr.shape[1])
        flat = pa.array(arr.reshape(-1))
        return pa.FixedSizeListArray.from_arrays(flat, width)
    # Rank >= 2 per row: keep the full shape as a fixed-shape-tensor column.
    from batcher.io.formats.ml.tensor import to_tensor_column

    return to_tensor_column(arr)


def _header_schema(fh: IO[Any]) -> pa.Schema | None:
    """The Arrow schema of a ``.npy`` file from its header alone, or None if unreadable.

    The type comes from one synthetic row pushed through `_array_to_arrow` — the *same*
    mapping the read uses — so the advertised schema cannot drift from the batches. A
    dtype the mapping cannot handle simply returns None and the caller loads the file.
    """
    from batcher.io.stats.free_counts import npy_header_shape_dtype

    header = npy_header_shape_dtype(fh)
    if header is None:
        return None
    shape, dtype = header
    np = _np()
    try:
        probe = np.zeros((1, *shape[1:]), dtype=dtype)
        return pa.schema([pa.field("data", _array_to_arrow(probe).type)])
    except Exception:
        return None


def _table_from_npy_handle(fh: IO[Any]) -> pa.Table:
    np = _np()
    loaded = np.load(fh, allow_pickle=False)
    if hasattr(loaded, "files"):  # .npz archive
        return pa.table({k: _array_to_arrow(loaded[k]) for k in loaded.files})
    return pa.table({"data": _array_to_arrow(loaded)})


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
