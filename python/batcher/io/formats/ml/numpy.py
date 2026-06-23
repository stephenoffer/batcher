"""NumPy ``.npy`` / ``.npz`` source — arrays as Arrow columns.

A 1-D array becomes a single ``data`` column; an ``(n, dim)`` array becomes a
``FixedSizeList`` column (the Ray Data ``read_numpy`` convention); a higher-rank
``(n, *shape)`` array becomes a fixed-shape-tensor column that preserves the full
per-row shape. ``.npz`` archives expose one column per stored array.
"""

from __future__ import annotations

from collections.abc import Iterator
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
        return _table_from_npy_handle(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        table = _table_from_npy_handle(fh)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

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
