"""HDF5 format — array-dataset read via `h5py`, sliced to Arrow.

`HDF5Source` reads one HDF5 dataset (a dense N-D array) and exposes *slice-level*
splits along the leading axis — one `HDF5SliceSplit` per row-block — so a
distributed read pulls only its slice via h5py's lazy indexing. Each slice becomes
Arrow columns: a 1-D dataset maps to a single ``value`` column; a 2-D dataset maps
to one column per trailing index (``c0``, ``c1``, …). Read-only; persist results as
Parquet/Arrow.

All `h5py` imports are deferred — importing this module never requires the optional
dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[hdf5]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split

__all__ = ["HDF5SliceSplit", "HDF5Source"]


def _require_h5py() -> Any:
    """Import and return the `h5py` module or raise `BackendError`."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError("HDF5 requires h5py: pip install 'batcher-engine[hdf5]'") from exc
    return h5py


def _slice_to_batch(array: Any, projection: list[str] | None) -> pa.RecordBatch:
    """Turn an in-memory numpy slice into an Arrow record batch."""
    if array.ndim == 1:
        data = {"value": array}
    else:
        data = {f"c{i}": array[:, i] for i in range(array.shape[1])}
    batch = pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in data.items()})
    return batch.select(projection) if projection is not None else batch


@dataclass(frozen=True, slots=True)
class HDF5SliceSplit:
    """A contiguous row-block of one HDF5 dataset, read in isolation."""

    path: str
    dataset: str
    start: int
    stop: int

    def _array(self) -> Any:
        h5py = _require_h5py()
        with h5py.File(self.path, "r") as handle:
            return handle[self.dataset][self.start : self.stop]

    def schema(self) -> pa.Schema:
        return _slice_to_batch(self._array(), None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return [_slice_to_batch(self._array(), projection)]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return self.stop - self.start

    def identity(self) -> str:
        return f"hdf5:{self.path}:{self.dataset}:{self.start}-{self.stop}"


@SOURCES.register("hdf5")
class HDF5Source:
    """One HDF5 dataset read to Arrow, sliced along its leading axis.

    Args:
        path: The ``.h5`` / ``.hdf5`` file.
        dataset: The dataset path within the file (e.g. ``"/data"``).
    """

    __slots__ = ("_dataset", "_path")

    def __init__(self, path: str, *, dataset: str) -> None:
        self._path = path
        self._dataset = dataset

    def _length(self) -> int:
        h5py = _require_h5py()
        with h5py.File(self._path, "r") as handle:
            return handle[self._dataset].shape[0]

    def schema(self) -> pa.Schema:
        h5py = _require_h5py()
        with h5py.File(self._path, "r") as handle:
            head = handle[self._dataset][0:1]
        return _slice_to_batch(head, None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        batches: list[pa.RecordBatch] = []
        for split in self.splits():
            batches.extend(split.read(projection))
        return batches

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for split in self.splits():
            yield from split.iter_batches(projection)

    def row_count(self) -> int | None:
        return self._length()

    def identity(self) -> str:
        return f"hdf5:{self._path}:{self._dataset}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        n = self._length()
        slice_rows = active_config().execution.morsel_rows
        return [
            HDF5SliceSplit(self._path, self._dataset, s, min(s + slice_rows, n))
            for s in range(0, max(n, 1), slice_rows)
        ]
