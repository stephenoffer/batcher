"""Zarr format — chunked array read via `zarr`, chunk-parallel to Arrow.

`ZarrSource` reads one Zarr array and exposes *chunk-aligned* splits — one
`ZarrChunkSplit` per block of leading-axis chunks — so a distributed read pulls
only its chunks (Zarr's native parallelism unit). Each block becomes Arrow columns:
a 1-D array maps to a single ``value`` column; a 2-D array maps to one column per
trailing index (``c0``, ``c1``, …). Read-only; persist results as Parquet/Arrow.

All `zarr` imports are deferred — importing this module never requires the optional
dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[zarr]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split

__all__ = ["ZarrChunkSplit", "ZarrSource"]


def _require_zarr() -> Any:
    """Import and return the `zarr` module or raise `BackendError`."""
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError("Zarr support requires zarr: pip install 'batcher[zarr]'") from exc
    return zarr


def _slice_to_batch(array: Any, projection: list[str] | None) -> pa.RecordBatch:
    """Turn an in-memory numpy slice into an Arrow record batch."""
    if array.ndim == 1:
        data = {"value": array}
    else:
        data = {f"c{i}": array[:, i] for i in range(array.shape[1])}
    batch = pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in data.items()})
    return batch.select(projection) if projection is not None else batch


@dataclass(frozen=True, slots=True)
class ZarrChunkSplit:
    """A contiguous, chunk-aligned row-block of one Zarr array, read in isolation."""

    path: str
    start: int
    stop: int

    def _array(self) -> Any:
        zarr = _require_zarr()
        return zarr.open(self.path, mode="r")[self.start : self.stop]

    def schema(self) -> pa.Schema:
        return _slice_to_batch(self._array(), None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return [_slice_to_batch(self._array(), projection)]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return self.stop - self.start

    def identity(self) -> str:
        return f"zarr:{self.path}:{self.start}-{self.stop}"


@SOURCES.register("zarr")
class ZarrSource:
    """One Zarr array read to Arrow, split along chunk boundaries on the leading axis.

    Args:
        path: The Zarr store path or URI.
    """

    __slots__ = ("_path",)

    def __init__(self, path: str) -> None:
        self._path = path

    def _array(self) -> Any:
        zarr = _require_zarr()
        try:
            return zarr.open(self._path, mode="r")
        except Exception as exc:
            raise BackendError(f"failed to open Zarr array {self._path!r}: {exc}") from exc

    def schema(self) -> pa.Schema:
        return _slice_to_batch(self._array()[0:1], None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        batches: list[pa.RecordBatch] = []
        for split in self.splits():
            batches.extend(split.read(projection))
        return batches

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for split in self.splits():
            yield from split.iter_batches(projection)

    def row_count(self) -> int | None:
        return int(self._array().shape[0])

    def identity(self) -> str:
        return f"zarr:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        array = self._array()
        n = int(array.shape[0])
        chunk = int(array.chunks[0]) or n or 1
        starts = range(0, max(n, 1), chunk)
        return [ZarrChunkSplit(self._path, s, min(s + chunk, n)) for s in starts]
