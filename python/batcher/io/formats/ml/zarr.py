"""Zarr format — chunked array read via `zarr`, chunk-parallel to Arrow.

`ZarrSource` reads one Zarr array and exposes *chunk-aligned* splits — one
`ZarrChunkSplit` per block of leading-axis chunks — so a distributed read pulls
only its chunks (Zarr's native parallelism unit). Each block becomes Arrow columns:
a 1-D array maps to a single ``value`` column; a 2-D array maps to one column per
trailing index (``c0``, ``c1``, …). Read-only; persist results as Parquet/Arrow.

All `zarr` imports are deferred — importing this module never requires the optional
dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[zarr]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.mathx import ceil_div
from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.ml._ndarray import schema_from_array_meta, slice_to_batch
from batcher.io.splits import Split
from batcher.plan.source_stats import SourceStatistics

__all__ = ["ZarrChunkSplit", "ZarrSource"]


def _require_zarr() -> Any:
    """Import and return the `zarr` module or raise `BackendError`."""
    return require("zarr", feature="Zarr", provides="zarr", extra="zarr")


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
        return slice_to_batch(self._array(), None).schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return [slice_to_batch(self._array(), projection)]

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
        return schema_from_array_meta(self._array())

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

    def statistics(self) -> SourceStatistics | None:
        """Exact row count and chunk count from the array's stored metadata — no data read.

        A Zarr array records its shape and chunk grid in ``.zarray`` metadata, so both
        the leading-axis row count (exact) and the number of leading-axis chunks (the
        granularity `splits` prunes and reads at) are free. No per-column bounds are
        advertised — Zarr v2 metadata carries no min/max, and an exact-looking guessed
        bound is worse than none.
        """
        try:
            array = self._array()
            n = int(array.shape[0])
            chunk = int(array.chunks[0]) or n or 1
        except Exception:
            return None
        groups = ceil_div(n, chunk) if n else 0  # ceil, matching `splits` block count
        return SourceStatistics(row_count=n, exact_rows=True, row_group_count=groups)

    def identity(self) -> str:
        return f"zarr:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        array = self._array()
        n = int(array.shape[0])
        chunk = int(array.chunks[0]) or n or 1
        starts = range(0, max(n, 1), chunk)
        return [ZarrChunkSplit(self._path, s, min(s + chunk, n)) for s in starts]
