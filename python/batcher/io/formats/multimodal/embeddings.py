"""Embedding source — vector files (.npy / .parquet) → an Arrow embedding column.

Unlike the reference-only media sources, `EmbeddingSource` materializes vectors:
each ``.npy`` file (a 1-D vector or a 2-D ``(n, dim)`` matrix) or ``.parquet``
file (a single fixed-width list/array column) becomes rows of an
``embedding: FixedSizeList<float32, dim>`` column, alongside ``uri`` and ``row``
provenance columns. This is the format that pairs with a vector store (Lance):
embeddings land as a contiguous fixed-size-list column ready for ANN indexing.

CORE — only numpy and pyarrow are needed (no optional extra). All ``.npy`` /
``.parquet`` files under the path must share one embedding dimension.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher._internal.errors import IOError as BatcherIOError
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SOURCES

__all__ = ["EmbeddingSource"]

_SUFFIXES = (".npy", ".parquet")


@SOURCES.register("embeddings")
class EmbeddingSource:
    """Vector files as an Arrow ``FixedSizeList`` embedding column.

    Lists ``.npy`` / ``.parquet`` files under `path` and assembles one Arrow
    `RecordBatch` per file-batch with columns ``uri:string``, ``row:int64``, and
    ``embedding:fixed_size_list<float32, dim>``. The dimension is inferred from
    the first file and enforced across the rest.
    """

    __slots__ = ("_batch_files", "_dim", "_files_cache", "_fs", "_path")

    def __init__(self, path: str, *, batch_files: int = 64) -> None:
        if batch_files < 1:
            raise ValueError("batch_files must be >= 1")
        self._path = path
        self._batch_files = batch_files
        self._fs = resolve_filesystem(path)
        self._files_cache: list[str] | None = None
        self._dim: int | None = None

    def _files(self) -> list[str]:
        if self._files_cache is None:
            seen: dict[str, None] = {}
            for suffix in _SUFFIXES:
                try:
                    matches = self._fs.expand(self._path, suffix=suffix)
                except BatcherIOError:
                    continue
                for f in matches:
                    seen.setdefault(f, None)
            if not seen:
                raise BatcherIOError(f"no embedding files (.npy/.parquet) under {self._path!r}")
            self._files_cache = sorted(seen)
        return self._files_cache

    def _dimension(self) -> int:
        if self._dim is None:
            self._dim = self._file_vectors(self._files()[0]).shape[1]
        return self._dim

    def schema(self) -> pa.Schema:
        dim = self._dimension()
        return pa.schema(
            [
                pa.field("uri", pa.string()),
                pa.field("row", pa.int64()),
                pa.field("embedding", pa.list_(pa.float32(), dim)),
            ]
        )

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        dim = self._dimension()
        files = self._files()
        for start in range(0, len(files), self._batch_files):
            chunk = files[start : start + self._batch_files]
            batch = self._build_batch(chunk, dim)
            yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        # Total vectors requires reading each file's shape; cheap for .npy headers
        # but we keep row_count to the known file count's batches → unknown rows.
        return None

    def identity(self) -> str:
        return f"embeddings:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        from batcher.io.formats.multimodal.media import MediaSplit

        files = self._files()
        return [
            MediaSplit("embeddings", tuple(files[s : s + self._batch_files]), False)
            for s in range(0, len(files), self._batch_files)
        ]

    # ---- batch assembly ---------------------------------------------------
    def _build_batch(self, chunk: list[str], dim: int) -> pa.RecordBatch:
        import numpy as np

        uris: list[str] = []
        rows: list[int] = []
        mats: list[Any] = []
        for path in chunk:
            mat = self._file_vectors(path)
            if mat.shape[1] != dim:
                raise FormatError(
                    f"embedding dim mismatch in {path!r}: got {mat.shape[1]}, expected {dim}"
                )
            mats.append(mat)
            uris.extend([path] * mat.shape[0])
            rows.extend(range(mat.shape[0]))
        flat = np.ascontiguousarray(np.concatenate(mats, axis=0).reshape(-1), dtype=np.float32)
        values = pa.array(flat, pa.float32())
        embedding = pa.FixedSizeListArray.from_arrays(values, dim)
        return pa.RecordBatch.from_arrays(
            [pa.array(uris, pa.string()), pa.array(rows, pa.int64()), embedding],
            names=["uri", "row", "embedding"],
        )

    def _file_vectors(self, path: str) -> Any:
        """Load one file as a 2-D ``(n, dim)`` float array (1-D promoted to one row)."""
        import numpy as np

        if path.endswith(".npy"):
            with self._fs.open(path) as fh:
                arr = np.load(fh, allow_pickle=False)
        else:
            arr = self._parquet_vectors(path)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise FormatError(f"embedding file {path!r} must be 1-D or 2-D, got {arr.ndim}-D")
        return arr

    def _parquet_vectors(self, path: str) -> Any:
        """Read a single fixed-width list column from a Parquet file as a matrix."""
        import numpy as np
        import pyarrow.parquet as pq

        with self._fs.open(path) as fh:
            table = pq.read_table(fh)
        if table.num_columns != 1:
            raise FormatError(
                f"embedding parquet {path!r} must have exactly one (list) column, "
                f"got {table.num_columns}"
            )
        col = table.column(0).combine_chunks()
        if not pa.types.is_fixed_size_list(col.type) and not pa.types.is_list(col.type):
            raise FormatError(
                f"embedding parquet {path!r} column must be a list type, got {col.type}"
            )
        flat = np.asarray(col.values, dtype=np.float32)
        return flat.reshape(len(col), -1)
