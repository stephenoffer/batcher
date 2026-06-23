"""Binary-blob source — whole files as ``{uri, bytes, size, mime}`` rows.

The substrate for unstructured and multimodal data: each file becomes one row
with its raw bytes and cheap, header-free metadata. Multimodal sources extend
this shape with header-only media metadata; decoding to pixels/tensors is a
downstream Rust expression, never done at read time.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterator

import pyarrow as pa

from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["BinarySource"]

_SCHEMA = pa.schema(
    [
        ("uri", pa.string()),
        ("bytes", pa.binary()),
        ("size", pa.int64()),
        ("mime", pa.string()),
    ]
)


@SOURCES.register("binary")
class BinarySource:
    """Whole files as binary rows, batched `batch_files` at a time.

    `suffix` narrows directory/glob discovery (default ``""`` matches all files).
    Each split reads one batch of files.
    """

    __slots__ = ("_batch_files", "_files_cache", "_fs", "_path", "_suffix")

    def __init__(self, path: str, *, suffix: str = "", batch_files: int = 64) -> None:
        self._path = path
        self._fs = resolve_filesystem(path)
        self._suffix = suffix
        self._batch_files = batch_files
        self._files_cache: list[str] | None = None

    def _files(self) -> list[str]:
        if self._files_cache is None:
            self._files_cache = self._fs.expand(self._path, suffix=self._suffix)
        return self._files_cache

    def schema(self) -> pa.Schema:
        return _SCHEMA

    def _batch(self, files: list[str]) -> pa.RecordBatch:
        uris, blobs, sizes, mimes = [], [], [], []
        for f in files:
            with self._fs.open(f) as fh:
                data = fh.read()
            uris.append(f)
            blobs.append(data)
            sizes.append(len(data))
            mimes.append(mimetypes.guess_type(f)[0] or "application/octet-stream")
        return pa.RecordBatch.from_arrays(
            [
                pa.array(uris, pa.string()),
                pa.array(blobs, pa.binary()),
                pa.array(sizes, pa.int64()),
                pa.array(mimes, pa.string()),
            ],
            names=["uri", "bytes", "size", "mime"],
        )

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        files = self._files()
        for i in range(0, len(files), self._batch_files):
            batch = self._batch(files[i : i + self._batch_files])
            yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        return len(self._files())

    def identity(self) -> str:
        return f"binary:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        files = self._files()
        out: list[Split] = []
        for i in range(0, len(files), self._batch_files):
            chunk = files[i : i + self._batch_files]
            src = BinarySource(chunk[0], suffix=self._suffix, batch_files=self._batch_files)
            src._files_cache = chunk  # this split reads exactly its file chunk
            out.append(WholeSourceSplit(src))
        return out or [WholeSourceSplit(self)]
