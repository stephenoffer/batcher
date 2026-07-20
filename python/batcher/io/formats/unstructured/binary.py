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
from batcher.io.formats.multimodal._batching import (
    pack_by_count_and_bytes,
    probe_sizes,
)
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["BinarySource"]

_SCHEMA = pa.schema(
    [
        ("uri", pa.string()),
        # 64-bit offsets — see the note in `multimodal/media.py`: a batch of whole files
        # overflows a 32-bit offset array at 2 GB total.
        ("bytes", pa.large_binary()),
        ("size", pa.int64()),
        ("mime", pa.string()),
    ]
)


# Default byte ceiling on one file-batch — see `multimodal/_batching`.
_DEFAULT_BATCH_BYTES = 256 << 20


@SOURCES.register("binary")
class BinarySource:
    """Whole files as binary rows, batched `batch_files` at a time.

    `suffix` narrows directory/glob discovery (default ``""`` matches all files).
    Each split reads one batch of files.
    """

    __slots__ = (
        "_batch_bytes",
        "_batch_files",
        "_chunk_cache",
        "_files_cache",
        "_fs",
        "_path",
        "_suffix",
    )

    def __init__(
        self,
        path: str,
        *,
        suffix: str = "",
        batch_files: int = 64,
        batch_bytes: int = _DEFAULT_BATCH_BYTES,
    ) -> None:
        self._path = path
        self._fs = resolve_filesystem(path)
        self._suffix = suffix
        self._batch_files = batch_files
        self._batch_bytes = batch_bytes
        self._files_cache: list[str] | None = None
        self._chunk_cache: list[list[str]] | None = None

    def _chunks(self) -> list[list[str]]:
        """The file-batches this source reads, bounded by both count and bytes.

        Whole-file blobs vary in size without limit, so a batch bounded only by file count
        is unbounded in memory. Shared by `iter_batches` and `splits` so a distributed read
        cuts the corpus exactly where a single-node one does.
        """
        if self._chunk_cache is None:
            files = self._files()
            self._chunk_cache = pack_by_count_and_bytes(
                files, probe_sizes(files, self._fs.size), self._batch_files, self._batch_bytes
            )
        return self._chunk_cache

    def _files(self) -> list[str]:
        if self._files_cache is None:
            self._files_cache = self._fs.expand(self._path, suffix=self._suffix)
        return self._files_cache

    def schema(self) -> pa.Schema:
        return _SCHEMA

    def _read_one(self, path: str) -> bytes:
        with self._fs.open(path) as fh:
            return fh.read()

    def _batch(self, files: list[str]) -> pa.RecordBatch:
        from batcher.io._concurrent import read_each_file

        # Read the chunk's files concurrently — a serial per-file open leaves a many-file
        # blob scan latency-bound on one connection (the same fix as the media reader).
        payloads = read_each_file(self._fs, files, lambda _fs, p: self._read_one(p))
        uris, blobs, sizes, mimes = [], [], [], []
        for f, data in zip(files, payloads, strict=True):
            uris.append(f)
            blobs.append(data)
            sizes.append(len(data))
            mimes.append(mimetypes.guess_type(f)[0] or "application/octet-stream")
        return pa.RecordBatch.from_arrays(
            [
                pa.array(uris, pa.string()),
                pa.array(blobs, pa.large_binary()),
                pa.array(sizes, pa.int64()),
                pa.array(mimes, pa.string()),
            ],
            names=["uri", "bytes", "size", "mime"],
        )

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for chunk in self._chunks():
            batch = self._batch(chunk)
            yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        return len(self._files())

    def identity(self) -> str:
        return f"binary:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:
        """One split per file-batch.

        Args:
            target_size: Rough bytes to aim for per split, overriding `batch_bytes` for
                this call. Previously ignored, which made a split a fixed *file count* —
                so a split of videos could outweigh a split of thumbnails a thousandfold.
        """
        if target_size is not None and target_size != self._batch_bytes:
            files = self._files()
            chunks = pack_by_count_and_bytes(
                files, probe_sizes(files, self._fs.size), self._batch_files, target_size
            )
        else:
            chunks = self._chunks()
        out: list[Split] = []
        for chunk in chunks:
            src = BinarySource(
                chunk[0],
                suffix=self._suffix,
                batch_files=self._batch_files,
                batch_bytes=self._batch_bytes,
            )
            src._files_cache = chunk  # this split reads exactly its file chunk
            out.append(WholeSourceSplit(src))
        return out or [WholeSourceSplit(self)]
