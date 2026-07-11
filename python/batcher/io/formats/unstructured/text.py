"""Plain-text source — one row per line or one row per whole file."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["TextSource"]

_LINE_SCHEMA = pa.schema(
    [("path", pa.string()), ("line_number", pa.int64()), ("text", pa.string())]
)
_FILE_SCHEMA = pa.schema([("path", pa.string()), ("text", pa.string())])

# Files read concurrently per streaming chunk (bounds memory to one chunk of whole files).
_TEXT_READ_CHUNK = 64


@SOURCES.register("text")
class TextSource:
    """Text files as rows. `mode="line"` → one row per line (with `line_number`);
    `mode="file"` → one row per whole file. Each split is a single file.
    """

    __slots__ = ("_encoding", "_files_cache", "_fs", "_mode", "_path")

    def __init__(self, path: str, *, mode: str = "line", encoding: str = "utf-8") -> None:
        if mode not in ("line", "file"):
            raise FormatError(f"TextSource mode must be 'line' or 'file', got {mode!r}")
        self._path = path
        self._fs = resolve_filesystem(path)
        self._mode = mode
        self._encoding = encoding
        self._files_cache: list[str] | None = None

    def _files(self) -> list[str]:
        if self._files_cache is None:
            self._files_cache = self._fs.expand(self._path, suffix=".txt")
        return self._files_cache

    def schema(self) -> pa.Schema:
        return _LINE_SCHEMA if self._mode == "line" else _FILE_SCHEMA

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def _read_text(self, path: str) -> str:
        with self._fs.open(path) as fh:
            return fh.read().decode(self._encoding)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        from batcher.io._concurrent import read_each_file

        files = self._files()
        # Read files a chunk at a time, each chunk concurrently, so a many-file text scan
        # isn't one serial open-per-file on a high-latency store — while staying streaming
        # (memory bounded to one chunk) and yielding in file order.
        for start in range(0, len(files), _TEXT_READ_CHUNK):
            chunk = files[start : start + _TEXT_READ_CHUNK]
            texts = read_each_file(self._fs, chunk, lambda _fs, p: self._read_text(p))
            for f, data in zip(chunk, texts, strict=True):
                yield self._build_batch(f, data, projection)

    def _build_batch(self, f: str, data: str, projection: list[str] | None) -> pa.RecordBatch:
        if self._mode == "line":
            lines = data.splitlines()
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array([f] * len(lines), pa.string()),
                    pa.array(range(1, len(lines) + 1), pa.int64()),
                    pa.array(lines, pa.string()),
                ],
                names=["path", "line_number", "text"],
            )
        else:
            batch = pa.RecordBatch.from_arrays(
                [pa.array([f], pa.string()), pa.array([data], pa.string())],
                names=["path", "text"],
            )
        return batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"text:{self._mode}:{self._path}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [
            WholeSourceSplit(TextSource(f, mode=self._mode, encoding=self._encoding))
            for f in self._files()
        ]
