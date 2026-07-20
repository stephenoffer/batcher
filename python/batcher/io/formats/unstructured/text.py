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
# Bytes pulled from a file per read in line mode. Line mode is the one shape where a
# single *file* can be arbitrarily large — a multi-GB log — so it reads in blocks rather
# than whole, and holds one block plus one batch instead of the whole decoded file plus a
# Python list of every line.
_TEXT_BLOCK_BYTES = 1 << 20  # 1 MiB
# Lines per emitted batch in line mode.
_TEXT_LINES_PER_BATCH = 16_384


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
        if self._mode == "line":
            # One file can be arbitrarily large in line mode, so read it in blocks rather
            # than whole. The whole-file path below would hold the decoded text *and* a
            # Python list of every line *and* one batch of them all at once — three copies
            # of a multi-GB log.
            for f in files:
                yield from self._iter_line_batches(f, projection)
            return
        # File mode: a row is a whole file, so the file is resident either way. Read a
        # chunk of files concurrently so a many-file scan isn't one serial open per file
        # on a high-latency store, while staying bounded to one chunk and in file order.
        for start in range(0, len(files), _TEXT_READ_CHUNK):
            chunk = files[start : start + _TEXT_READ_CHUNK]
            texts = read_each_file(self._fs, chunk, lambda _fs, p: self._read_text(p))
            for f, data in zip(chunk, texts, strict=True):
                yield self._build_batch(f, data, projection)

    def _iter_line_batches(
        self, path: str, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        """Stream one file's lines, in batches, without ever holding it whole."""
        lineno = 0
        lines: list[str] = []
        for line in self._iter_lines(path):
            lines.append(line)
            if len(lines) >= _TEXT_LINES_PER_BATCH:
                yield self._line_batch(path, lineno, lines, projection)
                lineno += len(lines)
                lines = []
        if lines or lineno == 0:
            # An empty file still yields one empty batch, so the schema is observable and
            # the whole-file path's behaviour is preserved.
            yield self._line_batch(path, lineno, lines, projection)

    def _iter_lines(self, path: str) -> Iterator[str]:
        """Yield the file's lines, matching `str.splitlines()` exactly.

        `splitlines` — not `split("\n")` — because it is what the whole-file path used,
        and it also breaks on `\r`, `\v`, `\f`, `\x85`, `\u2028` and friends. Reading in
        blocks means two boundary hazards that a whole-file split never had:

        * a multi-byte character split across a block, handled by an *incremental* decoder;
        * a `\r` at the end of a block that turns out to be the start of a `\r\n`, which
          would otherwise be reported as one line where the file has none. It is held back
          until the next block resolves it.
        """
        import codecs

        decoder = codecs.getincrementaldecoder(self._encoding)()
        carry = ""
        with self._fs.open(path) as fh:
            while True:
                block = fh.read(_TEXT_BLOCK_BYTES)
                if not block:
                    carry += decoder.decode(b"", final=True)
                    break
                carry += decoder.decode(block)
                pieces = carry.splitlines(keepends=True)
                if not pieces:
                    continue
                # The last piece is incomplete unless it carries its terminator — and a
                # trailing lone "\r" may still become "\r\n", so it waits too.
                carry = ""
                if pieces[-1] == pieces[-1].splitlines()[0] or pieces[-1].endswith("\r"):
                    carry = pieces.pop()
                for piece in pieces:
                    yield piece.splitlines()[0]
        for piece in carry.splitlines():
            yield piece

    def _line_batch(
        self, path: str, first_line: int, lines: list[str], projection: list[str] | None
    ) -> pa.RecordBatch:
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([path] * len(lines), pa.string()),
                pa.array(range(first_line + 1, first_line + len(lines) + 1), pa.int64()),
                pa.array(lines, pa.string()),
            ],
            names=["path", "line_number", "text"],
        )
        return batch.select(projection) if projection is not None else batch

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
