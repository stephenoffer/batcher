"""Plain-text source — one row per line or one row per whole file."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.base._lines import iter_line_blocks, one_array
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
        held: list[pa.Array] = []
        n_held = 0
        # Blocks arrive as Arrow arrays (or lists, when the exact Python fallback ran) and
        # are concatenated only at batch boundaries, so a line is never a Python object on
        # the fast path. See `io.base._lines` for why the splitting is Arrow's.
        for block in self._iter_lines(path):
            arr = block if isinstance(block, pa.Array) else pa.array(block, pa.string())
            held.append(arr)
            n_held += len(arr)
            while n_held >= _TEXT_LINES_PER_BATCH:
                # Sliced to the batch size rather than emitted whole. Emitting the whole
                # accumulation avoids a concatenation, and it also unbounds the batch: a
                # block is 16 MiB of *bytes*, which for short lines is millions of rows, so
                # the batch-size knob would stop bounding anything and line mode exists to
                # bound exactly that. `Array.slice` is a zero-copy view, so the remainder
                # costs nothing to carry — only rejoining it to the next block copies.
                lines = one_array(held)
                yield self._line_batch(
                    path, lineno, lines.slice(0, _TEXT_LINES_PER_BATCH), projection
                )
                lineno += _TEXT_LINES_PER_BATCH
                held = [lines.slice(_TEXT_LINES_PER_BATCH)]
                n_held -= _TEXT_LINES_PER_BATCH
        if n_held or lineno == 0:
            # An empty file still yields one empty batch, so the schema is observable and
            # the whole-file path's behaviour is preserved — and it reaches here with no
            # block at all, which `concat_arrays` refuses.
            yield self._line_batch(path, lineno, one_array(held), projection)

    def _iter_lines(self, path: str) -> Iterator[pa.Array | list[str]]:
        """Yield each block's complete lines, matching `str.splitlines()` exactly.

        UTF-8 — the default, and what text files overwhelmingly are — goes through the
        shared Arrow-backed splitter, which produces the column with no Python object per
        line. Any other encoding keeps the incremental-decoder path below, because that
        splitter reads UTF-8 and nothing else; a `latin-1` or `utf-16` file is decoded here
        exactly as it always was.
        """
        if self._encoding.lower().replace("-", "").replace("_", "") in ("utf8", "u8", "utf"):
            with self._fs.open(path) as fh:
                yield from iter_line_blocks(fh, splitlines=True)
            return
        yield from self._iter_lines_decoded(path)

    def _iter_lines_decoded(self, path: str) -> Iterator[list[str]]:
        """`_iter_lines` for a non-UTF-8 encoding, decoded incrementally in Python.

        `splitlines` — not `split("\n")` — because it is what the whole-file path used,
        and it also breaks on `\r`, `\v`, `\f`, `\x85`, `\u2028` and friends. Reading in
        blocks means two boundary hazards that a whole-file split never had:

        * a multi-byte character split across a block, handled by an *incremental* decoder;
        * a `\r` at the end of a block that turns out to be the start of a `\r\n`, which
          would otherwise be reported as one line where the file has none. It is held back
          until the next block resolves it.

        A **block** of lines is yielded rather than each line, and the terminators are
        stripped by splitting the block's body **once** rather than per line. The previous
        form called `piece.splitlines()[0]` on every individual line — to undo the terminator
        `keepends=True` had just attached — so each line cost a second scan, a fresh
        one-element list, an index, and a generator resume. That is per-row Python in the
        read path, and it was most of the cost: this source read **12.8 MB/s** where
        `pyarrow.csv` reads the same line-delimited bytes at over 1 GB/s. `keepends` is now
        used only to find where the complete body ends.
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
                tail = pieces[-1]
                if tail == tail.splitlines()[0] or tail.endswith("\r"):
                    body, carry = carry[: len(carry) - len(tail)], tail
                else:
                    body, carry = carry, ""
                if body:
                    yield body.splitlines()
        if carry:
            yield carry.splitlines()

    def _line_batch(
        self, path: str, first_line: int, lines: pa.Array, projection: list[str] | None
    ) -> pa.RecordBatch:
        """One batch of lines, with the two derived columns built without per-row Python.

        `path` is the same string on every row and `line_number` is a contiguous run, so
        neither needs a Python list per batch: `[path] * len(lines)` allocated a list of
        16,384 references to one string and `range(...)` boxed 16,384 integers, both only
        for Arrow to walk them straight back. `pa.repeat` and `numpy.arange` produce the
        same two columns entirely in C.
        """
        import numpy as np

        n = len(lines)
        batch = pa.RecordBatch.from_arrays(
            [
                pa.repeat(pa.scalar(path, pa.string()), n),
                pa.array(np.arange(first_line + 1, first_line + n + 1, dtype=np.int64)),
                lines,
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
        """The exact row count in file mode (one row per file), else None.

        In ``mode="file"`` a row *is* a whole file, so the count is exactly the number of
        files — known from the listing, with no read. In line mode an exact count needs a
        scan, so the honest answer is None and `statistics()` supplies an estimate instead.
        """
        return len(self._files()) if self._mode == "file" else None

    def statistics(self):  # type: ignore[no-untyped-def]
        """Cheap metadata for a text scan: exact rows in file mode, a line estimate otherwise.

        A text source reached the estimator with no size and no count, so a join against one
        was sized from the planner's default. File mode gives an exact row count (the file
        count) and byte size; line mode gives a byte-sample line estimate (advisory) and byte
        size. Best-effort — a byte size that cannot be read is simply omitted.
        """
        from batcher.io._concurrent import total_file_bytes
        from batcher.plan.source_stats import SourceStatistics

        files = self._files()
        if not files:
            return None
        byte_size = total_file_bytes(self._fs, files)
        if self._mode == "file":
            # One row is one whole file, so the byte total divided by the file count is a
            # row's size directly — which the `string`/`binary` column's 36-byte type prior
            # cannot approach for a corpus of documents.
            return SourceStatistics(
                row_count=len(files),
                byte_size=byte_size,
                exact_rows=True,
                content_byte_size=True,
            )
        from batcher.io.stats.row_estimate import estimate_delimited_rows

        rows = estimate_delimited_rows(self._fs, files, has_header=False, total_bytes=byte_size)
        if rows is None and byte_size is None:
            return None
        # Line mode: a row is a line and the byte total covers every line, so bytes per row
        # is again the row's own content — just derived from an estimated count rather than
        # an exact one, which `exact_rows=False` already records.
        return SourceStatistics(
            row_count=rows, byte_size=byte_size, exact_rows=False, content_byte_size=True
        )

    def identity(self) -> str:
        # `encoding` decides how bytes become text, so the same path read as utf-8 vs
        # latin-1 is a different relation: different characters, and a different line
        # count in line mode when a byte decodes to a line break in one encoding but not
        # the other. Omitting it collided their identities. Kept off the key for the
        # utf-8 default so the common identity is unchanged.
        base = f"text:{self._mode}:{self._path}"
        if self._encoding.lower().replace("-", "") != "utf8":
            return f"{base}#enc={self._encoding}"
        return base

    def splits(
        self, target_size: int | None = None, projection: list[str] | None = None
    ) -> list[Split]:
        """Independently-readable slices — byte ranges when the scan allows, else one per file.

        A whole file was the only unit here, so one 50 GB log was one task however many nodes
        the cluster had. Line mode is newline-delimited, so it can range-split exactly as
        NDJSON and CSV do — except for `line_number`, which counts from the start of the file
        and therefore cannot be produced by a range that starts in the middle of one. So the
        ranges are offered only when the pushed projection does not ask for it, which is the
        common shape (grep the lines, count matches, extract a field). See
        `io.splits.text` for the reasoning in full.

        File mode is one row per file and cannot subdivide at all; a non-UTF-8 encoding keeps
        the whole-file split because the range reader decodes UTF-8.

        Args:
            target_size: Rough bytes per split; `ExecutionConfig.split_bytes` when omitted.
            projection: The columns Kyber pushed to this scan, if any.

        Returns:
            The splits covering this source exactly once.
        """
        files = self._files()
        if self._can_range_split(projection):
            from batcher.config import active_config
            from batcher.io.splits import line_range_splits

            chunk = target_size or active_config().execution.split_bytes
            out: list[Split] = []
            for f in files:
                try:
                    size = self._fs.size(f)
                except (OSError, ValueError):
                    size = None
                if size is None or size <= chunk:
                    out.append(WholeSourceSplit(TextSource(f, mode=self._mode)))
                    continue
                out.extend(
                    line_range_splits(
                        "text",
                        f,
                        size,
                        chunk,
                        text_column="text",
                        columns=tuple(projection or ()),
                        splitlines=True,
                    )
                )
            return out
        return [
            WholeSourceSplit(TextSource(f, mode=self._mode, encoding=self._encoding)) for f in files
        ]

    def _can_range_split(self, projection: list[str] | None) -> bool:
        """Whether this scan can be served by byte ranges rather than whole files."""
        return (
            self._mode == "line"
            and projection is not None
            and "line_number" not in projection
            and bool(projection)
            and self._encoding.lower().replace("-", "").replace("_", "") in ("utf8", "u8", "utf")
        )
