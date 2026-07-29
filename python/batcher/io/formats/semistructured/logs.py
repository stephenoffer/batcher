"""Log format — line-delimited text logs read as raw lines (core, no extra).

`LogSource` reads any line-delimited text file into a fixed Arrow schema
``{path: str, line_number: int64, line: str}`` — one Arrow row per source line,
assembled at batch granularity. No regex/grok parsing happens in Python (that
would be per-row hot-path work); when a `pattern` is supplied it is stored and
surfaced to the api layer, which lowers grok extraction into Rust as
``col("line").str.regexp_extract(pattern, ...)``. One whole file is one `Split`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["LogSource"]

#: The fixed schema every `LogSource` produces (raw, unparsed lines).
LOG_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("line_number", pa.int64()),
        ("line", pa.string()),
    ]
)


@SOURCES.register("logs")
class LogSource(FileSource):
    """One or more line-delimited log files, read as raw lines.

    Args:
        path: The log file (single file, directory, or glob).
        pattern: Optional grok/regex pattern. NOT applied in Python — it is stored
            on `pattern` for the api layer to lower into a Rust
            ``str.regexp_extract`` over the ``line`` column.
    """

    suffix = ".log"
    format_name = "logs"

    __slots__ = ("pattern",)

    def __init__(self, path: str, *, pattern: str | None = None, **kwargs: Any) -> None:
        # Forward the base options; dropping them made `on_error="skip"` a silent no-op.
        super().__init__(path, **kwargs)
        self.pattern = pattern

    def _estimated_row_count(self, byte_total: int | None) -> int | None:
        """An advisory line count from a byte sample — a log file emits one row per line.

        A log source has no footer, so cardinality was the planner's default. Every line is
        one Arrow row (no header), so the shared delimited estimator scales the first file's
        average line width by the dataset's on-disk size. Advisory (`statistics()` marks it
        `exact_rows=False`), and cheap enough to run at plan time. `byte_total` is the size
        `statistics()` already computed, reused so the file sizes are not swept twice.
        """
        from batcher.io.stats.row_estimate import estimate_delimited_rows

        return estimate_delimited_rows(
            self._fs, self._files(), has_header=False, total_bytes=byte_total
        )

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return LOG_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Every batch of one log file, materialized — the `read()` contract."""
        return list(self._batches_from(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one log a morsel at a time rather than decoding it whole.

        `_read_file` batches internally but *accumulates every batch* before returning, so
        `iter_batches` was streaming in name only: a multi-GB log was fully resident as
        Arrow before its first batch reached the consumer. A log file is the archetypal
        larger-than-memory streaming input, and the neighbouring text source already fixed
        exactly this shape for its line mode ("three copies of a multi-GB log"); the log
        format, which exists for nothing else, kept the unbounded form.

        Args:
            path: The log file to stream.
            projection: Columns the scan must produce. All columns when omitted.

        Yields:
            One `RecordBatch` per morsel of lines, in file order.
        """
        with self._open(path) as fh:
            yield from self._batches_from(fh, projection, name=path)

    def _batches_from(
        self, fh: IO[Any], projection: list[str] | None, name: str | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Yield morsel-sized batches of `fh`'s lines. The one decoding loop both paths use.

        An *empty* file still yields one empty batch, because a reader that produced no
        batch at all would leave the caller with no schema to build an empty result from.
        """
        name = name or getattr(fh, "name", self._path)
        batch_rows = active_config().execution.morsel_rows
        paths: list[str] = []
        numbers: list[int] = []
        lines: list[str] = []
        emitted = False
        for i, raw in enumerate(fh, start=1):
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            paths.append(name)
            numbers.append(i)
            lines.append(text.rstrip("\n"))
            if len(lines) >= batch_rows:
                yield self._batch(paths, numbers, lines, projection)
                emitted = True
                paths, numbers, lines = [], [], []
        if lines or not emitted:
            yield self._batch(paths, numbers, lines, projection)

    @staticmethod
    def _batch(
        paths: list[str],
        numbers: list[int],
        lines: list[str],
        projection: list[str] | None,
    ) -> pa.RecordBatch:
        batch = pa.RecordBatch.from_pydict(
            {"path": paths, "line_number": numbers, "line": lines}, schema=LOG_SCHEMA
        )
        return batch.select(projection) if projection is not None else batch
