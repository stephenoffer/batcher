"""Log format — line-delimited text logs read as raw lines (core, no extra).

`LogSource` reads any line-delimited text file into a fixed Arrow schema
``{path: str, line_number: int64, line: str}`` — one Arrow row per source line,
assembled at batch granularity. No regex/grok parsing happens in Python (that
would be per-row hot-path work); when a `pattern` is supplied it is stored and
surfaced to the api layer, which lowers grok extraction into Rust as
``col("line").str.regexp_extract(pattern, ...)``. One whole file is one `Split`.
"""

from __future__ import annotations

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
        name = getattr(fh, "name", self._path)
        batch_rows = active_config().execution.morsel_rows
        out: list[pa.RecordBatch] = []
        paths: list[str] = []
        numbers: list[int] = []
        lines: list[str] = []
        for i, raw in enumerate(fh, start=1):
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            paths.append(name)
            numbers.append(i)
            lines.append(text.rstrip("\n"))
            if len(lines) >= batch_rows:
                out.append(self._batch(paths, numbers, lines, projection))
                paths, numbers, lines = [], [], []
        if lines or not out:
            out.append(self._batch(paths, numbers, lines, projection))
        return out

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
