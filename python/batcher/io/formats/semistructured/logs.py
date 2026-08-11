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
from batcher.io.base._lines import iter_line_blocks, one_array
from batcher.io.formats.base import SOURCES
from batcher.io.splits import FileSplit, Split

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

    def splits(
        self,
        target_size: int | None = None,
        predicate: dict | None = None,
        projection: list[str] | None = None,
    ) -> list[Split]:
        """Independently-readable slices — byte ranges when the scan allows, else per file.

        A log file was one split, so a single 50 GB log was a single task however many nodes
        the cluster had — and a log file is the archetypal input large enough for that to
        matter. Lines are newline-delimited, so the file range-splits exactly as NDJSON and
        CSV do, with one exception: `line_number` counts from the start of the file, and a
        range beginning in the middle cannot know it. The ranges are therefore offered only
        when the pushed projection does not ask for it. See `io.splits.text`.

        A bring-your-own filesystem, `storage_options`, or a tolerated (`on_error`) read
        keeps the whole-file split, because a range split resolves the backend itself and
        reads fail-fast — the same condition on which `ParquetSource` declines its own
        row-group fast path, for the same reason.

        Args:
            target_size: Rough bytes per split; `ExecutionConfig.split_bytes` when omitted.
            predicate: The predicate Kyber pushed to this scan, if any (a log has no
                metadata to prune with, so it is only forwarded to the base planner).
            projection: The columns Kyber pushed to this scan, if any.

        Returns:
            The splits covering this source exactly once.
        """
        if (
            projection
            and "line_number" not in projection
            and self._filesystem is None
            and self._storage_options is None
            and self._errors.mode == "raise"
        ):
            from batcher.config import active_config
            from batcher.io.splits import line_range_splits

            chunk = target_size or active_config().execution.split_bytes
            out: list[Split] = []
            for f in self._files():
                try:
                    size = self._fs.size(f)
                except (OSError, ValueError):
                    size = None
                if size is None or size <= chunk:
                    out.append(FileSplit(self.format_name, f, self._reader_kwargs()))
                    continue
                out.extend(
                    line_range_splits(
                        self.format_name,
                        f,
                        size,
                        chunk,
                        text_column="line",
                        columns=tuple(projection),
                        splitlines=False,
                    )
                )
            return out
        return super().splits(target_size, predicate)

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
        first = 1
        held: list[pa.Array] = []
        n_held = 0
        emitted = False
        # Blocks arrive as Arrow arrays (or lists, when the exact Python fallback ran) and
        # are concatenated only at batch boundaries, so a line is never a Python object on
        # the fast path. See `io.base._lines` for why the splitting is Arrow's.
        for block in iter_line_blocks(fh, splitlines=False):
            arr = block if isinstance(block, pa.Array) else pa.array(block, pa.string())
            held.append(arr)
            n_held += len(arr)
            while n_held >= batch_rows:
                # Sliced to the morsel size rather than emitted whole. Emitting the whole
                # accumulation avoids a concatenation, and it also unbounds the batch: a
                # block is 16 MiB of *bytes*, which for short lines is millions of rows, and
                # streaming a log a morsel at a time is the reason this loop exists.
                # `Array.slice` is a zero-copy view, so carrying the remainder is free.
                lines = one_array(held)
                yield self._batch(name, first, lines.slice(0, batch_rows), projection)
                emitted = True
                first += batch_rows
                held, n_held = [lines.slice(batch_rows)], n_held - batch_rows
        if n_held or not emitted:
            lines = one_array(held)
            yield self._batch(name, first, lines, projection)

    @staticmethod
    def _batch(
        path: str,
        first_line: int,
        lines: pa.Array,
        projection: list[str] | None,
    ) -> pa.RecordBatch:
        """One batch, with `path` and `line_number` built without a Python list per row.

        `path` repeats one string and `line_number` is a contiguous run, so building either
        as an N-element Python list only to hand back to Arrow is pure overhead — `pa.repeat`
        and `numpy.arange` produce the same two columns entirely in C.
        """
        import numpy as np

        n = len(lines)
        batch = pa.RecordBatch.from_arrays(
            [
                pa.repeat(pa.scalar(path, pa.string()), n),
                pa.array(np.arange(first_line, first_line + n, dtype=np.int64)),
                lines,
            ],
            schema=LOG_SCHEMA,
        )
        return batch.select(projection) if projection is not None else batch
