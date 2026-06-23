"""JSON format — newline-delimited (line) JSON read + write."""

from __future__ import annotations

import json
from typing import IO, Any

import pyarrow as pa

from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import FileSplit, LineRangeSplit, Split

__all__ = ["JSONSink", "JSONSource"]


@SOURCES.register("json")
class JSONSource(FileSource):
    """One or more newline-delimited (line) JSON files (file, directory, or glob).

    Large files are split into newline-aligned byte ranges (`LineRangeSplit`), so a
    single multi-GB NDJSON file is read in parallel across workers; small files use
    one split each. `pyarrow.json.read_json` reads each range whole, so per-task
    memory scales with the split size, not the whole file.
    """

    suffix = ".json"
    format_name = "json"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.json as pajson

        return pajson.read_json(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.json as pajson

        table = pajson.read_json(fh)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:
        # Default byte-range split size (so one huge NDJSON file fans across workers
        # instead of reading on a single node) is `ExecutionConfig.split_bytes`.
        chunk = target_size or active_config().execution.split_bytes
        try:
            size = self._fs.size(path)
        except (OSError, ValueError):
            return [FileSplit(self.format_name, path)]
        if size <= chunk:
            return [FileSplit(self.format_name, path)]
        return [
            LineRangeSplit(self.format_name, path, start, min(start + chunk, size))
            for start in range(0, size, chunk)
        ]


@SINKS.register("json")
class JSONSink(FileSink):
    """Write newline-delimited (line) JSON.

    pyarrow has no JSON writer, so each row is serialized as one JSON object per
    line via the stdlib — the shape `JSONSource` / `pyarrow.json.read_json` reads.
    """

    suffix = ".json"
    format_name = "json"

    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        for row in table.to_pylist():
            fh.write((json.dumps(row) + "\n").encode("utf-8"))
