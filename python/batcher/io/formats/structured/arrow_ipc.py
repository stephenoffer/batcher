"""Arrow IPC / Feather format — zero-conversion read + write via `pyarrow.ipc`.

The Arrow IPC file format (a.k.a. Feather v2) is the engine's native on-disk
shape: batches are already Arrow, so read/write are conversion-free. Reads expose
*block-level* splits — one `ArrowBlockSplit` per record-batch block in the file —
so a distributed read pulls only its assigned blocks via
``ipc.open_file(...).get_batch(i)``. Projection is applied per batch with
``batch.select``. Registered under ``arrow``, ``feather`` and ``ipc``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.base import FileSink, FileSource
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import Split

__all__ = ["ArrowBlockSplit", "ArrowIPCSink", "ArrowIPCSource"]


def _require_ipc() -> Any:
    """Import and return `pyarrow.ipc` or raise `BackendError`."""
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - pyarrow ships ipc by default
        raise BackendError(
            "Arrow IPC support requires pyarrow: pip install 'batcher-engine[all]'"
        ) from exc
    return ipc


def _select(batch: pa.RecordBatch, projection: list[str] | None) -> pa.RecordBatch:
    return batch.select(projection) if projection is not None else batch


@dataclass(frozen=True, slots=True)
class ArrowBlockSplit:
    """A contiguous run of record-batch blocks within one Arrow IPC file.

    Carries only ``(path, blocks)``; `read` reopens the file and pulls just those
    blocks via ``RecordBatchFileReader.get_batch``.
    """

    path: str
    blocks: tuple[int, ...]

    def _reader(self) -> Any:
        ipc = _require_ipc()
        fs = resolve_filesystem(self.path)
        return ipc.open_file(fs.open(self.path))

    def schema(self) -> pa.Schema:
        return self._reader().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        reader = self._reader()
        return [_select(reader.get_batch(i), projection) for i in self.blocks]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        reader = self._reader()
        for i in self.blocks:
            yield _select(reader.get_batch(i), projection)

    def row_count(self) -> int | None:
        reader = self._reader()
        return sum(reader.get_batch(i).num_rows for i in self.blocks)

    def identity(self) -> str:
        return f"arrow:{self.path}:blocks{','.join(map(str, self.blocks))}"


@SOURCES.register("arrow")
@SOURCES.register("feather")
@SOURCES.register("ipc")
class ArrowIPCSource(FileSource):
    """One or more Arrow IPC (Feather v2) files (single file, directory, or glob)."""

    suffix = ".arrow"
    format_name = "arrow"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        ipc = _require_ipc()
        return ipc.open_file(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        ipc = _require_ipc()
        reader = ipc.open_file(fh)
        return [_select(reader.get_batch(i), projection) for i in range(reader.num_record_batches)]

    def _file_row_count(self, path: str) -> int | None:
        ipc = _require_ipc()
        with self._fs.open(path) as fh:
            reader = ipc.open_file(fh)
            return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:  # noqa: ARG002
        ipc = _require_ipc()
        with self._fs.open(path) as fh:
            n = ipc.open_file(fh).num_record_batches
        return [ArrowBlockSplit(path, (i,)) for i in range(n)]


@SINKS.register("arrow")
@SINKS.register("feather")
@SINKS.register("ipc")
class ArrowIPCSink(FileSink):
    """Write an Arrow IPC (Feather v2) file."""

    suffix = ".arrow"
    format_name = "arrow"

    __slots__ = ("compression",)

    def __init__(self, compression: str | None = "zstd") -> None:
        self.compression = compression

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        ipc = _require_ipc()
        options = ipc.IpcWriteOptions(compression=self.compression)
        with ipc.new_file(fh, table.schema, options=options) as writer:
            writer.write_table(table)

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        ipc = _require_ipc()
        options = ipc.IpcWriteOptions(compression=self.compression)
        return ipc.new_file(fh, schema, options=options)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write_batch(batch)

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
