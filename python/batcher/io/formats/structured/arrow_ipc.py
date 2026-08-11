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

from batcher._internal.optional import require
from batcher.io.base import FileSink, FileSource
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import FileSplit, Split

__all__ = ["ArrowBlockSplit", "ArrowIPCSink", "ArrowIPCSource"]


def _require_ipc() -> Any:
    """Import and return `pyarrow.ipc` or raise `BackendError`."""
    return require("pyarrow.ipc", feature="Arrow IPC support", provides="pyarrow", extra="all")


def _select(batch: pa.RecordBatch, projection: list[str] | None) -> pa.RecordBatch:
    return batch.select(projection) if projection is not None else batch


#: Offset widths for the variable-length layouts, by the predicate that recognizes them.
_OFFSET_WIDTHS: tuple[tuple[Any, str], ...] = (
    (pa.types.is_large_string, "q"),
    (pa.types.is_large_binary, "q"),
    (pa.types.is_large_list, "q"),
    (pa.types.is_string, "i"),
    (pa.types.is_binary, "i"),
    (pa.types.is_list, "i"),
)


def _offset_base(array: pa.Array) -> int:
    """The first entry of `array`'s offsets buffer, or 0 when it has none.

    A variable-length Arrow array is free to start its offsets anywhere in the values
    buffer, and the engine produces exactly that for the trailing partial batch of a
    `limit`: the batch is a window onto its morsel, so its offsets begin mid-buffer.
    """
    fmt = next((code for check, code in _OFFSET_WIDTHS if check(array.type)), None)
    if fmt is None:
        return 0
    buffers = array.buffers()
    if len(buffers) < 2 or buffers[1] is None or buffers[1].size < 8:
        return 0
    return int(memoryview(buffers[1]).cast(fmt)[array.offset])


def _rebase_offsets(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Return `batch` with every variable-length column's offsets starting at zero.

    Works around a defect in `pyarrow.ipc` (reproduced on 19.0.1, and with no engine code
    involved): a string, binary or list array whose offsets buffer does *not* start at
    zero serializes to garbage. The array is valid Arrow — `validate(full=True)` passes,
    and reading it in memory is correct — so nothing upstream notices, and the corruption
    surfaces only after a round trip, as NUL bytes or invalid UTF-8 in the tail.

    Batcher hits it on the trailing partial batch of a `limit`, which is a window onto its
    morsel and therefore has non-zero offsets. `ds.head(50_000).write.arrow(path)` wrote
    848 corrupt rows before this.

    `concat_arrays` on a single array copies it into a fresh contiguous buffer with the
    offsets rebased, which is the cheapest normalization pyarrow offers. Columns that are
    already based at zero are passed through untouched, so a batch that does not need this
    pays one buffer read per column and no copy.
    """
    columns = list(batch.columns)
    rebased = False
    for index, column in enumerate(columns):
        if _offset_base(column):
            columns[index] = pa.concat_arrays([column])
            rebased = True
    if not rebased:
        return batch
    return pa.RecordBatch.from_arrays(columns, schema=batch.schema)


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

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one IPC file batch by batch rather than holding every batch at once.

        The IPC file format already stores discrete record batches; `_read_file` merely
        collects them into a list, which is the whole file in memory. Yielding them keeps
        peak memory at one batch.
        """
        ipc = _require_ipc()
        with self._fs.open(path) as fh:
            reader = ipc.open_file(fh)
            for i in range(reader.num_record_batches):
                yield _select(reader.get_batch(i), projection)

    def _file_row_count(self, path: str) -> int | None:
        ipc = _require_ipc()
        with self._fs.open(path) as fh:
            reader = ipc.open_file(fh)
            return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))

    def _file_splits(
        self,
        path: str,
        target_size: int | None,  # noqa: ARG002
        predicate: dict | None = None,  # noqa: ARG002 (no IPC block statistics to prune with)
    ) -> list[Split]:
        # An `ArrowBlockSplit` carries only `(path, blocks)` and re-resolves the filesystem
        # from the bare path on the worker, so it cannot carry a bring-your-own `filesystem=`
        # or `storage_options=`: the worker would resolve its own backend from the
        # environment and read a *different store* than the driver was configured for — a
        # dict-carried `endpoint_override` pointing at an on-prem MinIO is exactly the case,
        # and a wrong-store read is a different object, not a slower one.
        #
        # `on_error` rides the same fallback and for the same reason: the split carries no
        # reader kwargs, so a tolerated read would rebuild a fail-fast reader on the worker
        # and one truncated shard would abort the whole distributed query while the
        # tolerance looked wired up.
        #
        # A `FileSplit` reconstructs the source through `_reader_kwargs()` and does carry
        # them, trading sub-file block granularity for correct credentials and policy —
        # the same trade `ParquetSource._file_splits` makes for row-group splits.
        if (
            self._filesystem is not None
            or self._storage_options is not None
            or self._errors.mode != "raise"
        ):
            return [FileSplit(self.format_name, path, self._reader_kwargs())]
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

    def __init__(self, compression: str | None = "zstd", **kwargs: Any) -> None:
        super().__init__(**kwargs)  # carries filesystem= / storage_options=
        self.compression = compression

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        ipc = _require_ipc()
        options = ipc.IpcWriteOptions(compression=self.compression)
        with ipc.new_file(fh, table.schema, options=options) as writer:
            # Per batch rather than `write_table`, so each one passes through the offset
            # rebase. A table assembled from engine batches carries the same chunks.
            for batch in table.to_batches():
                writer.write_batch(_rebase_offsets(batch))

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        ipc = _require_ipc()
        options = ipc.IpcWriteOptions(compression=self.compression)
        return ipc.new_file(fh, schema, options=options)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write_batch(_rebase_offsets(batch))

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
