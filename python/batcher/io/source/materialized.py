"""`MaterializedSource` — a distributed stage's result, left partitioned on disk.

Produced when an adaptive stage keeps its output as Arrow IPC files (one per producer)
instead of collecting it to the driver. The next stage scans it in place, shared-nothing,
so a multi-stage query never funnels an intermediate through one node.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.splits import IpcFileSplit, Split

__all__ = ["MaterializedSource"]


class MaterializedSource:
    """A relation whose batches live on disk as Arrow IPC files (one per producer).

    Produced by a distributed stage that kept its result *partitioned* instead of
    collecting it to the driver: the adaptive executor scans it in place for the next
    stage (shared-nothing, via `IpcFileSplit`s), and its exact `row_count` feeds the
    optimizer's build-side/broadcast choices (provenance ``EXACT`` via the
    `row_count` fallback). `cleanup()` removes the backing files once the query no
    longer needs the intermediate.
    """

    __slots__ = ("_files", "_schema", "_work_dir")
    bounded = True

    def __init__(
        self,
        files: list[tuple[str, int]],
        schema: pa.Schema,
        work_dir: str | None = None,
    ) -> None:
        self._files = files  # (ipc_path, exact_row_count) per producer partition
        self._schema = schema
        self._work_dir = work_dir

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        for path, _rows in self._files:
            out.extend(IpcFileSplit(path).read(projection))
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for path, _rows in self._files:
            yield from IpcFileSplit(path).iter_batches(projection)

    def row_count(self) -> int | None:
        return sum(rows for _path, rows in self._files)

    def identity(self) -> str:
        return f"materialized:{self._schema}:{self.row_count()}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [IpcFileSplit(path, rows) for path, rows in self._files]

    def cleanup(self) -> None:
        """Delete the backing IPC files' work directory (best-effort)."""
        if self._work_dir:
            import shutil

            shutil.rmtree(self._work_dir, ignore_errors=True)
