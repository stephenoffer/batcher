"""Arrow IPC shuffle files — the object-store-bypassing data-plane transport.

Shuffle partitions are written as Arrow IPC stream files and passed between
stages by path. Only paths transit Ray; the bytes live on disk (local NVMe in
production), so the data plane never touches the Ray object store and is not
bounded by memory.
"""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa

__all__ = ["read_ipc", "write_ipc", "write_ipc_round_robin"]


def write_ipc(batches: list[pa.RecordBatch], path: str) -> str:
    """Write record batches to an Arrow IPC stream file. Returns `path`."""
    if not batches:
        raise ValueError("write_ipc requires at least one batch (for the schema)")
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_stream(sink, batches[0].schema) as writer:
        for b in batches:
            writer.write_batch(b)
    return path


def write_ipc_round_robin(
    batches: Iterable[pa.RecordBatch],
    fallback_schema: pa.Schema,
    paths: list[str],
) -> None:
    """Stream `batches` round-robin across per-partition IPC files.

    The driver holds **one batch at a time** — it never materializes the whole
    source — so a larger-than-RAM streaming input is partitioned in bounded memory.
    Each partition's IPC stream is seeded from the first batch's schema (a source
    yields a single consistent schema); a partition that receives no batch still
    gets one schema-only batch (from `fallback_schema` when the source was empty)
    so downstream map tasks always have a schema to operate over.

    Round-robin preserves the row multiset (each worker re-partitions by key before
    producing output, so the distributed result is unchanged); only which worker
    reads which batch differs.
    """
    n = len(paths)
    sinks: list[pa.OSFile | None] = [None] * n
    writers: list[object | None] = [None] * n
    schema: pa.Schema | None = None

    def _open(j: int, sch: pa.Schema) -> None:
        sinks[j] = pa.OSFile(paths[j], "wb")
        writers[j] = pa.ipc.new_stream(sinks[j], sch)

    try:
        for i, b in enumerate(batches):
            if schema is None:
                schema = b.schema
            j = i % n
            if writers[j] is None:
                _open(j, schema)
            writers[j].write_batch(b)  # type: ignore[union-attr]
        if schema is None:
            schema = fallback_schema
        empty = pa.RecordBatch.from_pylist([], schema=schema)
        for j in range(n):
            if writers[j] is None:
                _open(j, schema)
                writers[j].write_batch(empty)  # type: ignore[union-attr]
    finally:
        for w in writers:
            if w is not None:
                w.close()  # type: ignore[attr-defined]
        for s in sinks:
            if s is not None:
                s.close()


def read_ipc(path: str) -> list[pa.RecordBatch]:
    """Read all record batches from an Arrow IPC stream file."""
    with pa.OSFile(path, "rb") as src, pa.ipc.open_stream(src) as reader:
        return list(reader)
