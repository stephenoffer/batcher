"""Staged-file writes for the Iceberg sink.

Each worker writes its shard as a **real** Parquet file into a staging area under the
table (parallel, shared-nothing, bounded per-worker memory) and returns only the file
locator. The driver then registers every staged file in one transaction with
``add_files`` — the data never flows through the driver.

Delta no longer stages: its sink writes final data files straight into the table's own
layout and commits their `AddAction`s (see `_commit`), which removes the staging copy
entirely. Iceberg still stages because ``add_files`` re-reads each file's footer to
build the manifest entry, so the file must exist before the commit names it.

Shard file names are deterministic (``part-{shard}-{chunk}.parquet``), so a preempted
and re-run shard overwrites its own staged file — the commit stays idempotent.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pyarrow as pa

from batcher.io.base import _safe_size
from batcher.io.filesystem import resolve_filesystem
from batcher.io.manifest import WrittenFile

__all__ = [
    "cleanup_staging",
    "stage_shard",
    "stage_stream",
    "staging_root",
]

# Subdirectory (under the table root) that holds shard Parquet files until they are
# committed. Left in place only if a commit fails, so a retry finds the staged data.
_STAGING = "_batcher_staging"


def staging_root(table_path: str) -> str:
    """The staging directory for a table write rooted at `table_path`."""
    return f"{table_path.rstrip('/')}/{_STAGING}"


def _shard_name(staging: str, token: str | None, file_index: int, chunk_index: int) -> str:
    """Staged-file path, deterministic in ``(file_index, chunk_index)`` — plus a per-write
    `token` when the caller needs names unique across writes (Iceberg ``add_files``, which
    references files permanently, must not let a later write clobber a prior snapshot's)."""
    tok = f"{token}-" if token else ""
    return f"{staging}/part-{tok}{file_index:05d}-{chunk_index:05d}.parquet"


def stage_shard(
    table: pa.Table,
    staging: str,
    *,
    file_index: int,
    chunk_index: int = 0,
    token: str | None = None,
    compression: str = "zstd",
) -> WrittenFile:
    """Write `table` as one staged Parquet file and return its locator.

    Runs wherever the shard is produced (a worker in the distributed path, the driver
    single-node), so the encode is parallel and shared-nothing. The name is deterministic
    in ``(file_index, chunk_index)`` (a re-run shard overwrites itself — idempotent),
    optionally prefixed with a per-write `token`.
    """
    import pyarrow.parquet as pq

    fs = resolve_filesystem(staging)
    fs.mkdirs(staging, exist_ok=True)
    name = _shard_name(staging, token, file_index, chunk_index)
    with fs.atomic_writer(name) as fh:
        pq.write_table(table, fh, compression=compression)
    return WrittenFile(path=name, rows=table.num_rows, bytes=_safe_size(fs, name))


def stage_stream(
    batches: Iterator[pa.RecordBatch],
    staging: str,
    *,
    schema: pa.Schema | None = None,
    file_index: int = 0,
    token: str | None = None,
    compression: str = "zstd",
) -> WrittenFile:
    """Stream `batches` into one staged Parquet file, holding a single batch at a time.

    The bounded-memory counterpart of `stage_shard`: a breaker-free read→transform→write
    never materializes the whole result before the lakehouse commit. Returns the file
    locator (rows counted as they stream). An empty stream writes a valid empty file so
    the commit sees a schema-correct (zero-row) shard.
    """
    import pyarrow.parquet as pq

    fs = resolve_filesystem(staging)
    fs.mkdirs(staging, exist_ok=True)
    name = _shard_name(staging, token, file_index, 0)
    it = iter(batches)
    first = next(it, None)
    rows = 0
    with fs.atomic_writer(name) as fh:
        if first is None:
            empty = schema.empty_table() if schema is not None else pa.table({})
            pq.write_table(empty, fh, compression=compression)
        else:
            from itertools import chain

            # `closing`, not a bare `close()` at the end: the writer was closed only on the
            # success path, so a shard that raised partway — a preempted worker, an OOM, a
            # batch that fails to encode — left it open. That leaks the writer's buffers and
            # the file handle once per failed shard, and the `atomic_writer` context then
            # exits around a file still being written to.
            with contextlib.closing(
                pq.ParquetWriter(fh, first.schema, compression=compression)
            ) as writer:
                for batch in chain([first], it):
                    if batch.num_rows:
                        writer.write_batch(batch)
                        rows += batch.num_rows
    return WrittenFile(path=name, rows=rows, bytes=_safe_size(fs, name))


def cleanup_staging(files: list[WrittenFile], staging: str) -> None:
    """Delete the staged shard files and the staging directory (best-effort).

    Committed data lives in the table's own layout, so the staging copies are pure
    scratch; a failure to remove them is a tidiness issue, never a correctness one.
    """
    fs = resolve_filesystem(staging)
    for f in files:
        with contextlib.suppress(OSError, ValueError, NotImplementedError):
            fs.remove(f.path)
    # Remove the now-empty staging directory via the underlying pyarrow filesystem.
    inner = getattr(fs, "_fs", None)
    if inner is not None:
        with contextlib.suppress(Exception):
            inner.delete_dir(fs._p(staging))
