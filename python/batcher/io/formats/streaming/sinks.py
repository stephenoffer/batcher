"""Streaming sinks — per-micro-batch writers for the streaming-query engine.

A `StreamSink` consumes one Arrow table per micro-batch (`open` → many
`write_batch` → `close`). It is the sink-side counterpart of the unbounded
`Source`: the engine in `core` drives the loop and hands each finished micro-batch
here.

Where a batch sink writes durable files it **reuses the existing batch machinery**
— `FileStreamSink` wraps any `SINKS` file format and writes one atomic
``part-batch<NNNNN>`` file per micro-batch (exactly-once by position, the same
property `resume=` relies on), and `DeltaStreamSink` reuses the transactional Delta
append. Only the genuinely-new targets (console, memory, foreach-batch, foreach)
are implemented here from scratch. The `write_batch` return value is an opaque
*sink token* the checkpoint commit-log records (Workstream D).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from batcher._internal.registry import Registry

__all__ = [
    "STREAM_SINKS",
    "ConsoleStreamSink",
    "DeltaStreamSink",
    "FileStreamSink",
    "ForeachBatchStreamSink",
    "ForeachStreamSink",
    "MemoryStreamSink",
    "StreamSink",
    "memory_table",
]


@runtime_checkable
class StreamSink(Protocol):
    """A sink that consumes one Arrow table per micro-batch.

    `open` is called once before the first batch; `write_batch` is called per
    micro-batch and returns an opaque sink token (recorded in the checkpoint
    commit-log for idempotent recovery, or ``None`` for non-durable sinks); `close`
    is called once when the query stops.
    """

    def open(self) -> None: ...

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None: ...

    def close(self) -> None: ...


STREAM_SINKS: Registry[type] = Registry("stream_sink")


@STREAM_SINKS.register("console")
class ConsoleStreamSink:
    """Print each micro-batch to stdout (the `console` sink — for development)."""

    def __init__(self, *, num_rows: int = 20, **_: Any) -> None:
        self._num_rows = num_rows

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        print(f"-------- Batch: {batch_id} --------")
        print(table.slice(0, self._num_rows))
        return None

    def close(self) -> None:
        pass


# Process-global store for in-memory sinks, read back by `bt.read_memory(name)`.
_MEMORY: dict[str, list[pa.Table]] = {}


def memory_table(name: str) -> pa.Table:
    """Return the accumulated table for a named in-memory streaming sink.

    Raises `KeyError` (surfaced as a clear error by the caller) if no query has
    written to `name`.
    """
    parts = _MEMORY[name]
    return pa.concat_tables(parts) if parts else pa.table({})


@STREAM_SINKS.register("memory")
class MemoryStreamSink:
    """Accumulate micro-batches in memory under `name`, queryable via `bt.read_memory`.

    For `complete` output mode the table is replaced each micro-batch (the running
    result is the whole answer); for `append`/`update` it grows. The mode is passed
    by the engine so the sink keeps only what the semantics require.
    """

    def __init__(self, name: str, *, output_mode: str = "append", **_: Any) -> None:
        self._name = name
        self._replace = output_mode == "complete"

    def open(self) -> None:
        _MEMORY[self._name] = []

    def write_batch(self, _batch_id: int, table: pa.Table) -> str | None:
        if self._replace:
            _MEMORY[self._name] = [table]
        else:
            _MEMORY[self._name].append(table)
        return None

    def close(self) -> None:
        pass


@STREAM_SINKS.register("foreach_batch")
class ForeachBatchStreamSink:
    """Call a user function ``fn(table, batch_id)`` on each micro-batch.

    The function receives the whole Arrow table for the micro-batch (never a row),
    so it is the sink-side twin of `map_batches` — the sanctioned hook for custom
    upserts (`MERGE`/SCD), multi-sink fan-out, and any Python-side commit logic.
    """

    def __init__(self, fn: Callable[[pa.Table, int], Any], **_: Any) -> None:
        self._fn = fn

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        self._fn(table, batch_id)
        return None

    def close(self) -> None:
        pass


@STREAM_SINKS.register("foreach")
class ForeachStreamSink:
    """Call ``fn(row)`` for each row of each micro-batch (Spark `foreach`).

    Convenience over `foreach_batch` for row-at-a-time external writes; the batch is
    converted to row dicts in one vectorized `to_pylist` (no per-element Python in
    the engine's hot path — the iteration is the user's chosen sink semantics).
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Any], **_: Any) -> None:
        self._fn = fn

    def open(self) -> None:
        pass

    def write_batch(self, _batch_id: int, table: pa.Table) -> str | None:
        for row in table.to_pylist():
            self._fn(row)
        return None

    def close(self) -> None:
        pass


class FileStreamSink:
    """Write one atomic ``part-batch<NNNNN>`` file per micro-batch via a batch sink.

    Reuses any registered `SINKS` file format (parquet/csv/json/…). `resume=True`
    skips a part file already present, giving exactly-once output by batch position
    when the source offsets are replayable (Workstream D). The output directory is a
    valid dataset the existing readers can scan.
    """

    def __init__(self, path: str, fmt: str, *, resume: bool = True, **opts: Any) -> None:
        from batcher.io.formats import SINKS

        self._path = path.rstrip("/")
        self._sink = SINKS.get(fmt)(**opts)
        self._suffix = getattr(self._sink, "suffix", "")
        self._resume = resume

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        file_path = f"{self._path}/part-batch{batch_id:05d}{self._suffix}"
        written = self._sink.write(table, file_path, resume=self._resume)
        return written.path

    def close(self) -> None:
        pass


class DeltaStreamSink:
    """Append each micro-batch to a Delta table in one transactional commit.

    Reuses the existing transactional `delta` sink (`mode="append"`), so each
    micro-batch is an atomic Delta version — the sink token is the written manifest
    digest the commit-log records.
    """

    def __init__(self, uri: str, **opts: Any) -> None:
        self._uri = uri
        opts.setdefault("mode", "append")
        self._opts = opts

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        from batcher.io.formats import SINKS
        from batcher.io.manifest import WriteManifest

        # A fresh sink per micro-batch makes each batch its own atomic Delta
        # transaction (the sink stages in `_pending` and is not reusable across
        # commits). The committed table version is the sink token.
        sink = SINKS.get("delta")(**self._opts)
        written = sink.write(table, self._uri)
        sink.commit(WriteManifest((written,)), self._uri)
        return f"delta:{batch_id}:{written.rows}"

    def close(self) -> None:
        pass
