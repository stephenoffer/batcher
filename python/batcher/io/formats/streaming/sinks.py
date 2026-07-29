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

#: The output schema each named sink was opened with, so an *empty* sink still reads back as
#: the query's relation rather than as a table with no columns at all. A stream whose filter
#: matched nothing, or a `complete`-mode query whose running result went to zero rows, is a
#: perfectly ordinary outcome — and `bt.read_memory` answered it with a schema-less table,
#: which then fails to concatenate against the same query's non-empty run and reads as "no
#: such relation" rather than "no such rows".
_MEMORY_SCHEMA: dict[str, pa.Schema] = {}


def _check_memory_sink_size(name: str, held: int) -> None:
    """Raise a clear `ResourceError` if a named in-memory sink has outgrown the cap.

    In `append`/`update` mode this sink retains every micro-batch for the lifetime of
    the process, and `close()` frees nothing — so an unbounded stream written to it
    grows until the box dies, having looked healthy for hours. It is a debugging sink
    (the Spark `memory` sink is documented the same way), so the honest failure is an
    actionable error naming the sink, not a silent OOM. The cap is the same
    `memory.streaming_state_max_bytes` envelope that bounds watermark-held state.

    `held` is passed in rather than recomputed. Summing `nbytes` across every retained
    table on every micro-batch is O(batches) work per batch, so the guard against the sink
    growing without bound was itself quadratic in the number of micro-batches — the check
    got slower exactly as the thing it watches got bigger.

    Args:
        name: The in-memory sink's registered name, quoted back in the error.
        held: Bytes the sink currently retains.

    Raises:
        ResourceError: If ``held`` exceeds the streaming-state budget.
    """
    from batcher.config import active_config

    cap = active_config().memory.streaming_state_budget_bytes()
    if held > cap:
        from batcher._internal.errors import ResourceError

        raise ResourceError(
            f"in-memory streaming sink {name!r} reached {held} bytes (cap {cap}): it "
            "retains every micro-batch in append/update mode and never evicts. Use a "
            "durable sink (parquet/delta) for an unbounded stream, switch to "
            "outputMode='complete', or raise memory.streaming_state_max_bytes."
        )


def memory_table(name: str) -> pa.Table:
    """Return the accumulated table for a named in-memory streaming sink.

    Raises `KeyError` (surfaced as a clear error by the caller) if no query has
    written to `name`.
    """
    parts = _MEMORY[name]
    if parts:
        return pa.concat_tables(parts)
    schema = _MEMORY_SCHEMA.get(name)
    return pa.Table.from_batches([], schema=schema) if schema is not None else pa.table({})


@STREAM_SINKS.register("memory")
class MemoryStreamSink:
    """Accumulate micro-batches in memory under `name`, queryable via `bt.read_memory`.

    For `complete` output mode the table is replaced each micro-batch (the running
    result is the whole answer); for `append`/`update` it grows. The mode is passed
    by the engine so the sink keeps only what the semantics require.
    """

    def __init__(
        self,
        name: str,
        *,
        output_mode: str = "append",
        schema: pa.Schema | None = None,
        **_: Any,
    ) -> None:
        self._name = name
        self._replace = output_mode == "complete"
        self._held = 0
        # The query's output schema, when the conductor knew it. It is what lets an
        # empty sink read back as the query's relation instead of as no relation.
        self._schema = schema

    def open(self) -> None:
        _MEMORY[self._name] = []
        self._held = 0
        if self._schema is not None:
            _MEMORY_SCHEMA[self._name] = self._schema
        else:
            _MEMORY_SCHEMA.pop(self._name, None)

    def write_batch(self, _batch_id: int, table: pa.Table) -> str | None:
        # Learn the schema from the data when the conductor did not supply one, so a
        # `complete`-mode sink whose running result later empties still reads back typed.
        _MEMORY_SCHEMA.setdefault(self._name, table.schema)
        if self._replace:
            _MEMORY[self._name] = [table]
        else:
            _MEMORY[self._name].append(table)
            self._held += table.nbytes
            _check_memory_sink_size(self._name, self._held)
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
        return f"foreach_batch:{batch_id}"

    def close(self) -> None:
        pass


@STREAM_SINKS.register("foreach")
class ForeachStreamSink:
    """Call ``fn(row)`` for each row of each micro-batch (Spark `foreach`).

    Convenience over `foreach_batch` for row-at-a-time external writes; the batch is
    converted to row dicts in one vectorized `to_pylist` per record batch (no per-element
    Python in the engine's hot path — the iteration is the user's chosen sink semantics).

    Converting a *chunk* at a time rather than the whole table is what keeps the sink
    usable on a large micro-batch. `Table.to_pylist()` builds one Python list holding a dict
    per row of the entire micro-batch before the first `fn(row)` runs, and a Python dict of
    boxed scalars costs on the order of a hundred bytes per column — so a micro-batch that
    fits comfortably in Arrow can be an order of magnitude larger as Python objects, and the
    peak lands on top of the Arrow table it was built from. Per-record-batch conversion caps
    that transient at one morsel and lets each chunk's dicts be collected as it goes.
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Any], **_: Any) -> None:
        self._fn = fn

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        fn = self._fn
        for record_batch in table.to_batches():
            for row in record_batch.to_pylist():
                fn(row)
        return f"foreach:{batch_id}"

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
    """Append each micro-batch to a Delta table as exactly one idempotent transaction.

    Two properties the log must have, and how they are obtained:

    **Exactly one transaction per micro-batch.** Each batch gets a fresh `delta` sink
    that writes its data file and commits it, so the version history is a one-to-one
    record of the micro-batches — never one commit per worker, and never a commit per
    file.

    **Exactly-once under replay.** The engine records a micro-batch's source offset
    *before* processing it, so a crash between processing and committing leaves a batch
    the next run replays. A plain append would then write those rows a second time.
    Instead every commit carries a Delta ``txn`` action — ``(app_id, batch_id)`` — and
    the sink checks the log for it first: a replayed batch finds its own transaction
    already recorded, writes nothing, and commits nothing. That check is what turns the
    engine's at-least-once replay into end-to-end exactly-once, and it is why the log
    ends up with exactly one transaction per micro-batch no matter how often one was
    retried.

    `app_id` must be *stable across restarts* or the idempotency check would never find
    the previous run's transactions. It is the query name when one was given, and
    otherwise derived from the destination table — stable either way.
    """

    def __init__(self, uri: str, *, query_name: str | None = None, **opts: Any) -> None:
        self._uri = uri
        self._app_id = query_name or f"batcher-stream:{uri.rstrip('/')}"
        opts.setdefault("mode", "append")
        self._opts = opts

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        from batcher._internal.errors import CommitError
        from batcher.io.formats import SINKS
        from batcher.io.manifest import WriteManifest

        sink = SINKS.get("delta")(app_id=self._app_id, txn_version=batch_id, **self._opts)
        # Check *before* writing: a replayed batch must not even leave an orphan data
        # file behind, let alone commit one.
        if sink.is_committed(self._uri):
            return f"delta:{batch_id}:already-committed"
        written = sink.write(table, self._uri)
        try:
            sink.commit(WriteManifest((written,), schema=table.schema), self._uri)
        except CommitError:
            # The pre-check is not atomic with the commit. A second writer sharing this
            # `app_id` — a concurrent driver, or this query racing its own restart —
            # can pass the same check, and the Delta log's optimistic concurrency
            # rejects the loser's transaction at commit time. Re-read the log: if this
            # `(app_id, batch_id)` transaction is now recorded, the batch is durably
            # committed and the conflict is benign (the loser's data file is an
            # uncommitted orphan the log ignores and vacuum reclaims). Re-raise only
            # when the transaction still is not there — a genuine commit failure.
            if not sink.is_committed(self._uri):
                raise
            return f"delta:{batch_id}:already-committed"
        return f"delta:{batch_id}:{written.rows}"

    def close(self) -> None:
        pass
