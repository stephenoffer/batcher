"""Streaming sinks — per-micro-batch writers for the streaming-query engine.

A `StreamSink` consumes one Arrow table per micro-batch (`open` → many
`write_batch` → `close`). It is the sink-side counterpart of the unbounded
`Source`: the engine in `core` drives the loop and hands each finished micro-batch
here.

Where a batch sink writes durable files it **reuses the existing batch machinery**
— `FileStreamSink` wraps any `SINKS` file format and writes one atomic
``part-batch<NNNNN>`` file per micro-batch (exactly-once by position, the same
property `resume=` relies on), and `TransactionalStreamSink` reuses the destination
table format's own transactional append. Only the genuinely-new targets (console,
memory, foreach-batch, foreach) are implemented here from scratch. The `write_batch`
return value is an opaque *sink token* the checkpoint commit-log records.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

from batcher._internal.registry import Registry
from batcher.plan.types import retained_bytes

__all__ = [
    "STREAM_SINKS",
    "ConsoleStreamSink",
    "DeltaStreamSink",
    "FileStreamSink",
    "ForeachBatchStreamSink",
    "ForeachStreamSink",
    "ForeachWriter",
    "MemoryStreamSink",
    "NoopStreamSink",
    "StreamSink",
    "TransactionalStreamSink",
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


#: Characters a truncated console cell keeps, matching Spark's ``truncate=True`` default.
_TRUNCATE_WIDTH = 20


@STREAM_SINKS.register("console")
class ConsoleStreamSink:
    """Print each micro-batch to stdout (the `console` sink — for development).

    `num_rows` and `truncate` are Spark's two console options. Truncation is on by
    default there and was absent here, which is the difference between a readable
    development stream and a terminal filled by one column of JSON blobs — the exact
    shape a Kafka stream carries. Only *display* is truncated; nothing downstream sees it.
    """

    def __init__(self, *, num_rows: int = 20, truncate: bool | int = True, **_: Any) -> None:
        self._num_rows = num_rows
        # Spark accepts `truncate=True` (20 chars) or an explicit width; both are useful,
        # and an int is what a reader reaches for the moment 20 turns out to be too narrow.
        self._truncate = _TRUNCATE_WIDTH if truncate is True else (truncate or 0)

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        print(f"-------- Batch: {batch_id} --------")
        shown = table.slice(0, self._num_rows)
        print(_truncate_strings(shown, self._truncate) if self._truncate else shown)
        return None

    def close(self) -> None:
        pass


def _truncate_strings(table: pa.Table, width: int) -> pa.Table:
    """Shorten every string column to `width` characters, for display only.

    Vectorized per column rather than per cell: this is a development sink, but it runs on
    every micro-batch of a stream someone left running, and a per-row Python loop over a
    wide table is the kind of cost that turns "print the output" into the slowest operator
    in the query.
    """
    import pyarrow.compute as pc

    columns = []
    for column in table.columns:
        if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
            column = pc.utf8_slice_codeunits(column, 0, width)
        columns.append(column)
    return pa.Table.from_arrays(columns, schema=table.schema)


@STREAM_SINKS.register("noop")
class NoopStreamSink:
    """Accept every micro-batch and write nothing (Spark ``format("noop")``).

    The benchmark sink. Measuring a pipeline through a real sink measures the sink too — a
    Parquet write is compression and fsyncs, the console sink is a terminal, the memory sink
    grows until the box dies. Discarding the rows leaves the read, the transform and the
    engine, which is the thing under test.

    It is not a no-op: `write_batch` counts what it was handed, so a run reports the rows it
    processed rather than a silent zero. A sink that swallowed the count would make "the
    pipeline is fast" and "the pipeline produced nothing" look identical, which is the one
    way a benchmark sink can actively mislead.
    """

    def __init__(self, **_: Any) -> None:
        self.rows_written = 0
        self.batches_written = 0

    def open(self) -> None:
        self.rows_written = 0
        self.batches_written = 0

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        self.rows_written += table.num_rows
        self.batches_written += 1
        return f"noop:{batch_id}:{table.num_rows}"

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


def _check_memory_sink_size(name: str, held: int, *, replaces: bool = False) -> None:
    """Raise a clear `ResourceError` if a named in-memory sink has outgrown the cap.

    In `append`/`update` mode this sink retains every micro-batch for the lifetime of
    the process, and `close()` frees nothing — so an unbounded stream written to it
    grows until the box dies, having looked healthy for hours. It is a debugging sink
    (the Spark `memory` sink is documented the same way), so the honest failure is an
    actionable error naming the sink, not a silent OOM. The cap is the same
    `memory.streaming_state_max_bytes` envelope that bounds watermark-held state.

    `complete` mode is checked too, and used not to be. Replacing the table each
    micro-batch bounds the sink by the *running result* rather than by the stream, which
    is why the append-mode guard did not apply — but a running result is not small by
    construction: a `group_by` over a high-cardinality key is exactly the shape whose
    result grows with the stream, and it is the shape a `complete`-mode query is usually
    written for. So the one mode with no guard was the one whose remedy the other mode's
    error message recommends.

    `held` is passed in rather than recomputed. Summing `nbytes` across every retained
    table on every micro-batch is O(batches) work per batch, so the guard against the sink
    growing without bound was itself quadratic in the number of micro-batches — the check
    got slower exactly as the thing it watches got bigger.

    Args:
        name: The in-memory sink's registered name, quoted back in the error.
        held: Bytes the sink currently retains.
        replaces: Whether the sink replaces its contents each micro-batch (`complete`
            mode), which decides what the error can usefully suggest.

    Raises:
        ResourceError: If ``held`` exceeds the streaming-state budget.
    """
    from batcher.config import active_config

    cap = active_config().memory.streaming_state_budget_bytes()
    if held <= cap:
        return
    from batcher._internal.errors import ResourceError

    remedy = (
        "its running result alone exceeds the budget, so no output mode makes it smaller. "
        "Narrow the grouping key, use a durable sink (parquet/delta), or raise "
        "memory.streaming_state_max_bytes."
        if replaces
        else "it retains every micro-batch in append/update mode and never evicts. Use a "
        "durable sink (parquet/delta) for an unbounded stream, switch to "
        "outputMode='complete', or raise memory.streaming_state_max_bytes."
    )
    raise ResourceError(
        f"in-memory streaming sink {name!r} reached {held} bytes (cap {cap}): {remedy}"
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
            self._held = retained_bytes(table)
        else:
            _MEMORY[self._name].append(table)
            self._held += retained_bytes(table)
        _check_memory_sink_size(self._name, self._held, replaces=self._replace)
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

    **A `ForeachWriter` object works too**, which is how Spark's `foreach` is usually
    written for a real destination: `open(partition_id, epoch_id)` acquires the connection
    and returns whether to proceed, `process(row)` writes one row, and `close(error)`
    releases it — including when the epoch failed, which is the whole reason the shape
    exists. A plain function has nowhere to put a connection, so every call either reopens
    one or leans on a module-level global.
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Any] | Any, **_: Any) -> None:
        # A writer object is duck-typed on `process`, so any of the three Spark shapes
        # works: a bare function, a class instance with `process`, or one with all three.
        self._writer = fn if hasattr(fn, "process") else None
        self._fn = fn if self._writer is None else fn.process

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        if self._writer is None:
            self._consume(self._fn, table)
            return f"foreach:{batch_id}"
        return self._write_through_writer(batch_id, table)

    def _write_through_writer(self, batch_id: int, table: pa.Table) -> str | None:
        """Run one epoch through a `ForeachWriter`'s open / process / close lifecycle.

        `close` runs in a `finally` and receives the failure, so a destination connection
        is released whether the epoch succeeded or not — the property a bare function
        cannot offer, and the reason Spark's own docs steer real sinks to this shape.
        Partition id is 0: a single-node micro-batch is one partition, and inventing a
        number would make a user's per-partition bookkeeping quietly wrong.
        """
        opener = getattr(self._writer, "open", None)
        if opener is not None and opener(0, batch_id) is False:
            return f"foreach:{batch_id}:skipped"
        error: BaseException | None = None
        try:
            self._consume(self._fn, table)
        except BaseException as exc:
            error = exc
            raise
        finally:
            closer = getattr(self._writer, "close", None)
            if closer is not None:
                closer(error)
        return f"foreach:{batch_id}"

    @staticmethod
    def _consume(fn: Callable[[dict[str, Any]], Any], table: pa.Table) -> None:
        """Hand every row to `fn`, one record batch's worth of dicts at a time."""
        for record_batch in table.to_batches():
            for row in record_batch.to_pylist():
                fn(row)

    def close(self) -> None:
        pass


class ForeachWriter:
    """The open / process / close shape `write.for_each` accepts (Spark `ForeachWriter`).

    Subclass it for a destination that needs a connection: `open` acquires one and returns
    whether to proceed with the epoch, `process` writes a row, `close` releases it — with
    the exception when the epoch failed, so cleanup is not conditional on success.

    Nothing requires the base class; the sink duck-types on `process`. It exists so the
    contract is written down once and so `isinstance` works for a caller that wants it.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> class ToApi(bt.ForeachWriter):
            ...     def open(self, partition_id, epoch_id):
            ...         self.sent = []
            ...         return True
            ...     def process(self, row):
            ...         self.sent.append(row)
            ...     def close(self, error):
            ...         pass
            >>> writer = ToApi()
            >>> stream = bt.read.kafka("events")  # doctest: +SKIP
            >>> query = stream.write.for_each(writer)  # doctest: +SKIP
    """

    def open(self, partition_id: int, epoch_id: int) -> bool:  # noqa: ARG002 - override point
        """Acquire whatever this epoch needs; return False to skip it entirely.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.ForeachWriter().open(0, 3)
                True

        Args:
            partition_id: The partition being written. ``0`` on the single-node path,
                where one micro-batch is one partition.
            epoch_id: The micro-batch id, stable across a replay of the same epoch.

        Returns:
            True to process the epoch, False to skip it.
        """
        return True

    def process(self, row: dict[str, Any]) -> None:
        """Write one row to the destination.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.ForeachWriter().process({"x": 1}) is None
                True

        Args:
            row: One output row, as a dict of column name to value.
        """

    def close(self, error: BaseException | None) -> None:
        """Release the epoch's resources, whether or not it succeeded.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.ForeachWriter().close(None) is None
                True

        Args:
            error: The exception that ended the epoch, or None when it completed.
        """


class FileStreamSink:
    """Write one atomic ``part-batch<NNNNN>`` file per micro-batch via a batch sink.

    Reuses any registered `SINKS` file format (parquet/csv/json/…). `resume=True`
    skips a part file already present, giving exactly-once output by batch position
    when the source offsets are replayable (Workstream D). The output directory is a
    valid dataset the existing readers can scan.

    `max_rows_per_file` caps each output file, splitting a micro-batch across as many as
    it needs. Without it a batch is one file whatever its size, which on a long-running
    stream is the small-files problem in its purest form — the file size is whatever the
    trigger interval happened to produce, and nothing in the query says otherwise. The
    chunk index joins the batch id in the name, so every file is still named by position
    and `resume` still recognises what it already wrote.
    """

    def __init__(
        self,
        path: str,
        fmt: str,
        *,
        resume: bool = True,
        max_rows_per_file: int | None = None,
        **opts: Any,
    ) -> None:
        from batcher.io.formats import SINKS

        self._path = path.rstrip("/")
        self._sink = SINKS.get(fmt)(**opts)
        self._suffix = getattr(self._sink, "suffix", "")
        self._resume = resume
        self._max_rows_per_file = max_rows_per_file

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        """Write one micro-batch, as one file or as row-capped chunks of one.

        Returns the first file's path as the batch's token: the name identifies the batch
        by position, and any further chunks are the same name with a rising chunk index.
        """
        cap = self._max_rows_per_file
        if cap is None or table.num_rows <= cap:
            file_path = f"{self._path}/part-batch{batch_id:05d}{self._suffix}"
            return self._sink.write(table, file_path, resume=self._resume).path
        first: str | None = None
        for chunk_index, start in enumerate(range(0, table.num_rows, cap)):
            chunk_path = f"{self._path}/part-batch{batch_id:05d}-{chunk_index:05d}{self._suffix}"
            written = self._sink.write(table.slice(start, cap), chunk_path, resume=self._resume)
            first = first or written.path
        return first

    def close(self) -> None:
        pass


class TransactionalStreamSink:
    """Append each micro-batch to a *table* format as one transaction.

    Two properties the log should have, and how they are obtained:

    **Exactly one transaction per micro-batch.** Each batch gets a fresh sink for the
    destination format that writes its data file and commits it, so the version history
    is a one-to-one record of the micro-batches — never one commit per worker, and never
    a commit per file.

    **Exactly-once under replay, where the format can do it.** The engine records a
    micro-batch's source offset *before* processing it, so a crash between processing and
    committing leaves a batch the next run replays. A plain append would then write those
    rows a second time. A format with an application-transaction log — Delta, via its
    ``txn`` action carrying ``(app_id, batch_id)`` — is asked first whether it has already
    recorded this batch: a replayed one finds its own transaction, writes nothing, and
    commits nothing. That check is what turns the engine's at-least-once replay into
    end-to-end exactly-once.

    `app_id` must be *stable across restarts* or the idempotency check would never find
    the previous run's transactions. It is the query name when one was given, and
    otherwise derived from the destination table — stable either way.

    **A format without that log gets an ordinary append, and says so.** Iceberg and Hudi
    have no ``(app_id, batch_id)`` marker here, so a replayed epoch appends its rows
    again. That is at-least-once, and the sink warns once at `open()` rather than letting
    a reader assume the exactly-once story above applies to every table format. What it
    must never do is what this class replaced: `DeltaStreamSink` hard-coded
    ``SINKS.get("delta")`` while the conductor routed *every* mode-aware format to it, so
    ``write(path, format="iceberg", trigger=...)`` silently produced a Delta table — right
    rows, right path, wrong format, no error anywhere.
    """

    def __init__(
        self, uri: str, fmt: str = "delta", *, query_name: str | None = None, **opts: Any
    ) -> None:
        self._uri = uri
        self._fmt = fmt
        self._app_id = query_name or f"batcher-stream:{uri.rstrip('/')}"
        opts.setdefault("mode", "append")
        self._opts = opts

    def _idempotent(self) -> bool:
        """Whether this format can recognize a replayed micro-batch by its transaction id."""
        from batcher.io.formats import SINKS

        return hasattr(SINKS.get(self._fmt), "is_committed")

    def _new_sink(self, batch_id: int) -> Any:
        """A fresh sink for one micro-batch, carrying its transaction id where supported."""
        from batcher.io.formats import SINKS
        from batcher.io.sink import table_sink_kwargs

        cls = SINKS.get(self._fmt)
        # A table format may need its destination at *construction* — Iceberg's identifier
        # and per-write token. The batch write path knew that; this one did not, and failed
        # on the first micro-batch with a TypeError out of the constructor.
        extra = table_sink_kwargs(self._fmt, self._uri)
        if self._idempotent():
            return cls(app_id=self._app_id, txn_version=batch_id, **extra, **self._opts)
        return cls(**extra, **self._opts)

    def open(self) -> None:
        """Warn once when the destination format cannot absorb a replayed micro-batch."""
        if self._idempotent():
            return
        import warnings

        warnings.warn(
            f"the {self._fmt!r} streaming sink has no per-batch transaction marker, so a "
            "micro-batch replayed after a failure appends its rows a second time "
            "(at-least-once). Use format='delta' for exactly-once, or dedup downstream.",
            stacklevel=2,
        )

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        from batcher._internal.errors import CommitError
        from batcher.io.manifest import WriteManifest

        sink = self._new_sink(batch_id)
        idempotent = self._idempotent()
        # Check *before* writing: a replayed batch must not even leave an orphan data
        # file behind, let alone commit one.
        if idempotent and sink.is_committed(self._uri):
            return f"{self._fmt}:{batch_id}:already-committed"
        written = sink.write(table, self._uri)
        try:
            sink.commit(WriteManifest((written,), schema=table.schema), self._uri)
        except CommitError:
            # The pre-check is not atomic with the commit. A second writer sharing this
            # `app_id` — a concurrent driver, or this query racing its own restart —
            # can pass the same check, and the log's optimistic concurrency rejects the
            # loser's transaction at commit time. Re-read the log: if this
            # `(app_id, batch_id)` transaction is now recorded, the batch is durably
            # committed and the conflict is benign (the loser's data file is an
            # uncommitted orphan the log ignores and vacuum reclaims). Re-raise only
            # when the transaction still is not there — a genuine commit failure, and
            # always for a format that cannot answer the question.
            if not idempotent or not sink.is_committed(self._uri):
                raise
            return f"{self._fmt}:{batch_id}:already-committed"
        return f"{self._fmt}:{batch_id}:{written.rows}"

    def close(self) -> None:
        pass


class DeltaStreamSink(TransactionalStreamSink):
    """`TransactionalStreamSink` pinned to Delta — the exactly-once streaming table sink.

    Kept as its own name because Delta is the one format that carries the
    ``(app_id, batch_id)`` transaction the engine's replay story depends on, and because
    callers that mean Delta specifically should say so.
    """

    def __init__(self, uri: str, *, query_name: str | None = None, **opts: Any) -> None:
        super().__init__(uri, "delta", query_name=query_name, **opts)
