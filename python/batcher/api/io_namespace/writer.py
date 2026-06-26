"""The `ds.write` namespace — typed, per-format dataset sinks.

``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
is the explicit spelling. Methods are thin wrappers over `terminal._write` and the
merge helpers; sink implementations live in `io/formats/` and register into the
`SINKS` registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher.api.session import read as _read

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.api.dataset import Dataset
    from batcher.api.streaming import StreamingQuery
    from batcher.io.manifest import WriteManifest
    from batcher.plan.streaming import Trigger

__all__ = ["Writer"]


# Save modes (Spark `SaveMode` parity). `append` is only meaningful for the
# transactional lakehouse sinks, which consume `mode` as a constructor option; the
# file sinks always overwrite, so for them `mode` only drives the existence gate.
_SAVE_MODES = ("overwrite", "error", "ignore", "append")
_MODE_AWARE_SINKS = frozenset({"delta", "iceberg", "hudi"})
# Sinks with a native transactional MERGE; others fall back to a copy-on-write
# file merge (`api.merge.compose_file_merge`).
_MERGE_NATIVE_SINKS = frozenset({"delta"})


class Writer:
    """The `ds.write` namespace: callable for autodetect, typed methods per format.

    ``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
    is the explicit spelling. All methods accept `partition_by=` (Hive directory) and
    `distributed=`/`num_workers=` (parallel shard write + atomic commit), and return
    a `WriteManifest`.
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        """Bind the write namespace to the `Dataset` whose result it writes."""
        self._ds = ds

    def __call__(
        self,
        path: str,
        format: str | None = None,
        *,
        mode: str = "overwrite",
        partition_by: list[str] | None = None,
        distributed: bool = False,
        num_workers: int | None = None,
        resume: bool = False,
        max_rows_per_file: int | None = None,
        sort_by: list[str] | None = None,
        replace_where: Any = None,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        checkpoint: str | None = None,
        query_name: str | None = None,
        **opts: Any,
    ) -> WriteManifest | StreamingQuery:
        """Execute and write the result, inferring `format` from the path when omitted.

        **Batch vs streaming is one API.** With a bounded source and no `trigger`,
        this is a one-shot write returning a `WriteManifest` (the behavior below).
        If `trigger` is set, or any source is unbounded (a stream), the write runs as
        a streaming query — micro-batches are appended to `path` per the `trigger`
        cadence and `output_mode` — and it returns a `StreamingQuery` handle instead.

        `mode` is the save mode (Spark ``SaveMode`` parity):

        * ``"overwrite"`` (default) — write, replacing any existing output.
        * ``"error"`` — raise `PlanError` if `path` already exists.
        * ``"ignore"`` — skip the write (return an empty manifest) if `path` exists.
        * ``"append"`` — add to an existing table; only the transactional lakehouse
          sinks (`delta`/`iceberg`/`hudi`) support it (others raise).

        ``replace_where=<predicate>`` is a dynamic partition/range overwrite (Delta
        ``replaceWhere`` / the backfill pattern): atomically replace only the rows
        matching the predicate and keep the rest. This is predicate-scoped, not
        key-matched — for a key-matched upsert (update/insert by join key) use
        `merge` instead.

        ``sort_by=[cols]`` clusters the output: rows are sorted (ascending) before
        writing, so each file / row-group's min/max bounds are tight and downstream
        queries skip far more data via zonemaps and bloom filters — the engine-side
        slice of liquid clustering (you choose the keys; there is no managed table
        service). Bounded batch writes only.

        ``resume=True`` makes the write idempotent: output files already present
        (necessarily fully committed, since writes are atomic) are skipped, so a job
        re-run after a crash or spot preemption finishes only the unwritten shards —
        the resumability Ray Data lacks without external bookkeeping.

        **Correctness precondition (important):** resume identifies done work by file
        *position* (``part-NNNNN``), so it is exactly-once **only when the plan is
        deterministic** — the same input produces the same rows in the same order, so
        a given part file holds the same rows on every run. This holds for the
        read → ``map_batches``/``filter``/``select`` → write (ETL / batch-inference)
        path. It does **not** hold for a plan whose row→file assignment can vary
        between runs — a ``group_by``/``join``/``sort`` or distributed shuffle, where
        ordering is hash- and worker-count-dependent. Resuming such a plan could skip
        a file that now holds *different* rows, dropping or duplicating data. For
        those, write to a fresh path (no resume) or materialize a stable, keyed
        intermediate first.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> out = os.path.join(tempfile.mkdtemp(), "t.parquet")
                >>> _ = bt.from_pydict({"x": [1, 2, 3]}).write(out)
                >>> bt.read(out).count()
                3
        """
        from batcher._internal.errors import PlanError
        from batcher.api.terminal import _write
        from batcher.io.detect import detect_format
        from batcher.io.manifest import WriteManifest
        from batcher.io.source import is_bounded

        if mode not in _SAVE_MODES:
            raise PlanError(f"write(): unknown mode {mode!r}; use one of {list(_SAVE_MODES)}")
        fmt = detect_format(path, format)

        # `sort_by` clusters the output: sort rows (ascending) before writing so each
        # file / row-group's min/max bounds are tight, maximizing downstream zonemap +
        # bloom skipping (the engine-side slice of liquid clustering — no managed table
        # service). Delegate to a sorted dataset so all the write logic below runs on it.
        # A global sort is a bounded-batch notion; refuse it on an unbounded stream.
        if sort_by is not None:
            if trigger is not None or any(not is_bounded(s) for s in self._ds._sources):
                raise PlanError(
                    "write(sort_by=...) clusters a bounded batch write; it cannot sort an "
                    "unbounded stream"
                )
            return self._ds.sort(*sort_by).write(
                path,
                format,
                mode=mode,
                partition_by=partition_by,
                distributed=distributed,
                num_workers=num_workers,
                resume=resume,
                max_rows_per_file=max_rows_per_file,
                replace_where=replace_where,
                output_mode=output_mode,
                checkpoint=checkpoint,
                query_name=query_name,
                **opts,
            )

        # Unified surface: a trigger or an unbounded source means this is a streaming
        # write — append micro-batches to `path` and return a StreamingQuery.
        if trigger is not None or any(not is_bounded(s) for s in self._ds._sources):
            sink = self._stream_sink_for(path, fmt, opts)
            return self._start_stream(sink, trigger, output_mode, query_name, checkpoint)

        # Resume is exactly-once only on a deterministic plan: the same input must
        # produce the same rows in the same part file on every run. A pipeline breaker
        # (group_by, join, sort, distinct, window, union, limit) or its shuffle makes the
        # row-to-file assignment vary between runs, so resuming could skip a file that now
        # holds different rows, silently dropping or duplicating data. Refuse it up front.
        # The ETL and batch-inference path (read, map_batches, filter, select, write) stays
        # streamable, so it still resumes.
        if resume:
            from batcher.plan.logical import is_streamable

            if not is_streamable(self._ds._plan):
                raise PlanError(
                    "write(resume=True) is exactly-once only on a deterministic plan "
                    "(read → map_batches / filter / select → write). This plan contains a "
                    "group_by / join / sort / distinct / window / union / limit, whose "
                    "row→file assignment can vary between runs, so resuming risks dropping "
                    "or duplicating rows. Write to a fresh path without resume, or "
                    "materialize a stable keyed intermediate first."
                )

        # `replace_where` = dynamic partition/range overwrite (Delta `replaceWhere` /
        # the backfill pattern): atomically replace only the rows matching the
        # predicate, preserving the rest. Copy-on-write: keep the existing rows
        # *outside* the range, union the new data, overwrite. Single-writer only.
        if replace_where is not None:
            from batcher.io.filesystem import resolve_filesystem

            if resolve_filesystem(path).exists(path):
                kept = _read(path, format=fmt).filter(~replace_where)
                combined = kept.union(self._ds)
                return combined.write(
                    path,
                    fmt,
                    mode="overwrite",
                    partition_by=partition_by,
                    max_rows_per_file=max_rows_per_file,
                    **opts,
                )
        if mode == "append" and fmt not in _MODE_AWARE_SINKS:
            raise PlanError(
                f"write(): mode='append' is only supported for {sorted(_MODE_AWARE_SINKS)}, "
                f"not {fmt!r} (use a fresh path, or 'overwrite')"
            )
        # error/ignore are a pre-write existence gate (resume has its own per-file
        # idempotence, so it is exempt).
        if mode in ("error", "ignore") and not resume:
            from batcher.io.filesystem import resolve_filesystem

            if resolve_filesystem(path).exists(path):
                if mode == "error":
                    raise PlanError(f"write(): path {path!r} already exists and mode='error'")
                return WriteManifest()  # ignore: leave the existing output untouched

        # The lakehouse sinks consume append/overwrite as a constructor option; the
        # file sinks always overwrite, so `mode` only drives the gate above for them.
        sink_kwargs = dict(opts)
        if fmt in _MODE_AWARE_SINKS:
            sink_kwargs["mode"] = mode if mode in ("append", "overwrite") else "overwrite"

        # A `repartition(...)` layout (set via ds.repartition) supplies write defaults:
        # `by` Hive-partitions, `num_files`/`target_size_mb` set the per-file row cap
        # (resolved post-materialization in `_write`).
        num_files: int | None = None
        target_bytes: int | None = None
        spec = self._ds._repartition
        if spec is not None:
            if spec.by and partition_by is None:
                partition_by = list(spec.by)
            num_files = spec.num_files
            if spec.target_size_mb is not None:
                target_bytes = int(spec.target_size_mb * 1024 * 1024)

        return _write(
            self._ds._plan,
            self._ds._sources,
            self._ds.columns,
            path,
            fmt,
            partition_by=partition_by,
            distributed=distributed,
            num_workers=num_workers,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
            num_files=num_files,
            target_bytes_per_file=target_bytes,
            sink_kwargs=sink_kwargs,
        )

    # --- streaming sink targets -------------------------------------------
    def _start_stream(
        self,
        sink: Any,
        trigger: Trigger | None,
        output_mode: str,
        query_name: str | None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Launch a streaming query writing this dataset's stream to `sink`."""
        from batcher.api.streaming import start_streaming_query

        return start_streaming_query(
            self._ds._plan,
            self._ds._sources,
            sink,
            trigger=trigger,
            output_mode=output_mode,
            name=query_name,
            checkpoint=checkpoint,
        )

    def _stream_sink_for(self, path: str, fmt: str, opts: dict[str, Any]) -> Any:
        """Build the per-micro-batch `StreamSink` for a path/format streaming write."""
        from batcher.io.formats.streaming.sinks import DeltaStreamSink, FileStreamSink

        if fmt in _MODE_AWARE_SINKS:
            return DeltaStreamSink(path, **opts)
        return FileStreamSink(path, fmt, **opts)

    def console(
        self,
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        num_rows: int = 20,
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream each micro-batch to stdout (development sink).

        Examples:
            .. code-block:: python

                import batcher as bt

                stream = bt.read.json("events/", stream=True)
                query = stream.write.console(
                    num_rows=10, trigger=bt.Trigger.processing_time("5 seconds")
                )
                query.await_termination()
        """
        from batcher.io.formats.streaming.sinks import ConsoleStreamSink

        return self._start_stream(
            ConsoleStreamSink(num_rows=num_rows), trigger, output_mode, query_name, checkpoint
        )

    def memory(
        self,
        name: str,
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream into an in-memory table queryable via ``bt.read_memory(name)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> query = ds.write.memory("scratch")
                >>> _ = query.await_termination()
                >>> bt.read_memory("scratch").count()
                3
        """
        from batcher.io.formats.streaming.sinks import MemoryStreamSink

        return self._start_stream(
            MemoryStreamSink(name, output_mode=output_mode),
            trigger,
            output_mode,
            query_name,
            checkpoint,
        )

    def for_each_batch(
        self,
        fn: Callable[[pa.Table, int], Any],
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream each micro-batch into a user callback ``fn(table, batch_id)``.

        The callback gets the whole Arrow table for the micro-batch — the sanctioned
        hook for custom upserts (``MERGE``/SCD), multi-sink fan-out, or any per-batch
        commit logic (the sink-side twin of `map_batches`).

        Examples:
            .. code-block:: python

                import batcher as bt

                def upsert(table, batch_id):
                    print(f"batch {batch_id}: {table.num_rows} rows")
                    table.to_pandas().to_sql("events", engine, if_exists="append")

                stream = bt.read.json("events/", stream=True)
                query = stream.write.for_each_batch(upsert)
                query.await_termination()
        """
        from batcher.io.formats.streaming.sinks import ForeachBatchStreamSink

        return self._start_stream(
            ForeachBatchStreamSink(fn), trigger, output_mode, query_name, checkpoint
        )

    def for_each(
        self,
        fn: Callable[[dict[str, Any]], Any],
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream each row of each micro-batch into a user callback ``fn(row)``.

        Examples:
            .. code-block:: python

                import batcher as bt

                def send(row):
                    requests.post("https://example.com/ingest", json=row)

                stream = bt.read.json("events/", stream=True)
                query = stream.write.for_each(send)
                query.await_termination()
        """
        from batcher.io.formats.streaming.sinks import ForeachStreamSink

        return self._start_stream(
            ForeachStreamSink(fn), trigger, output_mode, query_name, checkpoint
        )

    def parquet(self, path: str, *, compression: str = "zstd", **opts: Any) -> WriteManifest:
        """Write as Parquet (see `__call__` for `partition_by`/`distributed`).

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.parquet(out)
                >>> bt.read.parquet(out).count()
                3
        """
        return self(path, "parquet", compression=compression, **opts)

    def csv(self, path: str, **opts: Any) -> WriteManifest:
        """Write as CSV.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.csv(out)
                >>> bt.read.csv(out).count()
                3
        """
        return self(path, "csv", **opts)

    def json(self, path: str, **opts: Any) -> WriteManifest:
        """Write as newline-delimited JSON.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.json(out)
                >>> bt.read.json(out).count()
                3
        """
        return self(path, "json", **opts)

    def orc(self, path: str, **opts: Any) -> WriteManifest:
        """Write as ORC.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.orc(out)
                >>> bt.read.orc(out).count()
                3
        """
        return self(path, "orc", **opts)

    def arrow(self, path: str, **opts: Any) -> WriteManifest:
        """Write as Arrow/Feather IPC.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.arrow(out)
                >>> bt.read.arrow(out).count()
                3
        """
        return self(path, "arrow", **opts)

    def avro(self, path: str, **opts: Any) -> WriteManifest:
        """Write as Avro (needs ``batcher-engine[avro]``).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"x": [1, 2, 3]})
                ds.write.avro("data.avro")
        """
        return self(path, "avro", **opts)

    def lance(self, path: str, **opts: Any) -> WriteManifest:
        """Write a Lance dataset (needs ``batcher-engine[lance]``).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"x": [1, 2, 3]})
                ds.write.lance("data.lance")
        """
        return self(path, "lance", **opts)

    def msgpack(self, path: str, **opts: Any) -> WriteManifest:
        """Write as MessagePack (needs ``batcher-engine[msgpack]``).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"x": [1, 2, 3]})
                ds.write.msgpack("data.msgpack")
        """
        return self(path, "msgpack", **opts)

    def merge(
        self,
        target: str,
        *,
        on: str | list[str],
        when_matched: str = "update",
        when_not_matched: str = "insert",
        format: str | None = None,
        **opts: Any,
    ) -> WriteManifest:
        """Upsert (``MERGE INTO``) this dataset into an existing `target`, keyed on `on`.

        For a transactional sink (Delta) this delegates to the native ``MERGE``. For a
        plain file target it is a copy-on-write merge: read the current target,
        reconcile with this source (``when_matched`` ∈ ``update``/``delete``,
        ``when_not_matched`` ∈ ``insert``/``ignore``), and atomically overwrite. If the
        target does not exist yet, the source is written as-is (all inserts).

        File merge is read-modify-write over the whole path — single-writer only; use a
        Delta target for concurrent writers.

        Use `merge` for **key-matched upserts** (reconcile rows by `on`). For replacing
        a known slice of a partitioned table wholesale — a backfill or idempotent
        partition reload — use ``write(path, replace_where=<predicate>)`` instead, which
        overwrites every row matching the predicate regardless of keys.

        Examples:
            .. code-block:: python

                import batcher as bt

                updates = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                updates.write.merge(
                    "warehouse/orders",
                    on="id",
                    when_matched="update",
                    when_not_matched="insert",
                )
        """
        from batcher.api.merge import execute_merge

        return execute_merge(
            self,
            target,
            on=on,
            when_matched=when_matched,
            when_not_matched=when_not_matched,
            format=format,
            native_sinks=_MERGE_NATIVE_SINKS,
            opts=opts,
        )

    # --- Lakehouse / catalog ----------------------------------------------
    def delta(
        self,
        uri: str,
        *,
        mode: str = "append",
        merge_on: str | list[str] | None = None,
        **opts: Any,
    ) -> WriteManifest:
        """Write to a Delta Lake table (one transactional commit).

        With `merge_on`, performs a ``MERGE INTO`` upsert keyed on those columns —
        matched rows are updated and new rows inserted (Spark/Delta ``MERGE``). The
        keys build the match predicate; pass `merge_predicate=` instead for a custom
        one. Otherwise `mode` is ``"append"`` (default) or ``"overwrite"``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                ds.write.delta("warehouse/orders", merge_on="id")
        """
        if merge_on is not None:
            from batcher.api.merge import merge_predicate_for

            opts["merge_predicate"] = merge_predicate_for(merge_on)
        return self(uri, "delta", mode=mode, **opts)

    def iceberg(self, identifier: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Iceberg table (``mode="append"|"overwrite"``).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                ds.write.iceberg("db.orders", mode="append")
        """
        return self(identifier, "iceberg", mode=mode, **opts)

    def hudi(self, table_uri: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Apache Hudi table (``mode="append"|"overwrite"``).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                ds.write.hudi("s3://lake/orders", mode="append")
        """
        return self(table_uri, "hudi", mode=mode, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a database table via ADBC/FlightSQL.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                ds.write.sql("orders", uri="postgresql://localhost/warehouse")
        """
        return self(table, "adbc", **opts)

    def snowflake(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a Snowflake table.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                ds.write.snowflake("ORDERS", account="acct", warehouse="WH", database="DB")
        """
        return self(table, "snowflake", **opts)

    def mongo(self, collection: str, **opts: Any) -> WriteManifest:
        """Write to a MongoDB collection.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                ds.write.mongo("orders", uri="mongodb://localhost:27017", database="shop")
        """
        return self(collection, "mongo", **opts)
