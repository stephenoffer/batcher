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
    from batcher.api.merge import MergeBuilder
    from batcher.api.streaming import StreamingQuery
    from batcher.io.manifest import WriteManifest
    from batcher.plan.streaming import Trigger

__all__ = ["Writer"]


# Save modes (Spark `SaveMode` parity). `append` is only meaningful for the sinks that
# can add to an existing target — the transactional lakehouse tables and the warehouse
# tables — which consume `mode` as a constructor option; the file sinks always overwrite,
# so for them `mode` only drives the existence gate.
#
# A sink that honors `mode` MUST be listed here. `snowflake` was not, and the result was
# the worst of both: `mode="append"` was rejected even though `SnowflakeSink` implements
# it, and `mode="overwrite"` passed the gate but never reached the sink, so the write
# quietly appended instead. A save mode that silently does the opposite of what it says is
# a data-corruption bug, not a missing feature.
_SAVE_MODES = ("overwrite", "error", "ignore", "append")
_MODE_AWARE_SINKS = frozenset({"delta", "iceberg", "hudi", "snowflake"})


def _is_distributable_aggregate(plan: Any) -> bool:
    """Whether `plan` is a streaming aggregation the cluster can fold in parallel.

    A top-level `Aggregate` over a breaker-free input is exactly the shape the mergeable
    algebra covers: each worker runs `partial` on its share of the epoch and the driver
    `combine`s the partials into the running state. Anything with a *second* breaker under
    it (a sort, a join) has no such decomposition here, and is refused rather than run with
    different semantics than the single-node path.
    """
    from batcher.plan.logical import Aggregate, is_streamable

    return isinstance(plan, Aggregate) and is_streamable(plan.input)


class Writer:
    """The `ds.write` namespace: callable for autodetect, typed methods per format.

    ``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
    is the explicit spelling. All methods accept `partition_by=` (Hive directory) and
    `distributed=`/`num_workers=` (parallel shard write + atomic commit), and return
    a `WriteManifest`.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> out = os.path.join(tempfile.mkdtemp(), "t.parquet")
            >>> _ = bt.from_pydict({"x": [1, 2, 3]}).write(out)
            >>> bt.read(out).count()
            3
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
        distributed: bool | str = "auto",
        num_workers: int | None = None,
        resume: bool = False,
        max_rows_per_file: int | None = None,
        sort_by: list[str] | None = None,
        replace_where: Any = None,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        checkpoint: str | None = None,
        query_name: str | None = None,
        auto_compact: bool = False,
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
        #
        # The distributed *stream* drain stays EXPLICIT opt-in (`distributed=True`), not
        # `"auto"`: it supports only a stateless `available_now()` backfill and raises for
        # a checkpointed / continuous / stateful stream. Letting `"auto"` resolve it True
        # on a cluster would turn those into errors for users who never asked to
        # distribute. The batch path below resolves `"auto"` normally.
        if trigger is not None or any(not is_bounded(s) for s in self._ds._sources):
            if distributed is True:
                drain = self._maybe_distributed_stream(
                    path, fmt, opts, trigger, checkpoint, num_workers, query_name, output_mode
                )
                if drain is not None:
                    return drain
            sink = self._stream_sink_for(path, fmt, opts, query_name)
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

        # `replace_where` = dynamic partition/range overwrite (Delta `replaceWhere` / the
        # backfill pattern): atomically replace only the rows matching the predicate and
        # keep the rest.
        #
        # On a **Delta** table this is a scoped commit: the workers write the new
        # partition's files and the driver retires exactly the matching partitions from the
        # log. Nothing else is read and nothing else is rewritten, so backfilling one day of
        # a 100 TB table costs one day. The copy-on-write path below cannot do that — it
        # reads the *whole* table, filters out the replaced range, unions, and overwrites
        # everything, which turns a one-day backfill into a full-table rewrite. That is what
        # every lakehouse target used to do.
        if replace_where is not None:
            from batcher.io.filesystem import resolve_filesystem

            if fmt in ("delta", "iceberg"):
                # A lakehouse target scopes the overwrite to the predicate, inside its own
                # transaction. Iceberg used to fall through the `exists(path)` check below —
                # an Iceberg "path" is a catalog identifier, not a file, so the check was
                # always False, `replace_where` was silently dropped, and the write ran as a
                # plain overwrite that DELETED THE REST OF THE TABLE.
                opts = dict(opts)
                opts["replace_where"] = replace_where.to_ir()
                mode = "overwrite"
            elif resolve_filesystem(path).exists(path):
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
            auto_compact=auto_compact,
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

    def _maybe_distributed_stream(
        self,
        path: str,
        fmt: str,
        opts: dict[str, Any],
        trigger: Trigger | None,
        checkpoint: str | None,
        num_workers: int | None,
        query_name: str | None,
        output_mode: str = "append",
    ) -> StreamingQuery | None:
        """Run a `distributed=True` streaming write across the cluster, if eligible.

        Two distributed shapes, one for each kind of trigger:

        * a **drain** (`available_now`/`once`) over a bounded, splittable source is a
          parallel backfill — every worker drains its own partition once (Spark's
          `Trigger.AvailableNow`);
        * a **continuous / processing-time** stream runs each micro-batch as a cluster-wide
          epoch: the workers stage the epoch's data files and the driver publishes them as
          a single transaction, so the log records one transaction per micro-batch and a
          replayed epoch adds neither a row nor a commit.

        A benign mismatch (a source with nothing to fan out) returns ``None`` so the caller
        falls back to the single-node engine. A request we cannot honor *correctly* raises
        `PlanError` rather than silently ignoring the flag — or, worse, quietly delivering a
        weaker guarantee than the API implies.
        """
        from batcher._internal.errors import PlanError
        from batcher.api.streaming import _DRAIN_TRIGGER_KINDS, _is_stateless
        from batcher.dist.executor import _is_splittable_source

        srcs = self._ds._sources
        if len(srcs) != 1 or not _is_splittable_source(srcs[0]):
            return None  # not worth (or not able to) fan out — fall back to single-node

        drain = trigger is not None and trigger.kind in _DRAIN_TRIGGER_KINDS
        if drain and checkpoint is None:
            from batcher.api.streaming import start_distributed_stream_drain

            return start_distributed_stream_drain(
                self._ds._plan,
                srcs,
                path,
                fmt,
                opts,
                self._ds.columns,
                num_workers=num_workers,
                name=query_name,
            )

        if fmt in _MODE_AWARE_SINKS and fmt != "delta":
            raise PlanError(
                f"distributed streaming to {fmt!r} is not supported: its writer has no "
                "transaction-id check, so a replayed micro-batch would duplicate rows "
                "instead of being recognized as already-committed. Write to Delta for an "
                "exactly-once distributed stream, or run single-node (distributed=False)."
            )
        if not _is_stateless(self._ds._plan) and trigger is not None:
            if trigger.kind == "continuous":
                raise PlanError(
                    "a continuous trigger supports only stateless pipelines (filter / "
                    "select / map_batches); a streaming aggregation needs a micro-batch "
                    "boundary to fold — use Trigger.processing_time(...)"
                )
            if not _is_distributable_aggregate(self._ds._plan):
                raise PlanError(
                    "distributed streaming supports a stateless pipeline or a top-level "
                    "streaming aggregation; this plan has another pipeline breaker "
                    "(sort / join / window) — restructure it, or omit distributed."
                )
        from batcher.api.streaming import start_distributed_stream
        from batcher.plan.streaming import Trigger as _Trigger

        return start_distributed_stream(
            self._ds._plan,
            srcs,
            path,
            fmt,
            opts,
            trigger=trigger or _Trigger.processing_time(0),
            output_mode=output_mode,
            name=query_name,
            checkpoint=checkpoint,
            num_workers=num_workers,
        )

    def _stream_sink_for(
        self, path: str, fmt: str, opts: dict[str, Any], query_name: str | None = None
    ) -> Any:
        """Build the per-micro-batch `StreamSink` for a path/format streaming write.

        `query_name` becomes the transactional sink's Delta ``txn`` application id, which
        is what makes a restarted query's replayed micro-batch idempotent — so it has to
        reach the sink, not just the query engine.
        """
        from batcher.io.formats.streaming.sinks import DeltaStreamSink, FileStreamSink

        if fmt in _MODE_AWARE_SINKS:
            return DeltaStreamSink(path, query_name=query_name, **opts)
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

        Args:
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            num_rows: Rows to print per micro-batch (default 20).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running console stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> stream = bt.read.json("events/", stream=True)  # doctest: +SKIP
                >>> query = stream.write.console(  # doctest: +SKIP
                ...     num_rows=10, trigger=bt.Trigger.processing_time("5 seconds")
                ... )
                >>> query.await_termination()  # doctest: +SKIP
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

        Args:
            name: Name of the in-memory table to write into.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running in-memory stream.

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

        Args:
            fn: Callback ``fn(table, batch_id)`` invoked once per micro-batch.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> def upsert(table, batch_id):
                ...     print(f"batch {batch_id}: {table.num_rows} rows")
                >>> stream = bt.read.json("events/", stream=True)  # doctest: +SKIP
                >>> query = stream.write.for_each_batch(upsert)  # doctest: +SKIP
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

        Args:
            fn: Callback ``fn(row)`` invoked once per row, ``row`` a dict.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> def send(row):
                ...     print(row)
                >>> stream = bt.read.json("events/", stream=True)  # doctest: +SKIP
                >>> query = stream.write.for_each(send)  # doctest: +SKIP
        """
        from batcher.io.formats.streaming.sinks import ForeachStreamSink

        return self._start_stream(
            ForeachStreamSink(fn), trigger, output_mode, query_name, checkpoint
        )

    def parquet(self, path: str, *, compression: str = "zstd", **opts: Any) -> WriteManifest:
        """Write as Parquet (see `__call__` for `partition_by`/`distributed`).

        Args:
            path: Output path/URI (file or directory) to write to.
            compression: Parquet compression codec (default ``"zstd"``).
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

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

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

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

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

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

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

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

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

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

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.write.avro("data.avro")  # doctest: +SKIP
        """
        return self(path, "avro", **opts)

    def lance(self, path: str, **opts: Any) -> WriteManifest:
        """Write a Lance dataset (needs ``batcher-engine[lance]``).

        Args:
            path: Output directory path for the Lance dataset.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.write.lance("data.lance")  # doctest: +SKIP
        """
        return self(path, "lance", **opts)

    def msgpack(self, path: str, **opts: Any) -> WriteManifest:
        """Write as MessagePack (needs ``batcher-engine[msgpack]``).

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.write.msgpack("data.msgpack")  # doctest: +SKIP
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

        Args:
            target: Path/URI of the table to merge into.
            on: Join key column name(s) matching source rows to target rows.
            when_matched: Action for matched rows (``"update"`` or ``"delete"``).
            when_not_matched: Action for new rows (``"insert"`` or ``"ignore"``).
            format: Sink format override; inferred from `target` when omitted.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the merged output.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> updates = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> updates.write.merge(  # doctest: +SKIP
                ...     "warehouse/orders",
                ...     on="id",
                ...     when_matched="update",
                ...     when_not_matched="insert",
                ... )
        """
        from batcher.api.merge import execute_merge

        return execute_merge(
            self,
            target,
            on=on,
            when_matched=when_matched,
            when_not_matched=when_not_matched,
            format=format,
            opts=opts,
        )

    def merge_into(
        self,
        target: str,
        *,
        on: str | list[str],
        prune: bool = True,
        format: str | None = None,
        **opts: Any,
    ) -> MergeBuilder:
        """Open a full ``MERGE INTO`` against `target`, keyed on `on` — the general form.

        `merge` is the two-clause shorthand (update the matched, insert the rest). This is
        the whole statement: any number of ordered ``WHEN`` clauses, each with its own
        condition, each writing whichever columns it likes — including ``WHEN NOT MATCHED
        BY SOURCE``, which acts on the target rows the change set never mentioned (how a
        snapshot load expires departed rows, and how SCD-2 closes a version). Clauses are
        tried in the order added; the first whose condition holds wins.

        The merge rewrites only the data files whose key statistics prove they could hold
        one of the source's keys — so an upsert costs the change set, not the table. (A
        ``when_not_matched_by_source`` clause is *about* the untouched rows, so it forces a
        full rewrite; see `when_not_matched_by_source`.)

        Args:
            target: Path/URI of the table to merge into.
            on: Key column(s) matching a source row to a target row.
            prune: Skip target files the source's keys provably cannot reach. Correctness
                does not depend on it — turning it off only rewrites everything.
            format: Sink format override; inferred from `target` when omitted.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `MergeBuilder`; add clauses, then call `execute`.

        Examples:
            .. doctest::

                >>> import tempfile, os
                >>> import batcher as bt
                >>> from batcher import lit, source_col
                >>> path = os.path.join(tempfile.mkdtemp(), "orders.parquet")
                >>> _ = bt.from_pydict({"id": [1, 2], "amount": [10, 20]}).write.parquet(path)
                >>> changes = bt.from_pydict({"id": [2, 3], "amount": [99, 30]})
                >>> _ = (
                ...     changes.write.merge_into(path, on="id")
                ...     .when_matched(source_col("amount") > lit(50))
                ...     .delete()
                ...     .when_matched()
                ...     .update_all()
                ...     .when_not_matched()
                ...     .insert_all()
                ...     .execute()
                ... )
                >>> sorted(bt.read.parquet(path).collect().to_pydict()["id"])
                [1, 3]
        """
        from batcher.api.merge import MergeBuilder

        keys = [on] if isinstance(on, str) else list(on)
        return MergeBuilder(self._ds, target, keys, prune=prune, format=format, opts=opts)

    # --- Lakehouse / catalog ----------------------------------------------
    def delta(
        self,
        uri: str,
        *,
        mode: str = "append",
        merge_on: str | list[str] | None = None,
        auto_compact: bool = False,
        merge_schema: bool = False,
        **opts: Any,
    ) -> WriteManifest:
        """Write to a Delta Lake table (one transactional commit).

        With `merge_on`, performs a ``MERGE INTO`` upsert keyed on those columns —
        matched rows are updated and new rows inserted (Spark/Delta ``MERGE``). The
        keys build the match predicate; pass `merge_predicate=` instead for a custom
        one. Otherwise `mode` is ``"append"`` (default) or ``"overwrite"``.

        ``auto_compact=True`` bin-packs the table after the commit if it has accumulated
        enough small files. An incremental writer leaves one small file per commit and the
        next write cannot fix that — it only adds another — so a table nobody compacts
        eventually costs more to *plan* than to read. The check counts the table's standing
        small files (from the log, not by opening anything), and the compaction is the same
        transaction `bt.compact` runs, so it never deletes a file an older version still
        references. It runs *after* the commit, so a failed compaction cannot fail the write.

        Args:
            uri: Path/URI of the Delta table root.
            mode: ``"append"`` (default) or ``"overwrite"`` when not merging.
            merge_on: Key column(s) to upsert on; triggers a ``MERGE INTO``.
            auto_compact: Bin-pack the table after the commit if small files have piled up.
            merge_schema: Evolve the table to accept columns this write has and the table
                does not. Off by default — an unexpected column is refused rather than
                silently written into the files where the table cannot see it.
            opts: Additional write options (e.g. ``merge_predicate=``) forwarded to the sink.

        Returns:
            A `WriteManifest` describing the committed Delta files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.delta("warehouse/orders", merge_on="id")  # doctest: +SKIP
        """
        if merge_on is not None:
            from batcher.api.merge import merge_predicate_for

            opts["merge_predicate"] = merge_predicate_for(merge_on)
        opts["merge_schema"] = merge_schema
        return self(uri, "delta", mode=mode, auto_compact=auto_compact, **opts)

    def iceberg(self, identifier: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Iceberg table (``mode="append"|"overwrite"``).

        Args:
            identifier: Table identifier within the catalog (``db.table``).
            mode: ``"append"`` (default) or ``"overwrite"``.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the committed Iceberg files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.iceberg("db.orders", mode="append")  # doctest: +SKIP
        """
        return self(identifier, "iceberg", mode=mode, **opts)

    def hudi(self, table_uri: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Apache Hudi table (``mode="append"|"overwrite"``).

        Args:
            table_uri: Path/URI of the Hudi table root.
            mode: ``"append"`` (default) or ``"overwrite"``.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the committed Hudi files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.hudi("s3://lake/orders", mode="append")  # doctest: +SKIP
        """
        return self(table_uri, "hudi", mode=mode, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a database table via ADBC/FlightSQL.

        Args:
            table: Destination table name.
            opts: Connection (``uri=``) and driver options passed as keywords.

        Returns:
            A `WriteManifest` describing the written rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.sql(  # doctest: +SKIP
                ...     "orders", uri="postgresql://localhost/warehouse"
                ... )
        """
        return self(table, "adbc", **opts)

    def snowflake(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a Snowflake table.

        Args:
            table: Destination Snowflake table name.
            opts: Connection credentials (account, warehouse, database, …) as keywords.

        Returns:
            A `WriteManifest` describing the written rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.snowflake(  # doctest: +SKIP
                ...     "ORDERS", account="acct", warehouse="WH", database="DB"
                ... )
        """
        return self(table, "snowflake", **opts)

    def mongo(self, collection: str, **opts: Any) -> WriteManifest:
        """Write to a MongoDB collection.

        Args:
            collection: Destination collection name.
            opts: Connection (``uri=``), ``database=``, and write options as keywords.

        Returns:
            A `WriteManifest` describing the written documents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.mongo(  # doctest: +SKIP
                ...     "orders", uri="mongodb://localhost:27017", database="shop"
                ... )
        """
        return self(collection, "mongo", **opts)
