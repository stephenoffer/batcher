"""The public streaming-query handle + launcher (the conductor's streaming entry).

`api` is the only layer that may sequence Kyber → Core, so the streaming launcher
lives here: it optimizes the plan once (stateless case), builds the per-micro-batch
processor and the `StreamSink`, constructs the `core.StreamingQueryEngine`, starts
it, and hands back a `StreamingQuery`. The handle is a thin façade over the engine
(`stop` / `await_termination` / `status` / `recent_progress` / `is_active`) plus
registration in the process-wide active-query registry exposed as `bt.streams`.

Batch and streaming share the one `ds.write(...)` surface — this module is reached
only when that terminal runs in streaming mode (a `Trigger` was set or a source is
unbounded), never for an ordinary bounded write.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from batcher.plan.streaming import (
    OutputMode,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.core.streaming_query import StreamingQueryEngine
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "StreamingQuery",
    "active_streams",
    "start_distributed_stream",
    "start_distributed_stream_drain",
    "start_streaming_query",
]

# Triggers that drain the currently-available data and stop (Spark `AvailableNow` /
# `Once`). These are the shapes the distributed backfill path supports today.
_DRAIN_TRIGGER_KINDS = ("available_now", "once")


# Process-wide registry of running queries, surfaced as `bt.streams`.
_ACTIVE: dict[str, StreamingQuery] = {}
_LOCK = threading.Lock()
_COUNTER = 0


def _next_name() -> str:
    global _COUNTER
    with _LOCK:
        _COUNTER += 1
        return f"query-{_COUNTER}"


def _warn_if_checkpoint_not_durable(location: str) -> None:
    """Under ``resilience="spot"``, warn when the checkpoint location looks node-local.

    A streaming query's exactly-once recovery lives in its `checkpoint_location`. On a
    spot/preemptible cluster a reclaimed node takes a node-local checkpoint with it, so
    a restart cannot resume — defeating the durability the checkpoint exists for. A
    durable location (object storage, or a shared mount) survives the node. Only a
    warning, not an error: a bare path may legitimately be a shared filesystem, which
    we cannot tell apart from node-local storage."""
    import re
    import warnings

    from batcher.config import active_config

    if active_config().distributed.resilience != "spot":
        return
    has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", location)
    node_local = has_scheme is None or location.lower().startswith("file://")
    if node_local:
        warnings.warn(
            f"streaming checkpoint_location {location!r} looks node-local; on a "
            "spot/preemptible cluster a reclaimed node loses the checkpoint and its "
            "exactly-once recovery. Use a durable location (s3://, gs://, hdfs://, or a "
            "shared mount that survives node loss).",
            stacklevel=3,
        )


def active_streams() -> list[StreamingQuery]:
    """All currently-active streaming queries (the `bt.streams` accessor)."""
    with _LOCK:
        return [q for q in _ACTIVE.values() if q.is_active]


def await_any_termination(timeout: float | None = None) -> bool:
    """Block until any active streaming query stops (Spark ``awaitAnyTermination``).

    Waits for the first of the currently-running queries to terminate, re-raising its
    exception if it failed. With no active queries, returns immediately.

    Args:
        timeout: Maximum seconds to wait; ``None`` waits indefinitely.

    Returns:
        ``True`` if a query stopped (or none were active), ``False`` on timeout.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.await_any_termination(timeout=0.0)  # no active queries
            True
    """
    import time

    watching = active_streams()
    if not watching:
        return True
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        for q in watching:
            if not q.is_active:
                q.await_termination(0.0)  # re-raise if it failed; deregister
                return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        # Poll: the queries run on their own daemon threads, so a short sleep between
        # liveness checks keeps this cheap without a shared condition variable.
        time.sleep(0.05)


class StreamingQuery:
    """A handle to a running streaming query (Spark `StreamingQuery` parity).

    Returned by `ds.write(..., trigger=...)` (and `ds.write.console()/memory()/...`)
    when the write runs in streaming mode. Methods mirror Spark: `stop()`,
    `await_termination(timeout)`, `status`, `recent_progress`, `is_active`.
    """

    __slots__ = ("_engine", "_name")

    def __init__(self, name: str, engine: StreamingQueryEngine) -> None:
        self._name = name
        self._engine = engine

    def __repr__(self) -> str:
        """Show the query name and whether it is still running."""
        state = "active" if self._engine.is_active else "stopped"
        return f"StreamingQuery(name={self._name!r}, {state})"

    def __enter__(self) -> StreamingQuery:
        """Enter a ``with`` block; the query keeps running until the block exits."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the query on leaving the ``with`` block (even if the body raised)."""
        self.stop()

    @property
    def name(self) -> str:
        """The query's name (auto-generated if not supplied)."""
        return self._name

    @property
    def is_active(self) -> bool:
        """Whether the micro-batch loop is still running."""
        return self._engine.is_active

    def stop(self) -> None:
        """Halt the query at the next micro-batch boundary and wait for it to finish."""
        self._engine.stop()
        with _LOCK:
            _ACTIVE.pop(self._name, None)

    def await_termination(self, timeout: float | None = None) -> bool:
        """Block until the query stops (or `timeout` seconds elapse).

        Returns whether the query has stopped. Re-raises any exception the query
        loop failed with.
        """
        stopped = self._engine.await_termination(timeout)
        if stopped:
            with _LOCK:
                _ACTIVE.pop(self._name, None)
        return stopped

    @property
    def status(self) -> StreamingQueryStatus:
        """A point-in-time snapshot of the query's state."""
        return self._engine.status()

    def recent_progress(self) -> list[StreamingQueryProgress]:
        """Per-micro-batch metrics for the most recent batches (rolling window)."""
        return self._engine.recent_progress()

    @property
    def last_progress(self) -> StreamingQueryProgress | None:
        """The most recent micro-batch's metrics, or None if none completed yet."""
        progress = self._engine.recent_progress()
        return progress[-1] if progress else None

    def exception(self) -> BaseException | None:
        """The exception that terminated the query, or None if it is healthy.

        Spark `StreamingQuery.exception()` parity: read the failure without letting
        it propagate (unlike `await_termination`, which re-raises). Returns None while
        the query is running normally or after a clean stop.

        Returns:
            The terminating exception, or None.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> stream = bt.read.rate(rows_per_second=5, num_rows=5, pace=False)
                >>> q = stream.write.memory(  # doctest: +SKIP
                ...     "m", trigger=bt.Trigger.available_now()
                ... )
                >>> q.await_termination()  # doctest: +SKIP
                True
                >>> q.exception() is None  # doctest: +SKIP
                True
        """
        return self._engine.exception


def _build_run_batch(plan: LogicalPlan, sources: list[Source]):
    """Build the Kyber-optimized per-micro-batch runner for a stateless pipeline.

    Mirrors `api/terminal/stream.py::_iter_streaming`: a `map_batches` pipeline runs
    its opaque UDF per batch; a relational pipeline is optimized once so the source
    projection/predicate is pushed down, and each batch feeds the metadata learner.
    """
    from batcher import core, kyber
    from batcher.io.source import InMemorySource

    if core.has_map_batches(plan):
        # Build the (class) UDFs once for the whole stream, so a load-once inference
        # model loads a single time and is reused across every micro-batch — not rebuilt
        # per micro-batch (which would reload the model on every trigger). This is the
        # single-node resident-inference path; the distributed streaming pool is W1.
        resident = core.prebuild_factories(plan)

        def run_batch(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
            return core.execute_with_udfs(resident, [InMemorySource([batch])])

        return run_batch

    hub = core.default_hub()
    opt_plan = kyber.optimize(plan, sources=sources, hub=hub)

    def run_batch(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        return core.execute_local(opt_plan, [[batch]], feedback=hub)

    return run_batch


def start_streaming_query(
    plan: LogicalPlan,
    sources: list[Source],
    sink,
    *,
    trigger: Trigger | None = None,
    output_mode: str = OutputMode.APPEND,
    name: str | None = None,
    checkpoint: str | None = None,
) -> StreamingQuery:
    """Optimize, build the engine, start it, and return a `StreamingQuery`.

    `sink` is a constructed `StreamSink`. `trigger` defaults to as-soon-as-possible
    micro-batches. `checkpoint` is a directory enabling exactly-once recovery
    (offset/commit logs + state snapshots). Raises `PlanError` for an unsupported
    shape (multi-source, or an output-mode/plan mismatch).
    """
    from batcher import core
    from batcher._internal.errors import PlanError

    if len(sources) != 1:
        raise PlanError(
            "streaming a sink currently supports a single source (stream-stream join "
            "is not yet available); collect or write each input separately"
        )
    output_mode = OutputMode.validate(output_mode)
    trigger = trigger or Trigger.processing_time(0)

    # Continuous processing supports only stateless map/filter/project pipelines
    # (Spark's restriction): an aggregation needs a micro-batch boundary to fold.
    if trigger.kind == "continuous" and not _is_stateless(plan):
        raise PlanError(
            "continuous trigger supports only stateless pipelines (filter / select / "
            "map_batches); use a processing-time trigger for aggregations"
        )

    store = None
    if checkpoint is not None:
        from batcher.io.formats.streaming.checkpoint import CheckpointStore

        _warn_if_checkpoint_not_durable(checkpoint)
        store = CheckpointStore(checkpoint)

    run_batch = _build_run_batch(plan, sources) if _is_stateless(plan) else None
    processor = core.make_processor(plan, output_mode, run_batch)
    query_name = name or _next_name()
    engine = core.StreamingQueryEngine(
        name=query_name,
        source=sources[0],
        sink=sink,
        processor=processor,
        trigger=trigger,
        output_mode=output_mode,
        checkpoint=store,
    )
    query = StreamingQuery(query_name, engine)
    with _LOCK:
        _ACTIVE[query_name] = query
    engine.start()
    return query


def _is_stateless(plan: LogicalPlan) -> bool:
    from batcher.plan.logical import Aggregate, Distinct, is_streamable

    return is_streamable(plan) and not isinstance(plan, (Aggregate, Distinct))


def start_distributed_stream(
    plan: LogicalPlan,
    sources: list[Source],
    path: str,
    fmt: str,
    sink_kwargs: dict,
    *,
    trigger: Trigger,
    output_mode: str = OutputMode.APPEND,
    name: str | None = None,
    checkpoint: str | None = None,
    num_workers: int | None = None,
) -> StreamingQuery:
    """Run a continuous streaming query whose micro-batches execute across the cluster.

    The same `StreamingQueryEngine` as the single-node path — same trigger, same offset
    log, same recovery — with the *epoch* handed to a `DistributedRunner` instead of run on
    this thread. Each micro-batch fans across workers, which write data files without
    committing, and the driver publishes the epoch as **one** transaction. The result is a
    log with one transaction per micro-batch and rows that land exactly once, however many
    workers or retries produced them.

    The caller (`io_namespace.writer`) has already checked eligibility.
    """
    import json

    from batcher import core, kyber
    from batcher.plan.logical import Aggregate

    output_mode = OutputMode.validate(output_mode)
    store = None
    if checkpoint is not None:
        _warn_if_checkpoint_not_durable(checkpoint)
        from batcher.io.formats.streaming.checkpoint import CheckpointStore

        store = CheckpointStore(checkpoint)

    # What each worker runs. A stateless epoch runs the whole Kyber-optimized plan (with
    # its projection pushed into the read); an aggregate's workers run only its *input*
    # pipeline and hand back a partial, exactly as the single-node fold does — so the two
    # paths compute the same thing from the same IR.
    agg = plan if isinstance(plan, Aggregate) else None
    if agg is None:
        physical = kyber.optimize(plan, sources=sources, hub=core.default_hub())
        plan_ir, projection = physical.to_json(), physical.source_projections.get(0)
    else:
        plan_ir, projection = json.dumps(agg.input.to_ir()), None

    query_name = name or _next_name()
    workers = num_workers if num_workers is not None else _drain_workers(sources[0])
    drain = trigger.kind in _DRAIN_TRIGGER_KINDS

    def make_runner(should_stop):
        from batcher.dist.streaming.microbatch import DistributedRunner

        return DistributedRunner(
            plan_ir=plan_ir,
            projection=projection,
            source=sources[0],
            path=path,
            fmt=fmt,
            sink_kwargs=sink_kwargs,
            query_name=query_name,
            num_workers=workers,
            drain=drain,
            should_stop=should_stop,
            agg=agg,
        )

    engine = core.StreamingQueryEngine(
        name=query_name,
        source=sources[0],
        sink=None,  # every sink lives on a worker (data files) or in the runner (commit)
        processor=None,  # the runner runs the plan itself, in Rust, on the workers
        trigger=trigger,
        output_mode=output_mode,
        checkpoint=store,
        runner_factory=make_runner,
    )
    query = StreamingQuery(query_name, engine)
    with _LOCK:
        _ACTIVE[query_name] = query
    engine.start()
    return query


class _DrainEngine:
    """A completed distributed-drain query — a `StreamingQuery`-compatible handle.

    A distributed `available_now`/`once` write runs the whole backfill to completion
    (every worker drains its source partition in parallel) before the handle returns,
    so the query is already stopped. This exposes the same read surface the micro-batch
    `StreamingQueryEngine` does (`is_active`, `status`, `recent_progress`, …) reporting
    that terminal state, so `ds.write(..., distributed=True)` returns one uniform
    `StreamingQuery` regardless of whether it ran on one node or many.
    """

    __slots__ = ("_progress",)

    is_active = False
    exception = None

    def __init__(self, progress: StreamingQueryProgress) -> None:
        self._progress = [progress]

    def stop(self) -> None:
        """No-op — the drain already ran to completion before the handle was returned."""

    def await_termination(self, timeout: float | None = None) -> bool:  # noqa: ARG002
        """Already terminated; always returns ``True``."""
        return True

    def status(self) -> StreamingQueryStatus:
        return StreamingQueryStatus(
            is_active=False,
            is_data_available=False,
            is_trigger_active=False,
            message="Stopped",
            batches_processed=len(self._progress),
        )

    def recent_progress(self) -> list[StreamingQueryProgress]:
        return list(self._progress)


def start_distributed_stream_drain(
    plan: LogicalPlan,
    sources: list[Source],
    path: str,
    fmt: str,
    sink_kwargs: dict,
    columns: list[str],
    *,
    num_workers: int | None = None,
    name: str | None = None,
) -> StreamingQuery:
    """Backfill a stream across the cluster: each worker drains one source partition.

    The distributed image of an `available_now`/`once` streaming write. Rather than the
    driver pulling the whole stream through one thread, the splittable source is
    partitioned and each worker reads its partition, runs the (stateless) pipeline, and
    writes its own shard files — only manifests return to the driver. The mergeable/
    shared-nothing write means the output is identical to the single-node drain; this
    just fans the read+transform+write across nodes. Returns a completed `StreamingQuery`.

    The caller (`io_namespace.writer`) guarantees eligibility (stateless plan, splittable
    single source, drain trigger, no checkpoint, file/lakehouse sink).
    """
    from time import perf_counter, time

    from batcher.api.terminal import _write

    t0 = perf_counter()
    manifest = _write(
        plan,
        sources,
        columns,
        path,
        fmt,
        distributed=True,
        num_workers=num_workers if num_workers is not None else _drain_workers(sources[0]),
        sink_kwargs=sink_kwargs,
    )
    rows = manifest.total_rows
    progress = StreamingQueryProgress(
        batch_id=0,
        num_input_rows=rows,
        num_output_rows=rows,
        duration_ms=(perf_counter() - t0) * 1000.0,
        timestamp=time(),
    )
    query_name = name or _next_name()
    query = StreamingQuery(query_name, _DrainEngine(progress))  # type: ignore[arg-type]
    with _LOCK:
        _ACTIVE[query_name] = query
    return query


def _drain_workers(source) -> int:
    """A data-aware default worker count for a distributed drain (unset ``num_workers``).

    Fan out to one worker per source partition, capped at the driver's CPU count — so the
    drain scales with the backlog's real parallelism instead of a fixed handful. On a Ray
    cluster the resulting task demand is what the autoscaler reacts to; the per-task
    placement then clamps this to the live cluster (`dist.executors...clamp_workers`).
    """
    from batcher._internal.hardware import available_cpu_count

    try:
        n_splits = len(source.splits())
    except Exception:
        n_splits = 1
    return max(1, min(n_splits, available_cpu_count()))
