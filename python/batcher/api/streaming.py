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

__all__ = ["StreamingQuery", "active_streams", "start_streaming_query"]


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


def _build_run_batch(plan: LogicalPlan, sources: list[Source]):
    """Build the Kyber-optimized per-micro-batch runner for a stateless pipeline.

    Mirrors `api/terminal/stream.py::_iter_streaming`: a `map_batches` pipeline runs
    its opaque UDF per batch; a relational pipeline is optimized once so the source
    projection/predicate is pushed down, and each batch feeds the metadata learner.
    """
    from batcher import core, kyber
    from batcher.io.source import InMemorySource

    if core.has_map_batches(plan):

        def run_batch(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
            return core.execute_with_udfs(plan, [InMemorySource([batch])])

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
