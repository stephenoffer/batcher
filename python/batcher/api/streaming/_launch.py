"""The single-node streaming launcher: optimize once, then drive micro-batches.

Builds the Kyber-optimized per-batch runner and the `MicroBatchProcessor` the engine
folds each micro-batch through, then starts `core.StreamingQueryEngine`. The distributed
launcher (`_distributed`) reuses that same engine with a different micro-batch runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.api.streaming._query import (
    StreamingQuery,
    _deregister,
    _next_name,
    _register,
    _warn_if_checkpoint_not_durable,
)
from batcher.plan.streaming import OutputMode, Trigger

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["start_streaming_query"]


def _build_run_batch(
    plan: LogicalPlan, sources: list[Source]
) -> tuple[object, list[str] | None, dict | None]:
    """Build the Kyber-optimized per-micro-batch runner for a stateless pipeline.

    Mirrors `api/terminal/stream/dispatch.py::_iter_streaming`: a `map_batches` pipeline
    runs its opaque UDF per batch; a relational pipeline is optimized once so the source
    projection/predicate is pushed down, and each batch feeds the metadata learner.

    Returns the runner **and** the source projection/predicate, because pushdown is only
    real if the caller reads the source through them. This used to return the runner
    alone, so `LocalRunner` read with `iter_batches(None)` and a `select("a")` over a
    Kafka topic decoded every column of every message forever — while the identical
    `iter_batches` pipeline pushed the projection down. The distributed launcher already
    threaded the projection, so single-node and distributed streaming disagreed too.

    Args:
        plan: The streaming pipeline's logical plan.
        sources: Its bound sources (single-source; validated by the caller).

    Returns:
        `(run_batch, projection, predicate)`. Projection and predicate are `None` for a
        `map_batches` pipeline, whose UDF is opaque to Kyber.
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

        return run_batch, None, None

    hub = core.default_hub()
    opt_plan = kyber.optimize(plan, sources=sources, hub=hub)

    def run_batch(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        return core.execute_local(opt_plan, [[batch]], feedback=hub)

    return run_batch, opt_plan.source_projections.get(0), opt_plan.source_predicates.get(0)


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
    shape (an output-mode/plan mismatch, or a multi-source plan no driver can produce).

    A **two-source** plan is a stream-stream join, whose driver reads both sides itself
    and yields finished output rows. It reaches the engine through `DriverRunner` instead
    of the one-source `LocalRunner` — the trigger, the progress record, the sink's
    exactly-once check and the listener events are all unchanged. Until this existed the
    only way to consume a stream-stream join was `iter_batches()`, which the cookbook
    called the sharpest edge in the streaming story.
    """
    from batcher import core
    from batcher._internal.errors import PlanError

    if len(sources) > 1 or _is_driver_shape(plan):
        return _start_driver_stream(plan, sources, sink, trigger, output_mode, name, checkpoint)
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

    # A top-level aggregate over a `map_batches` input is the one non-stateless shape that
    # still needs a per-batch runner: the UDF runs in Python and the fold consumes what it
    # returns. `iter_batches` has streamed this since S29 (`map_stream`); the sink path
    # answered it with a bare `NotImplementedError` from `MapBatches.to_ir()`, which is the
    # single most common shape an ML streaming job has -- inference, then a rollup.
    mapped_aggregate = _is_mapped_aggregate(plan)
    if mapped_aggregate and checkpoint is not None:
        raise PlanError(
            "checkpoint= is refused for an aggregate over map_batches: the running state is "
            "folded against whatever schema the UDF returns, which is not knowable before "
            "the UDF has run, so a restart would resume from an empty aggregate while the "
            "offset log said the rows were already counted. Aggregate without the UDF (or "
            "materialize the mapped output first) if you need resumption."
        )
    if mapped_aggregate:
        run_batch, projection, predicate = _build_run_batch(plan.input, sources)
    else:
        run_batch, projection, predicate = (
            _build_run_batch(plan, sources) if _is_stateless(plan) else (None, None, None)
        )
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
        projection=projection,
        predicate=predicate,
    )
    query = StreamingQuery(query_name, engine, plan, sources)
    _register(query_name, query)
    try:
        engine.start()
    except BaseException:
        # `start()` opens the sink and recovers from the checkpoint before the loop
        # thread launches; if either raises, the query never runs, so it must not linger
        # in the registry as a phantom active stream. The store never reaches the loop's
        # own teardown on this path either, so its connections are closed here.
        _deregister(query_name)
        if store is not None:
            import contextlib

            with contextlib.suppress(Exception):
                store.close()
        raise
    return query


def _start_driver_stream(
    plan: LogicalPlan,
    sources: list[Source],
    sink,
    trigger: Trigger | None,
    output_mode: str,
    name: str | None,
    checkpoint: str | None,
) -> StreamingQuery:
    """Drive a multi-source streaming plan into a sink — a join, or a union of streams.

    The driver is the same `_iter_batches` router `iter_batches()` uses, so the rows
    written are the rows that terminal would have yielded — one implementation, two
    consumers, rather than a second definition of what a stream-stream join means.

    `checkpoint=` is refused rather than accepted-and-ignored. The join's state is two
    buffers plus two watermarks, none of it offset-addressable, so there is nothing to
    resume from: a checkpoint here would restart the query from an empty join on every
    restart while looking exactly like exactly-once recovery.
    """
    from batcher import core
    from batcher._internal.errors import PlanError
    from batcher.api.terminal.stream import _iter_batches
    from batcher.api.terminal.stream.static_join import stream_static_sides
    from batcher.api.terminal.stream.union import union_streams_interleaved
    from batcher.plan.logical import (
        Limit,
        StreamingSessionWindow,
        Union,
        WatermarkDedup,
        WatermarkStreamJoin,
        is_partition_independent,
    )

    # Row-wise operators above the shape are peeled by the router and re-applied per batch,
    # so recognition has to look under them here too -- otherwise a `filter` on top of a
    # stream-stream join is a plan `iter_batches` streams and no sink accepts.
    root = plan
    while is_partition_independent(root):
        root = root.input
    joins = isinstance(root, WatermarkStreamJoin)
    unions = isinstance(root, Union) and union_streams_interleaved(root, sources)
    enriches = stream_static_sides(root, sources) is not None
    retains_rows = isinstance(root, (Limit, StreamingSessionWindow, WatermarkDedup))
    if not (joins or unions or enriches or retains_rows):
        raise PlanError(
            "streaming a sink from more than one source is supported for a stream-stream "
            "interval join (join_stream), a stream-static join, and a UNION ALL of streams; "
            f"this plan reads {len(sources)} sources some other way. Write each input "
            "separately, or materialize one side to a bounded source first."
        )
    output_mode = OutputMode.validate(output_mode)
    if output_mode != OutputMode.APPEND:
        raise PlanError(
            f"output_mode={output_mode!r} needs an aggregation; a stream-stream join, a "
            "stream-static join, a session window, a watermark dedup, a limit and a "
            "stream union each emit every row once, which is 'append'"
        )
    if checkpoint is not None:
        raise PlanError(
            "this streaming plan has no checkpointable position: a join's state is two "
            "buffered sides and two watermarks, a stream-static join's is a dimension "
            "snapshot, a session window's is the rows of every session still open, a "
            "watermark dedup's is the seen-key set, a limit's is how many rows have gone "
            "out, and a union's is one cursor per branch "
            "— none of them is a source offset the engine records. Drop checkpoint= (the "
            "sink's own idempotency still applies), or reduce to a plan the offset log can "
            "address."
        )

    query_name = name or _next_name()
    trigger = trigger or Trigger.processing_time(0)
    # Every shape on this path retains something between batches -- buffered sides, a
    # dimension snapshot, open sessions, a seen-key set, a row count. Continuous processing
    # has no micro-batch boundary to fold at, which is why Spark restricts it to stateless
    # pipelines and why the single-source launcher already refused it. This path never
    # checked, so a continuous trigger was accepted and then quietly run as micro-batches:
    # the answer was right and the latency the caller asked for was not what they got.
    if trigger.kind == "continuous":
        raise PlanError(
            "continuous trigger supports only stateless pipelines (filter / select / "
            "map_batches); this plan retains state between micro-batches, which needs a "
            "boundary to fold at. Use a processing-time trigger."
        )

    def make_runner(should_stop):
        from batcher.core.streaming_runner import DriverRunner

        for source in sources:
            attach = getattr(source, "set_stop_signal", None)
            if attach is not None:
                attach(should_stop)
        return DriverRunner(_iter_batches(plan, sources, plan.available_columns()), sink)

    engine = core.StreamingQueryEngine(
        name=query_name,
        source=sources[0],
        sink=sink,
        processor=None,  # the driver produces finished output rows
        trigger=trigger,
        output_mode=output_mode,
        checkpoint=None,
        runner_factory=make_runner,
    )
    query = StreamingQuery(query_name, engine, plan, sources)
    _register(query_name, query)
    try:
        engine.start()
    except BaseException:
        _deregister(query_name)
        raise
    return query


def _is_stateless(plan: LogicalPlan) -> bool:
    from batcher.plan.logical import Aggregate, Distinct, is_streamable

    return is_streamable(plan) and not isinstance(plan, (Aggregate, Distinct))


def _is_driver_shape(plan: LogicalPlan) -> bool:
    """Whether the driver produces this plan's rows, under any stack of row-wise operators.

    These shapes go down the driver path for the same reason a join does: their state is
    retained *rows* or a cursor rather than a fold, and the operators that turn them into
    output are relational ones living above `core`, which `core` may not import. One
    definition of each, reached from both terminals. A `Limit` is here because a limited
    stream *ends*, and the driver runner already knows what to do when its iterator stops.

    The peeling matters as much as the list. `iter_batches` strips row-wise operators off a
    breaker and re-applies them per batch, so `join_stream(...).filter(...)` -- the
    cookbook's own recipe for "impressions with no click" -- streams there. This launcher
    tested the top node only, so a single `filter` on top made every one of these plans
    unrecognizable and unwritable. The same `is_partition_independent` predicate the router
    peels by is used here, so the two cannot drift into disagreeing about which plans exist.
    """
    from batcher.plan.logical import (
        Limit,
        StreamingSessionWindow,
        WatermarkDedup,
        is_partition_independent,
    )

    while is_partition_independent(plan):
        plan = plan.input
    return isinstance(plan, (Limit, StreamingSessionWindow, WatermarkDedup))


def _is_mapped_aggregate(plan: LogicalPlan) -> bool:
    """A top-level aggregate whose input is a breaker-free `map_batches` pipeline."""
    from batcher import core
    from batcher.plan.logical import Aggregate, is_streamable

    return (
        isinstance(plan, Aggregate)
        and plan.watermark is None
        and is_streamable(plan.input)
        and core.has_map_batches(plan.input)
    )
