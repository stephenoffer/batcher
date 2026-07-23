"""The single-node streaming launcher: optimize once, then drive micro-batches.

Builds the Kyber-optimized per-batch runner and the `MicroBatchProcessor` the engine
folds each micro-batch through, then starts `core.StreamingQueryEngine`. The distributed
launcher (`_distributed`) reuses that same engine with a different micro-batch runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.api.streaming._query import (
    _ACTIVE,
    _LOCK,
    StreamingQuery,
    _next_name,
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
    query = StreamingQuery(query_name, engine)
    with _LOCK:
        _ACTIVE[query_name] = query
    engine.start()
    return query


def _is_stateless(plan: LogicalPlan) -> bool:
    from batcher.plan.logical import Aggregate, Distinct, is_streamable

    return is_streamable(plan) and not isinstance(plan, (Aggregate, Distinct))
