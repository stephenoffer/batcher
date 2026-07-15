"""Streaming with the micro-batch fanned across the cluster.

Two shapes, both on the same `core.StreamingQueryEngine` as the single-node path:

* `start_distributed_stream_drain` — an `available_now`/`once` backfill: every worker
  drains one source partition, once.
* `start_distributed_stream` — a continuous / processing-time stream: **each** micro-batch
  is an epoch that the workers stage (writing data files they do not commit) and the driver
  publishes as a single transaction, so the log records one transaction per micro-batch and
  a replayed epoch adds neither a row nor a commit.

The engine is unchanged; only the `MicroBatchRunner` differs (`dist.streaming.microbatch`).
`core` never imports `dist` — the conductor injects the runner here.
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
from batcher.plan.streaming import (
    OutputMode,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
)

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["start_distributed_stream", "start_distributed_stream_drain"]

# Triggers that drain the currently-available data and stop (Spark `AvailableNow` / `Once`).
_DRAIN_TRIGGER_KINDS = ("available_now", "once")


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

    The caller (`io_namespace.writer`) guarantees a splittable single source, a drain trigger,
    no checkpoint, a bounded source, and a file/lakehouse sink. It does **not** guarantee a
    stateless plan, and does not need to: a drain reads its bounded source once and hands the
    plan to `_write`, the ordinary distributed batch path, which already covers an aggregate
    through the mergeable algebra.

    That includes a *watermarked* plan, and this is the part worth stating explicitly, because
    it looks like the hole that was real on the micro-batch path. It is not one. A single pass
    over a bounded source sees its first batch with the watermark still unset, so no row can be
    behind it and the late-drop never fires — streaming and batch semantics coincide here, and
    both paths return the same rows (verified). Refusing a watermarked drain would cost a
    capability and buy no correctness.
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
