"""The three ways the conductor can execute an admitted plan, plus the source read.

Once Kyber has optimized and Carbonite has ruled, exactly one of these runs the plan:
across a Ray cluster, out-of-core through the spilling executor, or in memory. They are
alternatives, not layers — each returns the finished table (or `None`, meaning "this route
does not apply, try the next") and none of them decides *which* route to take. That choice
stays in `run`, where the verdict and the budgets are.

The mergeable algebra makes the three interchangeable: a partition count, a worker count,
or a spill threshold changes where data lives and how long it takes, never the answer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher.api.orchestration.sizing import (
    DEFAULT_PARTITIONS,
    declared_row_count,
    partitions_from_physical,
)
from batcher.io.source import read_source

if TYPE_CHECKING:
    from batcher.core import ExecutionContext
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan

__all__ = ["ResolvedSources", "execute_distributed", "resolve_sources", "spill_to_disk"]


def execute_distributed(
    logical_opt: LogicalPlan,
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    rm: Any,
    opt: PhysicalPlan,
    decisions: list,
    *,
    materialize: bool,
    phase,
    started: float,
) -> pa.Table | Source:
    """Run the plan across the Ray cluster and close the learned-scheduling loops.

    The *optimized* logical plan is what gets distributed, not the raw one. The distributed
    executor reads join keys and pushed predicates straight off the `LogicalPlan`, and a
    comma join (``FROM a, b WHERE a.k = b.k``) is raw-lowered as a cartesian inner join on a
    constant key with the equality stranded in a `Filter` above it. Run raw, every row
    hashes to one bucket and the shuffle collapses onto a single reducer.

    Args:
        logical_opt: The optimized logical plan, carrying the derived join keys.
        plan: The pre-optimization plan, the identity the learner records against.
        sources: The plan's bound sources.
        ctx: The execution context (hub, transport, profile).
        rm: The Carbonite resource manager that admitted the plan.
        opt: The optimized physical plan.
        decisions: Kyber's per-join build-side choices.
        materialize: Whether to return a table rather than a streaming source.
        phase: The per-phase timing recorder.
        started: The run's start clock, for the join-strategy bandit's reward.

    Returns:
        The distributed result.
    """
    from batcher import dist
    from batcher.api.terminal._metadata import collect_source_metadata
    from batcher.api.tuning import distributed_grant, record_distributed

    # Learned scheduling: size worker fan-out from the measured data volume (when the user
    # gave none) and warm-start the shuffle credit window from what this signature converged
    # on last time. Both are pure scheduling levers, so a cold hub grants the default.
    mark = time.perf_counter()
    workers, envelope = distributed_grant(rm, opt, plan, sources, ctx)
    phase("distributed_grant", time.perf_counter() - mark)

    mark = time.perf_counter()
    prof = ctx.profile
    worker_metrics: list = []
    result = dist.execute_distributed(
        logical_opt,
        sources,
        workers,
        transport=ctx.transport,
        envelope=envelope,
        hub=ctx.hub,
        materialize=materialize,
        metrics_out=worker_metrics if prof is not None else None,
    )
    phase("execute_distributed", time.perf_counter() - mark)

    mark = time.perf_counter()
    if prof is not None:
        prof.worker_metrics = worker_metrics
    collect_source_metadata(ctx.hub, sources)
    phase("collect_source_metadata", time.perf_counter() - mark)

    record_distributed(
        ctx.hub,
        plan,
        logical_opt,
        decisions,
        envelope.credits,
        (time.perf_counter() - started) * 1000.0,
    )
    return result


def spill_to_disk(
    logical_opt: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    rm: Any,
    opt: PhysicalPlan,
    verdict,
) -> pa.Table | None:
    """Run the plan out-of-core, or return `None` when this shape has no spilling path.

    Partition count is sharded by data volume: the learned recommendation first (from
    measured per-family peak memory), then Kyber's per-breaker fan-out, then a constant, so
    a bigger group-by or join uses more, smaller buckets. It is then floored at whatever
    admission's counter-offer implies, because the volume-derived count knows nothing about
    this machine's memory.

    The *optimized* logical plan is what spills: the optimizer derives real join keys (a
    comma join would otherwise blow up cartesian out-of-core) and lowers
    ``COUNT(DISTINCT x)`` to ``COUNT(*)`` over a `DISTINCT`, so the spilling executor dedups
    hash-partitioned instead of spilling a giant value list.

    Args:
        logical_opt: The optimized logical plan.
        sources: The plan's bound sources.
        ctx: The execution context.
        rm: The Carbonite resource manager.
        opt: The optimized physical plan.
        verdict: Admission's verdict, whose `suggested_bounds` floors the partition count.

    Returns:
        The spilled result, or `None` when the plan has no out-of-core path.
    """
    from batcher.api.tuning import spill_compression_scope
    from batcher.dist.spill import spill_collect

    partitions = (
        rm.recommend_spill_partitions(opt) or partitions_from_physical(opt) or DEFAULT_PARTITIONS
    )
    partitions = max(partitions, rm.partitions_for_bounds(opt, verdict.suggested_bounds))
    if ctx.profile is not None:
        from batcher.api.terminal.profile import record_spill

        record_spill(ctx.profile, partitions, rm.spill_reason(opt))
    # Force the learned spill codec (large IO-bound state compresses; small state does not).
    # IPC self-describes its codec, so the un-spilled result is byte-identical either way.
    with spill_compression_scope(rm, opt):
        return spill_collect(logical_opt, sources, partitions)


class ResolvedSources:
    """Sources read into Arrow, with the per-source facts the learner needs afterwards.

    `complete` records whether each read saw its source *whole* — no predicate filtered it,
    and the rows read match what the source declares. Only a whole scan may teach the
    learner a source-level distinct count, because a partial scan's distinct count is an
    under-count rather than an estimate. An unknown row count counts as partial: a distinct
    count Batcher might be wrong about is one it declines to record.
    """

    __slots__ = ("batches", "complete")

    def __init__(self, batches: list, complete: list[bool]) -> None:
        self.batches = batches
        self.complete = complete


def resolve_sources(sources: list[Source], opt: PhysicalPlan, ctx: ExecutionContext):
    """Read every source to Arrow, timing each read and recording its throughput.

    Reads happen here, not earlier: projection and predicate pushdown tell each source what
    to read, and both are only known once the plan is optimized. Core measures the I/O the
    hardware actually delivered so a later read of the same source can predict its cost.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan, carrying the pushed projections and predicates.
        ctx: The execution context, whose hub receives the throughput measurements.

    Returns:
        The resolved batches and the per-source complete-scan flags.
    """
    from batcher.api.source_stats import _source_identity
    from batcher.metadata.io_stats import record_source_io, scanned_byte_count

    batches_per_source = []
    complete: list[bool] = []
    for i, src in enumerate(sources):
        read_started = time.perf_counter()
        predicate = opt.source_predicates.get(i)
        batches = read_source(src, opt.source_projections.get(i), predicate)
        elapsed_ms = (time.perf_counter() - read_started) * 1000.0

        batches_per_source.append(batches)
        declared = declared_row_count(src)
        scanned = sum(b.num_rows for b in batches)
        complete.append(predicate is None and declared is not None and scanned == declared)

        identity = _source_identity(src)
        record_source_io(
            ctx.hub,
            identity,
            scanned_byte_count(identity, opt.source_projections.get(i), scanned, batches),
            elapsed_ms,
        )
    return ResolvedSources(batches_per_source, complete)
