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

from batcher._internal.errors import PlanError
from batcher._internal.logging import note_suppressed
from batcher.api.orchestration import phases
from batcher.api.orchestration.sizing import (
    DEFAULT_PARTITIONS,
    declared_row_count,
    partitions_from_physical,
)
from batcher.io.source import InMemorySource, read_source

if TYPE_CHECKING:
    from batcher.core import ExecutionContext
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan
    from batcher.plan.profile.usage import UsageStopwatch

__all__ = [
    "ResolvedSources",
    "execute_distributed",
    "measured_spill_collect",
    "resolve_sources",
    "spill_to_disk",
]


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

    staged = _stage_if_optimization_requires_it(
        plan, logical_opt, sources, ctx, materialize=materialize
    )
    if staged is not None:
        return staged

    # Learned scheduling: size worker fan-out from the measured data volume (when the user
    # gave none) and warm-start the shuffle credit window from what this signature converged
    # on last time. Both are pure scheduling levers, so a cold hub grants the default.
    phases.begin("distributed_grant")
    mark = time.perf_counter()
    workers, envelope = distributed_grant(rm, opt, plan, sources, ctx)
    phase("distributed_grant", time.perf_counter() - mark)

    # The long one. Before this the line showed whatever phase had ended last -- on a
    # seven-second distributed run, `admission`, which had taken 0.3 ms.
    phases.begin("execute_distributed")
    mark = time.perf_counter()
    prof = ctx.profile
    worker_metrics: list = []
    try:
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
    except PlanError:
        # Staging was skipped because the aligned executor claimed the plan, and it declined
        # once it could see what planning cannot: the size of a broadcast result. The staged
        # loop is the route this plan would have taken without it.
        if not (materialize and _aligned_skipped_staging(plan, logical_opt, sources, ctx.hub)):
            raise
        result = _run_staged(plan, sources, ctx)
    phase("execute_distributed", time.perf_counter() - mark)

    phases.begin("collect_source_metadata")
    mark = time.perf_counter()
    if prof is not None:
        prof.worker_metrics = worker_metrics
    collect_source_metadata(ctx.hub, sources, plan)
    phase("collect_source_metadata", time.perf_counter() - mark)

    record_distributed(
        ctx.hub,
        plan,
        logical_opt,
        decisions,
        envelope.credits,
        (time.perf_counter() - started) * 1000.0,
    )
    _record_distributed_cardinality(ctx.hub, plan, sources, result)
    return result


def _aligned_skipped_staging(
    plan: LogicalPlan, logical_opt: LogicalPlan, sources: list[Source], hub
) -> bool:
    """Whether staging was bypassed for this plan only because it looked aligned.

    The same question the gate asked (`aligned_claims`, of `plan`), not a fresh one of
    `logical_opt`: the two plans can differ, and a gate that said "aligned" followed by a
    probe here that said "not aligned" re-raised the decline instead of staging.
    """
    from batcher.api.adaptive.gating import aligned_claims
    from batcher.api.adaptive.staging import staged_depth_exhausted
    from batcher.dist import requires_staging

    return (
        not staged_depth_exhausted()
        and (requires_staging(plan) or requires_staging(logical_opt))
        and aligned_claims(plan, sources, hub)
    )


def _stage_if_optimization_requires_it(
    plan: LogicalPlan,
    logical_opt: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    *,
    materialize: bool,
) -> pa.Table | None:
    """Run the plan staged when *optimization* is what made staging the only route.

    `resolve_adaptive` asks `requires_staging` of the plan the caller wrote. This function is
    handed the plan **Kyber produced**, and the two are not the same question: a rewrite can
    introduce a breaker beneath a join that the raw plan did not have. Eager aggregation is
    the one that does it — `Aggregate(Join(lineitem, orders))` becomes
    `Aggregate(Join(orders, Aggregate(lineitem)))`, pre-reducing the fact table on the join
    key — which is a good rewrite and a shape `dist._dispatch` has no one-shot path for.

    It fires on *measured* statistics, so the raw plan and the cold-hub optimized plan both
    say "no staging needed" and the warm one says otherwise. The observable symptom was that
    a `join -> group_by -> agg` over parquet — TPC-H q3, q5, q9, q10 and most of any star
    schema — ran fine distributed the first time and raised `PlanError` on **every run after
    it**, with the query, the data and the arguments unchanged. Cross-run learning turning a
    working query into a failing one is the worst shape a learning loop can have, and nothing
    in the suites could see it: a differential test runs each query once.

    Asking the question again is the whole fix, and here is where it can first be asked —
    this module is the one place the optimized plan and the distributed route meet.

    Args:
        plan: The pre-optimization plan, which is what the staged loop re-optimizes.
        logical_opt: The optimized plan the one-shot dispatcher would otherwise be handed.
        sources: The plan's bound sources.
        ctx: The execution context carrying the hub, fan-out and transport.
        materialize: False when this call is producing a partitioned intermediate.

    Returns:
        The staged result, or `None` when the one-shot route is fine (the common case).
    """
    from batcher.api.adaptive.staging import staged_depth_exhausted

    # Bounded, not forbidden, from inside the loop. Every stage executes through
    # `run_relational`, which comes back through this function, and the loop's **final**
    # stage is the whole residual plan — so an *unbounded* re-entry let a plan the loop could
    # not cut re-enter the loop on itself: `RecursionError` rather than the `PlanError` the
    # caller should get, on
    # `test_diff_exists_mixed_correlation::test_mixed_exists_matches_distributed`.
    # `materialize` alone does not cover it: that final stage materializes, like any ordinary
    # query.
    #
    # Refusing the *first* re-entry also refused the one case that makes progress. The loop
    # picks its cuts from the plan the caller wrote, and Kyber then re-optimizes each stage —
    # so eager aggregation can put a breaker beneath a join in a stage the loop had already
    # decided needed no cut. That breaker is new and strictly below the root, so staging it
    # materializes something and the residual shrinks. `_MAX_STAGED_DEPTH` caps how often
    # that may happen, which is what makes the recorded `RecursionError` impossible rather
    # than unlikely: a rewrite that does not make progress spends the budget and then yields
    # the same `PlanError` as before.
    if not materialize or staged_depth_exhausted():
        return None
    from batcher.dist import requires_staging
    from batcher.dist.executors.aligned import aligned_route

    # The aligned executor runs a breaker beneath a join whole, per key range, so a plan it
    # takes needs no stage boundary -- and staging it would put an exchange back between
    # tables whose layout already co-partitions them. `resolve_adaptive` asked this of the
    # plan before eager aggregation; it has to be asked again of the plan after it.
    if not requires_staging(logical_opt) or aligned_route(logical_opt, sources):
        return None
    return _run_staged(plan, sources, ctx)


def _run_staged(plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext) -> pa.Table:
    """Run `plan` through the staged loop because the one-shot route cannot run it."""
    from batcher.api.adaptive import execute_adaptive

    # Deliberately NOT recorded through `record_adaptive_route`. That feeds the bandit which
    # chooses between the staged and one-shot routes on cost, and here there is no choice to
    # learn from: the one-shot route cannot run this plan at all. Timing it as if it had won
    # a comparison would teach the chooser about an arm the other queries do have.
    return execute_adaptive(
        plan,
        sources,
        ctx.hub,
        distributed=True,
        num_workers=ctx.num_workers,
        transport=ctx.transport,
        # Staging is the only route for this plan, which the loop cannot work out for
        # itself: it is handed `plan`, and it is `logical_opt` that has no one-shot path.
        # Without saying so, the loop would fall back to `_worth_staging`, whose answer
        # depends on what the hub has learned — so the query would run or raise according
        # to how many times it had been run before.
        force_structural=True,
    ).table


def _record_distributed_cardinality(hub, plan: LogicalPlan, sources: list[Source], result) -> None:
    """Close the *cardinality* loop for a distributed run, as the single-node path does.

    `record_distributed` above learns the scheduling knobs — worker fan-out, credit window.
    The plan-level learned state is a different loop and was single-node only: this path
    never reached `record_cardinality_outcome`, so a distributed query measured its own
    output on every run and learned nothing from it, while the identical query on one node
    learned. That is backwards. A distributed workload is the long-running, repeatedly-issued
    one, which is exactly where cross-run learning is worth the most — and it was the one
    configuration with the loop switched off.

    The row count is taken only where it is already known. A materialized result is a table
    and carries it; a partitioned result left on the fleet reports it from its handles
    without fetching a batch. Anything else declines rather than forcing a materialization —
    learning must not change what the run costs.

    Best-effort, like every other write on this path.

    Args:
        hub: The metadata hub; a `None` hub is a no-op.
        plan: The pre-optimization plan — the identity the learner keys on.
        sources: The plan's bound sources, for the selectivity denominator.
        result: The run's result, a table or a partitioned source.
    """
    # The whole body is inside the guard, counting included. Asking a partitioned result for
    # its row count reaches the fleet's handles, and a worker lost between the answer landing
    # and this call would otherwise turn a *completed* query into a failed one — the run has
    # already produced its rows, and nothing about learning from them is worth that.
    try:
        rows = getattr(result, "num_rows", None)
        if rows is None:
            counter = getattr(result, "row_count", None)
            rows = counter() if callable(counter) else None
        if rows is None:
            return
        # Imported here, not at module scope: `run` imports this module, so the dependency
        # only runs one way at import time.
        from batcher.api.orchestration.run import record_cardinality_outcome

        record_cardinality_outcome(hub, plan, sources, int(rows))
    except Exception as exc:  # learning must never break a completed run
        note_suppressed("api", "record distributed cardinality", exc)


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
    from batcher.plan.profile.usage import UsageStopwatch

    partitions = (
        rm.recommend_spill_partitions(opt) or partitions_from_physical(opt) or DEFAULT_PARTITIONS
    )
    partitions = max(partitions, rm.partitions_for_bounds(opt, verdict.suggested_bounds))
    # The out-of-core path runs thousands of *unmetered* engine dispatches rather than one
    # metered call, so the engine's own whole-execution reading never happens on it — the
    # queries most worth observing reported no CPU, memory or disk cost at all. Two syscalls
    # around the whole phase measure the same counters, and the reading is shaped exactly
    # like the engine's, so the profile consumes either without knowing which it got.
    watch = UsageStopwatch()
    # The query is now committed to disk, so hand the engine's retained arena back before it
    # starts writing. mimalloc keeps freed pages by design -- 408 MiB of a 1,397 MiB resident
    # set, measured on three group-bys whose results had already been dropped -- and holding
    # that through a spill is a third of a gigabyte closer to an OOM kill on a node with no
    # swap, which is the default on Kubernetes. Here rather than in the spill *gate*: the gate
    # has three independent routes and only one of them reads live pressure, so trimming there
    # covered a third of the spills. A shape with no out-of-core path reaches this line and then
    # falls back to memory, having paid one trim; `memory.reclaim`'s backoff is what stops that
    # from being paid twice.
    rm.going_out_of_core()
    # Force the learned spill codec (large IO-bound state compresses; small state does not).
    # IPC self-describes its codec, so the un-spilled result is byte-identical either way.
    with spill_compression_scope(rm, opt):
        spilled = measured_spill_collect(logical_opt, sources, partitions, ctx, watch)
    # Record the spill only once it has actually happened. `spill_collect` returns `None` for
    # a shape with no out-of-core path (a string-keyed sort, a filter/project with no state),
    # and the caller then falls through to the in-memory path — so recording *before* the
    # call told every such run it had "executed out-of-core under bounded memory". That is
    # the one claim a reader reaches for while diagnosing an OOM, and it was exactly backwards
    # on the runs that stayed resident.
    if spilled is not None and ctx.profile is not None:
        from batcher.api.terminal.profile import record_spill

        record_spill(ctx.profile, partitions, rm.spill_reason(opt))
    return spilled


def measured_spill_collect(
    logical_opt: LogicalPlan,
    sources: list[Source],
    partitions: int,
    ctx: ExecutionContext,
    watch: UsageStopwatch,
) -> pa.Table | None:
    """`spill_collect`, with what it cost and what it wrote to disk recorded into the profile.

    Both readings are needed because this path runs no metered engine call: `watch` is the
    whole-phase resource reading, and the spill meter is the only account of the buckets the
    Python executors wrote, without which the profile reported `spilled: False` for a query
    that went out of core. Nothing is recorded when the shape has no out-of-core path.

    Args:
        logical_opt: The optimized logical plan.
        sources: The plan's bound sources.
        partitions: The out-of-core bucket count.
        ctx: The execution context, whose profile (if any) receives the readings.
        watch: A stopwatch started where the phase's cost should begin.

    Returns:
        The out-of-core result, or `None` when the plan has no out-of-core path.
    """
    from batcher.dist.spill import spill_collect
    from batcher.plan.profile.spill import spill_meter

    with spill_meter() as meter:
        spilled = spill_collect(logical_opt, sources, partitions)
    if spilled is not None and ctx.profile is not None:
        ctx.profile.record_usage(watch.finish())
        ctx.profile.record_out_of_core_spill(meter)
    return spilled


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


def read_scanned(
    sources: list[Source], opt: PhysicalPlan, ids: set[int] | frozenset[int]
) -> dict[int, tuple[list[pa.RecordBatch], float]]:
    """Read the sources in `ids` with the plan's pushdowns, concurrently: `{id: (batches, ms)}`.

    One source after another used to be the rule, and on a join over several tables it put
    every read in series although each spends its time in native code with the GIL released:
    the reads of a TPC-H query's `orders`, `customer` and `nation` then cost their sum instead
    of the longest. Each read keeps its own concurrency across its files, so this adds one
    thread per source, not per file. `ms` is the read's own wall time — measured while the
    others ran, which is the throughput the query actually got.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan, carrying the pushed projections and predicates.
        ids: The sources to read.

    Returns:
        Each read source's batches and its wall time in milliseconds.
    """

    groups = _shared_reads(sources, opt, sorted(i for i in ids if i < len(sources)))

    def one(group: list[int]) -> tuple[list[pa.RecordBatch], float]:
        started = time.perf_counter()
        first = group[0]
        if len(group) == 1:
            projection = opt.source_projections.get(first)
            predicate = opt.source_predicates.get(first)
        else:
            projection = _union_projection([opt.source_projections.get(i) for i in group])
            predicates = [opt.source_predicates.get(i) for i in group]
            predicate = predicates[0] if all(p == predicates[0] for p in predicates) else None
        batches = read_source(
            sources[first],
            projection,
            predicate,
            opt.source_limits.get(first),
            opt.source_orderings.get(first),
        )
        return batches, (time.perf_counter() - started) * 1000.0

    # The pool buys overlap between reads that each spend their time in native code; a
    # resident relation's "read" is a projection over batches already in memory, and a pool
    # spun up and torn down around those cost more than the reads (~3% of a warm TPC-H sf1
    # q8, sampled, in thread start/join alone). So it is used only when two or more groups
    # actually read something.
    if sum(not isinstance(sources[g[0]], InMemorySource) for g in groups) <= 1:
        results = [one(g) for g in groups]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            results = list(pool.map(one, groups))
    out: dict[int, tuple[list[pa.RecordBatch], float]] = {}
    for group, (batches, ms) in zip(groups, results, strict=True):
        for i in group:
            wanted = opt.source_projections.get(i)
            out[i] = (batches if len(group) == 1 else _narrowed(batches, wanted), ms)
    return out


def _shared_reads(sources: list[Source], opt: PhysicalPlan, ids: list[int]) -> list[list[int]]:
    """`ids` grouped so each group is read once: the bindings of one source object together.

    A source bound twice — a self-join, or a correlated subquery decorrelated into a second
    scan of the table it correlates with (TPC-H q17, q18, q20, q21) — was decoded once per
    binding. One read of the union of their columns serves every binding, because what each
    binding pushed is superset-safe: the engine keeps its `Filter`, so a binding handed rows its
    predicate would have pruned still computes the same answer. Only bindings with no pushed row
    cap and no pushed ordering share, since those change *which* rows a read returns.
    """
    groups: dict[int, list[int]] = {}
    alone: list[list[int]] = []
    for i in ids:
        if opt.source_limits.get(i) is not None or opt.source_orderings.get(i):
            alone.append([i])
        else:
            groups.setdefault(id(sources[i]), []).append(i)
    return list(groups.values()) + alone


def _union_projection(projections: list[list[str] | None]) -> list[str] | None:
    """The columns every projection needs, first-seen order; None when any reads every column."""
    columns: dict[str, None] = {}
    for projection in projections:
        if projection is None:
            return None
        columns.update(dict.fromkeys(projection))
    return list(columns)


def _narrowed(batches: list[pa.RecordBatch], columns: list[str] | None) -> list[pa.RecordBatch]:
    """`batches` reduced to `columns` in that order (zero-copy), or unchanged for None."""
    if columns is None:
        return batches
    return [b.select(columns) for b in batches]


def resolve_sources(sources: list[Source], opt: PhysicalPlan, ctx: ExecutionContext):
    """Read every source to Arrow, timing each read and recording its throughput.

    Reads happen here, not earlier: projection and predicate pushdown tell each source what
    to read, and both are only known once the plan is optimized. Core measures the I/O the
    hardware actually delivered so a later read of the same source can predict its cost.

    **A source the plan does not scan is not read at all.** The bound source list is the
    *query's*, while `opt` is often a sub-plan of it: the adaptive staging loop executes one
    breaker's subtree at a time against the full list, so a stage scanning two tables used to
    read all six. That is not merely wasted I/O — the pushdown analysis records a projection
    only for scans the plan contains, and an absent entry means "read every column", so the
    sources a stage does not use are exactly the ones it reads *widest*. On TPC-H q9 at sf10
    the first stage resolved 8.9 GiB where its own plan needed 2.9 GiB, and the query was
    OOM-killed. Skipped sources keep their index and resolve to no batches, which is what a
    plan that never scans them asks the engine for.

    Args:
        sources: The plan's bound sources.
        opt: The optimized physical plan, carrying the pushed projections and predicates.
        ctx: The execution context, whose hub receives the throughput measurements.

    Returns:
        The resolved batches and the per-source complete-scan flags.
    """
    from batcher.api.source_stats import _source_identity
    from batcher.metadata.io_stats import record_source_io, scanned_byte_count

    scanned_ids = opt.scanned_source_ids()
    reads = read_scanned(sources, opt, scanned_ids)
    batches_per_source = []
    complete: list[bool] = []
    for i, src in enumerate(sources):
        if i not in scanned_ids:
            # No rows, and `complete=False`: nothing was read, so nothing was proven about
            # this source's size. Recording a zero-row "complete scan" would teach the
            # cardinality model that the table is empty.
            batches_per_source.append([])
            complete.append(False)
            continue
        predicate = opt.source_predicates.get(i)
        limit = opt.source_limits.get(i)
        batches, elapsed_ms = reads[i]

        batches_per_source.append(batches)
        declared = declared_row_count(src)
        scanned = sum(b.num_rows for b in batches)
        # `limit` joins `predicate` here, for the same belt-and-braces reason `predicate`
        # is already here: `scanned == declared` would catch a capped read on its own,
        # since a source that stopped early returns fewer rows than it declares. But the
        # cost of the two mistakes is lopsided — a missing row count merely leaves Kyber
        # estimating, while a wrong one is recorded as *exact* and mis-plans the relation
        # on every later run — so a scan that was offered a subset is not asked to prove
        # it read everything.
        complete.append(
            predicate is None and limit is None and declared is not None and scanned == declared
        )

        identity = _source_identity(src)
        record_source_io(
            ctx.hub,
            identity,
            scanned_byte_count(identity, opt.source_projections.get(i), scanned, batches),
            elapsed_ms,
        )
    return ResolvedSources(batches_per_source, complete)
