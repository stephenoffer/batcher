"""The contract loop: Kyber optimizes, Carbonite admits, Core executes, metadata flows back."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher._internal.logging import get_logger, log_kv, note_suppressed
from batcher.api._join_helpers import _empty_result_schema
from batcher.api.orchestration.sizing import (
    DEFAULT_PARTITIONS,
    distributed_hardware,
    partitions_from_physical,
    projected_input_bytes,
    proven_empty_table,
)
from batcher.api.orchestration.stages import execute_distributed, resolve_sources, spill_to_disk
from batcher.api.source_stats import collect_source_stats, column_bounds_needed
from batcher.config import active_config, config_context
from batcher.core.runtime import query_scope
from batcher.metadata.hardware_scope import planning_for

if TYPE_CHECKING:
    from batcher.core import ExecutionContext
    from batcher.io.source import Source
    from batcher.kyber.rules.selection import BuildSideDecision
    from batcher.plan.logical import LogicalPlan

__all__ = ["DEFAULT_PARTITIONS", "partitions_from_physical", "run_relational"]

_log = get_logger("api.run")

#: Sentinel for `_execute_in_memory(feedback=...)` meaning "use this context's hub". A plain
#: `None` default cannot express it: `None` is the meaningful value the fast path passes to
#: turn metric recording off, so the two would be indistinguishable.
_HUB = object()


def _phase(name: str, seconds: float) -> None:
    """Record one contract-loop phase timing (stats, Kyber, Carbonite, Core) at DEBUG.

    On the `batcher.api.run` logger these follow the one `log_level` setting, and the phase
    and duration are structured fields, so "where did the control plane spend its time" is
    answerable without re-parsing text.
    """
    log_kv(_log, logging.DEBUG, "run phase", phase=name, seconds=round(seconds, 3))


def _log_decisions(opt, decisions, verdict, *, distributed: bool) -> None:
    """Record the plan verdict at INFO and each join's build-side choice beside it.

    This is what makes ``verbosity="verbose"`` mean what it says: the optimizer's choices
    exist as `Decision` objects on the profile, but the profile is an artifact you read
    afterwards. INFO rather than DEBUG, because a join order flipping or an admission
    verdict turning infeasible is what an operator wants to see without opting into
    per-phase timing noise.
    """
    log_kv(
        _log,
        logging.INFO,
        "plan admitted" if verdict.feasible else "plan infeasible",
        ops=len(opt.ops),
        feasible=verdict.feasible,
        binding=verdict.binding_constraint or "none",
        binding_op=verdict.binding_op or "none",
        distributed=distributed,
    )
    for i, decision in enumerate(decisions):
        log_kv(
            _log,
            logging.INFO,
            "join build side",
            join=i,
            build=("right" if decision.swapped else "left"),
            broadcast=decision.broadcast,
            left_rows=round(decision.left_rows),
            right_rows=round(decision.right_rows),
            why=decision.provenance,
        )


def run_relational(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    *,
    distributed: bool = False,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list[BuildSideDecision]]:
    """Run one relational (non-UDF) plan through Kyber → Carbonite → Core.

    `distributed` fans the plan out across Ray workers using Carbonite's scheduling
    envelope. The distributed executor makes its own shape and partition decisions, so the
    *logical* plan is shipped and single-node rewrites are not overlaid; the mergeable
    algebra is what guarantees the result equals single-node.

    When `execution.adaptive_morsel_sizing` is on (the default) and memory is under
    pressure, Carbonite's pressure-scaled morsel target is activated for the execution
    scope, reaching both the in-process engine and the shipped worker config. That is a
    smaller streaming working set when memory is tight, and a no-op when it is not, so an
    unpressured query stays byte-identical on every path.

    Args:
        plan: The logical plan to run.
        sources: The plan's bound sources, in scan order.
        ctx: The execution context: metadata hub, transport, profile, requested columns.
        distributed: Run across the Ray cluster rather than on this machine.
        materialize: Return a table rather than a streaming source.

    Returns:
        The result and the optimizer's per-join build-side decisions — telemetry the
        adaptive executor reports and the one-shot executors ignore.

    Raises:
        PlanError: Carbonite's admission policy rejected the plan.
    """
    import contextlib

    from batcher import carbonite

    scope: contextlib.AbstractContextManager = contextlib.nullcontext()
    if active_config().execution.adaptive_morsel_sizing:
        # Pass the hub so the morsel target reflects the *learned* per-family peak memory
        # (Carbonite's `LearnedMemoryModel` over recorded `m_peak_bytes`), not just live
        # pressure. Restricting the learned width to *this plan's* operator families keeps a
        # narrow plan from being throttled by a wide aggregate measured in an earlier query.
        from batcher.plan.visitor import walk

        families = {type(node).__name__ for node in walk(plan)}
        # The plan itself goes with them, so a **cold** store can still size the morsel: the
        # learned width only exists after a query of this shape has run, and the first run
        # of a multimodal pipeline is the one that OOMs. The schema knows the width before a
        # row is read, and a measured width still wins wherever there is one.
        adapted = carbonite.ResourceManager(hub=ctx.hub).recommended_config(families, plan)
        if adapted is not None:
            scope = config_context(adapted)
    # Hold an execution slot for the whole run, and narrow this query's pool to its share
    # of the machine. Both are no-ops when `execution.max_concurrent_queries` is 0, which
    # is the default — so an unconfigured deployment executes exactly as before.
    #
    # The grant is threaded through the SAME `config_context` the adaptive morsel sizing
    # already uses rather than a second mechanism: one scoped-config path means one place
    # where "what does this query think the machine looks like" is answered.
    #
    # `query_scope` wraps the whole thing so the execution has a cancellable id and Ctrl-C
    # reaches it. It is outermost of the three because a query queued for an admission slot
    # should be interruptible too — a user waiting on a full queue is exactly who reaches
    # for Ctrl-C, and a scope inside `admit()` would not have been entered yet.
    with (
        query_scope(),
        carbonite.ResourceManager(hub=ctx.hub).admit() as grant,
        _with_grant(scope, grant),
    ):
        return _run_seeded_topn(plan, sources, ctx, distributed, materialize)


def _run_seeded_topn(plan, sources, ctx, distributed, materialize):
    """Run `plan`, first trying it seeded with a remembered top-N bound.

    Kyber decides whether the shape is seedable and what bound to use
    (`kyber.learned_tuning.topn_bound`); what happens *here* is the half a plan → plan pass
    cannot express — run it, look at how many rows came back, and run the original instead
    when the answer says the bound was stale.

    The check is a row count and nothing more, because that is all it has to be: the seeded
    plan removes only rows strictly beyond the bound, so any `k` survivors are the true
    top-k regardless of what the bound was learned from. A short result is the *only* way
    seeding can go wrong, and it is not a wrong answer, just a wasted (cheap) scan.
    """
    from batcher.kyber.learned_tuning.topn_bound import record_topn_bound, seed_topn_bound

    def run(target):
        return _run_relational(
            target, sources, ctx, distributed=distributed, materialize=materialize
        )

    # Seeding needs a materialized result to count, and the whole loop is keyed on the plan
    # *as written* — never on the seeded rewrite, which is a different shape and would learn
    # a bound under a signature no later run asks for.
    seed = seed_topn_bound(plan, ctx.hub) if materialize else None
    if seed is not None:
        table, decisions = run(seed.plan)
        if table is not None and table.num_rows >= seed.k:
            return table, decisions
        # The remembered bound no longer separates `k` rows. Nothing about the seeded run is
        # reusable — a wider bound could admit rows it never looked at — so redo as written.

    table, decisions = run(plan)
    if materialize:
        record_topn_bound(ctx.hub, plan, table)
    return table, decisions


@contextlib.contextmanager
def _with_grant(scope, grant):
    """Apply `grant.workers` to the active config for the block, inside `scope`.

    Nested inside the adaptive-sizing scope rather than replacing it: that scope may have
    adjusted the morsel target from learned statistics, and narrowing the pool must not
    discard it.
    """
    with scope:
        workers = getattr(grant, "workers", 0)
        if not workers:
            yield  # unbounded: the single-query case and the unconfigured default
            return
        current = active_config()
        narrowed = current.replace(
            execution=dataclasses.replace(current.execution, parallelism=workers)
        )
        with config_context(narrowed):
            yield


def _optimize(plan, sources, ctx, *, hardware=None):
    """Collect source statistics, seed the learner, and run Kyber once.

    One optimizer run yields both the physical plan (for admission and costing) and the
    optimized *logical* plan, which the distributed and out-of-core executors read their
    derived join keys and pushed predicates from. Computing both here is what keeps those
    paths from re-running the whole pipeline.
    """
    from batcher import kyber
    from batcher.api.terminal._metadata import seed_column_ndv

    mark = time.perf_counter()
    # Reuse the conductor's already-collected stats when present: the metadata-answer
    # attempt for a missed count()/is_empty() collected them, so a terminal op reads each
    # source's footer once across both passes.
    source_stats = (
        ctx.source_stats
        if ctx.source_stats is not None
        else collect_source_stats(sources, ctx.hub, need_columns=column_bounds_needed(plan))
    )
    _phase("collect_source_stats", time.perf_counter() - mark)

    mark = time.perf_counter()
    # No file footer carries a distinct count, so without this seeding a query's *first*
    # run orders its joins blind.
    seed_column_ndv(ctx.hub, sources, plan)
    # The hardware Kyber plans for, resolved once by the caller: it also keys the machine-
    # scoped learned state (`planning_for`), and resolving it twice across an autoscale is
    # how a read and a write come to disagree.
    result = kyber.optimize_full(
        plan, sources=sources, hub=ctx.hub, source_stats=source_stats, hardware=hardware
    )
    _phase("kyber.optimize_full", time.perf_counter() - mark)
    return result


def _admit(opt, decisions, ctx, *, distributed: bool):
    """Ask Carbonite whether the plan fits, and record what it decided.

    A memory-binding "infeasible" verdict is a spill-friendly counter-offer rather than a
    hard stop: the plan will not fit memory, so it routes out-of-core. Any *other* binding
    constraint has no spill remedy, so it is a real failure.

    Returns:
        The resource manager, the verdict, and whether the plan must spill.

    Raises:
        PlanError: The plan is infeasible for a reason spilling cannot fix.
    """
    from batcher import carbonite

    # Hub-backed so admission, spill, and reservation size from the learned per-family peak
    # memory (measured `m_peak_bytes`) rather than the plan estimate alone. Result-invariant:
    # a spill or reservation choice only changes where data lives.
    rm = carbonite.ResourceManager(hub=ctx.hub)
    verdict = rm.validate(opt)
    _log_decisions(opt, decisions, verdict, distributed=distributed)
    if ctx.profile is not None:
        from batcher.api.terminal.profile import admission_decision, verdict_summary

        ctx.profile.carbonite_summary = verdict_summary(verdict)
        ctx.profile.decisions.append(admission_decision(verdict))

    must_spill = not verdict.feasible and verdict.binding_constraint == "memory"
    if not verdict.feasible and not must_spill:
        raise PlanError(
            f"plan is infeasible (binding constraint: {verdict.binding_constraint}"
            + (f" at {verdict.binding_op}" if verdict.binding_op else "")
            + ")"
        )
    return rm, verdict, must_spill


def _execute_in_memory(logical_opt, plan, opt, ctx, resolved, *, feedback=_HUB):
    """Run the plan through the local engine over already-resolved batches.

    `feedback` is the sink the engine's per-operator `ExecMetrics` are recorded into, and
    defaults to the sentinel meaning "this context's hub" — the ordinary path, where Core
    measuring is the point. `fast_path` passes `None` to turn the recording off, which is a
    quarter of what that path has left to spend; see its module docstring for the trade.
    """
    from batcher import core

    if feedback is _HUB:
        feedback = ctx.hub
    prof = ctx.profile
    # A bare `Scan` is already done: the reader decoded the files and applied the pushed
    # projection, so its batches *are* the result. Handing them back to the engine only to
    # pass them through a no-op operator costs a full round trip across the FFI boundary —
    # about a quarter of the wall clock of `read_parquet(...).collect()`.
    # Profiling wants the per-operator metrics, so it takes the real path.
    table = (
        None
        if prof is not None
        else core.scan_only_result(logical_opt, resolved.batches, opt.source_predicates)
    )
    if table is not None:
        return table

    # The metered path still feeds the hub; it additionally returns the per-operator
    # `ExecMetrics` the conductor's `QueryProfile` needs. An ordinary run takes the plain
    # path and skips even the small metrics serialization.
    if prof is not None:
        batches, metric_ops, usage = core.execute_local_metered(
            opt, resolved.batches, feedback=feedback
        )
        prof.metric_ops = metric_ops
        prof.record_usage(usage)
    else:
        batches = core.execute_local(opt, resolved.batches, feedback=feedback)
    return pa.Table.from_batches(
        batches,
        schema=batches[0].schema if batches else _empty_result_schema(plan, ctx.columns),
    )


def _close_learning_loops(
    plan, logical_opt, ctx, rm, sources, resolved, table, decisions, *, started: float
):
    """Feed this run's measured outcome back to the learners.

    Output size, per-column distinct counts and quantiles, the filter's measured
    selectivity, the conductor's tuning loops, and the memory-pressure flap rate. Every one
    of these only steers a later *performance* choice, so recording is result-invariant.

    Only `learn_column_stats` needs the *resident* input batches, which is why the
    out-of-core paths call `_close_resident_free_loops` instead of this — they never
    materialize the input, but everything else they measured is just as real.
    """
    from batcher.api.terminal._metadata import learn_column_stats

    learn_column_stats(ctx.hub, resolved.batches, sources, plan, resolved.complete)
    _close_resident_free_loops(
        plan, logical_opt, ctx, rm, sources, table.num_rows, decisions, started=started
    )


def record_cardinality_outcome(hub, plan, sources, out_rows: int) -> None:
    """Record the two loops every completed run can close, whatever route it took.

    The measured output count (which corrects this plan signature's cardinality next run)
    and the filter's measured selectivity ratio. Neither needs the input resident, a
    resource manager, or the optimizer's per-join decisions — only the plan, its sources and
    how many rows came out — which is what lets the in-memory, out-of-core and explicit
    ``spill=True`` routes all close them from the one definition instead of each recording
    a different subset (the explicit spill route recorded nothing at all).

    Args:
        hub: The metadata hub to record into; a `None` hub is a no-op.
        plan: The pre-optimization plan — the identity the learner keys on.
        sources: The plan's bound sources, for the selectivity denominator.
        out_rows: Rows the run actually produced.
    """
    from batcher import kyber

    kyber.record_execution(hub, plan, out_rows)
    kyber.record_selectivity(hub, plan, sources, out_rows)


def _close_resident_free_loops(
    plan, logical_opt, ctx, rm, sources, out_rows: int, decisions, *, started: float
):
    """The learning loops that need no resident input batches.

    Split out because the spill paths used to record the output row count and nothing else,
    which left the optimizer learning least from exactly the queries that most needed it:
    a query big enough to go out-of-core taught it no filter selectivity, no shuffle volume,
    no group-reduction ratio, no join outcome, and no memory-pressure flap rate. That is
    self-reinforcing — those are the very inputs whose absence keeps the next run's estimate
    poor enough to spill again.

    None of these reads a batch. `record_selectivity` divides the output count by the
    source's footer row count, `record_run_feedback` takes counts and a wall time, and the
    flap rate comes off the resource manager — all available whether the input was resident
    or streamed from disk. (`learn_column_stats` genuinely cannot run here: it measures
    ndv/quantiles from the scanned batches, and an out-of-core run never holds them.)

    The flap rate is the sharpest of them: it exists to stiffen de-escalation for a workload
    that oscillates under memory pressure, and it was being recorded only on the path that
    did *not* hit pressure.
    """
    from batcher.api.tuning import record_run_feedback, total_source_rows

    record_cardinality_outcome(ctx.hub, plan, sources, out_rows)
    record_run_feedback(
        ctx.hub,
        plan,
        logical_opt,
        decisions,
        out_rows=out_rows,
        input_rows=total_source_rows(sources),
        wall_ms=(time.perf_counter() - started) * 1000.0,
    )
    _record_flap_rate(ctx.hub, rm)
    # The envelope's high-water mark, the spill volume and the cache's hit rate, onto the
    # bus — read here for the same reason the flap rate is, and only here: at admission none
    # of it has happened yet. Self-gating on whether anything is listening, so a process
    # exporting no metrics does not pay for a reading nothing reads.
    _publish_resource_gauges(rm)


def _run_relational(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    *,
    distributed: bool = False,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list[BuildSideDecision]]:
    """The Kyber → Carbonite → Core body, run under the (possibly adapted) config."""
    from batcher._internal.logging import ensure_configured

    ensure_configured()
    started = time.perf_counter()  # the join-strategy bandit's per-run reward clock

    # The hardware Kyber plans for. Distributed runs plan against the cluster's *binding*
    # worker (smallest cores and RAM), so a cache- or memory-sized threshold tracks the
    # workers rather than a possibly-fat driver.
    #
    # Resolved **here**, once, because it does two jobs: it sizes the plan, and it names the
    # machine class every machine-unit learned value in this run is keyed by. The bandit's
    # arm, the broadcast crossover and the build-side priors are all written after the run
    # and read while planning the next one, and both halves run on this driver about work
    # done on those workers. Keyed by the driver they name the wrong machine; keyed by two
    # independent lookups they can disagree across an autoscale, which files a value under a
    # key nothing ever reads. One resolution, one ambient scope, both halves inside it.
    hardware = distributed_hardware() if distributed else None
    with planning_for(getattr(hardware, "fingerprint", "") or ""):
        return _run_relational_scoped(
            plan, sources, ctx, hardware, started, distributed=distributed, materialize=materialize
        )


def _run_relational_scoped(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    hardware,
    started: float,
    *,
    distributed: bool,
    materialize: bool,
) -> tuple[pa.Table | Source, list[BuildSideDecision]]:
    """The body of `_run_relational`, inside the machine-scoping context it opens."""
    from batcher import kyber

    opt, logical_opt, decisions = _optimize(plan, sources, ctx, hardware=hardware)
    if ctx.profile is not None:
        from batcher.api.terminal.profile import record_plan

        record_plan(ctx.profile, opt, plan, distributed, decisions)

    # Kyber's zone-map rules can *prove* a plan empty and record that by rewriting the root
    # to `Limit(x, 0)`. Executing it would read every source in full and throw every row
    # away — worse than not having proved it, since the proof drops the `Filter` and with it
    # the predicate the source would have used to skip files. The proof only exists after
    # optimization, which is why the raw-plan pre-gate in `metadata_answer` cannot see it.
    if materialize:
        empty = proven_empty_table(logical_opt, plan)
        if empty is not None:
            kyber.record_execution(ctx.hub, plan, 0)
            return empty, decisions

    mark = time.perf_counter()
    rm, verdict, must_spill = _admit(opt, decisions, ctx, distributed=distributed)
    _phase("carbonite.validate", time.perf_counter() - mark)

    if distributed:
        result = execute_distributed(
            logical_opt,
            plan,
            sources,
            ctx,
            rm,
            opt,
            decisions,
            materialize=materialize,
            phase=_phase,
            started=started,
        )
        return result, decisions

    # Three signals route a single-node plan out-of-core: admission's counter-offer, the
    # spill estimate, and the input size. The input is the dominant one on a scan-heavy
    # query, because the in-memory path resolves every source to Arrow *before* the engine
    # runs — a 600M-row scan is resident in full even when the query returns four rows.
    input_bytes = projected_input_bytes(sources, opt.source_projections)
    # `resident_total_exceeds_budget` subsumes the input-only check: the input and the
    # plan's peak state are concurrent on this path, so what matters is their sum.
    if must_spill or rm.should_spill(opt) or rm.resident_total_exceeds_budget(input_bytes, opt):
        spilled = spill_to_disk(logical_opt, sources, ctx, rm, opt, verdict)
        if spilled is not None:
            _close_resident_free_loops(
                plan, logical_opt, ctx, rm, sources, spilled.num_rows, decisions, started=started
            )
            return spilled, decisions
        # An *advisory* infeasibility rests on a `Provenance.DEFAULT` guess: worth routing
        # out-of-core, but a guess must never fail a legitimate query (the admission
        # contract), so fall through to the in-memory path instead of raising.
        if must_spill and not verdict.advisory:
            raise PlanError(
                "plan does not fit the memory envelope and has no out-of-core path "
                f"(binding constraint: {verdict.binding_constraint}"
                + (f" at {verdict.binding_op}" if verdict.binding_op else "")
                + ")"
            )

    resolved = resolve_sources(sources, opt, ctx)

    # Reserve the estimated envelope against the process-wide buffer pool for the duration
    # of execution, so concurrent queries draw on one budget. When the reservation does not
    # fit, prefer the out-of-core path over racing them into an OOM — reserve-before-allocate
    # is only real if a `False` actually changes behavior.
    with rm.reserve(rm.estimated_bytes(opt)) as granted:
        if not granted:
            from batcher.dist.spill import spill_collect
            from batcher.plan.profile.usage import UsageStopwatch

            parts = partitions_from_physical(opt) or DEFAULT_PARTITIONS
            # As in `spill_to_disk`: this path runs unmetered engine dispatches, so the
            # whole-phase reading is the only account of what it cost.
            watch = UsageStopwatch()
            spilled = spill_collect(logical_opt, sources, parts)
            if spilled is not None:
                if ctx.profile is not None:
                    ctx.profile.record_usage(watch.finish())
                _close_resident_free_loops(
                    plan,
                    logical_opt,
                    ctx,
                    rm,
                    sources,
                    spilled.num_rows,
                    decisions,
                    started=started,
                )
                return spilled, decisions
        table = _execute_in_memory(logical_opt, plan, opt, ctx, resolved)

    _close_learning_loops(
        plan, logical_opt, ctx, rm, sources, resolved, table, decisions, started=started
    )
    if ctx.profile is not None:
        # Recorded *after* execution, not at admission: the interesting figures are the
        # envelope's high-water mark and the cache's hit rate, and at admission neither has
        # happened yet. Without it a profile shows the plan and the timings but not the
        # memory conditions, and the same plan reads as fast with headroom and slow at the
        # edge of the budget.
        from batcher.api.terminal.profile import resource_decision

        ctx.profile.decisions.append(resource_decision(rm))
    return table, decisions


def _publish_resource_gauges(rm: object) -> None:
    """Publish Carbonite's reading to the event bus. Best-effort; never breaks a query."""
    try:
        rm.publish_stats()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - telemetry must never fail a run
        note_suppressed("api", "publish Carbonite resource gauges", exc)


def _record_flap_rate(hub: object, rm: object) -> None:
    """Persist this run's measured pressure-flap rate. Best-effort; never breaks a query.

    The manager reads a past run's flap rate at construction to stiffen de-escalation for an
    oscillating workload, so without this write that read stays permanently cold.
    """
    try:
        from batcher.carbonite.memory.pressure import record_flap_rate

        rate = rm.flap_rate()  # type: ignore[attr-defined]
        if rate is not None:
            record_flap_rate(hub, rate)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - a learned write must never fail a run
        return
