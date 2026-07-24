"""The adaptive stage loop: execute one breaker, re-optimize the rest (control plane, `api`).

The seam: this module is the *engine* of adaptivity — it owns the loop that runs one
pipeline breaker at a time, splices its exact-sized result back into the plan, and
manages the resources a staged query holds (the query-lifetime shuffle fleet, on-disk
intermediates, the worker-loss retry). It asks `gating` whether an estimate held and
`plan_surgery` where to cut; it makes no decisions of its own about *whether* to be
adaptive. That split keeps the resource-owning code (which needs a live cluster to
exercise) apart from the pure decision code.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging

import pyarrow as pa

from batcher._internal.logging import get_logger, log_kv
from batcher.api.adaptive.gating import _estimate_accurate, _estimate_rows, _record_adaptive_flip
from batcher.api.adaptive.plan_surgery import lowest_breaker, replace
from batcher.io.source import InMemorySource, Source
from batcher.plan.logical import LogicalPlan, Scan, empty_result_schema
from batcher.plan.schema import SchemaRef

_log = get_logger("api.adaptive")


__all__ = ["AdaptiveResult", "execute_adaptive"]


@dataclasses.dataclass(frozen=True, slots=True)
class AdaptiveResult:
    table: pa.Table
    decisions: list  # BuildSideDecision per join, across all re-optimized stages
    stages: int


def execute_adaptive(
    plan: LogicalPlan,
    sources: list[Source],
    hub,
    *,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
) -> AdaptiveResult:
    """Run a plan with stage-boundary re-optimization.

    When `distributed`, each breaker stage fans out across Ray workers and its
    *exact* output cardinality feeds the next stage's optimizer — so even at scale
    join build-side and broadcast choices use measured sizes, not estimates. This
    is strictly stronger than Spark AQE (which adapts only at stage boundaries on
    coarse stats); the mergeable algebra guarantees the result equals single-node.

    Intermediate distributed stages keep their result *partitioned on disk* (a
    `MaterializedSource`) or on a persistent Flight fleet rather than collecting it
    to the driver, so a large multi-stage query never funnels every breaker's output
    through driver memory. Those intermediates are cleaned up once the query finishes.

    Fault tolerance: a worker that dies *within* a stage is recovered by the shuffle's
    lineage recompute. A persistent-fleet worker that dies holding an *already
    materialized* intermediate has no fine-grained recompute yet, so on that loss the
    whole query is retried (bounded) on a **fresh** fleet — the failed attempt tore the
    dead fleet down, and a new fleet on the surviving workers re-runs the deterministic
    query to the same result. The retry stays on the Flight path (a fresh single fleet),
    which avoids the cross-stage placement-group deadlock that *disabling* the fleet
    would reintroduce for a multi-stage query. So the persistent fleet is never *less*
    fault-tolerant than the default path it optimizes.
    """
    from batcher.config import active_config

    if not (distributed and active_config().distributed.persistent_fleet):
        return _execute_adaptive(
            plan,
            sources,
            hub,
            distributed=distributed,
            num_workers=num_workers,
            transport=transport,
        )

    from batcher._internal.logging import get_logger

    log = get_logger("api")
    # Bounded so a *persistent* failure (e.g. a real cluster shrink) surfaces instead of
    # looping; config-driven so a churning spot cluster (the `spot` profile) can ride out
    # more preemptions than the conservative default of 2.
    max_attempts = active_config().distributed.fleet_max_attempts
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return _execute_adaptive(
                plan,
                sources,
                hub,
                distributed=distributed,
                num_workers=num_workers,
                transport=transport,
            )
        except _worker_loss_errors() as exc:
            # The fleet lost a worker holding a cross-stage intermediate. The failed
            # attempt already freed the dead fleet (its `finally`); retry on a fresh one.
            last = exc
            log.warning(
                "persistent-fleet worker loss (%s); retry %d/%d on a fresh fleet",
                type(exc).__name__,
                attempt + 1,
                max_attempts,
            )
    raise last  # exhausted retries — surface the last worker-loss error


def _worker_loss_errors() -> tuple[type[BaseException], ...]:
    """Exception types that signal a lost worker/task (safe to retry deterministically),
    not a logic error. Built lazily so `ray` stays an optional import."""
    from batcher._internal.errors import ResourceError

    errs: tuple[type[BaseException], ...] = (ResourceError,)
    try:
        import ray

        errs = (*errs, ray.exceptions.RayActorError, ray.exceptions.RayTaskError)
    except (ImportError, AttributeError):  # pragma: no cover - ray absent, or too old
        pass
    return errs


def _execute_adaptive(
    plan: LogicalPlan,
    sources: list[Source],
    hub,
    *,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
    _fault_inject_stage=None,
) -> AdaptiveResult:
    """The adaptive stage loop (one attempt). `_fault_inject_stage` is a test hook
    invoked with the live fleet after each intermediate stage, to exercise cross-stage
    worker loss."""
    from batcher import kyber

    orig_plan = plan  # capture the signature key before the loop rewrites `plan`
    flipped = False  # did any stage's measured size diverge from its estimate (a re-opt flip)?
    srcs = list(sources)
    # Re-optimize each stage starting from the *optimized* logical structure, not the
    # raw plan. A stage is a subtree rooted at a pipeline breaker; in the raw plan a
    # join's condition can live in a `Filter` *above* that breaker (a comma/cross join
    # whose `WHERE` equality has not yet been folded into the join keys), so splitting
    # there would execute the join as a cartesian product. Folding keys, pushing
    # predicates, and reordering joins over the whole plan first makes every breaker
    # subtree self-contained — the per-stage loop then only refines cost-based choices
    # with measured cardinalities. Holds for single-node and distributed alike.
    #
    # But a `map_batches` operator is opaque to the IR (`to_ir` raises by design), and this
    # whole-plan optimize lowers to IR to run the rule engine — so skip it for a UDF plan
    # (the one-shot path never lowers one whole either; `_run_stage` dispatches every
    # map-carrying stage to `core.execute_with_udfs`, which walks it operator-by-operator).
    # Each self-contained relational stage is still optimized on its own by `run_relational`,
    # so the result matches the non-adaptive run instead of raising `NotImplementedError`.
    from batcher import core

    if not core.has_map_batches(plan):
        plan = kyber.optimize_logical(plan, sources=srcs, hub=hub)
    decisions: list = []
    stages = 0
    intermediates: list = []  # partitioned-on-disk/Flight sources, cleaned up at the end

    # A persistent shuffle fleet (when enabled) lets the distributed Flight path keep
    # each stage's result on the workers instead of collecting it to the driver: one
    # placement group + fleet is reserved for the whole query and every stage borrows
    # it, so there is no per-stage placement churn to deadlock against. Owned here and
    # freed once, in the `finally`. `None` (the default, or single-node/disk) leaves
    # each operator to spawn its own fleet — bit-identical to before.
    fleet = None
    fleet_token = None
    stack = contextlib.ExitStack()
    if distributed:
        from batcher.dist.fleet import maybe_spawn_query_fleet, session_fleet_lease, set_fleet

        # Hold the warm *session* fleet for the whole staged query. Each stage's operator
        # takes its own short lease, but between stages nothing would — and an intermediate
        # left partitioned on the workers is read in place by the next stage, so a teardown
        # in that gap destroys it. (The per-operator lease alone also let the idle timer
        # kill the fleet under any single stage running longer than it.)
        stack.enter_context(session_fleet_lease())
        fleet = maybe_spawn_query_fleet(num_workers, transport)
        if fleet is not None:
            fleet_token = set_fleet(fleet)

    from batcher.config import active_config

    reopt_error = active_config().optimizer.reoptimize_error
    from batcher.dist import requires_staging  # lazy: ray is optional

    try:
        while True:
            target = lowest_breaker(plan)
            if target is None:
                break
            final = target is plan
            # Pre-execution row estimate for this stage — gauges, after it runs,
            # whether the optimizer's cardinalities are proving trustworthy.
            est_rows = 0 if final else _estimate_rows(target, srcs, hub)
            # Intermediate stages may stay partitioned (materialize=False); the final
            # stage must collect a table to return.
            import time as _t

            _t0 = _t.perf_counter()
            log_kv(
                _log,
                logging.DEBUG,
                "stage start",
                stage=stages,
                op=type(target).__name__,
                final=final,
                est_rows=round(est_rows),
            )
            result, decs = _run_stage(
                target, srcs, hub, distributed, num_workers, transport, materialize=final
            )
            log_kv(
                _log,
                logging.DEBUG,
                "stage done",
                stage=stages,
                op=type(result).__name__,
                rows=getattr(result, "num_rows", None),
                seconds=round(_t.perf_counter() - _t0, 3),
            )
            decisions.extend(decs)
            stages += 1
            if final:
                _record_adaptive_flip(hub, orig_plan, flipped)
                return AdaptiveResult(_as_table(result, target), decisions, stages)
            # An intermediate whose measured size missed its estimate is exactly a stage
            # where re-optimizing on the real cardinality can flip a downstream choice —
            # learn that this signature benefits from staying adaptive.
            measured = _stage_row_count(result)
            if measured is not None and not _estimate_accurate(measured, est_rows, reopt_error):
                flipped = True
            # Splice a Scan over the breaker's result (exact-size) for the rest of the
            # plan. A `MaterializedSource` is scanned in place; a collected table is
            # re-wrapped as an in-memory source (the single-node / fallback path).
            src, schema = _stage_source(result)
            # A partitioned intermediate (disk `MaterializedSource` or
            # `FlightMaterializedSource`) owns resources (files / worker actors) freed
            # after the final result; duck-type on `cleanup` so both are tracked.
            if callable(getattr(src, "cleanup", None)):
                intermediates.append(src)
            # Test hook: simulate a cross-stage worker loss once this stage's result is
            # parked on the fleet but before the next stage reads it.
            if fleet is not None and _fault_inject_stage is not None:
                _fault_inject_stage(fleet)
            sid = len(srcs)
            srcs.append(src)
            plan = replace(plan, target, Scan(sid, schema))
            # `reoptimize_error` gate: if this stage's measured size matched its estimate
            # within tolerance, the optimizer's cardinalities are accurate — the rest
            # would re-plan to the same shape, so finish in one shot and stop breaking the
            # pipeline. Single-node (collected-table) path only; a distributed partitioned
            # intermediate stays adaptive and keeps measuring each stage.
            #
            # Never shortcut while the *residual* plan still has no one-shot distributed
            # path (a 4+-table bushy join) — the dispatcher would refuse it. The early exit
            # may skip re-optimization, never the staging the plan structurally requires.
            if (
                isinstance(result, pa.Table)
                and _estimate_accurate(result.num_rows, est_rows, reopt_error)
                and not (distributed and requires_staging(plan))
            ):
                break

        result, decs = _run_stage(
            plan, srcs, hub, distributed, num_workers, transport, materialize=True
        )
        decisions.extend(decs)
        _record_adaptive_flip(hub, orig_plan, flipped)
        return AdaptiveResult(_as_table(result, plan), decisions, stages + 1)
    finally:
        # The final result is a fully in-memory table, independent of the on-disk
        # intermediates, so they can be removed now (best-effort).
        for m in intermediates:
            m.cleanup()
        # Free the query-lifetime fleet once, after every intermediate that borrowed
        # it has been read (the final stage already collected its result to a table).
        if fleet is not None:
            from batcher.dist.fleet import reset_fleet

            reset_fleet(fleet_token)
            fleet.cleanup()
        # Drop the query-scoped session-fleet lease last: every intermediate that read
        # from the warm fleet is gone, so it may now go idle (and time out) safely.
        stack.close()


def _run_stage(
    node: LogicalPlan,
    sources: list[Source],
    hub,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
    *,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list]:
    """Optimize + execute one stage, returning its result and join decisions.

    Each stage runs through the shared `run_relational` orchestrator — the same
    Kyber → Carbonite → Core contract loop the one-shot executors use — so an
    adaptive stage gets the full rule set, resource admission, spill, and the
    metadata feedback loop. Its inputs are already materialized sources with exact
    `row_count`, so the optimizer's estimator reads *measured* sizes for its
    build-side/broadcast/join-order choices, not guesses. With ``materialize=False``
    a distributed stage may return a `MaterializedSource` (result kept on disk).

    A stage carrying `map_batches` is opaque to Kyber, so it bypasses `run_relational` — but
    it still honours `distributed`: the `DistributedExecutor` fans the UDF/inference chain out
    across the workers (`dist.executors.map`), exactly as the one-shot path does. Running it
    through the single-node UDF orchestrator regardless — which is what this did — pinned every
    distributed `map_batches` pipeline (the batch-inference hot path, and Ray Data's home turf)
    to the **driver alone**, using one node of the cluster while the other fifteen sat idle.
    """
    from batcher import core

    ctx = core.ExecutionContext(
        columns=node.available_columns(),
        hub=hub,
        num_workers=num_workers,
        transport=transport,
    )

    if core.has_map_batches(node):
        if distributed:
            from batcher.api import executors

            return executors.select(node, distributed=True).execute(node, sources, ctx), []
        batches = core.execute_with_udfs(node, sources)
        return _table(batches, node), []

    from batcher.api.orchestration import run_relational

    return run_relational(node, sources, ctx, distributed=distributed, materialize=materialize)


def _as_table(result: pa.Table | Source, node: LogicalPlan) -> pa.Table:
    """The stage result as a table — reading a `MaterializedSource` back if needed."""
    if isinstance(result, pa.Table):
        return result
    return _table(list(result.iter_batches()), node)


def _stage_row_count(result: pa.Table | Source) -> int | None:
    """A stage's measured output rows, or `None` when the count is not known exactly.

    A distributed stage parks a `MaterializedSource`/`FlightMaterializedSource`, not a
    `pa.Table`; both carry an exact `row_count` from their reduce tasks. Reading only
    `pa.Table.num_rows` silently skipped the estimate-accuracy check on the distributed
    path, so `learned_adaptive_helps` could never turn on for a distributed shape.
    """
    if isinstance(result, pa.Table):
        return result.num_rows
    row_count = getattr(result, "row_count", None)
    return row_count() if callable(row_count) else None


def _stage_source(result: pa.Table | Source) -> tuple[Source, SchemaRef]:
    """A source + schema to splice in for the next stage's scan over `result`.

    A `MaterializedSource` is passed through (scanned in place, shared-nothing); a
    collected table is wrapped as an `InMemorySource` (its exact `row_count` still
    feeds the optimizer).

    The wrap asks for no zone maps: this relation lives for one stage, so an O(rows)
    min/max pass over it would be recomputed and discarded on every run of the query —
    at sf10 that was 130-200 ms per collect, 13-17% of TPC-H Q9. The measured
    `row_count`, which is what re-optimization actually reads, costs nothing.
    """
    if isinstance(result, pa.Table):
        batches = result.to_batches() or [pa.RecordBatch.from_pylist([], schema=result.schema)]
        return InMemorySource(batches, zone_maps=False), SchemaRef.from_arrow(result.schema)
    return result, SchemaRef.from_arrow(result.schema())


def _table(batches, node) -> pa.Table:
    if batches:
        return pa.Table.from_batches(batches, schema=batches[0].schema)
    # An empty stage result still has to carry the types a matching run would produce.
    # This used to be `pa.table({c: [] for c in ...})`, which types every column `null`
    # — so an adaptive re-plan that staged zero rows handed the next stage a schema the
    # non-adaptive path would have typed `int64`. Share the one neutral spelling.
    names = node.available_columns()
    return pa.Table.from_batches([], schema=empty_result_schema(node, names))
