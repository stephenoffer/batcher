"""Adaptive (intra-query) execution: stage-boundary re-optimization.

A static optimizer plans the whole query once against cardinality *estimates*.
The adaptive executor instead materializes the plan one pipeline breaker at a
time and re-optimizes the remaining plan with that breaker's **exact** output
cardinality fed back as a known-size source. Downstream decisions — notably join
build-side — therefore use *measured* sizes (provenance `exact`) rather than
guesses, even when the estimate would have been badly wrong (e.g. a very
selective filter feeding a join). This is the metadata-driven moat that static
engines (DuckDB) and stage-plan-only adapters can't match.

Mechanism: find the lowest breaker whose inputs are all breaker-free, execute it
through the normal optimize→engine path, replace its subtree with a `Scan` over
an in-memory source holding the result (whose `row_count` is now exact), and
repeat. Each stage is optimized with its inputs already materialized, so a join
over two aggregates picks its build side from the two real sizes.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pyarrow as pa

from batcher.io.source import InMemorySource, Source
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    Scan,
    Sort,
    Union,
    Window,
    is_streamable,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import Provenance

__all__ = ["AdaptiveResult", "execute_adaptive", "resolve_adaptive"]

_BREAKERS = (Aggregate, Sort, Distinct, Window, Limit, Join, Union)


@dataclasses.dataclass(frozen=True, slots=True)
class AdaptiveResult:
    table: pa.Table
    decisions: list  # BuildSideDecision per join, across all re-optimized stages
    stages: int


def resolve_adaptive(
    adaptive: bool | str,
    plan: LogicalPlan,
    sources: list[Source],
    hub,
    *,
    distributed: bool = False,
) -> bool:
    """Resolve ``adaptive="auto"`` to a concrete on/off decision.

    ``"auto"`` (the default) turns stage-by-stage re-optimization on *only* when it
    could change a downstream decision: a join whose operand is produced by a pipeline
    breaker the loop can materialize, and whose size is a pure estimate (a Selinger
    guess, `Provenance.DEFAULT`). That is exactly when measuring the real cardinality
    flips a build-side / broadcast / join-order choice. A plan whose join inputs are
    confidently sized — from source statistics, sketches, or a prior run — gains nothing
    from the extra per-stage materialization, so it stays on the cheaper one-shot path
    (zero adaptive overhead). An explicit ``True``/``False`` always wins.

    When `distributed`, ``"auto"`` ALSO turns it on for a shape the one-shot dispatcher
    cannot route at all — a join whose operand already spans two sources (every 3+-table
    star/snowflake query), which used to raise `PlanError`. There staging is not an
    optimization but the only distributed path. Explicit ``adaptive=False`` still wins.
    """
    if adaptive != "auto":
        return bool(adaptive)
    if distributed:
        from batcher.dist import requires_staging

        if requires_staging(plan):
            return True
    # The cold heuristic (a join over a breaker-produced, guessed-size operand) fires on
    # the first run. Once history exists, ALSO enable for any signature whose measured
    # re-optimization actually *flipped* a plan often enough — a shape whose estimates the
    # loop keeps correcting is worth the per-stage cost even if the structural gate misses
    # it. Gating adaptivity only trades planning overhead, never the result.
    return _adaptive_would_help(plan, sources, hub) or _learned_adaptive_helps(plan, hub)


def _learned_adaptive_helps(plan: LogicalPlan, hub) -> bool:
    """Whether the hub has learned that stage-by-stage re-opt historically helps `plan`."""
    if hub is None:
        return False
    try:
        from batcher.kyber.learned_tuning import learned_adaptive_helps
        from batcher.kyber.signature import plan_signature

        return learned_adaptive_helps(hub, plan_signature(plan))
    except Exception:  # pragma: no cover - a learned read must never break routing
        return False


def _record_adaptive_flip(hub, plan: LogicalPlan, flipped: bool) -> None:
    """Fold this adaptive run's flip outcome into the learned adaptive gate. Best-effort."""
    if hub is None:
        return
    try:
        from batcher.kyber.learned_tuning import record_adaptive_flip
        from batcher.kyber.signature import plan_signature

        record_adaptive_flip(hub, plan_signature(plan), flipped)
    except Exception:  # pragma: no cover - recording must never break a query
        return


# Below this total input-row count, stage-by-stage re-optimization is not worth its
# cost. Adaptive re-opt trades a per-stage materialize + re-plan (~20-40ms of control
# plane) for a better downstream join/build-side choice — a win only when the data is
# large enough that a mis-estimated plan would cost *more* than that overhead. At
# interactive / dev scale (a few million rows, the whole query well under a second) the
# one-shot plan is already fast and the re-plan is pure overhead. The gate reads EXACT
# source row counts, so it separates scales cleanly (TPC-H sf1≈9M off, sf10≈90M on)
# without ever depending on the guessed operand size it is trying to protect against.
_ADAPTIVE_MIN_INPUT_ROWS = 20_000_000


def _total_input_rows(plan: LogicalPlan, estimator) -> float:
    """Sum of every `Scan`'s estimated rows — the query's total input size.

    Scan estimates come straight from EXACT source statistics (footer/catalog row
    counts), so this is a trustworthy size gauge even when downstream operand sizes are
    only guessed.
    """
    from batcher.plan.logical import Scan

    total = 0.0
    for node in _walk(plan):
        if isinstance(node, Scan):
            total += estimator.estimate(node).rows
    return total


def _walk(node: LogicalPlan):
    """Pre-order walk over the plan tree (local helper, no visitor import cycle)."""
    yield node
    for child in _children(node):
        yield from _walk(child)


def _adaptive_would_help(plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Whether any join has a breaker-produced operand whose size is only guessed —
    *and* the total input is large enough for re-optimization to pay for itself."""
    joins = _joins(plan)
    if not joins:
        return False
    estimator = _build_estimator(sources, hub)
    if _total_input_rows(plan, estimator) < _ADAPTIVE_MIN_INPUT_ROWS:
        return False  # small inputs: the one-shot plan is already fast (see threshold note)
    return any(
        not is_streamable(operand) and estimator.estimate(operand).provenance >= Provenance.DEFAULT
        for join in joins
        for operand in (join.left, join.right)
    )


def _build_estimator(sources: list[Source], hub):
    """A `CardinalityEstimator` configured exactly as Kyber's, for the confidence gate."""
    from batcher.api.orchestration import collect_source_stats
    from batcher.config import active_config
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator

    cfg = active_config()
    learned = load_learned_stats(hub) if hub is not None else {}
    return CardinalityEstimator(
        sources,
        learned,
        cfg.optimizer.cardinality,
        source_stats=collect_source_stats(sources, hub),
    )


def _joins(node: LogicalPlan) -> list[Join]:
    """Every `Join` node in the plan (pre-order)."""
    out: list[Join] = [node] if isinstance(node, Join) else []
    for child in _children(node):
        out.extend(_joins(child))
    return out


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
    except Exception:  # pragma: no cover - ray optional
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
            target = _lowest_breaker(plan)
            if target is None:
                break
            final = target is plan
            # Pre-execution row estimate for this stage — gauges, after it runs,
            # whether the optimizer's cardinalities are proving trustworthy.
            est_rows = 0 if final else _estimate_rows(target, srcs, hub)
            # Intermediate stages may stay partitioned (materialize=False); the final
            # stage must collect a table to return.
            import os as _os
            import time as _t

            _dbg = _os.environ.get("BATCHER_DEBUG_STAGES")
            _t0 = _t.perf_counter()
            if _dbg:
                print(
                    f"[stage {stages}] {type(target).__name__} final={final} est={est_rows:.0f}",
                    flush=True,
                )
            result, decs = _run_stage(
                target, srcs, hub, distributed, num_workers, transport, materialize=final
            )
            if _dbg:
                _n = getattr(result, "num_rows", None)
                _elapsed = _t.perf_counter() - _t0
                print(
                    f"[stage {stages}] -> {type(result).__name__} rows={_n} in {_elapsed:.1f}s",
                    flush=True,
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
            plan = _replace(plan, target, Scan(sid, schema))
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
    return pa.table({c: [] for c in node.available_columns()})


def _estimate_rows(node: LogicalPlan, sources: list[Source], hub) -> int:
    """The optimizer's pre-execution row estimate for `node` over `sources` (0 on error).

    Built from the same `CardinalityEstimator` Kyber uses, over the *current* sources
    (which include any exact-sized intermediates spliced in by earlier stages).
    """
    try:
        return int(_build_estimator(sources, hub).estimate(node).rows)
    except Exception:
        return 0


def _estimate_accurate(actual: int, estimate: int, reopt_error: float) -> bool:
    """Whether `actual` and `estimate` agree within a factor of `1 + reopt_error`.

    The **symmetric q-error**, not a relative error normalized by the estimate: the latter is
    bounded by 1 for any over-estimate, so it called every over-estimate accurate — and an
    over-estimate is exactly what this loop exists to catch. Error is multiplicative, so the
    band is too. A positive estimate that produced nothing is a total miss.
    """
    if estimate <= 0 or actual <= 0:
        return False
    return max(actual / estimate, estimate / actual) <= 1.0 + reopt_error


def _children(node: LogicalPlan) -> list[LogicalPlan]:
    if isinstance(node, Join):
        return [node.left, node.right]
    if isinstance(node, Union):
        return list(node.inputs)
    if hasattr(node, "input"):
        return [node.input]
    return []


def _lowest_breaker(node: LogicalPlan):
    """A breaker whose inputs are all breaker-free (so it can run now)."""
    for child in _children(node):
        found = _lowest_breaker(child)
        if found is not None:
            return found
    if isinstance(node, _BREAKERS) and all(is_streamable(c) for c in _children(node)):
        return node
    return None


def _replace(node: LogicalPlan, target: LogicalPlan, repl: LogicalPlan) -> LogicalPlan:
    if node is target:
        return repl
    if isinstance(node, Join):
        return Join(
            _replace(node.left, target, repl),
            _replace(node.right, target, repl),
            node.left_keys,
            node.right_keys,
            node.join_type,
            node.output,
            node.strategy,
        )
    if isinstance(node, Union):
        return Union(tuple(_replace(i, target, repl) for i in node.inputs), node.distinct)
    if hasattr(node, "input"):
        return dataclasses.replace(node, input=_replace(node.input, target, repl))
    return node
