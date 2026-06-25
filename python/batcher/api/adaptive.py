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

# How many times to (re-)attempt a persistent-fleet query on a fresh fleet before
# giving up on worker loss. Bounded so a *persistent* failure (e.g. a real cluster
# shrink) surfaces instead of looping; the first attempt plus one retry covers a
# transient preemption.
_FLEET_MAX_ATTEMPTS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class AdaptiveResult:
    table: pa.Table
    decisions: list  # BuildSideDecision per join, across all re-optimized stages
    stages: int


def resolve_adaptive(adaptive: bool | str, plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Resolve ``adaptive="auto"`` to a concrete on/off decision.

    ``"auto"`` (the default) turns stage-by-stage re-optimization on *only* when it
    could change a downstream decision: a join whose operand is produced by a pipeline
    breaker the loop can materialize, and whose size is a pure estimate (a Selinger
    guess, `Provenance.DEFAULT`). That is exactly when measuring the real cardinality
    flips a build-side / broadcast / join-order choice. A plan whose join inputs are
    confidently sized — from source statistics, sketches, or a prior run — gains nothing
    from the extra per-stage materialization, so it stays on the cheaper one-shot path
    (zero adaptive overhead). An explicit ``True``/``False`` always wins.
    """
    if adaptive != "auto":
        return bool(adaptive)
    return _adaptive_would_help(plan, sources, hub)


def _adaptive_would_help(plan: LogicalPlan, sources: list[Source], hub) -> bool:
    """Whether any join has a breaker-produced operand whose size is only guessed."""
    joins = _joins(plan)
    if not joins:
        return False
    estimator = _build_estimator(sources, hub)
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

    import logging

    last: BaseException | None = None
    for attempt in range(_FLEET_MAX_ATTEMPTS):
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
            logging.getLogger(__name__).warning(
                "persistent-fleet worker loss (%s); retry %d/%d on a fresh fleet",
                type(exc).__name__,
                attempt + 1,
                _FLEET_MAX_ATTEMPTS,
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
    srcs = list(sources)
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
    if distributed:
        from batcher.dist.fleet import maybe_spawn_query_fleet, set_fleet

        fleet = maybe_spawn_query_fleet(num_workers, transport)
        if fleet is not None:
            fleet_token = set_fleet(fleet)

    from batcher.config import active_config

    reopt_error = active_config().optimizer.reoptimize_error
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
            result, decs = _run_stage(
                target, srcs, hub, distributed, num_workers, transport, materialize=final
            )
            decisions.extend(decs)
            stages += 1
            if final:
                return AdaptiveResult(_as_table(result, target), decisions, stages)
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
            if isinstance(result, pa.Table) and _estimate_accurate(
                result.num_rows, est_rows, reopt_error
            ):
                break

        result, decs = _run_stage(
            plan, srcs, hub, distributed, num_workers, transport, materialize=True
        )
        decisions.extend(decs)
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
    """
    from batcher import core
    from batcher.api.orchestration import run_relational

    if core.has_map_batches(node):
        batches = core.execute_with_udfs(node, sources)
        return _table(batches, node), []

    ctx = core.ExecutionContext(
        columns=node.available_columns(),
        hub=hub,
        num_workers=num_workers,
        transport=transport,
    )
    return run_relational(node, sources, ctx, distributed=distributed, materialize=materialize)


def _as_table(result: pa.Table | Source, node: LogicalPlan) -> pa.Table:
    """The stage result as a table — reading a `MaterializedSource` back if needed."""
    if isinstance(result, pa.Table):
        return result
    return _table(list(result.iter_batches()), node)


def _stage_source(result: pa.Table | Source) -> tuple[Source, SchemaRef]:
    """A source + schema to splice in for the next stage's scan over `result`.

    A `MaterializedSource` is passed through (scanned in place, shared-nothing); a
    collected table is wrapped as an `InMemorySource` (its exact `row_count` still
    feeds the optimizer).
    """
    if isinstance(result, pa.Table):
        batches = result.to_batches() or [pa.RecordBatch.from_pylist([], schema=result.schema)]
        return InMemorySource(batches), SchemaRef.from_arrow(result.schema)
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
    """Whether `actual` is within `reopt_error` relative error of a positive `estimate`."""
    return estimate > 0 and abs(actual - estimate) / estimate <= reopt_error


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
