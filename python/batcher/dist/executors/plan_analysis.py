"""Plan-shape analysis for the distributed dispatcher.

Pure functions over `LogicalPlan` that decide whether (and how) a plan can be
distributed: locating a pipeline breaker, walking single-input chains, counting
sources, and relabelling a single-source subplan so its scan reads source 0.
No execution, no Ray — just plan inspection.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    Scan,
    Sort,
    Unnest,
    remap_sources,
)
from batcher.plan.schema import SchemaRef

# Single-input nodes we can carry as "post-aggregation" work above the breaker.
_PASS_THROUGH = (Filter, Project, Sort, Limit, Distinct, Unnest)

# Schema for an intermediate stage's scan: only read when the upstream stage produced
# zero rows (`_execute_node` falls back to it), where the downstream result is empty
# regardless — so an empty schema is correct and never truncates real data.
_EMPTY_SCHEMA = pa.schema([])


def _has_breaker(node: LogicalPlan) -> bool:
    """True if `node`'s subtree contains a pipeline breaker (so it can't be a
    plain map input to a distributed sort)."""
    if isinstance(node, (Aggregate, Sort, Join, Distinct, Limit)):
        return True
    if _is_row_wise(node):
        return _has_breaker(node.input)
    return isinstance(node, Scan) is False  # unknown node → be conservative


def _is_row_wise(node: LogicalPlan) -> bool:
    """Whether `node` is a stateless, partition-independent transform of its input.

    Running these on each partition and concatenating gives exactly the single-node result:
    `Unnest` (explode) and `Unpivot` (melt) multiply rows but hold no state, and a
    **fraction** `Sample` keeps a row iff a seeded hash of its values falls under the
    fraction — a per-row predicate, so partitioning cannot change which rows survive. (That
    hash reads every column, which is why `kyber.rules.projections` must not prune below a
    `Sample`; with pruning, a worker sampled a different column set than single-node did.)

    A *fixed-count* `Sample` (`n=`) is NOT row-wise: it keeps the `n` smallest-hash rows of
    the WHOLE relation, so running it per partition would keep `n` rows on every worker.
    """
    from batcher.plan.logical import Sample, Unpivot

    if isinstance(node, Sample):
        return node.n is None
    return isinstance(node, (Filter, Project, Unnest, Unpivot))


def _split_at(plan: LogicalPlan, breaker_type: type):
    """Walk down single-input nodes to the first node of `breaker_type`.

    Returns `(above, breaker)` where `above` is the chain of nodes from the root
    down to (not including) the breaker, or None if none is reachable that way.
    """
    above: list[LogicalPlan] = []
    node = plan
    while True:
        if isinstance(node, breaker_type):
            return above, node
        if isinstance(node, _PASS_THROUGH):
            above.append(node)
            node = node.input
        else:
            return None


def _single_source(plan: LogicalPlan) -> bool:
    return len(_source_ids(plan)) == 1


def _relabel_single_source(plan: LogicalPlan) -> tuple[LogicalPlan, int]:
    """Rewrite a single-source subplan so its scan reads source 0; return its id."""
    ids = _source_ids(plan)
    assert len(ids) == 1, "expected a single-source subplan"
    sid = next(iter(ids))
    return remap_sources(plan, -sid), sid


def _is_linear_map_pipeline(plan: LogicalPlan) -> bool:
    """True if the plan is a linear chain of scan / filter / project / map_batches and the
    row-wise reshapers (`unnest`, `unpivot`, fraction `sample`) — embarrassingly parallel per
    partition, with no relational breaker.

    These are stateless and row-wise, so running them on each partition and concatenating
    gives exactly the single-node result. Excluding `Unnest` once forced the whole RAG
    ingest shape — scan, chunk, explode, embed — onto one node; `Unpivot` and fraction
    `sample` were likewise stranded, though `plan.logical.is_streamable` already called them
    partition-independent.
    """
    node = plan
    while True:
        if isinstance(node, Scan):
            return True
        if isinstance(node, MapBatches) or _is_row_wise(node):
            node = node.input
        else:
            return False


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One resource-class stage of a linear `map_batches` pipeline.

    `sub_plan` is a linear plan whose leaf scan reads source 0 — the real input
    partition for the first stage, the upstream stage's published output for the
    rest — so one actor (pool) runs exactly this stage. The resource attributes size
    and place that pool: a CPU preprocess stage (`num_gpus == 0`, stateless) feeding a
    GPU/load-once inference stage is the canonical two-stage split this enables.
    """

    sub_plan: LogicalPlan
    num_gpus: float
    accelerator_type: str | None
    wants_pool: bool
    concurrency: object  # int | tuple[int, int] | None


def _is_pool_class(node: MapBatches) -> bool:
    """Whether a map stage runs a GPU or load-once model (so it wants a resident actor
    pool): a positive `num_gpus`, an explicit `concurrency`, or a class/factory `fn`
    that builds its model once. The first such stage in a linear chain is where the
    stateless-CPU prefix hands off to the model."""
    return node.num_gpus > 0 or node.concurrency is not None or isinstance(node.fn, type)


def _linear_nodes(plan: LogicalPlan) -> list[LogicalPlan]:
    """A linear scan→…→root plan as a bottom-up node list `[scan, …, root]`."""
    chain: list[LogicalPlan] = []
    node = plan
    while True:
        chain.append(node)
        if isinstance(node, Scan):
            break
        node = node.input
    chain.reverse()
    return chain


def _rebuild_stage(group: list[LogicalPlan], base: LogicalPlan) -> LogicalPlan:
    """Fold a stage's bottom-up node `group` onto `base` (the stage's input scan),
    skipping the original scan (replaced by `base`)."""
    cur = base
    for node in group:
        if isinstance(node, Scan):
            continue
        cur = dataclasses.replace(node, input=cur)
    return cur


def _stage_spec(group: list[LogicalPlan], base: LogicalPlan) -> StageSpec:
    """Build the `StageSpec` for one node `group`, summarizing its map stages'
    resource class (max GPU, first pinned accelerator, any load-once pool)."""
    maps = [n for n in group if isinstance(n, MapBatches)]
    num_gpus = max((m.num_gpus for m in maps), default=0.0)
    accel = next((m.accelerator_type for m in maps if m.accelerator_type is not None), None)
    concurrency: object = None
    wants_pool = False
    for m in maps:
        cls_pool = m.concurrency is not None or isinstance(m.fn, type)
        wants_pool = wants_pool or cls_pool
        if m.concurrency is not None:
            concurrency = m.concurrency if concurrency is None else concurrency
    return StageSpec(_rebuild_stage(group, base), num_gpus, accel, wants_pool, concurrency)


def split_at_first_pool_boundary(plan: LogicalPlan) -> tuple[StageSpec, StageSpec] | None:
    """Split a linear `map_batches` pipeline at the first GPU/load-once stage.

    Returns `(producer, consumer)`: the producer is the stateless-CPU prefix (scan +
    read/decode/preprocess `map_batches`) up to — but not including — the first
    GPU/load-once stage; the consumer is that stage onward (inference plus any
    postprocess maps). The producer streams its output to the consumer pool so the CPU
    prefix overlaps the model stage, for *any* linear pipeline — not just an exactly
    two-stage one (a CPU→GPU→CPU-postprocess chain splits into CPU producer + GPU+post
    consumer). Returns `None` when there is no GPU/load-once stage, or no CPU
    `map_batches` precedes it (nothing worth a Flight hand-off) — the caller then keeps
    the non-overlapped distributed-map path.

    Pure plan inspection. The producer's leaf scan is relabeled to read source 0 (the
    real partition); the consumer's leaf scan reads an intermediate source 0 holding the
    producer's published output (an empty schema is fine — it is only consulted when the
    producer yielded no rows, where the consumer result is empty regardless).
    """
    nodes = _linear_nodes(plan)
    boundary = next(
        (i for i, n in enumerate(nodes) if isinstance(n, MapBatches) and _is_pool_class(n)),
        None,
    )
    if boundary is None:
        return None
    prefix, suffix = nodes[:boundary], nodes[boundary:]
    # Require real CPU compute in the prefix (a decode/preprocess map) to overlap; a
    # bare scan prefix isn't worth a Flight hand-off for an in-memory partition.
    if not any(isinstance(n, MapBatches) for n in prefix):
        return None
    producer = _stage_spec(prefix, prefix[0])  # prefix[0] is the original Scan
    relabeled, _sid = _relabel_single_source(producer.sub_plan)
    producer = dataclasses.replace(producer, sub_plan=relabeled)
    consumer = _stage_spec(suffix, Scan(0, SchemaRef.from_arrow(_EMPTY_SCHEMA)))
    return producer, consumer


def _source_ids(plan: LogicalPlan) -> set[int]:
    if isinstance(plan, Scan):
        return {plan.source_id}
    ids: set[int] = set()
    for field in dataclasses.fields(plan):
        value = getattr(plan, field.name)
        if isinstance(value, LogicalPlan):
            ids |= _source_ids(value)
        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, LogicalPlan):
                    ids |= _source_ids(v)
    return ids


def requires_staging(plan: LogicalPlan) -> bool:
    """Whether distributing `plan` in one shot is impossible, but staging it would work.

    Two shapes qualify, both because a single-shot executor would have to run a subplan
    **once per partition**, which is only sound for a map-only (breaker-free) subplan:

    1. *A join whose operand spans two sources* — every 3+-table (star/snowflake) query. The
       dispatcher co-partitions exactly two sources per join, so there is no one-shot path.
    2. *A breaker beneath a breaker* — `limit(100).group_by(k).agg(...)`, `agg(agg(x))`,
       `limit(5).join(dim)`, `window(limit(...))`. The aggregate and join executors ship the
       inner plan to every worker as a "map prefix"; a `Limit` then keeps `limit` rows **per
       partition** and a nested `Aggregate` produces per-partition partial groups the outer
       aggregate re-aggregates. Both silently return wrong answers, not errors.

    Staged, the lowest breaker runs distributed on its own, its result is materialized, and
    the operator above it then sees a breaker-free scan — which is exactly the contract each
    executor already assumes. So this is a *routing* fact, not a missing operator.

    An `Aggregate` over a `Join` or a clean `Distinct` is excluded: the dispatcher has real
    fused paths for those. Shapes no amount of staging can distribute (an ordered global
    window, `sample`, `row_id`) are not listed here and still surface loudly.
    """
    from batcher.plan.logical import AsofJoin, Window

    if isinstance(plan, (Join, AsofJoin)):
        for side in (plan.left, plan.right):
            if len(_source_ids(side)) > 1 or _has_breaker(side):
                return True
    elif isinstance(plan, Aggregate) and _has_breaker(plan.input):
        if not _dispatcher_handles_aggregate_input(plan.input):
            return True
    elif isinstance(plan, Window) and _has_breaker(plan.input):
        return True  # `Window` is not a `_split_at` pass-through, so nothing carries it up
    return any(
        requires_staging(child)
        for child in _child_plans(plan)  # a breaker nested under any operator counts
    )


def _dispatcher_handles_aggregate_input(node: LogicalPlan) -> bool:
    """Whether `_dispatch` has a real fused path for an aggregate over this (breaker) input.

    A join is fused into the aggregate's reducers (or staged by `_staged_aggregate_over_join`),
    and a breaker-free `Distinct` is the `COUNT(DISTINCT)` rewrite the dispatcher redirects.
    Everything else would be run per-partition as a map prefix, which is unsound.
    """
    from batcher.plan.logical import AsofJoin

    if isinstance(node, (Join, AsofJoin)):
        return True
    return isinstance(node, Distinct) and not _has_breaker(node.input)


def _child_plans(plan: LogicalPlan):
    """The `LogicalPlan` children of `plan`, in field order (including tuple fields)."""
    for field in dataclasses.fields(plan):
        value = getattr(plan, field.name)
        if isinstance(value, LogicalPlan):
            yield value
        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, LogicalPlan):
                    yield v


def empty_result_table(plan: LogicalPlan, names: list[str]) -> pa.Table:
    """A zero-row table carrying the plan's REAL column types, in `names` order.

    A distributed stage that produced no rows must return the schema a stage with rows would,
    or `distributed == single-node` is false for every empty result and a downstream concat /
    `write_parquet` / typed projection breaks only on the empty case. Falls back to null-typed
    placeholders when the plan cannot state its types (an opaque `map_batches` output) or when
    they disagree with `names` — strictly safer than trusting a mismatched schema.
    """
    schema = plan.available_schema()
    if schema is None or list(schema.arrow.names) != list(names):
        return pa.table({n: pa.array([], pa.null()) for n in names})
    return pa.Table.from_batches([], schema=schema.arrow)


def _empty_agg_table(agg: Aggregate) -> pa.Table:
    """The typed, zero-row result of an aggregate that saw no rows."""
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return empty_result_table(agg, names)
