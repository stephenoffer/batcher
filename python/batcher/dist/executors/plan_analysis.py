"""Plan-shape analysis for the distributed dispatcher.

Pure functions over `LogicalPlan` that decide whether (and how) a plan can be
distributed: locating a pipeline breaker, walking single-input chains, counting
sources, and relabelling a single-source subplan so its scan reads source 0.
No execution, no Ray — just plan inspection.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    RowId,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
    remap_sources,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, scanned_source_ids

# Single-input nodes we can carry as "post-aggregation" work above the breaker — re-run by
# `_apply_above` over the breaker's fully-assembled driver-side result. Every one is a row-wise
# or whole-relation-over-the-assembled-table transform, so applying it on the driver after the
# distributed breaker equals applying it single-node: `Unpivot`/`Sample` are row-wise, and a
# `RowId` (`with_row_index`/`with_random`/`tail`) numbers the assembled result in its final global
# order — exactly what single-node does. Without `Unpivot`/`Sample`/`RowId` here, a breaker
# followed by one of them (`group_by(...).agg(...).unpivot(...)`, `sort(...).with_row_index()`)
# matched no dispatch branch and raised `PlanError` on splittable data.
_PASS_THROUGH = (Filter, Project, Sort, Limit, Distinct, Unnest, Unpivot, Sample, RowId)

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

    The classification itself lives in `plan.logical.transforms` (neutral), because the
    streaming path needs the identical rule: a node that is safe to run per *batch* is
    exactly a node that is safe to run per *partition*. These were two hand-maintained
    copies carrying a comment that they MUST agree; now there is one definition and
    nothing to keep in sync.
    """
    from batcher.plan.logical.transforms import is_partition_independent

    return is_partition_independent(node)


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
    return len(scanned_source_ids(plan)) == 1


def _relabel_single_source(plan: LogicalPlan) -> tuple[LogicalPlan, int]:
    """Rewrite a single-source subplan so its scan reads source 0; return its id."""
    ids = scanned_source_ids(plan)
    if len(ids) != 1:
        raise PlanError(
            f"expected a single-source subplan to relabel, found {len(ids)} sources: {sorted(ids)}"
        )
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


def shuffle_branches(node: LogicalPlan) -> list[LogicalPlan] | None:
    """The map-side sub-plans that may feed ONE shuffle for an operator over `node`.

    `[node]` for the ordinary single-source, breaker-free input every shuffle operator
    already accepts. For a `UNION ALL` of such branches it is the branch list, because the
    shuffle routes by key hash and `partial → combine → finalize` is associative and
    commutative: the partials of N branches merged in one reducer are the partials of their
    concatenation, which is what the union *is*. Returns `None` for any other shape.

    That equality is what makes this a routing fact rather than a second semantics — and it
    is worth having, because the alternative is the driver. `_distributed_union` runs each
    branch to a driver table and concatenates there, so every shape that reduces a union —
    `union(...).group_by(...)`, and `intersect`/`except_`, which lower to an aggregate over a
    union of tagged branches — moved both inputs whole through one node before reducing them.

    Branch *types* must match exactly, not merely their names. `Union.available_schema`
    promotes one branch's `Int64` key against another's `Float64`; independent mappers skip
    that promotion, so `1` and `1.0` would hash to different reducers and one group would
    come back as two — the split-group failure `CLAUDE.md` names, invisible single-node. An
    unreadable schema is not a match: it is refused, leaving the shape on the path it had.
    """
    if not isinstance(node, Union):
        return [node] if _single_source(node) and not _has_breaker(node) else None
    # A DISTINCT union carries a dedup of its own, which map-side partials do not perform;
    # the dispatcher rewrites that shape to `Distinct(UNION ALL)` before it arrives here.
    if node.distinct:
        return None
    branches = list(node.inputs)
    if not all(_single_source(b) and not _has_breaker(b) for b in branches):
        return None
    schemas = [b.available_schema() for b in branches]
    if any(s is None for s in schemas):
        return None
    first = schemas[0].arrow
    if any(s.arrow != first for s in schemas[1:]):
        return None
    return branches


def fused_union_ids(plan: LogicalPlan) -> set[int]:
    """`id()` of every UNION the aggregate directly above it maps into one shuffle.

    The adaptive loop stages the *lowest* runnable breaker, and a union under an aggregate is
    one — so it would run the union on its own and splice its result in, which for a union
    means concatenating every branch on the driver. That is the materialization
    `shuffle_branches` exists to avoid, and staging it would quietly undo the fusion: the
    aggregate would then see a single already-concatenated source and never take the fused
    path at all.

    Identity, not equality, and scoped to a union whose parent is an absorbing aggregate:
    a union under a sort or a limit still stages exactly as it did, because nothing above it
    can absorb it and its staged result is what gives that operator a single source to
    distribute over.
    """
    fused: set[int] = set()

    def walk(node: LogicalPlan) -> None:
        if (
            isinstance(node, Aggregate)
            and isinstance(node.input, Union)
            and shuffle_branches(node.input) is not None
        ):
            fused.add(id(node.input))
        for child in children(node):
            walk(child)

    walk(plan)
    return fused


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


def _pool_key(node: LogicalPlan) -> object | None:
    """What pool a node belongs to, or `None` when it is ordinary stateless CPU work.

    A pool-class stage is keyed by **identity**, not by its resource numbers: two load-once
    models chained one after the other each want their own actor pool, even when both ask for
    one GPU, because putting them in one actor loads both models into one device and runs them
    in series — which is the starvation this whole module exists to remove.
    """
    if not isinstance(node, MapBatches) or not _is_pool_class(node):
        return None
    return id(node)


def split_into_resource_stages(plan: LogicalPlan) -> list[StageSpec] | None:
    """Split a linear `map_batches` pipeline at **every** resource-class boundary.

    This used to split *once*: the CPU prefix, then the first pool-class stage *and everything
    above it*. That one cut is the right one for a two-stage pipeline and wrong for anything
    longer. A
    ``decode → embed → rerank → write`` chain ran its two models in one actor, so they shared
    a device and took turns instead of overlapping; and a CPU postprocess after inference ran
    on the GPU actor, spending device time on host work and forcing the two to scale together.

    The grouping rule is that consecutive stateless-CPU maps form one stage — a Flight hop
    between two host transforms costs more than it saves — and every pool-class map (a GPU
    stage, an explicit `concurrency`, or a class UDF that loads a model once) is a stage of
    its own. A leading scan with no CPU map before the first pool stage is folded *into* that
    stage rather than becoming a hand-off of its own, for the same reason the single cut used
    to decline that shape outright: streaming an unprocessed partition over Flight is not worth
    the hop.

    Args:
        plan: The linear `Scan → map → … → map` plan to split.

    Returns:
        The stages bottom-up, each reading its upstream's published output as source 0, or
        `None` when the chain has no pool-class stage or does not divide into at least two.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.dist.executors.plan_analysis import split_into_resource_stages
            >>> class Model:  # a load-once class UDF: its own pool
            ...     def __call__(self, batch):
            ...         return batch
            >>> ds = bt.from_pydict({"x": [1]})
            >>> ds = ds.ml.map_batches(lambda b: b).ml.map_batches(Model)
            >>> ds = ds.ml.map_batches(lambda b: b)
            >>> [s.wants_pool for s in split_into_resource_stages(ds._plan)]
            [False, True, False]
    """
    nodes = _linear_nodes(plan)
    if not any(_pool_key(n) is not None for n in nodes):
        return None
    groups: list[list[LogicalPlan]] = []
    current_key: object = object()  # a key nothing can equal, so the first node opens a group
    for node in nodes:
        key = _pool_key(node)
        # A pool node always opens a group (its key is unique to it); a CPU node joins the
        # open group only when that group is itself CPU work.
        if key is not None or current_key is not None or not groups:
            groups.append([node])
        else:
            groups[-1].append(node)
        current_key = key
    # A leading scan-only group is not a hand-off worth making — fold it into the stage above,
    # which then reads the partition itself.
    if len(groups) > 1 and not any(isinstance(n, MapBatches) for n in groups[0]):
        groups[1] = [*groups[0], *groups[1]]
        del groups[0]
    if len(groups) < 2:
        return None
    stages = [_stage_spec(groups[0], groups[0][0])]
    relabeled, _sid = _relabel_single_source(stages[0].sub_plan)
    stages[0] = dataclasses.replace(stages[0], sub_plan=relabeled)
    # Every later stage reads its upstream's published morsels as source 0. The empty schema
    # is only consulted when the upstream produced no rows, where this stage's result is empty
    # whatever the schema says.
    stages.extend(
        _stage_spec(group, Scan(0, SchemaRef.from_arrow(_EMPTY_SCHEMA))) for group in groups[1:]
    )
    return stages


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
            if len(scanned_source_ids(side)) > 1 or _has_breaker(side):
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

    A join is fused into the aggregate's reducers (or staged by `_staged_aggregate_over_join`)
    ONLY when both of its sides are a single, breaker-free source — the same `_join_sides_are_map
    _only` precondition the dispatcher's fused paths enforce. A breaker-free `Distinct` is the
    `COUNT(DISTINCT)` rewrite the dispatcher redirects. Everything else would be run per-partition
    as a map prefix, which is unsound, so it must stage.

    Returning True for *any* join (the previous behavior) was correct only because
    `requires_staging` also recurses into the join's children and catches a breaker side there.
    Mirroring the real predicate here removes that fragile coupling: a join with a breaker side
    now stages directly, never risking a silent `PlanError` if the child recursion ever misses it.

    A UNION ALL of map-only branches is handled too: they map into one shuffle
    (`shuffle_branches`), so staging the union first would materialize on the driver the very
    concatenation the shuffle exists to avoid.

    An **ASOF** join is deliberately not a `Join` here, and this is the whole point of
    mirroring rather than approximating. `_fusable_join_aggregate` and `_aggregate_over_join`
    both test `isinstance(j, Join)`, so the dispatcher has no fused route over an ASOF (or
    range) join at all — and claiming one made `requires_staging` answer False for
    `join_asof(...).agg(...)`, which turned off the staging that shape's only distributed path
    runs through. The query then reached the dispatcher, matched nothing, and raised
    `PlanError` on splittable data. Widening the claim to cover a shape the dispatcher does
    not route does not add a route; it removes the fallback.
    """
    if isinstance(node, Union):
        return shuffle_branches(node) is not None
    if isinstance(node, Join):
        return (
            len(scanned_source_ids(node.left)) == 1
            and len(scanned_source_ids(node.right)) == 1
            and not _has_breaker(node.left)
            and not _has_breaker(node.right)
        )
    return isinstance(node, Distinct) and not _has_breaker(node.input)


def _child_plans(plan: LogicalPlan):
    """The `LogicalPlan` children of `plan`, in field order (including tuple fields).

    Delegates to `plan.visitor.children`, which is the one implementation of this walk and
    caches each node class's child-bearing fields. The hand-rolled copy that used to live
    here re-derived them per node *and* was a second place the discovery rules could drift
    from the canonical one.
    """
    return children(plan)


def empty_result_table(plan: LogicalPlan, names: list[str]) -> pa.Table:
    """A zero-row table carrying the plan's REAL column types, in `names` order.

    A distributed stage that produced no rows must return the schema a stage with rows would,
    or `distributed == single-node` is false for every empty result and a downstream concat /
    `write.parquet` / typed projection breaks only on the empty case. Falls back to null-typed
    placeholders when the plan cannot state its types (an opaque `map_batches` output) or when
    they disagree with `names` — strictly safer than trusting a mismatched schema.
    """
    # The schema rule itself lives in neutral `plan` so `api`, `dist`, and `core` cannot
    # drift apart on it (they had, in three different directions). This function stays as
    # the `dist`-facing spelling that returns a *table* rather than a schema.
    from batcher.plan.logical import empty_result_schema

    return pa.Table.from_batches([], schema=empty_result_schema(plan, names))


def _empty_agg_table(agg: Aggregate) -> pa.Table:
    """The typed, zero-row result of an aggregate that saw no rows."""
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return empty_result_table(agg, names)
