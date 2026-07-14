"""The distributed executor — the dispatcher.

Inspects a plan's shape and routes it to the matching distributed operator
(map_batches, aggregate, join, sort, distinct, window, union), each of which reuses
the engine's mergeable primitives so its result is identical to single-node
execution. Shapes that can't be distributed yet fall back to the multi-core
single-node engine.

The per-operator implementations live in the `executors` subpackage; plan
analysis in `executors.plan_analysis`, partitioning/post-breaker helpers in
`executors.partition_io`, and Ray lifecycle + fallback in `executors.ray_runtime`.
The internal helpers re-exported
here (`_apply_above`, `_empty_agg_table`, `_ensure_ray`, `_partition_source`,
`_relabel_single_source`, `_rmtree`) keep `from batcher.dist.executor import ...`
working for the Flight and spill paths.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count

# Re-exported (`X as X`) so the Flight + spill paths can keep importing these
# helpers from `batcher.dist.executor` after the split.
from batcher.dist.executors.partition_io import _apply_above as _apply_above
from batcher.dist.executors.partition_io import _partition_source as _partition_source
from batcher.dist.executors.partition_io import source_pushdown
from batcher.dist.executors.plan_analysis import _empty_agg_table as _empty_agg_table

# Used by the dispatcher below.
from batcher.dist.executors.plan_analysis import (
    _has_breaker,
    _is_linear_map_pipeline,
    _single_source,
    _split_at,
    empty_result_table,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source as _relabel_single_source
from batcher.dist.executors.plan_analysis import _source_ids as _source_ids
from batcher.dist.executors.ray_runtime import _ensure_ray as _ensure_ray
from batcher.dist.executors.ray_runtime import _rmtree as _rmtree
from batcher.dist.executors.ray_runtime import (
    _single_node,
    alive_node_count,
    await_autoscale,
    clamp_workers,
    engine_config_json,
    release_autoscale,
    request_autoscale,
    reset_scheduling_envelope,
    resolve_transport,
    set_scheduling_envelope,
    topology_scope,
    worker_node_memory_bytes,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    RowId,
    Sort,
    Union,
    Window,
)
from batcher.plan.resource import SchedulingEnvelope

__all__ = ["execute_distributed", "resolve_worker_fanout"]


def resolve_worker_fanout(num_workers: int | None) -> int:
    """The worker fan-out for a distributed stage the caller did not size explicitly.

    `execute_distributed` derives its fan-out from the live cluster (`_cluster_fill_workers`
    — enough workers to fill every node's cores). Paths that bypass it and drive Ray tasks
    directly — notably the distributed **write** — must not invent their own constant: a
    hard-coded fan-out uses 4 workers on a 100-node cluster, stranding 96 nodes. This is
    that same sizing, exposed for those callers. An explicit `num_workers` always wins.
    """
    if num_workers is not None:
        return max(1, num_workers)
    _ensure_ray(available_cpu_count())
    fill = _cluster_fill_workers()
    return fill[0] if fill is not None else available_cpu_count()


def execute_distributed(
    plan: LogicalPlan,
    sources: list[Source],
    num_workers: int | None = None,
    transport: str = "disk",
    envelope: SchedulingEnvelope | None = None,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    """Execute a plan across Ray workers, falling back to single-node when needed.

    `transport="flight"` shuffles aggregation partials over Arrow Flight between
    worker actors (object store bypassed) instead of the default disk Arrow-IPC
    files; the result is identical. `transport="auto"` (the surface default) is
    resolved here from cluster topology — Flight on a real multi-node cluster
    (where the disk shuffle's driver-local `work_dir` is unreachable cross-node),
    disk on a single node / shared filesystem.

    `envelope` is Carbonite's metadata-driven scheduling grant: its `n_tasks` sets
    the worker fan-out (replacing a blind `os.cpu_count()`) and its per-task
    resources are applied to every Ray task via `.options(...)` at wrap time. It is
    installed as the ambient grant for the duration of this call. `hub` lets the
    GPU map/inference path record measured utilization for next-run adaptation.

    `materialize=False` lets a stage keep its result partitioned on disk and return a
    `MaterializedSource` (over the disk-shuffle path, where it is supported) instead
    of collecting every reducer's output back to the driver — the adaptive executor
    scans that intermediate in place for the next stage. Shapes that don't support it
    still return a collected `pa.Table`, so the caller must handle either.
    """
    if envelope is not None and num_workers is None:
        workers = max(1, envelope.n_tasks)
    else:
        workers = num_workers or available_cpu_count()

    # Set the grant first so the up-front `_ensure_ray` wraps tasks with it; then ask
    # the autoscaler for the cores this query wants (released in the `finally` so a
    # one-off big job doesn't pin the cluster scaled-up), clamp the fan-out to
    # schedulable capacity, and pick the transport from the resulting topology.
    num_cpus, num_gpus = (envelope.num_cpus, envelope.num_gpus) if envelope else (1.0, 0.0)
    token = set_scheduling_envelope(envelope)
    request_autoscale(math.ceil(workers * num_cpus), workers * num_gpus)
    try:
        _ensure_ray(workers)
        # Wait (bounded, growth-detected) for the autoscaler to bring the cluster toward
        # the cores this query asked for BEFORE sizing the fan-out — otherwise the
        # worker-per-node fill below reads the pre-scale (small) topology and the query
        # never uses the nodes it triggered (the wait inside `clamp_workers` can't help
        # once the fill has made `workers == capacity`). A fixed cluster / unavailable
        # capacity bails fast via the stall window; disabled by `autoscale_wait_s <= 0`.
        # Pure scheduling — the result is identical whether it waits or not.
        if num_workers is None:
            await_autoscale(math.ceil(workers * num_cpus), workers * num_gpus)
        # On a multi-node cluster, fan out to exactly ONE worker per node, each owning
        # that node's cores — the cluster-filling, evenly-distributed shape. This (a) uses
        # every node for any non-trivial query (max, even utilization — more workers than
        # nodes can't add CPU parallelism since cores are the limit, but each node-worker
        # saturates its cores via morsel parallelism + spill), (b) keeps shuffle fan-out
        # minimal (one bucket stream per node, not per core), and (c) gives the reused
        # session fleet a stable, adequate size independent of which query first spawned
        # it — the data-driven count would size the fleet to the first (maybe tiny) query
        # and then under-provision a later big one. An explicit `num_workers` overrides it;
        # a single node falls back to the data-driven `_even_cpu_share` path below.
        fill = None if num_workers is not None else _cluster_fill_workers()
        if fill is not None:
            desired, (workers, num_cpus) = workers, fill
            mem = (
                int(envelope.memory_bytes * max(1, desired) / workers)
                if envelope is not None and envelope.memory_bytes
                else (envelope.memory_bytes if envelope is not None else 0)
            )
            envelope = (
                dataclasses.replace(envelope, n_tasks=workers, num_cpus=num_cpus, memory_bytes=mem)
                if envelope is not None
                else SchedulingEnvelope(num_cpus=num_cpus, n_tasks=workers)
            )
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
        # Ray is up, so the live topology is readable: give each worker an EVEN SHARE of
        # the cluster's CPUs (capped at one node's cores), not the single core Carbonite's
        # per-operator `num_cpus` models — a `_FlightWorker` runs the multi-core executor
        # over a whole partition and a managed cgroup would pin it to 1 core, throttling
        # the scan ~Ncores×. MUST run after `_ensure_ray` (`ray.nodes()` is empty before).
        share = _even_cpu_share(workers)
        if share > num_cpus:
            envelope = (
                dataclasses.replace(envelope, num_cpus=share)
                if envelope is not None
                else SchedulingEnvelope(num_cpus=share, n_tasks=workers)
            )
            num_cpus = share
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
        clamped = clamp_workers(workers, num_cpus, num_gpus)
        # Carbonite sized the per-task memory hint against its *desired* fan-out;
        # once the cluster clamp reduces (or the data-driven want exceeds) it, each
        # real task holds a larger share. Rescale the soft memory hint to the actual
        # worker count and re-install the grant so `.options(memory=)` is honest.
        if envelope is not None and clamped != workers:
            envelope = _rescale_envelope(envelope, workers, clamped)
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
            _ensure_ray(clamped)
        workers = clamped
        # Size the per-worker memory budget from the WORKER node's RAM (hardware metadata →
        # decision): a large driver (e.g. a 197 GiB head) must not hand a 34 GiB worker a
        # budget derived from the driver's memory. Under SPREAD a worker owns its node, so its
        # spill threshold is the node's RAM × soft_limit split across the workers packed on it
        # — enough to use the machine before spilling, bounded so it never OOMs the smallest
        # node. A tighter Carbonite estimate still wins; the topology having no memory info
        # leaves the grant untouched (today's behavior).
        sized = _size_worker_memory(envelope, workers, num_cpus)
        if sized is not envelope:
            envelope = sized
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
        # The worker count and cluster size are now settled (autoscale-wait + clamp done),
        # so snapshot the topology once for the whole placement/transport/shuffle phase
        # instead of paying a fresh `ray.nodes()` RPC at each of its ~5 read sites — the
        # O(nodes)-per-read amplification that bites at thousands of nodes.
        with topology_scope():
            transport = resolve_transport(transport, workers)
            return _dispatch(
                plan,
                sources,
                workers,
                transport,
                hub,
                materialize=materialize,
                metrics_out=metrics_out,
            )
    finally:
        release_autoscale()  # let the autoscaler reclaim what this query scaled up
        reset_scheduling_envelope(token)


def _worker_node_cpus() -> list[float]:
    """CPU counts of the nodes eligible to run distributed workers.

    Excludes the Ray **head** node (marker ``node:__internal_head__``) when at least one
    other node exists: the head runs the GCS / dashboard / job supervisor, and scheduling
    data operators on it causes contention and instability (Ray Data hits this — the
    guides' "set `num_cpus=0` on the head" rule). Many managed clusters already give the
    head 0 CPU, so the `> 0` filter handles it there; excluding by marker makes Batcher
    correct on a raw Ray cluster whose head has cores too — "works on any cluster type". A
    single-node cluster (head only) keeps the head, since it must run the work.
    """
    import ray

    alive = [n for n in ray.nodes() if n.get("Alive")]
    non_head = [n for n in alive if "node:__internal_head__" not in n.get("Resources", {})]
    nodes = non_head if non_head else alive  # keep the head only if it's the whole cluster
    return [c for c in (float(n.get("Resources", {}).get("CPU", 0.0)) for n in nodes) if c > 0]


def _cluster_fill_workers() -> tuple[int, float] | None:
    """The cluster-filling fan-out: enough `min`-core workers to fill EVERY node's cores.

    Returns `(workers, num_cpus)` on a genuine multi-node cluster. `num_cpus` = the smallest
    worker node's cores, so a worker is placeable on any node (SPREAD-safe); `workers` =
    ``Σ floor(node_cores / num_cpus)`` over the worker nodes — one worker per `num_cpus`-core
    slice. A **homogeneous** cluster reduces to one worker per node exactly as before; a
    **heterogeneous** one gives a node with k times the smallest node's cores k workers, so its
    extra cores are used instead of stranded idle under a uniform one-worker-per-node grant
    (a 64-core node next to 32-core nodes was running at half utilization). Any worker count
    is result-correct under the mergeable algebra, so this only affects saturation; the
    per-worker CPU grant stays `min`-sized so the SPREAD placement group still fits every
    node. Returns `None` on a single node or unreadable topology, so the caller keeps the
    data-driven `_even_cpu_share` sizing. Ray must already be initialized.
    """
    try:
        node_cpus = _worker_node_cpus()
        if len(node_cpus) <= 1:
            return None
        num_cpus = max(1.0, float(int(min(node_cpus))))
        workers = sum(max(1, int(c // num_cpus)) for c in node_cpus)
        return workers, num_cpus
    except Exception:
        return None


def _even_cpu_share(workers: int) -> float:
    """CPUs to grant each distributed worker so the fan-out isn't single-core-starved.

    Two hard constraints: the grant must be **placeable on every node** (capped at
    `min(node cores)`, since workers are SPREAD across nodes) and must not over-subscribe
    (capped at `floor(total / workers)`). Within those, hand each worker as many cores as
    possible (`>= 1`) so its parallel scan-read + fold use the node, not one cgroup-pinned
    core. The grant is deliberately *uniform* — skew is handled orthogonally (LPT-balanced
    splits in `_balance`; salted hot join keys in `join_par`). Returns 1.0 (historical
    default) when topology is unavailable.
    """
    try:
        node_cpus = _worker_node_cpus()
        if not node_cpus or workers <= 0:
            return 1.0
        placeable = float(int(min(node_cpus)))  # fits the smallest node (SPREAD-safe)
        non_oversubscribing = float(sum(node_cpus) // workers)  # workers x grant <= cluster
        return max(1.0, min(placeable, non_oversubscribing))
    except Exception:
        return 1.0


def _size_worker_memory(
    envelope: SchedulingEnvelope | None, workers: int, num_cpus: float
) -> SchedulingEnvelope | None:
    """Set the per-worker memory budget from the worker node's RAM (hardware-aware spill
    threshold). Returns `envelope` unchanged (same object) when the topology advertises no
    node memory, so a cluster that doesn't report it keeps today's behavior.

    A worker owns its node under SPREAD; when several workers pack one node they split its
    RAM. The budget is `min(node_mem * soft_limit / workers_per_node, Carbonite's estimate)`
    — Carbonite's tighter data-driven estimate still wins, but an unset (unbounded) or
    driver-oversized grant is clamped to what the worker machine can actually hold.
    """
    node_mem = worker_node_memory_bytes()
    if node_mem <= 0:
        return envelope
    nodes = max(1, alive_node_count())
    per_node_workers = max(1, math.ceil(workers / nodes))
    from batcher.config import active_config

    budget = int(node_mem * active_config().memory.soft_limit / per_node_workers)
    if budget <= 0:
        return envelope
    if envelope is None:
        return SchedulingEnvelope(num_cpus=num_cpus, n_tasks=workers, memory_bytes=budget)
    current = envelope.memory_bytes
    new_mem = budget if current <= 0 else min(current, budget)
    if new_mem == current:
        return envelope
    return dataclasses.replace(envelope, memory_bytes=new_mem)


def _rescale_envelope(
    envelope: SchedulingEnvelope, desired: int, actual: int
) -> SchedulingEnvelope:
    """Rescale a scheduling grant from its desired fan-out to the actual one.

    The per-task memory hint was `peak // desired`; with `actual` tasks each holds
    `peak // actual`, so scale the hint by `desired / actual` (and update `n_tasks`).
    """
    actual = max(1, actual)
    memory_bytes = int(envelope.memory_bytes * desired / actual) if envelope.memory_bytes else 0
    return dataclasses.replace(envelope, n_tasks=actual, memory_bytes=memory_bytes)


def _is_splittable_source(source: Source) -> bool:
    """Whether `source` yields real per-chunk splits workers can read directly (Parquet
    row-groups, lakehouse fragments) — vs an in-memory/iterator source that returns one
    `WholeSourceSplit` and would have to be shipped to the workers. Only the former is
    worth distributing for a breaker-free scan."""
    from batcher.io.splits import WholeSourceSplit

    try:
        splits = source.splits()
    except Exception:
        return False
    return bool(splits) and not (len(splits) == 1 and isinstance(splits[0], WholeSourceSplit))


# Max `LIMIT k` for the shuffle-free distributed top-N (driver merges `workers x k` rows).
_TOPN_MAX_ROWS = 1_000_000


def _collapse_limits(lim: Limit) -> tuple[int, int, LogicalPlan]:
    """Fold a chain of nested `Limit`s into one `(n, offset, base)` over the chain's base.

    Kyber's limit pushdown leaves a `Limit` above the one it pushed into the scan, so the
    dispatcher routinely sees `Limit(Limit(scan))`; treating the inner `Limit` as an opaque
    (non-map) input would strand the whole shape on `_unsupported`.

    An outer `Limit(n, offset)` keeps rows `[offset, offset + n)` of its input, and the
    inner one already restricted its child to `[i_offset, i_offset + i_n)`. Composing:
    the child rows kept are `[i_offset + offset, ...)`, and at most `min(n, i_n - offset)`
    of them survive (clamped at 0 when the outer offset skips past the inner window).
    """
    n, offset = lim.n, lim.offset
    base = lim.input
    while isinstance(base, Limit):
        n = max(0, min(n, base.n - offset))
        offset = base.offset + offset
        base = base.input
    return n, offset, base


def _join_sides_are_map_only(join) -> bool:
    """Whether both join operands are a single source with a breaker-free (map-only) plan.

    Each side is shipped to every worker and re-run against that worker's partition, so a
    breaker on a side is evaluated per-partition: `limit(5).join(dim)` kept 5 rows on each of
    4 workers and returned 20. Such joins are `requires_staging` — the inner breaker runs as
    its own distributed stage first — so refusing here turns a wrong answer into the staged
    path (or, with `adaptive=False`, a loud error).
    """
    return (
        _single_source(join.left)
        and _single_source(join.right)
        and not _has_breaker(join.left)
        and not _has_breaker(join.right)
    )


def _fusable_join_aggregate(agg: Aggregate) -> bool:
    """Whether `agg` can be distributed by reusing the join's co-partitioning — no shuffle.

    **This is exchange elimination, and it is decided by the physical-property layer.** The
    shuffle join co-partitions both sides by the join key, so its output is hash-partitioned
    by that key (`kyber.properties.hash_partitioned_on`). An aggregate whose group keys are a
    *superset* of it therefore has every group entirely inside one bucket: each reducer joins
    *and* aggregates its own bucket, the union of the reducer outputs is the complete result,
    and no second shuffle is needed. Because no cross-bucket combine is needed, even a
    non-mergeable aggregate is correct here.

    Kyber owns the question "what distribution does this relation already have"; `dist`
    schedules against the answer rather than re-deriving it, so the two cannot drift.
    """
    from batcher.kyber.properties import PhysicalProperties, hash_partitioned_on, satisfies

    j = agg.input
    if not isinstance(j, Join) or not _join_sides_are_map_only(j):
        return False
    group_cols = tuple(gk.expr.name for gk in agg.group_keys if isinstance(gk.expr, Col))
    if not group_cols:
        return False
    partitioned_on = hash_partitioned_on(j)
    if not partitioned_on:
        return False  # nothing guaranteed (an outer join, or no equi-key): must re-shuffle
    # The group keys must *cover* the partitioning, so no group straddles two buckets — the
    # containment `satisfies` checks, with the group keys as what is on hand.
    return satisfies(
        PhysicalProperties(hash_partitioned_on=group_cols),
        PhysicalProperties(hash_partitioned_on=partitioned_on),
    )


def _aggregate_over_join(agg: Aggregate) -> bool:
    """Whether `agg` is an aggregate over a join of two single sources (any join type or
    group keys) — the general case `_fusable_join_aggregate` does not cover.

    Distributed as partial-aggregate-per-reducer + a driver-side `combine_finalize` of the
    small partials (the mergeable two-phase): the join is aggregated *on the workers* and
    only group-cardinality-many partial rows reach the driver, instead of collecting the
    whole join to the head to aggregate it single-node (the 70s→~1s join fix).
    """
    j = agg.input
    return isinstance(j, Join) and _join_sides_are_map_only(j)


# A whole-relation window aggregate is the same computation as the equivalent GROUP BY
# aggregate over zero keys; only the SQL spelling of `avg` differs from the engine's `mean`.
_WINDOW_AGG_TO_AGG = {"sum": "sum", "avg": "mean", "min": "min", "max": "max", "count": "count"}


def _is_empty_relation(plan: LogicalPlan) -> bool:
    """Whether `plan` provably yields zero rows, so there is nothing to distribute.

    Kyber folds a predicate it can prove false against the source's statistics into
    `Limit(0)`. Every row-preserving or row-reducing operator above it still yields zero
    rows. The walk stops at an `Aggregate` (a zero-key aggregate over an empty relation
    returns exactly ONE row), a `Join` (an outer join emits the surviving side), and a
    `Union` (another branch may have rows) — none of those is empty just because an input is.
    """
    from batcher.plan.logical import Filter, Project, Unnest

    node = plan
    while True:
        if isinstance(node, Limit) and node.n == 0:
            return True
        if isinstance(node, (Filter, Project, Sort, Distinct, Limit, Window, Unnest)):
            node = node.input
            continue
        return False


def _is_broadcastable_global_window(window: Window) -> bool:
    """Whether `window` is `<agg>(x) OVER ()` — one scalar per function, over every row.

    With no PARTITION BY *and* no ORDER BY, an aggregate window's frame is the whole
    relation, so every row receives the same value. That is exactly a zero-key aggregate,
    which is mergeable — so it distributes, unlike an ordered global window (a running sum,
    `row_number()`, `lag`), which needs one global row order and has no distributed path.
    An explicit frame re-introduces per-row bounds, so it is excluded too.
    """
    return (
        not window.partition_keys
        and not window.order_keys
        and window.rank_limit is None
        and all(
            f.func in _WINDOW_AGG_TO_AGG and f.frame is None and f.input is not None
            for f in window.functions
        )
    )


def _distributed_global_window(
    above: list[LogicalPlan], window: Window, sources: list[Source], workers: int, transport: str
) -> pa.Table:
    """`<agg>(x) OVER ()` — aggregate the whole relation, then broadcast the scalars.

    Two distributed passes, both linear: a zero-key mergeable aggregate reduces the relation
    to one row on the driver (O(1) memory, not O(rows)), then a stateless map appends each
    result as a literal column. Collecting every row onto one node to compute the window —
    what a naive "one partition" implementation does — is precisely the cliff this avoids.
    """
    from batcher.plan.expr_ir import AggExpr, Lit
    from batcher.plan.logical import Aggregate, AggregateSpec, Project, Projection

    totals = _dispatch(
        Aggregate(
            input=window.input,
            group_keys=(),
            aggregates=tuple(
                AggregateSpec(alias=f.alias, agg=AggExpr(_WINDOW_AGG_TO_AGG[f.func], f.input))
                for f in window.functions
            ),
        ),
        sources,
        workers,
        transport,
    )
    if totals.num_rows == 0:
        # No input rows ⇒ no output rows; the window preserves its input's cardinality.
        result = empty_result_table(window, window.available_columns())
        return result if not above else _apply_above(above, result)

    scalars = totals.to_pydict()
    columns = window.input.available_columns()
    broadcast = Project(
        input=window.input,
        items=(
            *(Projection(alias=c, expr=Col(c)) for c in columns),
            *(Projection(alias=f.alias, expr=Lit(scalars[f.alias][0])) for f in window.functions),
        ),
    )
    result = _dispatch(broadcast, sources, workers, transport)
    return result if not above else _apply_above(above, result)


def _hoist_computed_sort_key(sort: Sort):
    """Rewrite `ORDER BY <expr>, …` so the LEADING key is a plain column.

    Returns `(sort', drop_key)` — a sort over a `Project` that materializes the computed
    leading key as a hidden column, plus the `Project` that drops it again — or `None` when
    the leading key is already a column (the common case, left byte-identical).

    The distributed sort range-partitions on the leading key's *values*, which it can only
    read from a column, so `df.sort(col("a") + col("b"))` had no distributed path at all.
    Materializing the key once per row in the map prefix is exactly what the single-node
    sort does internally, so this is a rewrite, not an approximation: the hidden column is
    computed before the shuffle, drives the partitioning, and is projected away afterwards.
    """
    from batcher.plan.logical import Project, Projection, SortKeySpec

    key = sort.keys[0]
    if isinstance(key.expr, Col):
        return None

    columns = sort.input.available_columns()
    hidden = "__sort_key"
    while hidden in columns:  # never shadow a user column
        hidden += "_"

    with_key = Project(
        input=sort.input,
        items=(
            *(Projection(alias=c, expr=Col(c)) for c in columns),
            Projection(alias=hidden, expr=key.expr),
        ),
    )
    rewritten = dataclasses.replace(
        sort,
        input=with_key,
        keys=(
            SortKeySpec(Col(hidden), descending=key.descending, nulls_first=key.nulls_first),
            *sort.keys[1:],
        ),
    )
    drop_key = Project(
        input=rewritten, items=tuple(Projection(alias=c, expr=Col(c)) for c in columns)
    )
    return rewritten, drop_key


def _staged_aggregate_over_join(
    above: list[LogicalPlan],
    agg: Aggregate,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
) -> pa.Table:
    """Disk-transport aggregate over an arbitrary join: shuffle the join, then the aggregate.

    The Flight path folds the partial aggregate into the join's reducers, so only partials
    cross the network — strictly better, and it stays the flight branch. The disk shuffle has
    no such fold, and collecting the whole join to the driver to aggregate it single-node is
    the exact cliff `_unsupported` exists to prevent. So run two distributed shuffles: the
    join keeps its result partitioned on disk (`materialize=False`), and the aggregate then
    treats that intermediate as an ordinary splittable source. Driver memory stays O(groups),
    and the result is the single-node one (both stages are mergeable).
    """
    from batcher.dist.executors.aggregate import _distributed_aggregate
    from batcher.dist.executors.join import _distributed_join
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    joined = _distributed_join([], agg.input, sources, workers, materialize=False, hub=hub)
    if isinstance(joined, pa.Table):
        # The join shape had no partitioned form and already collected; aggregate it here
        # rather than re-shuffling a table the driver is holding anyway.
        from batcher.io.source import InMemorySource

        intermediate: Source = InMemorySource(joined.to_batches())
        cleanup = None
    else:
        intermediate, cleanup = joined, joined.cleanup

    sid = len(sources)
    staged = dataclasses.replace(
        agg, input=Scan(source_id=sid, schema=SchemaRef(intermediate.schema()))
    )
    try:
        return _distributed_aggregate(
            above, staged, [*sources, intermediate], workers, hub, metrics_out=metrics_out
        )
    finally:
        if cleanup is not None:
            cleanup()


def _dispatch(
    plan: LogicalPlan,
    sources: list[Source],
    workers: int,
    transport: str,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    # A provably-empty relation (Kyber folded a false predicate to `Limit(0)`) has no data
    # to distribute, so run it on one node: correct, instant, and not the perf cliff
    # `_unsupported` guards against. Without this, `filter(<provably false>)` under an
    # operator that is not a `_split_at` pass-through (a window) reached `_unsupported` and
    # raised, purely because the folded `Limit` reads as a pipeline breaker.
    if _is_empty_relation(plan):
        return _single_node(plan, sources)

    # Batch-inference / embedding pipelines (map_batches): distribute the linear
    # map chain across workers — the Ray Data competitor path.
    from batcher.core.udf import has_map_batches

    if has_map_batches(plan):
        if _is_linear_map_pipeline(plan) and _single_source(plan):
            # A linear chain with a stateless-CPU prefix feeding a GPU/load-once stage
            # streams the two with the stages overlapped (CPU prepares k+1 while the GPU
            # runs k), when enabled; otherwise it runs embarrassingly parallel. Any such
            # chain qualifies (incl. CPU→GPU→postprocess); shapes with no overlap to win
            # fall back to the non-overlapped map.
            from batcher.config import active_config

            if active_config().distributed.stream_inference:
                from batcher.dist.executors.plan_analysis import split_at_first_pool_boundary

                if split_at_first_pool_boundary(plan) is not None:
                    from batcher.dist.streaming import stream_distributed_pipeline

                    return stream_distributed_pipeline(plan, sources, workers, hub)
            from batcher.dist.executors.map import _distributed_map

            return _distributed_map(plan, sources, workers, hub)
        # An aggregate over a linear map/UDF pipeline: distribute the UDF across workers,
        # partial-aggregate on each, combine on the driver (the Ray Data map_batches→agg
        # shape). The UDF runs cluster-wide instead of single-node on the driver.
        agg_split = _split_at(plan, Aggregate)
        if agg_split is not None:
            above, agg = agg_split
            sub = agg.input
            if has_map_batches(sub) and _is_linear_map_pipeline(sub) and _single_source(sub):
                from batcher.dist.executors.map import _distributed_map_aggregate

                return _distributed_map_aggregate(above, agg, sources, workers)
        # Any other map+breaker shape has no distributed path yet.
        return _unsupported(plan, sources, "a map_batches/UDF pipeline feeding this operator")

    # Breaker-free scan/filter/project over a SPLITTABLE source (Parquet row-groups,
    # lakehouse fragments): fan the read out so each worker reads its own splits in
    # parallel — the distributed-scan case — instead of one node reading the whole
    # source. In-memory/iterator sources stay single-node (shipping them to workers
    # costs more than the parallel CPU saves). Reuses `_distributed_map`'s stateless
    # task path (no UDF/GPU ⇒ one task per partition).
    if _is_linear_map_pipeline(plan) and _single_source(plan):
        sid = next(iter(_source_ids(plan)))
        if sid < len(sources) and _is_splittable_source(sources[sid]):
            from batcher.dist.executors.map import _distributed_map

            return _distributed_map(plan, sources, workers, hub)

    # A bare `LIMIT n OFFSET k` over a breaker-free single source (`df.limit(10)`,
    # `df.head()`, `df.filter(...).limit(10)` — the most common interactive shape, and
    # until now a hard failure on distributed data). Each worker keeps only the first
    # `k + n` rows of its OWN partition and the driver re-slices their concatenation.
    #
    # This is exact, not a sample: the global first `k + n` rows are a prefix of the
    # source, so every one of them lies in some partition's own first `k + n` rows; and
    # `_distributed_map` assembles partition results **by index** (`gather_map_results`
    # is index-addressed), so their concatenation is the source's row order and
    # `slice(k, n)` selects precisely the rows the single-node engine returns. Only
    # `workers x (k + n)` rows ever reach the driver, never the whole source.
    #
    # `hub=None`: the per-worker plan is truncated, so its row count must not be learned
    # as the source's cardinality. A `map_batches` prefix returned above, so the pipeline
    # here is pure scan/filter/project/unnest.
    limit_split = _split_at(plan, Limit)
    if limit_split is not None:
        above, lim = limit_split
        n, offset, base = _collapse_limits(lim)
        if _single_source(base) and _is_linear_map_pipeline(base):
            sid = next(iter(_source_ids(base)))
            if sid < len(sources) and _is_splittable_source(sources[sid]):
                from batcher.dist.executors.map import _distributed_map

                per_worker = Limit(input=base, n=offset + n, offset=0)
                table = _distributed_map(per_worker, sources, workers, None)
                table = table.slice(offset, n)
                return table if not above else _apply_above(above, table)

    # `with_row_index` — a single global counter, so a per-partition run would restart it at
    # zero on every worker. Instead run the (row-wise) input distributed and number the rows
    # on the driver: `_distributed_map` assembles partitions BY INDEX, so their concatenation
    # is the source's own row order and the counter lands on exactly the rows single-node
    # numbers. `with_random` (position-keyed hash) and `tail` (row index + filter) both lower
    # to `RowId`, so all three distribute through this one path.
    rowid_split = _split_at(plan, RowId)
    if rowid_split is not None:
        above, rowid = rowid_split
        if _single_source(rowid.input) and _is_linear_map_pipeline(rowid.input):
            sid = next(iter(_source_ids(rowid.input)))
            if sid < len(sources) and _is_splittable_source(sources[sid]):
                from batcher.dist.executors.map import _distributed_map

                table = _distributed_map(rowid.input, sources, workers, None)
                index = pa.array(
                    range(rowid.offset, rowid.offset + table.num_rows), type=pa.int64()
                )
                # `RowId.available_columns()` puts the index FIRST, and the engine emits it
                # non-nullable (a counter is never null). Match both exactly, or the schema
                # differs from single-node and `pa.concat_tables` rejects the pair.
                field = pa.field(rowid.alias, pa.int64(), nullable=False)
                table = table.add_column(0, field, index)
                return table if not above else _apply_above(above, table)

    agg_split = _split_at(plan, Aggregate)
    if agg_split is not None:
        above, agg = agg_split
        # Aggregate over a DISTINCT (the `count_distinct → distinct + count` rewrite, or a
        # user `distinct().agg(...)`) must be caught BEFORE the map/shuffle aggregate path:
        # that path runs `agg.input` as a per-partition map prefix, but a DISTINCT has GLOBAL
        # semantics — run map-local, each partition dedups independently and the reducer sums
        # the per-partition counts, double-counting any value spanning two source partitions
        # (the COUNT(DISTINCT) overcount). Distribute the DISTINCT first (globally deduped),
        # then aggregate over its result. NOTE (perf): a high-cardinality DISTINCT collects the
        # deduped rows to the driver for the outer aggregate — correct, but slow for a 15M-key
        # COUNT(DISTINCT); the correct-and-fast form is a 2nd distributed stage (future work).
        # Scoped to a direct `Distinct` input: OTHER breakers (nested aggregate / sort) keep
        # the map-local path, which is correct for the composable aggregates that dominate.
        if (
            isinstance(agg.input, Distinct)
            and _single_source(agg.input)
            and not _has_breaker(agg.input.input)
        ):
            from batcher.dist.executors.distinct import _distributed_distinct

            return _distributed_distinct(
                [*above, agg],
                agg.input,
                sources,
                workers,
                transport,
                materialize=materialize,
                hub=hub,
                metrics_out=metrics_out,
            )
        # The map/shuffle aggregate path: run `agg.input` as the per-partition map prefix, then
        # partial-aggregate + shuffle + combine. Sound ONLY over a breaker-free prefix — a map
        # prefix is evaluated independently on every partition. A `Limit` prefix would then keep
        # `n` rows *per partition* (`limit(100).group_by(k).agg(count())` counted 4x on 4
        # workers), and a nested `Aggregate` would hand this one per-partition partial groups
        # (`max(sum per k)` read a per-partition max). Both returned wrong answers silently.
        # Such shapes are `requires_staging`, so the staged executor runs the inner breaker
        # first; here we refuse rather than compute. A join input has two sources, so
        # `_single_source` skips it to the join handlers below.
        if _single_source(agg.input) and not _has_breaker(agg.input):
            if transport == "flight":
                from batcher.dist.flight_aggregate import execute_aggregate_flight

                # `materialize=False` (when an adaptive-loop fleet is ambient) keeps the
                # result on the workers as a `FlightMaterializedSource` the next stage
                # reads in place — no driver collect; else it spawns + collects as before.
                return execute_aggregate_flight(
                    above, agg, sources, workers, materialize=materialize
                )
            from batcher.dist.executors.aggregate import _distributed_aggregate

            return _distributed_aggregate(
                above, agg, sources, workers, hub, materialize=materialize, metrics_out=metrics_out
            )
        # Aggregate over an inner join grouped by ⊇ the join key: fold each reducer's bucket
        # to groups (exchange elimination) — full join never collects on head.
        if _fusable_join_aggregate(agg):
            if transport == "flight":
                from batcher.dist.flight_join import execute_join_flight

                return execute_join_flight(above, agg.input, sources, workers, fused_agg=agg)
            from batcher.dist.executors.join import _distributed_join_aggregate

            return _distributed_join_aggregate(above, agg, agg.input, sources, workers, hub)
        # General aggregate over a (non-key-aligned, or non-inner) join.
        if _aggregate_over_join(agg):
            if transport == "flight":
                # Fold the partial aggregate into the join's reducers, so only partials
                # cross the network and the join never collects on the head.
                from batcher.dist.flight_join import execute_join_flight

                return execute_join_flight(
                    above, agg.input, sources, workers, fused_agg=agg, combine_partials=True
                )
            return _staged_aggregate_over_join(above, agg, sources, workers, hub, metrics_out)

    join_split = _split_at(plan, Join)
    if join_split is not None:
        above, join = join_split
        if _join_sides_are_map_only(join):
            if transport == "flight":
                from batcher.dist.flight_join import execute_join_flight

                # `materialize=False` (an intermediate stage of a multi-join query) keeps
                # each reducer's joined bucket on its worker and returns a
                # `FlightMaterializedSource` the next stage reads in place — no driver
                # round-trip per join, which is what makes a 3+-table query scale.
                return execute_join_flight(above, join, sources, workers, materialize=materialize)
            from batcher.dist.executors.join import _distributed_join

            return _distributed_join(
                above, join, sources, workers, materialize=materialize, hub=hub
            )

    # ASOF join with `by` keys: co-partition both sides by the `by` keys (equal `by`
    # values hash together, so each bucket is an independent ASOF join). A keyless
    # ASOF needs one global order on `on` → stays single-node.
    asof_split = _split_at(plan, AsofJoin)
    if asof_split is not None:
        above, asof = asof_split
        if asof.left_by and _join_sides_are_map_only(asof):
            return _distributed_asof(above, asof, sources, workers)

    # A top-level sort over a scannable input distributes via range partitioning on the
    # leading key. That key must be a plain COLUMN (the range partitioner splits on its
    # values); a computed one — `ORDER BY a + b`, `ORDER BY lower(name)` — is hoisted into a
    # hidden column first. Secondary keys may be any expression: only the leading key drives
    # the partitioning, the rest are evaluated by each reducer's local sort.
    sort_split = _split_at(plan, Sort)
    if sort_split is not None:
        above, sort = sort_split
        if _single_source(sort.input) and sort.keys and not _has_breaker(sort.input):
            hoisted = _hoist_computed_sort_key(sort)
            if hoisted is not None:
                sort, drop_key = hoisted
                above = [*above, drop_key]  # innermost: drops the hidden key from the result
            if transport == "flight":
                from batcher.dist.flight_sort import execute_sort_flight, execute_topn_flight

                # Small `ORDER BY ... LIMIT k` → mergeable top-N (no shuffle); else full sort.
                if sort.limit is not None and sort.limit <= _TOPN_MAX_ROWS:
                    return execute_topn_flight(above, sort, sources, workers)
                return execute_sort_flight(above, sort, sources, workers)
            from batcher.dist.executors.sort import _distributed_sort

            return _distributed_sort(above, sort, sources, workers, hub)

    # DISTINCT over a breaker-free single source: dedup via the aggregate shuffle.
    distinct_split = _split_at(plan, Distinct)
    if distinct_split is not None:
        above, distinct = distinct_split
        if _single_source(distinct.input) and not _has_breaker(distinct.input):
            from batcher.dist.executors.distinct import _distributed_distinct

            return _distributed_distinct(
                above,
                distinct,
                sources,
                workers,
                transport,
                materialize=materialize,
                hub=hub,
                metrics_out=metrics_out,
            )

    # A window partitioned by plain columns over a breaker-free source: hash-shuffle
    # rows by the partition keys so each partition is computed whole on one reducer.
    # A window with NO partition keys has nothing to shuffle on; when it is also
    # order-free it is a whole-relation aggregate broadcast, which distributes (an
    # *ordered* global window needs one global row order and still has no path).
    window_split = _split_at(plan, Window)
    if window_split is not None:
        above, window = window_split
        if _single_source(window.input) and not _has_breaker(window.input):
            if _is_broadcastable_global_window(window):
                return _distributed_global_window(above, window, sources, workers, transport)
            if window.partition_keys and all(isinstance(k, Col) for k in window.partition_keys):
                if transport == "flight":
                    from batcher.dist.flight_window import execute_window_flight

                    return execute_window_flight(above, window, sources, workers)
                from batcher.dist.executors.window import _distributed_window

                return _distributed_window(above, window, sources, workers, hub)

    # UNION: distribute each branch independently, then concatenate (+ dedup).
    union_split = _split_at(plan, Union)
    if union_split is not None:
        above, union = union_split
        from batcher.dist.executors.union import _distributed_union

        return _distributed_union(above, union, sources, workers, transport)

    # No distributed path matched this shape.
    return _unsupported(plan, sources, "an unsupported operator combination")


def _unsupported(plan: LogicalPlan, sources: list[Source], reason: str):
    """Either fail loudly (the silent-single-node antipattern) or run a legitimately
    single-node-only plan on one node.

    Silent single-node fallback for a plan that *should* be distributed is an
    antipattern: it masks a missing distributed path behind a quiet perf cliff (the whole
    job on one node) and an OOM risk. So when any input is a **splittable** storage source
    (real distributed data), raise loudly with the shape — the gap must be fixed, not
    hidden. When every source is in-memory / non-splittable there is no distributed data
    to speak of, so executing it on one node is the correct plan, not a fallback.

    The splittable check is scoped to the sources **this plan actually reads**
    (`_source_ids`), not the whole ambient `sources` list: a later adaptive stage — e.g. a
    trailing `project` over an in-memory intermediate materialized by an earlier stage —
    reads only that in-memory source and is correctly a single-node local transform, even
    though the original splittable scan source is still present in `sources` (unused here).
    Without this scoping such a tail stage would wrongly raise as "unsupported."
    """
    read_ids = _source_ids(plan)
    if any(_is_splittable_source(sources[i]) for i in read_ids if i < len(sources)):
        from batcher._internal.errors import PlanError
        from batcher.dist.executors.plan_analysis import requires_staging

        # A join over a multi-source operand HAS a distributed path — the staged one. The
        # caller reached here only by forcing `adaptive=False`, so say that rather than
        # implying the operator is missing.
        hint = (
            "this shape distributes stage by stage (a join whose operand spans two sources, "
            "or a pipeline breaker beneath another breaker); it was disabled by an explicit "
            '`adaptive=False`. Re-run with `adaptive=True` (or the default `"auto"`). '
            "Running it in one shot would evaluate the inner plan once per partition and "
            "return wrong values, so it is refused rather than computed."
            if requires_staging(plan)
            else "File/extend the distributed operator, or run with distributed=False "
            "to force single-node explicitly."
        )
        raise PlanError(
            "distributed execution has no path for this plan shape "
            f"({reason}); refusing to silently fall back to single-node on distributed "
            f"data. {hint}"
        )
    return _single_node(plan, sources)


# --- ASOF join (co-partition by the `by` keys) --------------------------------
# Lives here (not in the `executors` subpackage, which is at its file-count ceiling)
# alongside the dispatch that routes to it. An ASOF match only ever pairs rows that
# share a `by` group, and `partition_batches` hashes equal `by` values to the same
# bucket on both sides, so each bucket is an independent ASOF join whose union is the
# full result. It reuses the equi-join's generic map/reduce tasks verbatim — only the
# partition keys (`by`) and the reducer IR (`asof_join`) differ.


def _require_shared_scratch(op: str) -> None:
    """Refuse a disk-shuffle-only operator on a multi-node cluster with no shared scratch.

    The disk shuffle hands only *paths* between tasks, so `work_dir` must resolve on every
    node. Every other operator is steered off disk by `resolve_transport` (which picks
    Flight the moment the cluster spans more than one node), but `op` has no Flight path, so
    nothing protects it: a worker would open a driver-local `/tmp` path that does not exist
    on its own node — a `FileNotFoundError` at best, and silently missing rows if a
    same-named directory happens to exist there. Fail with the fix instead.
    """
    from batcher.dist.shuffle_io import shared_scratch_root

    if shared_scratch_root() is not None or alive_node_count() <= 1:
        return
    from batcher._internal.errors import PlanError

    raise PlanError(
        f"distributed `{op}` uses the disk shuffle, which needs a scratch directory every "
        "worker node can reach, and this cluster has no shared mount. Point "
        "`MemoryConfig.spill_dir` at a shared filesystem, or run it single-node "
        "(`distributed=False`)."
    )


def _asof_reducer_ir(asof: AsofJoin) -> dict:
    """IR for the per-bucket ASOF join of a left input (source 0) and right input
    (source 1). Mirrors `AsofJoin.to_ir()` but substitutes the per-task scans."""
    return {
        "op": "asof_join",
        "left": {"op": "scan", "source_id": 0},
        "right": {"op": "scan", "source_id": 1},
        "left_on": asof.left_on,
        "right_on": asof.right_on,
        "left_by": list(asof.left_by),
        "right_by": list(asof.right_by),
        "backward": asof.direction == "backward",
        "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in asof.output],
    }


def _distributed_asof(
    above: list[LogicalPlan], asof: AsofJoin, sources: list[Source], workers: int
) -> pa.Table:
    """Co-partition both sides by the `by` keys and ASOF-join each bucket in parallel."""
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.join import _join_map_task, _join_reduce_task
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.shuffle_io import read_ipc

    _ensure_ray(workers)
    _require_shared_scratch("asof_join")
    cfg_json = engine_config_json()  # driver config → shipped to workers

    left_plan, left_sid = _relabel_single_source(asof.left)
    right_plan, right_sid = _relabel_single_source(asof.right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    asof_ir = json.dumps(_asof_reducer_ir(asof))
    left_proj, left_pred = source_pushdown(left_plan, 0)
    right_proj, right_pred = source_pushdown(right_plan, 0)

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dir = distributed_work_dir("batcher_asof_")
    try:
        left_parts = _partition_source(
            sources[left_sid], workers, work_dir, tag="L", projection=left_proj, predicate=left_pred
        )
        right_parts = _partition_source(
            sources[right_sid],
            workers,
            work_dir,
            tag="R",
            projection=right_proj,
            predicate=right_pred,
        )

        pol = speculation_policy()

        # Co-partition each side by its `by` keys with the generic join map task.
        def _left_map_for(i: int):
            return _join_map_task.remote(
                left_ir, list(asof.left_by), left_parts[i], workers, work_dir, "L", i, cfg_json
            )

        def _right_map_for(i: int):
            return _join_map_task.remote(
                right_ir, list(asof.right_by), right_parts[i], workers, work_dir, "R", i, cfg_json
            )

        left_paths = gather_with_backups(
            [_left_map_for(i) for i in range(len(left_parts))], _left_map_for, pol
        )  # [mapper][bucket]
        right_paths = gather_with_backups(
            [_right_map_for(i) for i in range(len(right_parts))], _right_map_for, pol
        )

        def _reduce_for(r: int):
            l_inputs = [paths[r] for paths in left_paths]
            r_inputs = [paths[r] for paths in right_paths]
            return _join_reduce_task.remote(asof_ir, l_inputs, r_inputs, work_dir, r, cfg_json)

        result_paths = gather_with_backups(
            [_reduce_for(r) for r in range(workers)], _reduce_for, pol
        )

        batches: list[pa.RecordBatch] = []
        # `_join_reduce_task` yields `(path, rows, ...)` — it grew a trailing metrics field —
        # so unpack only what this path needs rather than pinning the tuple's arity.
        for entry in result_paths:
            path = entry[0]
            if path is not None:
                batches.extend(read_ipc(path))
    finally:
        _rmtree(work_dir)

    if not batches:
        result = empty_result_table(asof, [o.alias for o in asof.output])
    else:
        result = pa.Table.from_batches(batches)
    return result if not above else _apply_above(above, result)
