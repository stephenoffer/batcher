"""Single-node out-of-core aggregation via partition-and-spill, plus the spill dispatcher.

This is the same radix-partition machinery the distributed shuffle uses, run locally and
sequentially against disk — realizing the plan's thesis that single-node out-of-core and
PB-scale distribution are *one* mechanism with disk vs. network as the sink.

Pipeline (memory bounded by a single source batch + one bucket's partial state):

    for each source batch (streamed):          # bounded input memory
        mapped   = run the map sub-plan on the batch
        partial  = partial_aggregate(mapped)    # pre-aggregate to shrink data
        buckets  = hash-partition partial by group key into K on-disk files
    for each bucket (one at a time):            # bounded reduce memory
        result  += combine_finalize(read(bucket))

Because each group key hashes to exactly one bucket, combining per bucket yields the correct
global result — identical to the in-memory aggregation, but a group-by over more distinct
groups than fit in RAM still completes. The scratch plumbing lives in `scratch`; the
ordering/binary breakers that share it live in `dist.spill_breakers` (sort/join/window).
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source, _single_source
from batcher.dist.executors.plan_analysis import empty_result_table, restore_declared_types
from batcher.dist.spill.buckets import (
    GRACE_DEPTH,
    GRACE_SUB_BUCKETS,
    BucketWriters,
    over_envelope,
    read_reserved_bucket,
    regrace,
    spill_scratch,
    split_salt,
)
from batcher.dist.spill.scratch import (
    _fd_safe,
    _iter_spill_morsels,
    map_projection,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import col
from batcher.plan.ir_specs import agg_spec_json
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    Sort,
    Window,
)

__all__ = [
    "execute_spilling_aggregate",
    "spill_collect",
]


def _peel_to_breaker(plan: LogicalPlan) -> LogicalPlan | None:
    """The spillable breaker under a chain of leading row-wise/limit ops, or `None`.

    Peels `Project`/`Filter`/`Limit` and returns the underlying node if it is a spillable
    breaker (`Distinct`/`Aggregate`/`Join`/`Sort`/`Window`) — the marker that an operator's
    input must itself be spilled out-of-core rather than streamed per batch.
    """
    node = plan
    while isinstance(node, (Project, Filter, Limit)):
        node = node.input
    return node if isinstance(node, (Distinct, Aggregate, Join, Sort, Window)) else None


def spill_collect(
    plan: LogicalPlan, sources: list[Source], num_partitions: int = 16
) -> pa.Table | None:
    """Run `plan` out-of-core if its top operator supports spilling, else `None`.

    Dispatches a top-level Aggregate / Join / (range-partitionable) Sort / window to
    the matching partition-and-spill executor. Returns `None` when the plan shape has
    no spilling path, so the caller can fall back to the in-memory engine. Shared
    by the explicit `collect(spill=True)` request and Carbonite's automatic
    spill decision, so both route through one place.
    """
    if isinstance(plan, Aggregate):
        # If the aggregate reads a spillable *breaker* (the optimizer lowers
        # `COUNT(DISTINCT x)` to `COUNT(*)` over a `DISTINCT`), spill that inner breaker
        # out-of-core first and aggregate its bounded result — far cheaper than the
        # value-list spill, and correct (the streaming map path would run the breaker
        # per-batch, which a `DISTINCT`/nested aggregate cannot be).
        if _peel_to_breaker(plan.input) is not None:
            inner = spill_collect(plan.input, sources, num_partitions)
            if inner is None:
                return None
            from batcher.dist.executors.partition_io import _apply_above

            return _apply_above([plan], inner)
        # INTERSECT/EXCEPT lower to `Aggregate(bool_or) over Union(left, right)` — an aggregate
        # whose input spans TWO sources, which the one-shot spilling aggregate cannot relabel
        # (`_relabel_single_source` asserts a single source). Decline so the caller runs it in
        # memory (the same mergeable oracle), exactly as the Join path declines a multi-source
        # join via `supports_spilling_join`.
        if not _single_source(plan.input):
            return None
        return execute_spilling_aggregate(plan, sources, num_partitions)
    # DISTINCT is a group-by over every column with no aggregates, so it rides the same
    # hash-partition-and-spill path — the fix for a high-cardinality `DISTINCT` (and the
    # `COUNT(DISTINCT)` the planner lowers to `DISTINCT → COUNT`) failing fast under a tight
    # memory envelope instead of completing out-of-core, which it must at PB scale.
    if isinstance(plan, Distinct):
        # A *keyed* dedup is not a group-by over every column — its surviving row carries
        # columns the key does not determine — so the equivalence below does not hold for it
        # and building it anyway silently returns a whole-row DISTINCT (every row of a table
        # with no duplicate rows). Decline: the engine's own grace path
        # (`bc_interp::distinct_on_spill`) reduces it out of core under the same envelope.
        if plan.keys:
            return None
        # Same multi-source guard: a `DISTINCT` over a `Union` (a set-op shape) can't ride the
        # single-source spill path — decline to the in-memory engine.
        if not _single_source(plan.input):
            return None
        cols = plan.input.available_columns()
        group_keys = tuple(Projection(alias=c, expr=col(c)) for c in cols)
        equiv = Aggregate(input=plan.input, group_keys=group_keys, aggregates=())
        # The lowering is what loses the column type: a whole-row `DISTINCT` keeps an
        # extension-typed column, but as a group-by its keys come back as plain storage.
        # `DISTINCT`'s own schema is the one to restore, not the group-by's.
        return execute_spilling_aggregate(
            equiv, sources, num_partitions, declared=empty_result_table(plan, cols).schema
        )
    # The ordering/binary breakers live in `spill_breakers` (imported lazily so this
    # module stays import-cycle-free: `spill_breakers` depends on this one's helpers).
    if isinstance(plan, (Join, Sort, Window)):
        from batcher.dist import spill_breakers as br

        if isinstance(plan, Join):
            # A join whose side spans several sources cannot be grace-partitioned (see
            # `supports_spilling_join`); decline so the caller runs it in memory rather
            # than asserting.
            if br.supports_spilling_join(plan):
                return br.execute_spilling_join(plan, sources, num_partitions)
            return None
        if isinstance(plan, Sort) and br.supports_spilling_sort(plan, sources):
            return br.execute_spilling_sort(plan, sources, num_partitions)
        if isinstance(plan, Window):
            # PARTITION BY window → grace-partition by those keys; a *global* window
            # (no PARTITION BY, single plain-column ORDER BY) → ordered-bucket offset.
            gen = None
            if br.supports_spilling_window(plan):
                gen = br.stream_spilling_window(plan, sources, num_partitions)
            else:
                from batcher.dist.global_window import (
                    stream_spilling_global_window,
                    supports_ordered_bucket_offsets,
                )

                if supports_ordered_bucket_offsets(plan):
                    gen = stream_spilling_global_window(plan, sources, num_partitions)
            if gen is not None:
                batches = list(gen)
                if batches:
                    return pa.Table.from_batches(batches)
                return empty_result_table(plan, plan.available_columns())
    # Peel the row-wise / limit operators sitting *above* a spillable breaker (e.g. the
    # output `Project` of a `COUNT(DISTINCT)`, whose raw plan is `Project → Aggregate`),
    # spill the breaker out-of-core, then re-apply the peeled ops to its bounded result.
    # Without this a `Project`/`Filter`/`Limit` on top made the whole plan look
    # non-spillable, so a large query would fail fast under a tight memory envelope
    # instead of completing out-of-core.
    above: list[LogicalPlan] = []
    node: LogicalPlan = plan
    while isinstance(node, (Project, Filter, Limit)):
        above.append(node)
        node = node.input
    if above:
        # A peeled `Limit` is order-sensitive, and only `Sort` gives the breaker below it a
        # defined output order. `Distinct`, `Aggregate`, `Join` and `Window` all emit in
        # hash/partition order out-of-core, while the in-memory path emits in input order —
        # so re-applying `LIMIT k` here keeps a *different k rows* than `collect()` does,
        # which is a wrong answer rather than a slower one. `bc_ir::RelOp::Distinct` states
        # the contract this breaks: "the rows kept are the first k in input order", chosen
        # precisely so one node and many agree. Decline instead, and let the in-memory path
        # answer; the shapes are pinned in `NON_SPILLING_SHAPES`
        # (tests/integration/test_spill_route_is_taken.py).
        if not isinstance(node, Sort) and any(isinstance(n, Limit) for n in above):
            return None
        inner = spill_collect(node, sources, num_partitions)
        if inner is None:
            return None
        from batcher.dist.executors.partition_io import _apply_above

        return _apply_above(above, inner)
    # A plan whose peeling reaches a bare `Scan` has no stateful operator, so there is no
    # state to partition and nothing here can help it. The caller falls back to the
    # in-memory path, which resolves the whole input — for a selective filter that is a
    # poor trade (measured: `scan -> filter` returning ONE row from 1.5M resolved the whole
    # input), but running the row-wise plan chunk-by-chunk over `_iter_spill_morsels`
    # instead was tried and is *worse*: 644 MB of growth against 532 MB on a 6M-row parquet
    # scan, because the per-chunk `execute_plan` dispatches and the morsel re-chunking cost
    # more than one pass does. The bound for this shape has to come from a streaming source
    # handoff at the FFI boundary, not from re-chunking on the Python side.
    return None


def execute_spilling_aggregate(
    agg: Aggregate,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
    declared: pa.Schema | None = None,
) -> pa.Table:
    """Aggregate `agg` out-of-core, spilling hash-partitioned partials to disk.

    `declared` is the output schema to restore onto the result. It defaults to the
    aggregate's own, and the whole-row `DISTINCT` lowering overrides it with `DISTINCT`'s,
    because the two disagree about the key columns' types and `DISTINCT`'s is the one a
    caller asked for.

    Restoring it matters because an Arrow **extension** type -- what every tensor column
    carries -- does not survive a group-key round trip: the key comes back as its plain
    storage, which then picks up the FFI boundary's narrow-type widening. Without this an
    explicit ``collect(spill=True)`` returns a different column *type* from the same query
    run in memory, which is the one thing a spilled result may not do.
    """
    nat = engine()
    cfg_json = active_config().engine_config_json()
    group_keys_json, aggregates_json = agg_spec_json(agg)
    n_keys = len(agg.group_keys)
    # A global aggregate (no keys) cannot shuffle by key → a single bucket.
    n_buckets = 1 if n_keys == 0 else _fd_safe(num_partitions)
    key_idx = list(range(n_keys))

    map_plan, source_id = _relabel_single_source(agg.input)
    map_ir = json.dumps(map_plan.to_ir())
    source = sources[source_id]

    with spill_scratch("batcher_spill_", spill_dir) as store:
        # --- partition phase: stream source, partial-aggregate, spill by key ---
        writers = BucketWriters(store, "bucket")
        for batch in _iter_spill_morsels(source, map_projection(agg, source_id)):
            mapped = nat.execute_plan(map_ir, [[batch]], cfg_json)
            if not mapped:
                continue
            partial = nat.partial_aggregate(group_keys_json, aggregates_json, mapped)
            # One bucket (global aggregate, or num_partitions=1) needs no shuffle.
            if n_buckets == 1:
                writers.write(0, partial)
            else:
                writers.add(nat.partition_batches([partial], key_idx, n_buckets))
        handles = writers.close()

        # --- reduce phase: combine+finalize one bucket at a time, recursing into
        # any bucket too large to fit (skew) ------------------------------------
        out: list[pa.RecordBatch] = []
        for b in range(n_buckets):
            handle = handles.get(b)
            if handle is None:
                continue  # bucket received no rows
            _reduce_agg_bucket(
                store, handle, group_keys_json, aggregates_json, nat, key_idx, n_keys, out, 0
            )

        if out:
            # Same reason the distributed reducer restores them: a group-key round trip
            # hands an extension-typed column back as its plain storage.
            table = pa.Table.from_batches(out)
            return restore_declared_types(table, declared or _empty_table(agg).schema)
        # Empty input. A *global* aggregate over zero rows still returns exactly one row
        # (`count() -> 0`, `median() -> NULL`), which is what both the single-node engine
        # and DuckDB do — so it cannot take the zero-row `_empty_table` path.
        #
        # `combine_finalize(..., [])` cannot serve it: with no partial state it has no
        # schema to type the result from, and raises. Route a schema-carrying *empty*
        # batch through the same map -> partial -> finalize pipeline the non-empty path
        # uses; the aggregate's identity element then falls out of the mergeable algebra
        # rather than being special-cased per function.
        if n_keys == 0:
            empty_in = pa.RecordBatch.from_pylist([], schema=source.schema())
            mapped = nat.execute_plan(map_ir, [[empty_in]], cfg_json)
            partial = nat.partial_aggregate(group_keys_json, aggregates_json, mapped)
            return pa.Table.from_batches(
                [nat.combine_finalize(group_keys_json, aggregates_json, [partial])]
            )
        return _empty_table(agg)


def _empty_table(agg: Aggregate) -> pa.Table:
    # Typed, not null-typed: an empty aggregate result must carry the same column types a
    # non-empty one would, or `distributed == single-node` is false for every empty result.
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return empty_result_table(agg, names)


# Named here because the skew test and this module's own reduce read them; the values, the
# reasoning behind the depth bound, and the salt are `dist.spill.buckets`' — every breaker
# graces the same way, and three of them used to say so separately.
_MAX_SPILL_RECURSION = GRACE_DEPTH
_SUB_BUCKETS = GRACE_SUB_BUCKETS
_split_salt = split_salt


def _reduce_agg_bucket(store, handle, gk, aj, nat, key_idx, n_keys, out, depth):
    """Reduce one spilled aggregate bucket, recursing into it if it is too large.

    A bucket within budget (or a keyless global aggregate, or at the recursion floor) is
    combined+finalized directly. An over-large bucket is re-partitioned by a secondary hash
    of the group key into `_SUB_BUCKETS` sub-buckets — streamed, so the whole bucket is
    never resident — and each sub-bucket is reduced recursively. Every group's partial rows
    hash together, so per-sub-bucket finalize is exact: skew degrades gracefully instead of
    OOMing the reduce.

    A *global* aggregate has one group and no key to re-hash, so it is never split however
    large it gets — the ceiling `GRACE_DEPTH` documents, arriving immediately.
    """
    if n_keys and over_envelope(handle, depth, max_depth=_MAX_SPILL_RECURSION):
        subs = regrace(
            nat,
            store,
            handle,
            key_idx,
            _split_salt(depth),
            f"{handle.path.rsplit('/', 1)[-1]}_d{depth}",
            n_sub=_SUB_BUCKETS,
        )
        for sb in range(_SUB_BUCKETS):
            h = subs.get(sb)
            if h is not None:
                _reduce_agg_bucket(store, h, gk, aj, nat, key_idx, n_keys, out, depth + 1)
        return

    partials = read_reserved_bucket(store, handle)
    if partials:
        out.append(nat.combine_finalize(gk, aj, partials))
    # This bucket is finished, and every group key hashes to exactly one bucket, so nothing
    # will read it again. Giving its disk back here bounds peak *scratch* to the buckets
    # still outstanding rather than to the whole spilled state — the disk analogue of the
    # credit window, and the difference between a PB-scale aggregate needing room for its
    # largest bucket and needing room for all of them at once.
    store.release(handle)
