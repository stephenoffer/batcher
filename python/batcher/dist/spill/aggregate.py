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
ordering/binary breakers that share it live in `dist.spill_breakers` (sort/join/window),
and `staging` decides which of an operator's inputs must be spilled before it runs.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source, _single_source
from batcher.dist.executors.plan_analysis import (
    _empty_agg_table,
    empty_result_table,
    restore_declared_types,
)
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
from batcher.dist.spill.staging import peel_to_breaker, stage_breaker_inputs
from batcher.io.source import Source
from batcher.plan.expr_ir import col
from batcher.plan.ir_specs import agg_spec_json
from batcher.plan.types import logical_bytes
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    Sort,
    Window,
    hoist_sort_key,
    hoist_window_keys,
)

__all__ = [
    "execute_spilling_aggregate",
    "narrow_to_stage",
    "spill_collect",
]


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
        if peel_to_breaker(plan.input) is not None:
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
    if isinstance(plan, (AsofJoin, Join, Sort, Window)):
        from batcher.dist import spill_breakers as br

        # An ASOF join with `by` keys grace-partitions on exactly the same argument an
        # equi-join does -- rows that can match share a `by` group, so hashing on `by` puts
        # them in one bucket -- so it rides the same path rather than a second one.
        # `supports_spilling_join` refuses the keyless form, which needs the range
        # decomposition instead.
        # A breaker beneath *either side* is spilled first and spliced back as a staged scan
        # -- see `_stage_breaker_inputs`. `supports_spilling_join` does not catch this: it
        # asks whether each side names a single source, and `Join(Aggregate(Scan(0)),
        # Scan(1))` answers yes. Measured on a 30 MiB fixture with 7 groups, the
        # grace-partitioned join returned **28 rows for 7** -- one per (chunk, group).
        if isinstance(plan, (AsofJoin, Join)):
            staged = stage_breaker_inputs(plan, sources, num_partitions)
            if staged is not None:
                return spill_collect(*staged, num_partitions)
            # A join whose side spans several sources cannot be grace-partitioned (see
            # `supports_spilling_join`); decline so the caller runs it in memory rather
            # than asserting.
            if br.supports_spilling_join(plan):
                return br.execute_spilling_join(plan, sources, num_partitions)
            return None
        # A breaker *underneath* the ordering breaker is spilled out-of-core first, and the
        # sort/window then applies to its bounded result -- the same move the `Aggregate`
        # branch above makes, for the same reason, and it was missing here.
        #
        # This is a wrong-answer bug, not a memory one. `stage_and_partition` runs the
        # operator's input sub-plan **once per morsel** (`execute_plan(map_ir, [[batch]])`),
        # which is exactly right for the linear scan/filter/project chain it was written for
        # and silently wrong for a breaker: each morsel yields a *partial* aggregate, the
        # partials are range-partitioned and sorted, and nothing ever combines them. Nothing
        # in the predicate stack caught it — `supports_spilling_sort` asks whether the key
        # can be range-partitioned and whether the input names one source, and
        # `Sort(Aggregate(Filter(Scan)))` answers yes to both.
        #
        # TPC-H q1 is that shape, and it is the shape most of TPC-H ends in (`GROUP BY ...
        # ORDER BY ...`). At sf10 under a 2 GiB envelope it returned **1,224 rows where the
        # answer is 4** — one row per surviving per-morsel partial — with no error, and the
        # same query uncapped returned 4. A query that gets a different answer for being
        # short of memory is the worst failure this path can have: the envelope is exactly
        # what changes between a laptop and a cluster node.
        #
        # A join beneath a sort was already safe, by way of `_single_source`: a join spans
        # two sources, so `supports_spilling_sort` declines it. That is a true guard but an
        # incidental one, and it does not cover the unary breakers.
        if isinstance(plan, (Sort, Window)):
            staged = stage_breaker_inputs(plan, sources, num_partitions)
            if staged is not None:
                return spill_collect(*staged, num_partitions)
        # A *computed* shuffle key — `sort(col("a") + col("b"))`,
        # `partition_by=[col("v") % 4]` — is materialized as a hidden column first, exactly
        # as the distributed dispatcher does it and through the same
        # `plan.logical.hoist_sort_key` / `hoist_window_keys`. Both cut the operator into
        # pieces on the same partitioner, which reads a key's values from a *column*, so
        # without this the identical query distributed and then declined to spill: it fell
        # back to the in-memory kernel under the very memory envelope the spill exists for.
        # `keep` is the operator's original output, so the hidden column is dropped from the
        # result — in pyarrow, since the rows are already materialized here.
        #
        # Applied to a *copy*, and only where the spilling path then accepts it: falling
        # through with the rewrite still in place would carry the hidden column into the
        # generic peeling path below and return it to the caller as a real output column.
        keep: tuple[str, ...] | None = None
        if isinstance(plan, Sort) and not br.supports_spilling_sort(plan, sources):
            hoisted = hoist_sort_key(plan)
            if hoisted is not None and br.supports_spilling_sort(hoisted[0], sources):
                plan, keep = hoisted
        elif isinstance(plan, Window) and not br.supports_spilling_window(plan):
            hoisted = hoist_window_keys(plan)
            if hoisted is not None and br.supports_spilling_window(hoisted[0]):
                plan, keep = hoisted
        if isinstance(plan, Sort) and br.supports_spilling_sort(plan, sources):
            return _dropping(br.execute_spilling_sort(plan, sources, num_partitions), keep)
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
                    return _dropping(pa.Table.from_batches(batches), keep)
                return _dropping(empty_result_table(plan, plan.available_columns()), keep)
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
        inner = spill_collect(narrow_to_stage(above, node), sources, num_partitions)
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


def narrow_to_stage(above: list[LogicalPlan], node: LogicalPlan) -> LogicalPlan:
    """`node` with a `Project` inserted below it selecting only what this stage needs.

    A pass-through breaker narrows nothing by itself: a sort emits every column it is given
    and a window those plus its aliases, so every out-of-core path below reads the whole
    table. What narrows the read is the projection the caller just peeled into `above` — and
    the out-of-core path is exactly where that costs the most, because every unread column is
    also decoded, hash-partitioned, compressed, written to a bucket file and read back.
    `map_projection`'s own docstring makes the point for the source read; this extends it to
    the columns the stage as a whole can prove it never needs.

    Rewriting the *plan* rather than threading a projection through six signatures is what
    keeps this correct by construction: every path below — the source read, the bucket write,
    the reducer, the empty-result schema — derives from the plan it is handed, so none of
    them can disagree about which columns exist.

    The distributed dispatcher answers the same question with
    `dist.executors.partition_io.stage_pushdown`, and the two are deliberately not one
    function: that one produces a projection for the *source read* and may therefore speak in
    source columns alone, while this one puts a `Project` under the breaker and so must also
    keep every column the plan derived. See the note on `stage_pushdown`.

    `required_columns_per_source` is asked of `above` rebuilt over `node`, so a `Filter` that
    was peeled upward keeps the columns it tests even though the breaker does not use them.
    Returns `node` unchanged whenever the analysis cannot narrow (an opaque node, a source
    that needs everything, a plan whose columns cannot be resolved), so this is only ever an
    optimization.
    """
    import dataclasses

    from batcher.dist.executors.plan_analysis import _single_source, scanned_source_ids
    from batcher.plan.logical import project_columns

    if not _single_source(node):
        return node
    try:
        from batcher.kyber.rules.projections import required_columns_per_source

        stage: LogicalPlan = node
        for outer in reversed(above):
            stage = dataclasses.replace(outer, input=stage)
        source_id = next(iter(scanned_source_ids(node)))
        needed = required_columns_per_source(stage).get(source_id)
        available = node.input.available_columns()
    except Exception:  # pragma: no cover - an opaque node the analysis cannot walk
        return node
    if not needed:
        return node
    # `needed` names **source** columns, but `available` may hold columns the plan *derived*
    # below the breaker — a `with_columns` the breaker then keys on, or a hoisted shuffle key.
    # Dropping one of those is not a missed optimization, it is a broken plan: the breaker's
    # own `__post_init__` re-validates its keys against the narrowed input and raises
    # `ColumnNotFoundError`. So a column is dropped only when it is a source column the stage
    # proved it does not need; anything the plan computed is kept regardless. That still
    # removes the whole unread tail of a wide table, which is the entire win.
    source_columns = _scan_columns(node)
    if source_columns is None:
        return node
    keep = [c for c in available if c in set(needed) or c not in source_columns]
    if not keep or len(keep) == len(available):
        return node
    return dataclasses.replace(node, input=project_columns(node.input, keep))


def _scan_columns(node: LogicalPlan) -> set[str] | None:
    """The column names the single scan beneath `node` reads, or `None` if there isn't one.

    Needed to tell a column the *source* supplies from one the plan *computed*, which is the
    distinction `required_columns_per_source` cannot make on its own: it answers about the
    source, and its answer is only safe to subtract from a set that contains nothing else.
    """
    from batcher.plan.logical import Scan

    seen: LogicalPlan | None = node
    while seen is not None and not isinstance(seen, Scan):
        seen = getattr(seen, "input", None)
    if not isinstance(seen, Scan) or seen.schema is None:
        return None
    return set(seen.schema.arrow.names)


def _dropping(table: pa.Table, keep: tuple[str, ...] | None) -> pa.Table:
    """`table` cut back to `keep`, or unchanged when no key was hoisted.

    The hidden shuffle key a hoist materializes is real data in the spilled result, so the
    caller's relation is the result minus that column. Selecting by name is exact and costs
    nothing: the columns are already in memory and Arrow selection is a buffer reference.
    """
    return table if keep is None else table.select(list(keep))


#: Partial-aggregate state the out-of-core aggregate may hold before it starts bucketing.
#:
#: `memory.spill_bucket_max_bytes` is already the size a *bucket* may reach before the reduce
#: re-partitions it, so it is the figure this path has been tuned against; holding one
#: bucket's worth in memory instead of writing it is the same bound, spent on the other side
#: of the disk. It is a ceiling on the whole held state, not on any one partial.
#:
#: The configured value is used as-is, with **no floor under it**. A floor would silently
#: hold more than a caller who tightened the bound asked for, and the caller that tightens it
#: is the one that means it: `_tight()` in `tests/integration/test_carbonite_skew_out_of_core.py`
#: sets 4096 bytes precisely to force the bucketing path, and a 1 MiB floor held that whole
#: fixture in memory instead — the nine skew tests stopped reaching the code they exist to
#: cover while still reporting the right answer, which is the one failure shape a correct
#: answer cannot reveal. `config.validation` already rejects a non-positive value, so there is
#: nothing left for a floor to defend against.
def _held_partial_budget() -> int:
    """Bytes of partial state the partition phase may keep in memory before it spills."""
    return int(active_config().memory.spill_bucket_max_bytes)


def _spill_partial(writers, nat, partial, key_idx, n_buckets: int) -> None:
    """Write one partial to its bucket(s) — the one place the shuffle-or-not test lives."""
    if n_buckets == 1:
        writers.write(0, partial)  # global aggregate, or a single bucket: no shuffle
    else:
        writers.add(nat.partition_batches([partial], key_idx, n_buckets))


def _finalize_held(nat, gk: str, aj: str, held, agg, declared):
    """Finalize partials that never left memory, restoring the declared output types."""
    if not held:
        return None  # caller falls through to its empty-input handling
    table = pa.Table.from_batches([nat.combine_finalize(gk, aj, held)])
    return restore_declared_types(table, declared or _empty_agg_table(agg).schema)


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
        # A *reducing* aggregate's whole partial state fits in memory however large its
        # input is, and then none of the machinery below is needed: no hash-partition per
        # chunk, no bucket files, no reduce pass. Hold the partials until they prove
        # otherwise, and only start bucketing once they actually exceed the budget.
        #
        # This is a measurement, not an estimate, and that is the point. The estimate is
        # routinely wrong in exactly the direction that hurts: `GROUP BY l_returnflag,
        # l_linestatus` over TPC-H `lineitem` has **four** groups, but with no column
        # statistics Kyber reads the group count as at least a morsel's worth, so
        # `kyber.annotate._aggregate_resident_bytes` sizes the operator's envelope at the
        # whole input — 17.6 GB at sf100 — which both routes the query here and asks for 132
        # buckets. Measured on that query, sf100 on local NVMe: **40.4 s -> 28.8 s (1.40x)**,
        # with the per-chunk `partition_batches` going from 9.8 s to zero and the reduce pass
        # from 5.6 s to 1.8 s. A high-cardinality aggregate exceeds the cap within a few
        # chunks and pays one extra byte count per chunk before it does.
        held: list[pa.RecordBatch] = []
        held_bytes = 0
        bucketing = False
        for batch in _iter_spill_morsels(source, map_projection(agg, source_id)):
            mapped = nat.execute_plan(map_ir, [[batch]], cfg_json)
            if not mapped:
                continue
            partial = nat.partial_aggregate(group_keys_json, aggregates_json, mapped)
            if not bucketing:
                held.append(partial)
                held_bytes += logical_bytes(partial)
                if held_bytes <= _held_partial_budget():
                    continue
                # The state outgrew memory: flush what is held through the same
                # partitioning every later chunk takes, and carry on as before. Ordering is
                # irrelevant — `combine` is associative and commutative, so a group's rows
                # meet in their bucket whichever side of the switch they arrived on.
                bucketing = True
                for spilled_partial in held:
                    _spill_partial(writers, nat, spilled_partial, key_idx, n_buckets)
                held = []
                held_bytes = 0
                continue
            _spill_partial(writers, nat, partial, key_idx, n_buckets)
        if bucketing:
            handles = writers.close()
        else:
            # Nothing was ever written, so there is nothing to read back. An *empty* input
            # held nothing either, and that case still owes a global aggregate its one
            # identity row — so it falls through to the same empty-input handling below
            # rather than being answered here.
            finalized = _finalize_held(nat, group_keys_json, aggregates_json, held, agg, declared)
            if finalized is not None:
                return finalized
            handles = {}

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
            return restore_declared_types(table, declared or _empty_agg_table(agg).schema)
        # Empty input. A *global* aggregate over zero rows still returns exactly one row
        # (`count() -> 0`, `median() -> NULL`), which is what both the single-node engine
        # and DuckDB do — so it cannot take the zero-row `_empty_agg_table` path.
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
        return _empty_agg_table(agg)


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
