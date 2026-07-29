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
import shutil

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source, _single_source
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.spill.scratch import (
    _fd_safe,
    _iter_spill_morsels,
    _make_store,
    _work_dir,
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
        # Same multi-source guard: a `DISTINCT` over a `Union` (a set-op shape) can't ride the
        # single-source spill path — decline to the in-memory engine.
        if not _single_source(plan.input):
            return None
        cols = plan.input.available_columns()
        group_keys = tuple(Projection(alias=c, expr=col(c)) for c in cols)
        equiv = Aggregate(input=plan.input, group_keys=group_keys, aggregates=())
        return execute_spilling_aggregate(equiv, sources, num_partitions)
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
                from batcher.dist.window_stream import (
                    stream_spilling_global_window,
                    supports_streaming_global_window,
                )

                if supports_streaming_global_window(plan):
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
) -> pa.Table:
    """Aggregate `agg` out-of-core, spilling hash-partitioned partials to disk."""
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

    work_dir, owns_dir = _work_dir(spill_dir, "batcher_spill_")
    store = _make_store(work_dir)
    writers: dict[int, object] = {}
    handles: dict[int, object] = {}

    try:
        # --- partition phase: stream source, partial-aggregate, spill by key ---
        for batch in _iter_spill_morsels(source, map_projection(agg, source_id)):
            mapped = nat.execute_plan(map_ir, [[batch]], cfg_json)
            if not mapped:
                continue
            partial = nat.partial_aggregate(group_keys_json, aggregates_json, mapped)
            # One bucket (global aggregate, or num_partitions=1) needs no shuffle.
            if n_buckets == 1:
                buckets = [[partial]]
            else:
                buckets = nat.partition_batches([partial], key_idx, n_buckets)
            for b, part_batches in enumerate(buckets):
                for pb in part_batches:
                    if pb.num_rows == 0:
                        continue
                    w = writers.get(b)
                    if w is None:
                        w = store.writer(f"bucket_{b}")
                        writers[b] = w
                    w.write(pb)
        for b, w in writers.items():
            handles[b] = w.close()

        # --- reduce phase: combine+finalize one bucket at a time, recursing into
        # any bucket too large to fit (skew) ------------------------------------
        out: list[pa.RecordBatch] = []
        key_idx_groups = list(range(n_keys))
        for b in range(n_buckets):
            handle = handles.get(b)
            if handle is None:
                continue  # bucket received no rows
            _reduce_agg_bucket(
                store, handle, group_keys_json, aggregates_json, nat, key_idx_groups, n_keys, out, 0
            )

        if out:
            return pa.Table.from_batches(out)
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
    finally:
        # `cleanup` before the `rmtree`, and unconditionally. It aborts any writer still
        # open (a partition phase abandoned by an exception) and deletes both tiers' files
        # — the `rmtree` only ever reached the *local* one, so a failed query that had
        # overflowed left orphaned objects in the remote bucket, accumulating and billable,
        # with nothing recording that they existed.
        store.cleanup()
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _empty_table(agg: Aggregate) -> pa.Table:
    # Typed, not null-typed: an empty aggregate result must carry the same column types a
    # non-empty one would, or `distributed == single-node` is false for every empty result.
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return empty_result_table(agg, names)


# Max grace-recursion depth. Past it, re-partitioning has stopped helping: a bucket still
# over budget after this many secondary hashes is dominated by group state that *no* hash of
# the group key can separate — every row of a group hashes together by construction. Such a
# bucket is finalized in memory, and that is a genuine ceiling: a single group whose state
# exceeds RAM cannot be reduced out of core by partitioning, because partitioning is the
# only tool the mergeable algebra gives us here.
#
# Routing this case to `combine_finalize_spilling` was tried and reverted. It grace-partitions
# by the same group key, so it cannot split what this recursion already could not: measured
# over 86 buckets that reached the floor over budget, peak RSS was identical (716 MB either
# way) and it added a re-read and re-write of every one. Lifting this ceiling needs a
# per-group spillable state in `bc-runtime`, not another partitioning pass.
_MAX_SPILL_RECURSION = 3
_SUB_BUCKETS = 8


def _split_salt(depth: int) -> int:
    """The re-partition salt for recursion `depth`.

    Non-zero -- 0 is the unsalted, cluster-wide bucket assignment, which a *shuffle* must
    agree on and a *local* re-split must not reuse -- and distinct per level, so keys that
    collided at one level spread at the next instead of re-colliding identically.
    """
    return (0x9E3779B97F4A7C15 * (depth + 1)) % (1 << 64) | 1


def _reduce_agg_bucket(store, handle, gk, aj, nat, key_idx, n_keys, out, depth):
    """Reduce one spilled aggregate bucket, recursing into it if it is too large.

    A bucket within budget (or a keyless global aggregate, or at the recursion
    floor) is combined+finalized directly. An over-large bucket is re-partitioned by
    a secondary hash of the group key into `_SUB_BUCKETS` sub-buckets — streamed, so
    the whole bucket is never resident — and each sub-bucket is reduced recursively.
    Every group's partial rows hash together, so per-sub-bucket finalize is exact
    (N13: skew degrades gracefully instead of OOMing the reduce).
    """
    bucket_max = active_config().memory.spill_bucket_max_bytes
    # Budget against the bucket's *uncompressed* (in-memory) size — reading it back
    # decompresses into RAM. `handle.nbytes` is the on-disk compressed size, which for a
    # compressible bucket (many repeated group keys/values) can be far smaller than the
    # resident footprint, so using it would let an over-large bucket skip re-spill recursion
    # and OOM `combine_finalize`. `logical_nbytes` is the size `combine_finalize` actually pays.
    resident = handle.logical_nbytes or handle.nbytes
    if n_keys == 0 or resident <= bucket_max or depth >= _MAX_SPILL_RECURSION:
        # `read_reserved` accounts the resident footprint against the process-wide buffer
        # pool for the duration of the read. Reading a bucket back is the one step of
        # spilling that can undo it — the state went to disk because it did not fit — and
        # nothing was checking the envelope before pulling it in again, so a concurrent
        # query sizing its own state saw headroom this reduce was already using.
        with store.read_reserved(handle) as stream:
            partials = list(stream)
        if partials:
            out.append(nat.combine_finalize(gk, aj, partials))
        # This bucket is finished, and every group key hashes to exactly one bucket, so
        # nothing will read it again. Giving its disk back here bounds peak *scratch* to
        # the buckets still outstanding rather than to the whole spilled state — the disk
        # analogue of the credit window, and the difference between a PB-scale aggregate
        # needing room for its largest bucket and needing room for all of them at once.
        store.release(handle)
        return

    sub_writers: dict[int, object] = {}
    sub_handles: dict[int, object] = {}
    # A **salted** re-partition, and the salt is the whole point. The parent bucket count and
    # `_SUB_BUCKETS` are both powers of two, and bucket assignment reads the low hash bits at a
    # power-of-two count -- so an unsalted re-partition of a 16-way bucket into 8 sub-buckets
    # sends every row to `bucket & 7`. One sub-bucket, always, at every level: the recursion
    # rewrote and re-read the whole bucket three times and changed nothing, then combined the
    # over-large bucket anyway. (That is what the "peak RSS was identical either way"
    # measurement recorded above was actually measuring.) The salt varies by depth and never
    # by row, so equal keys still co-locate and each sub-bucket stays an exact partial reduce.
    salt = _split_salt(depth)
    for batch in store.read_stream(handle):
        for sb, parts in enumerate(
            nat.partition_batches_salted([batch], key_idx, _SUB_BUCKETS, salt)
        ):
            for pb in parts:
                if pb.num_rows == 0:
                    continue
                w = sub_writers.get(sb)
                if w is None:
                    w = store.writer(f"{handle.path.rsplit('/', 1)[-1]}_d{depth}_s{sb}")
                    sub_writers[sb] = w
                w.write(pb)
    for sb, w in sub_writers.items():
        sub_handles[sb] = w.close()
    # The parent has been fully re-partitioned into the sub-buckets, so its own file is
    # dead weight from here on — and it is the *largest* file in the recursion, since every
    # sub-bucket is a fraction of it. Holding it while the sub-buckets are reduced is what
    # makes grace recursion cost more disk at each level instead of the same.
    store.release(handle)
    for sb in range(_SUB_BUCKETS):
        h = sub_handles.get(sb)
        if h is not None:
            _reduce_agg_bucket(store, h, gk, aj, nat, key_idx, n_keys, out, depth + 1)
