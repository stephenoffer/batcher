"""Single-node out-of-core aggregation via partition-and-spill, plus the spill
dispatcher.

This is the same radix-partition machinery the distributed shuffle uses, run
locally and sequentially against disk — realizing the plan's thesis that
single-node out-of-core and PB-scale distribution are *one* mechanism with disk
vs. network as the sink.

Pipeline (memory bounded by a single source batch + one bucket's partial state):

    for each source batch (streamed):          # bounded input memory
        mapped   = run the map sub-plan on the batch
        partial  = partial_aggregate(mapped)    # pre-aggregate to shrink data
        buckets  = hash-partition partial by group key into K on-disk files
    for each bucket (one at a time):            # bounded reduce memory
        result  += combine_finalize(read(bucket))

Because each group key hashes to exactly one bucket, combining per bucket yields
the correct global result — identical to the in-memory aggregation, but a
group-by over more distinct groups than fit in RAM still completes. The shared
helpers here (`_work_dir`, `_make_store`, `_fd_safe`) and `spill_collect` back the
ordering/binary breakers too — those live in `spill_breakers` (sort/join/window).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pyarrow as pa

from batcher._internal.native import engine
from batcher.carbonite.spill import TieredSpillStore
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source, _single_source
from batcher.dist.executors.plan_analysis import empty_result_table
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


def _work_dir(spill_dir: str | None, prefix: str) -> tuple[str, bool]:
    """Resolve the local scratch dir for a spill, and whether we own it (rmtree it).

    An explicit `spill_dir` is caller-owned (not removed). Otherwise, if the config
    sets `MemoryConfig.spill_dir`, create a unique per-query subdir *under* that root
    (so striping onto fast/large disks is honored and rmtree only ever removes our
    own subdir — never a shared root). With neither, fall back to a system tempdir.
    """
    if spill_dir is not None:
        return spill_dir, False
    root = active_config().memory.spill_dir
    if root:
        os.makedirs(root, exist_ok=True)
        return tempfile.mkdtemp(prefix=prefix, dir=root), True
    return tempfile.mkdtemp(prefix=prefix), True


def _make_store(work_dir: str) -> TieredSpillStore:
    """A tiered spill store for `work_dir`, configured from the active `Config`.

    Local NVMe by default; overflows to `MemoryConfig.spill_remote_uri` once the
    local budget is exhausted, so an out-of-core query survives a full local disk.
    Spilled batches are compressed with the configured codec.
    """
    mem = active_config().memory
    return TieredSpillStore(
        work_dir,
        remote_uri=mem.spill_remote_uri,
        local_budget_bytes=mem.spill_local_budget_bytes,
        compression=mem.spill_compression,
    )


# Byte target the out-of-core partition phase feeds the engine at once. A source's
# batch size is not the engine's to trust: it can be far too large (`from_arrow` of a
# whole table, a fat parquet row group) or far too small (a streaming reader, a
# per-file scan, an exploded/filtered upstream emitting thousands of tiny batches).
# Both hurt.
#
#   * Too large: the parallel partial-aggregate builds per-thread hash tables over the
#     entire batch's cardinality, so peak memory scales with the batch, not the morsel
#     — a high-cardinality group-by peaked ~2.6x higher on one 20M-row batch than on
#     the same rows normalized here.
#   * Too small: the partition phase makes one engine dispatch per batch, and a batch
#     far under a morsel-group can't fill the cores — 256-row batches ran ~30x slower
#     than 256K-row batches through the identical spill.
#
# Normalizing every source to ~this target (split the over-large, coalesce runs of the
# under-large) caps the partition phase's working set *and* keeps each chunk wide
# enough to fan across all cores — so out-of-core throughput no longer depends on how
# the source happened to chunk its output.
_SPILL_INPUT_CHUNK_BYTES = 8 << 20  # 8 MiB


def _iter_spill_morsels(source: Source, projection: list[str] | None = None):
    """Yield `source`'s batches normalized to ~``_SPILL_INPUT_CHUNK_BYTES``.

    Over-large batches are split into zero-copy `slice` views (bounded without a
    copy); runs of small batches are coalesced into one chunk so the partition phase
    always processes an efficiently-sized, all-cores-wide morsel-group regardless of
    the source's batching. This is the single input tap every out-of-core partition
    phase (aggregate/join/sort/window) reads through; coalescing/splitting only
    reshapes the row stream, so every spill result is byte-identical.
    """
    pending: list[pa.RecordBatch] = []
    pending_bytes = 0

    def _flush() -> pa.RecordBatch | None:
        nonlocal pending_bytes
        if not pending:
            return None
        # One buffered batch needs no copy; a run is compacted into a single 0-offset
        # batch so the engine sees one contiguous chunk, not a chain of tiny ones.
        out = (
            pending[0]
            if len(pending) == 1
            else pa.Table.from_batches(pending).combine_chunks().to_batches()[0]
        )
        pending.clear()
        pending_bytes = 0
        return out

    for batch in source.iter_batches(projection):
        n = batch.num_rows
        if n == 0:
            continue
        nbytes = batch.nbytes
        if nbytes >= _SPILL_INPUT_CHUNK_BYTES:
            # Emit any buffered small batches first (order-preserving), then split.
            buffered = _flush()
            if buffered is not None:
                yield buffered
            if n == 1:
                yield batch
            else:
                rows = max(1, (_SPILL_INPUT_CHUNK_BYTES * n) // nbytes)
                for off in range(0, n, rows):
                    yield batch.slice(off, min(rows, n - off))
        else:
            pending.append(batch)
            pending_bytes += nbytes
            if pending_bytes >= _SPILL_INPUT_CHUNK_BYTES:
                yield _flush()
    tail = _flush()
    if tail is not None:
        yield tail


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
        for batch in _iter_spill_morsels(source):
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
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _empty_table(agg: Aggregate) -> pa.Table:
    # Typed, not null-typed: an empty aggregate result must carry the same column types a
    # non-empty one would, or `distributed == single-node` is false for every empty result.
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return empty_result_table(agg, names)


# Max grace-recursion depth: a bucket that is still over budget after this many
# secondary re-partitions is finalized as-is (a single dominant *group* cannot be
# split by hashing the key, so deeper recursion would not shrink it).
_MAX_SPILL_RECURSION = 3
_SUB_BUCKETS = 8
# Cap on simultaneously-open spill files: the partition phase holds one writer per
# non-empty bucket open at once, so an unbounded `num_partitions` would exhaust the
# process file-descriptor limit at scale (N14). Capping keeps FDs bounded; a bigger
# data volume is then absorbed by grace recursion (N13) into larger-then-split
# buckets rather than more files.
_FD_SAFE_PARTITIONS = 1024


def _fd_safe(n_buckets: int) -> int:
    return max(1, min(n_buckets, _FD_SAFE_PARTITIONS))


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
        partials = store.read(handle)
        if partials:
            out.append(nat.combine_finalize(gk, aj, partials))
        return

    sub_writers: dict[int, object] = {}
    sub_handles: dict[int, object] = {}
    for batch in store.read_stream(handle):
        for sb, parts in enumerate(nat.partition_batches([batch], key_idx, _SUB_BUCKETS)):
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
    for sb in range(_SUB_BUCKETS):
        h = sub_handles.get(sb)
        if h is not None:
            _reduce_agg_bucket(store, h, gk, aj, nat, key_idx, n_keys, out, depth + 1)
