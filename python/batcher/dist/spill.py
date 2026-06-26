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

from batcher.carbonite.spill import TieredSpillStore
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.io.source import Source
from batcher.plan.logical import Aggregate, Join, LogicalPlan, Sort, Window

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
        return execute_spilling_aggregate(plan, sources, num_partitions)
    # The ordering/binary breakers live in `spill_breakers` (imported lazily so this
    # module stays import-cycle-free: `spill_breakers` depends on this one's helpers).
    if isinstance(plan, (Join, Sort, Window)):
        from batcher.dist import spill_breakers as br

        if isinstance(plan, Join):
            return br.execute_spilling_join(plan, sources, num_partitions)
        if isinstance(plan, Sort) and br.supports_spilling_sort(plan):
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
                return pa.table({c: [] for c in plan.available_columns()})
    return None


def execute_spilling_aggregate(
    agg: Aggregate,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
) -> pa.Table:
    """Aggregate `agg` out-of-core, spilling hash-partitioned partials to disk."""
    import batcher._native as nat

    cfg_json = active_config().engine_config_json()
    group_keys_json = json.dumps(
        [{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys]
    )
    aggregates_json = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
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
        for batch in source.iter_batches():
            if batch.num_rows == 0:
                continue
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
        # Empty input: produce the correct empty/zero-row aggregate schema.
        if n_keys == 0:
            return pa.Table.from_batches(
                [nat.combine_finalize(group_keys_json, aggregates_json, [])]
            )
        return _empty_table(agg)
    finally:
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _empty_table(agg: Aggregate) -> pa.Table:
    names = [k.alias for k in agg.group_keys] + [s.alias for s in agg.aggregates]
    return pa.table({n: [] for n in names})


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
    if n_keys == 0 or handle.nbytes <= bucket_max or depth >= _MAX_SPILL_RECURSION:
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
