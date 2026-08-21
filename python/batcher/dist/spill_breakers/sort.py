"""Out-of-core sort: range-partition into ordered buckets, sort each, yield in key order.

Globally sorted with no k-way merge, bounded by one bucket. A top-N `limit` stops once
`limit` rows have been emitted.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.executors.partition_io import range_partitionable, sample_key_grid
from batcher.dist.executors.plan_analysis import _single_source
from batcher.dist.spill import (
    _fd_safe,
    _iter_spill_morsels,
    map_projection,
)
from batcher.dist.spill.buckets import (
    GRACE_DEPTH,
    GRACE_SUB_BUCKETS,
    BucketWriters,
    bucket_envelope,
    over_envelope,
    read_reserved_bucket,
    resident_bytes,
    spill_scratch,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.ir_specs import sort_keys_ir, task_scan_ir
from batcher.plan.logical import Sort


def supports_spilling_sort(sort: Sort, sources: list[Source] | None = None) -> bool:
    """Whether this sort can run out-of-core via the ordered range partition.

    The leading key must be a plain column (the range partition splits on it — equal
    values share a bucket — and each bucket is then sorted by the full key list) *and*
    it must be a type the shared sampler and range partitioner both handle: a numeric
    key sampled from the KLL sketch, or a string one sampled lexically (`sample_key_grid`
    picks the order and `bucketize` routes on the matching one). Returning `False` is a
    graceful fallback — the caller runs the ordinary
    (non-spilling) sort — so a key we cannot range-partition costs memory, never
    correctness. Without `sources` the type cannot be checked, so only the shape is.

    The input must also name a *single* source (the key's type is read from its schema). A
    sort over a multi-source input — `ORDER BY` above a join — used to reach
    `_relabel_single_source` anyway and die on its assertion; a predicate that answers
    *whether* a path applies must never raise when the answer is "no"."""
    if len(sort.keys) < 1 or not isinstance(sort.keys[0].expr, Col):
        return False
    if sources is None:
        return True
    if not _single_source(sort.input):
        return False
    _, sid = _relabel_single_source(sort.input)
    schema = sources[sid].schema()
    idx = schema.get_field_index(sort.keys[0].expr.name)
    # A derived key (not in the source schema) has an unknown type — stay out of the
    # range partition rather than fail inside it.
    if idx < 0:
        return False
    return range_partitionable(schema.field(idx).type)


def execute_spilling_sort(
    sort: Sort,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
) -> pa.Table:
    """Sort out-of-core, returning the full materialized table.

    Thin consumer of `stream_spilling_sort` — the bounded-memory bucket pipeline is
    one implementation; this collects it, `iter_batches()` streams it."""
    batches = list(stream_spilling_sort(sort, sources, num_partitions, spill_dir))
    if batches:
        return pa.Table.from_batches(batches)
    # Typed, not null-typed, and the *sort's* schema rather than the source's. Building the
    # empty from `{f.name: [] for f in source.schema()}` gave every column a null type and
    # the pre-map column set, so an empty out-of-core sort disagreed with the in-memory one
    # on both the types and the names — the same trap `execute_spilling_aggregate` avoids
    # with `empty_result_table`, and the reason "spilled == in-memory" has to hold for empty
    # results too.
    from batcher.dist.executors.plan_analysis import empty_result_table

    return empty_result_table(sort, sort.available_columns())


def _buckets_for_staged(stage_handle: object, hint: int) -> int:
    """How many ordered buckets the staged input should be split into.

    The bucket count used to be a constant (16, or whatever the caller passed), and that is
    the one number in an out-of-core sort that must not be a constant: each bucket is read
    back **whole** to be sorted, so a fixed count makes peak memory `input / 16` — it grows
    linearly with the input. A sort large enough to need this path is exactly the sort that
    then OOMs on its first bucket, and the failure looks like an ordinary OOM rather than a
    misconfigured spill.

    Staging has already measured the mapped input, so the count can come from the bucket
    envelope (`bucket_envelope` — the shared ceiling every breaker budgets against, capped
    by the memory budget), sized by `resident_bytes` — the same uncompressed measure.

    The count bounds the *average* bucket, not the largest: range boundaries come from a
    sampled grid, so a bucket can land a few percent over. Reading one back is still what
    the envelope has to cover, so a caller that must not exceed it needs the per-bucket
    re-split the grace paths use (`over_envelope`/`regrace`), which the ordered paths do
    not yet have — see `dist.global_window.stream`.

    The caller's `hint` is a floor, so a small sort still gets the parallelism it had, and
    `_fd_safe` is the ceiling, because the partition phase holds one writer per non-empty
    bucket open at once.

    A range partition cannot split a *single* key value across buckets — equal keys must
    share one, or the concatenation is not sorted — so a sort whose key is one hot value is
    still bounded by that value's rows. That is inherent to ordering, not to this sizing.
    """
    if stage_handle is None:
        return max(1, hint)
    resident = resident_bytes(stage_handle)
    bucket_max = bucket_envelope()
    if bucket_max <= 0 or resident <= 0:
        return _fd_safe(max(1, hint))
    return _fd_safe(max(hint, -(-resident // bucket_max)))


# The ordered counterpart of `dist.spill.buckets.regrace`: same depth and width bounds, and
# the same reason for having them, but it re-splits by **range on the order key** rather than
# by a hash salt. A salted split scatters the key range across sub-buckets, which is fine for
# a group-by (every row of a key still lands together) and useless here, where the whole
# contract is that bucket `i` precedes bucket `i+1`.
_MAX_ORDERED_SPLIT_DEPTH = GRACE_DEPTH
_ORDERED_SUB_BUCKETS = GRACE_SUB_BUCKETS


def iter_ordered_buckets(store, handle, key_name, nulls_first, descending, depth=0):
    """Yield the contents of `handle` in key order, re-splitting it first if it is too big.

    `stage_and_partition` sizes the bucket count from the *total* staged bytes, so it bounds
    the average bucket, not the largest: the boundaries come from a sampled grid, so buckets
    land uneven. Measured on 60k fat rows with a distinct key under a 1 MiB envelope, **33 of
    65 buckets exceeded it** -- and a bucket is read back *whole*, so the envelope is the one
    number that bounds this path's memory. Every other breaker re-splits an over-envelope
    bucket before reading it (`over_envelope`/`regrace`); the ordered paths did not, so the
    out-of-core sort and global window quietly used several times the budget they were given.

    Correctness rests on `bucketize`: sub-bucket `i` holds a contiguous key sub-range of the
    parent and equal keys never span a boundary, so walking the sub-buckets in key order is
    indistinguishable from having partitioned that finely to begin with. The recursion floor
    is the genuine ceiling `GRACE_DEPTH` documents -- a bucket that is one hot key cannot be
    split by any partitioning, because equal keys must be ordered together.

    Args:
        store: The scratch store holding `handle`.
        handle: The bucket, released once consumed.
        key_name: The order key the parent was routed by -- the same one a re-split uses.
        nulls_first: Which end nulls belong at, so a sub-split agrees with the parent.
        descending: Whether the order is descending, which decides sub-bucket visiting order.
        depth: Current recursion depth, `0` at the top.

    Yields:
        Each piece of `handle` as a list of record batches, in key order.
    """
    if over_envelope(handle, depth, max_depth=_MAX_ORDERED_SPLIT_DEPTH):
        subs = _range_regrace(store, handle, key_name, nulls_first, descending, depth)
        order = range(len(subs) - 1, -1, -1) if descending else range(len(subs))
        for i in order:
            if subs[i] is not None:
                yield from iter_ordered_buckets(
                    store, subs[i], key_name, nulls_first, descending, depth + 1
                )
        return
    bucket = read_reserved_bucket(store, handle)
    try:
        if bucket:
            yield bucket
    finally:
        store.release(handle)


def _range_regrace(store, handle, key_name, nulls_first, descending, depth):
    """Re-partition one over-large ordered bucket into finer *ordered* sub-buckets.

    Two streamed passes over the parent -- sample its own key grid, then route on the merged
    boundaries -- so the bucket that did not fit is never held whole, which is the point. The
    parent is released once re-partitioned, as `regrace` does, because it is the largest file
    in the recursion.

    Args:
        store: The scratch store holding `handle` and receiving the sub-buckets.
        handle: The bucket to split.
        key_name: The order key to split on.
        nulls_first: Which end nulls belong at.
        descending: Whether the order is descending.
        depth: The current depth, used to name the sub-bucket files uniquely.

    Returns:
        The sub-bucket handles, ascending by key (`None` where one received no rows), or the
        parent unchanged when its keys admit no boundary (a single hot value).
    """
    from batcher.dist.executors.partition_io import SAMPLE_PROBS, bucketize, merge_boundaries

    grids: list[tuple[list, int]] = []
    for rb in store.read_stream(handle):
        grid = sample_key_grid([rb], key_name, list(SAMPLE_PROBS))
        if grid:
            grids.append((grid, rb.num_rows))
    boundaries = merge_boundaries(grids, _ORDERED_SUB_BUCKETS)
    if not boundaries:
        return [handle]  # one hot key: no boundary exists, so process it as it stands
    writers = BucketWriters(store, f"ord_d{depth}_{id(handle):x}")
    for rb in store.read_stream(handle):
        writers.add(
            bucketize([rb], key_name, boundaries, _ORDERED_SUB_BUCKETS, nulls_first, descending)
        )
    subs = writers.close_dense(_ORDERED_SUB_BUCKETS)
    store.release(handle)
    return subs


def stage_and_partition(
    source, map_ir, key_name, nulls_first, descending, n_buckets, store, cfg_json, projection=None
):
    """Map `source` through `map_ir`, sample the key, and range-partition the mapped output
    into `n_buckets` ordered disk buckets (key-ascending; `None` where a bucket got no rows).

    Shared by the streaming sort and the streaming global window: both need equal keys to land
    in one bucket, which is what makes per-bucket processing + key-order concatenation globally
    correct. Sampling (`sample_key_grid`) and bucketing (`bucketize`) are the *same* primitives
    the distributed range-sort uses, so the two sorts cannot drift apart, and the per-row work
    stays in Rust. `descending` is load-bearing: the caller emits buckets high→low for a
    descending sort, so which end the nulls belong at depends on it, not on `nulls_first` alone.
    """
    from batcher.dist.executors.partition_io import (
        SAMPLE_PROBS,
        bucketize,
        merge_boundaries,
        sample_key_grid,
    )

    nat = engine()

    # --- pass 1: map the source ONCE, stage the mapped output to disk (so pass 2
    # re-reads locally, not re-mapping a possibly-remote source), and sketch the key.
    grids: list[tuple[list[float], int]] = []
    stage = store.writer("stage")
    for batch in _iter_spill_morsels(source, projection):
        for rb in nat.execute_plan(map_ir, [[batch]], cfg_json):
            if not rb.num_rows:
                continue
            stage.write(rb)
            grid = sample_key_grid([rb], key_name, list(SAMPLE_PROBS))
            if grid:
                grids.append((grid, rb.num_rows))
    stage_handle = stage.close()
    # Staging has measured the whole mapped input, so the bucket count no longer has to be
    # guessed — size it from what was actually staged. See `_buckets_for_staged`.
    n_buckets = _buckets_for_staged(stage_handle, n_buckets)
    boundaries = merge_boundaries(grids, n_buckets) if n_buckets > 1 else []

    # --- pass 2: assign staged rows to ordered buckets and spill ----------
    writers = BucketWriters(store, "bucket")
    staged = store.read_stream(stage_handle) if stage_handle is not None else iter(())
    for rb in staged:
        writers.add(bucketize([rb], key_name, boundaries, n_buckets, nulls_first, descending))
    handles = writers.close_dense(n_buckets)
    # The staged copy is the *whole* mapped input and pass 2 has finished re-reading it, so
    # holding it while the buckets are processed doubled peak scratch for no reader. Giving
    # it back here is what makes out-of-core sort cost one copy of the data on disk, not two.
    if stage_handle is not None:
        store.release(stage_handle)
    return handles


def stream_spilling_sort(
    sort: Sort,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
):
    """Sort out-of-core by *range*-partitioning, yielding the globally-sorted result
    one ordered bucket at a time — bounded by a single bucket, never the whole result.

    Range-partition into ordered buckets (`stage_and_partition`), then sort and yield
    each bucket in key order (no k-way merge). Single plain-column key only. A top-N
    `limit` stops once `limit` rows have been emitted."""
    nat = engine()

    cfg_json = active_config().engine_config_json()
    # Leading key drives the range partition; each bucket is sorted by the FULL key list.
    key = sort.keys[0]
    desc, nulls_first = key.descending, key.nulls_first
    n_buckets = _fd_safe(num_partitions)

    map_plan, sid = _relabel_single_source(sort.input)
    map_ir = json.dumps(map_plan.to_ir())
    keys_ir = sort_keys_ir(sort.keys)
    scan = task_scan_ir()
    sort_ir = json.dumps({"op": "sort", "input": scan, "keys": keys_ir, "limit": sort.limit})

    with spill_scratch("batcher_sort_spill_", spill_dir) as store:
        handles = stage_and_partition(
            sources[sid],
            map_ir,
            key.expr.name,
            nulls_first,
            desc,
            n_buckets,
            store,
            cfg_json,
            map_projection(sort, sid),
        )
        # Sort each bucket, yield in key order (reversed for descending). The count comes
        # from `handles`, not from `n_buckets`: staging measures the input and sizes the
        # split from it, so the number of buckets is decided in there, not here.
        order = range(len(handles) - 1, -1, -1) if desc else range(len(handles))
        emitted = 0
        for b in order:
            if handles[b] is None:
                continue
            for bucket in iter_ordered_buckets(store, handles[b], key.expr.name, nulls_first, desc):
                for rb in nat.execute_plan(sort_ir, [bucket], cfg_json):
                    if not rb.num_rows:
                        continue
                    if sort.limit is not None:
                        take = min(rb.num_rows, sort.limit - emitted)
                        if take <= 0:
                            return
                        yield rb.slice(0, take)
                        emitted += take
                        if emitted >= sort.limit:
                            return
                    else:
                        yield rb
