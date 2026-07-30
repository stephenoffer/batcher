"""Out-of-core window: grace-partition by the PARTITION BY keys so each bucket holds
*complete* partitions, run the window kernel per bucket, and yield — bounded by one bucket.

A keyless (global) window has no splittable partition, so it has no spill path here.
"""

from __future__ import annotations

import json
import shutil

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.spill import (
    _fd_safe,
    _iter_spill_morsels,
    _make_store,
    _split_salt,
    _work_dir,
    map_projection,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Window


def supports_spilling_window(window: Window) -> bool:
    """A PARTITION BY window over plain-column keys can grace-partition by those keys.

    A keyless (global) window has a single partition that cannot be split, so it has
    no bounded-memory spill path (it stays in-memory / materialized)."""
    return bool(window.partition_keys) and all(isinstance(k, Col) for k in window.partition_keys)


def stream_spilling_window(
    window: Window,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
):
    """Window out-of-core by grace-partitioning on the PARTITION BY keys, yielding each
    bucket's windowed output as it is produced — bounded by one bucket, not the whole
    input. Equal partition keys hash to the same bucket, so each bucket holds *complete*
    partitions and the window kernel computes the same values per bucket that it would
    single-node; the union of the buckets equals single-node. Plain-column keys only."""
    nat = engine()

    cfg_json = active_config().engine_config_json()
    cols = window.input.available_columns()
    pk_indices = [cols.index(k.name) for k in window.partition_keys]
    map_plan, sid = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    win_ir = window.to_ir()
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_json = json.dumps(win_ir)
    n_buckets = _fd_safe(num_partitions)
    source = sources[sid]

    work_dir, owns_dir = _work_dir(spill_dir, "batcher_win_spill_")
    store = _make_store(work_dir)
    writers: dict[int, object] = {}
    handles: list[object] = [None] * n_buckets
    try:
        for batch in _iter_spill_morsels(source, map_projection(window, sid)):
            rows = nat.execute_plan(map_ir, [[batch]], cfg_json)
            if not rows:
                continue
            for b, parts in enumerate(nat.partition_batches(rows, pk_indices, n_buckets)):
                for pb in parts:
                    if pb.num_rows == 0:
                        continue
                    w = writers.get(b)
                    if w is None:
                        w = store.writer(f"win_bucket_{b}")
                        writers[b] = w
                    w.write(pb)
        for b, w in writers.items():
            handles[b] = w.close()

        for b in range(n_buckets):
            if handles[b] is None:
                continue
            yield from _window_bucket(nat, store, handles[b], win_json, cfg_json, pk_indices, 0)
    finally:
        # `cleanup` before the `rmtree`, and unconditionally. It aborts any writer still
        # open (a partition phase abandoned by an exception) and deletes both tiers' files
        # — the `rmtree` only ever reached the *local* one, so a failed query that had
        # overflowed left orphaned objects in the remote bucket, accumulating and billable,
        # with nothing recording that they existed.
        store.cleanup()
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


# How many times a window bucket that still exceeds the envelope may be re-split. The bound
# exists for the case no hash can fix: a single hot PARTITION BY value re-hashes to one
# sub-bucket however it is salted, because a window partition must land whole in one bucket
# or it would be ranked twice. Past the bound the bucket runs as it stands, which is what
# this path did unconditionally before.
_MAX_WINDOW_SPLIT_DEPTH = 3
_WINDOW_SUB_BUCKETS = 8


def _window_bucket(nat, store, handle, win_json, cfg_json, pk_indices, depth):
    """Run the window kernel over one bucket, re-splitting it first if it does not fit.

    The bucket count is sized from the caller's `num_partitions`, which is a constant, so a
    skewed PARTITION BY leaves one bucket far over the envelope — and the kernel needs its
    bucket materialized, so that bucket OOMs at exactly the point spilling was meant to
    prevent it. Asking the handle for its resident size *before* reading is what lets the
    split happen without first pulling in the thing that does not fit.

    `logical_nbytes` is the **uncompressed** size, which is what reading it back costs in
    RAM; `nbytes` is the on-disk size and for a compressible bucket can be several times
    smaller, which would let an over-large bucket through.

    A re-split re-partitions by the **same** PARTITION BY keys under a depth-derived salt, so
    every window partition still lands whole in one sub-bucket. That is the correctness
    condition, and it is stricter than a join's: a partition split across two sub-buckets
    would be ranked twice rather than merely joined more slowly. The salt matters as much as
    the split — an unsalted re-partition of a power-of-two bucket count into another reads
    the same low hash bits and moves no rows at all.
    """
    bucket_max = active_config().memory.spill_bucket_max_bytes
    resident = handle.logical_nbytes or handle.nbytes
    if bucket_max > 0 and resident > bucket_max and depth < _MAX_WINDOW_SPLIT_DEPTH:
        sub_writers: dict[int, object] = {}
        salt = _split_salt(depth)
        for batch in store.read_stream(handle):
            parts = nat.partition_batches_salted([batch], pk_indices, _WINDOW_SUB_BUCKETS, salt)
            for sb, sub in enumerate(parts):
                for pb in sub:
                    if not pb.num_rows:
                        continue
                    w = sub_writers.get(sb)
                    if w is None:
                        w = store.writer(f"win_d{depth}_s{sb}_{id(handle):x}")
                        sub_writers[sb] = w
                    w.write(pb)
        sub_handles = {sb: w.close() for sb, w in sub_writers.items()}
        # The parent has been fully re-partitioned, and it is the largest file in the
        # recursion, so giving it back now keeps grace recursion costing the same disk at
        # each level instead of more.
        store.release(handle)
        for sb in range(_WINDOW_SUB_BUCKETS):
            h = sub_handles.get(sb)
            if h is not None:
                yield from _window_bucket(nat, store, h, win_json, cfg_json, pk_indices, depth + 1)
        return

    # Reserved so the pool sees this read, and released once consumed so peak scratch is the
    # outstanding buckets rather than the whole spilled input. Each bucket holds complete
    # partitions and is read exactly once.
    with store.read_reserved(handle) as stream:
        bucket = list(stream)
    store.release(handle)
    if not bucket:
        return
    for rb in nat.execute_plan(win_json, [bucket], cfg_json):
        if rb.num_rows:
            yield rb
