"""Out-of-core window: grace-partition by the PARTITION BY keys so each bucket holds
*complete* partitions, run the window kernel per bucket, and yield — bounded by one bucket.

A keyless (global) window has no splittable partition, so it has no spill path here.
"""

from __future__ import annotations

import json

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.spill import (
    _fd_safe,
    _iter_spill_morsels,
    map_projection,
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
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.ir_specs import task_scan_ir
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
    # The reduce runs the window over its bucket as a single in-memory source 0. `to_ir()`
    # **memoizes per node and hands back the plan's own dict**, so re-rooting it in place
    # does not build a second plan — it edits the caller's, permanently. Streaming a window
    # once and then touching the same `Dataset` again therefore ran a plan whose window had
    # lost whatever produced its input: `ds.with_columns(x=...).window(..., functions on x)`
    # came back as `window ← scan`, and the next `collect()` either returned the wrong rows
    # or raised `unknown column: x`. Copy before rewriting.
    win_ir = dict(window.to_ir())
    win_ir["input"] = task_scan_ir()
    win_json = json.dumps(win_ir)
    n_buckets = _fd_safe(num_partitions)
    source = sources[sid]

    with spill_scratch("batcher_win_spill_", spill_dir) as store:
        writers = BucketWriters(store, "win_bucket")
        for batch in _iter_spill_morsels(source, map_projection(window, sid)):
            rows = nat.execute_plan(map_ir, [[batch]], cfg_json)
            if rows:
                writers.add(nat.partition_batches(rows, pk_indices, n_buckets))
        handles = writers.close_dense(n_buckets)

        for b in range(n_buckets):
            if handles[b] is None:
                continue
            yield from _window_bucket(nat, store, handles[b], win_json, cfg_json, pk_indices, 0)


# The grace recursion is `dist.spill.buckets`': same depth bound, same width, same salt as the
# aggregate's and the join's. A window's correctness condition is the strictest of the three —
# a PARTITION BY value split across two sub-buckets would be *ranked twice*, not merely joined
# more slowly — and it is met the same way, by re-splitting on the same keys.
_MAX_WINDOW_SPLIT_DEPTH = GRACE_DEPTH
_WINDOW_SUB_BUCKETS = GRACE_SUB_BUCKETS


def _window_bucket(nat, store, handle, win_json, cfg_json, pk_indices, depth):
    """Run the window kernel over one bucket, re-splitting it first if it does not fit.

    The bucket count is sized from the caller's `num_partitions`, which is a constant, so a
    skewed PARTITION BY leaves one bucket far over the envelope — and the kernel needs its
    bucket materialized, so that bucket would OOM at exactly the point spilling was meant to
    prevent it. The handle is measured *before* being read, which is what lets the split
    happen without first pulling in the thing that does not fit.
    """
    if over_envelope(handle, depth, max_depth=_MAX_WINDOW_SPLIT_DEPTH):
        subs = regrace(
            nat,
            store,
            handle,
            pk_indices,
            split_salt(depth),
            f"win_d{depth}_{id(handle):x}",
            n_sub=_WINDOW_SUB_BUCKETS,
        )
        for sb in range(_WINDOW_SUB_BUCKETS):
            h = subs.get(sb)
            if h is not None:
                yield from _window_bucket(nat, store, h, win_json, cfg_json, pk_indices, depth + 1)
        return

    # Released once consumed, so peak scratch is the outstanding buckets rather than the whole
    # spilled input. Each bucket holds complete partitions and is read exactly once.
    bucket = read_reserved_bucket(store, handle)
    store.release(handle)
    if not bucket:
        return
    for rb in nat.execute_plan(win_json, [bucket], cfg_json):
        if rb.num_rows:
            yield rb
