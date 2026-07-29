"""Bounded-memory streaming for a *global* (no-``PARTITION BY``) window.

A global window over an ``ORDER BY`` key can't grace-partition the way a
``PARTITION BY`` window does (there is one partition over all rows), so it has no
per-bucket-independent spill path. But it *can* be streamed by **ordered-bucket
offsetting**:

1. Range-partition the input by the single plain-column order key into ordered
   buckets (`stage_and_partition`) — equal keys land in one bucket, so no peer group
   or value ever spans a boundary.
2. Run the ordinary (in-memory) window kernel on each bucket in key order, then apply
   a uniform vectorized **offset** so the within-bucket result becomes the global one:
   `row_number`/`rank` += rows in prior buckets; `dense_rank` += distinct keys in prior
   buckets; running `sum`/`count` += the prior buckets' total; running `min`/`max` =
   element-wise against the prior buckets' running extreme; `avg` = the offset running
   `sum` over the offset running `count` (a running mean is not a constant shift, but its
   sum and count each are, so it is offsettable through them); `first_value` = the first
   bucket's first value. Each offset is correct precisely because no group spans a
   bucket, so prior buckets contribute a single constant (or element-wise) shift.

Peak memory is one bucket, never the whole relation. Restricted to a single
plain-column order key, no explicit frame, and the offsettable function subset
(`lag`/`lead`/`last_value`/`ntile`/`percent_rank`/`cume_dist` fall back to the
materializing kernel — still correct, just not bounded). Output is yielded
bucket-by-bucket in key order — a valid permutation of the rows (a window result is
an unordered relation, like the distributed window shuffle).
"""

from __future__ import annotations

import json
import shutil

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.spill import _fd_safe, _make_store, _work_dir, map_projection
from batcher.dist.spill_breakers import stage_and_partition
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Window

__all__ = ["stream_spilling_global_window", "supports_streaming_global_window"]

# Window functions whose global value is recovered from the within-bucket value plus a
# constant/element-wise per-bucket offset (so per-bucket compute + offset == single-node).
# `avg` qualifies through its running `sum` and `count`, each of which is a constant shift.
_OFFSETTABLE = frozenset({"row_number", "rank", "dense_rank", "sum", "count", "min", "max", "avg"})
_NEEDS_COL_INPUT = frozenset({"sum", "count", "min", "max", "avg", "first_value"})
_UNSET = object()


def supports_streaming_global_window(window: Window) -> bool:
    """Whether `window` is a global window the ordered-bucket-offset stream supports.

    Requires: no partition keys (global); exactly one plain-column order key (to
    range-partition on); every function offsettable or `first_value`, with no explicit
    frame; and aggregate/`first_value` inputs are plain columns.
    """
    if window.rank_limit is not None or window.partition_keys:
        return False
    if len(window.order_keys) != 1 or not isinstance(window.order_keys[0].expr, Col):
        return False
    for fn in window.functions:
        if fn.frame is not None:
            return False
        if fn.func not in _OFFSETTABLE and fn.func != "first_value":
            return False
        if fn.func in _NEEDS_COL_INPUT and not isinstance(fn.input, Col):
            return False
    return True


def _inject_avg_helpers(window: Window, win_ir: dict) -> dict[str, tuple[str, str]]:
    """Append a running `sum` and `count` to `win_ir` for every `avg` function.

    Returns ``{avg_alias: (sum_alias, count_alias)}``. The private aliases carry a prefix
    that cannot collide with a user column (window aliases are validated against the input
    schema, which never contains one), so the kernel computes them for free next to the
    average, and the streamer divides one offset running total by the other.
    """
    helpers: dict[str, tuple[str, str]] = {}
    for fn in window.functions:
        if fn.func == "avg":
            sa, ca = f"__ws_sum::{fn.alias}", f"__ws_cnt::{fn.alias}"
            helpers[fn.alias] = (sa, ca)
            inp = fn.input.to_ir()
            win_ir["functions"].append({"func": "sum", "alias": sa, "offset": 1, "input": inp})
            win_ir["functions"].append({"func": "count", "alias": ca, "offset": 1, "input": inp})
    return helpers


def stream_spilling_global_window(
    window: Window,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
):
    """Stream a global window in bounded memory via ordered-bucket offsetting."""
    import pyarrow.compute as pc

    nat = engine()

    cfg_json = active_config().engine_config_json()
    key = window.order_keys[0]
    key_name = key.expr.name
    desc, nulls_first = key.descending, key.nulls_first
    n_buckets = _fd_safe(num_partitions)

    map_plan, sid = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    # `to_ir()` memoizes and returns the plan's shared dict/list, so copy the pieces this
    # function rewrites (the scanned input, and the functions list `avg` appends to) before
    # touching them — mutating the cached structures would corrupt every later use of the
    # same plan.
    win_ir = dict(window.to_ir())
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_ir["functions"] = list(win_ir["functions"])
    # `avg` is offset through its running sum and count, so ask the kernel for those two
    # alongside it under private aliases; they are read back per bucket and dropped before
    # the rows are yielded, so the output schema is unchanged.
    avg_helpers = _inject_avg_helpers(window, win_ir)
    win_json = json.dumps(win_ir)

    work_dir, owns_dir = _work_dir(spill_dir, "batcher_winstream_")
    store = _make_store(work_dir)
    try:
        handles = stage_and_partition(
            sources[sid],
            map_ir,
            key_name,
            nulls_first,
            desc,
            n_buckets,
            store,
            cfg_json,
            map_projection(window, sid),
        )
        # Process buckets in *global sort order* (reversed for descending) so the
        # running offsets accumulate correctly.
        order = range(n_buckets - 1, -1, -1) if desc else range(n_buckets)
        prior_rows = 0
        # Per-function running offset accumulators (in global sort order so far).
        dense = {f.alias: 0 for f in window.functions}
        psum = {f.alias: 0 for f in window.functions}
        pcount = {f.alias: 0 for f in window.functions}
        pmin: dict = {f.alias: None for f in window.functions}
        pmax: dict = {f.alias: None for f in window.functions}
        first: dict = {f.alias: _UNSET for f in window.functions}

        for b in order:
            if handles[b] is None:
                continue
            # Reserved, so the pool sees the bucket this reduce is holding; released after
            # the bucket is consumed, so peak scratch is the outstanding buckets rather than
            # the whole spilled input. Buckets are read once, in global sort order.
            with store.read_reserved(handles[b]) as stream:
                bucket = list(stream)
            store.release(handles[b])
            if not bucket:
                continue
            out = nat.execute_plan(win_json, [bucket], cfg_json)
            if not out:
                continue
            wt = pa.Table.from_batches(out)
            n = wt.num_rows
            for fn in window.functions:
                idx = wt.schema.get_field_index(fn.alias)
                col = wt.column(idx)
                if fn.func in ("row_number", "rank"):
                    col = pc.add(col, prior_rows)
                elif fn.func == "dense_rank":
                    bucket_distinct = pc.max(col).as_py() or 0
                    col = pc.add(col, dense[fn.alias])
                    dense[fn.alias] += bucket_distinct
                elif fn.func == "sum":
                    col = pc.add(col, psum[fn.alias])
                    s = pc.sum(wt.column(fn.input.name)).as_py()
                    psum[fn.alias] += s if s is not None else 0
                elif fn.func == "count":
                    col = pc.add(col, pcount[fn.alias])
                    pcount[fn.alias] += pc.count(wt.column(fn.input.name)).as_py()
                elif fn.func == "min":
                    if pmin[fn.alias] is not None:
                        col = pc.min_element_wise(col, pa.scalar(pmin[fn.alias], col.type))
                    bmin = pc.min(wt.column(fn.input.name)).as_py()
                    if bmin is not None:
                        pmin[fn.alias] = (
                            bmin if pmin[fn.alias] is None else min(pmin[fn.alias], bmin)
                        )
                elif fn.func == "max":
                    if pmax[fn.alias] is not None:
                        col = pc.max_element_wise(col, pa.scalar(pmax[fn.alias], col.type))
                    bmax = pc.max(wt.column(fn.input.name)).as_py()
                    if bmax is not None:
                        pmax[fn.alias] = (
                            bmax if pmax[fn.alias] is None else max(pmax[fn.alias], bmax)
                        )
                elif fn.func == "avg":
                    sa, ca = avg_helpers[fn.alias]
                    # The kernel's running sum is NULL until the first non-null input, but
                    # for the offset that means a 0 contribution — coalesce before adding
                    # the prior buckets' total. The running count is 0 (never null) there.
                    within_sum = pc.fill_null(pc.cast(wt.column(sa), pa.float64()), 0.0)
                    tot_sum = pc.add(within_sum, float(psum[fn.alias]))
                    tot_cnt = pc.add(wt.column(ca), pcount[fn.alias])
                    # No non-null value through this row (globally) ⇒ the mean is NULL, and
                    # the 0/0 the divide would produce there is discarded by the mask.
                    col = pc.if_else(
                        pc.equal(tot_cnt, 0),
                        pa.scalar(None, pa.float64()),
                        pc.divide(tot_sum, pc.cast(tot_cnt, pa.float64())),
                    )
                    bs = pc.sum(wt.column(fn.input.name)).as_py()
                    psum[fn.alias] += bs if bs is not None else 0
                    pcount[fn.alias] += pc.count(wt.column(fn.input.name)).as_py()
                elif fn.func == "first_value":
                    if first[fn.alias] is _UNSET:
                        first[fn.alias] = col[0].as_py() if n else None
                    else:
                        col = pa.array([first[fn.alias]] * n, type=col.type)
                wt = wt.set_column(idx, wt.schema.field(idx), col)
            if avg_helpers:
                # Drop the private sum/count columns the avg offset borrowed.
                hidden = {a for pair in avg_helpers.values() for a in pair}
                wt = wt.select([c for c in wt.column_names if c not in hidden])
            prior_rows += n
            for batch in wt.to_batches():
                if batch.num_rows:
                    yield batch
    finally:
        # `cleanup` before the `rmtree`, and unconditionally. It aborts any writer still
        # open (a partition phase abandoned by an exception) and deletes both tiers' files
        # — the `rmtree` only ever reached the *local* one, so a failed query that had
        # overflowed left orphaned objects in the remote bucket, accumulating and billable,
        # with nothing recording that they existed.
        store.cleanup()
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
