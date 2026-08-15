"""Bounded-memory streaming for a *global* (no-``PARTITION BY``) window, on one node.

A global window over an ``ORDER BY`` key can't grace-partition the way a ``PARTITION BY``
window does (there is one partition over all rows), so it has no per-bucket-independent
spill path. But it *can* be streamed by **ordered-bucket offsetting**: range-partition the
input by the single plain-column order key into ordered buckets (`stage_and_partition`),
run the ordinary in-memory window kernel on each bucket in key order, and shift each
bucket's result to its global value. The algebra and its limits live in `offsets`; this
module is the single-node scheduling of it -- one bucket resident at a time, so peak memory
is one bucket rather than the whole relation.

`flight` is the same algebra scheduled the other way: every bucket at once, on different
machines. That the two share `offsets` is what keeps the streamed answer and the distributed
answer the same answer.

Output is yielded bucket-by-bucket in key order -- a valid permutation of the rows (a window
result is an unordered relation, like the distributed window shuffle).
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.global_window.offsets import (
    OrderedBucketOffsets,
    bucket_order,
    inject_avg_helpers,
)
from batcher.dist.spill import _fd_safe, map_projection
from batcher.dist.spill.buckets import read_reserved_bucket, spill_scratch
from batcher.dist.spill_breakers import stage_and_partition
from batcher.io.source import Source
from batcher.plan.ir_specs import task_scan_ir
from batcher.plan.logical import Window

__all__ = ["stream_spilling_global_window"]


def stream_spilling_global_window(
    window: Window,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
):
    """Stream a global window in bounded memory via ordered-bucket offsetting."""
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
    win_ir["input"] = task_scan_ir()
    win_ir["functions"] = list(win_ir["functions"])
    # `avg` is offset through its running sum and count, so ask the kernel for those two
    # alongside it under private aliases; they are read back per bucket and dropped before
    # the rows are yielded, so the output schema is unchanged.
    avg_helpers = inject_avg_helpers(window, win_ir)
    win_json = json.dumps(win_ir)

    with spill_scratch("batcher_winstream_", spill_dir) as store:
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
        # Process buckets in *global sort order* (reversed for descending) so the running
        # offsets accumulate correctly. `handles`, not `n_buckets`: `stage_and_partition`
        # sizes the split from the staged bytes, so the bucket count is decided there.
        offsets = OrderedBucketOffsets(window, avg_helpers)
        for b in bucket_order(len(handles), desc):
            if handles[b] is None:
                continue
            # Released after the bucket is consumed, so peak scratch is the outstanding
            # buckets rather than the whole spilled input. Read once, in global sort order.
            bucket = read_reserved_bucket(store, handles[b])
            store.release(handles[b])
            if not bucket:
                continue
            out = nat.execute_plan(win_json, [bucket], cfg_json)
            if not out:
                continue
            for batch in offsets.apply(pa.Table.from_batches(out)).to_batches():
                if batch.num_rows:
                    yield batch
