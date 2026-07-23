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
from batcher.dist.spill import _fd_safe, _iter_spill_morsels, _make_store, _work_dir
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
        for batch in _iter_spill_morsels(source):
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
            bucket = store.read(handles[b])
            if not bucket:
                continue
            for rb in nat.execute_plan(win_json, [bucket], cfg_json):
                if rb.num_rows:
                    yield rb
    finally:
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
