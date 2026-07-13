"""Out-of-core sort: range-partition into ordered buckets, sort each, yield in key order.

Globally sorted with no k-way merge, bounded by one bucket. A top-N `limit` stops once
`limit` rows have been emitted.
"""

from __future__ import annotations

import json
import shutil

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.executors.plan_analysis import _single_source
from batcher.dist.spill import _fd_safe, _iter_spill_morsels, _make_store, _work_dir
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Sort


def supports_spilling_sort(sort: Sort, sources: list[Source] | None = None) -> bool:
    """Whether this sort can run out-of-core via the ordered range partition.

    The leading key must be a plain column (the range partition splits on it — equal
    values share a bucket — and each bucket is then sorted by the full key list) *and*
    it must be numeric: the boundaries come from the KLL sketch (`column_quantiles`),
    which only sketches numeric columns, and `range_partition_batches` rejects anything
    else. Returning `False` is a graceful fallback — the caller runs the ordinary
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
    dt = schema.field(idx).type
    return pa.types.is_integer(dt) or pa.types.is_floating(dt)


def execute_spilling_sort(
    sort: Sort,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
) -> pa.Table:
    """Sort out-of-core, returning the full materialized table.

    Thin consumer of `stream_spilling_sort` — the bounded-memory bucket pipeline is
    one implementation; this collects it, `iter_batches()` streams it."""
    _, sid = _relabel_single_source(sort.input)
    batches = list(stream_spilling_sort(sort, sources, num_partitions, spill_dir))
    if batches:
        return pa.Table.from_batches(batches)
    return pa.table({f.name: [] for f in sources[sid].schema()})


def stage_and_partition(
    source, map_ir, key_name, nulls_first, descending, n_buckets, store, cfg_json
):
    """Map `source` through `map_ir`, sample the key, and range-partition the mapped output
    into `n_buckets` ordered disk buckets (key-ascending; `None` where a bucket got no rows).

    Shared by the streaming sort and the streaming global window: both need equal keys to land
    in one bucket, which is what makes per-bucket processing + key-order concatenation globally
    correct. Sampling (`column_quantiles`) and bucketing (`bucketize`) are the *same* primitives
    the distributed range-sort uses, so the two sorts cannot drift apart, and the per-row work
    stays in Rust. `descending` is load-bearing: the caller emits buckets high→low for a
    descending sort, so which end the nulls belong at depends on it, not on `nulls_first` alone.
    """
    from batcher.dist.executors.partition_io import SAMPLE_PROBS, bucketize, merge_boundaries

    nat = engine()

    # --- pass 1: map the source ONCE, stage the mapped output to disk (so pass 2
    # re-reads locally, not re-mapping a possibly-remote source), and sketch the key.
    grids: list[tuple[list[float], int]] = []
    stage = store.writer("stage")
    for batch in _iter_spill_morsels(source):
        for rb in nat.execute_plan(map_ir, [[batch]], cfg_json):
            if not rb.num_rows:
                continue
            stage.write(rb)
            grid = nat.column_quantiles([key_name], [rb], list(SAMPLE_PROBS)).get(key_name, [])
            if grid:
                grids.append((grid, rb.num_rows))
    stage_handle = stage.close()
    boundaries = merge_boundaries(grids, n_buckets) if n_buckets > 1 else []

    # --- pass 2: assign staged rows to ordered buckets and spill ----------
    writers: dict[int, object] = {}
    handles: list[object] = [None] * n_buckets
    staged = store.read_stream(stage_handle) if stage_handle is not None else iter(())
    for rb in staged:
        parts = bucketize([rb], key_name, boundaries, n_buckets, nulls_first, descending)
        for b, part in enumerate(parts):
            if not part:
                continue
            w = writers.get(b)
            if w is None:
                w = store.writer(f"bucket_{b}")
                writers[b] = w
            for pb in part:
                if pb.num_rows:
                    w.write(pb)
    for b, w in writers.items():
        handles[b] = w.close()
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
    keys_ir = [
        {"expr": k.expr.to_ir(), "descending": k.descending, "nulls_first": k.nulls_first}
        for k in sort.keys
    ]
    scan = {"op": "scan", "source_id": 0}
    sort_ir = json.dumps({"op": "sort", "input": scan, "keys": keys_ir, "limit": sort.limit})

    work_dir, owns_dir = _work_dir(spill_dir, "batcher_sort_spill_")
    store = _make_store(work_dir)
    try:
        handles = stage_and_partition(
            sources[sid], map_ir, key.expr.name, nulls_first, desc, n_buckets, store, cfg_json
        )
        # Sort each bucket, yield in key order (reversed for descending).
        order = range(n_buckets - 1, -1, -1) if desc else range(n_buckets)
        emitted = 0
        for b in order:
            if handles[b] is None:
                continue
            bucket = store.read(handles[b])
            if not bucket:
                continue
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
    finally:
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
