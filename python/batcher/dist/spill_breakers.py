"""Out-of-core streaming for the binary/ordering breakers: sort, join, window.

These reuse the same radix/range-partition-to-disk machinery as the aggregate spill
(`spill`), but each is shaped as a **generator** that yields its result one bounded
bucket at a time, so `iter_batches()` streams a sort / join / window in bounded
memory instead of materializing the whole result. The `execute_spilling_*` wrappers
collect the same generator for `collect()` — one implementation, two consumers.

The bounded-memory contract per operator:

* **sort** — range-partition by the (single, plain-column) key into ordered buckets,
  sort each bucket, yield in key order: globally sorted with no k-way merge, bounded
  by one bucket. A top-N `limit` stops once `limit` rows are emitted.
* **join** — co-partition both sides by join key (equal keys share a bucket), join
  and yield each bucket pair independently: the union is the full join, bounded by
  one bucket pair rather than the whole build side.
* **window** — grace-partition by the (plain-column) PARTITION BY keys so each bucket
  holds *complete* partitions, run the window kernel per bucket and yield: bounded by
  one bucket. A keyless (global) window has no splittable partition, so no spill path.
"""

from __future__ import annotations

import json
import shutil

import pyarrow as pa

from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
from batcher.dist.spill import _fd_safe, _make_store, _work_dir
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Join, Sort, Window

__all__ = [
    "execute_spilling_join",
    "execute_spilling_sort",
    "reduce_join_paths_spilling",
    "stage_and_partition",
    "stream_spilling_join",
    "stream_spilling_sort",
    "stream_spilling_window",
    "supports_spilling_sort",
    "supports_spilling_window",
]


# --- join ---------------------------------------------------------------------


def execute_spilling_join(
    join: Join,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
) -> pa.Table:
    """Join out-of-core, returning the full materialized table.

    Thin consumer of `stream_spilling_join`; `iter_batches()` streams the same
    bounded-memory per-bucket pipeline."""
    batches = list(stream_spilling_join(join, sources, num_partitions, spill_dir))
    if batches:
        return pa.Table.from_batches(batches)
    return pa.table({o.alias: [] for o in join.output})


def stream_spilling_join(
    join: Join,
    sources: list[Source],
    num_partitions: int = 16,
    spill_dir: str | None = None,
):
    """Join out-of-core, yielding each co-partitioned bucket-pair's join output as it
    is produced — bounded by one bucket pair, never the whole build side.

    Co-partition both sides by join key (equal keys hash to the same bucket on both
    sides), then join and yield each bucket pair independently; the union of the
    per-bucket outputs is the full join."""
    import batcher._native as nat

    cfg_json = active_config().engine_config_json()
    left_plan, left_sid = _relabel_single_source(join.left)
    right_plan, right_sid = _relabel_single_source(join.right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    join_ir = json.dumps(
        {
            "op": "hash_join",
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
            "left_keys": list(join.left_keys),
            "right_keys": list(join.right_keys),
            "join_type": join.join_type,
            "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in join.output],
        }
    )
    n_buckets = _fd_safe(num_partitions)

    work_dir, owns_dir = _work_dir(spill_dir, "batcher_join_spill_")
    store = _make_store(work_dir)
    try:
        # Output schema of each side (from a 0-row probe) so empty buckets still
        # carry types for the null-extended side of an outer join.
        left_schema = _side_schema(nat, left_ir, sources[left_sid], cfg_json)
        right_schema = _side_schema(nat, right_ir, sources[right_sid], cfg_json)

        left_handles = _spill_side(
            nat, left_ir, list(join.left_keys), sources[left_sid], n_buckets, store, "L", cfg_json
        )
        right_handles = _spill_side(
            nat,
            right_ir,
            list(join.right_keys),
            sources[right_sid],
            n_buckets,
            store,
            "R",
            cfg_json,
        )

        for b in range(n_buckets):
            if left_handles[b] is None and right_handles[b] is None:
                continue
            left_b = (store.read(left_handles[b]) if left_handles[b] else None) or [
                _empty_batch(left_schema)
            ]
            right_b = (store.read(right_handles[b]) if right_handles[b] else None) or [
                _empty_batch(right_schema)
            ]
            for rb in nat.execute_plan(join_ir, [left_b, right_b], cfg_json):
                if rb.num_rows > 0:
                    yield rb
    finally:
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def reduce_join_paths_spilling(
    join_ir: str,
    left_keys: list[str],
    right_keys: list[str],
    left_paths: list[str],
    right_paths: list[str],
    work_dir: str,
    n_buckets: int,
    engine_config: str,
) -> list[pa.RecordBatch]:
    """Reduce a co-partitioned shuffle join in bounded memory from on-disk buckets.

    Each input path is one mapper's contribution to this reducer's bucket. Both sides
    are re-partitioned into `n_buckets` sub-buckets on disk — one mapper file read at a
    time — then joined one sub-bucket pair at a time, so peak memory is one contribution
    plus one pair, never the whole (possibly skewed) bucket. Equal keys hash to the same
    sub-bucket on both sides, so the union of the per-sub-bucket joins is exactly the
    full join (a still-large pair spills again inside the native join). The alternative —
    reading every path into one Python list before the join — peaks at the whole bucket.
    """
    import batcher._native as nat
    from batcher.dist.shuffle_io import read_ipc

    n = _fd_safe(n_buckets)
    left_sub, left_schema = _spill_paths_to_subbuckets(
        nat, left_paths, left_keys, n, work_dir, "rl"
    )
    right_sub, right_schema = _spill_paths_to_subbuckets(
        nat, right_paths, right_keys, n, work_dir, "rr"
    )
    out: list[pa.RecordBatch] = []
    for i in range(n):
        if left_sub[i] is None and right_sub[i] is None:
            continue
        # A missing side becomes a 0-row, schema-bearing input so an outer join still
        # null-extends the present side (matching the non-spilling reducer's behavior).
        left_b = read_ipc(left_sub[i]) if left_sub[i] else _maybe_empty(left_schema)
        right_b = read_ipc(right_sub[i]) if right_sub[i] else _maybe_empty(right_schema)
        out.extend(
            rb for rb in nat.execute_plan(join_ir, [left_b, right_b], engine_config) if rb.num_rows
        )
    return out


def _maybe_empty(schema: pa.Schema | None) -> list[pa.RecordBatch]:
    """A one-element schema-bearing empty batch list, or `[]` when no schema was seen
    (the side had no data at all — the native join infers types from the other side)."""
    return [_empty_batch(schema)] if schema is not None else []


def _spill_paths_to_subbuckets(nat, paths, key_names, n, work_dir, tag):
    """Hash-partition the batches in `paths` into `n` sub-bucket IPC files by `key_names`.

    Reads one path at a time and appends each sub-bucket's batches to an incrementally
    written stream file, so peak memory is one path's batches — not the whole side.
    Returns `(sub_bucket_paths, schema)` where an absent sub-bucket is `None`.
    """
    import os

    from batcher.dist.shuffle_io import read_ipc

    writers: list = [None] * n
    sinks: list = [None] * n
    out_paths: list[str | None] = [None] * n
    schema: pa.Schema | None = None
    key_idx: list[int] = []
    for p in paths:
        batches = read_ipc(p)
        if not batches:
            continue
        if schema is None:
            schema = batches[0].schema
            key_idx = [schema.get_field_index(k) for k in key_names]
        for i, bucket in enumerate(nat.partition_batches(batches, key_idx, n)):
            for b in bucket:
                if not b.num_rows:
                    continue
                if writers[i] is None:
                    out_paths[i] = os.path.join(work_dir, f"{tag}_{i}.arrow")
                    sinks[i] = pa.OSFile(out_paths[i], "wb")
                    writers[i] = pa.ipc.new_stream(sinks[i], schema)
                writers[i].write_batch(b)
    for w, s in zip(writers, sinks, strict=True):
        if w is not None:
            w.close()
        if s is not None:
            s.close()
    return out_paths, schema


def _side_schema(nat, sub_ir: str, source: Source, engine_config: str) -> pa.Schema:
    """The sub-plan's output schema, via a 0-row probe through the engine."""
    empty = pa.RecordBatch.from_pylist([], schema=source.schema())
    out = nat.execute_plan(sub_ir, [[empty]], engine_config)
    return out[0].schema if out else source.schema()


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist([], schema=schema)


def _spill_side(nat, sub_ir, key_names, source, n_buckets, store, tag, engine_config):
    """Stream a source through its sub-plan, hash-partition by key, spill by tier.
    Returns a list of per-bucket `SpillHandle`s (None where a bucket received no
    rows). Buckets overflow local→remote through the shared tiered `store`."""
    writers: dict[int, object] = {}
    handles: list[object] = [None] * n_buckets
    key_idx: list[int] | None = None

    for batch in source.iter_batches():
        if batch.num_rows == 0:
            continue
        rows = nat.execute_plan(sub_ir, [[batch]], engine_config)
        if not rows:
            continue
        if key_idx is None:
            key_idx = [rows[0].schema.get_field_index(k) for k in key_names]
        buckets = [rows] if n_buckets == 1 else nat.partition_batches(rows, key_idx, n_buckets)
        for b, part_batches in enumerate(buckets):
            for pb in part_batches:
                if pb.num_rows == 0:
                    continue
                w = writers.get(b)
                if w is None:
                    w = store.writer(f"{tag}_bucket_{b}")
                    writers[b] = w
                w.write(pb)
    for b, w in writers.items():
        handles[b] = w.close()
    return handles


# --- sort ---------------------------------------------------------------------


def supports_spilling_sort(sort: Sort) -> bool:
    """External sort spill currently supports a single plain-column sort key."""
    return len(sort.keys) == 1 and isinstance(sort.keys[0].expr, Col)


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


def stage_and_partition(source, map_ir, key_name, nulls_first, n_buckets, store, cfg_json):
    """Map `source` through `map_ir`, sample the key, and range-partition the mapped
    output into `n_buckets` ordered disk buckets (bucket 0 = smallest keys; nulls to
    the front/back bucket per `nulls_first`). Returns per-bucket spill handles (`None`
    where a bucket got no rows), in **key-ascending** bucket order. Memory is bounded
    by a single staged batch + one bucket.

    Shared by the streaming sort (`stream_spilling_sort`) and the streaming global
    window (`window_stream`): both need the identical ordered-bucket partition so that
    no key value spans a bucket boundary (equal keys land together), which is what
    makes per-bucket processing + a key-order concatenation globally correct.
    """
    import numpy as np

    import batcher._native as nat

    # --- pass 1: map the source ONCE, stage the mapped output to disk (so pass 2
    # re-reads locally, not re-mapping a possibly-remote source), and sample the key.
    sample: list = []
    stage = store.writer("stage")
    for batch in source.iter_batches():
        if batch.num_rows == 0:
            continue
        for rb in nat.execute_plan(map_ir, [[batch]], cfg_json):
            if not rb.num_rows:
                continue
            stage.write(rb)
            arr = rb.column(rb.schema.get_field_index(key_name))
            vals = arr.drop_null().to_numpy(zero_copy_only=False)
            if len(vals):
                sample.append(vals[:: max(1, len(vals) // 256)])  # stride-sample
    stage_handle = stage.close()
    boundaries = None
    if n_buckets > 1 and sample:
        allk = np.concatenate(sample)
        if len(allk):
            qs = np.linspace(0, 1, n_buckets + 1)[1:-1]
            boundaries = np.unique(np.quantile(allk, qs))

    # --- pass 2: assign staged rows to ordered buckets and spill ----------
    writers: dict[int, object] = {}
    handles: list[object] = [None] * n_buckets
    null_bucket = 0 if nulls_first else n_buckets - 1
    staged = store.read_stream(stage_handle) if stage_handle is not None else iter(())
    for rb in staged:
        arr = rb.column(rb.schema.get_field_index(key_name))
        if boundaries is None or len(boundaries) == 0:
            bucket_ids = np.zeros(rb.num_rows, dtype=np.int64)
        else:
            vals = arr.to_numpy(zero_copy_only=False)
            bucket_ids = np.searchsorted(boundaries, vals, side="right").astype(np.int64)
        if arr.null_count:
            null_mask = arr.is_null().to_numpy(zero_copy_only=False)
            bucket_ids[null_mask] = null_bucket
        for b in range(n_buckets):
            mask = bucket_ids == b
            if not mask.any():
                continue
            sub = rb.filter(pa.array(mask))
            w = writers.get(b)
            if w is None:
                w = store.writer(f"bucket_{b}")
                writers[b] = w
            w.write(sub)
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
    import batcher._native as nat

    cfg_json = active_config().engine_config_json()
    key = sort.keys[0]
    desc, nulls_first = key.descending, key.nulls_first
    n_buckets = _fd_safe(num_partitions)

    map_plan, sid = _relabel_single_source(sort.input)
    map_ir = json.dumps(map_plan.to_ir())
    sort_ir = json.dumps(
        {
            "op": "sort",
            "input": {"op": "scan", "source_id": 0},
            "keys": [{"expr": key.expr.to_ir(), "descending": desc, "nulls_first": nulls_first}],
            "limit": sort.limit,
        }
    )

    work_dir, owns_dir = _work_dir(spill_dir, "batcher_sort_spill_")
    store = _make_store(work_dir)
    try:
        handles = stage_and_partition(
            sources[sid], map_ir, key.expr.name, nulls_first, n_buckets, store, cfg_json
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


# --- window -------------------------------------------------------------------


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
    import batcher._native as nat

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
        for batch in source.iter_batches():
            if batch.num_rows == 0:
                continue
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
