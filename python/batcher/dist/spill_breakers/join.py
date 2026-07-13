"""Out-of-core join: co-partition both sides by key, join one bucket pair at a time.

Equal keys hash to the same bucket on both sides, so the union of the per-bucket-pair joins
is exactly the full join — bounded by one bucket pair rather than the whole build side.
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
from batcher.plan.logical import Join


def supports_spilling_join(join: Join) -> bool:
    """Whether this join can grace-partition out-of-core: each side must name one source.

    A side spanning two sources — a join whose operand is itself a join, i.e. any 3+-table
    query — cannot, and this path used to assert on it rather than decline. Sort and Window
    already gate this way. `False` falls back to the in-memory join: costs memory, never
    correctness."""
    return _single_source(join.left) and _single_source(join.right)


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
    nat = engine()

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
    nat = engine()
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

    for batch in _iter_spill_morsels(source):
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
