"""Out-of-core join: co-partition both sides by key, join one bucket pair at a time.

Equal keys hash to the same bucket on both sides, so the union of the per-bucket-pair joins
is exactly the full join — bounded by one bucket pair rather than the whole build side.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.dist.executor import _relabel_single_source
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
    over_envelope,
    read_reserved_bucket,
    regrace,
    resident_bytes,
    spill_scratch,
    split_salt,
)
from batcher.io.source import Source
from batcher.plan.logical import Join

# The grace recursion is `dist.spill.buckets`': same depth bound, same width, same salt as the
# aggregate's and the window's. Named locally because the reduce below reads them.
_MAX_JOIN_SPLIT_DEPTH = GRACE_DEPTH
_JOIN_SUB_BUCKETS = GRACE_SUB_BUCKETS
# The reducer's sub-bucket re-partition is a *single* level, so it takes the depth-0 salt.
_SUBBUCKET_SALT = split_salt(0)


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
    # Typed, not null-typed. `pa.table({alias: []})` gave every column a null type, so an
    # empty out-of-core join disagreed with the in-memory one on the result *schema* —
    # "spilled == in-memory" has to hold for empty results too, which is why the spilling
    # aggregate routes its empty through `empty_result_table` rather than a dict of lists.
    from batcher.dist.executors.plan_analysis import empty_result_table

    return empty_result_table(join, [o.alias for o in join.output])


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
            **join.shape_ir(),
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
        }
    )
    n_buckets = _fd_safe(num_partitions)

    with spill_scratch("batcher_join_spill_", spill_dir) as store:
        # Output schema of each side (from a 0-row probe) so empty buckets still
        # carry types for the null-extended side of an outer join.
        left_schema = _side_schema(nat, left_ir, sources[left_sid], cfg_json)
        right_schema = _side_schema(nat, right_ir, sources[right_sid], cfg_json)

        left_handles = _spill_side(
            nat,
            left_ir,
            list(join.left_keys),
            sources[left_sid],
            n_buckets,
            store,
            "L",
            cfg_json,
            map_projection(join, left_sid),
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
            map_projection(join, right_sid),
        )

        key_idx = (
            _key_indices(left_schema, join.left_keys),
            _key_indices(right_schema, join.right_keys),
        )
        for b in range(n_buckets):
            if left_handles[b] is None and right_handles[b] is None:
                continue
            yield from _join_bucket_pair(
                nat,
                store,
                left_handles[b],
                right_handles[b],
                join_ir,
                cfg_json,
                left_schema,
                right_schema,
                key_idx,
                0,
            )


def reduce_join_paths_spilling(
    join_ir: str,
    left_keys: list[str],
    right_keys: list[str],
    left_paths: list[str],
    right_paths: list[str],
    work_dir: str,
    n_buckets: int,
    engine_config: str,
    left_schema: pa.Schema | None = None,
    right_schema: pa.Schema | None = None,
) -> list[pa.RecordBatch]:
    """Reduce a co-partitioned shuffle join in bounded memory from on-disk buckets.

    Each input path is one mapper's contribution to this reducer's bucket. Both sides
    are re-partitioned into `n_buckets` sub-buckets on disk — one mapper file read at a
    time — then joined one sub-bucket pair at a time, so peak memory is one contribution
    plus one pair, never the whole (possibly skewed) bucket. Equal keys hash to the same
    sub-bucket on both sides, so the union of the per-sub-bucket joins is exactly the
    full join (a still-large pair spills again inside the native join). The alternative —
    reading every path into one Python list before the join — peaks at the whole bucket.

    `left_schema`/`right_schema` are optional fallbacks for a side that produced **no**
    rows at all (so no schema can be inferred from its data): an outer join must still
    null-extend the present side against a schema-bearing empty, and without the fallback
    that side comes through untyped and the null-extension is lost. When a side has data,
    its inferred schema wins and the fallback is unused.
    """
    nat = engine()
    from batcher.dist.shuffle_io import read_ipc

    n = _fd_safe(n_buckets)
    left_sub, left_data_schema = _spill_paths_to_subbuckets(
        nat, left_paths, left_keys, n, work_dir, "rl"
    )
    right_sub, right_data_schema = _spill_paths_to_subbuckets(
        nat, right_paths, right_keys, n, work_dir, "rr"
    )
    left_schema = left_data_schema or left_schema
    right_schema = right_data_schema or right_schema
    out: list[pa.RecordBatch] = []
    for i in range(n):
        if left_sub[i] is None and right_sub[i] is None:
            continue
        # A missing side becomes a 0-row, schema-bearing input so an outer join still
        # null-extends the present side (matching the non-spilling reducer's behavior).
        #
        # One pair at a time, and measurably so: joining several concurrently in a thread pool
        # was tried and is 35% *slower* (TPC-H sf100 q3, three runs each: 23.0s serial against
        # 31.7s at eight-way, and utilization fell from 39% to 32%). `execute_plan` already
        # spreads a pair across this worker's whole core grant, so Python-level concurrency
        # only oversubscribes them — the same thread-thrash `engine_config_json` sizes
        # `parallelism` to avoid.
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

    **Salted**, and that is what makes it a re-partition rather than a copy. These paths are
    one reducer's bucket, already assigned by the shuffle's hash; re-partitioning them with
    that same hash is inert whenever both counts are powers of two, because bucket assignment
    reads the low bits there — every row lands in `bucket & (n - 1)`, one sub-bucket, and the
    reduce pays a full write and re-read to change nothing. Both sides take the same salt, so
    equal keys still co-locate and each sub-bucket pair remains an independent join.
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
        for i, bucket in enumerate(
            nat.partition_batches_salted(batches, key_idx, n, _SUBBUCKET_SALT)
        ):
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


def _spill_side(
    nat, sub_ir, key_names, source, n_buckets, store, tag, engine_config, projection=None
):
    """Stream a source through its sub-plan, hash-partition by key, spill by tier.
    Returns a list of per-bucket `SpillHandle`s (None where a bucket received no
    rows). Buckets overflow local→remote through the shared tiered `store`."""
    writers = BucketWriters(store, f"{tag}_bucket")
    key_idx: list[int] | None = None

    for batch in _iter_spill_morsels(source, projection):
        rows = nat.execute_plan(sub_ir, [[batch]], engine_config)
        if not rows:
            continue
        if key_idx is None:
            key_idx = [rows[0].schema.get_field_index(k) for k in key_names]
        writers.add([rows] if n_buckets == 1 else nat.partition_batches(rows, key_idx, n_buckets))
    return writers.close_dense(n_buckets)


def _key_indices(schema: pa.Schema, names) -> list[int]:
    """Positions of the join key columns in a side's spilled schema."""
    return [schema.get_field_index(n) for n in names]


def _join_bucket_pair(
    nat, store, lh, rh, join_ir, cfg_json, left_schema, right_schema, key_idx, depth
):
    """Join one co-partitioned bucket pair, re-splitting it first if it does not fit.

    The bucket count is a constant, so under key skew one pair holds far more than its share
    — and both sides are read whole before the join, so a skewed pair would OOM at exactly
    the point spilling was meant to have prevented it.

    The **probe** side counts as much as the build side, which is why the larger of the two
    decides. The bucket count is sized from the build side alone, so a fact table with a hot
    key leaves a probe bucket orders of magnitude over the envelope even when every build
    bucket fits.

    Both sides are re-split with the *same* salt and count, so equal keys still co-locate and
    each sub-pair is an independent join whose union is the same relation.
    """
    biggest = lh if resident_bytes(lh) >= resident_bytes(rh) else rh
    if over_envelope(biggest, depth, max_depth=_MAX_JOIN_SPLIT_DEPTH):
        salt = split_salt(depth)
        lsub = _resplit_handle(nat, store, lh, key_idx[0], salt, f"jl_d{depth}")
        rsub = _resplit_handle(nat, store, rh, key_idx[1], salt, f"jr_d{depth}")
        for sb in range(_JOIN_SUB_BUCKETS):
            if lsub.get(sb) is None and rsub.get(sb) is None:
                continue
            yield from _join_bucket_pair(
                nat,
                store,
                lsub.get(sb),
                rsub.get(sb),
                join_ir,
                cfg_json,
                left_schema,
                right_schema,
                key_idx,
                depth + 1,
            )
        return

    left_b = read_reserved_bucket(store, lh) or [_empty_batch(left_schema)]
    right_b = read_reserved_bucket(store, rh) or [_empty_batch(right_schema)]
    try:
        for rb in nat.execute_plan(join_ir, [left_b, right_b], cfg_json):
            if rb.num_rows > 0:
                yield rb
    finally:
        # Equal keys hash to one bucket on both sides, so this pair will never be read again.
        # Releasing it here bounds peak scratch to the outstanding bucket pairs rather than to
        # both spilled sides in full.
        for handle in (lh, rh):
            if handle is not None:
                store.release(handle)


def _resplit_handle(nat, store, handle, key_idx, salt, tag) -> dict:
    """One side's over-large bucket, re-partitioned into salted sub-buckets on disk.

    A missing side is `{}` rather than an error: an outer join's bucket pair can have rows on
    only one side, and that pair still has to be split when the side it does have is too big.
    """
    if handle is None:
        return {}
    return regrace(
        nat, store, handle, key_idx, salt, f"{tag}_{id(handle):x}", n_sub=_JOIN_SUB_BUCKETS
    )
