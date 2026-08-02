"""Streaming, byte-bounded folds of a shuffle map-side partition.

A shuffle map task must not hold its whole partition in memory just to reduce it: at scale a
partition is ~one node's share of the input (~125M rows at 1B), and materializing it before
the reduce is the #1 distributed memory peak — the difference between a bounded worker and an
OOM-killed actor. Each fold here walks the partition's batches a `_FOLD_CHUNK_BYTES` chunk at
a time, keeping only the running reduced state, so peak memory is one chunk plus that state.

Both folds are correct by the *mergeable* invariant: the reduced state is combined
chunk-by-chunk, and combine-of-per-chunk-states equals one state over the whole partition
(the map prefix is breaker-free, so per-chunk application matches whole-partition application).
"""

from __future__ import annotations

import os

from batcher.plan.types import retained_bytes

_FOLD_CHUNK_BYTES = max(1 << 20, int(os.environ.get("BATCHER_FOLD_CHUNK_BYTES", str(256 << 20))))


def streaming_partial_aggregate(
    nat, map_ir, gk, aj, batches, engine_config, chunk_bytes=_FOLD_CHUNK_BYTES
):
    """Fold a partition's batches through the (breaker-free) map prefix + partial aggregate
    into one running partial, a byte-bounded chunk at a time.

    The map side never holds the whole partition or the whole mapped output: peak is one chunk
    plus the running partial. `combine` of the per-chunk partials equals one partial over the
    whole partition (see the module note).
    """
    running = None
    chunk: list = []
    size = 0

    def fold(rows):
        nonlocal running
        mapped = nat.execute_plan(map_ir, [rows], engine_config)
        partial = nat.partial_aggregate(gk, aj, mapped)
        running = partial if running is None else nat.combine(gk, aj, [running, partial])

    for b in batches:
        chunk.append(b)
        size += retained_bytes(b)
        if size >= chunk_bytes:
            fold(chunk)
            chunk, size = [], 0
    if chunk:
        fold(chunk)
    if running is None:  # empty partition → the empty (schema-bearing) partial
        running = nat.partial_aggregate(gk, aj, nat.execute_plan(map_ir, [[]], engine_config))
    return running


def streaming_topn(nat, local_ir, merge_ir, batches, engine_config, chunk_bytes=_FOLD_CHUNK_BYTES):
    """Fold a partition through the map prefix + top-N heap into one running top-N, a
    byte-bounded chunk at a time.

    A top-N is mergeable (top-k of a union is the top-k of the per-part top-ks), so the worker
    never needs the whole partition — peak is one chunk plus `k` rows. Reading the split whole
    made a worker hold ~125M rows at 1B scale to pick 100: ~25 GB, an OOM-killed actor, 130 s
    for a 100-row answer. `local_ir` is map prefix + sort + limit over the raw split; `merge_ir`
    is the same sort + limit over the projected schema — the driver's merge plan, one chunk
    earlier.
    """
    running: list = []
    chunk: list = []
    size = 0

    def fold(rows):
        nonlocal running
        top = nat.execute_plan(local_ir, [rows], engine_config)
        candidates = running + [b for b in top if b.num_rows > 0]
        running = (
            list(nat.execute_plan(merge_ir, [candidates], engine_config)) if candidates else []
        )

    for b in batches:
        chunk.append(b)
        size += retained_bytes(b)
        if size >= chunk_bytes:
            fold(chunk)
            chunk, size = [], 0
    if chunk:
        fold(chunk)
    if not running:  # empty partition → an empty, schema-bearing result
        return nat.execute_plan(local_ir, [[]], engine_config)
    return running


def streaming_map_buckets(
    nat, sub_ir, key_names, batches, n_buckets, engine_config, chunk_bytes=_FOLD_CHUNK_BYTES
):
    """Run a join's map prefix over a partition and hash it into `n_buckets`, a chunk at a time.

    The bucketing counterpart of `streaming_partial_aggregate`, and it exists for the same
    reason. The aggregate map side was taught to stream because materializing a whole partition
    is the largest distributed memory peak there is; the *join* map side was not, and it is the
    one that hurts more — it held the partition, the whole mapped output, and then a second full
    copy of that output, because `partition_batches` gathers into fresh buffers rather than
    aliasing its input. Measured on TPC-H sf100, that is a quarter of a 600M-row `lineitem`
    landing on a 30 GB node: q9 OOM-killed two workers instead of spilling.

    Streaming removes the two terms that scale with the partition. Peak is now one chunk, that
    chunk's mapped output, and the accumulating buckets — and the buckets are what has to exist
    regardless, since they are the mapper's actual result.

    Correct for the same reason the fold is: the map prefix is breaker-free, so applying it per
    chunk equals applying it to the whole partition, and hashing on the same keys sends a row to
    the same bucket whichever chunk it arrived in.

    That precondition is not an assumption — it is enforced before the query reaches here.
    `executor._join_sides_are_map_only` refuses the distributed join path for a side carrying a
    breaker (that shape is `requires_staging`, and the inner breaker runs as its own stage
    first), precisely because a breaker evaluated per *partition* already returns wrong answers:
    `limit(5).join(dim)` kept five rows on each of four workers and returned twenty. Chunking
    within a partition is safe exactly where partitioning already is.

    Args:
        nat: The native engine module.
        sub_ir: The map prefix's plan IR, as JSON.
        key_names: The join key columns to hash on.
        batches: The partition's batches, as an iterator (not a materialized list).
        n_buckets: How many hash buckets to produce.
        engine_config: The worker's engine config JSON.
        chunk_bytes: Bytes of input to accumulate before running the prefix over them.

    Returns:
        `n_buckets` lists of `RecordBatch`, empty ones included, so a reducer's failed fetch
        still means a lost worker rather than an empty bucket.
    """
    buckets: list[list] = [[] for _ in range(n_buckets)]
    chunk: list = []
    size = 0
    key_idx: list[int] = []

    def run(rows):
        nonlocal key_idx
        mapped = nat.execute_plan(sub_ir, [rows], engine_config)
        if not mapped:
            return
        if n_buckets == 1:
            buckets[0].extend(mapped)
            return
        if not key_idx:
            key_idx = [mapped[0].schema.get_field_index(k) for k in key_names]
        for i, part in enumerate(nat.partition_batches(mapped, key_idx, n_buckets)):
            buckets[i].extend(part)

    for b in batches:
        chunk.append(b)
        size += retained_bytes(b)
        if size >= chunk_bytes:
            run(chunk)
            chunk, size = [], 0
    if chunk:
        run(chunk)
    return buckets
