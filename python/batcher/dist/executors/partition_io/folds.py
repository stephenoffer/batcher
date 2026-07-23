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
        size += b.nbytes
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
        size += b.nbytes
        if size >= chunk_bytes:
            fold(chunk)
            chunk, size = [], 0
    if chunk:
        fold(chunk)
    if not running:  # empty partition → an empty, schema-bearing result
        return nat.execute_plan(local_ir, [[]], engine_config)
    return running
