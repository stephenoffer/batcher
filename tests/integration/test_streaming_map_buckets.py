"""Hashing a join's map side a chunk at a time must equal hashing it all at once.

The join map task used to read its whole partition, run the map prefix over it, and hash the
entire result — holding the partition, the full mapped output, and the second complete copy
`partition_batches` gathers into, all at the same time. `memory_budget_bytes` does not cover
any of that: it bounds allocations *inside* `execute_plan`, not what the worker holds around
it. On TPC-H sf100 that is a quarter of a 600M-row `lineitem` on a 30 GB node, and q9
OOM-killed two workers rather than spilling.

`streaming_map_buckets` walks the partition in byte-bounded chunks instead, exactly as the
aggregate map side already did. It is correct because the map prefix is breaker-free — applying
it per chunk equals applying it to the whole partition — and because hashing on the same keys
sends a row to the same bucket whichever chunk carried it. These cases pin that equality,
including the chunk boundaries and the empty cases where an off-by-one would hide.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def _sub_ir(ds) -> str:
    """The plan IR for a map prefix reading source 0."""
    from batcher.dist.executor import _relabel_single_source

    plan, _sid = _relabel_single_source(ds._plan)
    return json.dumps(plan.to_ir())


def _table(n: int, keys: int, seed: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table(
        {
            "k": rng.integers(0, keys, n).astype("int64"),
            "v": rng.integers(0, 1000, n).astype("int64"),
        }
    )


def _rows(buckets: list[list]) -> list[list[tuple]]:
    """Each bucket's rows, order-independent within the bucket."""
    out = []
    for b in buckets:
        rows: list[tuple] = []
        for batch in b:
            cols = batch.to_pydict()
            rows.extend(zip(*cols.values(), strict=True))
        out.append(sorted(rows))
    return out


def _whole(nat, sub_ir, keys, batches, n_buckets, cfg) -> list[list]:
    """What the map task did before: one `execute_plan`, one `partition_batches`."""
    mapped = nat.execute_plan(sub_ir, [list(batches)], cfg)
    if not mapped:
        return [[] for _ in range(n_buckets)]
    if n_buckets == 1:
        return [mapped]
    idx = [mapped[0].schema.get_field_index(k) for k in keys]
    return list(nat.partition_batches(mapped, idx, n_buckets))


@pytest.mark.parametrize("n_buckets", [1, 4, 7])
@pytest.mark.parametrize("chunk_bytes", [1, 4096, 1 << 30])
def test_chunked_bucketing_equals_whole_partition_bucketing(n_buckets, chunk_bytes):
    """The invariant. `chunk_bytes=1` forces a chunk per batch, `1 << 30` forces a single
    chunk, so the two extremes and a boundary in between all have to agree."""
    import batcher._native as nat

    from batcher.config import active_config
    from batcher.dist.executors.partition_io import streaming_map_buckets

    table = _table(20_000, keys=37)
    batches = table.to_batches(max_chunksize=1000)
    ds = bt.from_arrow(table).filter(bt.col("v") > 100)
    sub_ir, cfg = _sub_ir(ds), active_config().engine_config_json()

    got = streaming_map_buckets(
        nat, sub_ir, ["k"], iter(batches), n_buckets, cfg, chunk_bytes=chunk_bytes
    )
    assert len(got) == n_buckets, "every bucket is published, empty ones included"
    assert _rows(got) == _rows(_whole(nat, sub_ir, ["k"], batches, n_buckets, cfg))


def test_an_empty_partition_still_yields_every_bucket():
    """A reducer's failed fetch must mean a lost worker, never an empty bucket — so an empty
    partition publishes empties rather than nothing."""
    import batcher._native as nat

    from batcher.config import active_config
    from batcher.dist.executors.partition_io import streaming_map_buckets

    ds = bt.from_arrow(_table(10, keys=3)).filter(bt.col("v") > 10_000)  # matches nothing
    got = streaming_map_buckets(
        nat, _sub_ir(ds), ["k"], iter([]), 5, active_config().engine_config_json()
    )
    assert got == [[], [], [], [], []]


def test_rows_land_in_the_bucket_their_key_hashes_to():
    """The property the reduce depends on: one key never splits across two buckets, however
    the partition was chunked."""
    import batcher._native as nat

    from batcher.config import active_config
    from batcher.dist.executors.partition_io import streaming_map_buckets

    table = _table(5_000, keys=11, seed=3)
    ds = bt.from_arrow(table)
    got = streaming_map_buckets(
        nat,
        _sub_ir(ds),
        ["k"],
        iter(table.to_batches(max_chunksize=97)),
        6,
        active_config().engine_config_json(),
        chunk_bytes=1,
    )
    seen: dict[int, int] = {}
    for i, rows in enumerate(_rows(got)):
        for key, _v in rows:
            assert seen.setdefault(key, i) == i, f"key {key} landed in two buckets"
