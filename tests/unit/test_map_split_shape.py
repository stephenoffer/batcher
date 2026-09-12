"""The map/UDF route's split SHAPE must not be driven by its compute-derived partition count.

`_scan_splits` coalesces adjacent Parquet row-groups up to `_SPLIT_TARGET_BYTES`, but only
while the coalesced result still holds `workers x _SCAN_PREFETCH` splits — the guard that
stops a *small* dataset collapsing below the fan-out. `workers` there is the value the caller
passes `partition_descriptors` as its worker count, so a route that expresses "cut this into
465 partitions" by passing 465 sets the floor at 14,880 splits, which no ordinary read meets,
and coalescing is silently refused.

Measured on TPC-H sf100 `lineitem`, same query and projection: the map route planned **4,902**
splits where `flight_aggregate` — which passes the fleet width and asks for its partition count
via `max_partitions` — planned **2,494**. Twice the object-store requests for the same bytes.

These pin the idiom rather than the numbers: the descriptor count must come out the same, and
the split count must not grow with it.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.dist.executors.partition_io import _sources as psrc
from batcher.dist.executors.partition_io import partition_descriptors

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus(tmp_path):
    """Seventy small Parquet files of six row-groups each.

    Seventy so the *coalesced* plan (one split per file at a 64 MB target) still clears
    `_scan_splits`' real floor of ``workers x _SCAN_PREFETCH`` at a two-worker fleet — 64.
    That is what lets this run against the production constants instead of monkeypatching
    them, which matters: the constants are the thing under test.
    """
    for f in range(70):
        table = pa.table(
            {"i": pa.array(np.arange(f * 600, (f + 1) * 600)), "v": pa.array(np.random.rand(600))}
        )
        pq.write_table(table, tmp_path / f"p{f:03d}.parquet", row_group_size=100)
    import batcher as bt

    return bt.read.parquet(f"{tmp_path}/*.parquet")._sources[0]


def _splits(descriptors) -> int:
    return sum(len(d.get("splits", [])) for d in descriptors)


def test_max_partitions_keeps_the_count_and_coalesces_the_reads(corpus):
    """The whole point: the same partitions, a fraction of the object-store reads."""
    naive = partition_descriptors(corpus, 70, projection=["i"])
    idiomatic = partition_descriptors(corpus, 2, projection=["i"], max_partitions=70)

    assert len(idiomatic) == len(naive) == 70, "the partition count must be unchanged"
    assert _splits(idiomatic) < _splits(naive), (
        f"max_partitions must let the read coalesce: "
        f"{_splits(idiomatic)} vs {_splits(naive)} splits"
    )
    # Every row-group still reachable exactly once, whichever shape was planned.
    assert _splits(naive) == 70 * 6
    assert _splits(idiomatic) == 70


def test_a_partition_count_passed_as_workers_defeats_coalescing(corpus):
    """The failure mode itself, so the fix cannot be reverted quietly.

    `_scan_splits` refuses to coalesce unless the result still holds
    ``workers x _SCAN_PREFETCH`` splits. Passing a partition count there raises that floor
    with the partition count, so the larger the fan-out the more certain it is that the read
    stays fine-grained — the opposite of what a wide read wants.
    """
    assert len(psrc._scan_splits(corpus, 2, None, ["i"])) == 70, "a 2-wide fleet coalesces"
    assert len(psrc._scan_splits(corpus, 70, None, ["i"])) == 70 * 6, "a 70-wide one cannot"


def test_the_map_route_asks_for_partitions_with_max_partitions(monkeypatch):
    """`_distributed_map_aggregate` must pass the fleet width, not its partition count.

    A call-shape assertion, deliberately: the difference this guards is invisible in the
    result (same rows, same partition count) and shows up only as twice the object-store
    requests, so there is nothing else to assert on.
    """
    from batcher.dist.executors import map as mapmod

    seen = {}

    def _spy(source, workers, **kw):
        seen["workers"] = workers
        seen["max_partitions"] = kw.get("max_partitions")
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(mapmod, "partition_descriptors", _spy)
    monkeypatch.setattr(mapmod, "_adaptive_partition_count", lambda *a, **k: 465)
    monkeypatch.setattr(mapmod, "_ensure_ray", lambda *a, **k: None)

    import batcher as bt

    ds = (
        bt.from_pydict({"a": [1, 2, 3]})
        .map_batches(lambda b: b, output_columns=["a"])
        .agg(s=bt.col("a").sum())
    )
    agg = ds._plan
    with pytest.raises(_Stop):
        mapmod._distributed_map_aggregate(None, agg, ds._sources, 64)

    assert seen["max_partitions"] == 465, "the partition count belongs in max_partitions"
    assert seen["workers"] == 64, "the worker count must be the fleet width, not the partitions"
