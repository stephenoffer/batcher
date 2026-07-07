"""Distributed sort / top-N equivalence: single-node == multi-partition.

A small ``ORDER BY ... LIMIT k`` distributes as a shuffle-free mergeable top-N (each
worker's local top-N, merged on the driver); a large/absent limit takes the full
range-partition sort. Both must produce exactly the single-node ordering.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_T = pa.table({"k": (list(range(5000)) * 3)[::-1], "v": list(range(15000))})


def _rows(d: dict) -> list[tuple]:
    return [tuple(r) for r in zip(*d.values(), strict=True)]


def test_distributed_topn_equals_single_node():
    ds = bt.from_arrow(_T).sort("k", descending=True).limit(20)
    single = _rows(ds.collect().to_pydict())
    multi = _rows(ds.collect(distributed=True, num_workers=3).to_pydict())
    assert single == multi


def test_distributed_topn_ascending_with_ties_equals_single_node():
    ds = bt.from_arrow(_T).sort("k").limit(37)
    single_keys = ds.collect().to_pydict()["k"]
    multi_keys = ds.collect(distributed=True, num_workers=4).to_pydict()["k"]
    assert single_keys == multi_keys  # leading-key order identical (ties may reorder v)


def test_distributed_topn_reuses_fleet_across_changing_worker_counts():
    # The session fleet is sized on first use and reused; a later top-N with a DIFFERENT
    # num_workers must still be correct. The per-worker partitioning has to follow the
    # fleet's actual size, or parts/actors mismatch — a larger fleet indexes past the
    # partitions (crash), a smaller one silently drops the tail partitions' rows.
    ds = bt.from_arrow(_T).sort("k").limit(37)
    single = ds.collect().to_pydict()["k"]
    for nw in (4, 4, 3, 4, 2, 5):
        multi = ds.collect(distributed=True, num_workers=nw).to_pydict()["k"]
        assert multi == single, f"num_workers={nw} disagreed with single-node"
