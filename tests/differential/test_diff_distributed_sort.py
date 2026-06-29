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
