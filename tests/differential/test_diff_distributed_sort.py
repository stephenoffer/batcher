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


#: Above `run_sort::MIN_ROWS` (4,096) per partition, so each worker's own sort reaches the
#: natural-run detection rather than falling under its floor.
_RUN_ROWS = 60_000


def _run_structured(kind: str) -> pa.table:
    """A key column with a named run structure, plus the input position as a payload."""
    if kind == "sorted":
        keys = sorted((i * 7919) % 1_000_003 for i in range(_RUN_ROWS))
    elif kind == "runs":
        part = _RUN_ROWS // 8
        keys = []
        for p in range(8):
            keys += sorted((i * 7919 + p * 13) % 1_000_003 for i in range(part))
    elif kind == "descending":
        keys = sorted(((i * 7919) % 1_000_003 for i in range(_RUN_ROWS)), reverse=True)
    else:  # pragma: no cover - the parametrization covers the three above
        raise ValueError(kind)
    return pa.table({"k": keys, "p": list(range(len(keys)))})


@pytest.mark.parametrize("kind", ["sorted", "runs", "descending"])
@pytest.mark.parametrize("descending", [False, True])
def test_distributed_run_structured_sort_equals_single_node(kind, descending):
    """Natural-run detection must not make a partitioned sort disagree with a whole one.

    `ops::run_sort` merges the key's ordered runs instead of radix-sorting them, and it sees
    whatever slice of the relation it is handed — the whole input on one node, one partition's
    share on many. Those are different run structures over the same rows, so the two paths
    take genuinely different routes to what must be the identical permutation. That is exactly
    the class of bug invariant #7 exists for, and an order-independent comparison cannot see
    it, so this compares the row sequence.
    """
    ds = bt.from_arrow(_run_structured(kind)).sort("k", descending=descending)
    single = _rows(ds.collect().to_pydict())
    multi = _rows(ds.collect(distributed=True, num_workers=3).to_pydict())
    assert single == multi
