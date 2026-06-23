"""Distributed execution under a tight memory envelope spills on the workers and
still produces the single-node result.

This is the distributed arm of the "Carbonite protects against OOM" invariant: the
grace-spill engine is single-node-complete, and `engine_config_json` now folds the
per-task memory grant into every worker's `execute_plan`, so a reducer bucket that
exceeds its share spills to disk instead of OOMing. A small `max_memory_bytes` cap
forces the workers onto the spill path here; the results must be bit-for-bit equal
to an unbounded single-node run regardless.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _need_ray():
    pytest.importorskip("ray")


# A cap small enough that a few-thousand-row build side / group table exceeds it,
# forcing the worker `execute_plan` onto the grace-spill path.
_TIGHT_MEMORY = Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 16))


def _sort_key(rows: list[dict], keys: tuple[str, ...]) -> list[tuple]:
    return sorted(tuple(r[k] for k in keys) for r in rows)


def test_distributed_aggregate_spills_and_matches_single_node():
    rng = np.random.default_rng(0)
    t = pa.table(
        {
            "k": rng.integers(0, 4000, 20000).astype("int64"),
            "v": rng.integers(0, 100, 20000).astype("int64"),
        }
    )
    ds = bt.from_arrow(t).group_by("k").agg(n=count(), s=col("v").sum())
    single = ds.collect().to_pylist()
    with config_context(_TIGHT_MEMORY):
        dist = ds.collect(distributed=True, num_workers=3).to_pylist()
    assert _sort_key(dist, ("k",)) == _sort_key(single, ("k",))


def test_distributed_join_spills_and_matches_single_node():
    rng = np.random.default_rng(1)
    left = pa.table({"id": rng.integers(0, 5000, 12000).astype("int64"), "v": np.arange(12000)})
    right = pa.table({"id": rng.integers(0, 5000, 12000).astype("int64"), "w": np.arange(12000)})
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="id")
    single = ds.collect().to_pylist()
    with config_context(_TIGHT_MEMORY):
        dist = ds.collect(distributed=True, num_workers=3).to_pylist()
    assert _sort_key(dist, ("id", "v", "w")) == _sort_key(single, ("id", "v", "w"))


def test_distributed_sort_spills_and_matches_single_node():
    rng = np.random.default_rng(2)
    t = pa.table({"v": rng.integers(0, 1_000_000, 20000).astype("int64")})
    ds = bt.from_arrow(t).sort("v")
    single = ds.collect().to_pylist()
    with config_context(_TIGHT_MEMORY):
        dist = ds.collect(distributed=True, num_workers=3).to_pylist()
    # Sort is order-significant: compare the sequences directly.
    assert [r["v"] for r in dist] == [r["v"] for r in single]


def test_distributed_skewed_join_spills_and_matches_single_node():
    # One hot key holds most rows, so its reducer bucket dwarfs the per-task share —
    # exactly the OOM case. Under the tight cap that bucket must spill, not die.
    rng = np.random.default_rng(3)
    ids = np.concatenate(
        [np.zeros(15000, dtype="int64"), rng.integers(1, 3000, 5000).astype("int64")]
    )
    left = pa.table({"id": ids, "v": np.arange(ids.size)})
    right = pa.table({"id": np.arange(3000, dtype="int64"), "w": np.arange(3000)})
    ds = bt.from_arrow(left).join(bt.from_arrow(right), on="id")
    single = ds.collect().to_pylist()
    with config_context(_TIGHT_MEMORY):
        dist = ds.collect(distributed=True, num_workers=3).to_pylist()
    assert _sort_key(dist, ("id", "v", "w")) == _sort_key(single, ("id", "v", "w"))
