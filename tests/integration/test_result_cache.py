"""End-to-end `Dataset.cache()`: a hit is identical to re-executing, never stale.

The cache is opt-in and keyed by the plan signature plus each input's object identity,
so it serves an equivalent prior result without re-running and never returns one
dataset's result for a different (even same-shaped) dataset.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(autouse=True)
def _fresh_cache():
    from batcher.carbonite.cache import current_result_cache

    c = current_result_cache()
    if c is not None:
        c.clear()
    yield
    if c is not None:
        c.clear()


def test_cache_hit_matches_first_result():
    ds = (
        bt.from_pydict({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
        .cache()
    )
    first = ds.collect().to_pydict()
    second = ds.collect().to_pydict()  # served from the cache
    assert first == second


def test_cache_matches_uncached_result():
    base = bt.from_pydict({"k": [1, 1, 2], "v": [10, 20, 30]})
    uncached = base.group_by("k").agg(s=bt.col("v").sum()).collect().to_pydict()
    cached = base.group_by("k").agg(s=bt.col("v").sum()).cache().collect().to_pydict()
    assert cached == uncached


def test_no_false_hit_for_same_shape_different_data():
    # Both inputs share schema + row count, so their shape-based `identity()` is equal;
    # the cache must still keep them distinct (object-identity key) — no stale result.
    a = bt.from_pydict({"v": [1, 2, 3]}).cache()
    b = bt.from_pydict({"v": [4, 5, 6]}).cache()
    assert a.collect().to_pydict() == {"v": [1, 2, 3]}
    assert b.collect().to_pydict() == {"v": [4, 5, 6]}  # not a's cached result


def test_cache_actually_stores_an_entry():
    from batcher.carbonite.cache import current_result_cache

    ds = bt.from_pydict({"v": [1, 2, 3]}).cache()
    ds.collect()
    cache = current_result_cache()
    assert cache is not None and cache.used_bytes > 0


def test_distributed_cache_shares_key_with_single_node():
    # A cached relational result is identical single-node vs distributed (mergeable
    # algebra), so they share one cache entry: caching one way serves the other.
    pytest.importorskip("ray", reason="ray not installed")
    ds = (
        bt.from_pydict({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
        .cache()
    )
    single = ds.collect().to_pydict()  # populates the cache
    distrib = ds.collect(distributed=True, num_workers=2).to_pydict()  # served from cache
    assert single == distrib
