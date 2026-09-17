"""End-to-end `Dataset.cache()`: a hit is identical to re-executing, never stale.

The cache is opt-in and keyed by the plan signature plus each input's object identity,
so it serves an equivalent prior result without re-running and never returns one
dataset's result for a different (even same-shaped) dataset.

The storage level chooses which media a result may occupy, and none of them may change
the answer: every level and every tier must return exactly what an uncached run returns.
That is the property most of the cases below assert, because it is the one a caching bug
breaks silently — a wrong cached answer looks like a correct answer that arrived fast.
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


# --- storage levels: where a result lives may never change what it is ---------------


@pytest.mark.parametrize("level", ["memory_only", "memory_and_disk", "disk_only", None])
def test_every_storage_level_matches_the_uncached_result(level):
    rows = {"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]}

    def query(ds):
        return ds.group_by("k").agg(s=bt.col("v").sum(), n=bt.col("v").count())

    expected = query(bt.from_pydict(rows)).collect().to_pydict()
    cached = query(bt.from_pydict(rows)).cache() if level is None else None
    if cached is None:
        cached = query(bt.from_pydict(rows)).cache(level)
    first = cached.collect().to_pydict()
    second = cached.collect().to_pydict()  # this one is served from the cache
    assert _sorted(first) == _sorted(expected)
    assert _sorted(second) == _sorted(expected)


def _sorted(d: dict) -> list[tuple]:
    """Row tuples in a stable order — a group-by result has no guaranteed row order."""
    return sorted(zip(*d.values(), strict=True))


def test_disk_only_holds_no_memory_but_still_serves():
    from batcher.carbonite.cache import current_result_cache

    ds = bt.from_pydict({"v": list(range(1000))}).cache("disk_only")
    assert ds.collect().num_rows == 1000
    cache = current_result_cache()
    assert cache is not None
    assert cache.used_bytes == 0, "DISK_ONLY must not charge the memory envelope"
    assert cache.stats()["disk_entries"] == 1
    assert ds.collect().num_rows == 1000  # served from disk
    assert cache.stats()["disk_hits"] >= 1


def test_an_unknown_storage_level_is_refused_at_the_api_edge():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="unknown storage level"):
        bt.from_pydict({"v": [1]}).cache("MEMORY_ONLY_SER")


def test_uncache_drops_the_entry_and_the_next_collect_recomputes():
    from batcher.carbonite.cache import current_result_cache

    ds = bt.from_pydict({"v": [1, 2, 3]}).cache()
    assert ds.collect().num_rows == 3
    cache = current_result_cache()
    assert cache is not None and len(cache) == 1
    ds.uncache()
    assert len(cache) == 0
    misses = cache.stats()["misses"]
    assert ds.collect().to_pydict() == {"v": [1, 2, 3]}  # recomputed, same answer
    assert cache.stats()["misses"] > misses


def test_uncache_on_an_uncached_dataset_is_a_no_op():
    ds = bt.from_pydict({"v": [1]})
    assert ds.uncache() is ds
    assert ds.collect().num_rows == 1


def test_unpersist_is_the_spark_spelling_of_uncache():
    from batcher.carbonite.cache import current_result_cache

    ds = bt.from_pydict({"v": [1, 2]}).cache("memory_only")
    ds.collect()
    cache = current_result_cache()
    assert cache is not None and len(cache) == 1
    ds.uncache()
    assert len(cache) == 0


def test_cache_stats_separates_a_hit_from_a_miss():
    # Deltas, not absolutes: the counters are lifetime figures for the process — the
    # reading that says whether the cache is earning its RAM — so `clear_cache()`
    # deliberately does not reset them, and an earlier test in this session has already
    # moved them.
    before = bt.cache_stats()
    ds = bt.from_pydict({"v": [1, 2, 3]}).cache()
    ds.collect()  # miss
    ds.collect()  # hit
    after = bt.cache_stats()
    assert after["hits"] - before["hits"] == 1
    assert after["misses"] - before["misses"] == 1


def test_clear_cache_empties_both_tiers():
    bt.from_pydict({"v": [1, 2]}).cache().collect()
    bt.from_pydict({"v": [3, 4]}).cache("disk_only").collect()
    assert bt.cache_stats()["entries"] + bt.cache_stats()["disk_entries"] > 0
    bt.clear_cache()
    assert bt.cache_stats()["entries"] == 0
    assert bt.cache_stats()["disk_entries"] == 0


# --- every terminal that reads a cached result must actually consult the cache -------
#
# Only `collect()` did. `to_pydict`, `to_pylist`, `to_pandas`, `to_polars`, `to_arrow`,
# `count`, `is_empty` and `iter_batches` all re-executed the whole plan on a dataset the
# user had explicitly cached, and recorded neither a hit nor a miss while doing it — so
# `cache_stats()` showed a cache that looked idle rather than one that was being bypassed.


def _hits(fn) -> int:
    """How many cache hits `fn()` records."""
    before = bt.cache_stats()["hits"]
    fn()
    return bt.cache_stats()["hits"] - before


@pytest.fixture
def warm():
    """A cached dataset whose result has already been computed and stored."""
    ds = (
        bt.from_pydict({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
        .cache()
    )
    ds.collect()
    return ds


@pytest.mark.parametrize(
    "terminal",
    [
        "collect",
        "to_arrow",
        "to_pydict",
        "to_pylist",
        "count",
        "is_empty",
    ],
)
def test_every_materializing_terminal_serves_from_the_cache(warm, terminal):
    assert _hits(getattr(warm, terminal)) == 1


@pytest.mark.parametrize(("terminal", "module"), [("to_pandas", "pandas"), ("to_polars", "polars")])
def test_framework_conversions_serve_from_the_cache(warm, terminal, module):
    pytest.importorskip(module)
    assert _hits(getattr(warm, terminal)) == 1


def test_count_from_the_cache_matches_an_uncached_count(warm):
    rows = {"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]}
    plain = bt.from_pydict(rows).group_by("k").agg(s=bt.col("v").sum())
    assert warm.count() == plain.count()
    assert warm.is_empty() == plain.is_empty()


def test_iter_batches_streams_a_warm_cache(warm):
    before = bt.cache_stats()["hits"]
    streamed = sum(batch.num_rows for batch in warm.iter_batches())
    assert bt.cache_stats()["hits"] - before == 1
    assert streamed == warm.collect().num_rows


def test_iter_batches_honors_batch_size_on_a_cache_hit(warm):
    sizes = [batch.num_rows for batch in warm.iter_batches(batch_size=2)]
    assert sum(sizes) == 3
    assert max(sizes) <= 2


def test_iter_batches_does_not_populate_the_cache():
    # Filling the cache from a stream means materializing the whole result, which is the
    # one thing a caller reaching for iter_batches has asked not to happen.
    bt.clear_cache()
    ds = bt.from_pydict({"v": list(range(100))}).filter(bt.col("v") > 10).cache()
    assert sum(b.num_rows for b in ds.iter_batches()) == 89
    assert bt.cache_stats()["entries"] == 0


def test_an_uncached_dataset_records_no_miss_on_count():
    # The probe is guarded on the marker, not run unconditionally: a query that never
    # asked to be cached must not drive the hit rate toward zero.
    before = bt.cache_stats()
    bt.from_pydict({"v": [1, 2, 3]}).count()
    after = bt.cache_stats()
    assert after["misses"] == before["misses"]
    assert after["hits"] == before["hits"]
