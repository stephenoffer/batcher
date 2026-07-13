"""The optimizer's plan memo returns a plan only for the query it was built for.

Reusing an optimized plan is safe exactly as far as the key is: Kyber's rewrites are
semantics-preserving, so a stale plan is a slower plan — but a plan keyed by something that
*ignores* part of the query is a wrong answer. Two of those traps are live here:
`plan_signature` normalizes literals (so it must never key this cache), and an in-memory
source's `identity()` is only its shape (so two different relations collide on it, and
zone-map pruning reads a source's real bounds).

These tests pin the key, the invalidation, and the disable switch.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config, config_context
from batcher.io.source import InMemorySource
from batcher.kyber import learning, plan_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_cache():
    plan_cache.clear()
    yield
    plan_cache.clear()


def _source(rows: list[int]) -> InMemorySource:
    return InMemorySource(pa.table({"x": pa.array(rows, type=pa.int64())}).to_batches())


def _plan_ir(predicate_literal: int) -> str:
    """The plan's content fingerprint (what `cache_key` now keys on) for `x > literal`."""
    table = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
    return bt.from_arrow(table).filter(bt.col("x") > predicate_literal)._plan.content_key()


def _key(plan_key, sources, hub=None, config=None, kind="full"):
    return plan_cache.cache_key(plan_key, sources, config or active_config(), hub, kind)


# --- what the key must separate ----------------------------------------------


def test_a_different_literal_is_a_different_key():
    """The trap: `plan_signature` normalizes literals, so it could never key this cache.
    A plan built for `x > 1` must never be served to `x > 2`."""
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src) != _key(_plan_ir(2), src)


def test_the_same_query_over_the_same_source_is_the_same_key():
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src) == _key(_plan_ir(1), src)


def test_two_in_memory_sources_of_the_same_shape_do_not_collide():
    """`InMemorySource.identity()` is schema + row count, so different data shares it. Zone-map
    pruning reads the real bounds, so a shared plan could be a wrong answer, not a slow one."""
    a, b = _source([1, 2, 3]), _source([100, 200, 300])
    assert a.identity() == b.identity()  # the collision this key must survive
    assert _key(_plan_ir(1), [a]) != _key(_plan_ir(1), [b])


def test_a_different_optimizer_config_is_a_different_key():
    src = [_source([1, 2, 3])]
    base = active_config()
    tweaked = base.replace(optimizer=dataclasses.replace(base.optimizer, plan_cache_entries=8))
    assert _key(_plan_ir(1), src, config=base) != _key(_plan_ir(1), src, config=tweaked)


def test_a_different_hub_is_a_different_key():
    """Resetting the MetadataHub must invalidate every plan learned against the old one."""
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src, hub=object()) != _key(_plan_ir(1), src, hub=object())


def test_the_two_optimizer_entry_points_do_not_collide():
    """`optimize_full` and `optimize_logical` are pure in the same inputs but return
    different shapes. Sharing the memo means the key must separate them, or a caller
    asking for a physical plan is handed a logical one."""
    src = [_source([1, 2, 3])]
    assert _key(_plan_ir(1), src, kind="full") != _key(_plan_ir(1), src, kind="logical")


def test_an_unkeyable_source_is_never_cached():
    class NoIdentity:
        pass

    assert _key(_plan_ir(1), [NoIdentity()]) is None


# --- invalidation -------------------------------------------------------------


def test_learning_something_plan_relevant_invalidates():
    """The generation advances only for a *material* correction, and the key follows it."""
    src = [_source([1, 2, 3])]
    before = _key(_plan_ir(1), src)
    learning._bump_generation()
    assert _key(_plan_ir(1), src) != before


def test_a_settled_estimate_does_not_invalidate():
    """A smoothed average drifting in its fourth decimal must not throw the plan away —
    that is why fingerprinting the stats' *content* never hits."""
    assert not learning._is_material(1000.0, 1001.0)  # 0.1%
    assert learning._is_material(1000.0, 1200.0)  # 20%
    assert learning._is_material(None, 5.0)  # nothing was known
    assert learning._is_material(0.0, 5.0)  # a provably-empty prior, now non-empty


# --- the store ----------------------------------------------------------------


def test_store_and_lookup_round_trip():
    src = [_source([1, 2, 3])]
    key = _key(_plan_ir(1), src)
    assert plan_cache.lookup(key) is None
    plan_cache.store(key, "plan", src, max_entries=4)
    assert plan_cache.lookup(key) == "plan"


def test_a_zero_cap_disables_the_cache():
    src = [_source([1, 2, 3])]
    key = _key(_plan_ir(1), src)
    plan_cache.store(key, "plan", src, max_entries=0)
    assert plan_cache.lookup(key) is None


def test_the_cap_evicts_least_recently_used():
    sources = [[_source([i])] for i in range(3)]
    keys = [_key(_plan_ir(1), s) for s in sources]
    for k, s in zip(keys, sources, strict=True):
        plan_cache.store(k, "plan", s, max_entries=2)
    assert plan_cache.lookup(keys[0]) is None  # evicted
    assert plan_cache.lookup(keys[1]) == "plan"
    assert plan_cache.lookup(keys[2]) == "plan"


def test_a_hit_refreshes_its_lru_position():
    sources = [[_source([i])] for i in range(3)]
    keys = [_key(_plan_ir(1), s) for s in sources]
    plan_cache.store(keys[0], "a", sources[0], max_entries=2)
    plan_cache.store(keys[1], "b", sources[1], max_entries=2)
    assert plan_cache.lookup(keys[0]) == "a"  # touch the oldest
    plan_cache.store(keys[2], "c", sources[2], max_entries=2)
    assert plan_cache.lookup(keys[1]) is None  # `b` was the least recently used
    assert plan_cache.lookup(keys[0]) == "a"


# --- end to end ---------------------------------------------------------------


def test_a_stage_boundary_source_reports_rows_without_scanning_for_bounds():
    """The adaptive executor wraps each stage's output in a throwaway source. Its exact
    row count is what re-optimization reads; the O(rows) min/max pass would be recomputed
    and discarded every run, so it must not happen."""
    from batcher.plan.stats import Provenance

    batches = pa.table({"x": pa.array([5, 1, 9], type=pa.int64())}).to_batches()
    ephemeral = InMemorySource(batches, zone_maps=False)
    assert ephemeral.statistics().row_count == 3
    assert ephemeral.statistics().columns == {}

    registered = InMemorySource(batches)  # a user table is queried again: bounds pay off
    stats = registered.statistics()
    assert stats.row_count == 3
    assert (stats.columns["x"].min, stats.columns["x"].max) == (1, 9)
    assert stats.columns["x"].provenance is Provenance.EXACT


def test_caching_never_changes_a_result_across_literals():
    """The whole point: run the same shape with different literals, cache on, and every
    answer must be the one that query asks for."""
    table = pa.table({"x": pa.array([1, 2, 3, 4, 5], type=pa.int64())})
    dataset = bt.from_arrow(table)
    for threshold in (0, 2, 4, 2, 0):
        got = dataset.filter(bt.col("x") > threshold).collect().to_pydict()["x"]
        assert got == [v for v in [1, 2, 3, 4, 5] if v > threshold]


def test_disabling_the_cache_gives_the_same_results():
    table = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
    base = active_config()
    off = base.replace(optimizer=dataclasses.replace(base.optimizer, plan_cache_entries=0))
    with config_context(off):
        first = bt.from_arrow(table).filter(bt.col("x") > 1).collect().to_pydict()
    second = bt.from_arrow(table).filter(bt.col("x") > 1).collect().to_pydict()
    assert first == second == {"x": [2, 3]}
