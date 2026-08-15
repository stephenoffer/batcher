"""A materialized subplan names how it was derived, so its plan can be memoized.

`api.subplan_reuse` executes a repeated subtree once and binds the result as an
`InMemorySource`. That source is `ephemeral` — it exists for one execution — which used to
mean `kyber.plan_cache` refused to cache anything built over it, so the *second* optimize of
such a query ran in full on every run forever. A `derivation` key (the subplan's content key
over its inputs' own keys) is stable across runs by construction, and these tests pin both
halves of that: it recurs when the derivation is the same, and it does not when it differs.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.io.source import InMemorySource
from batcher.kyber import plan_cache
from batcher.plan.source_stats import source_stats_key


def _src(**kwargs) -> InMemorySource:
    return InMemorySource([pa.record_batch({"x": [1, 2, 3]})], **kwargs)


def test_an_ephemeral_source_without_a_derivation_is_still_unkeyable():
    """The rule this is an exception to: an `id()`-keyed relation can never be read back."""
    assert plan_cache._source_keys([_src(ephemeral=True)]) is None


def test_a_derivation_makes_an_ephemeral_source_cacheable():
    keys = plan_cache._source_keys([_src(ephemeral=True, derivation="abc123")])
    assert keys is not None and keys[0].endswith("derived:abc123")


def test_the_same_derivation_keys_the_same_way_across_objects():
    """Two objects, one derivation — which is the whole point: the next run rebuilds it."""
    first = source_stats_key(_src(ephemeral=True, derivation="abc123"))
    second = source_stats_key(_src(ephemeral=True, derivation="abc123"))
    assert first == second


def test_different_derivations_key_differently():
    assert source_stats_key(_src(ephemeral=True, derivation="abc123")) != source_stats_key(
        _src(ephemeral=True, derivation="def456")
    )


def test_an_ordinary_in_memory_source_still_keys_per_instance():
    """Unchanged for everything else: shape-based identity collides, so it keys per object."""
    assert source_stats_key(_src()) != source_stats_key(_src())


def test_a_derived_source_is_not_pinned_by_the_cache():
    """The keepalive exists for `id()` reuse, which a derivation key does not have.

    Pinning would hold the whole materialized intermediate resident until the entry is
    evicted, which is the cost that made caching over these sources unattractive in the first
    place.
    """
    plan_cache.clear()
    derived = _src(ephemeral=True, derivation="abc123")
    ordinary = _src()
    plan_cache.store("k", "plan", [derived, ordinary], 8)
    _, keepalive = plan_cache._CACHE["k"]
    assert ordinary in keepalive
    assert derived not in keepalive
    plan_cache.clear()
