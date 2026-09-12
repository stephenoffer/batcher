"""The map barrier's initial deal must follow worker capacity, not worker count.

`map_partitions` sizes the source count at `workers x slots`, which is exactly how deep the
barrier's idle pool is pre-filled -- so every source is handed out from that initial fill and
the go-idle path that would have corrected an imbalance never runs. On a uniform fleet that is
invisible and correct. On an unequal one it is a static even deal wearing a dynamic barrier's
clothes.

Measured on the 28-node / 384-core mixed cluster, 128 sources over 32 workers, before the fix:
every worker took exactly 4 regardless of size, so a 3-core worker carried eight times the
per-core load of a 24-core one and the stage waited on the small machines.

These tests are about the *pool*, which is the thing that was wrong. They do not prove the
barrier as a whole redistributes correctly under load -- only a cluster run does that, and CI
has no Ray.
"""

from __future__ import annotations

from collections import Counter

import pytest

from batcher.dist.executors.ray_runtime.policies._barrier import _idle_pool


@pytest.fixture
def capacities(monkeypatch):
    """Set what `fleet_worker_cpus` reports to the barrier."""

    def _set(caps):
        monkeypatch.setattr(
            "batcher.dist.executors.ray_runtime.policies._barrier.fleet_worker_cpus",
            lambda workers: caps,
        )

    return _set


def test_a_uniform_fleet_is_dealt_evenly(capacities):
    """The positive control: nothing about the even case may change.

    A weighting that also perturbs uniform fleets would pass every test below while quietly
    changing the behaviour of every homogeneous cluster, which is most of them.
    """
    capacities([8.0] * 4)
    pool = _idle_pool(4, 3)

    assert len(pool) == 12
    assert Counter(pool) == {0: 3, 1: 3, 2: 3, 3: 3}


def test_no_capacity_information_keeps_the_even_deal(capacities):
    """`fleet_worker_cpus` returns `None` off the per-node path; that must stay unchanged."""
    capacities(None)
    pool = _idle_pool(4, 3)

    assert Counter(pool) == {0: 3, 1: 3, 2: 3, 3: 3}


def test_a_big_worker_is_dealt_more_than_a_small_one(capacities):
    """The defect: a 24-core and a 3-core worker were handed the same number of sources."""
    capacities([24.0, 3.0])
    pool = _idle_pool(2, 4)

    counts = Counter(pool)
    assert counts[0] > counts[1], f"the big worker was not favoured: {dict(counts)}"


def test_the_deal_tracks_the_core_ratio(capacities):
    """Sources per core should be roughly flat, which is what "equal finishing time" means."""
    caps = [24.0, 24.0, 15.0, 3.0, 3.0]
    capacities(caps)
    pool = _idle_pool(len(caps), 4)

    counts = Counter(pool)
    per_core = [counts[i] / caps[i] for i in range(len(caps))]
    assert max(per_core) / min(per_core) < 2.0, (
        f"per-core load still varies {max(per_core) / min(per_core):.1f}x: {dict(counts)}"
    )


def test_the_pool_depth_is_unchanged(capacities):
    """Weighting redistributes the pool; it must not deepen or shrink it.

    Depth is the barrier's in-flight bound. A pool that grew would raise concurrency as a side
    effect of a fairness change, and the two would be impossible to tell apart afterwards.
    """
    capacities([24.0, 23.0, 15.0, 3.0, 3.0, 3.0])
    assert len(_idle_pool(6, 4)) == 24


def test_every_worker_still_gets_at_least_one_source(capacities):
    """A worker starved to zero would idle for the whole stage, however small it is."""
    capacities([96.0, 1.0, 1.0, 1.0])
    counts = Counter(_idle_pool(4, 2))

    assert all(counts[i] >= 1 for i in range(4)), f"a worker was starved: {dict(counts)}"


def test_the_first_round_reaches_every_worker(capacities):
    """Ordering is round-robin, not blocked, so the stage starts on the whole fleet.

    A blocked fill would hand the first twenty-four sources to one worker and leave the rest
    idle until it finished -- correct totals, and a serialized start.
    """
    capacities([24.0, 24.0, 3.0, 3.0])
    pool = list(_idle_pool(4, 4))

    assert set(pool[:4]) == {0, 1, 2, 3}, f"the first round did not touch every worker: {pool[:8]}"


def test_a_degenerate_capacity_list_falls_back(capacities):
    """A zero or negative grant must not divide the deal by zero or starve the fleet."""
    capacities([0.0, 8.0, 8.0])
    counts = Counter(_idle_pool(3, 2))

    assert sum(counts.values()) == 6
    assert all(counts[i] >= 1 for i in range(3))
