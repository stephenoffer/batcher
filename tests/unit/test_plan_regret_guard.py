"""The plan memo puts back a plan a re-plan displaced when the replacement measures slower.

`kyber.plan_cache.memo`'s regret guard: learning re-plans a memoized key, the new plan runs, and
if its execution is clearly slower than the best the displaced plan measured, the displaced
plan is restored and the key pinned to it. A replacement that runs as fast or faster stands.
"""

from __future__ import annotations

import pytest

from batcher.kyber import plan_cache

_KEY = "L1:x|full|plan|cfg|hub|stats|hw|[]"


@pytest.fixture(autouse=True)
def _fresh_memo():
    plan_cache.clear()
    yield
    plan_cache.clear()


def _store_and_run(result: object, ms: float) -> None:
    plan_cache.store(_KEY, result, None, 16)
    plan_cache.served(result, _KEY)
    plan_cache.record_outcome(result, ms)


def _replan() -> None:
    """A lookup whose dependencies moved: the entry is dropped and reported as a miss."""
    assert plan_cache.lookup(_KEY, lambda deps, rounds: False) is None


def test_a_replan_that_runs_slower_is_reverted_and_pinned():
    first, second = object(), object()
    _store_and_run(first, 40.0)
    _replan()
    _store_and_run(second, 230.0)  # the learning loop's choice ran 5x slower
    # The displaced plan is back, and stays: moved dependencies no longer re-plan the key.
    assert plan_cache.lookup(_KEY, lambda deps, rounds: False) is first
    assert plan_cache.lookup(_KEY, lambda deps, rounds: False) is first


def test_a_replan_that_runs_faster_stands():
    first, second = object(), object()
    _store_and_run(first, 230.0)
    _replan()
    _store_and_run(second, 40.0)
    assert plan_cache.lookup(_KEY) is second


def test_jitter_below_the_floor_does_not_revert():
    first, second = object(), object()
    _store_and_run(first, 1.0)
    _replan()
    _store_and_run(second, 2.5)  # 2.5x, but 1.5 ms: noise on a millisecond query
    assert plan_cache.lookup(_KEY) is second


def test_a_plan_never_measured_is_not_held_against_its_replacement():
    first, second = object(), object()
    plan_cache.store(_KEY, first, None, 16)  # served, never reported
    _replan()
    _store_and_run(second, 500.0)
    assert plan_cache.lookup(_KEY) is second


def test_a_time_reported_against_an_unserved_result_is_ignored():
    first, second = object(), object()
    plan_cache.store(_KEY, first, None, 16)
    plan_cache.record_outcome(object(), 1.0)  # not a result the memo handed out
    _replan()
    _store_and_run(second, 500.0)
    assert plan_cache.lookup(_KEY) is second  # no best was filed for the key to revert to


def test_a_key_stops_replanning_after_its_budget():
    from batcher.kyber.plan_cache.memo import _MAX_REPLANS

    plans = [object() for _ in range(_MAX_REPLANS + 1)]
    for i, plan in enumerate(plans):
        _store_and_run(plan, 50.0 - i)  # each re-plan a little faster: none is reverted
        if i < _MAX_REPLANS:
            _replan()
    # The budget is spent: a dependency moving again no longer drops the plan.
    assert plan_cache.lookup(_KEY, lambda deps, rounds: False) is plans[-1]
