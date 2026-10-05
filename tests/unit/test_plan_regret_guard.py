"""The plan memo puts back a plan a re-plan displaced when the replacement measures slower.

`kyber.plan_cache.memo`'s regret guard: learning re-plans a memoized key, the new plan runs, and
if its execution is clearly slower than the best the displaced plan measured, the displaced
plan is restored and the key pinned to it. A replacement that runs as fast or faster stands.
"""

from __future__ import annotations

import pytest

from batcher.kyber import plan_cache
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds

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


def _physical(est_rows: float, algorithm: str = "hash") -> PhysicalPlan:
    """A one-operator plan; `est_rows` is an annotation, `algorithm` is what the engine runs."""
    op = PhysicalOp(
        op_id=OpId(0),
        kind="Filter",
        backend="engine",
        algorithm=algorithm,
        bounds=ResourceBounds(m_max_bytes=int(est_rows) * 8, c_max_credits=1, n_max_parallelism=1),
        inputs=(),
        properties=PlanProperties(est_rows=est_rows),
    )
    return PhysicalPlan(ir={"op": "filter"}, output_schema=None, ops=(op,))


def test_a_replan_that_moved_only_estimates_is_never_reverted_for_time():
    """A busy machine slowing the one run after a re-plan is not a regression of the plan.

    A measured filter selectivity re-plans its key and changes only the annotated estimate;
    reverting on time put the unmeasured estimate back, and pinned it, whenever that run was
    slow (`tests/unit/test_plan_deps.py` failed only on a CPU-saturated box).
    """
    unmeasured, measured = _physical(1_667.0), _physical(0.0)
    _store_and_run(unmeasured, 1.3)
    _replan()
    _store_and_run(measured, 9.0)  # 7x and 7.7 ms: past both thresholds, purely load
    assert plan_cache.lookup(_KEY) is measured
    _replan()  # not pinned: the key still re-plans when its dependencies move


def test_a_replan_that_changed_what_runs_is_still_reverted():
    """Positive control for the test above: the same timings revert a genuinely different plan."""
    first, second = _physical(1_667.0, "hash"), _physical(0.0, "sort")
    _store_and_run(first, 1.3)
    _replan()
    _store_and_run(second, 9.0)
    assert plan_cache.lookup(_KEY, lambda deps, rounds: False) is first


def test_a_replan_that_reproduced_its_plan_settles_the_key():
    """A re-plan that rebuilt what it replaced tells `holds` the key is settled.

    The positive control is the first re-plan: before it, the key is not settled.
    """
    seen: list[bool] = []

    def holds(_deps, _rounds, settled=False):
        seen.append(settled)
        return False

    holds.accepts_settled = True  # type: ignore[attr-defined]
    plan = _physical(1_667.0)
    plan_cache.store(_KEY, (plan, None, ()), None, 16)
    assert plan_cache.lookup(_KEY, holds) is None
    plan_cache.store(_KEY, (_physical(1_000.0), None, ()), None, 16)  # estimates moved only
    assert plan_cache.lookup(_KEY, holds) is None
    assert seen == [False, True]
