"""A memoized plan is re-validated against the measurements its own planning read.

A filter's measured selectivity is folded from operator feedback rather than written through
the learning generation, so without this a plan cached before the filter was measured was
served forever: TPC-H q18's `HAVING` subquery kept 57 of 1.5M orders on every run and was
planned at a third of them on every run. The check must also stay *per plan*: one query
learning something must not invalidate another's plan (the global-bump attempt did, and cost
TPC-DS q34 5-12x inside a mixed workload).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber import plan_cache

pytestmark = pytest.mark.unit


def _orders() -> bt.Dataset:
    n = 20_000
    return bt.from_arrow(
        pa.table({"o": [i % 5_000 for i in range(n)], "q": [i % 7 for i in range(n)]})
    )


def _having(ds: bt.Dataset) -> bt.Dataset:
    # 5,000 groups; only the groups whose summed quantity clears 23 survive (a small share).
    return ds.group_by("o").agg(t=bt.col("q").sum()).filter(bt.col("t") > 23)


def _filter_estimate(ds: bt.Dataset) -> str:
    lines = [
        ln for ln in ds.explain().splitlines() if "filter" in ln and "t >" in ln.replace("  ", " ")
    ]
    assert lines, ds.explain()
    return lines[0]


def test_a_measured_filter_is_used_once_it_is_measured():
    ds = _having(_orders())
    actual = ds.count()
    cold = _filter_estimate(ds)
    assert "(default)" in cold, cold  # positive control: nothing is measured yet
    for _ in range(3):
        ds.collect()
    warm = _filter_estimate(ds)
    assert "(learned)" in warm, warm
    est = float(warm.split("est≈")[1].split()[0].replace(",", ""))
    assert actual / 2 <= est <= actual * 2, (est, actual)


def test_a_measured_filter_survives_a_busy_machine(monkeypatch):
    """The run after each re-plan is made to look slow, as a CPU-saturated box makes it.

    The re-plans this query takes change only the filter's estimate, never what the engine
    runs, so the memo's regret guard must not read their timing as a plan regression. It did:
    it restored the unmeasured plan and pinned it, and this file's tests failed only under
    load. Deterministic: the slow-down is injected at the guard, not generated.
    """
    from batcher.kyber.plan_cache import memo

    plan_cache.clear()
    real = memo.record_outcome
    spiked: list[float] = []

    def busy(result, elapsed_ms):
        hit = memo._SERVED.get(id(result))
        if hit is not None and hit[1] in memo._DISPLACED:  # the first run of a re-plan
            spiked.append(elapsed_ms)
            elapsed_ms = elapsed_ms * 5 + 10.0
        real(result, elapsed_ms)

    monkeypatch.setattr(plan_cache, "record_outcome", busy)
    ds = _having(_orders())
    for _ in range(4):
        ds.collect()
    assert spiked, "no re-plan was timed, so the guard was never exercised"
    warm = _filter_estimate(ds)
    assert "(learned)" in warm, warm


def test_another_query_learning_does_not_invalidate_this_plans_dependencies():
    """B learning its own filter leaves every dependency A's cached plans recorded intact.

    Scoped to what the per-plan check decides. A's cache key can still move for other reasons
    (B learning column statistics advances the global learning generation, which predates
    this); what must not happen is the *dependency* check dropping A for B's measurements.
    """
    from batcher import core
    from batcher.kyber.optimizer.plan_deps import dependencies_hold

    plan_cache.clear()
    a = _having(_orders())
    for _ in range(4):
        a.collect()  # warm: every one of A's own measurements has settled
    hub = core.default_hub()
    # Only the entries still valid now: the cache also holds A's older plans, stored before its
    # measurements existed, and those are correctly invalid.
    a_deps = [
        e[2] for e in plan_cache.memo._CACHE.values() if e[2] and dependencies_hold(hub, e[2])
    ]
    assert a_deps, "A's plans recorded no live dependencies, so there is nothing to check"

    b = bt.from_pydict({"x": list(range(5_000))}).filter(bt.col("x") % 3 == 0)
    for _ in range(3):
        b.collect()  # B learns its own filter's selectivity meanwhile
    assert all(dependencies_hold(hub, deps) for deps in a_deps)


class _Hub:
    """Stands in for the metadata hub: `plan_deps` only reads it through `_measured`."""


def test_a_newly_measured_dependency_replans_only_within_the_learning_rounds(monkeypatch):
    """A measurement appearing re-plans for `LEARNING_ROUNDS` rounds, then no longer does.

    Every round is checked against the same appeared measurement, so the only thing that moves
    the verdict is the round count. A measurement that moves by a full bucket still re-plans
    at any round: the damping is on appearances, not on the plan's own numbers moving.
    """
    from batcher.kyber.optimizer import plan_deps

    measured = {"sig": 0.5}
    monkeypatch.setattr(plan_deps, "_measured", lambda _hub: (measured, {}, {}, {}))
    unmeasured = (("sig", None, None, None),)
    verdicts = [
        plan_deps.dependencies_hold(_Hub(), unmeasured, r)
        for r in range(plan_deps.LEARNING_ROUNDS + 2)
    ]
    assert verdicts == [False] * plan_deps.LEARNING_ROUNDS + [True, True]

    snapshot = plan_deps.dependency_snapshot(_Hub(), {"sig"})
    assert plan_deps.dependencies_hold(_Hub(), snapshot, plan_deps.LEARNING_ROUNDS + 5)
    measured["sig"] = 0.5 / 4  # two octaves, four half-octave buckets
    assert not plan_deps.dependencies_hold(_Hub(), snapshot, plan_deps.LEARNING_ROUNDS + 5)


def test_the_plan_cache_counts_the_rounds_a_key_was_replanned():
    """The round passed to `holds` is how many times this key's entry was dropped for it."""
    plan_cache.clear()
    seen: list[int] = []

    def holds(_deps, rounds):
        seen.append(rounds)
        return False

    for _ in range(3):
        assert plan_cache.lookup("k", holds) is None
        plan_cache.store("k", "plan", None, 8, ("dep",))
    assert seen == [0, 1]
    plan_cache.clear()


def test_a_measurement_landing_on_the_planned_value_does_not_replan(monkeypatch):
    """A first measurement is judged against what the plan used, not against its absence.

    Re-planning on a measurement that confirms the estimate rebuilds the same plan from the
    same number: TPC-DS q37 re-planned three times that way. One that lands two octaves away
    re-plans, which is the positive control that the comparison is live.
    """
    from batcher.kyber.optimizer import plan_deps

    measured: dict[str, float] = {}
    monkeypatch.setattr(plan_deps, "_measured", lambda _hub: (measured, {}, {}, {}))
    snapshot = plan_deps.dependency_snapshot(_Hub(), {"sig"}, {"sig": [0.3, None, None]})
    assert plan_deps.dependencies_hold(_Hub(), snapshot, 0)  # nothing measured yet
    measured["sig"] = 0.33  # within a bucket of the planned 0.3
    assert plan_deps.dependencies_hold(_Hub(), snapshot, 0)
    measured["sig"] = 0.3 / 4
    assert not plan_deps.dependencies_hold(_Hub(), snapshot, 0)


def test_a_correctable_operator_is_judged_by_its_rows_not_its_factor(monkeypatch):
    """A drifting correction factor re-plans nothing while the measured rows hold still.

    The factor multiplies a structural estimate that sharpens as column statistics arrive,
    so it moves run to run (q37: 0.03, then 0.14, then 0.28) while the corrected row count
    the plan turns on does not. Rows far from the planned estimate still re-plan.
    """
    from batcher.kyber.optimizer import plan_deps

    corr, rows = {"sig": 0.03}, {"sig": 16.0}
    monkeypatch.setattr(plan_deps, "_measured", lambda _hub: ({}, corr, {}, rows))
    snapshot = plan_deps.dependency_snapshot(_Hub(), {"sig"}, {"sig": [None, 18.0, None]})
    corr["sig"] = 0.28  # six buckets: alone, this used to re-plan
    assert plan_deps.dependencies_hold(_Hub(), snapshot, 0)
    rows["sig"] = 16.0 * 8
    assert not plan_deps.dependencies_hold(_Hub(), snapshot, 0)


def test_a_settled_plan_replans_only_for_drift_or_a_new_measurement(monkeypatch):
    """`settled`: what the reproducing re-plan already saw re-plans only by drifting.

    A dependency measured *after* it (a filter's selectivity, folded a run later than the
    corrections that triggered the re-plan, the TPC-H q18 case) is still judged as usual.
    """
    from batcher.kyber.optimizer import plan_deps

    sel, rows = {}, {"r": 16.0}
    monkeypatch.setattr(plan_deps, "_measured", lambda _hub: (sel, {}, {}, rows))
    used = {"r": [None, 1_000.0, None], "f": [0.33, None, None]}
    snapshot = plan_deps.dependency_snapshot(_Hub(), {"r", "f"}, used)
    assert not plan_deps.dependencies_hold(_Hub(), snapshot, 0)  # rows far from the plan
    assert plan_deps.dependencies_hold(_Hub(), snapshot, 0, settled=True)
    sel["f"] = 57 / 1_500_000  # measured after the settling re-plan, far from 0.33
    assert not plan_deps.dependencies_hold(_Hub(), snapshot, 0, settled=True)
    sel.clear()
    rows["r"] = 16.0 * 8  # drift of what was already measured
    assert not plan_deps.dependencies_hold(_Hub(), snapshot, 0, settled=True)
