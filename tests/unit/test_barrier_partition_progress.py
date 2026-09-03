"""A long map or inference stage has to say how far through it is, while it runs.

`PARTITION` is the event that answers "N of M done", and it is what turns the live progress
line from an indeterminate sweep into a real bar with an ETA. Two barriers gather
distributed work and only one published it: the shuffle barrier
(`carbonite.resilience.gather_with_backups`) has since it existed, and `gather_map_results`
-- the barrier the map, inference and write paths go through, which is where a multi-hour
job actually spends its time -- published nothing at all. So the one shape whose progress a
person most needs was the shape that reported none.

Driven against the same fake Ray the window tests use, so the count is verified
deterministically rather than against a live cluster.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher._internal import events

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_ray(monkeypatch):
    """A `ray` whose refs are thunks, completing in submission order."""
    exceptions = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    exceptions.RayError = RayError
    exceptions.RayTaskError = RayTaskError
    module = types.ModuleType("ray")
    module.exceptions = exceptions
    module.wait = lambda refs, num_returns=1, timeout=None: ([refs[0]], refs[1:])
    module.get = lambda ref: ref()
    module.cluster_resources = lambda: {"CPU": 0.0}
    monkeypatch.setitem(sys.modules, "ray", module)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exceptions)
    return module


@pytest.fixture
def bus():
    """Every `PARTITION` event published inside the test."""
    seen: list[events.Event] = []
    unsubscribe = events.subscribe(lambda e: seen.append(e) if e.kind == events.PARTITION else None)
    try:
        yield seen
    finally:
        unsubscribe()


def _gather(n, stage=""):
    from batcher.dist.executors.ray_runtime import gather_map_results

    return gather_map_results(lambda idx: lambda i=idx: [i], n, stage=stage)


def test_one_event_per_partition_that_lands(fake_ray, bus):
    """The count is the point; without it the bar has no denominator and no ETA."""
    assert _gather(6) == [[i] for i in range(6)]
    assert len(bus) == 6


def test_each_event_carries_the_width_of_the_stage(fake_ray, bus):
    """ "N of M" needs the M, and this barrier is the only place that holds both."""
    _gather(4)
    assert [e.fields["total"] for e in bus] == [4, 4, 4, 4]


def test_the_running_count_advances(fake_ray, bus):
    """A consumer that cannot difference events still gets the position directly."""
    _gather(3)
    assert [e.fields["done"] for e in bus] == [1, 2, 3]


def test_the_stage_is_labelled_when_the_caller_names_it(fake_ray, bus):
    """Otherwise two stages of one query are indistinguishable in the same feed."""
    _gather(2, stage="map")
    assert {e.name for e in bus} == {"map"}


def test_an_unlabelled_caller_still_reports_its_count(fake_ray, bus):
    """The label is optional; the progress is not."""
    _gather(2)
    assert len(bus) == 2
    assert {e.name for e in bus} == {"stage"}


def test_an_empty_stage_reports_nothing(fake_ray, bus):
    """Zero partitions is not a stage that made progress, and must not read as one."""
    assert _gather(0) == []
    assert bus == []


def test_a_resubmitted_partition_is_counted_once(fake_ray, bus, monkeypatch):
    """`finished` counts wakeups, including transient failures that get resubmitted.

    Counting those would make the bar overshoot its own denominator on exactly the runs
    where a person is watching it, which is why the event is published after the result is
    handled rather than at the top of the loop.
    """
    from batcher.carbonite.resilience import RecoveryPolicy
    from batcher.dist.executors.ray_runtime import gather_map_results

    attempts = {0: 1}

    def submit(idx):
        if attempts.get(idx, 0) > 0:
            attempts[idx] -= 1

            def _boom():
                raise sys.modules["ray.exceptions"].RayError("preempted")

            return _boom
        return lambda i=idx: [i]

    out = gather_map_results(submit, 3, RecoveryPolicy(max_attempts=3), stage="map")
    assert out == [[0], [1], [2]]
    assert [e.fields["done"] for e in bus] == [1, 2, 3]
