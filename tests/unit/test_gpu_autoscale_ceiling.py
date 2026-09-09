"""A fixed GPU fleet must learn its device ceiling, and must never learn it from a zero.

`await_autoscale` is called before every fan-out is sized, so a query that asked the autoscaler
for more devices than the fleet will ever have pays the startup grace — 12 s by default — while
the poll loop rediscovers that nothing is coming. The CPU half has always learned that once and
short-circuited afterwards. The GPU half deliberately did not, and the reason was sound: a fleet
whose GPU node has not registered yet reports **0** devices, and capping future requests at 0
would disable the accelerator for the life of the driver on exactly the cluster that was about
to have one.

Excluding devices entirely is not the only way to avoid that, and it made the function's own
promise ("a fixed cluster pays the startup grace once, not per query") false for every GPU
stage. Refusing to record a *zero* is enough. These tests pin both halves: that a positive
stall is learned, and that a zero is not.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime import readiness

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    readiness._reset_capacity_ceiling()
    yield
    readiness._reset_capacity_ceiling()


def _gpu_ceiling() -> float:
    return readiness._reachable_gpu_ceiling


def test_a_fresh_process_has_no_device_ceiling():
    assert _gpu_ceiling() == float("inf")


def test_a_positive_stall_is_learned():
    readiness._note_gpu_ceiling(6.0)
    assert _gpu_ceiling() == 6.0


def test_a_zero_stall_is_never_learned():
    """The case the old GPU exclusion existed for: a fleet whose GPU node has not registered."""
    readiness._note_gpu_ceiling(0.0)
    assert _gpu_ceiling() == float("inf")


def test_a_negative_reading_is_never_learned():
    readiness._note_gpu_ceiling(-1.0)
    assert _gpu_ceiling() == float("inf")


def test_the_ceiling_only_ever_tightens():
    readiness._note_gpu_ceiling(8.0)
    readiness._note_gpu_ceiling(12.0)
    assert _gpu_ceiling() == 8.0


def test_growth_past_the_ceiling_lifts_it():
    """A cluster that later grew must not stay capped by what a previous wait observed."""
    readiness._note_gpu_ceiling(6.0)
    readiness._note_gpus_reached(8.0)
    assert _gpu_ceiling() == float("inf")


def test_reaching_the_ceiling_exactly_does_not_lift_it():
    readiness._note_gpu_ceiling(6.0)
    readiness._note_gpus_reached(6.0)
    assert _gpu_ceiling() == 6.0


def test_the_two_ceilings_are_independent():
    """A CPU-only fleet stalling at 48 cores says nothing about its devices, and the reverse."""
    readiness._note_ceiling(48)
    assert _gpu_ceiling() == float("inf")
    readiness._reset_capacity_ceiling()
    readiness._note_gpu_ceiling(6.0)
    assert readiness._reachable_ceiling == float("inf")


def test_a_learned_device_ceiling_short_circuits_the_wait(monkeypatch):
    """The behaviour the whole entry exists for: no poll loop on a proven-unreachable target."""
    entered = []
    monkeypatch.setattr(
        readiness,
        "_cluster_topology",
        lambda: {"cpus": 48.0, "gpus": 6.0, "nodes": 6},
    )
    monkeypatch.setattr(
        readiness,
        "_await_autoscale",
        lambda *a, **k: entered.append(a) or 48,
    )
    import ray

    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    readiness._note_ceiling(48)
    readiness._note_gpu_ceiling(6.0)
    readiness.await_autoscale(8, target_gpus=8.0)
    assert entered == [], "a proven-unreachable device target must not re-enter the poll loop"


def test_an_unlearned_device_target_still_waits(monkeypatch):
    """The ceiling must short-circuit only what it has actually proven unreachable."""
    entered = []
    monkeypatch.setattr(
        readiness,
        "_cluster_topology",
        lambda: {"cpus": 48.0, "gpus": 6.0, "nodes": 6},
    )
    monkeypatch.setattr(
        readiness,
        "_await_autoscale",
        lambda *a, **k: entered.append(a) or 48,
    )
    import ray

    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    readiness._note_ceiling(48)
    readiness.await_autoscale(8, target_gpus=8.0)
    assert entered, "nothing has been learned about devices; the wait must still run"
