"""A killed job must not leave the cluster pinned at its scaled-up size.

`request_resources` sets a *sticky* floor that lives in the autoscaler, not in the driver,
so it outlives the process that set it. `release_autoscale` runs in a `finally`, which
covers every way a query can end and none of the ways a job can be killed — and being
killed is the normal case here: Slurm ends the allocation at its time limit, Kubernetes
evicts the driver pod, a spot reclamation takes the driver's node.

Left alone, one killed job holds a cluster at full size indefinitely with nothing running
against it. A drain notice arrives before the kill in all three cases, so that is where the
floor has to be dropped.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import Config, config_context
from batcher.dist.executors.ray_runtime import autoscale_request as ar

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_floor(monkeypatch):
    """Each test starts with no floor, nothing armed, and the Ray SDK call captured."""
    applied: list[tuple] = []
    monkeypatch.setattr(ar, "_apply_autoscale_floor", lambda *a: applied.append(a))
    monkeypatch.setattr(ar, "_autoscale_active", 0)
    monkeypatch.setattr(ar, "_autoscale_floor", 0)
    monkeypatch.setattr(ar, "_autoscale_gpu_floor", 0)
    monkeypatch.setattr(ar, "_autoscale_resources", ())
    monkeypatch.setattr(ar, "_drain_release_armed", False)
    return applied


def _spot() -> Config:
    base = Config()
    return base.replace(distributed=dataclasses.replace(base.distributed, resilience="spot"))


def _stable() -> Config:
    base = Config()
    return base.replace(distributed=dataclasses.replace(base.distributed, resilience="default"))


class _FakeMonitor:
    """Stands in for the process-wide `PreemptionMonitor`."""

    def __init__(self) -> None:
        self.callbacks: list = []
        self.started = 0

    def on_drain(self, callback) -> None:
        self.callbacks.append(callback)

    def start(self) -> None:
        self.started += 1

    def fire(self) -> None:
        for callback in self.callbacks:
            callback()


@pytest.fixture
def monitor(monkeypatch):
    fake = _FakeMonitor()
    import batcher.carbonite.resilience as res

    monkeypatch.setattr(res, "preemption_monitor", lambda: fake)
    return fake


def test_drain_drops_the_floor(_reset_floor, monitor):
    """The whole point: the capacity the query asked for is given back when it is killed."""
    with config_context(_spot()):
        ar.request_autoscale(64)
        assert ar._autoscale_floor == 64
        monitor.fire()
    assert ar._autoscale_floor == 0
    assert _reset_floor[-1] == (0, 0, ())


def test_drain_release_abandons_every_in_flight_scope(_reset_floor, monitor):
    """Unlike `release_autoscale` this does not decrement. Two concurrent queries are both
    about to stop existing, so a decrement would leave the larger one's floor pinned."""
    with config_context(_spot()):
        ar.request_autoscale(64)
        ar.request_autoscale(128)
        assert ar._autoscale_floor == 128
        monitor.fire()
    assert ar._autoscale_floor == 0
    assert ar._autoscale_active == 0


def test_gpu_and_custom_resource_floors_are_dropped_too(_reset_floor, monitor):
    """A pinned GPU or TPU floor is the expensive one to leak."""
    with config_context(_spot()):
        ar.request_autoscale(64, target_gpus=8.0, target_resources=(("TPU", 4.0),))
        assert ar._autoscale_gpu_floor == 8
        monitor.fire()
    assert (ar._autoscale_gpu_floor, ar._autoscale_resources) == (0, ())


def test_a_stable_cluster_arms_nothing(_reset_floor, monitor):
    """A non-preemptible deployment must not pay for a poll thread or a signal trap."""
    with config_context(_stable()):
        ar.request_autoscale(64)
    assert monitor.started == 0
    assert monitor.callbacks == []
    assert ar._autoscale_floor == 64  # the normal lifecycle is untouched


def test_arming_happens_once_across_queries(_reset_floor, monitor):
    """The monitor fires each callback once; re-registering per query would only grow the
    list and drop the floor N times."""
    with config_context(_spot()):
        for _ in range(5):
            ar.request_autoscale(8)
    assert monitor.started == 1
    assert len(monitor.callbacks) == 1


def test_a_broken_monitor_does_not_fail_the_query(_reset_floor, monkeypatch):
    """Arming is best-effort: a leaked floor is a cost, not a correctness problem, so it
    must never take down the query that was merely asking for capacity."""
    import batcher.carbonite.resilience as res

    def _boom():
        raise RuntimeError("no monitor here")

    monkeypatch.setattr(res, "preemption_monitor", _boom)
    with config_context(_spot()):
        ar.request_autoscale(64)
    assert ar._autoscale_floor == 64


def test_normal_teardown_still_releases(_reset_floor, monitor):
    """The ordinary path is unchanged — the drain hook is a backstop, not a replacement."""
    with config_context(_spot()):
        ar.request_autoscale(64)
        ar.release_autoscale()
    assert ar._autoscale_floor == 0
    assert _reset_floor[-1] == (0, 0, ())


class TestNoScaleUpNearTheDeadline:
    """Nodes that boot after the job is killed are a bill for zero work.

    The floor is sticky and lives in the autoscaler, so capacity requested in a job's last
    seconds outlives it and sits idle until something else drops it.
    """

    def _deadline_in(self, monkeypatch, seconds: float) -> None:
        import time

        for var in ("BATCHER_DEADLINE_EPOCH_S", "SLURM_JOB_END_TIME"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("BATCHER_DEADLINE_EPOCH_S", str(time.time() + seconds))

    def test_a_job_with_no_time_left_asks_for_nothing(self, _reset_floor, monitor, monkeypatch):
        self._deadline_in(monkeypatch, 10.0)
        with config_context(_spot()):  # drain_lead_s defaults to 120
            ar.request_autoscale(64)
        assert ar._autoscale_floor == 0
        assert _reset_floor == [], "no floor should have been applied at all"

    def test_the_scope_is_still_counted_so_release_stays_balanced(
        self, _reset_floor, monitor, monkeypatch
    ):
        """Skipping the ask must not desynchronize the request/release pairing."""
        self._deadline_in(monkeypatch, 10.0)
        with config_context(_spot()):
            ar.request_autoscale(64)
            assert ar._autoscale_active == 1
            ar.release_autoscale()
        assert ar._autoscale_active == 0

    def test_a_job_with_plenty_of_time_still_scales_up(self, _reset_floor, monitor, monkeypatch):
        self._deadline_in(monkeypatch, 3600.0)
        with config_context(_spot()):
            ar.request_autoscale(64)
        assert ar._autoscale_floor == 64

    def test_no_deadline_is_unchanged(self, _reset_floor, monitor, monkeypatch):
        for var in ("BATCHER_DEADLINE_EPOCH_S", "SLURM_JOB_END_TIME"):
            monkeypatch.delenv(var, raising=False)
        with config_context(_spot()):
            ar.request_autoscale(64)
        assert ar._autoscale_floor == 64

    def test_a_zero_drain_lead_does_not_disable_scale_up(self, _reset_floor, monitor, monkeypatch):
        """Turning the drain lead off must not read as 'there is never time to scale'."""
        import dataclasses

        for var in ("BATCHER_DEADLINE_EPOCH_S", "SLURM_JOB_END_TIME"):
            monkeypatch.delenv(var, raising=False)
        base = Config()
        cfg = base.replace(
            distributed=dataclasses.replace(base.distributed, resilience="spot", drain_lead_s=0.0)
        )
        with config_context(cfg):
            ar.request_autoscale(64)
        assert ar._autoscale_floor == 64
