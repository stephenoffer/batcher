"""The bounded autoscale wait grows the cluster before clamping — and gives up early.

`_await_autoscale` waits (up to `autoscale_wait_s`) for the Ray autoscaler to bring the
cluster up to a query's requested capacity, so a big job runs on the scaled-up cluster
instead of under-provisioned. Its safety valve is the `autoscale_stall_s` grace window:
if capacity stays flat that long the autoscaler has nothing more to add (a fixed cluster)
or cannot satisfy the request (spot capacity unavailable), so it stops rather than blocking
the whole budget for nodes that will not arrive. These are pure, Ray-free tests: the live
cluster behavior is covered in the distributed integration suite.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from batcher.config import active_config, config_context
from batcher.dist.executors.ray_runtime import readiness, scaling

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_ceiling():
    # The learned ceiling is process-global; reset it before each test so one test's fixed
    # cluster doesn't cap another's. The persistence tests drive their own two `_run` calls
    # within a single test, so this per-test reset doesn't undercut them.
    readiness._reset_capacity_ceiling()
    yield
    readiness._reset_capacity_ceiling()


class _FakeClock:
    """A deterministic monotonic clock advanced only by `sleep` — no real waiting."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def _run(monkeypatch, cpus_series, *, target, wait=180.0, poll=5.0, stall=90.0, startup=12.0):
    """Drive `_await_autoscale` with a scripted CPU-capacity series and a fake clock.

    `cpus_series` is a callable time -> cpus (GPUs held at 0). Returns
    `(result_cpus, elapsed_seconds)`.
    """
    clock = _FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    monkeypatch.setattr(
        scaling, "cluster_topology", lambda: {"nodes": 1, "cpus": cpus_series(clock.t), "gpus": 0.0}
    )
    monkeypatch.setattr(scaling, "_ray_initialized", lambda: True, raising=False)
    base = active_config()
    dc = dataclasses.replace(
        base.distributed,
        autoscale_wait_s=wait,
        autoscale_poll_s=poll,
        autoscale_stall_s=stall,
        autoscale_startup_grace_s=startup,
    )
    with config_context(base.replace(distributed=dc)):
        # Route through the public entry so the learned-ceiling short-circuit is exercised;
        # it reads the initial capacity from `cluster_topology` (already patched) itself.
        import ray

        monkeypatch.setattr(ray, "is_initialized", lambda: True)
        before = clock.t
        readiness.await_autoscale(target, 0.0)
        # `await_autoscale` returns None; recover the observed capacity from the topology at
        # the clock position it left off (the same value `_await_autoscale` would have
        # returned), and whether it actually waited from the elapsed time.
        result = int(cpus_series(clock.t))
        _ = before
    return result, clock.t


def test_flat_cluster_bails_after_startup_grace_not_stall_window(monkeypatch):
    # A fixed cluster (capacity never grows) never enters the "growing" regime, so it must
    # give up after the SHORT startup grace — not the long post-growth stall window, and far
    # from the whole 180s budget. This is the 90s-per-cold-query tax the startup grace kills:
    # a large aggregate's fan-out routinely asks for more cores than a fixed cluster has.
    result, elapsed = _run(
        monkeypatch, lambda _t: 8, target=32, wait=180.0, poll=5.0, stall=90.0, startup=12.0
    )
    assert result == 8
    assert elapsed <= 12.0 + 5.0  # startup grace + one poll, not the 90s stall window


def test_growing_cluster_reaches_target(monkeypatch):
    # Capacity climbs a node (16 CPUs) every 15s; the first node arrives within the startup
    # grace, flipping into the growing regime so the wait stays until it covers the request
    # and returns the satisfied capacity. `startup` is set past the first growth to model a
    # cluster whose nodes register inside the startup window.
    def series(t: float) -> float:
        return 8 + 16 * int(t // 15)  # 8, 24, 40, ...

    result, _elapsed = _run(
        monkeypatch, series, target=32, wait=180.0, poll=5.0, stall=90.0, startup=20.0
    )
    assert result >= 32


def test_growth_then_stall_returns_partial(monkeypatch):
    # Grows to 24 (below the target 32) then stalls — spot capacity ran out mid-scale-up.
    # It should ride the growth, then bail a STALL grace window after the last gain with the
    # partial capacity, not hang for the full budget. `startup` covers the first growth (15s).
    def series(t: float) -> float:
        return 24 if t >= 15 else 8

    result, elapsed = _run(
        monkeypatch, series, target=32, wait=300.0, poll=5.0, stall=60.0, startup=20.0
    )
    assert result == 24
    assert elapsed <= 15.0 + 60.0 + 5.0  # last growth at ~15s + STALL grace + a poll


def test_growth_within_startup_grace_earns_the_full_stall_window(monkeypatch):
    # A single early growth (a node registers at 10s, inside the 12s startup grace) flips the
    # wait into the "growing" regime: the LONG stall window then governs, so a cluster still
    # bringing nodes up is not abandoned. It stalls at 24 (never reaches 32) and bails a full
    # stall window after that lone gain — proving the regime switched, not the startup grace.
    def series(t: float) -> float:
        return 24 if t >= 10 else 8

    result, elapsed = _run(
        monkeypatch, series, target=32, wait=300.0, poll=5.0, stall=60.0, startup=12.0
    )
    assert result == 24
    # Bailed well after the 12s startup grace would have (10s growth + 60s stall), proving the
    # single growth earned the long window rather than the short one.
    assert elapsed > 12.0 + 5.0
    assert elapsed <= 10.0 + 60.0 + 5.0


def test_flat_wait_learns_a_ceiling_that_short_circuits_later_waits(monkeypatch):
    # After a wait stalls below target on a fixed cluster, the cluster's unreachable ceiling
    # is remembered: a later query asking for more than the observed capacity must skip the
    # wait entirely (0 elapsed), so a fixed-at-max cluster pays the startup grace ONCE, not on
    # every cold query — the whole point of the ceiling.
    readiness._reset_capacity_ceiling()
    try:
        _r1, e1 = _run(monkeypatch, lambda _t: 8, target=32, wait=180.0, poll=5.0, startup=12.0)
        assert e1 > 0.0  # the first wait actually probed (and learned 8 is the ceiling)
        _r2, e2 = _run(monkeypatch, lambda _t: 8, target=64, wait=180.0, poll=5.0, startup=12.0)
        assert e2 == 0.0  # a larger request than the learned ceiling short-circuits
    finally:
        readiness._reset_capacity_ceiling()


def test_reached_target_lifts_a_stale_ceiling(monkeypatch):
    # A ceiling is not permanent: once capacity has climbed past it (the cluster recovered /
    # more nodes joined), the bound is dropped so a later request that the grown cluster can
    # now satisfy is served immediately instead of being wrongly short-circuited.
    readiness._reset_capacity_ceiling()
    try:
        _run(monkeypatch, lambda _t: 8, target=32, wait=180.0, poll=5.0, startup=12.0)  # ceiling=8
        # The cluster has since grown to 40 cores; a 40-core request is now satisfiable and
        # must return immediately (current capacity already covers it), not stay capped at 8.
        _r, e = _run(monkeypatch, lambda _t: 40, target=40, wait=180.0, poll=5.0)
        assert _r >= 40 and e == 0.0
    finally:
        readiness._reset_capacity_ceiling()


def test_disabled_wait_is_immediate(monkeypatch):
    # `autoscale_wait_s == 0` keeps the non-blocking behavior: clamp to current capacity.
    result, elapsed = _run(monkeypatch, lambda _t: 8, target=32, wait=0.0)
    assert result == 8
    assert elapsed == 0.0


def test_await_autoscale_noop_when_disabled(monkeypatch):
    # The public wrapper is a no-op when the wait is off — it must not even touch Ray, so a
    # `ray.is_initialized` that would raise proves the short-circuit fires first.
    base = active_config()
    dc = dataclasses.replace(base.distributed, autoscale_wait_s=0.0)
    called = {"topo": 0}
    monkeypatch.setattr(scaling, "cluster_topology", lambda: called.__setitem__("topo", 1))
    with config_context(base.replace(distributed=dc)):
        readiness.await_autoscale(64, 0.0)
    assert called["topo"] == 0  # never consulted the cluster


def test_await_autoscale_noop_when_target_nonpositive(monkeypatch):
    # A zero/negative target (nothing requested) short-circuits regardless of the wait.
    base = active_config()
    dc = dataclasses.replace(base.distributed, autoscale_wait_s=180.0)
    monkeypatch.setattr(
        scaling, "cluster_topology", lambda: pytest.fail("should not read topology")
    )
    with config_context(base.replace(distributed=dc)):
        readiness.await_autoscale(0, 0.0)
