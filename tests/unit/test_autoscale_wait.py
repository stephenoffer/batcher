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
from batcher.dist.executors.ray_runtime import scaling

pytestmark = pytest.mark.unit


class _FakeClock:
    """A deterministic monotonic clock advanced only by `sleep` — no real waiting."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def _run(monkeypatch, cpus_series, *, target, wait=180.0, poll=5.0, stall=90.0):
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
    base = active_config()
    dc = dataclasses.replace(
        base.distributed, autoscale_wait_s=wait, autoscale_poll_s=poll, autoscale_stall_s=stall
    )
    with config_context(base.replace(distributed=dc)):
        result = scaling._await_autoscale(target, cpus_series(0.0), 0.0, 0.0)
    return result, clock.t


def test_flat_cluster_bails_after_grace_not_full_wait(monkeypatch):
    # A fixed cluster (capacity never grows) must give up after ~the stall grace, not
    # block for the whole 180s budget on nodes that will never arrive.
    result, elapsed = _run(monkeypatch, lambda _t: 8, target=32, wait=180.0, poll=5.0, stall=90.0)
    assert result == 8
    assert elapsed <= 90.0 + 5.0  # grace window + one poll, far below the 180s budget


def test_growing_cluster_reaches_target(monkeypatch):
    # Capacity climbs a node (16 CPUs) every 30s; the wait must stay until it covers the
    # request and then return the satisfied capacity.
    def series(t: float) -> float:
        return 8 + 16 * int(t // 30)  # 8, 24, 40, ...

    result, _elapsed = _run(monkeypatch, series, target=32, wait=180.0, poll=5.0, stall=90.0)
    assert result >= 32


def test_growth_then_stall_returns_partial(monkeypatch):
    # Grows to 24 (below the target 32) then stalls — spot capacity ran out mid-scale-up.
    # It should ride the growth, then bail a grace window after the last gain with the
    # partial capacity, not hang for the full budget.
    def series(t: float) -> float:
        return 24 if t >= 20 else 8

    result, elapsed = _run(monkeypatch, series, target=32, wait=300.0, poll=5.0, stall=60.0)
    assert result == 24
    assert elapsed <= 20.0 + 60.0 + 5.0  # last growth at ~20s + grace + a poll


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
        scaling.await_autoscale(64, 0.0)
    assert called["topo"] == 0  # never consulted the cluster


def test_await_autoscale_noop_when_target_nonpositive(monkeypatch):
    # A zero/negative target (nothing requested) short-circuits regardless of the wait.
    base = active_config()
    dc = dataclasses.replace(base.distributed, autoscale_wait_s=180.0)
    monkeypatch.setattr(
        scaling, "cluster_topology", lambda: pytest.fail("should not read topology")
    )
    with config_context(base.replace(distributed=dc)):
        scaling.await_autoscale(0, 0.0)
