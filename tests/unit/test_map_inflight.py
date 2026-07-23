"""WS3: per-actor submit-ahead depth for the GPU/inference actor pool.

A depth-1 pool leaves a GPU idle across the dispatch/gather round-trip; a deeper pipeline
keeps it fed. The change must NOT alter results — assembly is index-addressed, so a
depth-D run returns exactly the depth-1 output, and a preempted actor holding K in-flight
partitions requeues all K, each delivered exactly once. These run against a fake Ray so
the ordering and heal invariants are checked deterministically.
"""

from __future__ import annotations

import collections
import sys
import types

import pytest

from batcher.carbonite.resilience import RecoveryPolicy
from batcher.ml.gpu import _INFLIGHT_DEPTH_MAX, recommend_inflight_depth


def _raise(exc: BaseException):
    raise exc


def _install_fake_ray(monkeypatch) -> tuple[type, type]:
    exc = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    exc.RayError = RayError
    exc.RayTaskError = RayTaskError
    ray_mod = types.ModuleType("ray")
    ray_mod.exceptions = exc
    ray_mod.wait = lambda refs, num_returns=1: ([refs[0]], refs[1:])
    ray_mod.get = lambda ref: ref()
    ray_mod.kill = lambda actor: None
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return RayError, RayTaskError


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return lambda: self._fn(*args, **kwargs)


class _FakeActor:
    def __init__(self) -> None:
        self.run = _Remote(lambda part, idx=0: [f"out-{part}"])
        self.gpu_stats = _Remote(lambda: 0.4)


# --- recommend_inflight_depth ------------------------------------------------------------


def test_out_of_box_depth_is_double_buffered():
    # >90%-GPU-out-of-the-box lever: the shipped default submits 2 partitions ahead so the
    # device stays fed across the dispatch/gather round-trip on the FIRST run (no measurement
    # yet). One-at-a-time (1) would idle the GPU between partitions.
    from batcher.config.config import DistributedConfig

    assert DistributedConfig().map_inflight_depth == 2
    # With no utilization measurement, the out-of-box depth is kept as-is (the double buffer),
    # and the adaptive loop only ever deepens it further from a measured low-utilization run.
    assert recommend_inflight_depth(None, 2) == 2


def test_recommend_depth_none_keeps_default():
    assert recommend_inflight_depth(None, 1) == 1
    assert recommend_inflight_depth(None, 4) == 4


def test_recommend_depth_saturated_keeps_default():
    assert recommend_inflight_depth(0.95, 2) == 2


def test_recommend_depth_starved_grows():
    assert recommend_inflight_depth(0.3, 1) == 4  # 4x
    assert recommend_inflight_depth(0.6, 1) == 2  # partly fed -> 2x


def test_recommend_depth_capped():
    assert recommend_inflight_depth(0.1, 8) == _INFLIGHT_DEPTH_MAX  # 8*4 clamped to 16


# --- _pipeline_actor_pool: depth-D == depth-1, partition-ordered -------------------------


@pytest.mark.parametrize("depth", [1, 2, 4, 16])
def test_pipeline_pool_order_invariant_to_depth(monkeypatch, depth):
    from batcher.dist.executors import map as mapmod

    _install_fake_ray(monkeypatch)
    actors = [_FakeActor() for _ in range(3)]
    parts = [f"p{i}" for i in range(10)]
    results = mapmod._pipeline_actor_pool(actors, parts, depth)
    assert results == [[f"out-p{i}"] for i in range(10)]  # complete + partition-ordered


def test_pipeline_pool_empty_partitions(monkeypatch):
    from batcher.dist.executors import map as mapmod

    _install_fake_ray(monkeypatch)
    assert mapmod._pipeline_actor_pool([_FakeActor()], [], 4) == []


# --- _drive_actor_pool at depth>1: heal reclaims every orphaned partition exactly once ---


def test_drive_pool_depth_gt1_ordering(monkeypatch):
    from batcher.dist.executors import map as mapmod

    _install_fake_ray(monkeypatch)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 4)

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0, write_spec=None):
            return _FakeActor()

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)
    parts = [f"p{i}" for i in range(8)]
    results, _, _ = mapmod._drive_actor_pool(
        None, parts, {}, min_size=2, max_size=2, policy=RecoveryPolicy(max_attempts=3)
    )
    assert results == [[f"out-p{i}"] for i in range(8)]


def test_drive_pool_reclaims_all_refs_of_a_dead_actor(monkeypatch):
    from batcher.dist.executors import map as mapmod

    RayError, _ = _install_fake_ray(monkeypatch)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 2)

    runs: collections.Counter = collections.Counter()
    spawned: list = []

    class _PoisonActor:
        def __init__(self, idx: int) -> None:
            self.idx = idx
            self.run = _Remote(self._run)
            self.gpu_stats = _Remote(lambda: 0.3)

        def _run(self, part, idx=0):
            runs[part] += 1
            if self.idx == 0:  # the first actor is preempted: every in-flight call dies
                raise RayError("actor 0 preempted")
            return [f"out-{part}"]

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0, write_spec=None):
            a = _PoisonActor(len(spawned))
            spawned.append(a)
            return a

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)
    # One actor, depth 2 -> actor 0 holds both partitions in flight when it dies.
    parts = ["p0", "p1"]
    results, _, _ = mapmod._drive_actor_pool(
        None, parts, {}, min_size=1, max_size=1, policy=RecoveryPolicy(max_attempts=3)
    )
    assert results == [["out-p0"], ["out-p1"]]  # both delivered, in order
    assert results.count(None) == 0  # nothing dropped
    assert len(spawned) >= 2  # actor 0 died, a healthy replacement was spawned


def test_drive_pool_reraises_deterministic_error_at_depth(monkeypatch):
    from batcher.dist.executors import map as mapmod

    _, RayTaskError = _install_fake_ray(monkeypatch)
    monkeypatch.setattr(mapmod, "_actor_inflight_depth", lambda: 4)

    class _BugActor:
        def __init__(self) -> None:
            self.run = _Remote(lambda part, idx=0: _raise(RayTaskError("UDF bug")))
            self.gpu_stats = _Remote(lambda: None)

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0, write_spec=None):
            return _BugActor()

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)
    with pytest.raises(RayTaskError):
        mapmod._drive_actor_pool(
            None, ["p0", "p1"], {}, min_size=1, max_size=1, policy=RecoveryPolicy(max_attempts=3)
        )


# --- _actor_inflight_depth: envelope wins, clamped -----------------------------------------


def test_actor_inflight_depth_reads_envelope(monkeypatch):
    from batcher.dist.executors import map as mapmod
    from batcher.dist.executors import ray_runtime as rr
    from batcher.plan.resource import SchedulingEnvelope

    tok = rr.set_scheduling_envelope(SchedulingEnvelope(inflight_depth=8))
    try:
        assert mapmod._actor_inflight_depth() == 8
    finally:
        rr.reset_scheduling_envelope(tok)


def test_actor_inflight_depth_clamped(monkeypatch):
    from batcher.dist.executors import map as mapmod
    from batcher.dist.executors import ray_runtime as rr
    from batcher.plan.resource import SchedulingEnvelope

    tok = rr.set_scheduling_envelope(SchedulingEnvelope(inflight_depth=999))
    try:
        assert mapmod._actor_inflight_depth() == mapmod._MAP_INFLIGHT_MAX
    finally:
        rr.reset_scheduling_envelope(tok)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_inflight_depth_caps_on_tight_vram():
    # A starved GPU submits deeper; but a VRAM-tight pipeline keeps the shallow default so
    # deep submission (several partitions' activations in flight) can't OOM the device.
    from batcher.ml.gpu import recommend_inflight_depth

    assert recommend_inflight_depth(0.2, 2) > 2  # starved, ample VRAM → deepen
    assert recommend_inflight_depth(0.2, 2, peak_vram_fraction=0.9) == 2  # tight VRAM → shallow
    assert recommend_inflight_depth(0.2, 2, peak_vram_fraction=0.3) > 2  # roomy → deepen stands
