"""Phase-1 fault tolerance: the distributed map/inference recovery loop.

A preempted GPU node must not fail an inference stage. The stateless gather resubmits
a lost partition onto a survivor, and the actor pool replaces a dead actor and
reassigns its partition — bounded, and re-raising a *deterministic* UDF error
immediately rather than wasting recompute rounds. These exercise that logic against a
fake Ray facade, so the branches are tested deterministically without real worker
crashes.
"""

from __future__ import annotations

import collections
import sys
import types

import pytest

from batcher.carbonite.resilience import RecoveryPolicy


def _raise(exc: BaseException):
    raise exc


def _install_fake_ray(monkeypatch) -> tuple[type, type]:
    """Install a minimal `ray` whose refs are thunks: `ray.get(ref)` calls the thunk
    (returning its value or raising), `ray.wait` pops one in FIFO order."""
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


def test_gather_resubmits_a_lost_partition(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = _install_fake_ray(monkeypatch)
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        if idx == 1 and calls[idx] == 1:
            return lambda: _raise(RayError("preempted"))  # worker loss, once
        return lambda i=idx: [f"r{i}"]

    out = gather_map_results(submit, 3, RecoveryPolicy(max_attempts=3))
    assert out == [["r0"], ["r1"], ["r2"]]  # every partition produced, in order
    assert calls[1] == 2  # partition 1 was resubmitted exactly once


def test_gather_reraises_deterministic_udf_error(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    _, RayTaskError = _install_fake_ray(monkeypatch)

    def submit(idx):
        return lambda: _raise(RayTaskError("a real bug in the UDF"))

    # A deterministic error is not preemption — fail fast, do not burn attempts.
    with pytest.raises(RayTaskError):
        gather_map_results(submit, 1, RecoveryPolicy(max_attempts=5))


def test_gather_gives_up_after_max_attempts(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = _install_fake_ray(monkeypatch)
    calls: collections.Counter = collections.Counter()

    def submit(idx):
        calls[idx] += 1
        return lambda: _raise(RayError("node never comes back"))

    with pytest.raises(RayError):
        gather_map_results(submit, 1, RecoveryPolicy(max_attempts=2))
    # initial attempt + max_attempts resubmissions
    assert calls[0] == 3


def test_actor_pool_replaces_a_dead_actor(monkeypatch):
    from batcher.dist.executors import map as mapmod

    RayError, _ = _install_fake_ray(monkeypatch)
    crashed: set = set()

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return lambda: self._fn(*args, **kwargs)

    class _FakeActor:
        def __init__(self) -> None:
            self.run = _Remote(self._run)
            self.gpu_stats = _Remote(lambda: None)

        def _run(self, part):
            # The partition "p1" preempts its actor the first time it is seen.
            if part == "p1" and part not in crashed:
                crashed.add(part)
                raise RayError("actor preempted")
            return [f"out-{part}"]

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0):
            return _FakeActor()

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)

    parts = ["p0", "p1", "p2"]
    results, _util = mapmod._drive_actor_pool(
        plan0=None,
        partitions=parts,
        opts={},
        min_size=2,
        max_size=2,
        policy=RecoveryPolicy(max_attempts=3),
    )
    assert results == [["out-p0"], ["out-p1"], ["out-p2"]]  # all produced once
    assert crashed == {"p1"}  # the one simulated preemption, recovered


def test_actor_pool_reraises_deterministic_error(monkeypatch):
    from batcher.dist.executors import map as mapmod

    _, RayTaskError = _install_fake_ray(monkeypatch)

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return lambda: self._fn(*args, **kwargs)

    class _FakeActor:
        def __init__(self) -> None:
            self.run = _Remote(lambda part: _raise(RayTaskError("UDF bug")))
            self.gpu_stats = _Remote(lambda: None)

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0):
            return _FakeActor()

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)

    with pytest.raises(RayTaskError):
        mapmod._drive_actor_pool(
            plan0=None,
            partitions=["p0"],
            opts={},
            min_size=1,
            max_size=1,
            policy=RecoveryPolicy(max_attempts=3),
        )
