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

import pytest

from _fake_ray import install_fake_ray
from batcher.carbonite.resilience import RecoveryPolicy


def _raise(exc: BaseException):
    raise exc


def test_gather_resubmits_a_lost_partition(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = install_fake_ray(monkeypatch)
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

    _, RayTaskError = install_fake_ray(monkeypatch)

    def submit(idx):
        return lambda: _raise(RayTaskError("a real bug in the UDF"))

    # A deterministic error is not preemption — fail fast, do not burn attempts.
    with pytest.raises(RayTaskError):
        gather_map_results(submit, 1, RecoveryPolicy(max_attempts=5))


def test_gather_gives_up_after_max_attempts(monkeypatch):
    from batcher.dist.executors.ray_runtime import gather_map_results

    RayError, _ = install_fake_ray(monkeypatch)
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

    RayError, _ = install_fake_ray(monkeypatch)
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

        def _run(self, part, idx=0):
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
        def remote(cls, plan0, write_spec=None):
            return _FakeActor()

    monkeypatch.setattr(mapmod, "_MapActor", _FakeMapActor)

    parts = ["p0", "p1", "p2"]
    results, _util, _vram = mapmod._drive_actor_pool(
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

    _, RayTaskError = install_fake_ray(monkeypatch)

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return lambda: self._fn(*args, **kwargs)

    class _FakeActor:
        def __init__(self) -> None:
            self.run = _Remote(lambda part, idx=0: _raise(RayTaskError("UDF bug")))
            self.gpu_stats = _Remote(lambda: None)

    class _FakeMapActor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, plan0, write_spec=None):
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


def _install_fake_ray_with_fatal(monkeypatch) -> tuple[type, type]:
    """`_install_fake_ray`, plus the `RuntimeEnvSetupError` that Ray really defines —
    a `RayError` that is emphatically NOT a worker loss."""
    RayError, _ = install_fake_ray(monkeypatch)
    exc = sys.modules["ray.exceptions"]

    class RuntimeEnvSetupError(RayError):
        pass

    exc.RuntimeEnvSetupError = RuntimeEnvSetupError
    return RayError, RuntimeEnvSetupError


def test_gather_reraises_a_broken_runtime_env_instead_of_blaming_workers(monkeypatch):
    """A `RuntimeEnvSetupError` is a `RayError` but not a death: retrying it cannot help,
    and treating it as worker loss would blame a healthy host per retry until the whole
    fleet looked dead. It must surface immediately, with `on_lost` never fired."""
    from batcher.dist.executors.ray_runtime import gather_map_results

    _, RuntimeEnvSetupError = _install_fake_ray_with_fatal(monkeypatch)
    lost: list[int] = []

    def submit(idx: int):
        return lambda: _raise(RuntimeEnvSetupError("bad runtime_env"))

    with pytest.raises(RuntimeEnvSetupError):
        gather_map_results(submit, 2, RecoveryPolicy(max_attempts=3), on_lost=lost.append)
    assert lost == []  # no healthy worker was blamed


def test_map_barrier_relocates_a_lost_worker_onto_a_survivor(monkeypatch):
    """A worker that dies during the map barrier has its source relaunched on a live
    worker under the SAME src, and only the dead HOST is recorded — never the source id
    of a slot that merely happened to be relocated onto it."""
    from batcher.dist.executors.ray_runtime import map_barrier

    RayError, _ = _install_fake_ray_with_fatal(monkeypatch)
    dead_hosts = {1}
    launched: list[tuple[int, int]] = []

    def launch(host: int, src: int):
        launched.append((host, src))
        if host in dead_hosts:
            return lambda: _raise(RayError("preempted"))
        return lambda: f"addr{host}"

    addrs, dead = map_barrier(4, launch, RecoveryPolicy(max_attempts=3))

    assert dead == {1}  # only the dead host, and exactly once
    assert len(addrs) == 4 and all(a is not None for a in addrs)
    # src 1 was relaunched on some live host; every source produced an address.
    relocated = [h for h, s in launched if s == 1 and h != 1]
    assert relocated and all(h not in dead_hosts for h in relocated)


def test_map_barrier_raises_when_every_worker_is_gone(monkeypatch):
    """With no survivor there is nowhere to recompute: fail loud, don't loop."""
    from batcher._internal.errors import ResourceError
    from batcher.dist.executors.ray_runtime import map_barrier

    RayError, _ = _install_fake_ray_with_fatal(monkeypatch)

    def launch(host: int, src: int):
        return lambda: _raise(RayError("preempted"))

    with pytest.raises(ResourceError, match="no surviving worker"):
        map_barrier(2, launch, RecoveryPolicy(max_attempts=5))


def test_map_barrier_survives_a_correlated_preemption_wave(monkeypatch):
    """Several workers gone at once, and only the one whose slot fails first is observed.
    A relocation onto a not-yet-observed-dead host must not be charged to the source's
    retry budget — otherwise a `max_attempts=1` barrier fails a query that had survivors
    the whole time. Discovering a dead worker is progress, not a failed attempt."""
    from batcher.dist.executors.ray_runtime import map_barrier

    RayError, _ = _install_fake_ray_with_fatal(monkeypatch)
    dead_hosts = {0, 2}

    def launch(host: int, src: int):
        if host in dead_hosts:
            return lambda: _raise(RayError("preempted"))
        return lambda: f"addr{host}"

    addrs, dead = map_barrier(4, launch, RecoveryPolicy(max_attempts=1))

    assert dead == {0, 2}
    assert len(addrs) == 4 and all(a is not None for a in addrs)


def test_map_barrier_prefers_a_confirmed_live_host_for_relocation(monkeypatch):
    """Once a worker has completed its own source it is provably alive, so a later
    relocation targets it rather than gambling on an unproven host."""
    from batcher.dist.executors.ray_runtime import map_barrier

    RayError, _ = _install_fake_ray_with_fatal(monkeypatch)
    relocations: list[int] = []

    def launch(host: int, src: int):
        if host != src:
            relocations.append(host)
        if host == 1:  # only worker 1 is gone
            return lambda: _raise(RayError("preempted"))
        return lambda: f"addr{host}"

    # The fake `ray.wait` is FIFO, so slot 0 completes before slot 1 fails: worker 0 is
    # the only *confirmed* live host at relocation time.
    addrs, dead = map_barrier(3, launch, RecoveryPolicy(max_attempts=3))

    assert dead == {1}
    assert relocations == [0]  # the proven-live host, not the unproven worker 2
    assert len(addrs) == 3 and all(a is not None for a in addrs)
