"""A minimal in-process stand-in for `ray`, for the distributed tests that need no cluster.

The `dist` recovery, in-flight, and resident-pool tests all drive the same code paths and all
need the same thing: a `ray` module whose object refs are plain thunks, so a test can make one
"task" fail by handing back a callable that raises. Three test modules had written the same
18-line installer; it lives here once, imported the way `_harness` is.
"""

from __future__ import annotations

import sys
import types


def install_fake_ray(monkeypatch) -> tuple[type, type]:
    """Install a minimal `ray` whose refs are thunks.

    `ray.get(ref)` calls the thunk, returning its value or raising whatever it raises, and
    `ray.wait` pops one ref in FIFO order — enough for the gather loops under test, and
    deterministic in a way a real cluster is not.

    Args:
        monkeypatch: The pytest fixture; the modules are removed again at teardown.

    Returns:
        The fake `RayError` and `RayTaskError` classes, for tests that raise them.
    """
    exc = types.ModuleType("ray.exceptions")

    class RayError(Exception):
        pass

    class RayTaskError(RayError):
        pass

    class OutOfMemoryError(RayError):
        pass

    exc.RayError = RayError
    exc.RayTaskError = RayTaskError
    exc.OutOfMemoryError = OutOfMemoryError

    ray_mod = types.ModuleType("ray")
    ray_mod.exceptions = exc
    # `timeout` is accepted and ignored: a ref here is a thunk that is always ready, so
    # there is nothing to wait for. Matching the real signature matters — a barrier that
    # polls with a deadline (to report a stage the cluster cannot schedule) would otherwise
    # fail against the stub for a reason that has nothing to do with what it is testing.
    ray_mod.wait = lambda refs, num_returns=1, timeout=None: ([refs[0]], refs[1:])
    ray_mod.get = lambda ref: ref()
    ray_mod.kill = lambda actor: None

    # `ray.util.placement_group` is a SEPARATE module, so patching `ray` alone does not stub
    # it: `from ray.util.placement_group import placement_group` resolved to the real one,
    # whose auto-connect (`RAY_ENABLE_AUTO_CONNECT`) called `ray.init()` and reserved bundles
    # on whatever cluster the box could reach. On this machine that is the shared 65-node
    # fleet, from a test whose whole point is to need no cluster. It also outlived the test:
    # Ray's global state keeps answering `cluster_resources()` after `shutdown()`, so every
    # later test that sizes itself from the cluster read 1,024 cores instead of this box's
    # 15 — which is how `test_map_scalability` came to pass alone and fail in a suite.
    pg_mod = types.ModuleType("ray.util.placement_group")
    pg_mod.placement_group = lambda bundles, strategy="PACK", **kwargs: _FakePlacementGroup(
        bundles, strategy
    )
    pg_mod.remove_placement_group = lambda pg: None
    sched_mod = types.ModuleType("ray.util.scheduling_strategies")
    sched_mod.PlacementGroupSchedulingStrategy = _FakeStrategy
    sched_mod.NodeAffinitySchedulingStrategy = _FakeStrategy
    util_mod = types.ModuleType("ray.util")
    util_mod.placement_group = pg_mod
    util_mod.scheduling_strategies = sched_mod
    ray_mod.util = util_mod

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    monkeypatch.setitem(sys.modules, "ray.util", util_mod)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", pg_mod)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", sched_mod)
    return RayError, RayTaskError


class _FakeStrategy:
    """A scheduling strategy that records its arguments and schedules nothing."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakePlacementGroup:
    """A reservation that is ready the moment it is asked for.

    `ready()` returns a thunk because that is what a ref is here, so the caller's
    `ray.wait([pg.ready()], timeout=...)` sees it as ready without a cluster.
    """

    def __init__(self, bundles: list[dict], strategy: str) -> None:
        self.bundle_specs = list(bundles)
        self.strategy = strategy

    def ready(self):
        return lambda: True
