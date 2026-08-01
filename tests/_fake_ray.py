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

    exc.RayError = RayError
    exc.RayTaskError = RayTaskError

    ray_mod = types.ModuleType("ray")
    ray_mod.exceptions = exc
    # `timeout` is accepted and ignored: a ref here is a thunk that is always ready, so
    # there is nothing to wait for. Matching the real signature matters — a barrier that
    # polls with a deadline (to report a stage the cluster cannot schedule) would otherwise
    # fail against the stub for a reason that has nothing to do with what it is testing.
    ray_mod.wait = lambda refs, num_returns=1, timeout=None: ([refs[0]], refs[1:])
    ray_mod.get = lambda ref: ref()
    ray_mod.kill = lambda actor: None

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return RayError, RayTaskError
