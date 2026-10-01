"""Ray bring-up/tear-down for the distributed integration tests.

Bringing up Ray for a test has to survive a *managed* environment (some platforms,
Kubernetes) that exports a runtime-env hook pointing at a plugin module absent
from this process (e.g. ``cgroup_runtime_plugin``), which makes a bare
``ray.init`` raise ``ModuleNotFoundError`` before any test runs — and that
already has a cluster running, so ``num_cpus`` can't be pinned. `init_test_ray`
reuses the engine's neutralize-the-broken-hook fix and falls back to attaching to
the running cluster, so the distributed suite runs both on a laptop (a fresh local
cluster) and against a managed cluster (attach) instead of erroring at setup.

These live in a uniquely-named module rather than a `conftest` for the same reason
`tests/_harness.py` exists: a `conftest` is imported under the bare name ``conftest``,
so ``from conftest import init_test_ray`` binds to whichever `conftest` pytest imported
first. In a run spanning `tests/differential` and `tests/integration` that is the wrong
module, and the import fails. A uniquely-named module is unambiguous from anywhere.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator

import pytest

from batcher.dist.executors.ray_runtime.lifecycle import _platform_env_hook_disabled

__all__ = ["init_test_ray", "local_ray_resources", "ray_session_fixture", "shutdown_test_ray"]


#: The managed-workspace variable that pins the resources of every Ray node started in this
#: process. The Anyscale head exports it with `"CPU": 0` so no work lands on the head.
_RESOURCE_OVERRIDE = "RAY_OVERRIDE_RESOURCES"


@contextlib.contextmanager
def local_ray_resources() -> Iterator[None]:
    """Let a test's own `ray.init(address="local", num_cpus=N)` really get its `N` CPUs.

    A managed head exports `RAY_OVERRIDE_RESOURCES={"CPU": 0, ...}`, and Ray applies it to
    *every* node started in the process, a local one included — so the local instance a test
    asked for came back advertising no CPU, and anything submitted to it pended forever. That
    is why `tests/migrate/test_executed_ray_data.py` hung for 600 s per case on a groupby of
    three rows, and why the local fan-out in `test_gpu_fanout.py` skipped. With the variable
    unset the same groupby finishes in about 6 s.

    The variable is restored on exit: a later module attaching to the session's cluster does
    not start a node, so it never read the override, but leaving the process environment as it
    was found keeps modules order-independent.
    """
    prior = os.environ.pop(_RESOURCE_OVERRIDE, None)
    try:
        yield
    finally:
        if prior is not None:
            os.environ[_RESOURCE_OVERRIDE] = prior


def init_test_ray(num_cpus: int) -> bool:
    """Start a local Ray of `num_cpus` cpus, or attach to a cluster already running.

    Returns whether this call *started* Ray (so the fixture knows whether to shut it
    down — a pre-existing / attached cluster is shared and must be left running).
    """
    import ray

    if ray.is_initialized():
        return False
    with _platform_env_hook_disabled():
        try:
            ray.init(
                num_cpus=num_cpus,
                include_dashboard=False,
                logging_level="ERROR",
                ignore_reinit_error=True,
            )
        except (ValueError, ConnectionError):
            # A cluster is already running but wants to be attached to (no local pinning).
            ray.init(address="auto", ignore_reinit_error=True)
    _require_schedulable_cpu(ray, num_cpus)
    return True


def _require_schedulable_cpu(ray, num_cpus: int) -> None:
    """Refuse a Ray that advertises no CPU, loudly, instead of letting the suite hang on it.

    **A Ray with no `CPU` resource does not fail a distributed test, it hangs it.** Every task
    the engine submits asks for `num_cpus > 0`, so on such an instance they all pend forever
    with nothing running: `ray status` shows zero usage and no pending demands, and the driver
    sits in the shuffle barrier's `ray.wait` until the suite is killed. Nothing about that
    reads as "the wrong Ray" — it reads as a deadlock in the engine, and attributing it cost
    this session hours across `test_distributed_unordered_limit`, `test_result_cache` and
    several others, all of which resume failing *fast* once this fires.

    It is reachable because `ray.init()` above resolves its own address when none is given. On
    a managed workspace that can land on an instance the platform started with `num_cpus=0`
    precisely so no work runs on the head — measured here as `cluster_resources()` carrying
    `memory`, `object_store_memory` and node labels and **no `CPU` key at all**, while the
    real 1,024-core cluster ran on a different port.

    Deliberately an error rather than a repair: a helper that silently substitutes some other
    cluster would be worse than one that says what it got. A local Ray on such a head came back
    with `CPU: None` only because the platform's `RAY_OVERRIDE_RESOURCES` pins every node
    started there to zero CPUs; a test that wants a local instance starts it under
    [`local_ray_resources`].

    The fix is an address, and it is verified rather than suggested: with `RAY_ADDRESS` set to
    the scheduling cluster, `test_distributed_unordered_limit` goes from hanging indefinitely
    to **4 passed in 5.9 s**.

    Args:
        ray: The imported `ray` module.
        num_cpus: What the caller asked for, named in the error.

    Raises:
        RuntimeError: When the attached Ray advertises no schedulable CPU.
    """
    if float(ray.cluster_resources().get("CPU", 0.0)) > 0:
        return
    raise RuntimeError(
        f"the Ray this suite attached to advertises no CPU resource (asked for {num_cpus}); "
        "every distributed task would pend forever rather than fail. Set `RAY_ADDRESS` to a "
        "cluster that schedules work — on a managed workspace the head often runs with "
        "`num_cpus=0`, so both a bare `ray.init()` and a local one land on an instance that "
        "cannot run a task. `ray status` names the instances it can see."
    )


def shutdown_test_ray(started: bool) -> None:
    """Shut Ray down only if `init_test_ray` started it (never tear down a shared one)."""
    if started:
        import ray

        ray.shutdown()


def ray_session_fixture(num_cpus: int) -> Callable[[], Iterator[None]]:
    """Return a module-scoped, autouse fixture holding Ray up for the whole test module.

    Bind it to the name `_ray_session` at module level so pytest collects it:
    ``_ray_session = ray_session_fixture(4)``. The fixture starts (or attaches to) Ray
    with `init_test_ray(num_cpus)` before the module's first test and calls
    `shutdown_test_ray` after its last one.
    """

    @pytest.fixture(scope="module", autouse=True)
    def _ray_session() -> Iterator[None]:
        started = init_test_ray(num_cpus)
        yield
        shutdown_test_ray(started)

    return _ray_session
