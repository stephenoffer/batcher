"""Ray bring-up/tear-down for the distributed integration tests.

Bringing up Ray for a test has to survive a *managed* environment (some platforms,
Kubernetes) that exports a runtime-env hook pointing at a plugin module absent
from this process (e.g. ``cgroup_runtime_plugin``), which makes a bare
``ray.init`` raise ``ModuleNotFoundError`` before any test runs — and that
already has a cluster running, so ``num_cpus`` can't be pinned. `init_test_ray`
reuses the engine's neutralize-the-broken-hook fix and falls back to attaching to
the running cluster, so the distributed suite runs both on a laptop (a fresh local
cluster) and against a managed cluster (attach) instead of erroring at setup.

These live here rather than in `tests/integration/conftest.py` for the same reason
`tests/_harness.py` exists: a `conftest` is imported under the bare name ``conftest``,
so ``from conftest import init_test_ray`` binds to whichever `conftest` pytest imported
first. In a run spanning `tests/differential` and `tests/integration` that is the wrong
module, and the import fails. A uniquely-named module is unambiguous from anywhere.
"""

from __future__ import annotations

from batcher.dist.executors.ray_runtime.lifecycle import _platform_env_hook_disabled

__all__ = ["init_test_ray", "shutdown_test_ray"]


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

    Deliberately an error rather than a repair, because the repair does not exist here.
    Starting a *local* Ray is the obvious fallback and it is the one thing that cannot work
    on this platform: `ray.init(address="local", num_cpus=4)` came back with `CPU: None` and a
    one-CPU task pending on `No available node types can fulfill resource request {'CPU': 1.0}`
    — the workspace's head runs no work, and only the autoscaled fleet schedules. A helper
    that silently substitutes some other cluster would also be worse than one that says what
    it got.

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
