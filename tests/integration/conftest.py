"""Shared setup for the distributed integration tests.

Bringing up Ray for a test has to survive a *managed* environment (some platforms,
Kubernetes) that exports a runtime-env hook pointing at a plugin module absent
from this process (e.g. ``cgroup_runtime_plugin``), which makes a bare
``ray.init`` raise ``ModuleNotFoundError`` before any test runs — and that
already has a cluster running, so ``num_cpus`` can't be pinned. `init_test_ray`
reuses the engine's neutralize-the-broken-hook fix and falls back to attaching to
the running cluster, so the distributed suite runs both on a laptop (a fresh local
cluster) and against a managed cluster (attach) instead of erroring at setup.
"""

from __future__ import annotations


def init_test_ray(num_cpus: int) -> bool:
    """Start a local Ray of `num_cpus` cpus, or attach to a cluster already running.

    Returns whether this call *started* Ray (so the fixture knows whether to shut it
    down — a pre-existing / attached cluster is shared and must be left running).
    """
    import ray

    from batcher.dist.executors.ray_runtime.lifecycle import _platform_env_hook_disabled

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
    return True


def shutdown_test_ray(started: bool) -> None:
    """Shut Ray down only if `init_test_ray` started it (never tear down a shared one)."""
    if started:
        import ray

        ray.shutdown()
