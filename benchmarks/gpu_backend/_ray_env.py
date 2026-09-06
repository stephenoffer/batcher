"""Shared Ray bootstrap for the standalone GPU-backend benchmark scripts.

Every script in this directory needs the same two things before it can talk to a
cluster: drop a runtime-env hook the managed host exports but whose module is not
importable here (Ray imports it during ``ray.init`` and crashes), then connect to
the existing cluster with a neutralized ``pip`` block. That was copy-pasted into
eleven scripts verbatim; it lives here once instead.

Imported as a *sibling* module (``from _ray_env import init_ray``) rather than as
``benchmarks.gpu_backend._ray_env``, because these scripts are run directly
(``python benchmarks/gpu_backend/tpch_q6_gpu.py``), which puts this directory —
not the repo root — on ``sys.path``.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["init_ray", "strip_broken_runtime_env_hook", "worker_pip"]


def _req_name(requirement: str) -> str:
    """The distribution name a pip requirement string names, lowercased."""
    for sep in ("=", "<", ">", "[", "!", "~", " "):
        requirement = requirement.split(sep, maxsplit=1)[0]
    return requirement.strip().lower()


def worker_pip(extra: Any = None) -> list[str]:
    """The pip set worker actors need: `extra`, plus the driver's own numpy.

    Ray pickles a numpy array (and every dtype inside a batch handed to a UDF) by module
    path, and numpy 2 moved `numpy.core` to `numpy._core`. A driver on numpy 2 against a
    cluster image on numpy 1 therefore kills **every** actor it starts with
    `ModuleNotFoundError: No module named 'numpy._core.numeric'`, before any user code
    runs — so a benchmark reports a timeout rather than a number. Pinning the workers to
    the driver's version is the side that can be changed from here.

    A caller pinning numpy itself (a cuDF wheel needing numpy 1) keeps its own pin: that
    constraint is tighter, and the driver must then match *it*.

    The twin of `cluster._ray_env.worker_pip`. These two bootstraps are deliberately
    separate because each directory's scripts are run directly, putting only their own
    directory on `sys.path`.

    Args:
        extra: Requirements the caller needs on the workers.

    Returns:
        The pip requirement list to hand `runtime_env`.
    """
    import numpy

    reqs = list(extra or [])
    named = {_req_name(r) for r in reqs}
    if "numpy" in named:
        return reqs
    return [f"numpy=={numpy.__version__}", *reqs]


def strip_broken_runtime_env_hook(*, unconditional: bool = False) -> None:
    """Drop ``RAY_RUNTIME_ENV_HOOK``/``RAY_RUNTIME_ENV_PLUGINS`` before ``ray.init``.

    A managed host env (e.g. a ``cgroup_runtime_plugin``) may export a runtime-env hook
    that Ray imports during ``ray.init``; outside that runtime the module is absent and
    init crashes. A hook pointing at an unimportable module is broken regardless, so
    removing it is strictly safer — and a no-op where the module is present.

    Args:
        unconditional: Drop the hook even when its module *is* importable. Some
            workspaces export a hook that injects a default dev-pip set (containing a
            broken local editable) into every task, including ones that declare no
            runtime env of their own. Those scripts ship their dependencies per-task
            themselves, so the hook is pure liability.
    """
    import importlib.util

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        value = os.environ.get(var)
        if not value:
            continue
        if unconditional:
            os.environ.pop(var, None)
            continue
        head = value.lstrip("[{\"' ").split(".")[0].split("[")[0]
        if head and importlib.util.find_spec(head) is None:
            os.environ.pop(var, None)


def _require_release() -> None:
    """Refuse a dev-profile engine before any of these scripts measures anything.

    Every benchmark in this directory is invoked as ``python benchmarks/<dir>/<name>.py``,
    so only its own directory is on ``sys.path`` and ``envinfo`` two levels up is not
    importable without help. Doing it here rather than in nineteen scripts means a new one
    cannot forget: they all reach the cluster through the init functions below.

    A dev build is 8-60x slower than release, so a number taken from one compares an
    unoptimized Batcher against release comparators. ``BENCH_ALLOW_DEBUG_BUILD=1`` overrides.

    Deliberately *not* also calling ``require_quiet_box``: the work in these benchmarks
    happens on cluster workers, so the driver's run queue is not the contention signal that
    would invalidate the measurement, and refusing on it would be a false negative.
    """
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from envinfo import require_release_build

    require_release_build()


#: How long to wait for the one-task numpy probe before giving up. Short: the probe runs in
#: the cluster image's own environment with no `runtime_env`, so it either answers at once or
#: the cluster is too busy to tell us anything useful.
_NUMPY_PROBE_TIMEOUT_S = 20.0


def _cluster_numpy_major(ray) -> int | None:
    """The numpy MAJOR version a worker imports, or `None` when it cannot be read.

    Runs with no `runtime_env`, so it is answered by the cluster image itself and costs no
    environment build -- which is the whole point, since deciding whether to *ask* for one is
    what it is for.
    """

    @ray.remote(num_cpus=0)
    def _probe() -> str:
        import numpy

        return numpy.__version__

    try:
        ready, _ = ray.wait([_probe.remote()], timeout=_NUMPY_PROBE_TIMEOUT_S)
        if not ready:
            return None
        return int(str(ray.get(ready[0])).split(".", 1)[0])
    except Exception:
        return None


def _numpy_pin_needed(ray) -> bool:
    """Whether the workers need the driver's numpy pinned into a `runtime_env`.

    **A `pip` block is never free, even when every package in it is already installed.** Ray
    builds a virtualenv for that environment hash and resolves the requirements into it, once
    per node a task first lands on. Measured on this 6-GPU cluster, a fan-out that touches
    every node: the same six GPU shards took **168.1s on the first round and 0.3s on the
    second and third** -- identical tasks, identical data, in one process. All of it was the
    per-node build, and none of it was compute (the task bodies summed to ~1.7s).

    So the pin is only worth that when it actually protects something. What it protects
    against is the numpy **1 vs 2** boundary: Ray pickles arrays by module path and numpy 2
    moved `numpy.core` to `numpy._core`, so a numpy-2 driver against a numpy-1 image kills
    every actor before user code runs. That is a major-version question, and pinning the
    driver's *exact* version asked a stricter one -- an image on 2.1.0 under a 2.2.6 driver
    pickles fine and was paying a full environment build to be told so.

    Returns True when the majors differ or the probe could not reach a conclusion. An
    inconclusive probe keeps the old behaviour exactly, so this can cost what it cost before
    but never more, and never introduces a failure the previous version did not have.
    """
    import numpy

    theirs = _cluster_numpy_major(ray)
    return theirs is None or theirs != int(numpy.__version__.split(".", 1)[0])


def _batcher_py_modules() -> list[str]:
    """The driver's `batcher` package directory, for `runtime_env["py_modules"]`."""
    import os

    import batcher

    return [os.path.dirname(os.path.abspath(batcher.__file__))]


def init_ray(
    *,
    pip: Any = None,
    env_vars: dict[str, str] | None = None,
    unconditional_hook_strip: bool = False,
    ship_batcher: bool = False,
) -> None:
    """Strip the broken hook, then attach to the running cluster if not already attached.

    `ship_batcher` is required by any script here that drives **Batcher's own distributed
    path** rather than only Ray. Without it the workers must already carry the package, and
    on a workspace install they do not: what made `import batcher` work on a worker was the
    managed hook's `working_dir`, and `unconditional_hook_strip=True` removes exactly that.

    The failure is not a clean error. The shard tasks raise `ModuleNotFoundError` on the
    worker, Ray retries them under `max_retries`, and each retry pays a fresh worker start —
    so the stage *eventually* returns a correct answer having spent all of its time in
    retries. Measured on this 6-GPU cluster, `relational_vs_raydata.py`'s GPU arm: **168s for
    six shards whose task bodies sum to 1.7s**, and 0.2s on the immediately following round
    once the workers were warm. The whole of that gap was import retries, and none of it was
    the GPU.

    Args:
        pip: Extra requirements the workers need, job-wide. The driver's own numpy is
            added only when the cluster's numpy major differs (see `_numpy_pin_needed`).
        env_vars: Environment variables to propagate to worker actors. A driver-process
            ``os.environ`` does not otherwise reach a remote actor.
        unconditional_hook_strip: Forwarded to `strip_broken_runtime_env_hook`.
        ship_batcher: Ship the driver's Batcher to the workers as `py_modules`, so a script
            that uses Batcher's distributed path runs the same build on both sides.
    """
    _require_release()
    strip_broken_runtime_env_hook(unconditional=unconditional_hook_strip)
    import ray

    if ray.is_initialized():
        return
    # Attach first with no `pip` block, so the common case never builds an environment.
    # The pin can only be decided by asking a worker, and asking needs a live connection.
    modules = _batcher_py_modules() if ship_batcher else []
    base: dict[str, Any] = {"env_vars": env_vars} if env_vars else {}
    if modules:
        base["py_modules"] = modules
    ray.init(address="auto", runtime_env=base, logging_level="ERROR", log_to_driver=False)
    if not pip and not _numpy_pin_needed(ray):
        return
    runtime_env: dict[str, Any] = {"pip": worker_pip(pip)}
    if env_vars:
        runtime_env["env_vars"] = env_vars
    if modules:
        runtime_env["py_modules"] = modules
    ray.shutdown()
    ray.init(
        address="auto",
        runtime_env=runtime_env,
        logging_level="ERROR",
        log_to_driver=False,
    )
