"""Shared bootstrap and helpers for the standalone cluster benchmark scripts.

These scripts differ from the `gpu_backend/` ones in that they exercise Batcher's own
distributed path, so they must *ship the working-tree Batcher* to the workers
(``py_modules``) and point `batcher.config` at the same runtime env the driver used —
otherwise the driver and the workers disagree about which Batcher is running.

The bootstrap was copy-pasted into nine scripts, each differing only in which env vars
it forwarded; `init_batcher_ray` takes those as arguments.

Imported as a *sibling* module (``from _ray_env import init_batcher_ray``) rather than
as ``benchmarks.cluster._ray_env``, because these scripts are run directly
(``python benchmarks/cluster/gpu_pipeline.py``), which puts this directory — not the
repo root — on ``sys.path``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

__all__ = [
    "init_batcher_ray",
    "init_ray",
    "strip_broken_runtime_env_hook",
    "with_timeout",
    "worker_pip",
]


def _req_name(requirement: str) -> str:
    """The distribution name a pip requirement string names, lowercased."""
    for sep in ("=", "<", ">", "[", "!", "~", " "):
        requirement = requirement.split(sep, maxsplit=1)[0]
    return requirement.strip().lower()


def worker_pip(extra: list[str] | None = None) -> list[str]:
    """The pip set worker actors need: `extra`, plus the driver's own numpy.

    Ray pickles a numpy array (and every dtype inside a `RecordBatch` handed to a UDF) by
    module path, and numpy 2 moved `numpy.core` to `numpy._core`. A driver on numpy 2
    against a cluster image on numpy 1 therefore kills **every** actor it starts with
    `ModuleNotFoundError: No module named 'numpy._core.numeric'` — before any user code
    runs, and identically for Batcher and for Ray Data, so a benchmark reports a timeout
    and an `ActorDiedError` rather than a number. Pinning the workers to the driver's
    version is the side that can be changed from here.

    A caller pinning numpy itself (a cuDF build, say, whose wheel needs numpy 1) keeps its
    own pin: that constraint is tighter than this one and the driver must match *it*.

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


def strip_broken_runtime_env_hook() -> None:
    """Drop ``RAY_RUNTIME_ENV_HOOK``/``RAY_RUNTIME_ENV_PLUGINS`` before ``ray.init``.

    A managed host env (e.g. a ``cgroup_runtime_plugin``) may export a runtime-env hook
    that Ray imports during ``ray.init``; outside that runtime the module is absent and
    init crashes. A hook pointing at an unimportable module is broken regardless, so
    removing it is strictly safer — and a no-op where the module is present.
    """
    import importlib.util

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        value = os.environ.get(var)
        if not value:
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


def init_ray(*, env_vars: dict[str, str] | None = None, pip: list[str] | None = None) -> None:
    """Attach to the running cluster *without* shipping the working-tree Batcher.

    For the scripts here that only drive Ray directly (no Batcher distributed path), so
    they need no ``py_modules``. The pip set replaces the inherited workspace one rather
    than adding to it — the cluster image already carries torch and Ray — and always
    carries the driver's numpy (see :func:`worker_pip`).

    Args:
        env_vars: Environment variables to propagate to worker actors. A driver-process
            ``os.environ`` does not otherwise reach a remote actor.
        pip: Extra requirements the workers need (a cuDF build, say).
    """
    _require_release()
    strip_broken_runtime_env_hook()
    import ray

    if ray.is_initialized():
        return
    # Attach first with no `pip` block, so the common case never builds an environment.
    # The pin can only be decided by asking a worker, and asking needs a live connection.
    base: dict[str, Any] = {"env_vars": env_vars} if env_vars else {}
    ray.init(address="auto", runtime_env=base, logging_level="ERROR", log_to_driver=False)
    if not pip and not _numpy_pin_needed(ray):
        return
    runtime_env: dict[str, Any] = {"pip": worker_pip(pip)}
    if env_vars:
        runtime_env["env_vars"] = env_vars
    ray.shutdown()
    ray.init(
        address="auto",
        runtime_env=runtime_env,
        logging_level="ERROR",
        log_to_driver=False,
    )


def init_batcher_ray(
    *,
    forward: tuple[str, ...] = (),
    env_defaults: dict[str, str] | None = None,
    hf_cache: str | None = None,
    **distributed_overrides: Any,
) -> None:
    """Ship the working-tree Batcher to the cluster and attach to it.

    Sets `batcher.config`'s distributed block *and* calls ``ray.init`` with the same
    ``runtime_env``, so the driver and the workers run identical code.

    Args:
        forward: Env var names to propagate from the driver to worker actors, if set.
            A driver-process ``os.environ`` does not otherwise reach a remote actor.
        env_defaults: Env vars to set on workers when absent from the driver env.
        hf_cache: When given, default ``HF_HOME`` to this path (falling back to the
            driver's ``HF_HOME``) so a model downloads ONCE to shared cluster storage
            rather than once per worker.
        **distributed_overrides: Extra fields set on `config.distributed`, e.g.
            ``stream_inference=True``.
    """
    _require_release()
    strip_broken_runtime_env_hook()
    import batcher
    from batcher.config import active_config, set_config

    env_vars = dict(env_defaults or {})
    env_vars.update({k: os.environ[k] for k in forward if k in os.environ})
    if hf_cache is not None:
        env_vars.setdefault("HF_HOME", os.environ.get("HF_HOME", hf_cache))

    pkg = os.path.dirname(os.path.abspath(batcher.__file__))
    runtime_env = {"py_modules": [pkg], "pip": worker_pip(), "env_vars": env_vars}

    base = active_config()
    set_config(
        base.replace(
            distributed=dataclasses.replace(
                base.distributed,
                ray_address="auto",
                runtime_env=runtime_env,
                **distributed_overrides,
            )
        )
    )
    import ray

    if not ray.is_initialized():
        ray.init(
            address="auto",
            runtime_env=runtime_env,
            logging_level="ERROR",
            log_to_driver=False,
        )


def with_timeout(fn, timeout_s: float):
    """Wrap `fn` so each call raises `TimeoutError` if it runs past `timeout_s`.

    Runs the call on a **daemon** thread and waits up to `timeout_s`. A timed-out call's
    thread is abandoned but, being a daemon, never keeps the process alive — so a
    pathological engine (e.g. Ray Data's distributed join) cannot leave a zombie driver
    holding cluster actors after the sweep ends. (A plain ThreadPoolExecutor uses
    non-daemon threads, which did exactly that.)

    Args:
        fn: The zero-argument call to guard.
        timeout_s: Seconds to wait before giving up on it.

    Returns:
        A zero-argument callable running `fn` under that deadline.
    """
    import threading

    def wrapped():
        box: dict = {}

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:
                box["e"] = e

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            raise TimeoutError
        if "e" in box:
            raise box["e"]
        return box.get("v")

    return wrapped
