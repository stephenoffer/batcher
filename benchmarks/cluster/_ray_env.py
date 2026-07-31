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
        requirement = requirement.split(sep)[0]
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
    strip_broken_runtime_env_hook()
    import ray

    if ray.is_initialized():
        return
    runtime_env: dict[str, Any] = {"pip": worker_pip(pip)}
    if env_vars:
        runtime_env["env_vars"] = env_vars
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
