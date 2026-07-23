"""Shared Ray bootstrap for the standalone cluster benchmark scripts.

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

__all__ = ["init_batcher_ray", "init_ray", "strip_broken_runtime_env_hook"]


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


def init_ray(*, env_vars: dict[str, str] | None = None) -> None:
    """Attach to the running cluster *without* shipping the working-tree Batcher.

    For the scripts here that only drive Ray directly (no Batcher distributed path), so
    they need no ``py_modules``. ``pip=None`` *neutralizes* the inherited workspace pip
    set rather than requesting no packages — the cluster image already carries what
    these need.

    Args:
        env_vars: Environment variables to propagate to worker actors. A driver-process
            ``os.environ`` does not otherwise reach a remote actor.
    """
    strip_broken_runtime_env_hook()
    import ray

    if ray.is_initialized():
        return
    runtime_env: dict[str, Any] = {"pip": None}
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
    runtime_env = {"py_modules": [pkg], "pip": None, "env_vars": env_vars}

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
