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

__all__ = ["init_ray", "strip_broken_runtime_env_hook"]


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


def init_ray(
    *,
    pip: Any = None,
    env_vars: dict[str, str] | None = None,
    unconditional_hook_strip: bool = False,
) -> None:
    """Strip the broken hook, then attach to the running cluster if not already attached.

    Args:
        pip: The runtime-env ``pip`` block. Defaults to ``None``, which *neutralizes*
            the inherited workspace pip set rather than requesting no packages — the
            cluster image already carries what these benchmarks need. Pass an explicit
            list only when a script must install something job-wide.
        env_vars: Environment variables to propagate to worker actors. A driver-process
            ``os.environ`` does not otherwise reach a remote actor.
        unconditional_hook_strip: Forwarded to `strip_broken_runtime_env_hook`.
    """
    strip_broken_runtime_env_hook(unconditional=unconditional_hook_strip)
    import ray

    if ray.is_initialized():
        return
    runtime_env: dict[str, Any] = {"pip": pip}
    if env_vars:
        runtime_env["env_vars"] = env_vars
    ray.init(
        address="auto",
        runtime_env=runtime_env,
        logging_level="ERROR",
        log_to_driver=False,
    )
