"""Choosing where a shared quota lives, and holding one request's share of it.

On a Ray worker (or any process with Ray initialized) a quota is the named actor every worker
shares; anywhere else it falls back to a quota shared by the threads of this process. Both
hold the same `QuotaState`, so the policy is identical and only its reach differs.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from batcher.dist.limits.quota import LocalQuota, QuotaConfig

__all__ = ["close_quota", "lease", "uses_ray"]

_LOCAL: dict[str, LocalQuota] = {}


def uses_ray() -> bool:
    """Whether quotas resolve to the shared Ray actor in this process.

    Read from `sys.modules` rather than by importing Ray: a process that never imported Ray
    cannot be a Ray worker, and importing it to find that out costs a second or more.
    """
    ray = sys.modules.get("ray")
    return bool(ray is not None and getattr(ray, "is_initialized", lambda: False)())


def _local(name: str, config: QuotaConfig) -> LocalQuota:
    quota = _LOCAL.get(name)
    if quota is None:
        quota = _LOCAL.setdefault(name, LocalQuota(config))
    return quota


def _acquire_remote(name: str, config: QuotaConfig, tokens: int) -> tuple[object, str, float]:
    import ray

    from batcher.dist.limits.actor import quota_actor

    handle = quota_actor(name, config)
    try:
        held, wait = ray.get(handle.acquire.remote(tokens, config))
    except ray.exceptions.RayActorError:
        # The actor was closed or its node died; a fresh one is created under the same name.
        handle = quota_actor(name, config, refresh=True)
        held, wait = ray.get(handle.acquire.remote(tokens, config))
    return handle, held, wait


@contextmanager
def lease(name: str, config: QuotaConfig, tokens: int = 0) -> Iterator[float]:
    """Hold one request's share of quota `name` for the duration of the block.

    Waits for a concurrency slot, reserves the request and its tokens, sleeps the reservation's
    delay, then yields. The slot is returned when the block exits, however it exits.

    Args:
        name: The quota's name; every holder naming it shares it.
        config: The quota's configuration. The latest one stated wins.
        tokens: Tokens the request is expected to spend.

    Yields:
        The seconds spent waiting for the reservation.
    """
    if uses_ray():
        handle, held, wait = _acquire_remote(name, config, tokens)
        release = handle.release.remote  # type: ignore[attr-defined]
    else:
        quota = _local(name, config)
        held, wait = quota.acquire(tokens, config)
        release = quota.release
    try:
        if wait > 0:
            time.sleep(wait)
        yield wait
    finally:
        release(held)


def close_quota(name: str) -> None:
    """Remove quota `name`: kill its Ray actor if one exists, and drop the local fallback.

    Args:
        name: The quota's name.
    """
    _LOCAL.pop(name, None)
    if not uses_ray():
        return
    import ray

    from batcher.dist.limits.actor import NAMESPACE, actor_name, forget_handle

    forget_handle(name)
    try:
        handle = ray.get_actor(actor_name(name), namespace=NAMESPACE)
    except ValueError:
        return
    ray.kill(handle)
