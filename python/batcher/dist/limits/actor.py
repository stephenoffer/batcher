"""The cluster half of a shared provider quota: one named Ray actor per provider.

Every worker of every job that names the same quota reaches the same actor, by name, in the
``batcher`` namespace. The actor holds the `QuotaState`; a worker asks it for a lease before
each remote model call, sleeps for the delay it is told, sends, and returns the lease. That is
one small actor round trip per request, against a remote model call that costs hundreds of
milliseconds or more.

**Async, so a held slot never blocks the actor.** When every concurrency slot is taken, the
call awaits inside the actor's event loop rather than blocking it, so releases and other
acquires keep being served. Rate waits are *not* spent in the actor at all: the reservation
returns its delay and the worker sleeps locally.

**Detached, by design, and removed explicitly.** A quota is a property of the provider
account rather than of one job, so it outlives the job that created it and is shared by
concurrent jobs naming it. `ProviderLimit.close()` (or `close_quota`) kills it. It reserves no
CPU, so a forgotten one holds no schedulable capacity.
"""

from __future__ import annotations

from typing import Any

from batcher.dist.limits.quota import QuotaConfig

__all__ = ["NAMESPACE", "actor_name", "forget_handle", "quota_actor"]

#: The Ray namespace every quota actor lives in, so detached actors are found across jobs.
NAMESPACE = "batcher"

_ACTOR_CLASS: Any = None
_HANDLES: dict[str, Any] = {}


def actor_name(name: str) -> str:
    """The Ray actor name a quota called `name` is registered under."""
    return f"batcher-provider-quota:{name}"


def _actor_class() -> Any:
    """The Ray actor class, defined on first use so importing this module needs no Ray."""
    global _ACTOR_CLASS
    if _ACTOR_CLASS is not None:
        return _ACTOR_CLASS
    import asyncio

    import ray

    from batcher.dist.limits.quota import QuotaState

    @ray.remote(num_cpus=0, max_concurrency=10_000)
    class ProviderQuotaActor:
        """Holds one provider's `QuotaState` for every worker that names it."""

        def __init__(self, config: QuotaConfig) -> None:
            self._state = QuotaState(config)

        async def acquire(self, tokens: int, config: QuotaConfig) -> tuple[str, float]:
            self._state.configure(config)
            while True:
                lease, wait = self._state.try_acquire(tokens)
                if lease is not None:
                    return lease, wait
                await asyncio.sleep(wait)

        async def release(self, lease: str) -> None:
            self._state.release(lease)

        async def in_flight(self) -> int:
            return self._state.in_flight

    _ACTOR_CLASS = ProviderQuotaActor
    return _ACTOR_CLASS


def quota_actor(name: str, config: QuotaConfig, *, refresh: bool = False) -> Any:
    """The handle of the actor holding quota `name`, creating it on first use.

    `get_if_exists` makes creation race-free: when two workers start at once, one creates the
    actor and the other receives the same handle.

    Args:
        name: The quota's name.
        config: The configuration to create it with when it does not exist yet.
        refresh: Drop the cached handle first, after the actor it named has died.

    Returns:
        A Ray actor handle.
    """
    if refresh:
        forget_handle(name)
    handle = _HANDLES.get(name)
    if handle is None:
        handle = (
            _actor_class()
            .options(
                name=actor_name(name),
                namespace=NAMESPACE,
                lifetime="detached",
                get_if_exists=True,
            )
            .remote(config)
        )
        _HANDLES[name] = handle
    return handle


def forget_handle(name: str) -> None:
    """Drop the cached handle for quota `name`, so the next use looks the actor up again."""
    _HANDLES.pop(name, None)
