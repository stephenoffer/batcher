"""The state behind one shared provider quota: a reservation token bucket plus leases.

A per-worker token bucket (`batcher.ml.llm.engines.limits.RateLimiter`) blocks the calling
thread until capacity accrues, which is right inside one process and wrong for a quota many
processes share: whoever holds the bucket would sleep on behalf of everyone. This state never
sleeps. A request *reserves* capacity, the balance may go negative, and the answer is how long
the caller should wait before sending — the deficit divided by the refill rate. Reservations
queue in arrival order by construction, because each one deepens the deficit the next one
waits out.

Concurrency is counted with **leases** rather than a bare counter. A worker that dies between
acquiring and releasing would otherwise hold its slot forever and slowly strangle the fleet;
a lease expires `lease_seconds` after its send time, so a lost slot comes back.

`QuotaState` is plain Python with an injectable clock, so the whole policy is unit-tested
without Ray. `LocalQuota` wraps it for one process; the Ray actor in `limits.actor` wraps the
same object for a fleet.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["LocalQuota", "QuotaConfig", "QuotaState"]

#: Seconds a caller waits before asking again when every concurrency slot is held.
_SLOT_POLL = 0.05


@dataclass(frozen=True)
class QuotaConfig:
    """One provider quota: rates per minute, a concurrency ceiling, and the lease lifetime.

    Attributes:
        requests_per_minute: Requests per minute across every holder, or `None`.
        tokens_per_minute: Tokens per minute across every holder, or `None`.
        max_concurrency: Requests in flight at once across every holder, or `None`.
        lease_seconds: How long a concurrency slot stays held after its send time when it is
            never released, which is what reclaims a slot from a worker that died.
        burst: Bucket capacity as a multiple of one minute's allowance.
    """

    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    max_concurrency: int | None = None
    lease_seconds: float = 600.0
    burst: float = 1.0


class QuotaState:
    """A reservation token bucket over requests and tokens, with leased concurrency slots.

    Not thread-safe on its own; `LocalQuota` and the Ray actor each serialize access.
    """

    def __init__(self, config: QuotaConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._leases: dict[str, float] = {}
        self._updated = clock()
        self._config = config
        self._requests = self._capacity(config.requests_per_minute)
        self._tokens = self._capacity(config.tokens_per_minute)

    def _capacity(self, per_minute: float | None) -> float:
        return 0.0 if per_minute is None else per_minute * self._config.burst

    @property
    def config(self) -> QuotaConfig:
        """The configuration in force."""
        return self._config

    @property
    def in_flight(self) -> int:
        """Leases currently held (expired ones are reclaimed first)."""
        self._expire(self._clock())
        return len(self._leases)

    def configure(self, config: QuotaConfig) -> None:
        """Adopt `config`, keeping the balance already accrued (capped at the new capacity).

        The last holder to connect states the quota, so a change to the job's settings takes
        effect without restarting whatever holds the state.
        """
        if config == self._config:
            return
        self._refill(self._clock())
        self._config = config
        self._requests = min(self._requests, self._capacity(config.requests_per_minute))
        self._tokens = min(self._tokens, self._capacity(config.tokens_per_minute))

    def try_acquire(self, tokens: int = 0) -> tuple[str | None, float]:
        """Reserve one request of `tokens`: ``(lease, seconds to wait before sending)``.

        Returns ``(None, poll)`` when every concurrency slot is held; nothing is charged, and
        the caller asks again after `poll` seconds.
        """
        now = self._clock()
        self._expire(now)
        config = self._config
        if config.max_concurrency is not None and len(self._leases) >= config.max_concurrency:
            return None, _SLOT_POLL
        self._refill(now)
        delay = 0.0
        if config.requests_per_minute is not None:
            self._requests -= 1.0
            delay = max(delay, -self._requests / (config.requests_per_minute / 60.0))
        if config.tokens_per_minute is not None and tokens > 0:
            # Capped at one bucket, as the per-worker limiter caps it: a single prompt larger
            # than a minute's allowance waits for a full bucket rather than for minutes.
            self._tokens -= min(float(tokens), self._capacity(config.tokens_per_minute))
            delay = max(delay, -self._tokens / (config.tokens_per_minute / 60.0))
        lease = uuid.uuid4().hex
        self._leases[lease] = now + delay + config.lease_seconds
        return lease, delay

    def release(self, lease: str) -> None:
        """Return a concurrency slot. Releasing an unknown or expired lease is a no-op."""
        self._leases.pop(lease, None)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        config = self._config
        if config.requests_per_minute is not None:
            self._requests = min(
                self._capacity(config.requests_per_minute),
                self._requests + elapsed * config.requests_per_minute / 60.0,
            )
        if config.tokens_per_minute is not None:
            self._tokens = min(
                self._capacity(config.tokens_per_minute),
                self._tokens + elapsed * config.tokens_per_minute / 60.0,
            )

    def _expire(self, now: float) -> None:
        for lease in [k for k, deadline in self._leases.items() if deadline <= now]:
            del self._leases[lease]


class LocalQuota:
    """A `QuotaState` shared by the threads of one process — the fallback without Ray."""

    def __init__(self, config: QuotaConfig) -> None:
        self._state = QuotaState(config)
        self._cond = threading.Condition()

    @property
    def state(self) -> QuotaState:
        """The underlying state, for inspection."""
        return self._state

    def acquire(self, tokens: int, config: QuotaConfig) -> tuple[str, float]:
        """Block until a concurrency slot is free, then reserve: ``(lease, send delay)``."""
        with self._cond:
            self._state.configure(config)
            while True:
                lease, wait = self._state.try_acquire(tokens)
                if lease is not None:
                    return lease, wait
                self._cond.wait(timeout=wait)

    def release(self, lease: str) -> None:
        """Return the slot and wake one waiter."""
        with self._cond:
            self._state.release(lease)
            self._cond.notify()
