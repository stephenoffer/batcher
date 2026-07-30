"""Client-side rate limiting for a hosted LLM endpoint.

Retrying a 429 is recovery, not control. A fleet of workers that only retries still sends the
burst that caused the 429, then sends it again: the provider sheds the excess, the backoff
grows, and throughput settles well under the quota while every worker spends its time asleep.
Worse, some providers count rejected requests against the quota, so the retries fund their own
starvation.

A token bucket is the control. Each worker holds one, refilled continuously at the configured
rate, and a request waits for its capacity before going out rather than after being refused.
The result is a smooth send rate at the quota instead of a sawtooth under it.

**The limit is per worker, not per fleet.** There is no shared state here, deliberately —
coordinating a global limiter would put a synchronous round trip in front of every request.
Divide the account quota by the number of workers you run, and leave headroom: a provider
measures arrival at its edge, where two workers' bursts can coincide.
"""

from __future__ import annotations

import threading
import time

__all__ = ["RateLimiter", "build_limiter"]

# The character-per-token divisor the rest of the control plane estimates with. A tokenizer
# per request would be exact and would also mean loading a vocabulary into every worker to
# decide how long to wait, which costs more than the error does.
_CHARS_PER_TOKEN = 4.0


def _estimated_tokens(prompt: str, body: dict) -> int:
    """Tokens one request is expected to spend — the prompt, plus the reply it asked for.

    Both halves matter to a tokens-per-minute quota: providers count input and output
    together, and a request with `max_tokens=4096` reserves far more of the quota than its
    prompt suggests. The estimate is deliberately coarse (see `_CHARS_PER_TOKEN`); an error
    here costs a slightly wrong wait, not a wrong result.
    """
    reply = body.get("max_tokens")
    reserved = int(reply) if isinstance(reply, int | float) else 0
    return int(len(prompt) / _CHARS_PER_TOKEN) + reserved


class RateLimiter:
    """A thread-safe token bucket limiting requests and tokens per minute.

    One instance is shared by every thread in a worker's request pool, so the limit applies to
    the worker rather than to each in-flight request. `acquire` blocks until the bucket holds
    enough capacity, which is what makes the send rate smooth rather than bursty.

    Both dimensions are optional and independent: a provider that limits requests per minute
    and tokens per minute enforces whichever binds first, and so does this.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.engines.limits import RateLimiter
            >>> limiter = RateLimiter(requests_per_minute=600)
            >>> limiter.acquire(estimated_tokens=10)  # returns immediately, bucket is full
    """

    def __init__(
        self,
        *,
        requests_per_minute: float | None = None,
        tokens_per_minute: float | None = None,
        burst: float = 1.0,
    ) -> None:
        """Build a limiter over either or both dimensions.

        Args:
            requests_per_minute: Maximum requests per minute, or `None` for unlimited.
            tokens_per_minute: Maximum tokens per minute, or `None` for unlimited.
            burst: Bucket capacity as a multiple of one minute's allowance. ``1.0`` permits a
                full minute's worth at once after an idle period; lower it to smooth harder.

        Raises:
            ValueError: If a rate is not positive, or `burst` is not positive.
        """
        for name, value in (
            ("requests_per_minute", requests_per_minute),
            ("tokens_per_minute", tokens_per_minute),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if burst <= 0:
            raise ValueError(f"burst must be positive, got {burst}")
        self._request_rate = None if requests_per_minute is None else requests_per_minute / 60.0
        self._token_rate = None if tokens_per_minute is None else tokens_per_minute / 60.0
        self._request_capacity = (
            None if requests_per_minute is None else requests_per_minute * burst
        )
        self._token_capacity = None if tokens_per_minute is None else tokens_per_minute * burst
        self._requests = self._request_capacity or 0.0
        self._tokens = self._token_capacity or 0.0
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    @property
    def unlimited(self) -> bool:
        """Whether neither dimension is limited, so `acquire` is a no-op.

        Returns:
            True when no rate was configured.

        Examples:
            .. doctest::

                >>> from batcher.ml.llm.engines.limits import RateLimiter
                >>> RateLimiter().unlimited
                True
        """
        return self._request_rate is None and self._token_rate is None

    def acquire(self, estimated_tokens: int = 0) -> float:
        """Block until the bucket can pay for one request of `estimated_tokens`, then charge it.

        The token estimate is the *prompt's* size plus whatever the caller expects back. It does
        not have to be exact — an under-estimate spends the difference on the next request,
        because the bucket is charged before the call and refilled by wall time, not by the
        provider's accounting.

        Args:
            estimated_tokens: Tokens this request is expected to consume.

        Returns:
            The seconds spent waiting, which is zero when capacity was already available.

        Examples:
            .. doctest::

                >>> from batcher.ml.llm.engines.limits import RateLimiter
                >>> RateLimiter(requests_per_minute=6000).acquire()
                0.0
        """
        if self.unlimited:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                delay = self._shortfall_delay(estimated_tokens)
                if delay <= 0.0:
                    if self._request_rate is not None:
                        self._requests -= 1.0
                    if self._token_rate is not None:
                        self._tokens -= self._charge(estimated_tokens)
                    return waited
            # Sleep outside the lock so the other threads in the pool can be served the
            # moment the bucket refills, rather than queueing behind this one's wait.
            time.sleep(delay)
            waited += delay

    def _refill(self) -> None:
        """Add the capacity that accrued since the last call, capped at the bucket size."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        if self._request_rate is not None and self._request_capacity is not None:
            self._requests = min(
                self._request_capacity, self._requests + elapsed * self._request_rate
            )
        if self._token_rate is not None and self._token_capacity is not None:
            self._tokens = min(self._token_capacity, self._tokens + elapsed * self._token_rate)

    def _charge(self, estimated_tokens: int) -> float:
        """What a request actually costs the bucket, capped at the bucket's own size.

        Charging the raw estimate would drive the bucket arbitrarily negative on a request
        larger than a full minute's allowance, and the next caller would then wait for the
        whole overdraft to refill — minutes, for a single oversized prompt. Capping at the
        capacity spends everything the bucket can hold and no more, which is the most the
        limiter can meaningfully throttle a request it has already decided to admit.
        """
        capacity = self._token_capacity or 0.0
        return min(float(estimated_tokens), capacity)

    def _shortfall_delay(self, estimated_tokens: int) -> float:
        """Seconds until both dimensions can pay, or 0 when they already can.

        A request larger than the whole token bucket would otherwise wait forever, so it is
        admitted once the bucket is full: the limiter smooths the send rate, it does not reject
        work the caller has already decided to do.
        """
        delay = 0.0
        if self._request_rate is not None and self._requests < 1.0:
            delay = max(delay, (1.0 - self._requests) / self._request_rate)
        if (
            self._token_rate is not None
            and estimated_tokens > 0
            and self._tokens < estimated_tokens
        ):
            capacity = self._token_capacity or 0.0
            wanted = min(float(estimated_tokens), capacity)
            if self._tokens < wanted:
                delay = max(delay, (wanted - self._tokens) / self._token_rate)
        return delay


def build_limiter(
    requests_per_minute: float | None,
    tokens_per_minute: float | None,
) -> RateLimiter | None:
    """A limiter for the configured rates, or `None` when neither was set.

    Returning `None` rather than an unlimited limiter keeps the uncontended path free of a lock
    acquisition per request, which matters when the endpoint is a local server and the whole
    call is microseconds.

    Args:
        requests_per_minute: Maximum requests per minute, or `None`.
        tokens_per_minute: Maximum tokens per minute, or `None`.

    Returns:
        A `RateLimiter`, or `None` when no rate was configured.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.engines.limits import build_limiter
            >>> build_limiter(None, None) is None
            True
            >>> build_limiter(600, None).unlimited
            False
    """
    if requests_per_minute is None and tokens_per_minute is None:
        return None
    return RateLimiter(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
    )
