"""Client-side rate limiting for a hosted LLM endpoint.

Retrying a 429 is recovery, not control. A fleet of workers that only retries still sends the
burst that caused the 429, then sends it again: the provider sheds the excess, the backoff
grows, and throughput settles well under the quota while every worker spends its time asleep.
Worse, some providers count rejected requests against the quota, so the retries fund their own
starvation.

A token bucket is the control. Each worker holds one, refilled continuously at the configured
rate, and a request waits for its capacity before going out rather than after being refused.
The result is a smooth send rate at the quota instead of a sawtooth under it.

**`RateLimiter` is per worker, not per fleet.** It holds no shared state, so it costs nothing
per request; divide the account quota by the number of workers you run, and leave headroom:
a provider measures arrival at its edge, where two workers' bursts can coincide.

**`ProviderLimit` is the fleet-wide one.** When dividing by a worker count is not good enough
(an autoscaling pool, several jobs on one account, a concurrency ceiling rather than a rate),
it names one quota that every worker obeys. Under Ray that quota is a named actor in
`batcher.dist.limits`, at the price of one small actor round trip per request; without Ray it
is shared by the threads of the process. Both limits can be set at once, and a request then
waits for both.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from batcher._internal.errors import PlanError

__all__ = ["ProviderLimit", "RateLimiter", "admitted", "build_limiter"]

# The character-per-token divisor the rest of the control plane estimates with. A tokenizer
# per request would be exact and would also mean loading a vocabulary into every worker to
# decide how long to wait, which costs more than the error does.
_CHARS_PER_TOKEN = 4.0


#: Tokens charged for one image on a vision request. `requests._MAX_IMAGE_EDGE` bounds every
#: image to 1024px on its longest edge before it is sent, and the providers price an image at
#: roughly ``width * height / 750`` tokens, so ~1400 is the ceiling for one. A rate limiter
#: should over-estimate rather than under: too high costs a slightly longer wait, too low costs
#: the 429 the limiter exists to prevent.
_TOKENS_PER_IMAGE = 1400


def _estimated_tokens(prompt: str, body: dict) -> int:
    """Tokens one request is expected to spend — the prompt, its images, and the reply.

    All three matter to a tokens-per-minute quota: providers count input and output together,
    a request with `max_tokens=4096` reserves far more of the quota than its prompt suggests,
    and **an image is most of a vision request's input**. Counting only the text under-charged
    a vision batch by ~1400 tokens a row, so the limiter let it run far over quota and the 429
    it exists to prevent arrived anyway.

    The estimate is deliberately coarse (see `_CHARS_PER_TOKEN`); an error here costs a
    slightly wrong wait, not a wrong result.
    """
    reply = body.get("max_tokens")
    reserved = int(reply) if isinstance(reply, int | float) else 0
    images = _image_blocks(body) * _TOKENS_PER_IMAGE
    return int(len(prompt) / _CHARS_PER_TOKEN) + reserved + images


def _image_blocks(body: dict) -> int:
    """How many image blocks a request body carries, across both engines' wire shapes.

    Anthropic spells one ``{"type": "image", ...}`` and OpenAI ``{"type": "image_url", ...}``,
    so matching on the ``image`` prefix covers both without this module having to know which
    engine built the body.
    """
    count = 0
    for message in body.get("messages") or ():
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue  # a plain-string content is text-only
        count += sum(
            1
            for block in content
            if isinstance(block, dict) and str(block.get("type", "")).startswith("image")
        )
    return count


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
            >>> limiter.acquire(estimated_tokens=10)  # seconds waited; the bucket starts full
            0.0
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


@dataclass(frozen=True)
class ProviderLimit:
    """One provider quota shared by every worker that sends remote model calls under it.

    `requests_per_minute`/`tokens_per_minute` on an engine are enforced **per worker**, so a
    quota has to be divided by a worker count nobody may know in advance. A `ProviderLimit`
    is enforced once, for everyone: pass the same one (or one with the same `name`) as an
    engine's ``shared_limit=`` and every worker draws from a single bucket and a single pool
    of concurrency slots.

    On a Ray cluster the quota lives in a named, detached actor that every worker and every
    job naming it reaches, at the price of one small actor round trip per request. Without
    Ray it is shared by the threads of this process. Each request waits for a slot, reserves
    its estimated tokens, sleeps out any rate deficit, and returns its slot when the call ends;
    a slot a dead worker never returned comes back after `lease_seconds`. Not yet verified
    against a live multi-node Ray cluster; see tests/PENDING_VERIFICATION.md.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> quota = bt.ml.ProviderLimit(
            ...     "openai-prod", requests_per_minute=600, max_concurrency=32
            ... )
            >>> engine = bt.ml.http_engine(  # doctest: +SKIP
            ...     "https://api.openai.com/v1", "gpt-4o-mini", shared_limit=quota
            ... )
            >>> quota.max_concurrency
            32

    Args:
        name: The quota's name. Every limit with this name shares one quota; the most
            recently stated settings win.
        requests_per_minute: Requests per minute across every worker, or `None`.
        tokens_per_minute: Tokens per minute across every worker, counting the prompt plus
            the reply each request reserved, or `None`.
        max_concurrency: Requests in flight at once across every worker, or `None`.
        lease_seconds: How long a slot stays held when its worker never returns it.

    Raises:
        PlanError: If `name` is empty, a limit is not positive, or nothing is limited.
    """

    name: str
    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    max_concurrency: int | None = None
    lease_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PlanError(
                f"ProviderLimit(name=...) must be a non-empty string, got {self.name!r}"
            )
        limits = {
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "max_concurrency": self.max_concurrency,
            "lease_seconds": self.lease_seconds,
        }
        for arg, value in limits.items():
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise PlanError(f"ProviderLimit({arg}=...) must be positive, got {value!r}")
        if self.max_concurrency is not None and not isinstance(self.max_concurrency, int):
            raise PlanError(
                "ProviderLimit(max_concurrency=...) must be an integer, "
                f"got {self.max_concurrency!r}"
            )
        if all(v is None for k, v in limits.items() if k != "lease_seconds"):
            raise PlanError(
                f"ProviderLimit({self.name!r}) limits nothing: set requests_per_minute, "
                "tokens_per_minute or max_concurrency"
            )

    def lease(self, estimated_tokens: int = 0) -> Any:
        """Hold one request's share of the quota for the duration of a ``with`` block.

        Engines call this around each remote request; call it yourself to put a custom
        client (a `map_batches` UDF calling a provider SDK) under the same quota.

        Args:
            estimated_tokens: Tokens the request is expected to spend.

        Returns:
            A context manager yielding the seconds spent waiting.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> quota = bt.ml.ProviderLimit("doc-example", max_concurrency=2)
                >>> with quota.lease() as waited:
                ...     waited
                0.0
                >>> quota.close()
        """
        from batcher.dist.limits import QuotaConfig, lease

        config = QuotaConfig(
            requests_per_minute=self.requests_per_minute,
            tokens_per_minute=self.tokens_per_minute,
            max_concurrency=self.max_concurrency,
            lease_seconds=self.lease_seconds,
        )
        return lease(self.name, config, estimated_tokens)

    def close(self) -> None:
        """Remove the quota: kill its Ray actor when there is one, and reset the local one.

        The actor is detached so that concurrent jobs share it, which also means it outlives
        the job; call this when the quota is no longer wanted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.ml.ProviderLimit("doc-example-close", requests_per_minute=60).close()
        """
        from batcher.dist.limits import close_quota

        close_quota(self.name)


@contextmanager
def admitted(
    limiter: RateLimiter | None, shared: ProviderLimit | None, estimated: Callable[[], int]
) -> Iterator[None]:
    """Wait for the per-worker limiter and the shared quota, then hold the shared slot.

    The one admission step every served-endpoint engine takes before a request, so the two
    limits compose the same way in each. `estimated` is called only when a limit needs it,
    keeping the unlimited path free of the token estimate.
    """
    if limiter is None and shared is None:
        yield
        return
    tokens = estimated()
    if limiter is not None:
        limiter.acquire(tokens)
    if shared is None:
        yield
        return
    with shared.lease(tokens):
        yield
