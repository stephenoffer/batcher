"""Retry for the object-store failures that are worth retrying, and only those.

A read against object storage fails in two very different ways, and the IO layer used to
treat them identically — as one exception, with no retry anywhere. A 503 slow-down, a
throttle, a dropped connection or a timeout is a blip that the same GET a moment later
serves fine; a 404 or a 403 is a fact that will not change, and retrying it only spends
time before failing anyway.

Collapsing the two is worse than it first looks, because `ErrorPolicy` sits directly
downstream. Under ``on_error="skip"`` a transient timeout was recorded as a *corrupt file*
and its rows silently dropped — a healthy object turned into missing data by a blip, with
nothing but a warning to say so. Retrying before the policy is consulted is what keeps
`skip` meaning "this file is genuinely unreadable".

Classification is by exception type and message rather than by backend, deliberately: the
spine sees whatever s3fs, gcsfs, adlfs, pyarrow, or a plain `urllib` raised, and there is
no common error taxonomy across them. The default is **not** transient — an unrecognized
failure is surfaced rather than retried, so a real bug fails fast instead of taking the
retry budget with it.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

__all__ = ["is_transient", "with_retry"]

_T = TypeVar("_T")

# Substrings that mark a retryable condition in the message of whatever the backend raised.
# Matched case-insensitively against `str(exc)`. Kept to phrases that are unambiguous about
# *transience*: a "slow down" or a "timeout" says try again, where "not found" and "access
# denied" say the opposite. HTTP status codes are matched with their reason text so a bare
# number inside a key or a request id cannot trip them.
_TRANSIENT_MARKERS = (
    "slowdown",
    "slow down",
    "throttl",
    "too many requests",
    "rate limit",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "temporarily unavailable",
    "service unavailable",
    "internal error",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "request has expired",
    "operation aborted",
    "500 server error",
    "502 server error",
    "503 server error",
    "504 server error",
    "429 client error",
)

# Message substrings that veto a retry even when a transient marker also matched. A
# "404 Not Found" carrying the word "error" must never look retryable, and an expired or
# invalid credential fails identically on every attempt — retrying it just delays a clear
# auth failure behind three backoffs.
_PERMANENT_MARKERS = (
    "not found",
    "no such file",
    "no such bucket",
    "nosuchkey",
    "nosuchbucket",
    "access denied",
    "accessdenied",
    "forbidden",
    "unauthorized",
    "invalid access key",
    "signature",
    "permission denied",
)


def is_transient(exc: BaseException) -> bool:
    """Whether `exc` is worth retrying, as opposed to a fact that will not change.

    Args:
        exc: The failure raised by the storage backend.

    Returns:
        True for a throttle / timeout / dropped connection / 5xx, False for anything
        else — including anything unrecognized, so a real bug is never retried.
    """
    # A socket-level timeout is transient by construction, whatever its message says.
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    text = str(exc).lower()
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return False
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def with_retry(
    op: Callable[[], _T],
    *,
    attempts: int,
    backoff_base_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run `op`, retrying only transient failures with jittered exponential backoff.

    The backoff uses the same equal-jitter shape as the shuffle recovery loop
    (`carbonite.resilience.recovery`): half the ceiling always, plus a random half. Keeping
    at least half means a real outage is genuinely backed off, and randomizing the rest
    decorrelates a wide scan whose hundreds of concurrent file reads would otherwise be
    throttled together and then all retry on the same tick — turning one throttle into a
    self-sustaining stampede.

    Args:
        op: The read to run. Must be safe to repeat — every caller here re-reads an
            immutable object, so a retry returns the same bytes.
        attempts: Total tries, including the first. `1` disables retrying entirely.
        backoff_base_s: The first retry's backoff ceiling; doubles each round. `0`
            retries immediately, which is what a test wants and a network does not.
        sleep: Injected for tests, so they never actually wait.

    Returns:
        Whatever `op` returned on the attempt that succeeded.

    Raises:
        BaseException: The last failure, re-raised unchanged once the attempts are spent
            or as soon as one is classified non-transient.
    """
    for attempt in range(attempts):
        try:
            return op()
        except Exception as exc:
            if attempt + 1 >= attempts or not is_transient(exc):
                raise
            if backoff_base_s > 0:
                ceiling = backoff_base_s * (2**attempt)
                half = ceiling / 2.0
                sleep(half + random.uniform(0.0, half))
    raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover
