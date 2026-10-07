"""The HTTP request loop every API source shares: auth, retries, backoff and a concurrency cap.

Built on the standard library's `urllib`, deliberately: an API read is a sequence of small
JSON requests, and taking `requests` or `httpx` as a dependency of the neutral `io` layer
would put a client library and its TLS stack into every install for a feature most never
use. `io.secret_backends` makes the same trade for the same reason.

**What is retried.** A status in `RetryPolicy.retry_on` (429 and the 5xx gateway statuses by
default), a 403/429 that a rate-limit header marks as quota exhaustion, and a network error.
Anything else is a `BackendError` naming the status, the URL *without its query string*
(an API key is often a query parameter), and the head of the response body.

**How long it waits.** ``Retry-After`` first, when the policy honors it; then the reset time
in the policy's rate-limit header; then exponential backoff with full jitter. A wait longer
than `RetryPolicy.max_retry_after` fails the read instead of parking a worker for an hour.

**The concurrency cap is per process.** `max_concurrency` bounds the requests in flight from
this process to one host, shared by every source pointed at it, through a semaphore kept at
module level. A *cluster-wide* quota is not provided: several workers each reading an API
each get their own cap, so N workers can send N times the limit. Coordinating one quota
across workers needs a shared counter in the distributed scheduler, which this neutral layer
cannot reach; size `max_concurrency` (and the number of concurrent reads) with that in mind.

**Credentials never reach a log.** Auth headers are added with `add_unredirected_header`, so
`urllib` does not forward them across a redirect to another host (a pre-signed download URL),
and no log line carries a header or a query string.
"""

from __future__ import annotations

import email.utils
import json
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from batcher._internal.errors import BackendError
from batcher._internal.logging import get_logger
from batcher.io.formats.http.options import RetryPolicy

if TYPE_CHECKING:
    from batcher.io.formats.http.auth import AuthProvider

__all__ = ["HttpClient", "Response", "redact_url"]

_LOG = get_logger("io")

#: Semaphores bounding in-flight requests, keyed by ``(scheme://host, limit)``. Module
#: level so every source in the process aimed at one host shares the cap.
_LIMITS: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_LIMITS_LOCK = threading.Lock()

#: Indirections a test replaces to run the retry loop without sleeping.
_sleep = time.sleep
_now = time.time


def redact_url(url: str) -> str:
    """`url` without its query string or user info, for an error message or a log line.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.http.transport import redact_url
            >>> redact_url("https://u:p@api.example.com/v1/items?api_key=SECRET")
            'https://api.example.com/v1/items'
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _limiter(url: str, limit: int) -> threading.BoundedSemaphore:
    parts = urlsplit(url)
    key = (f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}", limit)
    with _LIMITS_LOCK:
        semaphore = _LIMITS.get(key)
        if semaphore is None:
            semaphore = _LIMITS[key] = threading.BoundedSemaphore(limit)
        return semaphore


@dataclass(frozen=True, slots=True)
class Response:
    """A completed HTTP exchange: final URL, status, lower-cased headers, and body bytes."""

    url: str
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        """The body decoded as JSON, or a `BackendError` naming the URL when it is not."""
        try:
            return json.loads(self.body or b"null")
        except ValueError as exc:
            raise BackendError(
                f"{redact_url(self.url)} answered HTTP {self.status} with a body that is not "
                f"JSON: {self.body[:120]!r}"
            ) from exc


def _retry_after(value: str | None) -> float | None:
    """Seconds a ``Retry-After`` header asks for: delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(when.timestamp() - _now(), 0.0)


class HttpClient:
    """Send requests with auth, retries, backoff and a per-process concurrency cap.

    Args:
        headers: Headers sent with every request. A value may be a secret reference
            (``"env:API_KEY"``), resolved per request.
        auth: An auth provider (`BearerToken`, `OAuth2ClientCredentials`), or None.
        retry: The retry policy; the default `RetryPolicy` when None.
        timeout: Seconds to wait for each response.
        max_concurrency: Requests in flight from this process to one host.
    """

    __slots__ = ("_auth", "_headers", "_limit", "_retry", "_timeout")

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        auth: AuthProvider | None = None,
        retry: RetryPolicy | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"max_concurrency must be >= 1, got {max_concurrency}")
        self._headers = dict(headers or {})
        self._auth = auth
        self._retry = retry or RetryPolicy()
        self._timeout = timeout
        self._limit = max_concurrency

    def get(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> Response:
        """``GET`` `url` with `params`; see `request`."""
        return self.request("GET", url, params=params, **kw)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        ok: tuple[int, ...] = (),
    ) -> Response:
        """Send one request, retrying per the policy, and return the successful response.

        Args:
            method: The HTTP method.
            url: The URL, which may already carry a query string.
            params: Query parameters to add; a None value is dropped.
            json_body: A value to send as a JSON body.
            data: Raw body bytes, when not sending JSON.
            headers: Extra headers for this request only.
            ok: Statuses outside 2xx to return rather than raise (``(410,)`` for an
                expired delta link the caller handles).

        Returns:
            The response with a 2xx status, or one listed in `ok`.

        Raises:
            BackendError: On a non-retryable status, or once retries are exhausted.
        """
        full_url = self._with_params(url, params)
        body = data
        extra = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode()
            extra.setdefault("Content-Type", "application/json")
        refreshed = False
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._send(method, full_url, body, extra)
            except (OSError, TimeoutError) as exc:  # URLError is an OSError
                if attempt >= self._retry.max_attempts:
                    raise BackendError(
                        f"{method} {redact_url(full_url)} failed after {attempt} attempt(s): "
                        f"{type(exc).__name__}: {exc}"
                    ) from None
                self._wait(self._backoff(attempt), method, full_url, f"{type(exc).__name__}")
                continue
            status = response.status
            if 200 <= status < 300 or status in ok:
                return response
            if status == 401 and self._auth is not None and not refreshed:
                # A token revoked or rotated before its stated expiry: refresh once.
                self._auth.invalidate()
                refreshed = True
                attempt -= 1
                continue
            wait = self._retry_wait(response, attempt)
            if wait is None:
                raise BackendError(
                    f"{method} {redact_url(full_url)} answered HTTP {status}"
                    + (f" after {attempt} attempts" if attempt > 1 else "")
                    + f": {response.body[:300].decode('utf-8', 'replace')}"
                )
            self._wait(wait, method, full_url, f"HTTP {status}")

    @staticmethod
    def _with_params(url: str, params: dict[str, Any] | None) -> str:
        query = {k: _param(v) for k, v in (params or {}).items() if v is not None}
        if not query:
            return url
        return f"{url}{'&' if urlsplit(url).query else '?'}{urlencode(query)}"

    def _send(self, method: str, url: str, body: bytes | None, extra: dict[str, str]) -> Response:
        import urllib.error
        import urllib.request

        from batcher.io.credentials import resolve_secret

        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("Accept", "application/json")
        request.add_header("User-Agent", "batcher-engine")
        for name, value in {**self._headers, **extra}.items():
            resolved = resolve_secret(value, what=f"HTTP header {name}") or ""
            if resolved == value:
                request.add_header(name, resolved)
            else:  # it held a secret: never forward it across a redirect
                request.add_unredirected_header(name, resolved)
        if self._auth is not None:
            for name, value in self._auth.headers().items():
                request.add_unredirected_header(name, value)
        with _limiter(url, self._limit):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as raw:
                    return Response(
                        raw.geturl(),
                        raw.status,
                        {k.lower(): v for k, v in raw.headers.items()},
                        raw.read(),
                    )
            except urllib.error.HTTPError as exc:
                return Response(
                    url,
                    exc.code,
                    {k.lower(): v for k, v in (exc.headers or {}).items()},
                    exc.read() or b"",
                )

    def _retry_wait(self, response: Response, attempt: int) -> float | None:
        """Seconds to wait before retrying `response`, or None when it is not retryable."""
        policy = self._retry
        quota_out = (
            policy.rate_limit_reset_header is not None
            and response.status in (403, 429)
            and response.headers.get(policy.rate_limit_remaining_header.lower()) == "0"
        )
        after = (
            _retry_after(response.headers.get("retry-after"))
            if policy.respect_retry_after
            else None
        )
        retryable = response.status in policy.retry_on or quota_out
        if response.status == 403 and after is not None:
            retryable = True  # GitHub's secondary rate limit: 403 with Retry-After
        if not retryable or attempt >= policy.max_attempts:
            return None
        if after is None and quota_out:
            reset = response.headers.get(str(policy.rate_limit_reset_header).lower(), "")
            if reset.strip().isdigit():
                after = max(float(reset) - _now(), 0.0) + 1.0
        if after is not None:
            if after > policy.max_retry_after:
                raise BackendError(
                    f"{redact_url(response.url)} asked to retry after {after:.0f}s "
                    f"(HTTP {response.status}), longer than "
                    f"RetryPolicy.max_retry_after={policy.max_retry_after:.0f}s"
                )
            return after
        return self._backoff(attempt)

    def _backoff(self, attempt: int) -> float:
        cap = min(self._retry.max_backoff, self._retry.backoff * (2 ** (attempt - 1)))
        return random.uniform(0.0, cap)  # full jitter

    @staticmethod
    def _wait(seconds: float, method: str, url: str, why: str) -> None:
        _LOG.info("%s %s: %s, retrying in %.2fs", method, redact_url(url), why, seconds)
        _sleep(seconds)


def _param(value: Any) -> str:
    """A query-parameter value: booleans the way JSON APIs spell them, the rest as text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
