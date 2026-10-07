"""Typed options for the HTTP JSON source: pagination styles and the retry policy.

An HTTP API is paged in one of a handful of ways, and each is a small frozen value object
here rather than a string plus a bag of keywords. A frozen dataclass pickles to a worker,
prints without a secret (none of these hold one), and states its contract in its own
signature, which a ``pagination="cursor", cursor_param=..., cursor_path=...`` keyword soup
cannot do.

Each pagination object answers two questions for the paging loop in `source`:

* ``_first(url, params)`` -- the first request, and
* ``_follow(request, body, headers, n_records)`` -- the next request after a page, or
  None when the page was the last one.

and one for resumption, ``_resume(url, params, cursor)``: the request that re-reads the page
a recorded cursor names. The *cursor* a pagination reports is the token that addresses a
page: the cursor value, the next-link URL, the offset, or the page number. It is what
`Incremental` persists as the last accepted position.

The layer is neutral `io`: these objects know nothing of plans or execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from batcher._internal.errors import PlanError

__all__ = [
    "CursorPagination",
    "NextLinkPagination",
    "OffsetPagination",
    "PagePagination",
    "Request",
    "RetryPolicy",
    "dig",
]

#: A field path into a JSON document: a dotted string, or a tuple of keys for a key that
#: itself contains a dot (Microsoft Graph's ``@odata.nextLink``).
FieldPath = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Request:
    """One page request: the URL, its query parameters, and the cursor that addresses it."""

    url: str
    params: dict[str, Any]
    cursor: Any = None


def dig(document: Any, path: FieldPath | None) -> Any:
    """The value at `path` inside a decoded JSON document, or None when any step is absent.

    Args:
        document: The decoded JSON value.
        path: A dotted path (``"data.items"``), a tuple of keys, or None/``""`` for the
            document itself.

    Returns:
        The value found, or None.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.http.options import dig
            >>> dig({"a": {"b": [1, 2]}}, "a.b")
            [1, 2]
            >>> dig({"@odata.nextLink": "u"}, ("@odata.nextLink",))
            'u'
    """
    if not path:
        return document
    keys = path if isinstance(path, tuple) else tuple(path.split("."))
    value = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _positive(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise PlanError(f"{name} must be >= 1, got {value}")


@dataclass(frozen=True, slots=True)
class CursorPagination:
    """Page with an opaque cursor the response body carries and the next request sends back.

    The next cursor is read from `cursor_path` in each response and sent as the query
    parameter `param`. Paging stops when the cursor is missing, null or empty, or when
    `has_more_path` is given and reads false. This is the Stripe, Slack and Notion style.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> p = bt.io.CursorPagination(cursor_path="meta.next_cursor", param="cursor")
            >>> p.param
            'cursor'

    Args:
        cursor_path: Where the next cursor sits in the response body.
        param: The query parameter the cursor is sent as.
        has_more_path: Optional boolean field that says whether another page exists.
    """

    cursor_path: FieldPath
    param: str = "cursor"
    has_more_path: FieldPath | None = None

    def _first(self, url: str, params: dict[str, Any]) -> Request:
        return Request(url, dict(params))

    def _resume(self, url: str, params: dict[str, Any], cursor: Any) -> Request:
        return Request(url, {**params, self.param: cursor}, cursor)

    def _follow(
        self, request: Request, body: Any, headers: dict[str, str], n_records: int
    ) -> Request | None:
        del headers, n_records
        if self.has_more_path is not None and not dig(body, self.has_more_path):
            return None
        cursor = dig(body, self.cursor_path)
        if cursor is None or cursor == "":
            return None
        return Request(request.url, {**request.params, self.param: cursor}, cursor)


_LINK_NEXT = re.compile(r"<([^>]*)>\s*;[^,]*?\brel\s*=\s*\"?([^\",]*)\"?", re.IGNORECASE)


def next_link_from_header(value: str | None) -> str | None:
    """The ``rel="next"`` target of an RFC 8288 ``Link`` header, or None."""
    if not value:
        return None
    for target, rels in _LINK_NEXT.findall(value):
        if "next" in rels.lower().split():
            return target
    return None


@dataclass(frozen=True, slots=True)
class NextLinkPagination:
    """Page by following a next-page URL, from the ``Link`` header or a body field.

    With no `path`, the URL is the ``rel="next"`` target of the RFC 8288 ``Link`` response
    header, which is how GitHub and many REST APIs page. With a `path`, it is read from that
    body field (``"next"``, ``"links.next"``, or ``("@odata.nextLink",)`` for Microsoft Graph).
    A relative URL is resolved against the page it came from. The next URL already carries
    its own query string, so the original query parameters are not re-sent with it.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.io.NextLinkPagination().path is None
            True
            >>> bt.io.NextLinkPagination(path="links.next").path
            'links.next'

    Args:
        path: The body field holding the next URL, or None for the ``Link`` header.
    """

    path: FieldPath | None = None

    def _first(self, url: str, params: dict[str, Any]) -> Request:
        return Request(url, dict(params))

    def _resume(self, url: str, params: dict[str, Any], cursor: Any) -> Request:
        del url, params
        return Request(str(cursor), {}, cursor)

    def _follow(
        self, request: Request, body: Any, headers: dict[str, str], n_records: int
    ) -> Request | None:
        del n_records
        if self.path is None:
            target = next_link_from_header(headers.get("link"))
        else:
            found = dig(body, self.path)
            target = str(found) if found else None
        if not target:
            return None
        absolute = urljoin(request.url, target)
        return Request(absolute, {}, absolute)


@dataclass(frozen=True, slots=True)
class OffsetPagination:
    """Page by row offset: ``?offset=0&limit=100``, then ``offset=100``, and so on.

    Paging stops on a page holding fewer than `limit` records, or once the offset reaches
    the total at `total_path` when the API reports one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.io.OffsetPagination(limit=100, offset_param="skip").offset_param
            'skip'

    Args:
        limit: Records requested per page.
        offset_param: The query parameter the offset is sent as.
        limit_param: The query parameter the page size is sent as.
        start: The first offset.
        total_path: Optional field holding the total record count.
    """

    limit: int
    offset_param: str = "offset"
    limit_param: str = "limit"
    start: int = 0
    total_path: FieldPath | None = None

    def __post_init__(self) -> None:
        _positive("OffsetPagination.limit", self.limit)

    def _at(self, url: str, params: dict[str, Any], offset: int) -> Request:
        return Request(
            url, {**params, self.offset_param: offset, self.limit_param: self.limit}, offset
        )

    def _first(self, url: str, params: dict[str, Any]) -> Request:
        return self._at(url, params, self.start)

    def _resume(self, url: str, params: dict[str, Any], cursor: Any) -> Request:
        return self._at(url, params, int(cursor))

    def _follow(
        self, request: Request, body: Any, headers: dict[str, str], n_records: int
    ) -> Request | None:
        del headers
        if n_records < self.limit:
            return None
        offset = int(request.cursor) + self.limit
        total = dig(body, self.total_path) if self.total_path is not None else None
        if total is not None and offset >= int(total):
            return None
        return self._at(request.url, request.params, offset)


@dataclass(frozen=True, slots=True)
class PagePagination:
    """Page by page number: ``?page=1``, ``?page=2``, and so on.

    Paging stops on an empty page, on a page shorter than `size` when `size` is given, or
    after the page count at `total_pages_path` when the API reports one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.io.PagePagination(size=50, size_param="per_page").start
            1

    Args:
        page_param: The query parameter the page number is sent as.
        start: The first page number.
        size: Records requested per page, or None to leave the API's default.
        size_param: The query parameter the page size is sent as.
        total_pages_path: Optional field holding the number of pages.
    """

    page_param: str = "page"
    start: int = 1
    size: int | None = None
    size_param: str = "per_page"
    total_pages_path: FieldPath | None = None

    def __post_init__(self) -> None:
        _positive("PagePagination.size", self.size)

    def _at(self, url: str, params: dict[str, Any], page: int) -> Request:
        query = {**params, self.page_param: page}
        if self.size is not None:
            query[self.size_param] = self.size
        return Request(url, query, page)

    def _first(self, url: str, params: dict[str, Any]) -> Request:
        return self._at(url, params, self.start)

    def _resume(self, url: str, params: dict[str, Any], cursor: Any) -> Request:
        return self._at(url, params, int(cursor))

    def _follow(
        self, request: Request, body: Any, headers: dict[str, str], n_records: int
    ) -> Request | None:
        del headers
        if n_records == 0 or (self.size is not None and n_records < self.size):
            return None
        page = int(request.cursor) + 1
        total = dig(body, self.total_pages_path) if self.total_pages_path is not None else None
        if total is not None and page > int(total) + self.start - 1:
            return None
        return self._at(request.url, request.params, page)


#: Every pagination style the paging loop accepts.
Pagination = CursorPagination | NextLinkPagination | OffsetPagination | PagePagination


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How an HTTP source retries: which statuses, how long to back off, and when to give up.

    A retryable response (by default 429 and the 5xx gateway statuses) or a network error is
    retried up to `max_attempts` times in all. The wait is the server's ``Retry-After``
    when it sends one (seconds or an HTTP date), else the reset time in
    `rate_limit_reset_header` when the remaining-quota header reads zero, else exponential
    backoff with full jitter from `backoff` up to `max_backoff` seconds. A server asking for
    a longer wait than `max_retry_after` fails the read rather than parking it.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.io.RetryPolicy(max_attempts=3).retry_on
            (429, 500, 502, 503, 504)

    Args:
        max_attempts: Attempts in all, the first included.
        backoff: The first backoff, in seconds.
        max_backoff: The longest single backoff, in seconds.
        retry_on: The HTTP statuses that are retried.
        respect_retry_after: Whether a ``Retry-After`` header sets the wait.
        max_retry_after: The longest server-requested wait honored, in seconds.
        rate_limit_reset_header: A header carrying the quota reset time as epoch seconds,
            such as GitHub's ``x-ratelimit-reset``. A 403 or 429 whose
            `rate_limit_remaining_header` reads ``0`` then waits until that time.
        rate_limit_remaining_header: The header carrying the remaining quota.
    """

    max_attempts: int = 5
    backoff: float = 0.5
    max_backoff: float = 60.0
    retry_on: tuple[int, ...] = (429, 500, 502, 503, 504)
    respect_retry_after: bool = True
    max_retry_after: float = 300.0
    rate_limit_reset_header: str | None = None
    rate_limit_remaining_header: str = "x-ratelimit-remaining"

    def __post_init__(self) -> None:
        _positive("RetryPolicy.max_attempts", self.max_attempts)
        if self.backoff < 0 or self.max_backoff < 0 or self.max_retry_after < 0:
            raise PlanError("RetryPolicy waits must be >= 0")
