"""`graphql`: a GraphQL query, paged by a cursor variable, read as a lazy relation.

The query is POSTed as ``{"query": ..., "variables": ...}`` (the GraphQL-over-HTTP request
shape). Records are read from `records_path` inside ``data``; paging follows the Relay
connection convention by default -- `page_info_path` names a ``pageInfo`` object whose
``endCursor`` is sent back as the variable `cursor_variable` while ``hasNextPage`` is true.

**An error is never a short table.** GraphQL reports failure in-band: a response can carry
HTTP 200, a partial ``data`` and a non-empty ``errors`` list at once, and a reader that only
looks at ``data`` returns a table that is silently missing whatever the failed fields held.
So *any* entry in ``errors`` fails the read with every message and path, even when ``data``
is present, and a response with no ``data`` at all fails too. A partial result is not
offered as an option: whether the missing part mattered is not something a reader can know.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.http.auth import AuthProvider
from batcher.io.formats.http.options import FieldPath, RetryPolicy, dig
from batcher.io.formats.http.records import page_records
from batcher.io.formats.http.source import Page, PagedJsonSource
from batcher.io.formats.http.state import Incremental, IncrementalRun
from batcher.io.formats.http.transport import HttpClient, redact_url

__all__ = ["GraphQLSource", "raise_on_errors"]


def raise_on_errors(document: Any, *, where: str) -> dict:
    """The ``data`` object of a GraphQL response, or a `BackendError` for any error.

    Args:
        document: The decoded response.
        where: The endpoint (redacted), for the message.

    Returns:
        The response's ``data`` object.

    Raises:
        BackendError: When ``errors`` is non-empty, even alongside partial ``data``, or
            when there is no ``data``.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.http.graphql import raise_on_errors
            >>> raise_on_errors({"data": {"x": 1}}, where="api")
            {'x': 1}
    """
    if not isinstance(document, dict):
        raise BackendError(f"{where}: GraphQL response is not a JSON object")
    errors = document.get("errors") or []
    data = document.get("data")
    if errors:
        messages = "; ".join(
            f"{e.get('message', e)}" + (f" (path {e['path']})" if e.get("path") else "")
            if isinstance(e, dict)
            else str(e)
            for e in errors
        )
        partial = " The response also carried partial data, which was discarded." if data else ""
        raise BackendError(
            f"{where}: GraphQL returned {len(errors)} error(s): {messages}.{partial}"
        )
    if not isinstance(data, dict):
        raise BackendError(f"{where}: GraphQL response carries no data object")
    return data


@SOURCES.register("graphql")
class GraphQLSource(PagedJsonSource):
    """A GraphQL query read as a relation, paging a cursor variable.

    Args:
        url: The GraphQL endpoint.
        query: The query document. It declares the cursor variable when it pages
            (``query($after: String) { ... }``).
        records_path: Where each page holds its records, relative to ``data``
            (``"repository.issues.nodes"``).
        variables: Variables sent with every page.
        page_info_path: The Relay ``pageInfo`` object, relative to ``data``; None
            reads one page.
        cursor_variable: The variable the next cursor is sent as.
        schema: The declared schema; inferred from the first page when None.
        headers: Headers for every request; a value may be a secret reference.
        auth: A `BearerToken` or `OAuth2ClientCredentials`.
        retry: A `RetryPolicy`; the default policy when None.
        timeout: Seconds to wait for each response.
        max_concurrency: Requests in flight from this process to the endpoint's host.
        max_pages: Stop after this many pages.
        incremental: An `Incremental`; with no `cursor_field` it resumes from the last
            accepted page cursor.
    """

    format_name = "graphql"

    def __init__(
        self,
        url: str,
        query: str,
        *,
        records_path: FieldPath,
        variables: dict[str, Any] | None = None,
        page_info_path: FieldPath | None = None,
        cursor_variable: str = "after",
        schema: pa.Schema | None = None,
        headers: dict[str, str] | None = None,
        auth: AuthProvider | None = None,
        retry: RetryPolicy | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 4,
        max_pages: int | None = None,
        incremental: Incremental | None = None,
    ) -> None:
        super().__init__(schema=schema, incremental=incremental, max_pages=max_pages)
        if not query.strip():
            raise PlanError("graphql needs a non-empty query")
        self._url = url
        self._query = query
        self._records_path = records_path
        self._variables = dict(variables or {})
        self._page_info_path = page_info_path
        self._cursor_variable = cursor_variable
        self._client_kwargs = {
            "headers": dict(headers or {}),
            "auth": auth,
            "retry": retry,
            "timeout": timeout,
            "max_concurrency": max_concurrency,
        }
        HttpClient(**self._client_kwargs)  # validate at construction

    def _pages(self, run: IncrementalRun | None) -> Iterator[Page]:
        client = HttpClient(**self._client_kwargs)
        where = redact_url(self._url)
        variables = dict(self._variables)
        if run is not None and self._incremental is not None:
            if self._incremental.param and run.lower_bound is not None:
                variables[self._incremental.param] = run.lower_bound
            if run.resume_cursor is not None:
                variables[self._cursor_variable] = run.resume_cursor
        cursor = variables.get(self._cursor_variable)
        fetched = 0
        while True:
            response = client.request(
                "POST", self._url, json_body={"query": self._query, "variables": variables}
            )
            data = raise_on_errors(response.json(), where=where)
            records = page_records(data, self._records_path, where=where)
            yield records, cursor, where
            fetched += 1
            if self._page_info_path is None or (self._max_pages and fetched >= self._max_pages):
                return
            info = dig(data, self._page_info_path) or {}
            cursor = info.get("endCursor")
            if not info.get("hasNextPage") or cursor is None:
                return
            variables = {**variables, self._cursor_variable: cursor}

    def governed_name(self) -> str:
        """The endpoint without its query or user info: the name a policy is written about.

        `identity` folds in a digest of the query, which no operator could type, so a
        policy keyed on it would never apply.
        """
        return redact_url(self._url)

    def identity(self) -> str:
        """The endpoint and a digest of the query and variables' names."""
        return (
            f"graphql:{redact_url(self._url)}:"
            f"{self._fingerprint(self._query, self._records_path, sorted(self._variables))}"
        )
