"""`http_json`: a paginated HTTP JSON API read as a lazy relation.

``bt.read.http_json(url, pagination=..., records_path=...)`` walks an API page by page,
turns each page's records into one Arrow batch (`records.PageBuilder`), and yields it. The
pagination style is a typed option (`options`), the requests go through the shared
`transport.HttpClient` (auth, retries, ``Retry-After``, a per-process concurrency cap), and
an optional `state.Incremental` makes the read resumable.

**Paging is sequential, so the read is one split.** A cursor or next-link API can only be
walked in order -- the address of page N+1 is in page N -- so `splits` returns a single
`WholeSourceSplit` and the read runs on one worker. That is the shape of the API, not a
limitation to work around: an offset API could be fanned out, but only with a total count
known up front, and a guessed fan-out silently truncates or overlaps (see
`nosql.base.offset_windows`).

**Progress is observable.** `progress()` reports the pages and records read and the cursor
of the last page whose records were all handed to the consumer -- the *accepted* cursor. A
page counts as accepted when the consumer asks for the next batch, not when it is fetched,
so the cursor never runs ahead of what the consumer actually took.

`PagedJsonSource` is the shared base: a subclass supplies `_pages()`, and the base owns
schema inference, the batch conversion, projection, incremental filtering and progress.
`graphql.GraphQLSource` and the SaaS connectors in `io/formats/saas` are built on it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.http.auth import AuthProvider
from batcher.io.formats.http.options import (
    CursorPagination,
    FieldPath,
    NextLinkPagination,
    Pagination,
    Request,
    RetryPolicy,
)
from batcher.io.formats.http.records import PageBuilder, infer_schema, page_records
from batcher.io.formats.http.state import Incremental, IncrementalRun
from batcher.io.formats.http.transport import HttpClient, redact_url

__all__ = ["HttpJsonSource", "Page", "PagedJsonSource"]

#: One fetched page: its records, the cursor that addresses it, and where it came from.
Page = tuple[list[dict], Any, str]


class PagedJsonSource:
    """Base for an API read as a sequence of JSON pages, one Arrow batch per page.

    Subclasses implement `_pages(run)`, yielding ``(records, cursor, where)`` per page, and
    `identity`. The base implements the rest of the `Source` surface.

    Args:
        schema: The declared schema, or None to infer it from the first page.
        incremental: Resume state and dedup policy, or None for a full read.
        max_pages: Stop after this many pages, or None for all of them.
    """

    format_name = ""
    #: An incremental read continues across passes (it resumes from its state) rather
    #: than replaying, so a streaming driver may re-open it.
    continues_across_passes = True

    def __init__(
        self,
        *,
        schema: pa.Schema | None = None,
        incremental: Incremental | None = None,
        max_pages: int | None = None,
    ) -> None:
        if max_pages is not None and max_pages < 1:
            raise PlanError(f"max_pages must be >= 1, got {max_pages}")
        self._declared = schema
        self._incremental = incremental
        self._max_pages = max_pages
        self._builder: PageBuilder | None = None
        self._progress: dict[str, Any] = {"pages": 0, "records": 0, "cursor": None}
        self._pending_run: IncrementalRun | None = None

    # ---- override points -------------------------------------------------------
    def _pages(self, run: IncrementalRun | None) -> Iterator[Page]:
        """Yield every page of the read, in order."""
        raise NotImplementedError

    def identity(self) -> str:
        """The learned-statistics key; never carries a credential."""
        raise NotImplementedError

    # ---- the Source surface ------------------------------------------------------
    def schema(self) -> pa.Schema:
        """The declared schema, or one inferred from the first page's records."""
        return self._page_builder().schema

    def _page_builder(self) -> PageBuilder:
        if self._builder is None:
            if self._declared is not None:
                self._builder = PageBuilder(self._declared, declared=True)
            else:
                self._builder = PageBuilder(self._sample_schema(), declared=False)
        return self._builder

    def _sample_schema(self) -> pa.Schema:
        run = IncrementalRun(self._incremental) if self._incremental is not None else None
        for records, _, _ in self._pages(run):
            if records:
                return infer_schema(records)
        raise PlanError(
            f"{self.format_name}: the first page holds no records, so there is no schema to "
            "infer; declare schema= to read an API that may return nothing"
        )

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Every page as a batch."""
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Fetch page by page, yielding one batch per non-empty page.

        The incremental state is staged only after the last page is consumed, so a read
        that stops part-way leaves the previous state in force.
        """
        builder = self._page_builder()
        run = IncrementalRun(self._incremental) if self._incremental is not None else None
        self._progress = {"pages": 0, "records": 0, "cursor": None}
        for records, cursor, where in self._pages(run):
            batch = builder.batch(records, where=where)
            if run is not None:
                batch = run.filter(batch)
            if batch.num_rows:
                yield batch.select(projection) if projection is not None else batch
            # Reaching here means the consumer took the page's batch: it is accepted.
            self._progress["pages"] += 1
            self._progress["records"] += batch.num_rows
            self._progress["cursor"] = cursor
            if run is not None:
                run.accept(cursor)
        if run is not None:
            self._pending_run = run
            run.finish()
            self._pending_run = None

    def progress(self) -> dict[str, Any]:
        """Pages and records accepted by the consumer so far, and the last accepted cursor.

        Returns:
            ``{"pages": int, "records": int, "cursor": <last accepted page's cursor>}``.
        """
        return dict(self._progress)

    def row_count(self) -> int | None:
        """Unknown: an API rarely states its size up front."""
        return None

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        """One whole-source split: a paged API can only be walked in order."""
        from batcher.io.splits import WholeSourceSplit

        return [WholeSourceSplit(self)]

    # ---- streaming checkpoint hooks ----------------------------------------------
    def snapshot_position(self) -> dict:
        """The incremental state committed so far (the read position a checkpoint records)."""
        if self._incremental is None:
            return {}
        return {"state": self._incremental.load()}

    def seek(self, position: dict) -> None:
        """Resume from a checkpointed position by restoring its committed state."""
        if self._incremental is None or not position.get("state"):
            return
        from batcher.io.formats.http.state import _write_json

        _write_json(self._incremental.state, position["state"])

    def confirm(self) -> None:
        """Commit a staged state once its epoch is published (streaming queries)."""
        if self._incremental is not None:
            self._incremental.commit()

    def _fingerprint(self, *parts: Any) -> str:
        blob = "\x1f".join(repr(p) for p in parts)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@SOURCES.register("http_json")
class HttpJsonSource(PagedJsonSource):
    """A paginated HTTP JSON API, read page by page into Arrow.

    Args:
        url: The first page's URL.
        pagination: A `CursorPagination`, `NextLinkPagination`, `OffsetPagination` or
            `PagePagination`; None reads the single page at `url`.
        records_path: Where each page holds its records (``"data"``, ``"result.items"``);
            None when the page body is itself the list.
        schema: The declared schema; inferred from the first page when None.
        headers: Headers for every request; a value may be a secret reference.
        params: Query parameters for the first request.
        auth: A `BearerToken` or `OAuth2ClientCredentials`.
        retry: A `RetryPolicy`; the default policy when None.
        timeout: Seconds to wait for each response.
        max_concurrency: Requests in flight from this process to the API's host.
        max_pages: Stop after this many pages.
        incremental: An `Incremental` making the read resumable.
        method: ``"GET"`` or ``"POST"``.
        body: A JSON body sent with every request (for a ``POST`` search API).
    """

    format_name = "http_json"

    def __init__(
        self,
        url: str,
        *,
        pagination: Pagination | None = None,
        records_path: FieldPath | None = None,
        schema: pa.Schema | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        auth: AuthProvider | None = None,
        retry: RetryPolicy | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 4,
        max_pages: int | None = None,
        incremental: Incremental | None = None,
        method: str = "GET",
        body: Any = None,
    ) -> None:
        super().__init__(schema=schema, incremental=incremental, max_pages=max_pages)
        if method.upper() not in ("GET", "POST"):
            raise PlanError(f"http_json method must be 'GET' or 'POST', got {method!r}")
        if (
            incremental is not None
            and incremental.cursor_field is None
            and pagination is not None
            and not isinstance(pagination, CursorPagination | NextLinkPagination)
            and incremental.key is None
        ):
            raise PlanError(
                "resuming an offset- or page-numbered API by page cursor re-reads a page "
                "whose contents may have shifted; give Incremental a key= to deduplicate "
                "the overlap, or a cursor_field= to resume by watermark"
            )
        self._url = url
        self._pagination = pagination
        self._records_path = records_path
        self._headers = dict(headers or {})
        self._params = dict(params or {})
        self._auth = auth
        self._retry = retry
        self._timeout = timeout
        self._max_concurrency = max_concurrency
        self._method = method.upper()
        self._body = body
        # Validates max_concurrency at construction rather than at the first page.
        self._client()

    def _client(self) -> HttpClient:
        return HttpClient(
            headers=self._headers,
            auth=self._auth,
            retry=self._retry,
            timeout=self._timeout,
            max_concurrency=self._max_concurrency,
        )

    def _first_request(self, run: IncrementalRun | None) -> Request:
        params = dict(self._params)
        if run is not None and self._incremental is not None:
            bound = run.lower_bound
            if self._incremental.param and bound is not None:
                params[self._incremental.param] = bound
            resume = run.resume_cursor
            if resume is not None and self._pagination is not None:
                return self._pagination._resume(self._url, params, resume)
        if self._pagination is None:
            return Request(self._url, params)
        return self._pagination._first(self._url, params)

    def _pages(self, run: IncrementalRun | None) -> Iterator[Page]:
        client = self._client()
        request: Request | None = self._first_request(run)
        fetched = 0
        while request is not None:
            response = client.request(
                self._method,
                request.url,
                params=request.params,
                json_body=self._body,
            )
            where = redact_url(response.url)
            document = response.json()
            records = page_records(document, self._records_path, where=where)
            yield records, request.cursor, where
            fetched += 1
            if self._pagination is None or (self._max_pages and fetched >= self._max_pages):
                return
            request = self._pagination._follow(request, document, response.headers, len(records))

    def identity(self) -> str:
        """The API endpoint and the read's shape; never a header, token or query value."""
        return (
            f"http_json:{redact_url(self._url)}:"
            f"{self._fingerprint(self._pagination, self._records_path, sorted(self._params))}"
        )
