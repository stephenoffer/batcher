"""`github`: a repository's issues, pull requests or releases through the GitHub REST API.

Built on `http.source.HttpJsonSource`, so it inherits the paging loop, retries, the
concurrency cap and incremental state; what this module adds is GitHub's conventions:

* **Paging** follows the RFC 8288 ``Link`` header, 100 records a page.
* **Rate limits.** A 403 or 429 whose ``x-ratelimit-remaining`` is ``0`` waits until
  ``x-ratelimit-reset`` (epoch seconds); the secondary limit's ``Retry-After`` is honored.
  A wait past `RetryPolicy.max_retry_after` fails the read instead of parking it.
* **Incremental.** ``/issues`` takes ``since`` (records updated at or after it), so an
  `Incremental` on ``updated_at`` is answered by the server. ``/pulls`` and ``/releases``
  have no such parameter: an `Incremental` there filters on the client, which is correct but
  re-reads every page.
* **Tokens** are secret references sent as an unredirected ``Authorization`` header and
  appear in no log line, repr or identity.

The schema is declared per resource, from the fields GitHub documents for each, so a
repository whose first page happens to have no milestones does not infer a null column.
Note that GitHub's ``/issues`` endpoint also lists pull requests; ``pull_request`` is
non-null on those rows.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.http.auth import BearerToken
from batcher.io.formats.http.options import NextLinkPagination, RetryPolicy
from batcher.io.formats.http.source import HttpJsonSource
from batcher.io.formats.http.state import Incremental

__all__ = ["GITHUB_SCHEMAS", "GitHubSource"]

_TS = pa.timestamp("s", tz="UTC")
_USER = pa.struct([("login", pa.string()), ("id", pa.int64())])
_REF = pa.struct([("ref", pa.string()), ("sha", pa.string())])

#: The declared schema of each resource: the documented fields a pipeline usually wants.
GITHUB_SCHEMAS: dict[str, pa.Schema] = {
    "issues": pa.schema(
        [
            ("id", pa.int64()),
            ("number", pa.int64()),
            ("title", pa.string()),
            ("state", pa.string()),
            ("user", _USER),
            ("labels", pa.list_(pa.struct([("name", pa.string())]))),
            ("comments", pa.int64()),
            ("created_at", _TS),
            ("updated_at", _TS),
            ("closed_at", _TS),
            ("body", pa.string()),
            ("html_url", pa.string()),
            ("pull_request", pa.struct([("html_url", pa.string())])),
        ]
    ),
    "pulls": pa.schema(
        [
            ("id", pa.int64()),
            ("number", pa.int64()),
            ("title", pa.string()),
            ("state", pa.string()),
            ("user", _USER),
            ("draft", pa.bool_()),
            ("created_at", _TS),
            ("updated_at", _TS),
            ("closed_at", _TS),
            ("merged_at", _TS),
            ("head", _REF),
            ("base", _REF),
            ("body", pa.string()),
            ("html_url", pa.string()),
        ]
    ),
    "releases": pa.schema(
        [
            ("id", pa.int64()),
            ("tag_name", pa.string()),
            ("name", pa.string()),
            ("draft", pa.bool_()),
            ("prerelease", pa.bool_()),
            ("author", _USER),
            ("created_at", _TS),
            ("published_at", _TS),
            ("body", pa.string()),
            ("html_url", pa.string()),
        ]
    ),
}

#: GitHub's rate-limit headers, on top of the default retry statuses.
_GITHUB_RETRY = RetryPolicy(rate_limit_reset_header="x-ratelimit-reset")


@SOURCES.register("github")
class GitHubSource(HttpJsonSource):
    """A repository's issues, pull requests or releases.

    Args:
        repo: ``"owner/name"``.
        resource: ``"issues"``, ``"pulls"`` or ``"releases"``.
        token: The token as a secret reference (``"env:GITHUB_TOKEN"``); None reads
            anonymously, at GitHub's much lower unauthenticated rate limit.
        state: ``"open"``, ``"closed"`` or ``"all"`` for issues and pulls.
        since: Issues updated at or after this ISO-8601 time.
        incremental: An `Incremental`; defaults its cursor field to ``updated_at``, its
            key to ``id`` and, for issues, its parameter to ``since``.
        base_url: The API root, for GitHub Enterprise Server.
        retry: A `RetryPolicy`; GitHub's rate-limit headers are honored by default.
        max_pages: Stop after this many pages.
    """

    format_name = "github"

    def __init__(
        self,
        repo: str,
        resource: str = "issues",
        *,
        token: str | None = None,
        state: str = "all",
        since: str | None = None,
        incremental: Incremental | None = None,
        base_url: str = "https://api.github.com",
        retry: RetryPolicy | None = None,
        max_pages: int | None = None,
    ) -> None:
        if resource not in GITHUB_SCHEMAS:
            raise PlanError(
                f"github resource must be one of {sorted(GITHUB_SCHEMAS)}, got {resource!r}"
            )
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise PlanError(f"github repo must be 'owner/name', got {repo!r}")
        params: dict[str, Any] = {"per_page": 100}
        if resource != "releases":
            params["state"] = state
        if resource == "pulls":
            params.update(sort="updated", direction="asc")
        if since is not None:
            if resource != "issues":
                raise PlanError("github since= is only accepted by the issues endpoint")
            params["since"] = since
        super().__init__(
            f"{base_url.rstrip('/')}/repos/{repo}/{resource}",
            pagination=NextLinkPagination(),
            schema=GITHUB_SCHEMAS[resource],
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            params=params,
            auth=BearerToken(token) if token else None,
            retry=retry or _GITHUB_RETRY,
            max_pages=max_pages,
            incremental=_github_incremental(incremental, resource),
        )


def _github_incremental(spec: Incremental | None, resource: str) -> Incremental | None:
    """`spec` with GitHub's defaults filled in where the caller left them unset."""
    if spec is None:
        return None
    cursor = spec.cursor_field or ("published_at" if resource == "releases" else "updated_at")
    return Incremental(
        state=spec.state,
        cursor_field=cursor,
        key=spec.key or "id",
        lookback=spec.lookback,
        start=spec.start,
        param=spec.param or ("since" if resource == "issues" else None),
        auto_commit=spec.auto_commit,
    )
