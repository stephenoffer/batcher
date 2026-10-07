"""`io.formats.http` — paginated HTTP JSON APIs as sources.

The generic `http_json` source (`source`), the GraphQL source (`graphql`), and the pieces
both are built from: typed pagination and retry options (`options`), auth providers that
hold secret *references* (`auth`), the shared request loop (`transport`), page-to-Arrow
conversion (`records`) and resumable incremental state (`state`). Importing this package
registers the sources. Everything here is standard library plus pyarrow, so it needs no
extra. The SaaS connectors in `io/formats/saas` are built on it.
"""

from __future__ import annotations

from batcher.io.formats.http.auth import BearerToken, OAuth2ClientCredentials
from batcher.io.formats.http.graphql import GraphQLSource
from batcher.io.formats.http.options import (
    CursorPagination,
    NextLinkPagination,
    OffsetPagination,
    PagePagination,
    RetryPolicy,
)
from batcher.io.formats.http.source import HttpJsonSource, PagedJsonSource
from batcher.io.formats.http.state import Incremental

__all__ = [
    "BearerToken",
    "CursorPagination",
    "GraphQLSource",
    "HttpJsonSource",
    "Incremental",
    "NextLinkPagination",
    "OAuth2ClientCredentials",
    "OffsetPagination",
    "PagePagination",
    "PagedJsonSource",
    "RetryPolicy",
]
