"""`sharepoint`: a SharePoint or OneDrive document library through Microsoft Graph delta.

The read is a Microsoft Graph ``driveItem: delta`` walk over the shared
`http.transport.HttpClient`: ``GET /drives/{id}/root/delta`` (or ``/sites/{id}/drive/...``)
returns pages linked by ``@odata.nextLink``, and the final page carries an
``@odata.deltaLink`` -- the cursor that, requested later, returns only what changed since.

**Reconciling renames, changes and deletes.** Delta reports an item by its stable ``id``
with its *current* name and parent path, so a rename arrives as the same ``id`` under a new
``name``; a change arrives with a new ``etag``/``ctag``; a delete arrives with the
``deleted`` facet, which becomes ``deleted = true`` here. A merge on ``id`` therefore
reconciles a downstream copy without listing the library again. With ``state=``, the
delta link is kept (staged after the whole walk is consumed, then committed, exactly like
`http.state.Incremental`), so the next read starts from it. A delta link the service has
expired answers 410, which fails the read with instructions rather than silently
re-listing everything.

**Content streams.** ``include_content=True`` adds a ``content`` binary column, fetched
per file from ``/drives/{id}/items/{item}/content``. Graph answers that with a redirect to
a pre-authenticated download URL; the bearer token is an unredirected header, so it is
not sent to the download host. Fetching bytes is one request per file -- keep it off for a
metadata sync.

The rows are flattened from Graph's nested facets with Arrow struct access, not a Python
loop: ``parentReference.path`` becomes ``parent_path`` and so on.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.http.auth import AuthProvider
from batcher.io.formats.http.options import RetryPolicy
from batcher.io.formats.http.records import PageBuilder, page_records
from batcher.io.formats.http.state import Incremental, IncrementalRun
from batcher.io.formats.http.transport import HttpClient, redact_url

__all__ = ["DRIVE_ITEM_SCHEMA", "SharePointSource"]

_GRAPH = "https://graph.microsoft.com/v1.0"
_NEXT = "@odata.nextLink"
_DELTA = "@odata.deltaLink"

#: What Graph sends, as far as this source reads it.
_WIRE = pa.schema(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("size", pa.int64()),
        ("eTag", pa.string()),
        ("cTag", pa.string()),
        ("webUrl", pa.string()),
        ("lastModifiedDateTime", pa.timestamp("ms", tz="UTC")),
        (
            "parentReference",
            pa.struct([("id", pa.string()), ("path", pa.string()), ("driveId", pa.string())]),
        ),
        ("file", pa.struct([("mimeType", pa.string())])),
        ("folder", pa.struct([("childCount", pa.int64())])),
        ("deleted", pa.struct([("state", pa.string())])),
    ]
)

#: The rows this source yields (``content`` is appended under ``include_content``).
DRIVE_ITEM_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("parent_path", pa.string()),
        ("parent_id", pa.string()),
        ("size", pa.int64()),
        ("last_modified", pa.timestamp("ms", tz="UTC")),
        ("etag", pa.string()),
        ("ctag", pa.string()),
        ("mime_type", pa.string()),
        ("web_url", pa.string()),
        ("is_folder", pa.bool_()),
        ("deleted", pa.bool_()),
    ]
)


def _flatten(wire: pa.RecordBatch) -> list[pa.Array]:
    parent = wire.column("parentReference")
    return [
        wire.column("id"),
        wire.column("name"),
        pc.struct_field(parent, "path"),
        pc.struct_field(parent, "id"),
        wire.column("size"),
        wire.column("lastModifiedDateTime"),
        wire.column("eTag"),
        wire.column("cTag"),
        pc.struct_field(wire.column("file"), "mimeType"),
        wire.column("webUrl"),
        pc.is_valid(wire.column("folder")),
        pc.is_valid(wire.column("deleted")),
    ]


@SOURCES.register("sharepoint")
class SharePointSource:
    """A SharePoint or OneDrive drive, listed with Microsoft Graph delta.

    Args:
        drive_id: The drive (document library) id.
        site_id: A SharePoint site id, to read the site's default library instead.
        auth: An `OAuth2ClientCredentials` for the tenant, or a `BearerToken`.
        state: Where the delta link is kept between reads; None lists everything.
        folder: Keep only items under this drive path (``"/Shared Documents/reports"``).
        include_content: Add a ``content`` column with each file's bytes.
        base_url: The Graph root (for a national cloud or a test double).
        retry: A `RetryPolicy`.
        max_pages: Stop after this many pages.
    """

    format_name = "sharepoint"
    continues_across_passes = True

    def __init__(
        self,
        *,
        drive_id: str | None = None,
        site_id: str | None = None,
        auth: AuthProvider | None = None,
        state: str | None = None,
        folder: str | None = None,
        include_content: bool = False,
        base_url: str = _GRAPH,
        retry: RetryPolicy | None = None,
        max_pages: int | None = None,
    ) -> None:
        if (drive_id is None) == (site_id is None):
            raise PlanError("sharepoint needs exactly one of drive_id= or site_id=")
        self._drive = (
            f"drives/{quote(drive_id, safe='')}"
            if drive_id is not None
            else f"sites/{quote(str(site_id), safe=',')}/drive"
        )
        self._auth = auth
        self._state = Incremental(state=state) if state else None
        self._folder = folder.rstrip("/") if folder else None
        self._content = include_content
        self._base = base_url.rstrip("/")
        self._retry = retry
        self._max_pages = max_pages
        self._progress: dict[str, Any] = {"pages": 0, "records": 0, "cursor": None}

    def schema(self) -> pa.Schema:
        """`DRIVE_ITEM_SCHEMA`, plus ``content`` under ``include_content``."""
        if self._content:
            return DRIVE_ITEM_SCHEMA.append(pa.field("content", pa.binary()))
        return DRIVE_ITEM_SCHEMA

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Every delta page as a batch."""
        return list(self.iter_batches(projection))

    def _client(self) -> HttpClient:
        return HttpClient(auth=self._auth, retry=self._retry)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Walk the delta pages; stage the new delta link once all were consumed."""
        client = self._client()
        run = IncrementalRun(self._state) if self._state is not None else None
        url = (run.resume_cursor if run is not None else None) or (
            f"{self._base}/{self._drive}/root/delta"
        )
        builder = PageBuilder(_WIRE, declared=True)
        self._progress = {"pages": 0, "records": 0, "cursor": None}
        fetched = 0
        while True:
            response = client.get(url, ok=(410,))
            where = redact_url(response.url)
            if response.status == 410:
                raise BackendError(
                    f"{where}: Microsoft Graph expired the stored delta link (HTTP 410). "
                    "Delete the state document to re-list the drive in full."
                )
            document = response.json()
            batch = self._rows(
                builder.batch(page_records(document, "value", where=where), where=where), client
            )
            if batch.num_rows:
                yield batch.select(projection) if projection is not None else batch
            fetched += 1
            self._progress["pages"] = fetched
            self._progress["records"] += batch.num_rows
            following = document.get(_NEXT)
            if following and not (self._max_pages and fetched >= self._max_pages):
                url = following
                continue
            delta = document.get(_DELTA)
            if run is not None and delta:
                run.accept(delta)
                self._progress["cursor"] = delta
                run.finish()
            return

    def _rows(self, wire: pa.RecordBatch, client: HttpClient) -> pa.RecordBatch:
        batch = pa.RecordBatch.from_arrays(_flatten(wire), schema=DRIVE_ITEM_SCHEMA)
        if self._folder is not None:
            # Graph spells a parent path ``/drive/root:/Shared Documents/reports``. A deleted
            # item often comes without a path, and it cannot be placed, so it is kept: a
            # dropped delete is a downstream copy that never learns the file is gone.
            parent = pc.fill_null(batch.column("parent_path"), "")
            tail = pc.replace_substring_regex(parent, pattern=r"^[^:]*root:", replacement="")
            under = pc.or_(
                pc.or_(pc.equal(tail, self._folder), pc.starts_with(tail, f"{self._folder}/")),
                batch.column("deleted"),
            )
            batch = batch.filter(pc.fill_null(under, False))
        if not self._content:
            return batch
        wanted = pc.and_(pc.invert(batch.column("is_folder")), pc.invert(batch.column("deleted")))
        content = [
            client.get(f"{self._base}/{self._drive}/items/{quote(item, safe='')}/content").body
            if keep
            else None
            for item, keep in zip(batch.column("id").to_pylist(), wanted.to_pylist(), strict=True)
        ]
        return batch.append_column("content", pa.array(content, pa.binary()))

    def progress(self) -> dict[str, Any]:
        """Pages and items read, and the delta link accepted at the end of the walk.

        Returns:
            ``{"pages": int, "records": int, "cursor": <delta link or None>}``.
        """
        return dict(self._progress)

    def row_count(self) -> int | None:
        """Unknown until the walk is done."""
        return None

    def identity(self) -> str:
        """The drive; never a token or a delta link."""
        return f"sharepoint:{self._drive}"

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        """One split: delta pages can only be walked in order."""
        from batcher.io.splits import WholeSourceSplit

        return [WholeSourceSplit(self)]

    def confirm(self) -> None:
        """Commit a staged delta link once its epoch is published."""
        if self._state is not None:
            self._state.commit()
