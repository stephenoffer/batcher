"""Delta Sharing format — read a shared table directly into Arrow.

Delta Sharing's REST protocol returns a set of *pre-signed* Parquet file URLs for
a shared table. `DeltaSharingSource` obtains those URLs via the `delta_sharing`
client and reads each file directly with pyarrow, bypassing the client's
pandas-only ``load_as_pandas`` to honor Batcher's Arrow-only contract. Each shared
file becomes its own `Split`, so a distributed read parallelizes file-by-file.

Reads are scoped to a shared table reference of the form
``<profile>#<share>.<schema>.<table>`` (the standard Delta Sharing convention).
All `delta_sharing` imports are deferred; a missing dependency raises
`BackendError` with a ``pip install 'batcher[delta-sharing]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split

__all__ = ["DeltaSharingFileSplit", "DeltaSharingSource"]


def _require_delta_sharing() -> Any:
    """Import and return the `delta_sharing` module or raise `BackendError`."""
    try:
        import delta_sharing
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Delta Sharing support requires the delta-sharing client: "
            "pip install 'batcher[delta-sharing]'"
        ) from exc
    return delta_sharing


def _parse_url(url: str) -> tuple[str, Any]:
    """Split ``<profile>#<share>.<schema>.<table>`` into (profile, Table)."""
    delta_sharing = _require_delta_sharing()
    if "#" not in url:
        raise BackendError(
            f"invalid Delta Sharing url {url!r}; expected '<profile>#<share>.<schema>.<table>'"
        )
    profile, _, table_ref = url.partition("#")
    parts = table_ref.split(".")
    if len(parts) != 3:
        raise BackendError(
            f"invalid Delta Sharing table ref {table_ref!r}; expected 'share.schema.table'"
        )
    share, schema, table = parts
    return profile, delta_sharing.Table(name=table, share=share, schema=schema)


def _list_files(url: str) -> list[Any]:
    """Return the pre-signed `AddFile` actions for a shared table."""
    delta_sharing = _require_delta_sharing()
    profile, table = _parse_url(url)
    try:
        rest_client = delta_sharing.rest_client.DataSharingRestClient(
            delta_sharing.protocol.DeltaSharingProfile.read_from_file(profile)
        )
        response = rest_client.list_files_in_table(table)
    except Exception as exc:
        raise BackendError(f"failed to list Delta Sharing files for {url!r}: {exc}") from exc
    return list(response.add_files)


def _read_presigned(
    file_url: str, projection: list[str] | None, predicate: dict | None = None
) -> pa.Table:
    """Read one pre-signed Parquet URL directly into an Arrow table.

    A pushed `predicate` becomes a pyarrow `filters` argument (row-group + page
    pruning via the footer statistics).
    """
    import pyarrow.parquet as pq

    filters = None
    if predicate is not None:
        from batcher.io.predicate import to_pyarrow_expression

        filters = to_pyarrow_expression(predicate)
    fs = resolve_filesystem(file_url)
    with fs.open(file_url) as fh:
        return pq.read_table(fh, columns=projection, filters=filters)


@SOURCES.register("delta_sharing")
class DeltaSharingSource:
    """A Delta Sharing shared table read as Arrow via pre-signed Parquet URLs.

    Args:
        url: A shared-table reference, ``<profile>#<share>.<schema>.<table>``.
    """

    __slots__ = ("_files_cache", "_url")

    def __init__(self, url: str) -> None:
        self._url = url
        self._files_cache: list[Any] | None = None

    def _files(self) -> list[Any]:
        if self._files_cache is None:
            self._files_cache = _list_files(self._url)
        return self._files_cache

    # Predicate pushdown: the shared data is Parquet, so a pushed predicate
    # becomes a pyarrow filter applied as each pre-signed file is read.
    supports_predicate = True

    def schema(self) -> pa.Schema:
        files = self._files()
        if not files:
            raise BackendError(f"Delta Sharing table {self._url!r} has no files to infer schema")
        return _read_presigned(files[0].url, None).schema

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        for f in self._files():
            out.extend(_read_presigned(f.url, projection, predicate).to_batches())
        return out

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        for f in self._files():
            yield from _read_presigned(f.url, projection, predicate).to_batches()

    def row_count(self) -> int | None:
        return None  # pre-signed URLs carry no guaranteed cheap count.

    def identity(self) -> str:
        return f"delta_sharing:{self._url}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One Split per pre-signed Parquet file in the shared table."""
        return [DeltaSharingFileSplit(file_url=f.url) for f in self._files()]


class DeltaSharingFileSplit:
    """One pre-signed Parquet file of a shared table, read directly via pyarrow.

    Carries only the (time-limited) pre-signed URL, so it serializes to a worker
    that reads its single file directly from object storage.
    """

    __slots__ = ("_file_url",)

    def __init__(self, *, file_url: str) -> None:
        self._file_url = file_url

    def schema(self) -> pa.Schema:
        return _read_presigned(self._file_url, None).schema

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return _read_presigned(self._file_url, projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from _read_presigned(self._file_url, projection, predicate).to_batches()

    def row_count(self) -> int | None:
        import pyarrow.parquet as pq

        fs = resolve_filesystem(self._file_url)
        with fs.open(self._file_url) as fh:
            return pq.ParquetFile(fh).metadata.num_rows

    def identity(self) -> str:
        return f"delta_sharing:{self._file_url}"
