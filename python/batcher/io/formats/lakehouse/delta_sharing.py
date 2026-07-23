"""Delta Sharing format — read a shared table directly into Arrow.

Delta Sharing's REST protocol returns a set of *pre-signed* Parquet file URLs for
a shared table. `DeltaSharingSource` obtains those URLs via the `delta_sharing`
client and reads each file directly with pyarrow, bypassing the client's
pandas-only ``load_as_pandas`` to honor Batcher's Arrow-only contract. Each shared
file becomes its own `Split`, so a distributed read parallelizes file-by-file.

Reads are scoped to a shared table reference of the form
``<profile>#<share>.<schema>.<table>`` (the standard Delta Sharing convention).
All `delta_sharing` imports are deferred; a missing dependency raises
`BackendError` with a ``pip install 'batcher-engine[delta-sharing]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.optional import require
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split
from batcher.plan.source_stats import SourceStatistics

__all__ = ["DeltaSharingFileSplit", "DeltaSharingSource"]


def _require_delta_sharing() -> Any:
    """Import and return the `delta_sharing` module or raise `BackendError`."""
    return require(
        "delta_sharing",
        feature="Delta Sharing support",
        provides="the delta-sharing client",
        extra="delta-sharing",
    )


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


def _declared_rows(files: list[Any]) -> dict[str, int]:
    """Each shared file's server-declared `numRecords`, keyed by its pre-signed URL."""
    import json

    out: dict[str, int] = {}
    for f in files:
        raw = getattr(f, "stats", None)
        if not raw:
            continue
        try:
            stats = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            continue
        count = stats.get("numRecords")
        if isinstance(count, int):
            out[f.url] = count
    return out


def _surviving_urls(files: list[Any], predicate: dict | None) -> list[str]:
    """The pre-signed URLs of the shared files that can contain a row matching `predicate`.

    Each `AddFile` carries a Delta ``stats`` JSON string. Normalizing those into the
    add-action manifest layout lets the shared table reuse the same pruning the local
    Delta reader uses (`io.stats.file_skipping`) instead of fetching every file. Any file
    whose statistics are missing or unparseable is kept — an unknown must never prune.
    """
    if predicate is None or not files:
        return [f.url for f in files]
    manifest = _stats_manifest(files)
    if manifest is None:
        return [f.url for f in files]
    from batcher.io.stats.file_skipping import surviving_files

    keep = surviving_files(predicate, manifest)
    if keep is None:
        return [f.url for f in files]
    surviving = set(keep)
    return [f.url for f in files if f.url in surviving]


def _stats_manifest(files: list[Any]) -> pa.Table | None:
    """The shared files' Delta ``stats`` as an add-action manifest, or None if unusable.

    A file with no recorded statistics contributes a row of nulls rather than being
    dropped from the manifest, so it survives pruning (a missing stat is an unknown).
    """
    import json

    rows: list[dict[str, Any]] = []
    for f in files:
        row: dict[str, Any] = {"path": f.url, "num_records": None}
        raw = getattr(f, "stats", None)
        if raw:
            try:
                stats = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (ValueError, TypeError):
                stats = {}
            row["num_records"] = stats.get("numRecords")
            for key, prefix in (
                ("minValues", "min."),
                ("maxValues", "max."),
                ("nullCount", "null_count."),
            ):
                for column, value in (stats.get(key) or {}).items():
                    row[f"{prefix}{column}"] = value
        rows.append(row)
    # `from_pylist` takes its column set from the FIRST dict alone. Files carry different
    # stat columns — a file with no `stats` contributes only `path`/`num_records` — so if
    # the first shared file happens to lack them, every `min.`/`max.`/`null_count.` column
    # vanishes and the manifest prunes nothing, for the whole table, silently. The
    # direction is safe (under-pruning never drops a row) but the cost is total: the zone
    # maps the server went to the trouble of sending are discarded. Normalizing every row
    # to the union of keys first is what makes the manifest reflect what actually arrived.
    columns = {key: None for row in rows for key in row}
    try:
        return pa.Table.from_pylist([{**columns, **row} for row in rows])
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return None  # heterogeneous stat types across files → prune nothing


def _presigned_schema(file_url: str) -> pa.Schema:
    """One shared file's Arrow schema, read from its Parquet footer alone.

    `schema()` used to be answered by `_read_presigned(files[0].url, None).schema` — a
    full fetch and decode of a shared data file, over the network, out of object storage,
    to learn its column names. Every byte of it was then discarded.

    The cost is not incidental. `schema()` is called at *plan* time, on every read, before
    the query has decided whether it wants any of those columns — and a Delta Sharing file
    is remote by construction, so this is the connector family where a materializing
    schema lookup is most expensive. The footer states the schema exactly and pyarrow
    fetches only the footer to read it, so the answer is identical and the transfer is a
    few kilobytes instead of the file.
    """
    import pyarrow.parquet as pq

    fs = resolve_filesystem(file_url)
    with fs.open(file_url) as fh:
        return pq.ParquetFile(fh).schema_arrow


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
        return _presigned_schema(files[0].url)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        from batcher.io._concurrent import read_each_file

        # Skip the files the shared statistics rule out *before* fetching: each one the
        # predicate eliminates is an object-storage round trip that never happens.
        urls = _surviving_urls(self._files(), predicate)
        # Each shared file is a separate presigned-URL parquet fetch that releases the GIL;
        # read them concurrently so a many-file shared table isn't fetched one at a time.
        tables = read_each_file(
            None, urls, lambda _fs, url: _read_presigned(url, projection, predicate)
        )
        return [b for table in tables for b in table.to_batches()]

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        for url in _surviving_urls(self._files(), predicate):
            yield from _read_presigned(url, projection, predicate).to_batches()

    def row_count(self) -> int | None:
        """The rows the sharing server declared, or None when it declared none.

        This used to return `None` unconditionally, on the reasoning that "pre-signed URLs
        carry no guaranteed cheap count". The URLs do not — but the `AddFile` actions
        alongside them do: the protocol sends each file's Delta `stats` JSON, `numRecords`
        included, and `_declared_rows` was already parsing it to size the splits. The count
        was in hand and thrown away.
        """
        stats = self.statistics()
        return None if stats is None else stats.row_count

    def statistics(self) -> SourceStatistics | None:
        """Exact row count and per-column bounds from the server's own file statistics.

        This is the metadata a shared table gets for free and was not using. The sharing
        protocol sends the same ``numRecords``/``minValues``/``maxValues``/``nullCount``
        a local `_delta_log` carries, `_stats_manifest` already normalizes it into the
        add-action layout, and `manifest_statistics` already aggregates that layout for
        Delta and Iceberg. Declaring it here costs one dict walk over metadata that has
        *already arrived over the wire* and gives the shared table the zone map its local
        sibling has: predicates provable empty from metadata, `min()`/`max()` answered
        without a fetch, and pruning rules that can fire at all.

        **Nothing is reported unless the server stated it.** If any shared file omits
        `numRecords`, the total is not the table's count — `pc.sum` skips nulls, so it
        would silently under-report — and an under-reported *exact* count is not a slow
        plan but a wrong answer, since `count()` is served straight from it without
        executing. So a single missing count withdraws the whole statistic rather than
        rounding it off. `bounds_include_nan` is left False: Delta statistics omit NaN, so
        a float `max` here is the largest non-NaN value and cannot answer `max()`.
        """
        files = self._files()
        if not files:
            return None
        declared = _declared_rows(files)
        if len(declared) != len(files):
            return None  # a file the server did not count: the total would understate it
        manifest = _stats_manifest(files)
        if manifest is None:
            return SourceStatistics(row_count=sum(declared.values()), exact_rows=True)
        from batcher.io.stats import manifest_statistics

        stats = manifest_statistics(manifest)
        if stats is None:
            return SourceStatistics(row_count=sum(declared.values()), exact_rows=True)
        return stats

    def identity(self) -> str:
        return f"delta_sharing:{self._url}"

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 - shared files are not coalescable
        predicate: dict | None = None,
    ) -> list[Split]:
        """One Split per pre-signed Parquet file that can match `predicate`.

        The sharing protocol already sends each file's Delta statistics alongside its
        pre-signed URL — the same ``numRecords``/``minValues``/``maxValues``/``nullCount``
        a local `_delta_log` carries — and they were being thrown away. Reading them lets
        a shared table be pruned exactly like a local one, which matters *more* here than
        locally: a skipped file is an object-storage GET over the wire that never happens.
        Files whose statistics are absent or undecidable are kept (see `file_skipping`).
        """
        files = self._files()
        surviving = _surviving_urls(files, predicate)
        # The server already told us each file's row count in its `stats`. Carrying it on
        # the split is what stops `row_count()` re-fetching a footer over a pre-signed URL
        # once per split — see `DeltaSharingFileSplit.row_count`.
        rows = _declared_rows(files)
        return [DeltaSharingFileSplit(file_url=url, rows=rows.get(url)) for url in surviving]


class DeltaSharingFileSplit:
    """One pre-signed Parquet file of a shared table, read directly via pyarrow.

    Carries only the (time-limited) pre-signed URL, so it serializes to a worker
    that reads its single file directly from object storage.
    """

    __slots__ = ("_file_url", "_rows")

    def __init__(self, *, file_url: str, rows: int | None = None) -> None:
        self._file_url = file_url
        self._rows = rows

    def schema(self) -> pa.Schema:
        return _presigned_schema(self._file_url)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return _read_presigned(self._file_url, projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from _read_presigned(self._file_url, projection, predicate).to_batches()

    def row_count(self) -> int | None:
        """The count the sharing server declared, falling back to the file's own footer.

        The server sends `numRecords` in each file's `stats`, and `splits()` now carries it
        here. Without it this opened a **pre-signed URL per split** just to read a footer —
        serially, on the driver, every time the planner balanced — re-fetching a number the
        manifest had already parsed. The sibling `DeltaFileSplit` has always carried `rows`;
        this is the same fix.

        Returns:
            The row count, or None if neither the server nor the footer states one.
        """
        if self._rows is not None:
            return self._rows
        import pyarrow.parquet as pq

        fs = resolve_filesystem(self._file_url)
        with fs.open(self._file_url) as fh:
            return pq.ParquetFile(fh).metadata.num_rows

    def identity(self) -> str:
        return f"delta_sharing:{self._file_url}"
