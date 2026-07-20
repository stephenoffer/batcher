"""Reading an Iceberg table: manifest-level file skipping, time travel, incremental scans.

`IcebergSource` resolves a catalog (`io.catalog`), loads a table, and scans it to Arrow.
The pushed predicate goes into `plan_files`, so pyiceberg answers it against the manifests'
partition values and column bounds and only the data files that can match become splits.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.catalog import CatalogSpec, resolve_catalog
from batcher.io.formats.base import SOURCES
from batcher.io.formats.lakehouse._arrow import engine_schema, normalize_engine_types
from batcher.io.formats.lakehouse.iceberg._common import _require_pyiceberg
from batcher.io.splits import Split, WholeSourceSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["IcebergSource", "IcebergTableSplit"]

#: Distinguishes "not looked up yet" from "looked up, and there is none".
_UNSET: Any = object()


@SOURCES.register("iceberg")
class IcebergSource:
    """An Apache Iceberg table read as Arrow.

    Args:
        identifier: The table identifier (``namespace.table``).
        catalog: A catalog spec (name or property mapping; see `io.catalog`).
        snapshot_id: Optional snapshot id for time travel.
        row_filter: Optional pyiceberg row-filter expression or string predicate
            pushed into the scan.
    """

    # Predicate pushdown: Kyber's pushed predicate → a pyiceberg row filter,
    # giving partition + file pruning at the scan planner. A pyiceberg version
    # mismatch degrades to no pushdown (the engine still filters).
    supports_predicate = True

    __slots__ = (
        "_catalog",
        "_identifier",
        "_manifest_cache",
        "_row_filter",
        "_snapshot_id",
        "_table_cache",
    )

    def __init__(
        self,
        identifier: str,
        *,
        catalog: CatalogSpec | str | None = None,
        snapshot_id: int | None = None,
        row_filter: Any = None,
    ) -> None:
        self._identifier = identifier
        self._catalog = catalog
        self._snapshot_id = snapshot_id
        self._row_filter = row_filter
        self._manifest_cache: Any = _UNSET
        self._table_cache: Any = _UNSET

    def _table(self) -> Any:
        """The loaded pyiceberg table, resolved once per source.

        Loading a table is a catalog round trip plus a metadata-JSON fetch — not something
        to repeat per call. It was: `_scan()` calls this, and every caller of `_scan()`
        called it again first, so an ordinary split read loaded the same table **twice**,
        and `_source()` builds a fresh source per split, so a 1,000-split scan was 2,000
        catalog round trips before a byte of data moved.

        Memoized on the instance, which is the right scope: a source is pinned to one
        identifier and (optionally) one snapshot, so the table it names cannot change
        underneath it within its own lifetime.
        """
        if self._table_cache is not _UNSET:
            return self._table_cache
        _require_pyiceberg()
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            self._table_cache = cat.load_table(self._identifier)
        except Exception as exc:
            raise BackendError(f"failed to load Iceberg table {self._identifier!r}: {exc}") from exc
        return self._table_cache

    def _pushed_filter(self, predicate: dict | None) -> Any:
        """The pyiceberg expression for `predicate`, or None.

        A pyiceberg version whose expression API differs degrades to no pushdown
        (the engine's `Filter` re-check keeps the result correct).
        """
        if predicate is None:
            return None
        try:
            from batcher.io.predicate import to_iceberg_expression

            return to_iceberg_expression(predicate)
        except Exception:
            return None

    def _row_filter_for(self, predicate: dict | None) -> Any:
        """Combine the constructor row filter with a pushed predicate via ``And``."""
        pushed = self._pushed_filter(predicate)
        if self._row_filter is not None and pushed is not None:
            from pyiceberg.expressions import And

            return And(self._row_filter, pushed)
        if pushed is not None:
            return pushed
        return self._row_filter

    def _scan(self, projection: list[str] | None = None, predicate: dict | None = None) -> Any:
        from pyiceberg.expressions import AlwaysTrue

        row_filter = self._row_filter_for(predicate)
        kwargs: dict[str, Any] = {
            "row_filter": row_filter if row_filter is not None else AlwaysTrue(),
        }
        if projection is not None:
            kwargs["selected_fields"] = tuple(projection)
        if self._snapshot_id is not None:
            kwargs["snapshot_id"] = self._snapshot_id
        return self._table().scan(**kwargs)

    def schema(self) -> pa.Schema:
        """The table's Arrow schema, in the types the engine speaks.

        pyiceberg maps Iceberg's `StringType` to Arrow `large_string`, which the engine's
        kernels do not compare against a plain `string` literal — a filter on a string column
        raised ``Invalid comparison operation: LargeUtf8 == Utf8`` from the Rust engine. Since
        every Spark- or Flink-written Iceberg table carries that mapping, normalizing is not
        an edge case (`_arrow`).
        """
        return engine_schema(self._table().schema().as_arrow())

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return normalize_engine_types(self._scan(projection, predicate).to_arrow()).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream the table, normalizing each batch to the engine's types.

        The normalization is a `cast` against a target schema derived **once** for the scan.
        It used to wrap every batch into a one-batch `pa.Table`, call
        `normalize_engine_types` (which re-derived the same target schema from scratch), and
        unwrap it again — three allocations and a full schema walk per batch, in the scan's
        hot path, to reach a conclusion that cannot change between batches of one reader.
        """
        target: pa.Schema | None = None
        for batch in self._scan(projection, predicate).to_arrow_batch_reader():
            if target is None:
                target = engine_schema(batch.schema)
            yield batch if target.equals(batch.schema) else batch.cast(target)

    def row_count(self) -> int | None:
        """Row count from the current snapshot's summary — or None if it would be wrong.

        ``total-records`` counts every row the snapshot's data files contain. It does **not**
        subtract the rows a merge-on-read delete file removes, and it knows nothing about a
        constructor `row_filter`. Answering `count()` from it in either case returns a number
        the table does not have: a filtered read of a 10-row table reported 10 and then
        produced 4. So the summary is used only when it is the true count, and otherwise the
        engine counts the rows it actually reads.
        """
        if self._row_filter is not None:
            return None  # the summary counts rows this source will never return
        snapshot = self._snapshot()
        if snapshot is None or snapshot.summary is None:
            return None
        if int(snapshot.summary.get("total-position-deletes", 0) or 0):
            return None  # merge-on-read deletes are not subtracted from `total-records`
        total = snapshot.summary.get("total-records")
        return int(total) if total is not None else None

    def _snapshot(self) -> Any:
        """The snapshot this source reads — the pinned one, or the table's current."""
        table = self._table()
        if self._snapshot_id is not None:
            return table.snapshot_by_id(self._snapshot_id)
        return table.current_snapshot()

    def statistics(self) -> SourceStatistics | None:
        """Exact row count **and per-column bounds** from the manifest, with no scan.

        The bounds are the part that was missing, and their absence was not a small gap: with
        no column zone map, Kyber's pruning rules cannot fire on an Iceberg scan at all — no
        predicate can be proven empty from metadata, and `min()`/`max()` cannot be answered
        without reading the table. Iceberg records exactly the same per-file bounds Delta
        does; they are read here (`_manifest`) and aggregated by the same neutral code.

        Returns None whenever the numbers would not describe what this source returns — a
        `row_filter`, or merge-on-read deletes. Reporting `exact_rows=True` on a count that
        is not the answer is how `count()` came to return 10 for a table that yields 4.
        """
        try:
            rows = self.row_count()
        except Exception:
            return None
        if rows is None:
            return None  # a filtered / merge-on-read source: the manifest overstates it

        from batcher.io.stats import manifest_statistics

        manifest = self._manifest()
        if manifest is not None:
            stats = manifest_statistics(manifest)
            if stats is not None:
                return stats
        return SourceStatistics(row_count=rows, exact_rows=True)

    def _manifest(self) -> pa.Table | None:
        """The snapshot's per-file manifest, read once per source.

        `inspect.data_files()` is a manifest read (a few ms), and `statistics()` and split
        planning both want it, so it is memoized on the source rather than re-read.
        """
        if self._manifest_cache is _UNSET:
            from batcher.io.formats.lakehouse.iceberg._manifest import file_manifest

            self._manifest_cache = file_manifest(self._table(), self._snapshot_id)
        return self._manifest_cache

    def read_incremental(
        self, from_snapshot_id: int, to_snapshot_id: int | None = None
    ) -> pa.Table:
        """Read rows appended between two snapshots as an Arrow table.

        Uses pyiceberg's incremental append scan; only append-produced rows are
        returned (overwrites/deletes are not included).
        """
        _require_pyiceberg()
        table = self._table()
        try:
            scan = table.incremental_append_scan(
                from_snapshot_id_exclusive=from_snapshot_id,
                to_snapshot_id=to_snapshot_id,
            )
            return scan.to_arrow()
        except AttributeError as exc:
            raise BackendError(
                "incremental append scan is unavailable in this pyiceberg version"
            ) from exc
        except Exception as exc:
            raise BackendError(f"Iceberg incremental scan failed: {exc}") from exc

    def identity(self) -> str:
        """What makes this source *this* source, for the statistics cache.

        The identity has to name everything that changes the rows the source returns, or the
        cache hands one source another's statistics. Three things must be in it, and all
        three were real bugs: the **catalog** (``db.t`` in one warehouse is a different table
        from ``db.t`` in another), the **row filter** (a filtered read returns fewer rows
        than an unfiltered one — sharing a cache entry between them is how a filtered
        `count()` came back with the whole table's total), and the **snapshot**.

        The snapshot is the one that bites hardest. `count()`, `is_empty()`, and `min()`/
        `max()` are answered from the cached statistics without executing, so an identity
        that says only ``@latest`` lets a row count outlive the table it described: after an
        append, `count()` kept returning 3 while `collect()` returned 4 — the same table,
        two different answers. Resolving ``latest`` to the current snapshot id makes a new
        commit a new identity, so there is no stale entry left to serve. It also fixes it
        against *any* writer: invalidating on our own commits never covered a table appended
        to by Spark or another process.

        A table that cannot be opened falls back to the unresolved form rather than raising.
        An identity is metadata about a source, not a read of it.
        """
        if self._snapshot_id is not None:
            ref: Any = self._snapshot_id
        else:
            try:
                snapshot = self._table().current_snapshot()
                ref = snapshot.snapshot_id if snapshot is not None else "empty"
            except Exception:
                ref = "latest"
        catalog = _catalog_key(self._catalog)
        row_filter = f"|{self._row_filter}" if self._row_filter is not None else ""
        return f"iceberg:{catalog}:{self._identifier}@{ref}{row_filter}"

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 — protocol signature; Iceberg splits per file
        predicate: dict | None = None,
    ) -> list[Split]:
        """One Split per surviving `FileScanTask`, after manifest-level file skipping.

        `predicate` is the filter Kyber pushed to this scan, and passing it into
        `plan_files` is the whole point: pyiceberg then answers it against the manifests'
        partition values and column bounds and returns only the data files that can
        contain a matching row. Planning used to call `plan_files()` with no predicate at
        all, so every data file in the snapshot became a task and the pruning Iceberg
        exists to provide was simply never asked for.

        Each split carries the manifest's `record_count`, so the distributed planner can
        bin-pack by real file size without opening a single footer — and it carries the
        planned `FileScanTask` itself, which is what lets the worker read the file the way
        Iceberg means it to be read: columns resolved by field id (so a renamed column still
        reads) and delete files applied (so a merge-on-read table does not resurrect deleted
        rows). See `IcebergTableSplit`.

        **Merge-on-read tables now split too.** They used to fall back to a single
        whole-source scan, because the per-file reader knew nothing about delete files and
        would have returned deleted rows. That was the correct trade at the time and it cost
        those tables every bit of parallelism; reading each task through Iceberg's own
        scanner removes the need for it.
        """
        try:
            tasks = list(self._scan(predicate=predicate).plan_files())
        except Exception:
            return [WholeSourceSplit(self)]
        if not tasks:
            return [WholeSourceSplit(self)]
        return [
            IcebergTableSplit(
                identifier=self._identifier,
                catalog=self._catalog,
                snapshot_id=self._snapshot_id,
                task=task,
                rows=getattr(task.file, "record_count", None),
                row_filter=self._row_filter,
            )
            for task in tasks
        ]


class IcebergTableSplit:
    """One Iceberg `FileScanTask`, read on the worker through Iceberg's own scanner.

    The split carries the *task* pyiceberg planned, not just a file path — it pickles at
    under a kilobyte — and reads it with `ArrowScan`, the same reader pyiceberg uses itself.
    That is what makes a distributed Iceberg read *Iceberg-correct*, and the previous
    approach was not:

    **Iceberg resolves columns by field id, not by name.** The split used to read the raw
    Parquet with ``pq.read_table(columns=[...])``, which resolves by name. So on a table
    whose column was ever *renamed*, the pre-rename file still carries the old name, and a
    split read of it either failed to find the column outright or returned a differently-named
    schema that could not be concatenated with its siblings::

        Schema at index 1 was different: id, value  vs  id, v

    Schema evolution is routine in Iceberg, and a distributed read of an evolved table was
    simply broken. `ArrowScan` resolves the file against the table's current schema by field
    id, so a renamed column reads back under its new name and an added one back-fills NULL.

    **Delete files are applied.** A merge-on-read task carries positional delete files;
    reading its data file directly resurrects every deleted row. That is why `splits()` used
    to refuse to split such a table at all, costing it every bit of parallelism. `ArrowScan`
    applies them, so merge-on-read tables now read distributed like any other.

    The trade, stated plainly: `ArrowScan` reads through pyiceberg's own `FileIO`, so this
    path does **not** go through `io.filesystem` — no `native_read_target` pre-buffering, and
    no read-through byte cache. That is a real cost, and it is the right one to pay: a
    correct read that misses a caching layer beats a fast read that returns the wrong columns
    or resurrects deleted rows.
    """

    __slots__ = ("_catalog", "_identifier", "_row_filter", "_rows", "_snapshot_id", "_task")

    def __init__(
        self,
        *,
        identifier: str,
        catalog: CatalogSpec | str | None,
        snapshot_id: int | None,
        task: Any,
        rows: int | None = None,
        row_filter: Any = None,
    ) -> None:
        self._identifier = identifier
        self._catalog = catalog
        self._snapshot_id = snapshot_id
        self._task = task
        self._rows = rows
        # The source's constructor `row_filter` MUST travel with the split. A worker sees
        # only this object — it rebuilds the source from these fields — so a filter left
        # behind here is a filter the worker never applies. `plan_files` prunes at *file*
        # granularity, so the surviving files still contain non-matching rows, and the
        # filter is not part of Kyber's pushed predicate, so no `Filter` re-checks them:
        # the distributed read simply returns extra rows. Measured before this was
        # carried: single-node 10 rows, distributed 100, same query, no error.
        self._row_filter = row_filter

    def _source(self) -> IcebergSource:
        return IcebergSource(
            self._identifier,
            catalog=self._catalog,
            snapshot_id=self._snapshot_id,
            row_filter=self._row_filter,
        )

    def _arrow_scan(self, projection: list[str] | None, predicate: dict | None) -> Any:
        """The pyiceberg `ArrowScan` for this split's one task, ready to read."""
        from pyiceberg.io.pyarrow import ArrowScan

        from batcher.io.filesystem import ensure_io_threads

        ensure_io_threads()  # lift the 8-thread IO cap so a wide S3 read isn't throttled
        source = self._source()
        table = source._table()
        scan = source._scan(projection, predicate)
        return ArrowScan(table.metadata, table.io, scan.projection(), scan.row_filter, True)

    def _read_table(self, projection: list[str] | None, predicate: dict | None = None) -> pa.Table:
        try:
            arrow = self._arrow_scan(projection, predicate).to_table([self._task])
        except ValueError as exc:
            # pyiceberg raises a bare ValueError for an equality-delete table (Flink CDC).
            raise BackendError(f"cannot read Iceberg table {self._identifier!r}: {exc}") from exc
        return normalize_engine_types(arrow)

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._read_table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream this data file's batches, never materializing the file.

        `to_table` builds the whole data file — deletes applied, every column decoded — and
        the old `.to_batches()` on the result merely re-chunked what was already resident.
        `to_record_batches` is the same read incrementally, so peak memory is one batch
        rather than one file. The `ValueError` translation has to be repeated here because
        pyiceberg raises it lazily, when the generator is first advanced, not when it is
        created — wrapping only the call would let the bare error escape.
        """
        try:
            batches = self._arrow_scan(projection, predicate).to_record_batches([self._task])
            target: pa.Schema | None = None
            for batch in batches:
                # Normalize the reader's *schema* once, then cast each batch to it. The old
                # per-batch `normalize_engine_types(pa.Table.from_batches([batch]))` rebuilt
                # a Table and re-derived the target schema for every batch in the scan's hot
                # path, to compute the same answer each time.
                if target is None:
                    target = engine_schema(batch.schema)
                yield batch if target.equals(batch.schema) else batch.cast(target)
        except ValueError as exc:
            # pyiceberg raises a bare ValueError for an equality-delete table (Flink CDC).
            raise BackendError(f"cannot read Iceberg table {self._identifier!r}: {exc}") from exc

    def _data_file_path(self) -> str:
        """This split's data file, taken from the scan task it already carries.

        The manifest is the authority here, and the split holds it: `plan_files()` hands back
        a `FileScanTask` whose `.file` is the manifest's `DataFile`. Nothing else needs to be
        stored or re-derived — and nothing else *should* be, because a path re-derived from a
        filesystem is absolute while every key the table uses (a deletion vector's mask, a
        statistics-cache entry) is the table-relative path the manifest wrote.
        """
        return str(self._task.file.file_path)

    def row_count(self) -> int | None:
        # The manifest's `record_count`, captured at split time. There is no fallback to
        # opening the footer: an Iceberg data file always carries its row count in the
        # manifest, so a miss here means the manifest was malformed, and reporting "unknown"
        # (which the planner handles) beats an extra S3 round-trip per split.
        return self._rows

    def identity(self) -> str:
        return f"iceberg:{self._identifier}:{self._data_file_path()}"


def _catalog_key(spec: Any) -> str:
    """A stable key for a catalog spec, so two catalogs never share a cache entry."""
    if spec is None:
        return "default"
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return ";".join(f"{k}={v}" for k, v in sorted(spec.items()))
    return str(spec)
