"""Apache Iceberg format — read + append/overwrite write via `pyiceberg`.

`IcebergSource` resolves a catalog (see `io.catalog`), loads a table, and scans it
to Arrow with row-filter and projection pushdown, snapshot-id time travel, and an
incremental append-scan helper. `IcebergSink` appends or overwrites via a single
catalog transaction (the driver commits one snapshot); distributed writes stage
data files and register them with `add_files`.

All `pyiceberg` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[iceberg]'`` hint. Operations pyiceberg does not implement
robustly yet (merge-on-read / equality-delete writes) raise `BackendError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.catalog import CatalogSpec, resolve_catalog
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["IcebergSink", "IcebergSource", "IcebergTableSplit"]


def _require_pyiceberg() -> None:
    """Raise `BackendError` if pyiceberg is not importable."""
    try:
        import pyiceberg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Iceberg support requires pyiceberg: pip install 'batcher-engine[iceberg]'"
        ) from exc


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

    __slots__ = ("_catalog", "_identifier", "_row_filter", "_snapshot_id")

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

    def _table(self) -> Any:
        _require_pyiceberg()
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            return cat.load_table(self._identifier)
        except Exception as exc:
            raise BackendError(f"failed to load Iceberg table {self._identifier!r}: {exc}") from exc

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
        return self._table().schema().as_arrow()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._scan(projection, predicate).to_arrow().to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._scan(projection, predicate).to_arrow_batch_reader()

    def row_count(self) -> int | None:
        """Row count from the current snapshot's summary, if recorded (else None)."""
        table = self._table()
        snapshot = (
            table.snapshot_by_id(self._snapshot_id)
            if self._snapshot_id is not None
            else table.current_snapshot()
        )
        if snapshot is None or snapshot.summary is None:
            return None
        total = snapshot.summary.get("total-records")
        return int(total) if total is not None else None

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
        ref = self._snapshot_id if self._snapshot_id is not None else "latest"
        return f"iceberg:{self._identifier}@{ref}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One Split per `FileScanTask` when planning succeeds, else whole-table."""
        try:
            tasks = list(self._scan().plan_files())
        except Exception:
            return [WholeSourceSplit(self)]
        if not tasks:
            return [WholeSourceSplit(self)]
        return [
            IcebergTableSplit(
                identifier=self._identifier,
                catalog=self._catalog,
                snapshot_id=self._snapshot_id,
                data_file_path=task.file.file_path,
            )
            for task in tasks
        ]


class IcebergTableSplit:
    """One Iceberg data file, read directly via pyarrow on the worker.

    Carries only locators (table identifier, catalog spec, snapshot, file path),
    so it serializes cheaply; the worker reconstructs an `IcebergSource` to obtain
    storage credentials/schema and reads its single Parquet file.
    """

    __slots__ = ("_catalog", "_data_file_path", "_identifier", "_snapshot_id")

    def __init__(
        self,
        *,
        identifier: str,
        catalog: CatalogSpec | str | None,
        snapshot_id: int | None,
        data_file_path: str,
    ) -> None:
        self._identifier = identifier
        self._catalog = catalog
        self._snapshot_id = snapshot_id
        self._data_file_path = data_file_path

    def _source(self) -> IcebergSource:
        return IcebergSource(
            self._identifier,
            catalog=self._catalog,
            snapshot_id=self._snapshot_id,
        )

    def _read_table(self, projection: list[str] | None) -> pa.Table:
        import pyarrow.parquet as pq

        from batcher.io.filesystem import resolve_filesystem

        fs = resolve_filesystem(self._data_file_path)
        with fs.open(self._data_file_path) as fh:
            return pq.read_table(fh, columns=projection)

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._read_table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._read_table(projection).to_batches()

    def row_count(self) -> int | None:
        import pyarrow.parquet as pq

        from batcher.io.filesystem import resolve_filesystem

        fs = resolve_filesystem(self._data_file_path)
        with fs.open(self._data_file_path) as fh:
            return pq.ParquetFile(fh).metadata.num_rows

    def identity(self) -> str:
        return f"iceberg:{self._identifier}:{self._data_file_path}"


@SINKS.register("iceberg")
class IcebergSink:
    """Append/overwrite writer for an Iceberg table (one driver-side snapshot).

    Workers stage Arrow tables via `write_partitioned`; the driver concatenates
    the manifest and commits a single snapshot in `commit`. Merge-on-read /
    equality-delete writes are not supported (pyiceberg's support is immature) and
    raise `BackendError`.

    Args:
        identifier: The table identifier (``namespace.table``).
        catalog: A catalog spec (name or property mapping; see `io.catalog`).
        mode: ``"append"`` (default) or ``"overwrite"``.
    """

    __slots__ = ("_catalog", "_identifier", "_mode", "_pending")

    def __init__(
        self,
        identifier: str,
        *,
        catalog: CatalogSpec | str | None = None,
        mode: str = "append",
    ) -> None:
        if mode not in ("append", "overwrite"):
            raise BackendError(f"unsupported Iceberg write mode {mode!r}; use append/overwrite")
        self._identifier = identifier
        self._catalog = catalog
        self._mode = mode
        self._pending: list[pa.Table] = []

    def write(self, table: pa.Table, path: str) -> WrittenFile:  # noqa: ARG002
        self._pending.append(table)
        return WrittenFile(path=self._identifier, rows=table.num_rows, bytes=0)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,  # noqa: ARG002
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002
        file_index: int = 0,  # noqa: ARG002
    ) -> list[WrittenFile]:
        """Stage one shard's `table` for the driver-side snapshot commit.

        Iceberg partitioning is a table-level property owned by the catalog, so
        `partition_by` is ignored here; pyiceberg lays files out per the table's
        partition spec at commit time.
        """
        self._pending.append(table)
        return [WrittenFile(path=self._identifier, rows=table.num_rows, bytes=0)]

    def commit(self, manifest: WriteManifest, path: str) -> None:  # noqa: ARG002
        """Commit all staged data as one snapshot (append or overwrite)."""
        _require_pyiceberg()
        if not self._pending:
            return
        data = pa.concat_tables(self._pending)
        cat = resolve_catalog(self._catalog if self._catalog is not None else "default")
        try:
            table = cat.load_table(self._identifier)
            if self._mode == "overwrite":
                table.overwrite(data)
            else:
                table.append(data)
        except Exception as exc:
            raise BackendError(f"Iceberg commit to {self._identifier!r} failed: {exc}") from exc
        finally:
            self._pending.clear()
