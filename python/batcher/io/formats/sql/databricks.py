"""Databricks source — direct lakehouse read, warehouse fallback.

Databricks tables are Delta tables in cloud storage fronted by Unity Catalog.
The fast path bypasses the SQL warehouse entirely: vend short-lived,
table-scoped storage credentials from Unity Catalog (`vend_unity_credentials`)
and read the managed table directly as Delta via `DeltaSource`, so the read is
Arrow-native, distributed (Delta's own splits), and never queues on a warehouse.

The fallback path runs the query through a SQL warehouse with
``databricks-sql-connector``, using ``fetchall_arrow`` (Cloud Fetch returns
Arrow result files) — for arbitrary SQL the lakehouse path can't express.

All optional imports are deferred to `BackendError` with a
``pip install 'batcher-engine[databricks]'`` hint. Tokens ride on splits as plain
values and are never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.credentials import vend_unity_credentials
from batcher.io.formats.base import SOURCES
from batcher.io.formats.lakehouse.delta import DeltaSource
from batcher.io.formats.sql._common import require_module

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["DatabricksSource"]

_EXTRA = "databricks"
_SQL_MODULE = "databricks.sql"


@dataclass(frozen=True, slots=True)
class _DatabricksWarehouseSplit:
    """A picklable warehouse read: connection params + SQL (no live conn)."""

    server_hostname: str
    http_path: str
    access_token: str
    query: str

    def _connect(self) -> Any:
        sql = require_module(_SQL_MODULE, extra=_EXTRA)
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        )

    def _table(self) -> pa.Table:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(self.query)
            result = cur.fetchall_arrow()
            if isinstance(result, pa.RecordBatch):
                result = pa.Table.from_batches([result])
            return result
        finally:
            conn.close()

    def schema(self) -> pa.Schema:
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"databricks-wh:{self.http_path}:{self.query}"


@SOURCES.register("databricks")
@dataclass(frozen=True, slots=True)
class DatabricksSource:
    """A relation read from Databricks — lakehouse-direct or warehouse fallback.

    Preferred (lakehouse-direct): pass `table` + `workspace` + `token`. Unity
    Catalog vends temporary storage credentials and the managed Delta table is
    read directly via `DeltaSource` (distributed, warehouse-free).

    Fallback (warehouse): pass `query` + `server_hostname` + `http_path` +
    `access_token`. The query runs on a SQL warehouse and results are fetched as
    Arrow via Cloud Fetch.

    Args:
        table: Fully-qualified Unity table (``catalog.schema.table``) for the
            direct lakehouse read.
        workspace: Databricks workspace URL (``https://<host>``) for vending.
        token: Workspace token for Unity credential vending. Never logged.
        query: Arbitrary SQL for the warehouse fallback.
        server_hostname: SQL warehouse hostname (warehouse fallback).
        http_path: SQL warehouse HTTP path (warehouse fallback).
        access_token: SQL warehouse access token (warehouse fallback). Never
            logged.

    Raises:
        BackendError: If neither a valid lakehouse nor warehouse configuration is
            provided, or a required dependency is missing.
    """

    # Predicate pushdown: on the lakehouse path the predicate is threaded into the
    # `DeltaSource` delegate (pyarrow dataset pruning); on the warehouse path it
    # becomes an appended SQL ``WHERE`` so the warehouse filters before Cloud
    # Fetch. The engine's `Filter` re-check keeps a partial push correct.
    supports_predicate: ClassVar[bool] = True

    table: str | None = None
    workspace: str | None = None
    token: str | None = None
    query: str | None = None
    server_hostname: str | None = None
    http_path: str | None = None
    access_token: str | None = None

    def __post_init__(self) -> None:
        if not self._is_lakehouse() and not self._is_warehouse():
            raise BackendError(
                "DatabricksSource requires either a lakehouse read "
                "(table=, workspace=, token=) or a warehouse read "
                "(query=, server_hostname=, http_path=, access_token=)"
            )

    def _is_lakehouse(self) -> bool:
        return bool(self.table and self.workspace and self.token)

    def _is_warehouse(self) -> bool:
        return bool(self.query and self.server_hostname and self.http_path and self.access_token)

    def _delta_source(self) -> DeltaSource:
        """Vend Unity credentials and build a direct Delta reader for the table."""
        storage_url, storage_options = vend_unity_credentials(
            self.table,  # type: ignore[arg-type] - guarded by _is_lakehouse
            self.workspace,  # type: ignore[arg-type]
            self.token,  # type: ignore[arg-type]
        )
        return DeltaSource(storage_url, storage_options=storage_options)

    def _warehouse_split(self, predicate: dict | None = None) -> _DatabricksWarehouseSplit:
        return _DatabricksWarehouseSplit(
            self.server_hostname,  # type: ignore[arg-type] - guarded by _is_warehouse
            self.http_path,  # type: ignore[arg-type]
            self.access_token,  # type: ignore[arg-type]
            self._warehouse_query(predicate),
        )

    def _warehouse_query(self, predicate: dict | None) -> str:
        """The warehouse query, wrapped in a pushdown ``WHERE`` when pushable."""
        query = self.query  # type: ignore[assignment] - guarded by _is_warehouse
        if predicate is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where:
                return f"SELECT * FROM ({query}) AS _bq_pred WHERE {where}"
        return query  # type: ignore[return-value]

    def schema(self) -> pa.Schema:
        if self._is_lakehouse():
            return self._delta_source().schema()
        return self._warehouse_split().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        if self._is_lakehouse():
            return self._delta_source().read(projection, predicate)
        return self._warehouse_split(predicate).read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        if self._is_lakehouse():
            yield from self._delta_source().iter_batches(projection, predicate)
        else:
            yield from self._warehouse_split(predicate).iter_batches(projection)

    def row_count(self) -> int | None:
        if self._is_lakehouse():
            return self._delta_source().row_count()
        return None

    def identity(self) -> str:
        if self._is_lakehouse():
            return f"databricks:{self.table}"
        return f"databricks-wh:{self.http_path}:{self.query}"

    def splits(self, target_size: int | None = None) -> list[Split]:
        if self._is_lakehouse():
            return self._delta_source().splits(target_size)
        return [self._warehouse_split()]
