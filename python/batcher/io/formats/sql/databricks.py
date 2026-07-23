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
from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.credentials import resolve_secret, vend_unity_credentials
from batcher.io.formats.base import SOURCES
from batcher.io.formats.lakehouse.delta import DeltaSource
from batcher.io.formats.sql._common import (
    connection_fingerprint,
    probe_is_typed,
    push_down,
    require_module,
    schema_probe,
)

if TYPE_CHECKING:
    from batcher.io.splits import Split

__all__ = ["DatabricksSource"]

_EXTRA = "databricks"
_SQL_MODULE = "databricks.sql"


#: Rows per ``fetchmany_arrow`` call on the warehouse streaming path. Chosen as a multiple of
#: the engine's 16,384-row morsel so a fetch lines up with whole morsels downstream.
_FETCH_ROWS = 65_536


def _fetch_chunks(fetch: Any) -> Iterator[Any]:
    """Drive ``fetchmany_arrow`` to exhaustion, yielding each chunk as it arrives."""
    while True:
        chunk = fetch(_FETCH_ROWS)
        if chunk is None or chunk.num_rows == 0:
            return
        yield chunk


@dataclass(frozen=True, slots=True)
class _DatabricksWarehouseSplit:
    """A picklable warehouse read: connection params + SQL (no live conn)."""

    server_hostname: str
    http_path: str
    access_token: str = field(repr=False)
    query: str

    def _connect(self) -> Any:
        sql = require_module(_SQL_MODULE, extra=_EXTRA)
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            # Resolved on the worker: the split carries the reference, not the token.
            access_token=resolve_secret(self.access_token, what="Databricks access_token"),
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
        """The result's column types, from its first chunk rather than the whole result.

        `fetchall_arrow` pulls every Cloud Fetch result file to read column names. That is
        normally hidden because `DatabricksSource.schema` asks a ``WHERE 1 = 0`` probe, but the
        fallback for an untyped probe runs the *real* query — and a warehouse result large
        enough to arrive as Cloud Fetch files is exactly the one that must not be downloaded
        to learn its column names.

        `closing` matters here: this abandons the generator after its first chunk, and without
        it the ``finally`` that closes the connection would only run at collection.
        """
        with closing(self.iter_batches()) as batches:
            for batch in batches:
                return batch.schema
        # An empty result yields no batch to read a schema off; only then pay for the fetch.
        return self._table().schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        table = self._table()
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the warehouse result, rather than materializing it and then chunking it.

        This was ``yield from self.read(...)``, so the "streaming" entry point called
        `fetchall_arrow` and pulled every Cloud Fetch result file into memory before yielding
        its first batch — defeating every caller that chose `iter_batches` to bound memory.

        A connector build without `fetchmany_arrow` falls back to the materializing fetch
        rather than failing, so this is never worse than the behavior it replaces.
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(self.query)
            fetch = getattr(cur, "fetchmany_arrow", None)
            if fetch is None:
                result = cur.fetchall_arrow()
                if isinstance(result, pa.RecordBatch):
                    result = pa.Table.from_batches([result])
                chunks: Iterator[Any] = iter(result.to_batches())
            else:
                chunks = _fetch_chunks(fetch)
            for chunk in chunks:
                table = chunk if isinstance(chunk, pa.Table) else pa.Table.from_batches([chunk])
                for batch in table.to_batches():
                    yield batch.select(projection) if projection is not None else batch
        finally:
            conn.close()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        fingerprint = connection_fingerprint(
            {"server_hostname": self.server_hostname, "http_path": self.http_path}
        )
        return f"databricks-wh:{fingerprint}:{self.query}"


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
    # `DeltaSource` delegate (pyarrow dataset pruning); on the warehouse path it and
    # the projection become the split's own ``SELECT``/``WHERE``, so the warehouse
    # filters and prunes columns before Cloud Fetch. The engine's `Filter` re-check
    # keeps a partial push correct.
    supports_predicate: ClassVar[bool] = True

    table: str | None = None
    workspace: str | None = None
    token: str | None = field(default=None, repr=False)
    query: str | None = None
    server_hostname: str | None = None
    http_path: str | None = None
    access_token: str | None = field(default=None, repr=False)

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

    def _warehouse_split(
        self, predicate: dict | None = None, projection: list[str] | None = None
    ) -> _DatabricksWarehouseSplit:
        """The warehouse split, with the pushdown already folded into its SQL (see `push_down`)."""
        return _DatabricksWarehouseSplit(
            self.server_hostname,  # type: ignore[arg-type] - guarded by _is_warehouse
            self.http_path,  # type: ignore[arg-type]
            self.access_token,  # type: ignore[arg-type]
            push_down(self.query, predicate, projection),  # type: ignore[arg-type]
        )

    def schema(self) -> pa.Schema:
        if self._is_lakehouse():
            return self._delta_source().schema()
        probed = _DatabricksWarehouseSplit(
            self.server_hostname,  # type: ignore[arg-type] - guarded by _is_warehouse
            self.http_path,  # type: ignore[arg-type]
            self.access_token,  # type: ignore[arg-type]
            schema_probe(self.query),
        ).schema()
        return probed if probe_is_typed(probed) else self._warehouse_split().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        if self._is_lakehouse():
            return self._delta_source().read(projection, predicate)
        return self._warehouse_split(predicate, projection).read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        if self._is_lakehouse():
            yield from self._delta_source().iter_batches(projection, predicate)
        else:
            yield from self._warehouse_split(predicate, projection).iter_batches(projection)

    def row_count(self) -> int | None:
        if self._is_lakehouse():
            return self._delta_source().row_count()
        return None

    def identity(self) -> str:
        """The learned-statistics key: the workspace *and* the table, never the table alone.

        ``catalog.schema.table`` is only unique *within* a workspace, so keying on it alone
        made the same fully-qualified name in a prod and a staging workspace one relation —
        and Kyber then planned one with the other's cardinalities. The warehouse path had the
        same gap: `http_path` names a warehouse but not the host it lives on. Tokens are
        excluded from the digest, so rotating one preserves the accumulated statistics.
        """
        if self._is_lakehouse():
            workspace = connection_fingerprint({"workspace": self.workspace})
            return f"databricks:{workspace}:{self.table}"
        fingerprint = connection_fingerprint(
            {"server_hostname": self.server_hostname, "http_path": self.http_path}
        )
        return f"databricks-wh:{fingerprint}:{self.query}"

    def splits(
        self,
        target_size: int | None = None,
        predicate: dict | None = None,
        projection: list[str] | None = None,
    ) -> list[Split]:
        """Splits for the table, each already carrying Kyber's pushdown.

        A Unity Catalog table *is* a Delta table, so threading the pushed predicate down
        to the Delta source is what gives a Databricks-catalog read the same file
        skipping a path-addressed Delta read gets. Without it, resolving a table by name
        silently cost every data file in the table. `DeltaSource.splits` prunes by
        predicate only — a Delta split is a data file, and its columns are pruned when the
        worker reads the footer, so `projection` is deliberately not forwarded there.

        On the warehouse path both are folded into the SQL the split carries, because a
        split is what a worker rebuilds its reader from: a filter that is not *in the
        split's own query* never reaches the warehouse. The worker issues an unfiltered,
        unprojected read, the whole table crosses the wire, and the engine's `Filter`
        discards the rows afterwards — correct, and arbitrarily expensive.
        """
        if self._is_lakehouse():
            return self._delta_source().splits(target_size, predicate)
        return [self._warehouse_split(predicate, projection)]
