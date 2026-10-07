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
values and are never logged. A lakehouse split renews its vended credentials on the
worker when they near expiry (`UnityDeltaFileSplit`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.logging import get_logger, note_suppressed
from batcher.io.credentials import UnityLease, resolve_secret, unity_lease
from batcher.io.formats.base import SOURCES
from batcher.io.formats.lakehouse.delta import DeltaSource
from batcher.io.formats.lakehouse.delta.source import DeltaFileSplit
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
_LOGGER = get_logger("io.sql")


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


def _warehouse_identity(host: Any, http_path: Any, options: dict[str, Any]) -> dict[str, Any]:
    """What names a warehouse relation: the warehouse, plus a default catalog/schema if set.

    An unqualified table name resolves against the session's catalog and schema, so the
    same query under two of them is two relations. Added only when set, so a read that
    names neither keeps the key it always had.
    """
    material: dict[str, Any] = {"server_hostname": host, "http_path": http_path}
    material.update({k: options[k] for k in ("catalog", "schema") if k in options})
    return material


def _cancel(cur: Any) -> None:
    """Cancel the cursor's running statement on the warehouse; best-effort, never masking."""
    try:
        cur.cancel()
    except Exception as exc:
        note_suppressed("io", "cancel Databricks statement", exc)


@dataclass(frozen=True, slots=True)
class _DatabricksWarehouseSplit:
    """A picklable warehouse read: connection params + SQL (no live conn).

    `connect_options` carries the session's ``catalog``, ``schema`` and
    ``session_configuration`` — plain values, so they travel with the split and a worker's
    session matches the driver's.
    """

    server_hostname: str
    http_path: str
    access_token: str = field(repr=False)
    query: str
    connect_options: dict[str, Any] = field(default_factory=dict, repr=False)

    def _connect(self) -> Any:
        sql = require_module(_SQL_MODULE, extra=_EXTRA)
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            # Resolved on the worker: the split carries the reference, not the token.
            access_token=resolve_secret(self.access_token, what="Databricks access_token"),
            **self.connect_options,
        )

    @contextmanager
    def _statement(self) -> Iterator[Any]:
        """A cursor that has run the query, cancelled remotely if the read is abandoned.

        A warehouse keeps executing a statement whose client went away, and keeps billing
        for it. So when the caller stops early — an exception, an interrupt, or a consumer
        that closes the stream after the rows it needed — the statement is cancelled on the
        warehouse before the connection closes. A failure names the warehouse's query id,
        which is what the query history and a support ticket are keyed on.
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(self.query)
            except Exception as exc:
                raise BackendError(
                    f"Databricks query failed (query id {getattr(cur, 'query_id', None)}): {exc}"
                ) from exc
            _LOGGER.info("Databricks query id %s", getattr(cur, "query_id", None))
            try:
                yield cur
            except BaseException:
                _cancel(cur)
                raise
        finally:
            conn.close()

    def _table(self) -> pa.Table:
        with self._statement() as cur:
            result = cur.fetchall_arrow()
            if isinstance(result, pa.RecordBatch):
                result = pa.Table.from_batches([result])
            return result

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
        with self._statement() as cur:
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

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        fingerprint = connection_fingerprint(
            _warehouse_identity(self.server_hostname, self.http_path, self.connect_options)
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
        catalog: The warehouse session's default catalog (warehouse fallback).
        db_schema: The warehouse session's default schema (warehouse fallback);
            ``bt.read.databricks(schema=...)`` sets it.
        session_configuration: Spark/SQL configuration for the warehouse session, passed
            to ``databricks.sql.connect(session_configuration=...)``.
        statement_timeout_s: Cancel the warehouse statement after this many seconds, via
            the session's ``STATEMENT_TIMEOUT`` configuration.

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
    catalog: str | None = None
    db_schema: str | None = None
    session_configuration: dict[str, Any] | None = field(default=None, repr=False)
    statement_timeout_s: int | None = None

    def _connect_options(self) -> dict[str, Any]:
        """The warehouse session options, as ``databricks.sql.connect`` spells them."""
        options: dict[str, Any] = {}
        if self.catalog:
            options["catalog"] = self.catalog
        if self.db_schema:
            options["schema"] = self.db_schema
        config = dict(self.session_configuration or {})
        if self.statement_timeout_s is not None:
            config["STATEMENT_TIMEOUT"] = str(int(self.statement_timeout_s))
        if config:
            options["session_configuration"] = config
        return options

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

    def _lakehouse(self) -> tuple[str, str, str]:
        """`(table, workspace, token)`, for a code path only a lakehouse read reaches."""
        if not (self.table and self.workspace and self.token):
            raise BackendError("a Databricks lakehouse read needs table=, workspace= and token=")
        return self.table, self.workspace, self.token

    def _warehouse(self) -> tuple[str, str, str, str]:
        """`(server_hostname, http_path, access_token, query)`, for a warehouse-only path."""
        if not (self.query and self.server_hostname and self.http_path and self.access_token):
            raise BackendError(
                "a Databricks warehouse read needs query=, server_hostname=, http_path= "
                "and access_token="
            )
        return self.server_hostname, self.http_path, self.access_token, self.query

    def _delta_source(self) -> DeltaSource:
        """Vend Unity credentials and build a direct Delta reader for the table."""
        lease = unity_lease(*self._lakehouse())
        return _LeasedDeltaSource(lease)

    def _warehouse_split(
        self, predicate: dict | None = None, projection: list[str] | None = None
    ) -> _DatabricksWarehouseSplit:
        """The warehouse split, with the pushdown already folded into its SQL (see `push_down`)."""
        host, http_path, token, query = self._warehouse()
        return _DatabricksWarehouseSplit(
            host, http_path, token, push_down(query, predicate, projection), self._connect_options()
        )

    def schema(self) -> pa.Schema:
        if self._is_lakehouse():
            return self._delta_source().schema()
        host, http_path, token, query = self._warehouse()
        probed = _DatabricksWarehouseSplit(
            host, http_path, token, schema_probe(query), self._connect_options()
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
            _warehouse_identity(self.server_hostname, self.http_path, self._connect_options())
        )
        return f"databricks-wh:{fingerprint}:{self.query}"

    def governed_name(self) -> str:
        """The Databricks table, when one was named rather than a query.

        Distinct from `identity`, which names a *relation*: it folds in a
        `connection_fingerprint` so the same table on staging and on production cannot share
        one statistics entry. A policy is the other thing. It is written before the first
        read by someone who has to be able to **type the name**, and a fingerprint is a
        sha256 of the connection options -- so keying governance on the identity meant a
        policy on this connector could not be written at all. Every read was ungoverned, and
        nothing said so.

        Returns:
            The table name a policy is keyed on, or ``""`` when there is none.
        """
        return self.table or ""

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
            delta = self._delta_source()
            planned = delta.splits(target_size, predicate)
            lease = getattr(delta, "lease", None)
            if lease is None:
                return planned
            table, workspace, token = self._lakehouse()
            _remember(workspace, table, lease)
            return [_renewing(split, table, workspace, token, lease) for split in planned]
        return [self._warehouse_split(predicate, projection)]


#: Renew a lease this long before Unity says it expires, so a file read that starts just
#: before the deadline does not run past it.
_RENEW_MARGIN_S = 300.0

#: This process's newest lease per ``(workspace, table)``. A worker holding many splits of
#: one table renews once for all of them rather than once per file.
_LEASES: dict[tuple[str, str], UnityLease] = {}
_LEASE_LOCK = threading.Lock()


class _LeasedDeltaSource(DeltaSource):
    """A Delta reader over vended credentials, remembering the lease they came from."""

    def __init__(self, lease: UnityLease) -> None:
        super().__init__(lease.storage_url, storage_options=lease.storage_options)
        self.lease = lease


def _remember(workspace: str, table: str, lease: UnityLease) -> None:
    with _LEASE_LOCK:
        _LEASES[(workspace, table)] = lease


def _fresh(lease: UnityLease, now: float) -> bool:
    return lease.expires_at_s is None or now < lease.expires_at_s - _RENEW_MARGIN_S


@dataclass(frozen=True, slots=True)
class UnityDeltaFileSplit(DeltaFileSplit):
    """A Delta file split whose vended credentials are renewed when they near expiry.

    Unity vends credentials that last minutes to an hour, and the driver vends them once, at
    planning time. A split that waits behind a long queue, or a scan that runs for hours,
    used to reach the object store with credentials that had already lapsed and fail with a
    403 partway through. This split carries what vending needs -- the table, the workspace,
    and the token or a reference to it -- and re-vends on the worker once the lease it was
    planned with is within `_RENEW_MARGIN_S` of expiring. The renewed lease is shared by
    every split of the table on that worker.

    The token travels with the split, as a warehouse split's does; pass it as a secret
    reference (``token="env:DATABRICKS_TOKEN"``) so what travels is the reference.
    """

    unity_table: str = ""
    workspace: str = ""
    token: str = field(default="", repr=False)
    expires_at_s: float | None = None

    def _live_options(self) -> dict[str, str] | None:
        """The storage options to read with: the planned ones, or a renewed lease's."""
        now = time.time()
        planned = UnityLease(self.table_uri, self.storage_options or {}, self.expires_at_s)
        if _fresh(planned, now):
            return self.storage_options
        key = (self.workspace, self.unity_table)
        with _LEASE_LOCK:
            cached = _LEASES.get(key)
            if cached is None or not _fresh(cached, now):
                cached = unity_lease(self.unity_table, self.workspace, self.token)
                _LEASES[key] = cached
        return cached.storage_options

    def _snapshot(self) -> Any:
        from batcher.io.formats.lakehouse.delta._snapshot import open_snapshot

        return open_snapshot(
            self.table_uri, version=self.version, storage_options=self._live_options()
        )

    def schema(self) -> pa.Schema:
        return self._snapshot().schema()


def _renewing(split: Split, table: str, workspace: str, token: str, lease: UnityLease) -> Split:
    """`split` as a renewing split, when it is a per-file Delta split; unchanged otherwise."""
    if type(split) is not DeltaFileSplit:
        return split
    return UnityDeltaFileSplit(
        split.table_uri,
        split.file_path,
        split.storage_options,
        split.version,
        split.rows,
        split.partition_columns,
        split.partition_values,
        unity_table=table,
        workspace=workspace,
        token=token,
        expires_at_s=lease.expires_at_s,
    )
