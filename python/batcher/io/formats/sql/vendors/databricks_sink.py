"""Databricks SQL sink — stage Parquet in a Unity Catalog volume, then ``COPY INTO``.

Row-by-row ``INSERT`` through a SQL warehouse is the slow way to load Databricks. The bulk
path the SQL connector supports is two statements: a ``PUT`` that uploads a local file into
a Unity Catalog volume (the connector only allows it under the ``staging_allowed_local_path``
the connection was opened with), and a ``COPY INTO`` that loads the staged file into the
table. Each shard does both with its own Parquet file and then ``REMOVE``s it.

**Every staged file gets a fresh name.** ``COPY INTO`` is idempotent per file path: a path
it has already loaded is skipped on the next run. Reusing a shard's name across writes would
make a second write silently load nothing, so the name carries a UUID.

The destination table must exist. Only ``mode="append"`` is offered: an overwrite would need
a ``TRUNCATE`` and a ``COPY INTO`` that do not commit together, and a write that can leave
the table empty on failure is worse than one that refuses.

The returned `WrittenFile.job` carries the ``COPY INTO`` statement's ``query_id`` and the
row counts the warehouse reported.

Not yet verified against a live Databricks workspace; see ``tests/PENDING_VERIFICATION.md``.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from batcher._internal.errors import BackendError
from batcher._internal.logging import note_suppressed
from batcher.io.credentials import resolve_secret
from batcher.io.formats.base import SINKS
from batcher.io.formats.sql._common import require_module
from batcher.io.manifest import WrittenFile

__all__ = ["DatabricksSink", "databricks_table_name"]

_EXTRA = "databricks"
_MODULE = "databricks.sql"


def databricks_table_name(table: str) -> str:
    """`table` with each dotted part delimited in backticks, as Databricks SQL quotes them.

    Args:
        table: ``table``, ``schema.table`` or ``catalog.schema.table``.

    Returns:
        The quoted, possibly qualified name.

    Raises:
        BackendError: If a part is empty.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors.databricks_sink import databricks_table_name
            >>> databricks_table_name("main.sales.order")
            '`main`.`sales`.`order`'
    """
    parts = table.split(".")
    if not all(parts):
        raise BackendError(f"invalid Databricks table name {table!r}")
    return ".".join(f"`{part.replace('`', '``')}`" for part in parts)


def _literal_path(path: str) -> str:
    """A volume path as a SQL string literal, refusing a quote rather than escaping one."""
    if "'" in path or "\\" in path:
        raise BackendError(
            f"Databricks staging path {path!r} may not contain quotes or backslashes"
        )
    return f"'{path}'"


def _first_row(cur: Any) -> dict[str, Any]:
    """The first result row of the last statement as a dict, or empty when there is none."""
    rows = cur.fetchall() if getattr(cur, "description", None) else []
    if not rows:
        return {}
    names = [d[0] for d in cur.description]
    return dict(zip(names, tuple(rows[0]), strict=False))


def _remove_staged(cur: Any, remote: str) -> None:
    """Delete a staged file; a failed cleanup is logged, never allowed to mask the load."""
    try:
        cur.execute(f"REMOVE {_literal_path(remote)}")
    except Exception as exc:
        note_suppressed("io", f"remove staged Databricks file {remote}", exc)


@SINKS.register("databricks")
@dataclass(frozen=True, slots=True)
class DatabricksSink:
    """Bulk-load Arrow tables into an existing Databricks table through a volume.

    Not yet verified against a live Databricks workspace; see tests/PENDING_VERIFICATION.md.

    Args:
        server_hostname: The SQL warehouse hostname.
        http_path: The SQL warehouse HTTP path.
        access_token: A token or an ``env:``/``file:`` reference to one. Never logged.
        volume_path: A Unity Catalog volume directory to stage files in, e.g.
            ``/Volumes/main/staging/loads``.
        catalog: The session's default catalog.
        db_schema: The session's default schema.
        mode: ``"append"``, the only mode offered.

    Raises:
        BackendError: If `mode` is not ``"append"`` or `volume_path` is not a volume.
    """

    server_hostname: str
    http_path: str
    access_token: str = field(repr=False)
    volume_path: str
    catalog: str | None = None
    db_schema: str | None = None
    mode: str = "append"

    def __post_init__(self) -> None:
        if self.mode != "append":
            raise BackendError(
                f"Databricks write mode {self.mode!r} is not supported: only 'append' is, "
                "because an overwrite would truncate and load in two commits. TRUNCATE the "
                "table first, or write the table's storage with ds.write.delta."
            )
        if not self.volume_path.startswith("/Volumes/"):
            raise BackendError(
                f"volume_path={self.volume_path!r} must be a Unity Catalog volume path "
                "starting with /Volumes/."
            )

    def _connect(self, staging_dir: str) -> Any:
        sql = require_module(_MODULE, extra=_EXTRA)
        options: dict[str, Any] = {}
        if self.catalog:
            options["catalog"] = self.catalog
        if self.db_schema:
            options["schema"] = self.db_schema
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=resolve_secret(self.access_token, what="Databricks access_token"),
            staging_allowed_local_path=staging_dir,
            **options,
        )

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Stage `table` as Parquet and ``COPY INTO`` the table named `path`.

        Args:
            table: The rows to load.
            path: The destination table, optionally ``catalog.schema.table``.

        Returns:
            The rows loaded, and the ``COPY INTO`` statement's identity under `job`.
        """
        name = f"batcher-{uuid.uuid4().hex}.parquet"
        remote = f"{self.volume_path.rstrip('/')}/{name}"
        with tempfile.TemporaryDirectory() as staging_dir:
            local = os.path.join(staging_dir, name)
            pq.write_table(table, local)
            size = os.path.getsize(local)
            conn = self._connect(staging_dir)
            try:
                cur = conn.cursor()
                cur.execute(f"PUT {_literal_path(local)} INTO {_literal_path(remote)} OVERWRITE")
                try:
                    job = self._copy_into(cur, path, remote)
                finally:
                    _remove_staged(cur, remote)
            finally:
                conn.close()
        loaded = job.get("num_inserted_rows")
        rows = int(loaded) if loaded is not None else table.num_rows
        return WrittenFile(path=path, rows=rows, bytes=size, job=job)

    @staticmethod
    def _copy_into(cur: Any, path: str, remote: str) -> dict[str, Any]:
        """Run the ``COPY INTO`` and describe it, naming its query id when it fails."""
        target, source = databricks_table_name(path), _literal_path(remote)
        statement = f"COPY INTO {target} FROM {source} FILEFORMAT = PARQUET"
        try:
            cur.execute(statement)
        except Exception as exc:
            query_id = getattr(cur, "query_id", None)
            raise BackendError(
                f"Databricks COPY INTO {path!r} failed (query id {query_id}): {exc}"
            ) from exc
        return {
            "system": "databricks",
            "query_id": getattr(cur, "query_id", None),
            **_first_row(cur),
        }

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - a table, not a directory
        file_index: int = 0,  # noqa: ARG002 - appends from every shard are independent
    ) -> list[WrittenFile]:
        """Load one shard; appends from concurrent shards stage distinct files."""
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No-op: each ``COPY INTO`` commits its own rows."""
