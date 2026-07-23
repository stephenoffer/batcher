"""ADBC bulk-ingest sink.

Kept beside the source rather than inside it because the two share only a connection
helper, and the sink carries a correctness concern the source does not: a distributed
write fans one logical write across shards that all target the *same* table.

`_connect` is reached as `_source._connect` rather than imported by name. That is
deliberate and load-bearing: `from ... import _connect` binds the function object at
import time, so a test patching `source._connect` would leave this module still calling
the original — a patch that silently becomes a no-op, which is exactly how a test keeps
passing while testing nothing. Going through the module means one patch target covers
both the source and the sink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS
from batcher.io.formats.sql.adbc import source as _source
from batcher.io.manifest import WrittenFile

__all__ = ["ADBCSink"]

#: `adbc_ingest` dispositions that discard whatever the table already held. Safe for a
#: single writer; ruinous when every shard of a distributed write applies one.
_DESTRUCTIVE_MODES = frozenset({"replace", "create"})


@SINKS.register("adbc")
@dataclass(frozen=True, slots=True)
class ADBCSink:
    """Bulk-ingest Arrow tables into a database table via ADBC.

    Args:
        driver: The ADBC driver to load.
        db_kwargs: Driver/database connection kwargs (never logged).
        conn_kwargs: Extra ``connect()`` kwargs.
        mode: Ingest disposition passed to ``adbc_ingest`` (``"create"``,
            ``"append"``, ``"replace"``, ``"create_append"``).
        uri: A standard connection URI (``postgresql://host:5432/db``) supplying
            `driver` and `db_kwargs`, exactly as `ADBCSource` accepts.
        password: The password, as a literal or an ``env:``/``file:`` reference
            resolved on the worker.

    Raises:
        BackendError: If no connection is given, or `uri` names a scheme with no
            ADBC driver.
    """

    driver: str | None = None
    db_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    conn_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    mode: str = "create_append"
    uri: str | None = None
    password: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from batcher._internal.errors import BackendError

        if self.uri is not None:
            from batcher.io.formats.sql.uri import adbc_connection

            driver, merged, sanitized = adbc_connection(
                self.uri, password=self.password, driver=self.driver, db_kwargs=self.db_kwargs
            )
            object.__setattr__(self, "driver", driver)
            object.__setattr__(self, "db_kwargs", merged)
            object.__setattr__(self, "uri", sanitized)
        if self.db_kwargs is None:
            object.__setattr__(self, "db_kwargs", {})
        if self.driver is None:
            raise BackendError(
                "ADBCSink requires either uri= (e.g. 'postgresql://host/db') or an "
                "explicit driver= and db_kwargs="
            )

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Ingest `table` into the destination table named by `path`."""
        conn = _source._connect(self.driver, self.db_kwargs, self.conn_kwargs)
        try:
            cur = conn.cursor()
            cur.adbc_ingest(path, table, mode=self.mode)
            conn.commit()
        finally:
            conn.close()
        return WrittenFile(path=path, rows=table.num_rows, bytes=0)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - DB ingest is unpartitioned
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Ingest one shard; each worker appends to the same destination table.

        A file sink gives every shard its own ``part-N`` file, so shards cannot collide.
        A database sink has no such luxury: every shard ingests into **one** table, and a
        disposition that replaces that table is applied by *each* shard independently.

        With ``mode="replace"`` that silently destroyed the write. Six rows across three
        shards left two rows in the table — each shard dropped and recreated what the
        previous one had just written, and nothing raised. It is invisible single-node,
        where there is only ever one shard, and appears only at cluster scale as a wrong
        answer rather than an error.

        Doing it correctly needs the replace to happen exactly once, before any shard
        writes. Shards run concurrently on separate workers, so "let shard 0 replace"
        does not work either — an append that lands before the replace is destroyed by
        it. There is no driver-side prepare hook on the `Sink` protocol to hang that on,
        so this refuses instead: a loud error beats losing rows quietly.

        Raises:
            BackendError: If a destructive `mode` meets a multi-shard write.
        """
        if file_index > 0 and self.mode in _DESTRUCTIVE_MODES:
            raise BackendError(
                f"mode={self.mode!r} cannot be used for a distributed write to table "
                f"{path!r}: every shard would apply it to the same table, so each one "
                "would discard the shards before it. Write with mode='append' (or "
                "'create_append'), truncating the table beforehand if you need it empty."
            )
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No-op: ADBC ingests are committed per shard on write."""
