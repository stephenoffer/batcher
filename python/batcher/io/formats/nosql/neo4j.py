"""Neo4j connector — Bolt driver, SKIP/LIMIT partitioned read to Arrow.

`Neo4jSource` runs a Cypher query over the Bolt protocol and assembles the
returned records into Arrow at batch granularity. Parallel reads partition an
*ordered* query with ``SKIP``/``LIMIT`` windows — one `Split` per window, a
disjoint and exhaustive cover. The query must be deterministic-ordered for the
windows to tile cleanly; the source appends a stable ``ORDER BY`` key the caller
supplies.

The ``neo4j`` import is deferred; a missing driver raises `BackendError` with the
``neo4j`` extra hint. Connection kwargs (uri, auth) are stored verbatim and never
logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.formats.nosql.base import (
    PartitionSpec,
    ScanSource,
    offset_windows,
    require_driver,
    rows_to_batches,
)
from batcher.io.formats.sql.uri import redact_uri

__all__ = ["Neo4jSource"]

# A window locator: ``(skip, limit)`` over an ordered Cypher result.
_Window = tuple[int, int]

# Rows per window when the source is partitioned.


@SOURCES.register("neo4j")
class Neo4jSource(ScanSource):
    """A Neo4j graph read via a Cypher query, partitioned by SKIP/LIMIT.

    Args:
        uri: A Bolt URI (``bolt://…`` / ``neo4j://…``); never logged.
        username: The database username; never logged.
        password: The database password; never logged.
        cypher: The Cypher query; must ``RETURN`` flat, named columns.
        order_by: A stable expression to order by so windows tile cleanly
            (required when partitioning into more than one window).
        database: Optional target database name (default the server default).
        partition_spec: Optional parallelism hint; ``segments`` sets how many
            ``SKIP``/``LIMIT`` windows to split into (default 1).
    """

    format_name = "neo4j"

    __slots__ = ()

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        cypher: str,
        order_by: str | None = None,
        database: str | None = None,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            uri=uri,
            username=username,
            password=password,
            cypher=cypher,
            order_by=order_by,
            database=database,
        )

    def _driver(self) -> Any:
        neo4j = require_driver("neo4j", "neo4j")
        kw = self._conn_kwargs
        return neo4j.GraphDatabase.driver(
            self._secret("uri"), auth=(kw["username"], self._secret("password"))
        )

    def _identity_suffix(self) -> str:
        """The Bolt target and database, with any password in the URI masked.

        A Bolt URI may carry credentials inline (``bolt://user:hunter2@host``), and this
        string goes into `identity()` — the key learned statistics are *persisted* under.
        The password was therefore being written to the metadata store, where it outlives
        the process; that is strictly worse than a `repr` leak, which at least dies with
        the traceback.
        """
        kw = self._conn_kwargs
        db = kw["database"] or "default"
        return f"{redact_uri(str(kw['uri']))}/{db}"

    def _fingerprint_material(self) -> dict[str, Any]:
        """Connection kwargs with the URI's password masked before it is fingerprinted.

        `connection_fingerprint` excludes `password` by key name, but a password embedded
        in `uri` is invisible to it. Masking first means rotating that password does not
        change the fingerprint — so the relation keeps the statistics it has already
        accumulated instead of silently reverting to cold estimates on every rotation.
        """
        return {**self._conn_kwargs, "uri": redact_uri(str(self._conn_kwargs["uri"]))}

    def _run(self, driver: Any, cypher: str) -> Iterator[dict[str, Any]]:
        with driver.session(database=self._conn_kwargs["database"]) as session:
            for record in session.run(cypher):
                yield dict(record)

    def _infer_schema(self) -> pa.Schema:
        driver = self._driver()
        try:
            rows = list(self._run(driver, f"{self._conn_kwargs['cypher']} LIMIT 1"))
        finally:
            driver.close()
        if not rows:
            return pa.schema([])
        return pa.RecordBatch.from_pylist(rows).schema

    def _enumerate_partitions(self) -> list[_Window]:
        """SKIP/LIMIT windows that cover the whole result — see `offset_windows`.

        A window cover over Cypher needs two things the old code had neither of: a total, so
        the windows can be sized to actually reach the end, and a deterministic order, so two
        windows cannot return the same row. Without an ``order_by`` the result order is
        undefined and *no* SKIP/LIMIT split is sound, so this refuses to split at all rather
        than return a plausible wrong answer.
        """
        segments = max(1, self._partition_spec.segments)
        if segments == 1:
            return [(0, 0)]  # one unbounded window: no count needed, and no connection made
        if not self._conn_kwargs.get("order_by"):
            # SKIP/LIMIT over an unordered result is not a cover: the windows overlap and miss.
            return [(0, 0)]
        return offset_windows(self._total_rows(), segments)

    def _total_rows(self) -> int | None:
        """The query's row count, via a Cypher ``CALL {…} RETURN count(*)`` subquery.

        Returns None on any failure (an older server with no CALL subqueries, a query that
        cannot be wrapped), which `offset_windows` reads as "do not split" — correct and
        serial, never truncated.
        """
        cypher = self._conn_kwargs["cypher"]
        counted = f"CALL {{ {cypher} }} RETURN count(*) AS __bc_n"
        driver = self._driver()
        try:
            rows = list(self._run(driver, counted))
        except Exception:
            return None
        finally:
            # This `finally` was missing entirely. The failure path here is the *expected*
            # one — an older server with no CALL subqueries, or a query that cannot be
            # wrapped — so the common case leaked a Bolt driver and its whole connection
            # pool on every partition enumeration.
            driver.close()
        if not rows:
            return None
        value = rows[0].get("__bc_n") if isinstance(rows[0], dict) else None
        return int(value) if isinstance(value, (int, float)) else None

    def _read_partition(
        self,
        partition: _Window,
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 (no server-side filter; the engine's Filter re-checks)
    ) -> Iterator[pa.RecordBatch]:
        skip, limit = partition
        cypher = self._conn_kwargs["cypher"]
        # A window is ``(skip, limit)`` where ``limit == 0`` means *unbounded* — the tail of
        # the cover, which must still honour its ``skip`` and run to the end of the result.
        # Guarding on ``limit`` alone dropped the ``SKIP`` for that tail window, so it re-read
        # the whole result and every prior window's rows came back a second time.
        if skip or limit:
            order = self._conn_kwargs["order_by"]
            if order:
                cypher += f" ORDER BY {order}"
            cypher += f" SKIP {skip}"
            if limit:
                cypher += f" LIMIT {limit}"
        # Resolved before the driver is opened: `self.schema()` opens a driver of its own,
        # so this held two at once, and a raise from it landed between `_driver()` and the
        # `try` — stranding the outer one with nothing left holding a reference to close it.
        schema = self.schema() if not projection else None
        driver = self._driver()
        try:
            for batch in rows_to_batches(self._run(driver, cypher), schema=schema):
                yield batch.select(projection) if projection else batch
        finally:
            driver.close()
