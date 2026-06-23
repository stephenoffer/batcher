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
    require_driver,
    rows_to_batches,
)

__all__ = ["Neo4jSource"]

# A window locator: ``(skip, limit)`` over an ordered Cypher result.
_Window = tuple[int, int]

# Rows per window when the source is partitioned.
_WINDOW_ROWS = 100_000


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
        return neo4j.GraphDatabase.driver(kw["uri"], auth=(kw["username"], kw["password"]))

    def _identity_suffix(self) -> str:
        kw = self._conn_kwargs
        db = kw["database"] or "default"
        return f"{kw['uri']}/{db}"

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
        segments = max(1, self._partition_spec.segments)
        if segments == 1:
            return [(0, 0)]  # 0 limit = the whole query in one window.
        return [(i * _WINDOW_ROWS, _WINDOW_ROWS) for i in range(segments)]

    def _read_partition(
        self, partition: _Window, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        skip, limit = partition
        cypher = self._conn_kwargs["cypher"]
        if limit:
            order = self._conn_kwargs["order_by"]
            if order:
                cypher += f" ORDER BY {order}"
            cypher += f" SKIP {skip} LIMIT {limit}"
        driver = self._driver()
        schema = self.schema() if not projection else None
        try:
            for batch in rows_to_batches(self._run(driver, cypher), schema=schema):
                yield batch.select(projection) if projection else batch
        finally:
            driver.close()
