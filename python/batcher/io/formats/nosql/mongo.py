"""MongoDB connector — Arrow-native read via ``pymongoarrow``, batch-upsert write.

`MongoSource` reads a collection through ``pymongoarrow.api.find_arrow_all``,
which returns an Arrow `Table` directly (no per-row Python). Parallel reads split
the ``_id`` key space into contiguous ObjectId ranges: each split issues a bounded
``find`` over its half-open ``[lo, hi)`` range, so the ranges are a disjoint,
exhaustive cover. `MongoSink` writes an Arrow table back as a batch of bulk
upserts keyed by a chosen field.

Both defer ``pymongo`` / ``pymongoarrow`` so importing this module never requires
the drivers; a missing driver raises `BackendError` with the ``mongo`` extra hint.
Connection kwargs (URI, auth) are stored verbatim and never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import PartitionSpec, ScanSource, require_driver
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.plan.source_stats import SourceStatistics

__all__ = ["MongoSink", "MongoSource"]

# An ``_id`` range split is a half-open ObjectId/key interval ``(lo, hi)``; either
# bound may be None to mean "unbounded on that side".
_IdRange = tuple[Any, Any]


@SOURCES.register("mongo")
class MongoSource(ScanSource):
    """A MongoDB collection read as Arrow via ``pymongoarrow``.

    Args:
        uri: A MongoDB connection URI (``mongodb://…``); never logged.
        database: The database name.
        collection: The collection name.
        query: Optional Mongo filter document applied to every read.
        partition_spec: Optional parallelism hint; ``segments`` sets the number
            of ``_id`` ranges to split into.
    """

    format_name = "mongo"

    # Predicate pushdown: Kyber's pushed predicate → a Mongo filter document
    # merged into the ``find`` filter, so the server prunes before returning Arrow.
    supports_predicate = True

    __slots__ = ()

    def __init__(
        self,
        *,
        uri: str,
        database: str,
        collection: str,
        query: dict[str, Any] | None = None,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            uri=uri,
            database=database,
            collection=collection,
            query=query or {},
        )

    def _with_pushed(self, predicate: dict | None) -> MongoSource:
        """A sibling source whose ``query`` merges in the pushable predicate.

        Returns ``self`` when `predicate` is absent or unpushable; otherwise a new
        `MongoSource` whose find filter is the existing query AND-combined with the
        pushed Mongo filter document. The engine's `Filter` re-check keeps the
        result correct regardless.
        """
        if predicate is None:
            return self
        from batcher.io.predicate import to_mongo_filter

        pushed = to_mongo_filter(predicate)
        if pushed is None:
            return self
        existing = self._conn_kwargs["query"]
        merged = {"$and": [existing, pushed]} if existing else pushed
        return MongoSource(
            uri=self._conn_kwargs["uri"],
            database=self._conn_kwargs["database"],
            collection=self._conn_kwargs["collection"],
            query=merged,
            partition_spec=self._partition_spec,
        )

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self._with_pushed(predicate).iter_batches(projection))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        source = self._with_pushed(predicate)
        for partition in source._enumerate_partitions():
            yield from source._read_partition(partition, projection)

    def _client(self) -> Any:
        pymongo = require_driver("pymongo", "mongo")
        return pymongo.MongoClient(self._conn_kwargs["uri"])

    def _coll(self, client: Any) -> Any:
        return client[self._conn_kwargs["database"]][self._conn_kwargs["collection"]]

    def row_count(self) -> int | None:
        """Exact matching-document count via `count_documents` (no data transfer)."""
        try:
            client = self._client()
            try:
                return int(self._coll(client).count_documents(self._conn_kwargs["query"]))
            finally:
                client.close()
        except Exception:
            return None

    def statistics(self) -> SourceStatistics | None:
        """Exact row count from `count_documents`; no column stats from Mongo."""
        rows = self.row_count()
        return None if rows is None else SourceStatistics(row_count=rows, exact_rows=True)

    def _identity_suffix(self) -> str:
        return f"{self._conn_kwargs['database']}.{self._conn_kwargs['collection']}"

    def _infer_schema(self) -> pa.Schema:
        require_driver("pymongoarrow", "mongo")
        from pymongoarrow.api import find_arrow_all

        client = self._client()
        try:
            table = find_arrow_all(self._coll(client), self._conn_kwargs["query"], limit=1)
            return table.schema
        finally:
            client.close()

    def _enumerate_partitions(self) -> list[_IdRange]:
        segments = max(1, self._partition_spec.segments)
        if segments == 1:
            return [(None, None)]
        client = self._client()
        try:
            coll = self._coll(client)
            return _id_ranges(coll, self._conn_kwargs["query"], segments)
        finally:
            client.close()

    def _read_partition(
        self, partition: _IdRange, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        require_driver("pymongoarrow", "mongo")
        from pymongoarrow.api import find_arrow_all

        lo, hi = partition
        query = dict(self._conn_kwargs["query"])
        id_filter: dict[str, Any] = {}
        if lo is not None:
            id_filter["$gte"] = lo
        if hi is not None:
            id_filter["$lt"] = hi
        if id_filter:
            query["_id"] = id_filter
        projection_doc = dict.fromkeys(projection, 1) if projection else None
        client = self._client()
        try:
            table = find_arrow_all(self._coll(client), query, projection=projection_doc)
            yield from table.to_batches()
        finally:
            client.close()


def _id_ranges(coll: Any, query: dict[str, Any], segments: int) -> list[_IdRange]:
    """Split the ``_id`` key space into `segments` contiguous half-open ranges.

    Samples evenly-spaced boundary ``_id`` values by sorted offset so each range
    holds a comparable row count; the first range is left-open and the last
    right-open, making the set a disjoint, exhaustive cover.
    """
    total = coll.count_documents(query)
    if total == 0:
        return [(None, None)]
    step = max(1, total // segments)
    boundaries: list[Any] = []
    for offset in range(step, total, step):
        cursor = coll.find(query, {"_id": 1}).sort("_id", 1).skip(offset).limit(1)
        doc = next(iter(cursor), None)
        if doc is not None:
            boundaries.append(doc["_id"])
    bounds = [None, *boundaries, None]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


@SINKS.register("mongo")
class MongoSink:
    """Write an Arrow table to a MongoDB collection as batched bulk upserts.

    Each row is upserted on `key_field` (replacing the matching document, or
    inserting if absent), in one ``bulk_write`` per call — never a per-row network
    round trip. Returns a `WrittenFile` recording the row count for the manifest.

    Args:
        uri: A MongoDB connection URI; never logged.
        database: The target database name.
        collection: The target collection name.
        key_field: The document field upserts match on (default ``"_id"``).
    """

    __slots__ = ("collection", "database", "key_field", "uri")

    def __init__(
        self,
        *,
        uri: str,
        database: str,
        collection: str,
        key_field: str = "_id",
    ) -> None:
        self.uri = uri
        self.database = database
        self.collection = collection
        self.key_field = key_field

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Upsert every row of `table`; `path` is the logical target identifier."""
        pymongo = require_driver("pymongo", "mongo")
        rows = table.to_pylist()
        if not rows:
            return WrittenFile(path=path, rows=0, bytes=table.nbytes)
        ops = [
            pymongo.ReplaceOne({self.key_field: row.get(self.key_field)}, row, upsert=True)
            for row in rows
        ]
        client = pymongo.MongoClient(self.uri)
        try:
            client[self.database][self.collection].bulk_write(ops, ordered=False)
        except Exception as exc:
            raise BackendError(f"mongo bulk upsert failed: {exc}") from exc
        finally:
            client.close()
        return WrittenFile(path=path, rows=len(rows), bytes=table.nbytes)

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - Mongo has no Hive layout
        file_index: int = 0,  # noqa: ARG002
    ) -> list[WrittenFile]:
        """Write one shard; Mongo collections are unpartitioned, so this is `write`."""
        return [self.write(table, path)]

    def commit(self, manifest: WriteManifest, path: str) -> None:
        """No-op: upserts are visible on write (no transactional commit phase)."""
