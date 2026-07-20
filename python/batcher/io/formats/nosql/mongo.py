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
from batcher.io.formats.sql._common import connection_fingerprint
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.plan.source_stats import SourceStatistics

__all__ = ["MongoSink", "MongoSource"]

# An ``_id`` range split is a half-open ObjectId/key interval ``(lo, hi)``; either
# bound may be None to mean "unbounded on that side".
_IdRange = tuple[Any, Any]


def redact_mongo_uri(uri: str | None) -> str | None:
    """A MongoDB URI with the password in its userinfo masked.

    A Mongo URI carries its credentials inline — ``mongodb://user:pw@host/db`` — so unlike
    a kwarg named ``password`` there is no field to drop. That matters here more than for a
    `repr`, because the URI is what makes a connection *identifying*, and `identity()` is
    **persisted** as the learned-statistics key. Fingerprinting the raw URI would write the
    password into the metadata store, where it outlives the process.

    Masking rather than removing the whole userinfo keeps the username identifying, so two
    different accounts against one host stay distinct relations.

    Args:
        uri: A MongoDB connection URI, or None.

    Returns:
        The URI with the password masked, or None.
    """
    if not uri:
        return uri
    scheme, sep, rest = uri.partition("://")
    if not sep or "@" not in rest:
        return uri
    userinfo, _, hostpart = rest.rpartition("@")
    user, has_pw, _ = userinfo.partition(":")
    userinfo = f"{user}:***" if has_pw else userinfo
    return f"{scheme}://{userinfo}@{hostpart}"


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

    def _client(self) -> Any:
        pymongo = require_driver("pymongo", "mongo")
        # The URI embeds credentials, so it resolves like any other secret.
        return pymongo.MongoClient(self._secret("uri"))

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
        """``<connection>:<db>.<collection>`` — the server is part of the relation's identity.

        Keyed on ``database.collection`` alone, the same collection name on **production**
        and on **staging** was one relation: `identity()` is the key learned statistics are
        stored under, so Kyber applied the billion-document collection's cardinalities to
        the thousand-document one and picked a plan for data that was not there. Nothing
        errors — it is a silently worse plan, which is the hardest kind of bug to see.

        The URI is masked before it is fingerprinted (`redact_mongo_uri`), and the digest is
        `sha256` rather than `hash()` so the key is stable across processes; a per-run key
        would make the stats loop look alive while never reusing anything.
        """
        kw = self._conn_kwargs
        fingerprint = connection_fingerprint({"uri": redact_mongo_uri(kw["uri"])})
        return f"{fingerprint}:{kw['database']}.{kw['collection']}"

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
        self,
        partition: _IdRange,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """One ``_id`` range, fetched as Arrow and yielded batch by batch.

        **This is not incrementally streamed, and deliberately does not pretend to be.**
        ``pymongoarrow`` exposes no batched reader — `find_arrow_all` is its only `find`
        entry point and it returns a fully materialized `Table`. The honest alternative is
        a plain ``pymongo`` cursor through `rows_to_batches`, which *would* stream, but it
        gives up ``pymongoarrow``'s BSON typing: an ``ObjectId`` or ``Decimal128`` that the
        Arrow reader types correctly is something `pa.RecordBatch.from_pylist` cannot
        encode at all. Trading a correctness property for a memory one is the wrong trade,
        so the bound stays where it is — at the *partition*, which is why `partition_spec`
        segments the ``_id`` space rather than reading the collection in one call.

        The client is closed **before** the first batch is yielded rather than in a
        `finally` around the yields. A generator abandoned after its first batch runs its
        `finally` only whenever the garbage collector gets to it, so the connection (and
        its socket pool) outlived every early-exit read — a `limit`, a `head`, an exception
        downstream. Since the rows are already materialized by the time we can yield, there
        is nothing to keep it open for.
        """
        require_driver("pymongoarrow", "mongo")
        from pymongoarrow.api import find_arrow_all

        lo, hi = partition
        # Push the predicate into the `find` filter here, not into per-read instance state:
        # a worker rebuilds this source from the split's `conn_kwargs` and would otherwise
        # scan the whole collection.
        query = dict(self._with_pushed(predicate)._conn_kwargs["query"])
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
        finally:
            client.close()
        yield from table.to_batches()


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
        # Resolve the URI here, where the connection is dialed, exactly as `MongoSource`
        # does via `_secret`. The sink was calling `MongoClient(self.uri)` on the raw
        # attribute, so an ``env:``/``file:`` reference that read fine was handed to the
        # driver verbatim and failed to connect — the reference form worked for reads and
        # silently did not for writes.
        from batcher.io.credentials import resolve_secret

        client = pymongo.MongoClient(resolve_secret(self.uri, what="mongo uri"))
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
