"""MongoDB connector — Arrow-native read via ``pymongoarrow``, batch-upsert write.

`MongoSource` reads a collection through ``pymongoarrow.api.find_arrow_all``,
which returns an Arrow `Table` directly (no per-row Python). Parallel reads split
the ``_id`` key space into contiguous ObjectId ranges: each split issues a bounded
``find`` over its half-open ``[lo, hi)`` range, so the ranges are a disjoint,
exhaustive cover. `MongoSink` maintains a collection with the
same `upsert`/`append`/`overwrite`/`delete` vocabulary the SQL sink uses, one bulk
round trip per call.

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
from batcher.io.formats.nosql.base import (
    BulkSink,
    PartitionSpec,
    ScanSource,
    require_driver,
)
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
        return f"{self._conn_kwargs['database']}.{self._conn_kwargs['collection']}"

    def _fingerprint_material(self) -> dict[str, Any]:
        """`_conn_kwargs` with the URI's password masked before it is fingerprinted.

        `ScanSource.identity()` fingerprints the connection so that the same
        ``database.collection`` on production and on staging are different relations. It
        drops credentials it can recognize *by key name*, which a Mongo URI defeats: the
        password lives inside ``mongodb://user:pw@host/db``, in a field named ``uri``.

        Left unmasked, the raw password is digested into `identity()` — the key learned
        statistics are **persisted** under. Two consequences, both silent. The digest is
        one-way so the secret does not read back out, but every credential rotation
        produces a *different key*, orphaning everything Kyber has learned about the
        collection and returning it to cold estimates on whatever schedule the security
        team rotates on. Masking the password (and keeping the username, which genuinely
        identifies the connection) makes the key stable across rotations.
        """
        return {**self._conn_kwargs, "uri": redact_mongo_uri(self._conn_kwargs.get("uri"))}

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
class MongoSink(BulkSink):
    """Write an Arrow table to a MongoDB collection, in one bulk round trip.

    Every mode issues a single ``bulk_write`` per call — never a per-row network round
    trip — and returns a `WrittenFile` recording the row count for the manifest.

    Args:
        uri: A MongoDB connection URI; never logged, and an ``env:``/``file:`` reference
            is resolved where the connection is dialed.
        database: The target database name.
        collection: The target collection, when it is not the write's destination name.
            `ds.write.mongo("orders", ...)` already names it, so this is normally omitted;
            requiring it made that documented call raise ``TypeError: missing 1 required
            keyword-only argument: 'collection'`` on every invocation, because the writer
            passes the collection as the *destination*, never as a keyword.
        key_field: The document field `upsert` and `delete` match on (default ``"_id"``).
        mode: One of `STORE_WRITE_MODES`.
    """

    format_name = "mongo"

    __slots__ = ("collection", "database", "uri")

    def __init__(
        self,
        *,
        uri: str,
        database: str,
        collection: str | None = None,
        key_field: str = "_id",
        mode: str = "upsert",
    ) -> None:
        super().__init__(key_field=key_field, mode=mode, uri=uri)
        self.uri = uri
        self.database = database
        self.collection = collection

    def _operations(self, pymongo: Any, rows: list[dict[str, Any]]) -> list[Any]:
        """The bulk operations `mode` turns `rows` into."""
        if self.mode == "delete":
            return [pymongo.DeleteOne({self.key_field: row.get(self.key_field)}) for row in rows]
        if self.mode in ("append", "overwrite"):
            # An overwrite empties the collection first (below) and then inserts, so both
            # modes insert here. Neither matches on a key: an append that silently replaced
            # a document holding the same `_id` would be an upsert under another name.
            return [pymongo.InsertOne(row) for row in rows]
        return [
            pymongo.ReplaceOne({self.key_field: row.get(self.key_field)}, row, upsert=True)
            for row in rows
        ]

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Apply `rows` to the collection `path` names, or to an explicit `collection`."""
        pymongo = require_driver("pymongo", "mongo")
        ops = self._operations(pymongo, rows)
        client = pymongo.MongoClient(self._secret("uri"))
        try:
            target = client[self.database][self.collection or path]
            if self.mode == "overwrite":
                target.delete_many({})
            if ops:
                target.bulk_write(ops, ordered=False)
        except Exception as exc:
            raise BackendError(f"mongo bulk {self.mode} failed: {exc}") from exc
        finally:
            client.close()
