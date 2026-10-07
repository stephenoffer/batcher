"""Milvus connector — query a collection's partitions as Arrow; upsert, insert and delete.

Both halves speak pymilvus' ``MilvusClient``, which reaches a Milvus server by URI
(``"http://localhost:19530"``) and Milvus Lite by a local file path (``"./milvus.db"``).

`MilvusSink` reads the collection's schema with ``describe_collection`` before the first
request and refuses a frame that does not fit it: a vector of the wrong dimension, a metric
that disagrees with the vector field's index, or a payload column that is not a field of the
collection (unless the collection has its dynamic field enabled). The id column is sent as the
collection's primary-key field and the vector column as its float-vector field, whatever each
is called there. ``upsert`` replaces rows by primary key. ``append`` is Milvus ``insert``,
which does not check the key, so it is never retried: a retry of a request whose response
was lost would store the rows twice. ``partition=`` targets one partition.

`MilvusSource` reads with ``query_iterator``, one split per partition, so the partitions of a
collection read in parallel. Field types map onto Arrow; a field of a type with no mapping here
(JSON, arrays, sparse or binary vectors) needs a declared ``schema=``.

The ``pymilvus`` import is deferred to the worker; a missing client raises with the
``milvus`` extra. Not yet verified against a live Milvus; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import ScanSource, rows_to_batches
from batcher.io.formats.vector.contract import (
    DEFAULT_ID_COLUMN,
    DEFAULT_VECTOR_COLUMN,
    RemoteTarget,
    VectorSink,
    vector_field,
)

__all__ = ["MilvusSink", "MilvusSource"]

#: Rows per ``query_iterator`` page.
_PAGE_ROWS = 1_000

#: Milvus scalar field types and the Arrow type each is read as.
_SCALARS = {
    "BOOL": pa.bool_(),
    "INT8": pa.int8(),
    "INT16": pa.int16(),
    "INT32": pa.int32(),
    "INT64": pa.int64(),
    "FLOAT": pa.float32(),
    "DOUBLE": pa.float64(),
    "VARCHAR": pa.string(),
    "STRING": pa.string(),
}


def _milvus() -> Any:
    """The ``pymilvus`` module, or a typed install hint."""
    from batcher._internal.optional import require

    return require("pymilvus", feature="Milvus", provides="pymilvus", extra="milvus")


def _open(kwargs: dict[str, Any], token: Any) -> Any:
    """A `MilvusClient` for ``uri`` (a server URL or a Milvus Lite file)."""
    params: dict[str, Any] = {"uri": kwargs["uri"]}
    if token is not None:
        params["token"] = token
    if kwargs.get("db_name"):
        params["db_name"] = kwargs["db_name"]
    return _milvus().MilvusClient(**params)


def _type_name(field: dict[str, Any]) -> str:
    """A field's type as its ``DataType`` member name, such as ``"FLOAT_VECTOR"``."""
    dtype = field["type"]
    return str(getattr(dtype, "name", dtype)).upper()


def _dim(field: dict[str, Any]) -> int:
    return int(field.get("params", {})["dim"])


def _describe_collection(client: Any, collection: str) -> dict[str, Any]:
    """``describe_collection`` for an existing collection.

    Raises:
        PlanError: If the collection does not exist.
    """
    if not client.has_collection(collection):
        raise PlanError(
            f"milvus collection {collection!r} does not exist",
            hint="Create it with its primary key and vector field before writing or reading.",
        )
    return client.describe_collection(collection)


@SOURCES.register("milvus")
class MilvusSource(ScanSource):
    """A Milvus collection, one split per partition, read with ``query_iterator``.

    Not yet verified against a live Milvus; see tests/PENDING_VERIFICATION.md.

    Args:
        collection: The collection to read.
        uri: A server URL, or a local Milvus Lite database file.
        token: ``"user:password"`` or an API key; never logged, resolved on the worker.
        db_name: The database, when not the default.
        partitions: The partitions to read; every partition by default.
        filter: A Milvus boolean expression, evaluated by the server.
        schema: A declared schema, which skips reading the collection's.
    """

    format_name = "milvus"

    __slots__ = ()

    def __init__(
        self,
        *,
        collection: str,
        uri: str,
        token: str | None = None,
        db_name: str | None = None,
        partitions: list[str] | None = None,
        filter: str = "",
        schema: pa.Schema | None = None,
    ) -> None:
        super().__init__(
            schema=schema,
            collection=collection,
            uri=uri,
            token=token,
            db_name=db_name,
            partitions=partitions,
            filter=filter,
        )

    def _client(self) -> Any:
        return _open(self._conn_kwargs, self._secret("token"))

    def _identity_suffix(self) -> str:
        return str(self._conn_kwargs["collection"])

    def _infer_schema(self) -> pa.Schema:
        """The collection's fields, mapped onto Arrow, in the collection's order."""
        client = self._client()
        try:
            described = _describe_collection(client, self._conn_kwargs["collection"])
        finally:
            client.close()
        fields = []
        for f in described["fields"]:
            kind = _type_name(f)
            if kind == "FLOAT_VECTOR":
                fields.append(vector_field(f["name"], _dim(f)))
            elif kind in _SCALARS:
                fields.append(pa.field(f["name"], _SCALARS[kind]))
            else:
                raise PlanError(
                    f"milvus field {f['name']!r} is {kind}, which has no Arrow mapping here",
                    hint="Pass schema= to declare the columns to read.",
                )
        return pa.schema(fields)

    def _enumerate_partitions(self) -> list[str]:
        if self._conn_kwargs["partitions"]:
            return list(self._conn_kwargs["partitions"])
        client = self._client()
        try:
            return list(client.list_partitions(self._conn_kwargs["collection"]))
        finally:
            client.close()

    def _read_partition(
        self,
        partition: str,
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 - the engine's Filter re-checks
    ) -> Iterator[pa.RecordBatch]:
        schema = self.schema()
        fields = projection or schema.names
        client = self._client()
        try:
            rows = self._query(client, partition, fields)
            for batch in rows_to_batches(rows, schema=schema):
                yield batch.select(projection) if projection else batch
        finally:
            client.close()

    def _query(self, client: Any, partition: str, fields: list[str]) -> Iterator[dict[str, Any]]:
        """Every row of one partition, page by page."""
        iterator = client.query_iterator(
            self._conn_kwargs["collection"],
            batch_size=_PAGE_ROWS,
            filter=self._conn_kwargs["filter"],
            output_fields=fields,
            partition_names=[partition],
        )
        try:
            while page := iterator.next():
                yield from page
        finally:
            iterator.close()


@SINKS.register("milvus")
class MilvusSink(VectorSink):
    """Upsert, insert or delete rows in an existing Milvus collection.

    Not yet verified against a live Milvus; see tests/PENDING_VERIFICATION.md.

    Args:
        uri: A server URL, or a local Milvus Lite database file.
        token: ``"user:password"`` or an API key; never logged, resolved on the worker.
        db_name: The database, when not the default.
        partition: The partition to write into; the default partition otherwise.
        vector_field: The collection's float-vector field, when it has more than one.
        id_column: The id column, sent as the collection's primary key.
        vector_column: The vector column.
        dimension: The dimension every vector must have.
        metric: ``"cosine"``, ``"euclidean"`` (``"L2"``) or ``"dot"`` (``"IP"``); checked
            against the vector field's index.
        batch_size: Rows per request.
        max_retries: Retries per failed request; ``append`` is never retried.
        mode: ``"upsert"`` (default), ``"append"`` (an insert) or ``"delete"``.
    """

    format_name = "milvus"
    native_metrics: ClassVar[Mapping[str, str]] = {
        "cosine": "COSINE",
        "euclidean": "L2",
        "dot": "IP",
    }
    default_batch_size = 1_000
    supported_modes = ("upsert", "append", "delete")
    unretryable_modes = frozenset({"append"})

    __slots__ = ("_fields", "partition", "vector_field")

    def __init__(
        self,
        *,
        uri: str,
        token: str | None = None,
        db_name: str | None = None,
        partition: str | None = None,
        vector_field: str | None = None,
        id_column: str = DEFAULT_ID_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        dimension: int | None = None,
        metric: str | None = None,
        batch_size: int | None = None,
        max_retries: int = 3,
        mode: str = "upsert",
    ) -> None:
        super().__init__(
            id_column=id_column,
            vector_column=vector_column,
            dimension=dimension,
            metric=metric,
            batch_size=batch_size,
            max_retries=max_retries,
            mode=mode,
            uri=uri,
            token=token,
            db_name=db_name,
        )
        self.partition = partition
        self.vector_field = vector_field
        #: Frame column to collection field, resolved by `_describe` on each write.
        self._fields: dict[str, str] = {}

    def _client(self) -> Any:
        return _open(self._conn_kwargs, self._secret("token"))

    def _describe(self, client: Any, path: str, payload: list[str]) -> RemoteTarget:
        """Resolve the primary key and vector field, and check every payload column fits."""
        described = _describe_collection(client, path)
        fields = {f["name"]: f for f in described["fields"]}
        primary = next(name for name, f in fields.items() if f.get("is_primary"))
        vectors = [name for name, f in fields.items() if _type_name(f) == "FLOAT_VECTOR"]
        target = self.vector_field or (vectors[0] if len(vectors) == 1 else None)
        if target not in vectors:
            raise PlanError(
                f"milvus collection {path!r} has no single float-vector field to write "
                f"{self.vector_column!r} into",
                available=vectors,
                available_label="Vector fields",
                hint="Name one with vector_field=.",
            )
        self._fields = {self.id_column: primary, self.vector_column: target}
        if not described.get("enable_dynamic_field"):
            taken = {primary, target}
            for name in payload:
                if name not in fields or name in taken:
                    raise PlanError(
                        f"milvus collection {path!r} has no field {name!r} and no dynamic "
                        "field to hold it; nothing was written",
                        available=sorted(fields),
                        available_label="Fields",
                        hint="Drop or rename the column, or enable the dynamic field.",
                    )
        return RemoteTarget(
            dimensions={"": _dim(fields[target])}, metric=_index_metric(client, path, target)
        )

    def _send(self, client: Any, path: str, chunk: pa.Table) -> None:
        partition = {"partition_name": self.partition} if self.partition else {}
        if self.mode == "delete":
            client.delete(path, ids=chunk.column(self.id_column).to_pylist(), **partition)
            return
        rows = chunk.rename_columns(
            [self._fields.get(c, c) for c in chunk.column_names]
        ).to_pylist()
        if self.mode == "append":
            count = client.insert(path, data=rows, **partition).get("insert_count")
        else:
            count = client.upsert(path, data=rows, **partition).get("upsert_count")
        if count != len(rows):
            raise BackendError(
                f"milvus {self.mode} into {path!r} applied {count} of {len(rows)} rows"
            )


def _index_metric(client: Any, collection: str, field: str) -> str | None:
    """The metric of `field`'s index, or None when it has none or the server will not say."""
    try:
        names = client.list_indexes(collection, field_name=field)
        if not names:
            return None
        return str(client.describe_index(collection, names[0])["metric_type"])
    except Exception:  # an unindexed field has no metric to disagree with
        return None
