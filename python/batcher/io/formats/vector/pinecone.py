"""Pinecone connector — list and fetch a namespace as Arrow, upsert and delete records.

`PineconeSink` checks the index with ``describe_index`` before the first upsert, so a frame
whose vectors have the wrong dimension, or a ``metric=`` that disagrees with the index, is
refused before anything is sent. Records go out in batches through ``Index.upsert`` into one
``namespace``, and the response's ``upserted_count`` is compared with the batch rather than
trusted. Pinecone ids are strings, so an integer id is sent as its decimal string.

Metadata is every column that is neither the id nor the vector. Pinecone metadata holds
strings, numbers, booleans and lists of strings, and no nulls: a column of another type is
refused before the write, and a null value is left out of that record's metadata, which is
how Pinecone represents an absent field.

`PineconeSource` reads a namespace by paging its ids with ``Index.list`` and fetching each
page with ``Index.fetch``. ``list`` is available on serverless indexes only, so a pod-based
index cannot be read this way.

The ``pinecone`` import is deferred to the worker; a missing client raises with the
``pinecone`` extra. Not yet verified against a live Pinecone; see
tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import ScanSource, rows_to_batches, schema_from_rows
from batcher.io.formats.vector.contract import (
    DEFAULT_ID_COLUMN,
    DEFAULT_VECTOR_COLUMN,
    RemoteTarget,
    VectorSink,
    vector_field,
)

__all__ = ["PineconeSink", "PineconeSource"]


def _pinecone() -> Any:
    """The ``pinecone`` module, or a typed install hint."""
    from batcher._internal.optional import require

    return require("pinecone", feature="Pinecone", provides="pinecone", extra="pinecone")


def _attr(obj: Any, name: str) -> Any:
    """`name` off a client response, which is a model object or a plain dict by version."""
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _metadata_type_ok(dtype: pa.DataType) -> bool:
    """Whether Pinecone metadata can hold a column of `dtype`."""
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.types.is_string(dtype.value_type) or pa.types.is_large_string(dtype.value_type)
    return (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_boolean(dtype)
        or pa.types.is_null(dtype)
    )


@dataclass
class _Connection:
    """A Pinecone client and the `Index` handles it has opened, one per index name."""

    client: Any
    indexes: dict[str, Any] = field(default_factory=dict)

    def index(self, name: str) -> Any:
        if name not in self.indexes:
            self.indexes[name] = self.client.Index(name=name)
        return self.indexes[name]


@SOURCES.register("pinecone")
class PineconeSource(ScanSource):
    """A Pinecone namespace, read by listing its ids and fetching them a page at a time.

    Not yet verified against a live Pinecone; see tests/PENDING_VERIFICATION.md.

    Args:
        index: The index to read.
        api_key: The API key; never logged, resolved on the worker.
        namespace: The namespace; ``""`` is the default namespace.
        id_column: The column the record id is read into.
        vector_column: The column the vector is read into.
        schema: A declared schema, which skips the sampling read.
    """

    format_name = "pinecone"

    __slots__ = ()

    def __init__(
        self,
        *,
        index: str,
        api_key: str | None = None,
        namespace: str = "",
        id_column: str = DEFAULT_ID_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        schema: pa.Schema | None = None,
    ) -> None:
        super().__init__(
            schema=schema,
            index=index,
            api_key=api_key,
            namespace=namespace,
            id_column=id_column,
            vector_column=vector_column,
        )

    def _client(self) -> Any:
        return _pinecone().Pinecone(api_key=self._secret("api_key"))

    def _identity_suffix(self) -> str:
        index, namespace = self._conn_kwargs["index"], self._conn_kwargs["namespace"]
        return f"{index}/{namespace}" if namespace else str(index)

    def _infer_schema(self) -> pa.Schema:
        """The id, the vector at the index's dimension, and metadata from the first page."""
        kw = self._conn_kwargs
        client = self._client()
        dimension = int(_attr(client.describe_index(kw["index"]), "dimension"))
        first = next(iter(self._pages(client.Index(name=kw["index"]))), [])
        metadata = schema_from_rows([self._row(r, values=False) for r in first])
        fields = [
            pa.field(kw["id_column"], pa.string()),
            vector_field(kw["vector_column"], dimension),
        ]
        taken = {f.name for f in fields}
        return pa.schema(fields + [f for f in metadata if f.name not in taken])

    def _pages(self, index: Any) -> Iterator[list[Any]]:
        """Each page of ids ``Index.list`` yields, fetched into its records."""
        namespace = self._conn_kwargs["namespace"]
        for ids in index.list(namespace=namespace):
            if ids:
                yield list(
                    _attr(index.fetch(ids=list(ids), namespace=namespace), "vectors").values()
                )

    def _row(self, record: Any, *, values: bool) -> dict[str, Any]:
        kw = self._conn_kwargs
        row = dict(_attr(record, "metadata") or {})
        row[kw["id_column"]] = _attr(record, "id")
        if values:
            row[kw["vector_column"]] = list(_attr(record, "values"))
        return row

    def _enumerate_partitions(self) -> list[None]:
        return [None]  # one id cursor per namespace

    def _read_partition(
        self,
        partition: None,  # noqa: ARG002 - a single partition
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 - the engine's Filter re-checks
    ) -> Iterator[pa.RecordBatch]:
        schema = self.schema()
        index = self._client().Index(name=self._conn_kwargs["index"])
        rows = (self._row(r, values=True) for page in self._pages(index) for r in page)
        for batch in rows_to_batches(rows, schema=schema):
            yield batch.select(projection) if projection else batch


@SINKS.register("pinecone")
class PineconeSink(VectorSink):
    """Upsert or delete records in one namespace of an existing Pinecone index.

    Not yet verified against a live Pinecone; see tests/PENDING_VERIFICATION.md.

    Args:
        api_key: The API key; never logged, resolved on the worker.
        namespace: The namespace to write into; ``""`` is the default namespace.
        id_column: The id column.
        vector_column: The vector column.
        dimension: The dimension every vector must have.
        metric: ``"cosine"``, ``"euclidean"``, ``"dot"`` (or ``"dotproduct"``); checked
            against the index.
        batch_size: Records per ``upsert`` request; Pinecone caps a request at 1,000
            records or 2 MB.
        max_retries: Retries per failed request.
        mode: ``"upsert"`` (default) or ``"delete"``.
    """

    format_name = "pinecone"
    native_metrics: ClassVar[Mapping[str, str]] = {
        "cosine": "cosine",
        "euclidean": "euclidean",
        "dot": "dotproduct",
    }
    default_batch_size = 100

    __slots__ = ("namespace",)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        namespace: str = "",
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
            api_key=api_key,
        )
        self.namespace = namespace

    def _client(self) -> _Connection:
        return _Connection(_pinecone().Pinecone(api_key=self._secret("api_key")))

    def _describe(self, client: _Connection, path: str, payload: list[str]) -> RemoteTarget:  # noqa: ARG002
        try:
            described = client.client.describe_index(path)
        except Exception as exc:
            raise PlanError(
                f"could not describe pinecone index {path!r}: {exc}",
                hint="Create the index, or check the name and the API key.",
            ) from exc
        return RemoteTarget(
            dimensions={"": int(_attr(described, "dimension"))},
            metric=str(_attr(described, "metric")),
        )

    def _payload_type_ok(self, dtype: pa.DataType) -> bool:
        return _metadata_type_ok(dtype)

    def _send(self, client: _Connection, path: str, chunk: pa.Table) -> None:
        index = client.index(path)
        ids = [str(i) for i in chunk.column(self.id_column).to_pylist()]
        if self.mode == "delete":
            index.delete(ids=ids, namespace=self.namespace)
            return
        values = chunk.column(self.vector_column).to_pylist()
        metadata = chunk.select(self._payload_columns(chunk)).to_pylist()
        records = []
        for rid, vec, meta in zip(ids, values, metadata, strict=True):
            record: dict[str, Any] = {"id": rid, "values": vec}
            present = {k: v for k, v in meta.items() if v is not None}
            if present:
                record["metadata"] = present
            records.append(record)
        response = index.upsert(vectors=records, namespace=self.namespace)
        count = _attr(response, "upserted_count")
        if count != len(records):
            raise BackendError(
                f"pinecone upsert into {path!r} applied {count} of {len(records)} records"
            )
