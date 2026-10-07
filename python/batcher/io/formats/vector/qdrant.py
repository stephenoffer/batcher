"""Qdrant connector — scroll a collection out as Arrow, upsert and delete points into it.

`QdrantSource` reads a collection with Qdrant's ``scroll`` API, one page of points at a time,
as one row per point: the point id, one ``fixed_size_list<float32>`` column per vector (an
unnamed vector is the ``embedding`` column, a named one is a column of its own name), and one
column per payload key. Scroll is a single cursor with no parallel form, so the read is one
split.

`QdrantSink` writes through ``upsert`` and ``delete`` with ``wait=True``, so a batch that
returns has been applied. Qdrant ids are unsigned integers or UUIDs; a string id that is
neither is mapped to a UUID by `stable_point_id`, the same UUID on every run, and the original
string is kept in the payload under the id column's name, which is where `QdrantSource` reads
it back from. Named vectors are written with ``vectors={"name": "column", ...}``.

The ``qdrant-client`` import is deferred to the worker; a missing client raises with the
``qdrant`` extra. Not yet verified against a live Qdrant; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import SCHEMA_SAMPLE_ROWS, ScanSource, rows_to_batches
from batcher.io.formats.nosql.base import schema_from_rows as _schema_from_rows
from batcher.io.formats.vector.contract import (
    DEFAULT_ID_COLUMN,
    DEFAULT_VECTOR_COLUMN,
    RemoteTarget,
    VectorSink,
    stable_point_id,
    vector_field,
)

__all__ = ["QdrantSink", "QdrantSource"]

#: Points per scroll page.
_PAGE_POINTS = 256


def _qdrant() -> Any:
    """The ``qdrant_client`` module, or a typed install hint."""
    from batcher._internal.optional import require

    return require("qdrant_client", feature="Qdrant", provides="qdrant-client", extra="qdrant")


def _open(kwargs: dict[str, Any], api_key: Any) -> Any:
    """A `QdrantClient` from the connection keywords (``location``/``url``/``path``)."""
    params = {k: kwargs[k] for k in ("location", "url", "path") if kwargs.get(k) is not None}
    return _qdrant().QdrantClient(api_key=api_key, **params)


def _vector_sizes(client: Any, collection: str) -> dict[str, tuple[int, str]]:
    """Vector name ("" when unnamed) to ``(size, distance)`` for an existing collection.

    Raises:
        PlanError: If the collection does not exist.
    """
    if not client.collection_exists(collection):
        raise PlanError(
            f"qdrant collection {collection!r} does not exist",
            hint="Create it with its vector size and distance before writing or reading.",
        )
    vectors = client.get_collection(collection).config.params.vectors
    named = vectors if isinstance(vectors, dict) else {"": vectors}
    return {
        name: (int(p.size), str(getattr(p.distance, "value", p.distance)))
        for name, p in named.items()
    }


@SOURCES.register("qdrant")
class QdrantSource(ScanSource):
    """A Qdrant collection, scrolled out as one row per point.

    Not yet verified against a live Qdrant; see tests/PENDING_VERIFICATION.md.

    Args:
        collection: The collection to read.
        url: The server URL, such as ``"http://localhost:6333"``.
        location: ``":memory:"`` for the client's in-process mode, or a URL.
        path: A directory for the client's on-disk local mode.
        api_key: The API key; never logged, and an ``env:``/``file:`` reference is resolved
            on the worker.
        id_column: The column the point id is read into.
        vector_column: The column an unnamed vector is read into.
        with_vectors: Whether to read vectors at all; off reads ids and payload only.
        schema: A declared schema, which skips the sampling read.
    """

    format_name = "qdrant"

    __slots__ = ()

    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        location: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        id_column: str = DEFAULT_ID_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        with_vectors: bool = True,
        schema: pa.Schema | None = None,
    ) -> None:
        super().__init__(
            schema=schema,
            collection=collection,
            url=url,
            location=location,
            path=path,
            api_key=api_key,
            id_column=id_column,
            vector_column=vector_column,
            with_vectors=with_vectors,
        )

    def _client(self) -> Any:
        return _open(self._conn_kwargs, self._secret("api_key"))

    def _identity_suffix(self) -> str:
        return str(self._conn_kwargs["collection"])

    def row_count(self) -> int | None:
        """The exact point count from Qdrant's ``count`` API; None if the call fails."""
        try:
            client = self._client()
            try:
                return int(client.count(self._conn_kwargs["collection"], exact=True).count)
            finally:
                client.close()
        except Exception:
            return None

    def _column_of(self, name: str) -> str:
        return name or self._conn_kwargs["vector_column"]

    def _infer_schema(self) -> pa.Schema:
        """The id, a column per vector from the collection config, and the sampled payload."""
        kw = self._conn_kwargs
        client = self._client()
        try:
            sizes = _vector_sizes(client, kw["collection"])
            points, _ = client.scroll(
                kw["collection"], limit=SCHEMA_SAMPLE_ROWS, with_payload=True, with_vectors=False
            )
        finally:
            client.close()
        rows = [self._row(p, vectors=False) for p in points]
        id_column = kw["id_column"]
        ids = [r[id_column] for r in rows]
        id_type = pa.int64() if ids and all(isinstance(i, int) for i in ids) else pa.string()
        payload = _schema_from_rows([{k: v for k, v in r.items() if k != id_column} for r in rows])
        vectors = [vector_field(self._column_of(n), s) for n, (s, _) in sizes.items()]
        fields = [pa.field(id_column, id_type), *(vectors if kw["with_vectors"] else [])]
        taken = {f.name for f in fields}
        return pa.schema(fields + [f for f in payload if f.name not in taken])

    def _row(self, point: Any, *, vectors: bool) -> dict[str, Any]:
        """One point as a row; a payload key named like the id column holds the original id."""
        id_column = self._conn_kwargs["id_column"]
        row: dict[str, Any] = {id_column: point.id, **(point.payload or {})}
        if vectors and point.vector is not None:
            named = point.vector if isinstance(point.vector, dict) else {"": point.vector}
            row.update({self._column_of(n): v for n, v in named.items()})
        return row

    def _enumerate_partitions(self) -> list[None]:
        return [None]  # scroll is one cursor: there is no disjoint parallel form

    def _read_partition(
        self,
        partition: None,  # noqa: ARG002 - a single partition
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 - the engine's Filter re-checks
    ) -> Iterator[pa.RecordBatch]:
        schema = self.schema()
        kw = self._conn_kwargs
        with_vectors = kw["with_vectors"] and (
            projection is None
            or any(pa.types.is_fixed_size_list(schema.field(c).type) for c in projection)
        )
        client = self._client()
        try:
            for batch in rows_to_batches(self._scroll(client, with_vectors), schema=schema):
                yield batch.select(projection) if projection else batch
        finally:
            client.close()

    def _scroll(self, client: Any, with_vectors: bool) -> Iterator[dict[str, Any]]:
        """Every point in the collection, page by page, as rows."""
        offset = None
        while True:
            points, offset = client.scroll(
                self._conn_kwargs["collection"],
                limit=_PAGE_POINTS,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            for point in points:
                yield self._row(point, vectors=with_vectors)
            if offset is None:
                return


@SINKS.register("qdrant")
class QdrantSink(VectorSink):
    """Upsert or delete points in an existing Qdrant collection.

    Not yet verified against a live Qdrant; see tests/PENDING_VERIFICATION.md.

    Args:
        url: The server URL.
        location: ``":memory:"`` or a URL.
        path: A directory for the client's on-disk local mode.
        api_key: The API key; never logged, resolved on the worker.
        vector_name: The collection's vector name, for a collection with named vectors.
        vectors: Several named vectors at once, as ``{vector name: column}``.
        id_column: The id column.
        vector_column: The vector column, when `vectors` is not given.
        dimension: The dimension every vector must have.
        metric: ``"cosine"``, ``"euclidean"``, ``"dot"``, or Qdrant's own spelling; checked
            against the collection.
        batch_size: Points per ``upsert`` request.
        max_retries: Retries per failed request.
        mode: ``"upsert"`` (default) or ``"delete"``.
    """

    format_name = "qdrant"
    native_metrics: ClassVar[Mapping[str, str]] = {
        "cosine": "Cosine",
        "euclidean": "Euclid",
        "dot": "Dot",
        "manhattan": "Manhattan",
    }
    default_batch_size = 256

    __slots__ = ("named_vectors",)

    def __init__(
        self,
        *,
        url: str | None = None,
        location: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        vector_name: str | None = None,
        vectors: dict[str, str] | None = None,
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
            url=url,
            location=location,
            path=path,
            api_key=api_key,
        )
        if vectors is not None and vector_name is not None:
            raise PlanError("pass vector_name= or vectors=, not both")
        self.named_vectors = dict(vectors) if vectors else {vector_name or "": vector_column}

    def vector_columns(self) -> dict[str, str]:
        return self.named_vectors

    def _client(self) -> Any:
        return _open(self._conn_kwargs, self._secret("api_key"))

    def _describe(self, client: Any, path: str, payload: list[str]) -> RemoteTarget:  # noqa: ARG002
        sizes = _vector_sizes(client, path)
        missing = sorted(set(self.named_vectors) - set(sizes))
        if missing:
            raise PlanError(
                f"qdrant collection {path!r} has no vector named {missing[0]!r}",
                available=[n or "<unnamed>" for n in sizes],
                available_label="Vectors",
            )
        metrics = {sizes[name][1] for name in self.named_vectors}
        return RemoteTarget(
            dimensions={name: sizes[name][0] for name in self.named_vectors},
            metric=metrics.pop() if len(metrics) == 1 else None,
        )

    def _send(self, client: Any, path: str, chunk: pa.Table) -> None:
        models = _qdrant().models
        raw_ids = chunk.column(self.id_column).to_pylist()
        ids = [stable_point_id(i) for i in raw_ids]
        if self.mode == "delete":
            result = client.delete(path, points_selector=models.PointIdsList(points=ids), wait=True)
        else:
            payloads = chunk.select(self._payload_columns(chunk)).to_pylist()
            columns = {
                name: chunk.column(col).to_pylist() for name, col in self.named_vectors.items()
            }
            points = []
            for row, (raw, pid) in enumerate(zip(raw_ids, ids, strict=True)):
                payload = payloads[row]
                if raw != pid:  # a mapped string id: keep the caller's id beside the UUID
                    payload[self.id_column] = raw
                vector = (
                    columns[""][row]
                    if list(columns) == [""]
                    else {name: values[row] for name, values in columns.items()}
                )
                points.append(models.PointStruct(id=pid, vector=vector, payload=payload))
            result = client.upsert(path, points=points, wait=True)
        status = str(getattr(result.status, "value", result.status)).lower()
        if status != "completed":
            raise BackendError(f"qdrant {self.mode} into {path!r} returned status {status!r}")
