"""Turbopuffer connector — page a namespace out as Arrow, upsert and delete documents.

`TurbopufferSink` writes column-wise through ``Namespace.write(upsert_columns=...)``, one
request per batch, which Turbopuffer applies atomically. The id column is sent as Turbopuffer's
``id`` and the vector column as its ``vector``; every other column is an attribute.
``distance_metric`` is sent with every upsert, ``cosine_distance`` unless ``metric=`` says
otherwise, which is the default Ray Data's ``write_turbopuffer`` uses too. ``delete`` sends
``deletes=[ids]``. A namespace is created by its first write, so there may be nothing to check
a frame against; when the namespace exists and reports a vector type, its dimension is checked
before anything is sent.

`TurbopufferSource` reads a namespace by ranking on ``id`` ascending and paging with an
``id > last`` filter, the export pattern Turbopuffer documents. Columns come from
``Namespace.schema()``.

The ``turbopuffer`` import is deferred to the worker; a missing client raises with the
``turbopuffer`` extra. Not yet verified against a live Turbopuffer; see
tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import ScanSource, rows_to_batches
from batcher.io.formats.vector.contract import (
    DEFAULT_ID_COLUMN,
    DEFAULT_VECTOR_COLUMN,
    RemoteTarget,
    VectorSink,
    vector_field,
)

__all__ = ["TurbopufferSink", "TurbopufferSource"]

#: Documents per query page.
_PAGE_ROWS = 1_000

#: Turbopuffer attribute types and the Arrow type each is read as.
_TYPES = {
    "string": pa.string(),
    "int": pa.int64(),
    "uint": pa.uint64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "uuid": pa.string(),
    "datetime": pa.string(),
    "[]string": pa.list_(pa.string()),
    "[]int": pa.list_(pa.int64()),
    "[]uint": pa.list_(pa.uint64()),
    "[]float": pa.list_(pa.float64()),
    "[]bool": pa.list_(pa.bool_()),
}

#: A vector attribute's type, such as ``[1536]f32``.
_VECTOR_TYPE = re.compile(r"^\[(\d+)\]f(16|32)$")


def _turbopuffer() -> Any:
    """The ``turbopuffer`` module, or a typed install hint."""
    from batcher._internal.optional import require

    return require(
        "turbopuffer", feature="Turbopuffer", provides="turbopuffer", extra="turbopuffer"
    )


def _open(kwargs: dict[str, Any], api_key: Any) -> Any:
    """A `Turbopuffer` client for exactly one of ``region`` or ``base_url``."""
    if (kwargs.get("region") is None) == (kwargs.get("base_url") is None):
        raise PlanError("turbopuffer needs exactly one of region= or base_url=")
    where = {k: kwargs[k] for k in ("region", "base_url") if kwargs.get(k) is not None}
    return _turbopuffer().Turbopuffer(api_key=api_key, **where)


def _schema_types(namespace: Any) -> dict[str, str]:
    """Attribute name to its type string, from ``Namespace.schema()``."""
    described = namespace.schema()
    items = described.items() if isinstance(described, dict) else vars(described).items()
    out = {}
    for name, spec in items:
        kind = spec.get("type") if isinstance(spec, dict) else getattr(spec, "type", spec)
        out[str(name)] = str(kind)
    return out


def _row_dict(row: Any) -> dict[str, Any]:
    """A query result row as a dict; the client returns a model object or a dict."""
    if isinstance(row, dict):
        return dict(row)
    to_dict = getattr(row, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else dict(vars(row))


@SOURCES.register("turbopuffer")
class TurbopufferSource(ScanSource):
    """A Turbopuffer namespace, paged out in id order.

    Not yet verified against a live Turbopuffer; see tests/PENDING_VERIFICATION.md.

    Args:
        namespace: The namespace to read.
        region: The region, such as ``"gcp-us-central1"``; or pass `base_url`.
        base_url: The API base URL, instead of `region`.
        api_key: The API key; never logged, resolved on the worker.
        id_column: The column the document id is read into.
        vector_column: The column the vector is read into.
        schema: A declared schema, which skips asking the namespace for its own.
    """

    format_name = "turbopuffer"

    __slots__ = ()

    def __init__(
        self,
        *,
        namespace: str,
        region: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        id_column: str = DEFAULT_ID_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        schema: pa.Schema | None = None,
    ) -> None:
        super().__init__(
            schema=schema,
            namespace=namespace,
            region=region,
            base_url=base_url,
            api_key=api_key,
            id_column=id_column,
            vector_column=vector_column,
        )

    def _namespace(self) -> Any:
        return _open(self._conn_kwargs, self._secret("api_key")).namespace(
            self._conn_kwargs["namespace"]
        )

    def _identity_suffix(self) -> str:
        return str(self._conn_kwargs["namespace"])

    def _rename(self, name: str) -> str:
        kw = self._conn_kwargs
        return {"id": kw["id_column"], "vector": kw["vector_column"]}.get(name, name)

    def _infer_schema(self) -> pa.Schema:
        """Columns from the namespace's attribute schema, ``id`` first."""
        types = _schema_types(self._namespace())
        fields = [
            pa.field(
                self._conn_kwargs["id_column"], _TYPES.get(types.pop("id", "string"), pa.string())
            )
        ]
        for name, kind in types.items():
            vector = _VECTOR_TYPE.match(kind)
            if vector:
                fields.append(vector_field(self._rename(name), int(vector.group(1))))
            elif kind in _TYPES:
                fields.append(pa.field(self._rename(name), _TYPES[kind]))
            else:
                raise PlanError(
                    f"turbopuffer attribute {name!r} is {kind!r}, which has no Arrow mapping here",
                    hint="Pass schema= to declare the columns to read.",
                )
        return pa.schema(fields)

    def _enumerate_partitions(self) -> list[None]:
        return [None]  # one id-ordered cursor per namespace

    def _read_partition(
        self,
        partition: None,  # noqa: ARG002 - a single partition
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 - the engine's Filter re-checks
    ) -> Iterator[pa.RecordBatch]:
        schema = self.schema()
        rows = (
            {self._rename(k): v for k, v in _row_dict(row).items()}
            for row in self._pages(self._namespace(), schema)
        )
        for batch in rows_to_batches(rows, schema=schema):
            yield batch.select(projection) if projection else batch

    def _pages(self, namespace: Any, schema: pa.Schema) -> Iterator[Any]:
        """Every document, ranked by id and paged with an ``id > last`` filter."""
        renamed = {self._rename(n): n for n in ("id", "vector")}
        attributes = [renamed.get(n, n) for n in schema.names if renamed.get(n, n) != "id"]
        last = None
        while True:
            query: dict[str, Any] = {
                "rank_by": ("id", "asc"),
                "top_k": _PAGE_ROWS,
                "include_attributes": attributes,
            }
            if last is not None:
                query["filters"] = ("id", "Gt", last)
            rows = list(namespace.query(**query).rows or [])
            yield from rows
            if len(rows) < _PAGE_ROWS:
                return
            last = _row_dict(rows[-1])["id"]


@SINKS.register("turbopuffer")
class TurbopufferSink(VectorSink):
    """Upsert or delete documents in a Turbopuffer namespace.

    Not yet verified against a live Turbopuffer; see tests/PENDING_VERIFICATION.md.

    Args:
        region: The region, such as ``"gcp-us-central1"``; or pass `base_url`.
        base_url: The API base URL, instead of `region`.
        api_key: The API key; never logged, resolved on the worker.
        schema: A Turbopuffer attribute schema, forwarded with every write.
        id_column: The id column, sent as ``id``.
        vector_column: The vector column, sent as ``vector``.
        dimension: The dimension every vector must have.
        metric: ``"cosine"`` (``"cosine_distance"``, the default) or ``"euclidean"``
            (``"euclidean_squared"``).
        batch_size: Documents per write request.
        max_retries: Retries per failed request.
        mode: ``"upsert"`` (default) or ``"delete"``.
    """

    format_name = "turbopuffer"
    native_metrics: ClassVar[Mapping[str, str]] = {
        "cosine": "cosine_distance",
        "euclidean": "euclidean_squared",
    }
    default_batch_size = 10_000

    __slots__ = ("attribute_schema",)

    def __init__(
        self,
        *,
        region: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        schema: dict[str, Any] | None = None,
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
            region=region,
            base_url=base_url,
            api_key=api_key,
        )
        if (region is None) == (base_url is None):
            raise PlanError("turbopuffer needs exactly one of region= or base_url=")
        self.attribute_schema = schema

    def _client(self) -> Any:
        return _open(self._conn_kwargs, self._secret("api_key"))

    def _check_payload(self, table: pa.Table, path: str) -> None:
        """Refuse an attribute column named like Turbopuffer's reserved ``id`` or ``vector``."""
        super()._check_payload(table, path)
        for name in self._payload_columns(table):
            if name in ("id", "vector"):
                raise PlanError(
                    f"column {name!r} collides with turbopuffer's reserved {name!r} attribute "
                    f"in {path!r}; nothing was written",
                    hint="Rename it, or name it as id_column= / vector_column=.",
                )

    def _describe(self, client: Any, path: str, payload: list[str]) -> RemoteTarget | None:  # noqa: ARG002
        """The existing namespace's vector dimension, or None for a namespace not yet written.

        A namespace springs into existence on its first write, so a failed schema lookup is
        the ordinary case for a new one rather than an error; the write itself then reports
        anything genuinely wrong, per batch.
        """
        try:
            types = _schema_types(client.namespace(path))
        except Exception:  # an absent namespace: nothing to check the frame against
            return None
        vector = _VECTOR_TYPE.match(types.get("vector", ""))
        return RemoteTarget(dimensions={"": int(vector.group(1))} if vector else {})

    def _send(self, client: Any, path: str, chunk: pa.Table) -> None:
        namespace = client.namespace(path)
        ids = chunk.column(self.id_column).to_pylist()
        if self.mode == "delete":
            namespace.write(deletes=ids)
            return
        columns: dict[str, Any] = {
            "id": ids,
            "vector": chunk.column(self.vector_column).to_pylist(),
        }
        for name in self._payload_columns(chunk):
            columns[name] = chunk.column(name).to_pylist()
        request: dict[str, Any] = {
            "upsert_columns": columns,
            "distance_metric": self.metric or self.native_metrics["cosine"],
        }
        if self.attribute_schema is not None:
            request["schema"] = self.attribute_schema
        namespace.write(**request)
