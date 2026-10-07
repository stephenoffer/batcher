"""In-memory stand-ins for the Qdrant, Pinecone, Milvus and Turbopuffer client libraries.

None of the four clients is installed on a test box, and none of the services runs in CI, so
each fake implements exactly the calls the connectors make, with the argument names the real
client documents, and records every request. A test then pins the request shape a connector
sends and drives the failure modes (a store that drops part of a batch, one that fails once
and then succeeds) that a live server would only produce by accident.

This proves the connector speaks the client API as documented; it does not prove a live
server accepts it. That is what `tests/integration/live/` is for.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace as NS
from typing import Any

__all__ = ["install_milvus", "install_pinecone", "install_qdrant", "install_turbopuffer"]


def _install(monkeypatch: Any, name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class Flaky:
    """Fails the first `failures` calls of a request kind, then delegates."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures

    def check(self) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("transient: connection reset")


# --- Qdrant ---------------------------------------------------------------------------


class QdrantStore:
    """One process-wide fake Qdrant: collections of ``{id: (vector, payload)}``."""

    def __init__(self, vectors: Any, distance: str = "Cosine") -> None:
        self.vectors_config = vectors
        self.distance = distance
        self.points: dict[Any, tuple[Any, dict]] = {}
        self.requests: list[tuple[str, Any]] = []
        self.clients: list[dict] = []
        self.flaky = Flaky()
        self.status = "completed"
        self.exists = True


def install_qdrant(
    monkeypatch: Any, *, size: int = 2, distance: str = "Cosine", named: dict | None = None
) -> QdrantStore:
    """Install a fake ``qdrant_client`` whose collection ``docs`` holds `size`-d vectors."""
    if named:
        vectors: Any = {n: NS(size=s, distance=NS(value=distance)) for n, s in named.items()}
    else:
        vectors = NS(size=size, distance=NS(value=distance))
    store = QdrantStore(vectors, distance)

    class QdrantClient:
        def __init__(self, **kwargs: Any) -> None:
            store.clients.append(kwargs)

        def collection_exists(self, collection_name: str) -> bool:
            return store.exists

        def get_collection(self, collection_name: str) -> Any:
            return NS(config=NS(params=NS(vectors=store.vectors_config)))

        def upsert(self, collection_name: str, points: list, wait: bool = True) -> Any:
            store.flaky.check()
            store.requests.append(("upsert", [(p.id, p.vector, dict(p.payload)) for p in points]))
            for p in points:
                store.points[p.id] = (p.vector, dict(p.payload))
            return NS(status=NS(value=store.status))

        def delete(self, collection_name: str, points_selector: Any, wait: bool = True) -> Any:
            store.requests.append(("delete", list(points_selector.points)))
            for pid in points_selector.points:
                store.points.pop(pid, None)
            return NS(status=NS(value=store.status))

        def scroll(
            self,
            collection_name: str,
            limit: int = 10,
            offset: Any = None,
            with_payload: bool = True,
            with_vectors: bool = False,
        ) -> tuple[list, Any]:
            store.requests.append(("scroll", (limit, offset, with_vectors)))
            ids = sorted(store.points, key=str)
            start = 0 if offset is None else ids.index(offset)
            page = ids[start : start + limit]
            following = ids[start + limit] if start + limit < len(ids) else None
            records = [
                NS(
                    id=pid,
                    payload=dict(store.points[pid][1]) if with_payload else None,
                    vector=store.points[pid][0] if with_vectors else None,
                )
                for pid in page
            ]
            return records, following

        def count(self, collection_name: str, exact: bool = True) -> Any:
            return NS(count=len(store.points))

        def close(self) -> None:
            pass

    models = NS(
        PointStruct=lambda **k: NS(**k),
        PointIdsList=lambda **k: NS(**k),
    )
    _install(monkeypatch, "qdrant_client", QdrantClient=QdrantClient, models=models)
    return store


# --- Pinecone -------------------------------------------------------------------------


class PineconeStore:
    def __init__(self, dimension: int, metric: str) -> None:
        self.dimension = dimension
        self.metric = metric
        self.records: dict[str, dict[str, dict]] = {}
        self.requests: list[tuple[str, Any]] = []
        self.api_keys: list[Any] = []
        self.drop_one = False
        self.exists = True


def install_pinecone(
    monkeypatch: Any, *, dimension: int = 2, metric: str = "cosine"
) -> PineconeStore:
    """Install a fake ``pinecone`` with one serverless index ``docs``."""
    store = PineconeStore(dimension, metric)

    class Index:
        def __init__(self, name: str) -> None:
            self.name = name

        def upsert(self, vectors: list[dict], namespace: str = "") -> Any:
            store.requests.append(("upsert", namespace, [dict(v) for v in vectors]))
            kept = vectors[:-1] if store.drop_one else vectors
            for v in kept:
                store.records.setdefault(namespace, {})[v["id"]] = v
            return NS(upserted_count=len(kept))

        def delete(self, ids: list[str], namespace: str = "") -> dict:
            store.requests.append(("delete", namespace, list(ids)))
            for i in ids:
                store.records.get(namespace, {}).pop(i, None)
            return {}

        def list(self, namespace: str = "") -> Any:
            ids = sorted(store.records.get(namespace, {}))
            for start in range(0, len(ids), 2):
                yield ids[start : start + 2]

        def fetch(self, ids: list[str], namespace: str = "") -> Any:
            store.requests.append(("fetch", namespace, list(ids)))
            held = store.records.get(namespace, {})
            return NS(
                vectors={
                    i: NS(id=i, values=held[i]["values"], metadata=held[i].get("metadata"))
                    for i in ids
                }
            )

    class Pinecone:
        def __init__(self, api_key: Any = None) -> None:
            store.api_keys.append(api_key)

        def describe_index(self, name: str) -> Any:
            if not store.exists:
                raise RuntimeError(f"(404) Reason: Not Found: index {name}")
            return NS(dimension=store.dimension, metric=store.metric, host="docs.svc")

        def Index(self, name: str = "", host: str = "") -> Index:
            return Index(name)

    _install(monkeypatch, "pinecone", Pinecone=Pinecone)
    return store


# --- Milvus ---------------------------------------------------------------------------


class MilvusStore:
    def __init__(self, fields: list[dict], dynamic: bool, metric: str | None) -> None:
        self.fields = fields
        self.dynamic = dynamic
        self.metric = metric
        self.rows: dict[str, dict[Any, dict]] = {"_default": {}, "p1": {}}
        self.requests: list[tuple[str, Any]] = []
        self.clients: list[dict] = []
        self.flaky = Flaky()


def install_milvus(
    monkeypatch: Any,
    *,
    dim: int = 2,
    dynamic: bool = False,
    metric: str | None = "COSINE",
    extra_fields: tuple[tuple[str, str], ...] = (("title", "VARCHAR"),),
) -> MilvusStore:
    """Install a fake ``pymilvus`` with a collection ``docs`` (pk ``pk``, vector ``vec``)."""
    dtype = NS(**{n: NS(name=n) for n in ("INT64", "VARCHAR", "FLOAT_VECTOR", "JSON", "DOUBLE")})
    fields = [
        {"name": "pk", "type": dtype.INT64, "params": {}, "is_primary": True},
        {"name": "vec", "type": dtype.FLOAT_VECTOR, "params": {"dim": dim}},
        *({"name": n, "type": getattr(dtype, t), "params": {}} for n, t in extra_fields),
    ]
    store = MilvusStore(fields, dynamic, metric)

    class Iterator:
        def __init__(self, rows: list[dict], batch_size: int) -> None:
            self.pages = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
            self.closed = False

        def next(self) -> list[dict]:
            return self.pages.pop(0) if self.pages else []

        def close(self) -> None:
            self.closed = True

    class MilvusClient:
        def __init__(self, **kwargs: Any) -> None:
            store.clients.append(kwargs)

        def has_collection(self, collection_name: str) -> bool:
            return collection_name == "docs"

        def describe_collection(self, collection_name: str) -> dict:
            return {"fields": store.fields, "enable_dynamic_field": store.dynamic}

        def list_indexes(self, collection_name: str, field_name: str = "") -> list[str]:
            return ["vec_idx"] if store.metric else []

        def describe_index(self, collection_name: str, index_name: str) -> dict:
            return {"metric_type": store.metric}

        def _write(self, kind: str, data: list[dict], partition_name: str | None) -> int:
            store.flaky.check()
            store.requests.append((kind, partition_name, [dict(r) for r in data]))
            for row in data:
                store.rows[partition_name or "_default"][row["pk"]] = dict(row)
            return len(data)

        def upsert(
            self, collection_name: str, data: list[dict], partition_name: str | None = None
        ) -> dict:
            return {"upsert_count": self._write("upsert", data, partition_name)}

        def insert(
            self, collection_name: str, data: list[dict], partition_name: str | None = None
        ) -> dict:
            return {"insert_count": self._write("insert", data, partition_name)}

        def delete(
            self, collection_name: str, ids: list, partition_name: str | None = None
        ) -> dict:
            store.requests.append(("delete", partition_name, list(ids)))
            for part in store.rows.values():
                for i in ids:
                    part.pop(i, None)
            return {"delete_count": len(ids)}

        def list_partitions(self, collection_name: str) -> list[str]:
            return list(store.rows)

        def query_iterator(
            self,
            collection_name: str,
            batch_size: int = 1000,
            filter: str = "",
            output_fields: list[str] | None = None,
            partition_names: list[str] | None = None,
        ) -> Iterator:
            store.requests.append(("query_iterator", partition_names, list(output_fields or [])))
            rows = [
                {k: v for k, v in r.items() if k in (output_fields or r)}
                for p in partition_names or list(store.rows)
                for r in store.rows[p].values()
            ]
            return Iterator(rows, batch_size)

        def close(self) -> None:
            pass

    _install(monkeypatch, "pymilvus", MilvusClient=MilvusClient, DataType=dtype)
    return store


# --- Turbopuffer ----------------------------------------------------------------------


class TurbopufferStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict[Any, dict]] = {}
        self.requests: list[tuple[str, str, Any]] = []
        self.clients: list[dict] = []
        self.flaky = Flaky()


def install_turbopuffer(monkeypatch: Any) -> TurbopufferStore:
    """Install a fake ``turbopuffer`` whose namespaces spring into being on first write."""
    store = TurbopufferStore()

    class NotFoundError(Exception):
        pass

    class Namespace:
        def __init__(self, name: str) -> None:
            self.name = name

        def write(
            self,
            upsert_columns: dict | None = None,
            deletes: list | None = None,
            distance_metric: str | None = None,
            schema: dict | None = None,
        ) -> Any:
            store.flaky.check()
            store.requests.append(
                (
                    "write",
                    self.name,
                    {
                        "upsert_columns": upsert_columns,
                        "deletes": deletes,
                        "distance_metric": distance_metric,
                        "schema": schema,
                    },
                )
            )
            docs = store.docs.setdefault(self.name, {})
            if upsert_columns:
                names = list(upsert_columns)
                for i, doc_id in enumerate(upsert_columns["id"]):
                    docs[doc_id] = {n: upsert_columns[n][i] for n in names}
            for doc_id in deletes or []:
                docs.pop(doc_id, None)
            return NS(rows_affected=len((upsert_columns or {}).get("id", deletes or [])))

        def schema(self) -> dict:
            if self.name not in store.docs:
                raise NotFoundError(f"namespace {self.name} not found")
            sample = next(iter(store.docs[self.name].values()))
            types_ = {}
            for key, value in sample.items():
                if key == "vector":
                    types_[key] = {"type": f"[{len(value)}]f32"}
                elif isinstance(value, bool):
                    types_[key] = {"type": "bool"}
                elif isinstance(value, int):
                    types_[key] = {"type": "int"}
                elif isinstance(value, float):
                    types_[key] = {"type": "float"}
                else:
                    types_[key] = {"type": "string"}
            return types_

        def query(
            self, rank_by: Any, top_k: int, include_attributes: Any, filters: Any = None
        ) -> Any:
            store.requests.append(
                (
                    "query",
                    self.name,
                    {"top_k": top_k, "filters": filters, "include_attributes": include_attributes},
                )
            )
            docs = sorted(store.docs.get(self.name, {}).values(), key=lambda d: d["id"])
            if filters is not None:
                docs = [d for d in docs if d["id"] > filters[2]]
            keep = set(include_attributes) | {"id"}
            return NS(rows=[{k: v for k, v in d.items() if k in keep} for d in docs[:top_k]])

    class Turbopuffer:
        def __init__(self, **kwargs: Any) -> None:
            store.clients.append(kwargs)

        def namespace(self, name: str) -> Namespace:
            return Namespace(name)

    _install(monkeypatch, "turbopuffer", Turbopuffer=Turbopuffer, NotFoundError=NotFoundError)
    return store
