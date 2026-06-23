"""Elasticsearch connector — ES|QL Arrow output with sliced-scroll splits.

Elasticsearch 8.18+ can return ES|QL query results in Arrow stream format
(``format="arrow"``), which `ElasticsearchSource` reads straight into Arrow with
no per-row Python. Parallel reads use Elasticsearch's *sliced scroll*: a search
declares ``slice = {id, max}`` and each slice scrolls a disjoint subset of the
matching documents — one `Split` per slice, a disjoint and exhaustive cover.

The ``elasticsearch`` import is deferred; a missing driver raises `BackendError`
with the ``elasticsearch`` extra hint. Connection kwargs (hosts, api_key) are
stored verbatim and never logged.
"""

from __future__ import annotations

import contextlib
import io
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

__all__ = ["ElasticsearchSource"]

# A slice locator: ``(slice_id, max_slices)`` for a sliced scroll.
_Slice = tuple[int, int]


@SOURCES.register("elasticsearch")
class ElasticsearchSource(ScanSource):
    """An Elasticsearch index read via ES|QL Arrow output (8.18+) or sliced scroll.

    With ``esql`` set, the whole result is fetched in one Arrow stream (best for
    aggregations / projections the cluster computes). Otherwise documents are read
    with sliced scroll, one slice per split, and assembled into Arrow rows.

    Args:
        hosts: One or more Elasticsearch URLs; never logged.
        index: The index (or pattern) to read.
        api_key: Optional API key for auth; never logged.
        esql: Optional ES|QL query string; when set, the Arrow-native path is used.
        query: Optional DSL query for the scroll path (default match-all).
        partition_spec: Optional parallelism hint; ``segments`` sets the slice
            count for the scroll path (default 1).
    """

    format_name = "elasticsearch"

    # Predicate pushdown: on the ES|QL path Kyber's pushed predicate becomes an
    # appended ``| WHERE`` clause; on the search/scroll path it becomes an ES bool
    # query AND-merged with the existing DSL query, so the cluster prunes before
    # returning rows. The engine's `Filter` re-check keeps a partial push correct.
    supports_predicate = True

    __slots__ = ()

    def __init__(
        self,
        *,
        hosts: str | list[str],
        index: str,
        api_key: str | None = None,
        esql: str | None = None,
        query: dict[str, Any] | None = None,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            hosts=hosts,
            index=index,
            api_key=api_key,
            esql=esql,
            query=query or {"match_all": {}},
        )

    def _with_pushed(self, predicate: dict | None) -> ElasticsearchSource:
        """A sibling source with the pushable predicate folded into its query.

        On the ES|QL path the predicate becomes an appended ``| WHERE`` clause; on
        the search/scroll path it becomes an ES ``bool`` query AND-merged with the
        existing DSL query. Returns ``self`` when `predicate` is absent or
        unpushable; the engine's `Filter` re-check keeps the result correct.
        """
        if predicate is None:
            return self
        kw = self._conn_kwargs
        esql, query = kw["esql"], kw["query"]
        if esql is not None:
            from batcher.io.predicate import to_sql_where

            where = to_sql_where(predicate)
            if where is None:
                return self
            esql = f"{esql} | WHERE {where}"
        else:
            pushed = _to_es_query(predicate)
            if pushed is None:
                return self
            query = {"bool": {"must": [query, pushed]}}
        return ElasticsearchSource(
            hosts=kw["hosts"],
            index=kw["index"],
            api_key=kw["api_key"],
            esql=esql,
            query=query,
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
        es = require_driver("elasticsearch", "elasticsearch")
        kw = self._conn_kwargs
        return es.Elasticsearch(hosts=kw["hosts"], api_key=kw["api_key"])

    def _identity_suffix(self) -> str:
        return str(self._conn_kwargs["index"])

    def _infer_schema(self) -> pa.Schema:
        if self._conn_kwargs["esql"]:
            client = self._client()
            table = _esql_arrow(client, self._conn_kwargs["esql"])
            return table.schema
        client = self._client()
        resp = client.search(
            index=self._conn_kwargs["index"],
            query=self._conn_kwargs["query"],
            size=1,
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return pa.schema([])
        return pa.RecordBatch.from_pylist([hits[0]["_source"]]).schema

    def _enumerate_partitions(self) -> list[_Slice]:
        if self._conn_kwargs["esql"]:
            return [(0, 1)]  # ES|QL fetches the whole result in one Arrow stream.
        segments = max(1, self._partition_spec.segments)
        return [(i, segments) for i in range(segments)]

    def _read_partition(
        self, partition: _Slice, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        client = self._client()
        if self._conn_kwargs["esql"]:
            table = _esql_arrow(client, self._conn_kwargs["esql"])
            yield from (table.select(projection) if projection else table).to_batches()
            return
        rows = _scroll_slice(
            client, self._conn_kwargs["index"], self._conn_kwargs["query"], partition
        )
        schema = self.schema() if not projection else None
        for batch in rows_to_batches(rows, schema=schema):
            yield batch.select(projection) if projection else batch


# IR comparison op → Elasticsearch ``range`` query operator (``eq`` uses ``term``).
_ES_RANGE = {"lt": "lt", "le": "lte", "gt": "gt", "ge": "gte"}
# When a literal sits on the left, flip the comparison direction.
_ES_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


def _to_es_query(ir: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the pushable subset of `ir` to an Elasticsearch query dict, or None.

    Handles column-vs-literal comparisons (``= != < <= > >=``), ``IS NULL`` /
    ``IS NOT NULL``, and ``AND`` / ``OR`` of pushable terms; anything else (e.g.
    column-vs-column) makes the whole expression unpushable and returns ``None``.
    """
    e = ir.get("e")
    if e == "is_null" and ir["input"].get("e") == "col":
        return {"bool": {"must_not": {"exists": {"field": ir["input"]["name"]}}}}
    if e == "is_not_null" and ir["input"].get("e") == "col":
        return {"exists": {"field": ir["input"]["name"]}}
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = _to_es_query(ir["left"])
        right = _to_es_query(ir["right"])
        if left is None or right is None:
            return None
        clause = "must" if op == "and" else "should"
        bool_body: dict[str, Any] = {clause: [left, right]}
        if op == "or":
            bool_body["minimum_should_match"] = 1
        return {"bool": bool_body}
    parsed = _col_and_literal(ir.get("left", {}), ir.get("right", {}))
    if parsed is None:
        return None
    col, value, flipped = parsed
    effective = _ES_FLIP[op] if flipped else op
    if effective == "eq":
        return {"term": {col: value}}
    if effective == "ne":
        return {"bool": {"must_not": {"term": {col: value}}}}
    if effective in _ES_RANGE:
        return {"range": {col: {_ES_RANGE[effective]: value}}}
    return None


def _col_and_literal(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, Any, bool] | None:
    """Return ``(column, value, flipped)`` for a column-vs-literal comparison."""
    if left.get("e") == "col" and right.get("e") == "lit":
        return left["name"], next(iter(right["value"].values())), False
    if left.get("e") == "lit" and right.get("e") == "col":
        return right["name"], next(iter(left["value"].values())), True
    return None


def _esql_arrow(client: Any, esql: str) -> pa.Table:
    """Run an ES|QL query asking for Arrow output and read it into a `pa.Table`."""
    resp = client.esql.query(query=esql, format="arrow")
    raw = resp.body if hasattr(resp, "body") else resp
    with pa.ipc.open_stream(io.BytesIO(raw)) as reader:
        return reader.read_all()


def _scroll_slice(
    client: Any, index: str, query: dict[str, Any], slice_loc: _Slice
) -> Iterator[dict[str, Any]]:
    """Scroll one slice of `index`, yielding each hit's ``_source`` document."""
    slice_id, slice_max = slice_loc
    body: dict[str, Any] = {"query": query}
    if slice_max > 1:
        body["slice"] = {"id": slice_id, "max": slice_max}
    resp = client.search(index=index, body=body, scroll="2m", size=1000)
    scroll_id = resp.get("_scroll_id")
    try:
        while True:
            hits = resp["hits"]["hits"]
            if not hits:
                return
            for hit in hits:
                yield hit["_source"]
            resp = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp.get("_scroll_id")
    finally:
        if scroll_id is not None:
            with contextlib.suppress(Exception):  # best-effort scroll cleanup
                client.clear_scroll(scroll_id=scroll_id)
