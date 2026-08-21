"""Elasticsearch connector — ES|QL Arrow output with sliced-scroll splits.

Elasticsearch 8.18+ can return ES|QL query results in Arrow stream format
(``format="arrow"``), which `ElasticsearchSource` reads straight into Arrow with
no per-row Python. Parallel reads use Elasticsearch's *sliced scroll*: a search
declares ``slice = {id, max}`` and each slice scrolls a disjoint subset of the
matching documents — one `Split` per slice, a disjoint and exhaustive cover.

`ElasticsearchSink` is the write half, over the ``_bulk`` API: one request per batch,
with the per-item results inspected rather than assumed. ``_bulk`` returns HTTP 200
with an ``errors`` flag and a per-item status, so a caller that only checks the
response code has indexed *some* of its documents and been told the call succeeded.

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

from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import (
    BulkSink,
    PartitionSpec,
    ScanSource,
    require_driver,
    rows_to_batches,
)
from batcher.io.formats.sql._common import probe_is_typed
from batcher.io.predicate import combine_conjunction
from batcher.io.predicate._literals import _col_and_untyped_literal
from batcher.plan.ir_tags import COMPARISON_FLIP

__all__ = ["ElasticsearchSink", "ElasticsearchSource"]

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

    def _client(self) -> Any:
        es = require_driver("elasticsearch", "elasticsearch")
        kw = self._conn_kwargs
        return es.Elasticsearch(hosts=kw["hosts"], api_key=self._secret("api_key"))

    def row_count(self) -> int | None:
        """The exact matching-document count via the ``_count`` API — no scroll, no scan.

        ``_count`` runs the same DSL query the scroll path reads but returns only the
        cardinality, so it is a single cheap round trip that answers ``count()`` and sizes
        the estimator's cardinality exactly. The result is the count of the base relation
        (the source's own ``query``, before Kyber's pushed predicate narrows it further),
        which is what a `Scan` leaf's statistics describe.

        Only the search/scroll path answers here: an ES|QL result has no cheap count (it
        would need ``| STATS COUNT(*)``, a second full query), so it stays None. Best-effort
        — any cluster error yields None and the planner falls back to its default.
        """
        if self._conn_kwargs["esql"]:
            return None
        try:
            with contextlib.closing(self._client()) as client:
                resp = client.count(
                    index=self._conn_kwargs["index"], query=self._conn_kwargs["query"]
                )
            count = resp["count"] if isinstance(resp, dict) else resp.body["count"]
            return int(count)
        except Exception:
            return None

    def _identity_suffix(self) -> str:
        return str(self._conn_kwargs["index"])

    def _infer_schema(self) -> pa.Schema:
        """The relation's columns, from a one-row probe rather than the whole result.

        The ES|QL path ran the user's *entire* query and took `.schema` off the
        materialized table — the column names of a cluster-wide aggregation cost the
        cluster-wide aggregation. Worse, the plan needs the schema *before* it executes,
        so an ordinary ``read(...).collect()`` submitted the query twice.

        ``| LIMIT 1`` is the ES|QL spelling of the ``WHERE 1 = 0`` probe the SQL
        connectors use: the coordinator stops after one row, and the Arrow stream's IPC
        header carries the full typed schema regardless. `probe_is_typed` guards the case
        a probe comes back untyped, falling back to the full query — slow, which is merely
        what it did before, rather than wrong.
        """
        esql = self._conn_kwargs["esql"]
        with contextlib.closing(self._client()) as client:
            if esql:
                probed = _esql_schema(client, f"{esql} | LIMIT 1")
                return probed if probe_is_typed(probed) else _esql_schema(client, esql)
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
        self,
        partition: _Slice,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        self = self._with_pushed(predicate)  # push into the ES query, on the worker too
        # Resolved before the client is opened: `schema()` dials Elasticsearch itself, and
        # a raise from it between `_client()` and the `try` would strand a live connection
        # with nothing left holding a reference to close it.
        schema = self.schema() if not projection else None
        client = self._client()
        try:
            if self._conn_kwargs["esql"]:
                # Streamed off the IPC reader rather than `read_all()`-ed into a table
                # first: the whole point of `iter_batches` is that a caller which stops
                # early, or which only ever holds one batch, never pays for the rest.
                for batch in _esql_batches(client, self._conn_kwargs["esql"]):
                    yield batch.select(projection) if projection else batch
                return
            rows = _scroll_slice(
                client, self._conn_kwargs["index"], self._conn_kwargs["query"], partition
            )
            for batch in rows_to_batches(rows, schema=schema):
                yield batch.select(projection) if projection else batch
        finally:
            # A consumer that abandons this generator (a LIMIT, an exception downstream)
            # otherwise leaves the connection pool open until the collector happens to run.
            client.close()


# IR comparison op → Elasticsearch ``range`` query operator (``eq`` uses ``term``).
_ES_RANGE = {"lt": "lt", "le": "lte", "gt": "gt", "ge": "gte"}
# When a literal sits on the left, flip the comparison direction.


def _to_es_query(ir: dict[str, Any]) -> dict[str, Any] | None:
    """Translate the pushable subset of `ir` to an Elasticsearch query dict, or None.

    Handles column-vs-literal comparisons (``= != < <= > >=``), ``IS NULL`` /
    ``IS NOT NULL``, and ``AND`` / ``OR`` of pushable terms; anything else (e.g.
    column-vs-column) is not pushable.

    An ``AND`` with one untranslatable side keeps the side that did translate, because a
    conjunction narrowed by fewer terms is a *superset* of the right answer and the
    engine's `Filter` re-checks every row anyway. An ``OR`` in the same position declines
    entirely: dropping one branch of a disjunction would drop rows it matched. That
    asymmetry is `combine_conjunction`, shared with the SQL, Mongo and Iceberg
    translators — this connector used to require both sides, so a single untranslatable
    term (a column-vs-column comparison, or a temporal literal) sent the *whole* filter to
    the client and dragged the index across the network to discard it.
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
        clause = "must" if op == "and" else "should"
        bool_body: dict[str, Any] = {clause: [left, right]}
        if op == "or":
            bool_body["minimum_should_match"] = 1
        return combine_conjunction(op, left, right, {"bool": bool_body})
    parsed = _col_and_untyped_literal(ir.get("left", {}), ir.get("right", {}))
    if parsed is None:
        return None
    col, value, flipped = parsed
    effective = COMPARISON_FLIP[op] if flipped else op
    if effective == "eq":
        return {"term": {col: value}}
    if effective == "ne":
        return {"bool": {"must_not": {"term": {col: value}}}}
    if effective in _ES_RANGE:
        return {"range": {col: {_ES_RANGE[effective]: value}}}
    return None


def _esql_stream(client: Any, esql: str) -> Any:
    """Open the Arrow IPC stream for an ES|QL query without reading its batches."""
    resp = client.esql.query(query=esql, format="arrow")
    raw = resp.body if hasattr(resp, "body") else resp
    return pa.ipc.open_stream(io.BytesIO(raw))


def _esql_batches(client: Any, esql: str) -> Iterator[pa.RecordBatch]:
    """Yield an ES|QL result's Arrow batches one at a time."""
    with contextlib.closing(_esql_stream(client, esql)) as reader:
        yield from reader


def _esql_schema(client: Any, esql: str) -> pa.Schema:
    """An ES|QL result's schema, read from the IPC header without consuming a batch."""
    with contextlib.closing(_esql_stream(client, esql)) as reader:
        return reader.schema


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


#: Documents per ``_bulk`` request.
#:
#: Elasticsearch's own guidance is to size a bulk request by *bytes* (5-15 MB) rather than
#: by document count, and there is no way to know the byte size of a document before
#: rendering it. A count is the approximation every client library makes; 1,000 keeps a
#: request comfortably inside the default 100 MB ``http.max_content_length`` for any
#: document a relational frame produces.
_BULK_DOCUMENTS = 1_000


@SINKS.register("elasticsearch")
class ElasticsearchSink(BulkSink):
    """Index rows into an Elasticsearch index over the ``_bulk`` API.

    Every mode is one ``_bulk`` request per chunk, and every response is inspected per
    item: ``_bulk`` reports failures inside a 200 response, so trusting the status code
    means a partially-indexed batch reported as a success.

    ``upsert`` indexes each document under its ``_id``, replacing what was there.
    ``append`` indexes without an ``_id`` so the cluster assigns one, which is the only
    honest reading of "add these documents" in a store where an id collision is a replace.
    ``overwrite`` deletes every document in the index with ``delete_by_query`` before
    indexing, and is refused past the first shard of a distributed write.

    Args:
        hosts: One or more Elasticsearch URLs; never logged.
        index: The target index, when it is not the write's destination name.
        api_key: Optional API key; never logged, and an ``env:``/``file:`` reference is
            resolved where the connection is opened.
        key_field: The column supplying each document's ``_id``, for ``upsert`` and
            ``delete``. The column is kept in the document body as well, because dropping
            it would make a round trip lose it.
        refresh: Whether to make the write visible to search before returning. Off by
            default, as it is in Elasticsearch itself: forcing a refresh per batch is the
            standard way to make a bulk load an order of magnitude slower.
        mode: One of `STORE_WRITE_MODES`.
    """

    format_name = "elasticsearch"

    __slots__ = ("index", "refresh")

    def __init__(
        self,
        *,
        hosts: str | list[str] = "http://localhost:9200",
        index: str | None = None,
        api_key: str | None = None,
        key_field: str = "id",
        refresh: bool = False,
        mode: str = "upsert",
    ) -> None:
        super().__init__(key_field=key_field, mode=mode, hosts=hosts, api_key=api_key)
        self.index = index
        self.refresh = refresh

    def _client(self) -> Any:
        """An Elasticsearch client, built here so the credential is resolved on the worker."""
        es = require_driver("elasticsearch", "elasticsearch")
        return es.Elasticsearch(hosts=self._conn_kwargs["hosts"], api_key=self._secret("api_key"))

    def _actions(self, rows: list[dict[str, Any]], index: str) -> list[dict[str, Any]]:
        """The ``_bulk`` action/document lines `mode` turns `rows` into."""
        from batcher._internal.errors import BackendError

        lines: list[dict[str, Any]] = []
        for row in rows:
            if self.mode == "append":
                lines.extend(({"index": {"_index": index}}, row))
                continue
            if self.key_field not in row:
                raise BackendError(
                    f"mode={self.mode!r} needs a {self.key_field!r} column to address each "
                    f"document; this row has {sorted(row)}. Name it with key_field=."
                )
            addressed = {"_index": index, "_id": str(row[self.key_field])}
            if self.mode == "delete":
                lines.append({"delete": addressed})
            else:
                lines.extend(({"index": addressed}, row))
        return lines

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Clear the index if overwriting, then bulk-index in chunks."""
        client = self._client()
        index = self.index or path
        try:
            if self.mode == "overwrite":
                client.delete_by_query(
                    index=index, query={"match_all": {}}, refresh=True, ignore_unavailable=True
                )
            for start in range(0, len(rows), _BULK_DOCUMENTS):
                chunk = rows[start : start + _BULK_DOCUMENTS]
                _bulk(client, self._actions(chunk, index), index, refresh=self.refresh)
        finally:
            with contextlib.suppress(Exception):
                client.close()


def _bulk(client: Any, actions: list[dict[str, Any]], index: str, *, refresh: bool) -> None:
    """Send one ``_bulk`` request and raise on any per-item failure.

    Raises:
        BackendError: If the response's ``errors`` flag is set, naming how many items
            failed and what the first one said.
    """
    from batcher._internal.errors import BackendError

    if not actions:
        return
    response = client.bulk(operations=actions, refresh="true" if refresh else "false")
    if not response.get("errors"):
        return
    failures = [
        result
        for item in response.get("items", [])
        for result in item.values()
        if result.get("error")
    ]
    raise BackendError(
        f"elasticsearch bulk write to {index!r}: {len(failures)} of "
        f"{len(response.get('items', []))} operations failed; the first was "
        f"{failures[0].get('error') if failures else 'unreported'}"
    )
