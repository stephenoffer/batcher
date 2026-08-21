"""DynamoDB connector — native parallel scan to Arrow.

DynamoDB's `Scan` API has first-class parallelism: a scan declares ``TotalSegments``
and each call scans one ``Segment``. `DynamoDBSource` maps that directly onto
splits — one `Split` per segment — so the segments are a disjoint, exhaustive
cover with no client-side range math. Each segment paginates through its items
(following ``LastEvaluatedKey``) and assembles them into Arrow at batch
granularity via `rows_to_batches`.

## Why a scan is not always the right call

A ``FilterExpression`` is applied *after* items are read, so a filtered scan costs the
same read capacity as an unfiltered one: DynamoDB bills for what it examined, not for
what it returned. On a table of any size, ``ds.filter(col("user_id") == "u-42")`` over a
scan is a full-table read to return one item, which is the difference between an
operational store and an expensive one.

``Query`` is the operation that avoids it. Given an equality on the table's **partition
key**, DynamoDB goes straight to that partition and reads only what is there, and a
comparison on the sort key narrows it further. So when the pushed predicate carries a
partition-key equality, this source issues one ``Query`` instead of N parallel ``Scan``
segments, and whatever else the predicate says becomes a ``FilterExpression`` on top.

That rewrite is only sound because the predicate reaching a scan is one the engine's
`Filter` also applies: if it says ``pk = 'u-42'``, then every row the query could have
returned lives in that one partition, so reading only that partition cannot drop a
matching row. Anything the source cannot prove that of — a top-level ``OR``, an inequality
on the partition key, a key schema it could not read — falls back to the scan it always did.

`DynamoDBSink` is the write half. DynamoDB's bulk primitive is
``BatchWriteItem``, which takes at most 25 requests per call and returns the ones it
could not process rather than raising, so the sink chunks at 25 and retries the
unprocessed remainder with backoff. Anything else is a silent partial write.

The ``boto3`` import is deferred; a missing driver raises `BackendError` with the
``dynamodb`` extra hint. Connection kwargs (region, credentials, endpoint) are
stored verbatim and never logged.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import (
    BulkSink,
    PartitionSpec,
    ScanSource,
    _ScanSplit,
    require_driver,
    rows_to_batches,
)
from batcher.io.predicate import combine_conjunction, conjuncts
from batcher.io.predicate._literals import _col_and_untyped_literal
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.source_stats import SourceStatistics

__all__ = ["DynamoDBSink", "DynamoDBSource"]

# A segment locator: ``(segment_index, total_segments)`` for a parallel Scan.
_Segment = tuple[int, int]

#: Comparators DynamoDB accepts inside a ``KeyConditionExpression`` on the **sort** key.
#:
#: Narrower than `_DYNAMO_OP` on purpose: the partition key admits only ``=``, and the sort
#: key admits every ordering comparison but not ``<>``. A ``<>`` on the sort key is a filter,
#: not a key condition, and putting it in the key condition is a hard validation error from
#: the service rather than a slow query.
_KEY_CONDITION_OP = {"eq": "=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


@SOURCES.register("dynamodb")
class DynamoDBSource(ScanSource):
    """A DynamoDB table read via native parallel `Scan`.

    Args:
        table: The table name.
        region_name: The AWS region; never logged.
        aws_access_key_id: Optional explicit access key; never logged.
        aws_secret_access_key: Optional explicit secret key; never logged.
        endpoint_url: Optional override (e.g. for DynamoDB Local); never logged.
        partition_key: The table's partition-key attribute. Stating it lets a predicate
            that pins it become a ``Query`` without a ``DescribeTable`` round trip, and
            works for a role that has no ``dynamodb:DescribeTable`` permission.
        sort_key: The table's sort-key attribute, when it has one.
        partition_spec: Optional parallelism hint; ``segments`` becomes the scan's
            ``TotalSegments`` (default 1 = a single sequential scan).
    """

    format_name = "dynamodb"

    # Predicate pushdown: Kyber's pushed predicate → a DynamoDB ``FilterExpression``
    # (plus its name/value maps) passed to ``Scan``, so the server drops
    # non-matching items before returning them. The engine's `Filter` re-check
    # keeps a partial or skipped push correct.
    supports_predicate = True

    __slots__ = ()

    def __init__(
        self,
        *,
        table: str,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        partition_key: str | None = None,
        sort_key: str | None = None,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            table=table,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            partition_key=partition_key,
            sort_key=sort_key,
        )

    def _client(self) -> Any:
        boto3 = require_driver("boto3", "dynamodb")
        kw = self._conn_kwargs
        return boto3.client(
            "dynamodb",
            region_name=kw["region_name"],
            aws_access_key_id=kw["aws_access_key_id"],
            aws_secret_access_key=self._secret("aws_secret_access_key"),
            endpoint_url=kw["endpoint_url"],
        )

    def _identity_suffix(self) -> str:
        region = self._conn_kwargs["region_name"] or "default"
        return f"{region}/{self._conn_kwargs['table']}"

    def _fingerprint_material(self) -> dict[str, Any]:
        """`_conn_kwargs` with the AWS keys dropped before the connection is fingerprinted.

        `connection_fingerprint` drops credentials by exact key name — ``password``,
        ``secret``, ``token``, ``api_key`` — and AWS spells its own with a prefix.
        ``aws_secret_access_key`` is not the string ``secret``, so it matched nothing and
        the live secret key was digested straight into `identity()`, the **persisted**
        learned-statistics key.

        One-way, so not a plaintext leak, but every key rotation re-keys the relation and
        silently orphans everything Kyber has learned about the table. What is left is the
        genuinely identifying part: `endpoint_url` is what separates DynamoDB Local (and one
        account's table) from the real service at the same region and name, so a laptop's
        ten-item fixture no longer teaches the optimizer a cardinality it applies to
        production.
        """
        return {k: v for k, v in self._conn_kwargs.items() if not k.startswith("aws_")}

    def statistics(self) -> SourceStatistics | None:
        """Advisory item count and on-disk byte size from ``DescribeTable`` — no scan.

        DynamoDB maintains ``ItemCount`` and ``TableSizeBytes`` on the table and returns
        both from a single ``DescribeTable`` metadata call, so the planner gets a
        cardinality and a size for free instead of paying read-capacity to scan the table
        to learn them. Both are updated roughly every six hours, so they are **estimates**:
        `exact_rows=False`, so the figure sizes joins and the worker fan-out but never
        answers an exact ``count()``. Best-effort — any AWS error yields None.
        """
        try:
            client = self._client()
            table = client.describe_table(TableName=self._conn_kwargs["table"])["Table"]
        except Exception:
            return None
        rows = table.get("ItemCount")
        size = table.get("TableSizeBytes")
        if rows is None and not size:
            return None
        return SourceStatistics(
            row_count=int(rows) if rows is not None else None,
            byte_size=int(size) if size else None,
            exact_rows=False,  # DescribeTable's counters refresh only ~every six hours
        )

    def _infer_schema(self) -> pa.Schema:
        client = self._client()
        resp = client.scan(TableName=self._conn_kwargs["table"], Limit=1)
        items = [_deserialize(item) for item in resp.get("Items", [])]
        if not items:
            return pa.schema([])
        return pa.RecordBatch.from_pylist(items).schema

    def key_schema(self) -> tuple[str | None, str | None]:
        """The table's ``(partition_key, sort_key)``, from the constructor or DescribeTable.

        Stating the keys with `partition_key=`/`sort_key=` skips the metadata call entirely,
        which matters for a role granted ``dynamodb:Query`` and ``dynamodb:Scan`` but not
        ``dynamodb:DescribeTable`` — a common least-privilege split. Without them this asks
        the service, and a failure is not an error: it means the query rewrite is unavailable
        and the read is the scan it always was.

        Returns:
            The partition-key and sort-key attribute names; either may be None.
        """
        declared = (self._conn_kwargs.get("partition_key"), self._conn_kwargs.get("sort_key"))
        if declared[0] is not None:
            return declared
        try:
            described = self._client().describe_table(TableName=self._conn_kwargs["table"])
            schema = described["Table"]["KeySchema"]
        except Exception as exc:
            note_suppressed("io", "describe dynamodb key schema", exc)
            return (None, None)
        by_type = {entry["KeyType"]: entry["AttributeName"] for entry in schema}
        return (by_type.get("HASH"), by_type.get("RANGE"))

    def splits(
        self,
        target_size: int | None = None,
        predicate: dict | None = None,
    ) -> list[Any]:
        """One ``Query`` split when the predicate pins the partition key; else the scan segments.

        A parallel scan is the right shape for reading a table. It is the wrong shape for
        reading *one item*, and a ``FilterExpression`` cannot fix that, because DynamoDB
        bills read capacity for every item it examines rather than for the ones it returns.

        Args:
            target_size: Ignored; a store splits by its own partitions.
            predicate: The predicate Kyber pushed to this scan.

        Returns:
            The splits to read, which is a single query split when one is provable.
        """
        query = _key_query(predicate, *self.key_schema()) if predicate is not None else None
        if query is None:
            return super().splits(target_size, predicate)
        return [
            _ScanSplit(
                source_cls=type(self),
                conn_kwargs=dict(self._conn_kwargs),
                partition=query,
                identity_prefix=self.identity(),
                predicate=predicate,
            )
        ]

    def _enumerate_partitions(self) -> list[_Segment]:
        total = max(1, self._partition_spec.segments)
        return [(i, total) for i in range(total)]

    def _read_partition(
        self,
        partition: _Segment | _KeyQuery,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        client = self._client()
        kwargs: dict[str, Any] = {"TableName": self._conn_kwargs["table"]}
        names: dict[str, str] = {}
        values: dict[str, Any] = {}
        if isinstance(partition, _KeyQuery):
            operation = client.query
            kwargs["KeyConditionExpression"] = partition.key_expression
            names.update(partition.names)
            values.update(partition.values)
            if partition.filter_expression:
                kwargs["FilterExpression"] = partition.filter_expression
        else:
            operation = client.scan
            segment, total = partition
            if total > 1:
                kwargs["Segment"] = segment
                kwargs["TotalSegments"] = total
            pushed = _to_dynamo_filter(predicate) if predicate is not None else None
            if pushed is not None:
                kwargs["FilterExpression"] = pushed.expression
                names.update(pushed.names)
                values.update(pushed.values)
        if projection:
            projected = {f"#c{i}": col for i, col in enumerate(projection)}
            kwargs["ProjectionExpression"] = ", ".join(projected)
            names.update(projected)
        if values:
            kwargs["ExpressionAttributeValues"] = {k: _serialize(v) for k, v in values.items()}
        if names:
            kwargs["ExpressionAttributeNames"] = names
        schema = self.schema() if not projection else None
        yield from rows_to_batches(_paginate(operation, kwargs), schema=schema)


@dataclass(frozen=True, slots=True)
class _KeyQuery:
    """A single-partition ``Query``: the key condition, plus whatever filter is left over.

    Picklable by construction — it holds placeholder maps of plain Python values, not
    serialized DynamoDB types — because it travels inside a `_ScanSplit` to a worker that
    rebuilds the connector from scratch.
    """

    key_expression: str
    names: dict[str, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    filter_expression: str | None = None


def _comparison(term: dict[str, Any]) -> tuple[str, str, Any] | None:
    """``(column, op, value)`` for a column-vs-literal comparison term, else None."""
    if term.get("e") != "binary" or term.get("op") not in _DYNAMO_OP:
        return None
    parsed = _col_and_untyped_literal(term.get("left", {}), term.get("right", {}))
    if parsed is None:
        return None
    col, value, flipped = parsed
    return col, (COMPARISON_FLIP[term["op"]] if flipped else term["op"]), value


def _key_query(
    predicate: dict[str, Any], partition_key: str | None, sort_key: str | None
) -> _KeyQuery | None:
    """A `_KeyQuery` when `predicate` pins `partition_key`, else None.

    The rewrite is sound only when the resulting read cannot miss a matching row. That
    holds exactly when the predicate has a top-level conjunct ``partition_key = <literal>``:
    every row satisfying the predicate then lives in that one partition, so a ``Query`` over
    it sees everything a ``Scan`` would have. Every other conjunct that translates becomes a
    ``FilterExpression``, and one that does not is simply left to the engine's own `Filter`,
    which re-checks every row regardless.

    Args:
        predicate: The pushed predicate, in the engine's JSON IR.
        partition_key: The table's partition-key attribute, or None if unknown.
        sort_key: The table's sort-key attribute, or None.

    Returns:
        The query to issue, or None to fall back to a scan.
    """
    if not partition_key:
        return None
    terms = conjuncts(predicate)
    builder = _DynamoBuilder()
    key_parts: list[str] = []
    leftover: list[dict[str, Any]] = []
    pinned = False
    sort_bounded = False
    for term in terms:
        parsed = _comparison(term)
        if parsed is None:
            leftover.append(term)
            continue
        column, op, value = parsed
        if column == partition_key and op == "eq" and not pinned:
            key_parts.append(f"{builder.name(column)} = {builder.value(value)}")
            pinned = True
        elif column == sort_key and op in _KEY_CONDITION_OP and not sort_bounded:
            # One sort-key condition only. DynamoDB accepts a single comparison (or a
            # BETWEEN), so a second would be a validation error; it stays a filter instead,
            # which is correct and merely reads a little more of the partition.
            key_parts.append(
                f"{builder.name(column)} {_KEY_CONDITION_OP[op]} {builder.value(value)}"
            )
            sort_bounded = True
        else:
            leftover.append(term)
    if not pinned:
        return None
    filter_expression = None
    if leftover:
        parts = [_build_dynamo(term, builder) for term in leftover]
        kept = [part for part in parts if part is not None]
        # A term that will not translate is dropped from the filter, never from the key
        # condition: the engine's `Filter` re-checks every row, so an unpushed term costs
        # bandwidth rather than correctness.
        filter_expression = " AND ".join(kept) if kept else None
    return _KeyQuery(
        key_expression=" AND ".join(key_parts),
        names=dict(builder.names),
        values=dict(builder.values),
        filter_expression=filter_expression,
    )


# IR comparison op → DynamoDB ``FilterExpression`` comparator. ``eq`` maps to
# DynamoDB's ``=``; ``ne`` to ``<>``. When a literal sits on the left, flip.
_DYNAMO_OP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


@dataclass(frozen=True, slots=True)
class _DynamoFilter:
    """A translated DynamoDB ``Scan`` filter: expression + name/value maps.

    `expression` is the ``FilterExpression`` string referencing ``#n*`` name
    placeholders and ``:v*`` value placeholders; `names` maps each placeholder to
    its real attribute name (handling reserved words / dotted paths) and `values`
    maps each value placeholder to its plain-Python literal.
    """

    expression: str
    names: dict[str, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)


class _DynamoBuilder:
    """Accumulates ``#n*``/``:v*`` placeholders while building a FilterExpression."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.values: dict[str, Any] = {}

    def name(self, col: str) -> str:
        placeholder = f"#n{len(self.names)}"
        self.names[placeholder] = col
        return placeholder

    def value(self, val: Any) -> str:
        placeholder = f":v{len(self.values)}"
        self.values[placeholder] = val
        return placeholder


def _to_dynamo_filter(ir: dict[str, Any]) -> _DynamoFilter | None:
    """Translate the pushable subset of `ir` to a `_DynamoFilter`, or None.

    Handles column-vs-literal comparisons (``= <> < <= > >=``), ``IS NULL`` /
    ``IS NOT NULL`` (via ``attribute_exists`` / ``attribute_not_exists``), and
    ``AND`` / ``OR`` of pushable terms.

    An ``AND`` with one untranslatable side keeps the side that did translate; an ``OR``
    in the same position declines entirely. See `combine_conjunction`, shared with the SQL,
    Mongo and Iceberg translators. This matters more on DynamoDB than anywhere else on that
    list: an unfiltered scan is billed for every item it examines *and* transfers all of
    them, so one untranslatable term used to turn a narrow read into a full table read.
    """
    builder = _DynamoBuilder()
    expr = _build_dynamo(ir, builder)
    if expr is None:
        return None
    return _DynamoFilter(expression=expr, names=builder.names, values=builder.values)


def _build_dynamo(ir: dict[str, Any], builder: _DynamoBuilder) -> str | None:
    """Build one FilterExpression sub-clause, registering placeholders, or None."""
    e = ir.get("e")
    if e == "is_null" and ir["input"].get("e") == "col":
        return f"attribute_not_exists({builder.name(ir['input']['name'])})"
    if e == "is_not_null" and ir["input"].get("e") == "col":
        return f"attribute_exists({builder.name(ir['input']['name'])})"
    if e != "binary":
        return None
    op = ir["op"]
    if op in ("and", "or"):
        left = _build_dynamo(ir["left"], builder)
        right = _build_dynamo(ir["right"], builder)
        return combine_conjunction(op, left, right, f"({left} {op.upper()} {right})")
    if op not in _DYNAMO_OP:
        return None
    parsed = _col_and_untyped_literal(ir.get("left", {}), ir.get("right", {}))
    if parsed is None:
        return None
    col, value, flipped = parsed
    effective = COMPARISON_FLIP[op] if flipped else op
    return f"{builder.name(col)} {_DYNAMO_OP[effective]} {builder.value(value)}"


def _serialize(value: Any) -> dict[str, Any]:
    """Encode a Python literal to a DynamoDB low-level value ``{type: value}``."""
    from boto3.dynamodb.types import TypeSerializer

    return TypeSerializer().serialize(value)


def _paginate(operation: Any, kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Paginate a `Scan` or a `Query`, following `LastEvaluatedKey`, yielding decoded items.

    Genuinely incremental: one page is held at a time and `LastEvaluatedKey` drives the
    next request, so a read of a table larger than memory streams rather than accumulating.
    Both operations paginate identically, which is why one function serves them: `operation`
    is the bound ``client.scan`` or ``client.query``.
    """
    deserializer = _type_deserializer()
    while True:
        resp = operation(**kwargs)
        for item in resp.get("Items", []):
            yield _deserialize(item, deserializer)
        last = resp.get("LastEvaluatedKey")
        if not last:
            return
        kwargs["ExclusiveStartKey"] = last


@lru_cache(maxsize=1)
def _type_deserializer() -> Any:
    """The one `TypeDeserializer` every decode shares.

    It is stateless, so the per-item construction it replaces bought nothing — but it was
    paid for on **every row of every scan**, along with an `import boto3.dynamodb.types`
    lookup, in the one loop the whole connector's throughput runs through.
    """
    from boto3.dynamodb.types import TypeDeserializer

    return TypeDeserializer()


def _deserialize(item: dict[str, Any], deserializer: Any = None) -> dict[str, Any]:
    """Decode a DynamoDB low-level item ``{attr: {type: value}}`` to plain Python."""
    deserializer = deserializer if deserializer is not None else _type_deserializer()
    return {k: _to_py(deserializer.deserialize(v)) for k, v in item.items()}


def _to_py(value: Any) -> Any:
    """Coerce boto3's `Decimal`/`set`/`Binary` types to Arrow-friendly Python."""
    from decimal import Decimal

    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if isinstance(value, set):
        return sorted(_to_py(v) for v in value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return value


#: Requests one ``BatchWriteItem`` call accepts. A hard AWS limit, not a tuning knob.
_BATCH_WRITE_LIMIT = 25

#: Attempts at the unprocessed remainder of a batch before giving up.
#:
#: ``BatchWriteItem`` does not raise when it is throttled: it returns the requests it did
#: not apply under ``UnprocessedItems`` and a 200 status. A caller that ignores that field
#: has written *some* of its rows and been told the call succeeded, which is the quietest
#: possible data-loss bug — so the remainder is retried, and a remainder that survives every
#: attempt raises rather than being dropped.
_UNPROCESSED_ATTEMPTS = 8


@SINKS.register("dynamodb")
class DynamoDBSink(BulkSink):
    """Write rows into a DynamoDB table with ``BatchWriteItem``.

    Two of the four store modes are declined rather than approximated, and both refusals
    are about DynamoDB's semantics rather than an unfinished implementation.

    ``append`` cannot be honest here: ``PutItem`` *replaces* the item holding the same
    primary key, so an "append" would silently be an upsert. A true insert-only append
    needs a conditional write per item (``attribute_not_exists(pk)``), which
    ``BatchWriteItem`` does not support at all — it takes no condition expression.

    ``overwrite`` would mean scanning the table to delete every item, at full read *and*
    write cost, to reach a state the caller can get atomically and for free by writing to
    a new table and re-pointing an alias. Offering it as a keyword would make an expensive
    and surprising operation look like a mode flag.

    Args:
        table: The target table, when it is not the write's destination name.
        region_name: The AWS region; never logged.
        aws_access_key_id: Optional explicit access key; never logged.
        aws_secret_access_key: Optional explicit secret key; never logged.
        endpoint_url: Optional override (e.g. for DynamoDB Local); never logged.
        key_field: Unused for `upsert`, which writes the whole item; for `delete` it is
            joined with `sort_key_field` to build each item's key.
        sort_key_field: The table's sort key, when it has one and `mode` is ``"delete"``.
        mode: ``"upsert"`` (default) or ``"delete"``.
    """

    format_name = "dynamodb"
    supported_modes = ("upsert", "delete")

    __slots__ = ("sort_key_field", "table")

    def __init__(
        self,
        *,
        table: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
        key_field: str = "id",
        sort_key_field: str | None = None,
        mode: str = "upsert",
    ) -> None:
        super().__init__(
            key_field=key_field,
            mode=mode,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
        )
        self.table = table
        self.sort_key_field = sort_key_field

    def _client(self) -> Any:
        """A boto3 client, built here so the credential is resolved on the worker."""
        boto3 = require_driver("boto3", "dynamodb")
        kw = self._conn_kwargs
        return boto3.client(
            "dynamodb",
            region_name=kw["region_name"],
            aws_access_key_id=kw["aws_access_key_id"],
            aws_secret_access_key=self._secret("aws_secret_access_key"),
            endpoint_url=kw["endpoint_url"],
        )

    def _request(self, row: dict[str, Any]) -> dict[str, Any]:
        """One ``BatchWriteItem`` request for `row`, per `mode`."""
        if self.mode == "delete":
            fields = [self.key_field, *([self.sort_key_field] if self.sort_key_field else [])]
            key = {f: _serialize(row[f]) for f in fields if f in row}
            if len(key) != len(fields):
                from batcher._internal.errors import BackendError

                raise BackendError(
                    f"mode='delete' needs every key field {fields} on each row; this row "
                    f"has {sorted(row)}. Name the table's keys with key_field= and, for a "
                    "composite key, sort_key_field=."
                )
            return {"DeleteRequest": {"Key": key}}
        item = {k: _serialize(v) for k, v in row.items() if v is not None}
        return {"PutRequest": {"Item": item}}

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Write `rows` in chunks of 25, retrying whatever the service did not process."""
        client = self._client()
        target = self.table or path
        requests = [self._request(row) for row in rows]
        for start in range(0, len(requests), _BATCH_WRITE_LIMIT):
            _write_batch(client, target, requests[start : start + _BATCH_WRITE_LIMIT])


def _write_batch(client: Any, table: str, requests: list[dict[str, Any]]) -> None:
    """Send one ``BatchWriteItem``, retrying the unprocessed remainder with backoff.

    Raises:
        BackendError: If a remainder survives `_UNPROCESSED_ATTEMPTS` rounds, which means
            the table is throttling harder than a retry loop can absorb.
    """
    from batcher._internal.errors import BackendError
    from batcher.io.base._transient import with_retry

    pending = requests
    for attempt in range(_UNPROCESSED_ATTEMPTS):
        # `batch=pending` binds this round's remainder as a default argument. A lambda
        # closing over `pending` would read whatever the *next* round reassigned it to.
        response = with_retry(
            lambda batch=pending: client.batch_write_item(RequestItems={table: batch}),
            attempts=3,
            backoff_base_s=0.2,
        )
        pending = response.get("UnprocessedItems", {}).get(table) or []
        if not pending:
            return
        # Equal-jitter backoff, the same shape the object-store retry uses: throttling is
        # what produced the remainder, so retrying it on the same tick reproduces it.
        ceiling = 0.1 * (2**attempt)
        time.sleep(ceiling / 2 + random.uniform(0.0, ceiling / 2))
    raise BackendError(
        f"dynamodb write to {table!r} left {len(pending)} item(s) unprocessed after "
        f"{_UNPROCESSED_ATTEMPTS} attempts. The table is throttling: raise its write "
        "capacity, or slow the write down with a smaller batch size."
    )
