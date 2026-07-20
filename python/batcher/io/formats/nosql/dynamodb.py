"""DynamoDB connector — native parallel scan to Arrow.

DynamoDB's `Scan` API has first-class parallelism: a scan declares ``TotalSegments``
and each call scans one ``Segment``. `DynamoDBSource` maps that directly onto
splits — one `Split` per segment — so the segments are a disjoint, exhaustive
cover with no client-side range math. Each segment paginates through its items
(following ``LastEvaluatedKey``) and assembles them into Arrow at batch
granularity via `rows_to_batches`.

The ``boto3`` import is deferred; a missing driver raises `BackendError` with the
``dynamodb`` extra hint. Connection kwargs (region, credentials, endpoint) are
stored verbatim and never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.formats.nosql.base import (
    PartitionSpec,
    ScanSource,
    require_driver,
    rows_to_batches,
)
from batcher.io.formats.sql._common import connection_fingerprint

__all__ = ["DynamoDBSource"]

# A segment locator: ``(segment_index, total_segments)`` for a parallel Scan.
_Segment = tuple[int, int]


@SOURCES.register("dynamodb")
class DynamoDBSource(ScanSource):
    """A DynamoDB table read via native parallel `Scan`.

    Args:
        table: The table name.
        region_name: The AWS region; never logged.
        aws_access_key_id: Optional explicit access key; never logged.
        aws_secret_access_key: Optional explicit secret key; never logged.
        endpoint_url: Optional override (e.g. for DynamoDB Local); never logged.
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
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            table=table,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
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
        """``<endpoint>:<region>/<table>`` — the endpoint is part of the relation's identity.

        The region was already in the key, which hides how much this matters: `endpoint_url`
        is what distinguishes DynamoDB Local from the real service, and one AWS account's
        table from another's at the same region and name. Both are the *same* relation under
        a region-only key, so a laptop's ten-item fixture table taught Kyber the cardinality
        it then used to plan the production table — and `identity()` is persisted, so that
        estimate outlives the process that made it.

        `aws_secret_access_key` and `aws_access_key_id` are excluded rather than passed to
        `connection_fingerprint`: its `_NON_IDENTIFYING` filter matches on key *name* and
        does not cover the ``aws_``-prefixed spellings, so handing it the full kwargs would
        digest the live credentials instead of dropping them. Excluding them also keeps a
        key rotation from orphaning the table's statistics.
        """
        kw = self._conn_kwargs
        region = kw["region_name"] or "default"
        fingerprint = connection_fingerprint(
            {"region_name": region, "endpoint_url": kw["endpoint_url"] or "aws"}
        )
        return f"{fingerprint}:{region}/{kw['table']}"

    def _infer_schema(self) -> pa.Schema:
        client = self._client()
        resp = client.scan(TableName=self._conn_kwargs["table"], Limit=1)
        items = [_deserialize(item) for item in resp.get("Items", [])]
        if not items:
            return pa.schema([])
        return pa.RecordBatch.from_pylist(items).schema

    def _enumerate_partitions(self) -> list[_Segment]:
        total = max(1, self._partition_spec.segments)
        return [(i, total) for i in range(total)]

    def _read_partition(
        self,
        partition: _Segment,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        segment, total = partition
        client = self._client()
        kwargs: dict[str, Any] = {"TableName": self._conn_kwargs["table"]}
        if total > 1:
            kwargs["Segment"] = segment
            kwargs["TotalSegments"] = total
        names: dict[str, str] = {}
        if projection:
            names = {f"#c{i}": col for i, col in enumerate(projection)}
            kwargs["ProjectionExpression"] = ", ".join(names)
        pushed = _to_dynamo_filter(predicate) if predicate is not None else None
        if pushed is not None:
            kwargs["FilterExpression"] = pushed.expression
            names.update(pushed.names)
            kwargs["ExpressionAttributeValues"] = {
                k: _serialize(v) for k, v in pushed.values.items()
            }
        if names:
            kwargs["ExpressionAttributeNames"] = names
        schema = self.schema() if not projection else None
        yield from rows_to_batches(_scan_items(client, kwargs), schema=schema)


# IR comparison op → DynamoDB ``FilterExpression`` comparator. ``eq`` maps to
# DynamoDB's ``=``; ``ne`` to ``<>``. When a literal sits on the left, flip.
_DYNAMO_OP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_DYNAMO_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


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
    ``AND`` / ``OR`` of pushable terms. Column-vs-column comparisons and anything
    else make the whole expression unpushable and return ``None``.
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
        if left is None or right is None:
            return None
        return f"({left} {op.upper()} {right})"
    if op not in _DYNAMO_OP:
        return None
    parsed = _col_and_literal(ir.get("left", {}), ir.get("right", {}))
    if parsed is None:
        return None
    col, value, flipped = parsed
    effective = _DYNAMO_FLIP[op] if flipped else op
    return f"{builder.name(col)} {_DYNAMO_OP[effective]} {builder.value(value)}"


def _col_and_literal(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, Any, bool] | None:
    """Return ``(column, value, flipped)`` for a column-vs-literal comparison."""
    if left.get("e") == "col" and right.get("e") == "lit":
        return left["name"], next(iter(right["value"].values())), False
    if left.get("e") == "lit" and right.get("e") == "col":
        return right["name"], next(iter(left["value"].values())), True
    return None


def _serialize(value: Any) -> dict[str, Any]:
    """Encode a Python literal to a DynamoDB low-level value ``{type: value}``."""
    from boto3.dynamodb.types import TypeSerializer

    return TypeSerializer().serialize(value)


def _scan_items(client: Any, kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Paginate a `Scan`, following `LastEvaluatedKey`, yielding decoded items.

    Genuinely incremental: one page is held at a time and `LastEvaluatedKey` drives the
    next request, so a scan of a table larger than memory streams rather than accumulating.
    """
    deserializer = _type_deserializer()
    while True:
        resp = client.scan(**kwargs)
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
