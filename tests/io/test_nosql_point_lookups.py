"""Reading one partition instead of the whole store, when the predicate proves it can.

A pushed predicate on a NoSQL store used to buy one thing: the server dropped
non-matching rows before returning them. On a partitioned store that is the *smaller* half
of the win and, on DynamoDB, not a win at all — a ``FilterExpression`` is applied after
items are read, and read capacity is billed for what was examined. So
``ds.filter(col("user_id") == "u-42")`` cost a full-table read to return one item.

Both stores have an operation that avoids it, and both need the same proof to use it: the
predicate must pin the partition key to a single value with a top-level ``AND`` term. Then
every matching row lives in one partition, and reading only that partition cannot drop one.
DynamoDB gets a ``Query``; Cassandra drops the token predicate and the 64-way fan-out.

What is asserted here is that proof, in both directions. The rewrite must fire on the
shapes where it is sound, and must **not** fire on the shapes where it is not — a top-level
``OR``, a range on the partition key, a composite key only partly pinned. Each of those
would return too few rows rather than an error, and nothing downstream could tell.

Two pushdown defects are pinned as well, both of which produced wrong results rather than
slow ones:

* Both connectors kept a private copy of `_col_and_literal` that skipped the temporal
  unwrapping the shared one does, so ``created > date(2024, 1, 1)`` was pushed as
  ``created > 19723``. On DynamoDB a number never compares equal to a stored string, so the
  server dropped every matching item; on Elasticsearch 19723 reads as epoch millis.
* Both required *both* sides of an ``AND`` to translate, so one untranslatable term sent
  the whole filter to the client — on DynamoDB, turning a narrow read into a full scan.
"""

from __future__ import annotations

import pickle

import pytest

from batcher.io.formats.nosql.cassandra import (
    CassandraSource,
    _PartitionRead,
    _single_partition_where,
)
from batcher.io.formats.nosql.dynamodb import (
    DynamoDBSource,
    _key_query,
    _KeyQuery,
    _to_dynamo_filter,
)
from batcher.io.formats.nosql.elasticsearch import _to_es_query
from batcher.io.predicate import conjuncts, pinned_columns

pytestmark = pytest.mark.io


def col(name: str) -> dict:
    return {"e": "col", "name": name}


def lit(value) -> dict:
    kind = "str" if isinstance(value, str) else "i64"
    return {"e": "lit", "value": {kind: value}}


def binop(op: str, left: dict, right: dict) -> dict:
    return {"e": "binary", "op": op, "left": left, "right": right}


DATE_LITERAL = {"e": "lit", "value": {"date": 19723}}
PK_EQ = binop("eq", col("user_id"), lit("u-42"))


# --- the shared proof -----------------------------------------------------------------


def test_only_an_and_chain_flattens() -> None:
    """An OR is one indivisible term: neither branch proves anything about the other."""
    assert len(conjuncts(binop("and", PK_EQ, binop("gt", col("ts"), lit(1))))) == 2
    assert len(conjuncts(binop("or", PK_EQ, binop("gt", col("ts"), lit(1))))) == 1


def test_only_a_top_level_equality_pins_a_column() -> None:
    assert pinned_columns(PK_EQ) == {"user_id"}
    assert pinned_columns(binop("gt", col("user_id"), lit("a"))) == set()
    assert pinned_columns(binop("or", PK_EQ, PK_EQ)) == set()


def test_an_equality_written_backwards_still_pins() -> None:
    """`'u-42' == col` is the same fact, and the SQL parser produces it."""
    assert pinned_columns(binop("eq", lit("u-42"), col("user_id"))) == {"user_id"}


# --- DynamoDB: Query instead of Scan --------------------------------------------------


def test_a_partition_key_equality_becomes_a_key_condition() -> None:
    query = _key_query(PK_EQ, "user_id", "ts")
    assert query is not None
    assert query.key_expression == "#n0 = :v0"
    assert query.values == {":v0": "u-42"}
    assert query.filter_expression is None


def test_a_sort_key_comparison_narrows_the_key_condition() -> None:
    predicate = binop("and", PK_EQ, binop("gt", col("ts"), lit(100)))
    query = _key_query(predicate, "user_id", "ts")
    assert query.key_expression == "#n0 = :v0 AND #n1 > :v1"


def test_a_sort_key_inequality_stays_a_filter() -> None:
    """DynamoDB rejects `<>` in a KeyConditionExpression; putting it there is a hard error."""
    predicate = binop("and", PK_EQ, binop("ne", col("ts"), lit(100)))
    query = _key_query(predicate, "user_id", "ts")
    assert query.key_expression == "#n0 = :v0"
    assert query.filter_expression is not None


def test_a_second_sort_key_condition_stays_a_filter() -> None:
    """One comparison per key condition; a second is a validation error from the service."""
    predicate = binop(
        "and",
        binop("and", PK_EQ, binop("gt", col("ts"), lit(1))),
        binop("lt", col("ts"), lit(9)),
    )
    query = _key_query(predicate, "user_id", "ts")
    assert query.key_expression.count("#n") == 2
    assert query.filter_expression is not None


def test_everything_else_becomes_a_filter_expression() -> None:
    predicate = binop("and", PK_EQ, binop("eq", col("status"), lit("open")))
    query = _key_query(predicate, "user_id", "ts")
    assert query.key_expression == "#n0 = :v0"
    assert query.filter_expression == "#n1 = :v1"
    assert query.values == {":v0": "u-42", ":v1": "open"}


@pytest.mark.parametrize(
    ("predicate", "why"),
    [
        (binop("gt", col("user_id"), lit("a")), "a range on the partition key names no partition"),
        (binop("or", PK_EQ, binop("eq", col("user_id"), lit("u-7"))), "an OR spans partitions"),
        (binop("eq", col("status"), lit("open")), "no partition-key term at all"),
    ],
)
def test_an_unprovable_shape_falls_back_to_the_scan(predicate, why) -> None:
    assert _key_query(predicate, "user_id", "ts") is None, why


def test_no_key_schema_means_no_query() -> None:
    """A role without `dynamodb:DescribeTable` reads exactly as it always did."""
    assert _key_query(PK_EQ, None, None) is None


class _FakeDynamoClient:
    def __init__(self, key_schema=None) -> None:
        self.operations: list[tuple[str, dict]] = []
        self._key_schema = key_schema or [
            {"KeyType": "HASH", "AttributeName": "user_id"},
            {"KeyType": "RANGE", "AttributeName": "ts"},
        ]

    def describe_table(self, TableName):
        return {"Table": {"KeySchema": self._key_schema}}

    def query(self, **kwargs):
        self.operations.append(("query", kwargs))
        return {"Items": [{"user_id": {"S": "u-42"}, "ts": {"N": "1"}}]}

    def scan(self, **kwargs):
        self.operations.append(("scan", kwargs))
        return {"Items": [{"user_id": {"S": "u-42"}, "ts": {"N": "1"}}]}


@pytest.fixture
def dynamo(monkeypatch):
    client = _FakeDynamoClient()
    monkeypatch.setattr(DynamoDBSource, "_client", lambda self: client)
    return client


def test_a_pinned_read_is_one_split_not_a_parallel_scan(dynamo) -> None:
    source = DynamoDBSource(table="events", region_name="us-east-1")
    splits = source.splits(predicate=PK_EQ)
    assert len(splits) == 1
    assert isinstance(splits[0].partition, _KeyQuery)


def test_the_query_split_is_picklable_like_every_other_locator(dynamo) -> None:
    source = DynamoDBSource(table="events", region_name="us-east-1")
    restored = pickle.loads(pickle.dumps(source.splits(predicate=PK_EQ)[0]))
    assert isinstance(restored.partition, _KeyQuery)


def test_the_split_issues_a_query_and_not_a_scan(dynamo) -> None:
    source = DynamoDBSource(table="events", region_name="us-east-1")
    split = source.splits(predicate=PK_EQ)[0]
    list(split.iter_batches(None, PK_EQ))
    issued = [name for name, _ in dynamo.operations if name == "query"]
    assert issued == ["query"]
    (_name, kwargs) = next(entry for entry in dynamo.operations if entry[0] == "query")
    assert kwargs["KeyConditionExpression"] == "#n0 = :v0"
    assert kwargs["ExpressionAttributeValues"] == {":v0": {"S": "u-42"}}


def test_declared_keys_skip_the_describe_table_round_trip() -> None:
    source = DynamoDBSource(table="events", region_name="us-east-1", partition_key="pk")
    assert source.key_schema() == ("pk", None)


def test_an_unpinned_read_still_fans_out(dynamo) -> None:
    source = DynamoDBSource(table="events", region_name="us-east-1")
    splits = source.splits(predicate=binop("eq", col("status"), lit("open")))
    assert all(isinstance(split.partition, tuple) for split in splits)


# --- Cassandra: one partition instead of the ring -------------------------------------


def test_a_pinned_partition_key_drops_the_token_predicate() -> None:
    assert _single_partition_where(PK_EQ, ("user_id",)) == "user_id = 'u-42'"


def test_a_composite_key_must_be_pinned_whole() -> None:
    """Cassandra hashes the entire partition key, so half of one names no partition."""
    assert _single_partition_where(PK_EQ, ("user_id", "tenant")) is None
    both = binop("and", PK_EQ, binop("eq", col("tenant"), lit("t1")))
    assert _single_partition_where(both, ("user_id", "tenant")) is not None


def test_a_range_on_the_partition_key_still_scans_the_ring() -> None:
    assert _single_partition_where(binop("gt", col("user_id"), lit("a")), ("user_id",)) is None


def test_the_ring_scan_is_what_happens_without_a_predicate() -> None:
    source = CassandraSource(contact_points=["h"], keyspace="k", table="t", partition_key="user_id")
    assert len(source.splits()) == 64


def test_a_pinned_read_is_one_split(monkeypatch) -> None:
    source = CassandraSource(contact_points=["h"], keyspace="k", table="t", partition_key="user_id")
    splits = source.splits(predicate=PK_EQ)
    assert len(splits) == 1
    assert isinstance(splits[0].partition, _PartitionRead)
    assert isinstance(pickle.loads(pickle.dumps(splits[0])).partition, _PartitionRead)


def test_the_partition_read_statement_has_no_token_clause(monkeypatch) -> None:
    executed: list[str] = []

    class _Session:
        def execute(self, statement):
            executed.append(statement)
            return []

    class _Cluster:
        def shutdown(self) -> None:
            pass

    source = CassandraSource(
        contact_points=["h"], keyspace="k", table="events", partition_key="user_id"
    )
    monkeypatch.setattr(CassandraSource, "_session", lambda self: (_Cluster(), _Session()))
    monkeypatch.setattr(CassandraSource, "schema", lambda self: None)
    list(source._read_partition(_PartitionRead("user_id = 'u-42'"), None))
    assert executed == ["SELECT * FROM events WHERE user_id = 'u-42' ALLOW FILTERING"]
    assert "token(" not in executed[0]


# --- the two pushdown defects ---------------------------------------------------------


def test_a_temporal_literal_is_not_pushed_to_a_store_with_no_date_type() -> None:
    """Pushing the raw epoch offset made the *server* drop rows that matched."""
    predicate = binop("gt", col("created"), DATE_LITERAL)
    assert _to_dynamo_filter(predicate) is None
    assert _to_es_query(predicate) is None


def test_an_and_keeps_the_half_that_translates() -> None:
    predicate = binop(
        "and", binop("eq", col("status"), lit("open")), binop("gt", col("c"), DATE_LITERAL)
    )
    pushed = _to_dynamo_filter(predicate)
    assert pushed is not None and pushed.values == {":v0": "open"}
    assert _to_es_query(predicate) == {"term": {"status": "open"}}


def test_an_or_with_an_untranslatable_branch_pushes_nothing() -> None:
    """Dropping one branch of a disjunction would drop the rows it matched."""
    predicate = binop(
        "or", binop("eq", col("status"), lit("open")), binop("gt", col("c"), DATE_LITERAL)
    )
    assert _to_dynamo_filter(predicate) is None
    assert _to_es_query(predicate) is None
