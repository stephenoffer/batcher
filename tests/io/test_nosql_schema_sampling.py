"""An operational-store schema is inferred from a sample, or declared and never inferred.

Every row-based connector used to type the relation from its *first* row or document. A
field only later documents carry was then absent from the schema, and a field whose first
value was null was typed ``null``. Inference now unions a bounded sample
(`SCHEMA_SAMPLE_ROWS`) column by column, refuses a column whose values no one Arrow type
holds, and a source's ``schema=`` skips inference entirely -- including on the worker,
which rebuilds its source from the split.
"""

from __future__ import annotations

import pickle
from typing import Any, ClassVar

import pyarrow as pa
import pytest

from batcher._internal.errors import FormatError
from batcher.io.formats.nosql.base import SCHEMA_SAMPLE_ROWS, schema_from_rows
from batcher.io.formats.nosql.dynamodb import DynamoDBSource
from batcher.io.formats.nosql.elasticsearch import ElasticsearchSource

pytestmark = pytest.mark.io


class TestSchemaFromRows:
    def test_a_field_absent_from_the_first_row_is_kept(self):
        schema = schema_from_rows([{"a": 1}, {"a": 2, "late": "x"}])
        assert schema.names == ["a", "late"]
        assert schema.field("late").type == pa.string()

    def test_a_null_first_value_does_not_fix_the_type(self):
        assert schema_from_rows([{"a": None}, {"a": 3}]).field("a").type == pa.int64()

    def test_numbers_widen(self):
        assert schema_from_rows([{"a": 1}, {"a": 2.5}]).field("a").type == pa.float64()

    def test_irreconcilable_types_are_refused_by_name(self):
        with pytest.raises(FormatError, match=r"'a'.*schema="):
            schema_from_rows([{"a": 1}, {"a": "one"}])

    def test_the_first_row_alone_would_have_missed_the_field(self):
        """The control: what the one-row inference returned for the same input."""
        rows = [{"a": 1}, {"a": 2, "late": "x"}]
        assert pa.RecordBatch.from_pylist(rows[:1]).schema.names == ["a"]


class _Search:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.calls = 0

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        size = kwargs["size"]
        return {"hits": {"hits": [{"_source": h} for h in self.hits[:size]]}}

    def close(self) -> None:
        pass


def test_elasticsearch_infers_over_the_sample(monkeypatch):
    client = _Search([{"a": 1}] * 5 + [{"a": 2, "rare": True}])
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda self: client)
    assert ElasticsearchSource(hosts="h", index="i").schema().names == ["a", "rare"]


def test_a_declared_schema_is_never_inferred(monkeypatch):
    client = _Search([{"a": 1}, {"a": "one"}])  # would be refused if inferred
    monkeypatch.setattr(ElasticsearchSource, "_client", lambda self: client)
    declared = pa.schema([("a", pa.string())])
    source = ElasticsearchSource(hosts="h", index="i", schema=declared)
    assert source.schema() == declared
    assert client.calls == 0


def test_a_declared_schema_survives_the_trip_to_a_worker(monkeypatch):
    declared = pa.schema([("id", pa.int64()), ("tag", pa.string())])
    source = DynamoDBSource(table="t", region_name="us-east-1", schema=declared)
    monkeypatch.setattr(DynamoDBSource, "_enumerate_partitions", lambda self: [(0, 1)])
    split = pickle.loads(pickle.dumps(source.splits()[0]))
    monkeypatch.setattr(
        DynamoDBSource, "_infer_schema", lambda self: pytest.fail("the worker re-inferred")
    )
    assert split.schema() == declared


def test_the_sample_is_bounded():
    assert 1 < SCHEMA_SAMPLE_ROWS <= 1000


def test_a_failed_mongo_bulk_write_names_what_stayed_applied(monkeypatch):
    """`ordered=False` keeps every success; the error has to say which keys failed."""
    from types import SimpleNamespace

    from batcher._internal.errors import BackendError
    from batcher.io.formats.nosql import mongo

    class BulkWriteError(Exception):
        details: ClassVar[dict[str, Any]] = {
            "nInserted": 0,
            "nUpserted": 1,
            "nModified": 0,
            "nRemoved": 0,
            "writeErrors": [{"index": 1, "code": 11000}],
        }

    class _Coll:
        def bulk_write(self, ops, ordered):
            raise BulkWriteError("batch op errors occurred")

    class _Client:
        def __init__(self, uri):
            pass

        def __getitem__(self, _name):
            return {"c": _Coll()}

        def close(self):
            pass

    fake = SimpleNamespace(
        MongoClient=_Client,
        ReplaceOne=lambda *a, **k: ("replace", a),
    )
    monkeypatch.setattr(mongo, "require_driver", lambda *_a: fake)
    sink = mongo.MongoSink(uri="mongodb://h", database="d", collection="c")
    with pytest.raises(BackendError, match=r"upserted=1.*1 operation\(s\) failed.*\['b'\]"):
        sink._apply([{"_id": "a"}, {"_id": "b"}], "c")


def test_a_redis_scan_page_is_one_pipelined_round_trip():
    """Two round trips per SCAN page (the scan and one pipeline), not one GET per key."""
    from batcher.io.formats.nosql.redis import _NUM_SLOTS, _crc16_slot, _scan_range

    class _Pipe:
        def __init__(self, owner):
            self.owner, self.keys = owner, []

        def get(self, key):
            self.keys.append(key)

        def execute(self):
            self.owner.round_trips += 1
            return [f"v:{k}" for k in self.keys]

    class _Client:
        round_trips = 0

        def scan(self, *, cursor, match, count):
            self.round_trips += 1
            return 0, [f"k{i}" for i in range(50)]

        def pipeline(self, transaction=False):
            assert transaction is False
            return _Pipe(self)

        def get(self, key):  # pragma: no cover - the per-key path this replaces
            raise AssertionError("a per-key GET")

    client = _Client()
    rows = list(_scan_range(client, (0, _NUM_SLOTS), "*"))
    assert sorted(r["key"] for r in rows) == sorted(f"k{i}" for i in range(50))
    assert all(r["value"] == f"v:{r['key']}" for r in rows)
    assert client.round_trips == 2
    # Slots computed locally still form a disjoint cover.
    low = list(_scan_range(_Client(), (0, _NUM_SLOTS // 2), "*"))
    high = list(_scan_range(_Client(), (_NUM_SLOTS // 2, _NUM_SLOTS), "*"))
    assert len(low) + len(high) == 50
    assert all(_crc16_slot(r["key"]) < _NUM_SLOTS // 2 for r in low)
