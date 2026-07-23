"""A multi-branch Avro union keeps every branch's values.

`_arrow_type` mapped a union to its **first non-null branch**, so an Avro
``["null", "long", "string"]`` column was advertised as `int64`. That is the same class as
the CSV bug in `test_csv_schema_agreement.py` — `schema()` promising a type the data does
not have — but it failed harder: the read did not return the wrong type, it *raised*
``ArrowInvalid: Could not convert 'hello' with type str`` from inside `from_pylist`, so a
valid Avro file was unreadable and the error pointed at pyarrow rather than at the mapping.

Multi-branch unions are ordinary in the wild — they are how Avro spells a nullable
sum type, and schema-registry payloads use them for evolving fields.

Arrow's own union types do not survive the rest of the engine (no operator consumes one),
so this follows Spark's Avro reader: a struct with one nullable `memberN` per branch. What
these tests pin is that the mapping is *lossless* and that the far commoner
``["null", T]`` idiom is left exactly as it was — a struct there would be a gratuitous
break for every existing Avro user.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit

fastavro = pytest.importorskip("fastavro")


def _write(tmp_path, fields: list[dict], records: list[dict]) -> str:
    path = tmp_path / "a.avro"
    schema = fastavro.parse_schema({"type": "record", "name": "R", "fields": fields})
    with open(path, "wb") as fh:
        fastavro.writer(fh, schema, records)
    return str(tmp_path)


def _source(directory: str):
    from batcher.io.formats.structured.avro import AvroSource

    return AvroSource(directory)


_UNION = [{"name": "v", "type": ["null", "long", "string"]}]
_RECORDS = [{"v": 1}, {"v": "hello"}, {"v": None}, {"v": 2}, {"v": "world"}]


def test_a_multi_branch_union_is_readable_at_all(tmp_path) -> None:
    """The headline: this used to raise on a valid file."""
    source = _source(_write(tmp_path, _UNION, _RECORDS))

    assert pa.Table.from_batches(list(source.read())).num_rows == 5


def test_no_branch_is_lost(tmp_path) -> None:
    """`branches[0]` kept the longs and destroyed the strings."""
    source = _source(_write(tmp_path, _UNION, _RECORDS))

    values = pa.Table.from_batches(list(source.read())).column("v").to_pylist()

    assert [v and v["member0"] for v in values] == [1, None, None, 2, None]
    assert [v and v["member1"] for v in values] == [None, "hello", None, None, "world"]


def test_exactly_one_member_is_set_per_row(tmp_path) -> None:
    """The struct stands in for a union, so it must behave like one."""
    source = _source(_write(tmp_path, _UNION, _RECORDS))

    for value in pa.Table.from_batches(list(source.read())).column("v").to_pylist():
        if value is not None:
            assert sum(v is not None for v in value.values()) == 1


def test_a_null_stays_a_null_not_an_all_empty_struct(tmp_path) -> None:
    """`{"member0": None, "member1": None}` is a distinct value from `None` and would
    make `is_null` false for a row that is null."""
    source = _source(_write(tmp_path, _UNION, _RECORDS))

    values = pa.Table.from_batches(list(source.read())).column("v").to_pylist()

    assert values[2] is None


def test_the_advertised_schema_is_what_the_read_produces(tmp_path) -> None:
    """The contract that was broken. The engine types its operators from `schema()`."""
    source = _source(_write(tmp_path, _UNION, _RECORDS))

    produced = pa.Table.from_batches(list(source.read())).schema

    assert produced == source.schema()
    assert pa.types.is_struct(produced.field("v").type)


def test_both_read_paths_agree(tmp_path) -> None:
    """`read()` and `iter_batches()` build batches through different call sites."""
    source = _source(_write(tmp_path, _UNION, _RECORDS))

    from_read = pa.Table.from_batches(list(source.read()))
    from_iter = pa.Table.from_batches(list(source.iter_batches()))

    assert from_read.equals(from_iter)


def test_the_nullable_scalar_idiom_is_unchanged(tmp_path) -> None:
    """`["null", "long"]` is how *most* Avro fields are written. It has an exact Arrow
    equivalent — a nullable int64 — and must not become a one-member struct."""
    directory = _write(
        tmp_path,
        [{"name": "v", "type": ["null", "long"]}],
        [{"v": 1}, {"v": None}, {"v": 3}],
    )
    source = _source(directory)

    table = pa.Table.from_batches(list(source.read()))

    assert table.schema.field("v").type == pa.int64()
    assert table.column("v").to_pylist() == [1, None, 3]


def test_a_union_without_null_is_still_a_struct(tmp_path) -> None:
    """`["long", "string"]` has no null branch but is just as unrepresentable."""
    directory = _write(
        tmp_path,
        [{"name": "v", "type": ["long", "string"]}],
        [{"v": 5}, {"v": "five"}],
    )

    table = pa.Table.from_batches(list(_source(directory).read()))

    assert [v["member0"] for v in table.column("v").to_pylist()] == [5, None]
    assert [v["member1"] for v in table.column("v").to_pylist()] == [None, "five"]


def test_a_boolean_branch_is_not_captured_by_the_integer_one(tmp_path) -> None:
    """`bool` is a subclass of `int` in Python, so a naive `isinstance` dispatch puts
    `True` in the long member and reads it back as `1`."""
    directory = _write(
        tmp_path,
        [{"name": "v", "type": ["null", "boolean", "long"]}],
        [{"v": True}, {"v": 7}, {"v": False}],
    )

    values = pa.Table.from_batches(list(_source(directory).read())).column("v").to_pylist()

    assert [v["member0"] for v in values] == [True, None, False]
    assert [v["member1"] for v in values] == [None, 7, None]


def test_a_union_column_survives_the_split_round_trip(tmp_path) -> None:
    """The distributed path rebuilds the reader from the pickled split alone."""
    import pickle

    source = _source(_write(tmp_path, _UNION, _RECORDS))

    shipped = [pickle.loads(pickle.dumps(s)) for s in source.splits()]
    table = pa.Table.from_batches([b for s in shipped for b in s.read()])

    assert table.equals(pa.Table.from_batches(list(source.read())))


def test_the_engine_can_read_a_union_column_end_to_end(tmp_path) -> None:
    """A schema the source can build but no operator can consume is not a fix."""
    import batcher as bt

    directory = _write(tmp_path, _UNION, _RECORDS)

    assert bt.read.avro(directory).count() == 5
