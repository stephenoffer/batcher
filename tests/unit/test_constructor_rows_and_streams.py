"""Row shapes `from_records` reads, the errors the constructors give, and when input is consumed.

Each case here was a defect or a gap: dataclass rows were refused, namedtuple rows were told
they "carry no column names", ``columns=["x", "x"]`` silently lost a column, a duplicate
Arrow column name surfaced as pyarrow's `KeyError` mid-execution, ``{"b": "xy"}`` became a
column of characters, ``[1, "x"]`` was reported as "a sequence of builtins.int, which Arrow
cannot represent", and `from_batches` collected a stream producer into one table.
"""

from __future__ import annotations

import datetime
import enum
from collections import namedtuple
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


class Unit(enum.Enum):
    C = "celsius"


@dataclass
class Location:
    lat: float
    lon: float


@dataclass
class Reading:
    sensor: str
    value: float | None
    at: datetime.datetime
    where: Location
    tags: list


@dataclass
class WithEnum:
    unit: Unit


def test_dataclass_rows_convert_field_by_field():
    rows = [
        Reading("a", 1.5, datetime.datetime(2024, 1, 1), Location(1.0, 2.0), ["x"]),
        Reading("b", None, datetime.datetime(2024, 1, 2), Location(3.0, 4.0), []),
    ]
    ds = bt.from_records(rows)
    assert ds.columns == ["sensor", "value", "at", "where", "tags"]
    assert pa.types.is_timestamp(ds.schema.field("at").type)
    assert pa.types.is_struct(ds.schema.field("where").type)
    assert ds.to_pydict()["where"] == [{"lat": 1.0, "lon": 2.0}, {"lat": 3.0, "lon": 4.0}]
    assert ds.to_pydict()["value"] == [1.5, None]


def test_an_all_none_optional_field_is_typed_by_the_schema():
    rows = [Reading("a", None, datetime.datetime(2024, 1, 1), Location(0.0, 0.0), [])]
    assert pa.types.is_null(bt.from_records(rows).schema.field("value").type)
    typed = bt.from_records(
        rows,
        schema={
            "sensor": "string",
            "value": "float64",
            "at": "timestamp(us)",
            "where": pa.struct([("lat", pa.float64()), ("lon", pa.float64())]),
            "tags": pa.list_(pa.string()),
        },
    )
    assert typed.schema.field("value").type == pa.float64()


def test_an_enum_field_is_told_to_store_its_value():
    with pytest.raises(PlanError, match=r"column 'unit'.*\.value"):
        bt.from_records([WithEnum(Unit.C)])


def test_mixing_dataclass_and_tuple_rows_is_refused():
    with pytest.raises(PlanError, match=r"mix dataclass instances with tuple"):
        bt.from_records([Location(1.0, 2.0), (3.0, 4.0)])


def test_namedtuple_rows_name_their_own_columns():
    point = namedtuple("Point", ["x", "y"])
    assert bt.from_records([point(1, 2), point(3, 4)]).to_pydict() == {"x": [1, 3], "y": [2, 4]}
    renamed = bt.from_records([point(1, 2)], columns=["a", "b"])
    assert renamed.to_pydict() == {"a": [1], "b": [2]}


def test_from_items_points_a_namedtuple_at_from_records():
    point = namedtuple("Point", ["x", "y"])
    with pytest.raises(PlanError, match=r"from_records.*namedtuple"):
        bt.from_items([point(1, "a")])


def test_duplicate_record_columns_are_refused_rather_than_dropped():
    with pytest.raises(PlanError, match=r"\['x'\] appear more than once"):
        bt.from_records([(1, 2)], columns=["x", "x"])


def test_duplicate_arrow_names_are_a_plan_error_at_construction():
    table = pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=["x", "x"])
    with pytest.raises(PlanError, match=r"from_arrow\(\).*\['x'\] appear more than once"):
        bt.from_arrow(table)


def test_a_mixed_type_column_names_both_types_and_their_rows():
    with pytest.raises(PlanError) as info:
        bt.from_pylist([{"a": 1}, {"a": "x"}])
    message = str(info.value)
    assert "column 'a' mixes int (first at row 0) and str (first at row 1)" in message
    assert "builtins.int" not in message


def test_a_scalar_value_is_named_as_a_scalar():
    with pytest.raises(PlanError, match=r"column 'b' is a single int \(5\), not a column"):
        bt.from_pydict({"a": [1, 2], "b": 5})


def test_a_string_value_is_not_split_into_characters():
    """``{"b": "xy"}`` with two rows used to succeed as ``b = ["x", "y"]``."""
    with pytest.raises(PlanError, match=r"column 'b' is a single str \('xy'\)"):
        bt.from_pydict({"a": [1, 2], "b": "xy"})


def test_from_iter_consumes_its_input_at_construction():
    drawn: list[int] = []

    def gen():
        for i in range(3):
            drawn.append(i)
            yield i

    ds = bt.from_iter(gen())
    assert drawn == [0, 1, 2]
    assert ds.count() == 3


def _counting_reader(drawn: list[int], batches: int = 3) -> pa.RecordBatchReader:
    schema = pa.schema([("x", pa.int64())])

    def gen():
        for i in range(batches):
            drawn.append(i)
            yield pa.record_batch({"x": [i, i]}, schema=schema)

    return pa.RecordBatchReader.from_batches(schema, gen())


def test_from_batches_streams_a_reader_at_execution_not_construction():
    drawn: list[int] = []
    ds = bt.from_batches(_counting_reader(drawn))
    assert drawn == []
    assert ds.columns == ["x"]
    assert drawn == []
    assert sorted(ds.to_pydict()["x"]) == [0, 0, 1, 1, 2, 2]
    assert drawn == [0, 1, 2]


def test_a_consumed_reader_raises_instead_of_reading_as_empty():
    ds = bt.from_batches(_counting_reader([]))
    assert ds.count() == 6
    with pytest.raises(PlanError, match=r"already read.*single-shot"):
        ds.count()


def test_a_table_producer_is_re_exported_for_every_execution():
    ds = bt.from_batches(pa.table({"x": [1, 2]}))
    assert ds.count() == 2
    assert ds.to_pydict() == {"x": [1, 2]}


def test_from_batches_with_a_schema_draws_nothing_at_construction():
    calls: list[int] = []

    def factory():
        calls.append(1)
        return iter([pa.record_batch({"x": [1]})])

    ds = bt.from_batches(factory, pa.schema([("x", pa.int64())]))
    assert calls == []
    assert ds.count() == 1
    assert calls


def test_from_pandas_drops_the_index_by_default():
    df = pd.DataFrame({"v": [1, 2]}, index=pd.MultiIndex.from_tuples([(1, "a"), (2, "b")]))
    assert bt.from_pandas(df).columns == ["v"]


def test_from_pandas_preserve_index_follows_reset_index():
    named = pd.DataFrame(
        {"v": [1, 2]},
        index=pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["i", "j"]),
    )
    assert bt.from_pandas(named, preserve_index=True).to_pydict() == {
        "i": [1, 2],
        "j": ["a", "b"],
        "v": [1, 2],
    }
    partly = named.rename_axis([None, "j"])
    assert bt.from_pandas(partly, preserve_index=True).columns == ["level_0", "j", "v"]
    plain = pd.DataFrame({"v": [7]})
    assert bt.from_pandas(plain, preserve_index=True).to_pydict() == {"index": [0], "v": [7]}


def test_from_pandas_preserve_index_refuses_a_name_collision():
    df = pd.DataFrame({"v": [1]}, index=pd.Index([0], name="v"))
    with pytest.raises(PlanError, match=r"index level name\(s\) \['v'\] are already data"):
        bt.from_pandas(df, preserve_index=True)
