"""`select_dtypes`, `by_dtype`, `match_to_schema` and `matched_columns` read dtypes one way.

Defects pinned here: `select_dtypes` refused `include` and `exclude` together and knew no
nested family; ``ds.match_to_schema(ds.schema)`` failed on its own schema for a list or
struct column because a pyarrow type was routed through the cast-name grammar;
``bt.by_dtype("decimal(10,2)")`` used ``pa.type_for_alias`` rather than the cast parser;
and ``Selector.matched_columns(ds.columns, ds.schema)`` raised `AttributeError`.
"""

from __future__ import annotations

import decimal

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


@pytest.fixture
def mixed() -> bt.Dataset:
    return bt.from_pydict(
        {
            "i": [1],
            "f": [0.5],
            "s": ["x"],
            "b": [b"x"],
            "d": pa.array([decimal.Decimal("1.25")], pa.decimal128(10, 2)),
            "l": [[1]],
            "st": [{"a": 1}],
            "m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
            "t": list(np.zeros((1, 2, 2))),
        }
    )


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("list", ["l"]),
        ("struct", ["st"]),
        ("map", ["m"]),
        ("binary", ["b"]),
        ("decimal", ["d"]),
        ("nested", ["l", "st", "m"]),
        ("tensor", ["t"]),
        (bytes, ["b"]),
    ],
)
def test_the_nested_and_binary_families(mixed, family, expected):
    assert mixed.select_dtypes(family).columns == expected


def test_include_minus_exclude_in_column_order(mixed):
    assert mixed.select_dtypes(include="number", exclude="integer").columns == ["f", "d"]
    assert mixed.select_dtypes(include=["string", "list"], exclude="decimal").columns == [
        "s",
        "l",
    ]


def test_a_family_in_both_is_refused(mixed):
    with pytest.raises(PlanError, match=r"name the same dtype family"):
        mixed.select_dtypes(include="numeric", exclude=["number"])


def test_neither_is_still_refused(mixed):
    with pytest.raises(PlanError, match=r"takes `include`, `exclude`, or both"):
        mixed.select_dtypes()


def test_a_concrete_name_selects_its_whole_family(mixed):
    """What the corrected docstring says: ``"float32"`` matches the ``float64`` column."""
    assert mixed.select_dtypes("float32").columns == ["f"]


def test_match_to_schema_accepts_its_own_nested_schema(mixed):
    conformed = mixed.match_to_schema(mixed.schema)
    assert conformed.columns == mixed.columns
    assert conformed.schema.equals(mixed.schema)


def test_match_to_schema_accepts_a_pyarrow_nested_type_in_a_mapping():
    ds = bt.from_pydict({"l": [[1, 2]]})
    assert ds.match_to_schema({"l": pa.list_(pa.int64())}).to_pydict() == {"l": [[1, 2]]}


def test_inserting_a_missing_nested_column_keeps_its_clear_refusal():
    ds = bt.from_pydict({"a": [1]})
    with pytest.raises(PlanError, match=r"cannot build column 'l'.*no name for it"):
        ds.match_to_schema({"a": "int64", "l": pa.list_(pa.int64())}, missing_columns="insert")


def test_by_dtype_reads_every_cast_spelling(mixed):
    assert mixed.select(bt.by_dtype("decimal(10,2)")).columns == ["d"]
    assert mixed.select(bt.by_dtype("BIGINT")).columns == ["i"]
    assert mixed.select(bt.by_dtype(float)).columns == ["f"]


def test_by_dtype_still_reads_pyarrow_aliases():
    ds = bt.from_arrow(pa.table({"ts": pa.array([0], pa.timestamp("ms"))}))
    assert ds.select(bt.by_dtype("timestamp[ms]")).columns == ["ts"]


def test_by_dtype_refuses_an_unknown_name():
    with pytest.raises(PlanError, match=r"by_dtype\(\) takes pyarrow types"):
        bt.by_dtype("not_a_type")


def test_matched_columns_takes_the_schema_a_user_holds(mixed):
    assert bt.integer().matched_columns(mixed.columns, mixed.schema) == ["i"]
    assert bt.matches("^s").matched_columns(mixed.columns, mixed.schema) == ["s", "st"]


def test_schema_reports_the_widened_execution_type():
    """What the corrected `Dataset.schema` docstring promises."""
    narrow = bt.from_arrow(pa.table({"x": pa.array([1], pa.int8())}))
    assert narrow.schema.field("x").type == pa.int64()
