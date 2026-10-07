"""Every in-memory constructor takes one `schema=` keyword, spelled the way `cast` spells a dtype.

`from_pydict` used to accept only a `pa.Schema`; `from_pylist`, `from_records`, `from_items`
and `from_iter` took none, so an empty input came back as a zero-column or `null`-typed
relation, and ``from_pydict({}, schema=s)`` raised pyarrow's raw `KeyError`. These pin the
shared vocabulary (a `pa.Schema` or a ``{column: dtype}`` mapping read by the cast parser)
and the typed-empty result, nested types included.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

NESTED = pa.schema(
    [
        ("s", pa.struct([("a", pa.int64()), ("b", pa.list_(pa.string()))])),
        ("m", pa.map_(pa.string(), pa.float64())),
        ("n", pa.int64()),
    ]
)

EMPTY_CONSTRUCTORS = {
    "from_pydict": lambda schema: bt.from_pydict({}, schema=schema),
    "from_pylist": lambda schema: bt.from_pylist([], schema=schema),
    "from_records": lambda schema: bt.from_records([], schema=schema),
    "from_items": lambda schema: bt.from_items([], schema=schema),
    "from_iter": lambda schema: bt.from_iter(iter([]), schema=schema),
}


@pytest.mark.parametrize("name", sorted(EMPTY_CONSTRUCTORS))
def test_an_empty_input_with_a_schema_is_the_typed_empty_dataset(name):
    ds = EMPTY_CONSTRUCTORS[name](NESTED)
    assert ds.schema.equals(NESTED)
    assert ds.count() == 0
    assert ds.collect().schema.equals(NESTED)


@pytest.mark.parametrize("name", sorted(EMPTY_CONSTRUCTORS))
def test_a_mapping_schema_means_what_the_pyarrow_schema_means(name):
    as_mapping = {"s": NESTED.field("s").type, "m": NESTED.field("m").type, "n": "BIGINT"}
    assert EMPTY_CONSTRUCTORS[name](as_mapping).schema.equals(NESTED)


def test_the_mapping_reads_every_cast_spelling():
    ds = bt.from_pydict(
        {"d": ["1.25"], "t": [None], "i": [1]},
        schema={"d": "string", "t": "timestamp(us, UTC)", "i": int},
    )
    assert [str(t) for t in ds.dtypes] == ["string", "timestamp[us, tz=UTC]", "int64"]
    dec = bt.from_pydict({"x": []}, schema={"x": "DECIMAL(10,2)"})
    assert dec.schema.field("x").type == pa.decimal128(10, 2)


def test_from_pydict_with_a_schema_follows_its_order():
    ds = bt.from_pydict({"a": [1], "b": ["x"]}, schema={"b": "string", "a": "int64"})
    assert ds.columns == ["b", "a"]
    assert ds.to_pydict() == {"b": ["x"], "a": [1]}


def test_from_pydict_names_a_schema_column_the_mapping_lacks():
    with pytest.raises(PlanError, match=r"schema column\(s\) \['b'\] are not in the mapping"):
        bt.from_pydict({"a": [1]}, schema={"a": "int64", "b": "string"})


def test_an_unknown_dtype_names_the_column():
    with pytest.raises(PlanError, match=r"schema column 'a' names unknown dtype 'nope'"):
        bt.from_pylist([{"a": 1}], schema={"a": "nope"})


def test_a_schema_that_is_neither_form_is_refused():
    with pytest.raises(PlanError, match=r"schema must be a pyarrow.Schema or a"):
        bt.from_items([1], schema=["item"])


def test_from_pylist_with_a_schema_drops_unnamed_keys_and_nulls_missing_ones():
    rows = [{"a": 1, "extra": "dropped"}, {"b": "q"}]
    ds = bt.from_pylist(rows, schema={"b": "string", "a": "int64"})
    assert ds.to_pydict() == {"b": [None, "q"], "a": [1, None]}


def test_the_strict_form_is_match_to_schema():
    """The docstring points here for an extra-key policy; prove the pointer works."""
    rows = [{"a": 1, "extra": "x"}]
    with pytest.raises(PlanError, match=r"not in the schema"):
        bt.from_pylist(rows).match_to_schema({"a": "int64"}, extra_columns="raise")


def test_from_records_takes_tuple_row_names_from_the_schema():
    ds = bt.from_records([(1, "a")], schema={"n": "int32", "s": "string"})
    assert ds.to_pydict() == {"n": [1], "s": ["a"]}


def test_scalar_items_fill_the_schemas_column():
    assert bt.from_items([1, 2], schema={"item": "float64"}).to_pydict() == {"item": [1.0, 2.0]}
    with pytest.raises(PlanError, match=r"fill the single column 'item'.*\['v'\]"):
        bt.from_iter([1, 2], schema={"v": "float64"})


def test_a_value_outside_the_declared_type_names_column_row_and_type():
    with pytest.raises(PlanError, match=r"column 'a' holds str value 'x' at row 1.*int64"):
        bt.from_pylist([{"a": 1}, {"a": "x"}], schema={"a": "int64"})


def test_no_schema_still_infers():
    """The keyword is opt-in: without it the union-of-keys inference is unchanged."""
    ds = bt.from_pylist([{"a": 1}, {"a": 2, "b": 3}])
    assert ds.to_pydict() == {"a": [1, 2], "b": [None, 3]}
