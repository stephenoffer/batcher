"""CSV refuses a nested column by name, instead of letting pyarrow blame a bare type.

CSV is a grid of scalar cells. A `list`, `struct`, `map` or `fixed_size_list` column has
nowhere to go in one, and pyarrow says so as ``ArrowInvalid: Unsupported Type:list<item:
int64>`` — a message naming neither the column, nor CSV, nor anything to do about it. On a
wide frame that leaves the caller grepping their schema for a type they never wrote down,
and the obvious conclusion (that the engine cannot hold the column) is wrong: Parquet,
Arrow IPC and JSON all carry it.

This is the same guard `avro` and `orc` already have for their own unrepresentable types,
and it is checked against the *schema* rather than around the writer call so it covers all
three of the CSV sink's encode paths — the serial write, the parallel row-range encode, and
the streaming window — rather than whichever one a given test happens to take.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import SchemaError

pytestmark = pytest.mark.io

#: One column of each type CSV has no cell for, with a value so the frame is non-empty.
_NESTED = {
    "list": pa.array([["a", "b"]], pa.list_(pa.string())),
    "large_list": pa.array([["a"]], pa.large_list(pa.string())),
    "fixed_size_list": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0]), 2),
    "struct": pa.StructArray.from_arrays([pa.array([1])], names=["x"]),
    "map": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
}


@pytest.mark.parametrize("kind", sorted(_NESTED))
def test_a_nested_column_is_refused_by_name(kind, tmp_path):
    column = _NESTED[kind]
    table = pa.table({"id": pa.array([1] * len(column)), "payload": column})
    with pytest.raises(SchemaError) as err:
        bt.from_arrow(table).write.csv(str(tmp_path / f"{kind}.csv"))
    message = str(err.value)
    assert "'payload'" in message, "the column the caller has to fix must be named"
    assert "csv" in message.lower()


def test_the_streaming_path_refuses_it_too(tmp_path):
    """`write_stream` encodes from an iterator, so it checks each batch's own schema."""
    table = pa.table({"id": pa.array([1]), "payload": _NESTED["list"]})
    batches = list(bt.from_arrow(table).iter_batches())
    with pytest.raises(SchemaError, match="'payload'"):
        bt.from_arrow(batches).write.csv(str(tmp_path / "stream.csv"))


@pytest.mark.parametrize(
    "remedy",
    [
        pytest.param(lambda: bt.col("payload").cast("string"), id="cast"),
        pytest.param(lambda: bt.col("payload").list.join(","), id="list_join"),
    ],
)
def test_the_remedies_the_message_names_actually_work(remedy, tmp_path):
    """A message that names a fix is only useful if the fix runs. Both of these are run here.

    An earlier draft of this error suggested `.json.encode()`, which does not exist on the
    expression API — so the one actionable sentence would have sent the caller to an
    `AttributeError`.
    """
    ds = bt.from_arrow(pa.table({"id": pa.array([1]), "payload": _NESTED["list"]}))
    out = tmp_path / "fixed.csv"
    ds.with_columns(payload=remedy()).write.csv(str(out))
    assert "payload" in out.read_text().splitlines()[0]


def test_a_scalar_only_frame_is_untouched(tmp_path):
    """The control: the guard must not stand between an ordinary frame and its file."""
    out = tmp_path / "plain.csv"
    bt.from_pydict({"id": [1, 2], "s": ["x", "y"]}).write.csv(str(out))
    assert out.read_text().splitlines()[1:] == ['1,"x"', '2,"y"']


def test_a_dictionary_column_of_strings_is_still_written(tmp_path):
    """A dictionary column is written as its values, so it follows its value type, not itself."""
    table = pa.table({"c": pa.array(["a", "b", "a"]).dictionary_encode()})
    out = tmp_path / "dict.csv"
    bt.from_arrow(table).write.csv(str(out))
    assert out.read_text().count("a") >= 2
