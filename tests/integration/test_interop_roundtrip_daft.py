"""Daft <-> Batcher, both directions, holding column types as well as values.

The shared fixture and pinned schemas are in `tests/_interop_cases.py`; the Polars leg's
module docstring says why the engines are split across files. Comparisons are ordered: the
fixture is sorted before it leaves Batcher, and a Daft frame built from one table keeps its
row order through ``to_arrow_iter``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _interop_cases import ENGINE_SCHEMA, FOREIGN_SCHEMA, ROWS, TABLE, assert_types, dataset, empty
from batcher import PlanError
from batcher.io import interop
from batcher.io.source import IteratorSource

daft = pytest.importorskip("daft")

pytestmark = pytest.mark.integration

#: `Dataset.to_daft` of the fixture, as Daft names its dtypes. The dictionary column arrives
#: as `String` because the engine decodes it; Daft would otherwise hold it as a Python object.
DAFT_DTYPES = {
    "id": "Int64",
    "i": "Int64",
    "s": "String",
    "b": "Binary",
    "l": "List[Int64]",
    "st": "Struct[x: Int64, y: String]",
    "ts": "Timestamp[us; Asia/Tokyo]",
    "d": "Decimal[precision: 10, scale: 2]",
    "dict": "String",
}


def _dtypes(frame) -> dict[str, str]:
    return {field.name: str(field.dtype) for field in frame.schema()}


def _daft_table() -> pa.Table:
    """The fixture with its dictionary column decoded, the form Daft can hold as Arrow."""
    return TABLE.set_column(8, "dict", TABLE.column("dict").cast(pa.string()))


def test_batcher_to_daft_keeps_types_values_and_order():
    frame = dataset().to_daft()
    assert _dtypes(frame) == DAFT_DTYPES
    assert frame.to_arrow().to_pylist() == ROWS


def test_daft_to_batcher_keeps_types_values_and_order():
    ds = bt.from_daft(daft.from_arrow(_daft_table()))
    assert_types(ds.schema, FOREIGN_SCHEMA)
    out = ds.to_arrow()
    assert_types(out.schema, FOREIGN_SCHEMA)
    assert out.to_pylist() == ROWS


def test_round_trip_through_daft_returns_the_same_rows():
    back = bt.from_daft(dataset().to_daft())
    assert_types(back.to_arrow().schema, FOREIGN_SCHEMA)
    assert back.to_pylist() == ROWS
    assert_types(dataset().schema, ENGINE_SCHEMA)


def test_an_empty_result_keeps_its_schema_both_ways():
    frame = empty().to_daft()
    assert _dtypes(frame) == DAFT_DTYPES
    assert frame.to_arrow().num_rows == 0
    out = bt.from_daft(frame).to_arrow()
    assert out.num_rows == 0
    assert_types(out.schema, FOREIGN_SCHEMA)


def test_from_any_dispatches_a_daft_frame():
    frame = daft.from_pydict({"x": [3, 1, 2]})
    assert bt.from_any(frame).sort("x").to_pydict() == {"x": [1, 2, 3]}


class _CountingFrame:
    """A Daft frame that counts how often its result is pulled."""

    def __init__(self, frame) -> None:
        self._frame = frame
        self.pulls = 0

    def schema(self):
        return self._frame.schema()

    def to_arrow_iter(self):
        self.pulls += 1
        yield from self._frame.to_arrow_iter()


def test_from_daft_streams_and_does_not_run_the_query_up_front():
    counting = _CountingFrame(daft.from_pydict({"x": [1, 2, 3]}))
    source = interop.from_daft(counting)
    assert isinstance(source, IteratorSource)
    ds = bt.from_daft(counting)
    assert counting.pulls == 0, "building the Dataset ran the Daft query"
    assert ds.to_pydict() == {"x": [1, 2, 3]}
    assert counting.pulls == 1


def test_a_python_dtype_column_is_declined_with_the_cast_that_fixes_it():
    # Daft holds a dictionary-encoded Arrow column as a Python object column, which has no
    # engine type: before the decline this failed deep in execution on an unhashable type.
    frame = daft.from_arrow(pa.table({"k": pa.array(["u", None]).dictionary_encode()}))
    with pytest.raises(PlanError, match=r"\['k'\].*cast"):
        bt.from_daft(frame)
    fixed = frame.with_column("k", frame["k"].cast(daft.DataType.string()))
    assert bt.from_daft(fixed).to_pydict() == {"k": ["u", None]}
