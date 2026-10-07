"""`Dataset.schema` over a Python callback, and a declared `output_columns` schema.

A callback stage is opaque, so its output types are learned in one of two ways: from a
`pyarrow.Schema` given as `output_columns` (no call at all), or from a probe that hands each
undeclared batch callback an empty batch, and a per-row callback (which is never called on an
empty batch) one row. Before the probe was bounded, `schema` ran the whole
callback pipeline over every input row -- seven calls over 100,000 rows -- for a property
read. Each test counts what the callback actually saw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import PlanError, SchemaError

pytestmark = pytest.mark.integration

_ROWS = 100_000
_Y = pa.schema([pa.field("y", pa.int64())])


class _Counter:
    """A `map_batches` callable doubling `x` into `y`, counting calls and rows."""

    def __init__(self) -> None:
        self.calls = 0
        self.rows = 0

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        self.calls += 1
        self.rows += batch.num_rows
        return pa.record_batch({"y": pc.multiply(batch.column("x"), 2)})


def _big() -> bt.Dataset:
    return bt.from_pydict({"x": list(range(_ROWS))})


def test_a_declared_schema_answers_without_calling_the_function() -> None:
    fn = _Counter()
    ds = _big().map_batches(fn, output_columns=_Y)
    assert ds.schema == _Y
    assert (fn.calls, fn.rows) == (0, 0)


def test_an_undeclared_batch_stage_is_probed_on_an_empty_batch() -> None:
    fn = _Counter()
    schema = _big().map_batches(fn, output_columns=["y"]).schema
    assert schema.names == ["y"]
    assert schema.field("y").type == pa.int64()
    assert fn.rows == 0
    assert fn.calls <= 1


def test_every_stage_of_a_chain_is_bounded() -> None:
    first, second = _Counter(), _Counter()

    def rename(batch: pa.RecordBatch) -> pa.RecordBatch:
        return pa.record_batch({"x": batch.column("y")})

    ds = (
        _big()
        .map_batches(first, output_columns=["y"])
        .map_batches(rename, output_columns=["x"])
        .map_batches(second, output_columns=["y"])
    )
    assert ds.schema.names == ["y"]
    assert (first.rows, second.rows) == (0, 0)


def test_a_batch_stage_beneath_a_row_stage_gets_the_row_it_needs() -> None:
    # A per-row stage needs a row to answer, so the batch stage feeding it is handed one too
    # rather than emitting nothing for the row stage to be called on.
    first = _Counter()
    ds = _big().map_batches(first, output_columns=["y"]).map(lambda r: {"z": str(r["y"])})
    schema = ds.schema
    assert schema.names == ["z"]
    assert schema.field("z").type == pa.string()
    assert first.rows == 1


def test_a_function_that_refuses_an_empty_batch_is_asked_on_one_row() -> None:
    seen: list[int] = []

    def needs_rows(batch: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(batch.num_rows)
        if batch.num_rows == 0:
            raise ValueError("no rows to stack")
        return pa.record_batch({"y": pc.multiply(batch.column("x"), 2)})

    schema = _big().map_batches(needs_rows, output_columns=["y"]).schema
    assert schema.field("y").type == pa.int64()
    assert sum(seen) == 1


def test_a_row_callback_reports_its_output_columns_not_its_input() -> None:
    # A per-row callback is never called on an empty batch, which is why the probe hands it a
    # row rather than none: a zero-row probe reported the *input* column `x` here.
    seen: list[int] = []

    def fn(row: dict[str, Any]) -> dict[str, Any]:
        seen.append(row["x"])
        return {"label": str(row["x"])}

    schema = _big().map(fn).schema
    assert schema.names == ["label"]
    assert schema.field("label").type == pa.string()
    assert len(seen) <= 1


def test_a_callable_filter_keeps_its_input_schema_without_running() -> None:
    fn = _Counter()

    def keep(batch: pa.RecordBatch) -> pa.Array:
        fn(batch)
        return pc.greater(batch.column("x"), 5)

    ds = _big().with_columns(s=bt.col("x").cast(pa.string())).filter(keep)
    assert ds.schema == ds.collect().schema
    fn.calls = fn.rows = 0
    assert ds.schema.names == ["x", "s"]
    assert (fn.calls, fn.rows) == (0, 0)


@pytest.mark.parametrize("verb", ["map_batches", "map", "flat_map"])
def test_an_empty_input_returns_the_declared_type(verb: str) -> None:
    # The F2 defect: no rows reached the function, so `y` came back Arrow `null`.
    empty = bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") < 0)
    schema = pa.schema([pa.field("y", pa.float64())])
    fns = {
        "map_batches": lambda b: {"y": pc.cast(b.column("x"), pa.float64())},
        "map": lambda r: {"y": float(r["x"])},
        "flat_map": lambda r: [{"y": float(r["x"])}],
    }
    out = getattr(empty, verb)(fns[verb], output_columns=schema).collect()
    assert out.num_rows == 0
    assert out.schema == schema


def test_a_pandas_round_trip_of_an_empty_ragged_tensor_keeps_its_type() -> None:
    # The probe hands a batch callback an empty batch, so an empty batch's round trip has to
    # keep the type: pandas made an empty tensor column `float64`, which came back `double`.
    images = [np.zeros((2, 2), "uint8"), np.zeros((3, 4), "uint8")]
    ds = bt.from_pydict({"id": [1, 2], "img": images})
    full = ds.map_batches(lambda b: b, batch_format="pandas")
    empty = ds.filter(bt.col("id") > 9).map_batches(lambda b: b, batch_format="pandas")
    assert empty.collect().schema.field("img").type == full.collect().schema.field("img").type
    assert full.schema.field("img").type == full.collect().schema.field("img").type


def test_a_name_only_declaration_is_unchanged_on_empty_input() -> None:
    # The list form keeps its old meaning: names only, types from whatever ran.
    empty = bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") < 0)
    out = empty.map(lambda r: {"y": r["x"]}, output_columns=["y"]).collect()
    assert out.schema.names == ["y"]


def test_every_output_batch_is_cast_to_the_declared_type() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3]}).map_batches(
        lambda b: {"y": pc.cast(b.column("x"), pa.int8())},
        output_columns=pa.schema([pa.field("y", pa.float64())]),
    )
    out = ds.collect()
    assert out.schema.field("y").type == pa.float64()
    assert out.column("y").to_pylist() == [1.0, 2.0, 3.0]


def test_a_narrow_declared_type_reports_what_collect_returns() -> None:
    # The engine boundary widens int32 -> int64 and float32 -> float64 for every operator
    # above the stage, so the stage's own output is held to the same types; otherwise
    # `schema` and `collect` disagreed at the top and agreed one `select` later.
    declared = pa.schema([("i", pa.int32()), ("f", pa.float32())])
    ds = bt.from_pydict({"x": [1, 2]}).map_batches(
        lambda b: {
            "i": pc.cast(b.column("x"), pa.int32()),
            "f": pc.cast(b.column("x"), pa.float32()),
        },
        output_columns=declared,
    )
    for q in (ds, ds.select("i"), ds.with_columns(j=bt.col("i") + 1), ds.sort("i")):
        assert q.schema.types == q.collect().schema.types


def test_a_tensor_column_is_declared_with_the_arrow_tensor_type() -> None:
    tensor = pa.fixed_shape_tensor(pa.float32(), [2, 2])
    declared = pa.schema([pa.field("t", tensor)])

    def fn(batch: pa.RecordBatch) -> dict[str, Any]:
        values = np.ones((batch.num_rows, 2, 2), dtype=np.float32)
        return {"t": pa.FixedShapeTensorArray.from_numpy_ndarray(values)}

    ds = bt.from_pydict({"x": [1, 2]}).map_batches(fn, output_columns=declared)
    assert ds.schema.field("t").type == tensor
    assert ds.collect().schema.field("t").type == tensor


def test_a_type_the_function_cannot_meet_is_named() -> None:
    ds = bt.from_pydict({"x": [1, 2]}).map_batches(
        lambda b: {"y": pa.array(["a", "b"])}, output_columns=_Y
    )
    with pytest.raises(SchemaError, match="'y'"):
        ds.collect()


def test_a_non_nullable_declaration_refuses_nulls() -> None:
    declared = pa.schema([pa.field("y", pa.int64(), nullable=False)])
    ds = bt.from_pydict({"x": [1, 2]}).map_batches(
        lambda b: {"y": pa.array([1, None], pa.int64())}, output_columns=declared
    )
    with pytest.raises(SchemaError, match="non-nullable"):
        ds.collect()


def test_a_pruned_pass_through_column_is_filled_not_refused() -> None:
    # `input_columns` lets projection pushdown drop a declared pass-through column nothing
    # above reads; the conformed output fills it rather than failing.
    ds = bt.from_pydict({"x": [1, 2], "z": ["a", "b"]}).map_batches(
        lambda b: b.append_column("y", pc.multiply(b.column("x"), 2)),
        input_columns=["x"],
        output_columns=pa.schema([("x", pa.int64()), ("z", pa.string()), ("y", pa.int64())]),
    )
    assert ds.select("y").to_pydict() == {"y": [2, 4]}
    assert ds.collect().to_pydict() == {"x": [1, 2], "z": ["a", "b"], "y": [2, 4]}


def test_a_string_output_columns_is_refused() -> None:
    with pytest.raises(PlanError, match="wrap it in a list"):
        bt.from_pydict({"x": [1]}).map_batches(lambda b: b, output_columns="x")
