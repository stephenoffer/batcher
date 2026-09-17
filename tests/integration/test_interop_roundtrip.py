"""Polars <-> Batcher, both directions, holding column types as well as values.

The Daft and Spark legs live beside this file (`test_interop_roundtrip_daft.py`,
`test_interop_roundtrip_spark.py`) so that each suite is gated on its own framework at module
level, where `just lint-skips` counts it. The fixture and the pinned schemas are shared, in
`tests/_interop_cases.py`.

Every comparison here is ordered. The fixture is sorted on its key before it leaves Batcher,
and a scan of a Polars frame keeps the frame's row order, so a reordering is a defect.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _interop_cases import (
    ENGINE_SCHEMA,
    FOREIGN_SCHEMA,
    ROWS,
    TABLE,
    assert_types,
    dataset,
    empty,
)
from batcher.io import interop
from batcher.io.source import IteratorSource

pl = pytest.importorskip("polars")

pytestmark = pytest.mark.integration

#: `Dataset.to_polars` of the fixture, as Polars names its dtypes.
POLARS_DTYPES = {
    "id": "Int64",
    "i": "Int64",
    "s": "String",
    "b": "Binary",
    "l": "List(Int64)",
    "st": "Struct({'x': Int64, 'y': String})",
    "ts": "Datetime(time_unit='us', time_zone='Asia/Tokyo')",
    "d": "Decimal(precision=10, scale=2)",
    "dict": "String",
}


def _dtypes(frame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def test_the_fixture_reaches_the_engine_with_the_pinned_types():
    # The baseline both schemas are stated against: without it, a change to the engine's own
    # boundary would read as a Polars defect below.
    ds = dataset()
    assert_types(ds.schema, ENGINE_SCHEMA)
    assert_types(ds.to_arrow().schema, ENGINE_SCHEMA)
    assert ds.to_pylist() == ROWS


def test_batcher_to_polars_keeps_types_values_and_order():
    frame = dataset().to_polars()
    assert _dtypes(frame) == POLARS_DTYPES
    assert frame.to_dicts() == ROWS


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
def test_polars_to_batcher_keeps_types_values_and_order(lazy):
    frame = pl.from_arrow(TABLE)
    ds = bt.from_polars(frame.lazy() if lazy else frame)
    assert_types(ds.schema, FOREIGN_SCHEMA)
    out = ds.to_arrow()
    assert_types(out.schema, FOREIGN_SCHEMA)
    assert out.to_pylist() == ROWS


def test_round_trip_through_polars_returns_the_same_rows():
    back = bt.from_polars(dataset().to_polars())
    assert_types(back.to_arrow().schema, FOREIGN_SCHEMA)
    assert back.to_pylist() == ROWS


def test_an_empty_result_keeps_its_schema_both_ways():
    frame = empty().to_polars()
    assert frame.height == 0
    assert _dtypes(frame) == POLARS_DTYPES
    for source in (frame, frame.lazy()):
        out = bt.from_polars(source).to_arrow()
        assert out.num_rows == 0
        assert_types(out.schema, FOREIGN_SCHEMA)


def test_no_view_layout_reaches_the_engine():
    # Polars stores strings as views, and its newest Arrow export says so. A regression to
    # that export would still pass the value checks above, because the engine's boundary
    # normalizes the column type, so the source's own schema is what is checked.
    newest = pl.from_arrow(TABLE).to_arrow(compat_level=pl.CompatLevel.newest())
    assert pa.types.is_string_view(newest.schema.field("s").type), "positive control"
    for frame in (pl.from_arrow(TABLE), pl.from_arrow(TABLE).lazy()):
        batches = interop.from_polars(frame).read()
        assert batches, "the source produced no batch to inspect"
        for batch in batches:
            assert not pa.types.is_string_view(batch.schema.field("s").type)
            assert pa.types.is_large_binary(batch.schema.field("b").type)


def test_a_lazy_frame_is_not_collected_until_the_dataset_runs():
    # A LazyFrame whose query fails when it runs: building the Dataset must not run it.
    failing = pl.LazyFrame({"a": [1_000]}).select(pl.col("a").cast(pl.Int8, strict=True))
    ds = bt.from_polars(failing)
    assert isinstance(interop.from_polars(failing), IteratorSource)
    assert ds.columns == ["a"]
    with pytest.raises(Exception, match=r"(?i)cast|conversion|overflow|strict"):
        ds.to_arrow()


def test_a_sliced_frame_with_a_null_struct_row_converts():
    # Polars 1.40 exports this slice with struct children one row short, which pyarrow
    # rejects. Every chunk `collect_batches` yields is a slice, so a LazyFrame over this data
    # failed on the very first chunk.
    frame = pl.from_arrow(TABLE).slice(1, 2)
    ds = bt.from_polars(frame)
    assert_types(ds.to_arrow().schema, FOREIGN_SCHEMA)
    assert ds.to_pylist() == ROWS[1:]
