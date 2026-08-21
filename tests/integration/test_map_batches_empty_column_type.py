"""`batch_format` must not change a column's type on the batches that carry no data.

A NumPy or pandas round-trip carries a string column as `object`. An `object` column holding
no non-null value says nothing about what it held, so Arrow infers `null` and the column comes
back `null`-typed -- an identity `fn` changing a column *type*, on exactly the inputs that are
easiest to miss: a filter that matched nothing, or one empty partition among many.

`batch_format` is documented as choosing what the `fn` speaks, not what the query returns, so
every format has to agree here. The device tier hit the same defect and fixed it by name
(`core/gpu_plan/backend.py::_restore_empty_strings`); these are the CPU pair of that test.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

FORMATS = ["pyarrow", "numpy", "pandas"]


def _types(table: pa.Table) -> dict[str, str]:
    return dict(zip(table.schema.names, (str(t) for t in table.schema.types), strict=True))


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_empty_result_keeps_its_string_column_typed(fmt):
    """The filter matches nothing, so every batch reaching `fn` is empty."""
    out = (
        bt.from_pydict({"i": [1, 2, 3], "s": ["a", "b", "c"]})
        .filter(bt.col("i") > 100)
        .map_batches(lambda b: b, batch_format=fmt)
        .collect()
    )
    assert out.num_rows == 0
    assert _types(out) == {"i": "int64", "s": "string"}


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_all_null_string_column_keeps_its_type(fmt):
    """Rows, but no non-null value -- the same blind spot as an empty column."""
    source = pa.table(
        {"s": pa.array([None, None], type=pa.string()), "i": pa.array([1, 2], type=pa.int64())}
    )
    out = bt.from_arrow(source).map_batches(lambda b: b, batch_format=fmt).collect()
    assert _types(out) == {"s": "string", "i": "int64"}
    assert out.column("s").to_pylist() == [None, None]


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_agrees_with_pyarrow_on_an_empty_result(fmt):
    """Stated as agreement with the lossless format, which is the actual contract."""

    def run(f):
        return (
            bt.from_pydict({"i": [1], "s": ["a"], "f": [1.5]})
            .filter(bt.col("i") < 0)
            .map_batches(lambda b: b, batch_format=f)
            .collect()
        )

    assert _types(run(fmt)) == _types(run("pyarrow"))


def test_a_column_the_fn_invents_is_not_given_a_type_it_never_had():
    """The restore reads the input schema; it does not guess for a column that was not there.

    This is the guard that keeps the fix honest -- it puts a type *back*, and inventing one for
    a new all-null column would be the same silent type decision in the other direction.
    """

    def add_column(frame):
        frame = frame.copy()
        frame["brand_new"] = [None] * len(frame)
        return frame

    source = pa.table({"s": pa.array([None], type=pa.string())})
    out = bt.from_arrow(source).map_batches(add_column, batch_format="pandas").collect()
    assert _types(out) == {"s": "string", "brand_new": "null"}


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_populated_result_is_unchanged(fmt):
    """The control: the restore must not touch the ordinary path's values or types."""
    out = (
        bt.from_pydict({"i": [1, 2, 3], "s": ["a", "b", "c"]})
        .map_batches(lambda b: b, batch_format=fmt)
        .collect()
    )
    assert _types(out) == {"i": "int64", "s": "string"}
    assert out.to_pydict() == {"i": [1, 2, 3], "s": ["a", "b", "c"]}


@pytest.mark.parametrize("fmt", FORMATS)
def test_map_groups_keeps_an_all_null_string_column_typed(fmt):
    """`map_groups` runs the same coercion and lost the same type, in a louder way.

    Without the input schema to restore from, the group reassembly read the type-less column
    as ``list<item: string>`` rather than `null` -- so this path diverged from `pyarrow` by a
    whole column shape, not just a type name.
    """
    source = pa.table(
        {
            "g": pa.array(["k1", "k1", "k2"], type=pa.string()),
            "s": pa.array([None, None, None], type=pa.string()),
            "v": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    out = bt.from_arrow(source).groupby("g").map_groups(lambda b: b, batch_format=fmt).collect()
    assert _types(out) == {"g": "string", "s": "string", "v": "int64"}
    assert out.num_rows == 3


@pytest.mark.parametrize("fmt", FORMATS)
def test_map_groups_on_populated_columns_is_unchanged(fmt):
    """The control for `map_groups`: a column with values was never at risk, and stays put."""
    source = pa.table({"g": ["k1", "k1", "k2"], "s": ["a", "b", "c"], "v": [1, 2, 3]})
    out = bt.from_arrow(source).groupby("g").map_groups(lambda b: b, batch_format=fmt).collect()
    assert _types(out) == {"g": "string", "s": "string", "v": "int64"}
    assert sorted(out.column("s").to_pylist()) == ["a", "b", "c"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_streamed_terminal_agrees_with_collect(fmt):
    """`iter_batches` is a different execution route, so the restore has to hold there too.

    `CLAUDE.md` names ``{collect, iter_batches}`` x ``{nulls, empty}`` as the cross-product a
    green suite keeps missing; a fix wired into only the materializing path is exactly the
    shape of miss it warns about.
    """
    source = pa.table(
        {"i": pa.array([1, 2, 3], type=pa.int64()), "s": pa.array([None] * 3, type=pa.string())}
    )
    dataset = bt.from_arrow(source).map_batches(lambda b: b, batch_format=fmt)
    collected = [str(t) for t in dataset.collect().schema.types]
    streamed = [[str(t) for t in b.schema.types] for b in dataset.iter_batches()]
    assert collected == ["int64", "string"]
    assert all(types == collected for types in streamed), streamed
