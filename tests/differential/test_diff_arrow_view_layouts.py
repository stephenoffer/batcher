"""Differential tests: the Arrow *view* and *run-end* layouts are ordinary columns.

`string_view`/`binary_view` and `run_end_encoded` are alternative physical spellings of
`string`/`binary` and of their value type. They carry the same values, so every query over
one must return what the same query over the plain spelling returns.

They are not exotic inputs. A Parquet reader with view types enabled, DuckDB's Arrow export,
Polars, and every Velox-backed producer hand them over by default, and Batcher used to fail
the *whole query* on one -- first in the control plane (`ArrowTypeError: Extracting byte
ranges not supported for type string_view`, raised out of statistics collection by a bare
`RecordBatch.nbytes`) and then in the engine (`Invalid comparison operation: Utf8View >
Utf8`). Both are boundary concerns, so the fix is at the boundary: `plan.types.logical_bytes`
reads the byte figure on a layout `nbytes` cannot walk, and `bc_py::normalize_to` /
`decode_run_ends` present one physical spelling to every operator.

The oracle is the plain-layout table: DuckDB has no `string_view` of its own to register, so
each test compares Batcher-over-the-view against DuckDB-over-the-equivalent, which is the
property that actually matters.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# The same four rows in both spellings: nulls, a duplicate, and an empty string, so the
# comparison covers the null bitmap and the short-value inlining a view array does.
_VALUES = ["aa", None, "", "aa"]
_NUMS = [1, 2, 3, 4]


def _view_table() -> pa.Table:
    return pa.table(
        {"s": pa.array(_VALUES, type=pa.string_view()), "n": pa.array(_NUMS, type=pa.int64())}
    )


def _plain_table() -> pa.Table:
    return pa.table(
        {"s": pa.array(_VALUES, type=pa.string()), "n": pa.array(_NUMS, type=pa.int64())}
    )


def test_string_view_declares_the_plain_type():
    """`Dataset.schema` reports what the engine produces, not the input spelling."""
    assert bt.from_arrow(_view_table()).schema.field("s").type == pa.string()


def test_binary_view_declares_the_plain_type():
    t = pa.table({"b": pa.array([b"x", None], type=pa.binary_view())})
    assert bt.from_arrow(t).schema.field("b").type == pa.binary()


def test_string_view_filter(duck):
    """A comparison against a view column -- the shape that raised `Utf8View > Utf8`."""
    duck.register("t", _plain_table())
    out = bt.from_arrow(_view_table()).filter(bt.col("s") > "a").collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s > 'a'"))


def test_string_view_group_by(duck):
    """Grouping keys on a view column, including the null group."""
    duck.register("t", _plain_table())
    out = bt.from_arrow(_view_table()).group_by("s").agg(c=bt.col("n").sum()).collect()
    assert_same(out, duck.sql("SELECT s, SUM(n) AS c FROM t GROUP BY s"))


def test_string_view_string_function(duck):
    """A string kernel over a view column."""
    duck.register("t", _plain_table())
    out = bt.from_arrow(_view_table()).select(u=bt.col("s").str.upper()).collect()
    assert_same(out, duck.sql("SELECT UPPER(s) AS u FROM t"))


def test_string_view_distinct(duck):
    duck.register("t", _plain_table())
    out = bt.from_arrow(_view_table()).select("s").distinct().collect()
    assert_same(out, duck.sql("SELECT DISTINCT s FROM t"))


def test_string_view_join(duck):
    """A view column as a join key against a plain-string one on the other side."""
    right = pa.table({"s": pa.array(["aa", "zz"], type=pa.string()), "m": pa.array([9, 8])})
    duck.register("t", _plain_table())
    duck.register("r", right)
    out = bt.from_arrow(_view_table()).join(bt.from_arrow(right), on="s", how="inner").collect()
    assert_same(out, duck.sql("SELECT t.s, t.n, r.m FROM t JOIN r USING (s)"))


def test_string_view_sort_is_ordered(duck):
    """Ordered, so it is compared as a sequence -- `assert_same` would not see a sort bug."""
    out = bt.from_arrow(_view_table()).sort("s", descending=True).collect()
    assert out.column("s").to_pylist() == ["aa", "aa", "", None]


def test_string_view_streams_identically():
    """`iter_batches` takes a different path out of the engine than `collect`."""
    d = bt.from_arrow(_view_table()).filter(bt.col("n") > 1)
    streamed = list(d.iter_batches())
    assert pa.Table.from_batches(streamed, schema=d.collect().schema).equals(d.collect())


def test_empty_string_view_relation():
    """An empty view column: the layout with no data buffer at all."""
    t = pa.table({"s": pa.array([], type=pa.string_view()), "n": pa.array([], type=pa.int64())})
    out = bt.from_arrow(t).filter(bt.col("s") > "a").collect()
    assert out.num_rows == 0
    assert out.column("s").type == pa.string()


def test_all_null_string_view():
    t = pa.table({"s": pa.array([None, None], type=pa.string_view()), "n": pa.array([1, 2])})
    out = bt.from_arrow(t).group_by("s").agg(c=bt.col("n").sum()).collect()
    assert out.column("c").to_pylist() == [3]


def _run_end_table() -> pa.Table:
    """`[10, 10, 20, 20]` stored as two runs, with a narrow (int16) value type."""
    ree = pa.RunEndEncodedArray.from_arrays(
        pa.array([2, 4], type=pa.int32()), pa.array([10, 20], type=pa.int16())
    )
    return pa.table({"r": ree, "n": pa.array(_NUMS, type=pa.int64())})


def _run_end_plain() -> pa.Table:
    return pa.table(
        {"r": pa.array([10, 10, 20, 20], type=pa.int64()), "n": pa.array(_NUMS, type=pa.int64())}
    )


def test_run_end_declares_its_widened_value_type():
    """Decoded *and* widened: the int16 value type lands on int64 like any narrow numeric."""
    assert bt.from_arrow(_run_end_table()).schema.field("r").type == pa.int64()


def test_run_end_group_by(duck):
    """Grouping on a run-end column -- the runs must expand before the keys are built."""
    duck.register("t", _run_end_plain())
    out = bt.from_arrow(_run_end_table()).group_by("r").agg(c=bt.col("n").sum()).collect()
    assert_same(out, duck.sql("SELECT r, SUM(n) AS c FROM t GROUP BY r"))


def test_run_end_passthrough_matches_its_declared_type(duck):
    """A column nothing touches still comes back decoded, or the schema would lie."""
    duck.register("t", _run_end_plain())
    out = bt.from_arrow(_run_end_table()).filter(bt.col("n") > 1).collect()
    assert out.column("r").type == pa.int64()
    assert_same(out, duck.sql("SELECT * FROM t WHERE n > 1"))


def test_run_end_slice_decodes_its_own_rows(duck):
    """A *sliced* run-end array indexes its parent's runs; the decode must respect the offset."""
    sliced = _run_end_table().slice(1, 2)
    duck.register("t", _run_end_plain().slice(1, 2).combine_chunks())
    out = bt.from_arrow(sliced).collect()
    assert out.column("r").to_pylist() == [10, 20]
    assert_same(out, duck.sql("SELECT * FROM t"))


# The *list*-view layouts are a harder case than `string_view` above, and fail differently.
# `string_view` crosses the FFI and is normalized in Rust; a `list_view` cannot cross at all
# -- arrow-rs's C Data Interface reader rejects it outright -- so the whole query died with
# `ArrowException: The datatype "+vl" is still not supported in Rust implementation` while
# `Dataset.schema` reported the `list` that `plan.types.widen` said it would arrive as. The
# respelling therefore happens on the Python side of the boundary (`plan.types.layout`).
_LISTS = [[1, 2], None, [], [3]]


def _list_view_table() -> pa.Table:
    return pa.table(
        {
            "v": pa.array(_LISTS, type=pa.list_view(pa.int64())),
            "n": pa.array(_NUMS, type=pa.int64()),
        }
    )


def _list_plain_table() -> pa.Table:
    return pa.table(
        {"v": pa.array(_LISTS, type=pa.list_(pa.int64())), "n": pa.array(_NUMS, type=pa.int64())}
    )


def test_list_view_declares_the_plain_type():
    assert bt.from_arrow(_list_view_table()).schema.field("v").type == pa.list_(pa.int64())


def test_list_view_passthrough_matches_its_declared_type(duck):
    """The shape that used to raise at the FFI: a list-view column merely carried along."""
    duck.register("t", _list_plain_table())
    out = bt.from_arrow(_list_view_table()).filter(bt.col("n") > 1).collect()
    assert out.column("v").type == pa.list_(pa.int64())
    assert_same(out, duck.sql("SELECT * FROM t WHERE n > 1"))


def test_list_view_length_kernel(duck):
    """A list kernel over the view spelling must read the same lengths as over `list`."""
    duck.register("t", _list_plain_table())
    out = bt.from_arrow(_list_view_table()).select(k=bt.col("v").list.len()).collect()
    assert_same(out, duck.sql("SELECT len(v) AS k FROM t"))


def test_list_view_sort_is_ordered():
    """Ordered, so it is compared as a sequence -- `assert_same` would not see a sort bug."""
    out = bt.from_arrow(_list_view_table()).sort("n", descending=True).collect()
    assert out.column("v").to_pylist() == [[3], [], None, [1, 2]]


def test_list_view_streams_identically():
    d = bt.from_arrow(_list_view_table()).filter(bt.col("n") > 1)
    streamed = list(d.iter_batches())
    assert pa.Table.from_batches(streamed, schema=d.collect().schema).equals(d.collect())


def test_large_list_view_declares_the_plain_type():
    t = pa.table({"v": pa.array([[1], None], type=pa.large_list_view(pa.int64()))})
    assert bt.from_arrow(t).schema.field("v").type == pa.list_(pa.int64())


def test_list_view_nested_in_a_struct(duck):
    """The layout is rejected at *any* depth, so a buried one has to be respelled too."""
    nested = pa.table(
        {
            "s": pa.array([{"a": [1]}, {"a": None}], pa.struct([("a", pa.list_view(pa.int64()))])),
            "n": pa.array([1, 2], type=pa.int64()),
        }
    )
    duck.register(
        "t",
        pa.table(
            {
                "s": pa.array([{"a": [1]}, {"a": None}], pa.struct([("a", pa.list_(pa.int64()))])),
                "n": pa.array([1, 2], type=pa.int64()),
            }
        ),
    )
    out = bt.from_arrow(nested).collect()
    assert out.column("s").type == pa.struct([("a", pa.list_(pa.int64()))])
    assert_same(out, duck.sql("SELECT * FROM t"))


def test_a_udf_may_return_a_list_view():
    """A UDF is where a layout the engine has never seen most often arrives.

    User code picks the type here rather than a source doing it, and a NumPy or Polars
    round-trip inside `fn` can hand back a view spelling. It used to reach the user as
    ``Extracting byte ranges not supported for type list_view<item: int64>``.
    """

    def fn(_batch):
        return pa.record_batch({"v": pa.array([[1], [2]], pa.list_view(pa.int64()))})

    out = bt.from_arrow(pa.table({"x": [1, 2]})).map_batches(fn).collect()
    assert out.column("v").type == pa.list_(pa.int64())
    assert out.column("v").to_pylist() == [[1], [2]]


def test_empty_list_view_relation():
    t = pa.table({"v": pa.array([], type=pa.list_view(pa.int64())), "n": pa.array([], pa.int64())})
    out = bt.from_arrow(t).filter(bt.col("n") > 1).collect()
    assert out.num_rows == 0
    assert out.column("v").type == pa.list_(pa.int64())
