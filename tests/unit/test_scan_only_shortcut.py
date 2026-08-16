"""A bare scan skips the engine — and must return exactly what the engine would have.

`read_parquet(path).collect()` optimizes to a plan that is a single `Scan`. The reader has
already decoded the files and applied the pushed projection, so its batches *are* the result:
handing them back to Rust to pass through a no-op operator and exporting them again costs a
full round trip of the data across the FFI boundary — measured at 189 ms on a 709 ms read of
a 1.6 GB file, a quarter of the wall clock spent accomplishing nothing.

`core.scan_only_result` skips it. The whole risk of that is *divergence*: a result that took
the shortcut must be indistinguishable from one that did not, or `read().collect()` and
`read().filter(...).collect()` would disagree about a column's dtype on the same file. These
tests hold the two paths against each other on the shapes where they could drift — narrow
integers and floats (which the FFI boundary silently widens), projections (which a Parquet
reader returns in *file* order, not the order asked for), nulls, and empties.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.core import scan_only_result
from batcher.plan.expr_ir import col


@pytest.fixture
def source(tmp_path):
    """A file whose every column type the boundary treats differently."""
    table = pa.table(
        {
            "i8": pa.array([1, -2, None], pa.int8()),
            "i32": pa.array([1, 2, 3], pa.int32()),
            "u16": pa.array([1, 2, 3], pa.uint16()),
            "i64": pa.array([10, 20, 30], pa.int64()),
            "f32": pa.array([1.5, None, 3.5], pa.float32()),
            "f64": pa.array([1.5, 2.5, 3.5], pa.float64()),
            "s": pa.array(["a", None, "c"]),
            "b": pa.array([True, False, None]),
        }
    )
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _engine_result(path: str) -> pa.Table:
    """The same rows, forced through the engine — an always-true filter it cannot fold away."""
    return bt.read.parquet(path).filter(col("i64") > -(10**9)).collect()


def test_the_shortcut_matches_the_engine_exactly(source) -> None:
    shortcut = bt.read.parquet(source).collect()
    engine = _engine_result(source)

    assert shortcut.schema.equals(engine.schema, check_metadata=False), (
        f"schema drift:\n  shortcut {shortcut.schema}\n  engine   {engine.schema}"
    )
    assert shortcut.to_pydict() == engine.to_pydict()


def test_narrow_numerics_are_widened_the_way_the_boundary_widens_them(source) -> None:
    """Int8/16/32 and every unsigned int → int64; Float32 → float64. Same as `bc_py::widen_to`."""
    got = bt.read.parquet(source).collect()
    types = {f.name: str(f.type) for f in got.schema}
    assert types["i8"] == "int64"
    assert types["i32"] == "int64"
    assert types["u16"] == "int64"
    assert types["f32"] == "double"
    # And the untouched ones stay untouched.
    assert types["i64"] == "int64"
    assert types["s"] == "string"
    assert types["b"] == "bool"


def test_a_projection_comes_back_in_the_order_asked_for(source) -> None:
    """A Parquet reader returns leaves in *file* order; the plan's order has to win."""
    got = bt.read.parquet(source).select("s", "i8", "f64").collect()
    assert got.column_names == ["s", "i8", "f64"]
    assert got.column("i8").to_pylist() == [1, -2, None]


def test_nulls_survive(source) -> None:
    got = bt.read.parquet(source).collect().to_pydict()
    assert got["i8"] == [1, -2, None]
    assert got["s"] == ["a", None, "c"]
    assert got["b"] == [True, False, None]


def test_an_empty_file_keeps_its_schema(tmp_path) -> None:
    table = pa.table({"a": pa.array([], pa.int32()), "b": pa.array([], pa.string())})
    path = str(tmp_path / "e.parquet")
    pq.write_table(table, path)

    got = bt.read.parquet(path).collect()
    assert got.num_rows == 0
    assert got.column_names == ["a", "b"]
    assert str(got.schema.field("a").type) == "int64"  # widened, empty or not


# --------------------------------------------------------------------------------------
# When the shortcut must decline
# --------------------------------------------------------------------------------------


def test_the_shortcut_declines_anything_that_is_not_a_bare_scan(source) -> None:
    """One operator above the scan is real work; the engine has to do it."""
    from batcher.io.source import read_source

    ds = bt.read.parquet(source).filter(col("i64") > 15)
    resolved = [read_source(s, None, None) for s in ds._sources]
    assert scan_only_result(ds._plan, resolved) is None


def test_the_shortcut_declines_when_a_predicate_was_pushed_to_the_source(source) -> None:
    """A pushed predicate prunes *row groups* — a superset. The engine still has to filter."""
    from batcher.io.source import read_source

    ds = bt.read.parquet(source)
    resolved = [read_source(s, None, None) for s in ds._sources]
    pushed = {0: {"e": "binary", "op": "gt", "left": {"e": "col", "name": "i64"}}}
    assert scan_only_result(ds._plan, resolved, pushed) is None
    # …and with no predicate it does apply.
    assert scan_only_result(ds._plan, resolved, {}) is not None


def test_a_filtered_read_still_agrees_with_the_shortcut_read(source) -> None:
    """The end-to-end invariant: which path a query took must not be observable."""
    everything = bt.read.parquet(source).collect()
    filtered = bt.read.parquet(source).filter(col("i64") > 15).collect()

    assert filtered.schema.equals(everything.schema, check_metadata=False)
    assert filtered.num_rows == 2


# --------------------------------------------------------------------------------------
# A pure column selection over the scan takes the same shortcut
# --------------------------------------------------------------------------------------


def test_a_column_selection_over_a_scan_takes_the_shortcut(source) -> None:
    """`SELECT a, b FROM <file>` is `Project(Scan)`, and the `Project` is an identity.

    Kyber pushes the column list into the reader, so the batches it hands back already hold
    exactly these columns — the projection above is a no-op and the round trip it forced was
    the whole cost. On 32M rows x 16 `int64` columns, selecting one: 97.7 ms against 33.7 ms
    for the same query written as `read(columns=...)`.
    """
    from batcher.io.source import read_source

    ds = bt.read.parquet(source).select("i64", "s")
    resolved = [read_source(s, None, None) for s in ds._sources]
    out = scan_only_result(ds._plan, resolved)
    assert out is not None
    assert out.column_names == ["i64", "s"]


def test_a_selection_that_is_not_an_identity_declines(source) -> None:
    """Everything past "the same columns under the same names" is real work.

    A rename is the sharp one: the reader's batches carry the *source* names and nothing on
    this path renames them, so taking the shortcut would return a column called `i64` where
    the plan promised `n`. The others are computation, which the engine has to do.
    """
    from batcher.io.source import read_source

    resolved = [read_source(s, None, None) for s in bt.read.parquet(source)._sources]
    for label, ds in [
        ("rename", bt.read.parquet(source).select(n=col("i64"))),
        ("computed", bt.read.parquet(source).select(i64=col("i64") + 1)),
        ("literal", bt.read.parquet(source).with_columns(k=bt.lit(1))),
        ("two operators", bt.read.parquet(source).select("i64").limit(1)),
    ]:
        assert scan_only_result(ds._plan, resolved) is None, label


def test_a_selection_agrees_with_the_engine_column_for_column(source) -> None:
    """The end-to-end invariant again, for the selection shape and in a non-file order.

    `i64` comes *after* `s` in the file, so asking for them the other way round is exactly
    the case where a shortcut that forgot to restate the plan's column order would return
    two correct columns under each other's names.
    """
    shortcut = bt.read.parquet(source).select("s", "i64").collect()
    engine = _engine_result(source).select(["s", "i64"])
    assert shortcut.column_names == ["s", "i64"]
    assert shortcut.schema.equals(engine.schema, check_metadata=False)
    assert shortcut.equals(engine)


def test_a_narrow_column_is_still_widened_under_a_selection(source) -> None:
    """The widening the FFI boundary applies must not depend on which path ran."""
    shortcut = bt.read.parquet(source).select("i8", "f32").collect()
    engine = _engine_result(source).select(["i8", "f32"])
    assert shortcut.schema.equals(engine.schema, check_metadata=False)
    assert shortcut.schema.field("i8").type == pa.int64()
    assert shortcut.schema.field("f32").type == pa.float64()
    assert shortcut.equals(engine)
