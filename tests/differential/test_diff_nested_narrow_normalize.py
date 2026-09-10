"""Differential tests: FFI narrow-type normalization must recurse into nested types.

The boundary widens a narrow numeric (Int8/16/32 → Int64, Float16/32 → Float64) so the
engine's kernels stay on Int64/Float64. That widening must reach a narrow numeric buried in
a ``struct``/``list``/``map``, or later arithmetic on the nested field wraps: an ``int32``
``2_000_000_000 + 2_000_000_000`` silently becomes ``-294967296`` where the same value as a
top-level column widens to ``int64`` and gives ``4_000_000_000``.

Two sides must agree — the Rust ``normalize_batch`` (widens the data) and the Python type
inference (``plan/types/lattice.py::widen`` and the ``InMemorySource`` schema), so
``Dataset.schema`` matches what the engine produces. The DuckDB oracle casts the nested field
to ``BIGINT`` to encode the engine's documented widening (DuckDB otherwise *errors* on INT32
overflow rather than wrapping).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.config import Config, ExecutionConfig, config_context

pytestmark = pytest.mark.differential


def test_nested_struct_narrow_int_widens(duck):
    """A narrow int inside a struct widens, so struct-field arithmetic does not wrap."""
    t = pa.table({"s": pa.array([{"a": 2_000_000_000}], pa.struct([("a", pa.int32())]))})
    d = bt.from_arrow(t)
    # The declared schema must report the widened nested type the engine actually produces.
    assert d.schema.field("s").type == pa.struct([("a", pa.int64())])
    out = d.select(r=bt.col("s").struct.field("a") + bt.col("s").struct.field("a")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT CAST(s.a AS BIGINT) + CAST(s.a AS BIGINT) AS r FROM t"))


def test_nested_list_narrow_int_keeps_its_width_but_the_element_still_widens(duck):
    """A narrow int inside a list keeps its width; the op that makes it a *column* widens it.

    The wrap this file is about is real and unchanged — `2_000_000_000 + 2_000_000_000` must be
    4e9 and not -294,967,296 — but it is paid one cast per row at `list.get` rather than 8x the
    bytes on every element of a decoded-image tensor. Both halves are asserted here, because
    either alone is satisfiable by the behaviour this change replaced: the column stays
    `int32`, and the arithmetic over it still agrees with DuckDB.
    """
    t = pa.table({"l": pa.array([[2_000_000_000, 1]], pa.list_(pa.int32()))})
    d = bt.from_arrow(t)
    assert d.schema.field("l").type == pa.list_(pa.field("item", pa.int32()))
    assert d.collect().schema.field("l").type.value_type == pa.int32(), "declared must match"
    got = d.select(r=bt.col("l").list.get(0))
    assert got.schema.field("r").type == pa.int64(), "the element becomes a column, so it widens"
    out = d.select(r=bt.col("l").list.get(0) + bt.col("l").list.get(0)).collect()
    assert out.column("r").to_pylist() == [4_000_000_000], "a wrap would give -294967296"
    duck.register("t", t)
    # DuckDB list indexing is 1-based; list.get(0) selects the same (first) element.
    assert_same(out, duck.sql("SELECT CAST(l[1] AS BIGINT) + CAST(l[1] AS BIGINT) AS r FROM t"))


def test_a_narrow_int_list_column_matches_duckdbs_own_element_width(duck):
    """DuckDB is the oracle for the width too, exactly as it is for the float case below."""
    t = pa.table({"l": pa.array([[1, 2], [3, 4]], pa.list_(pa.int32()))})
    produced = bt.from_arrow(t).collect()
    duck.register("t", t)
    oracle = duck.sql("SELECT l FROM t").arrow().schema.field("l").type.value_type
    assert produced.schema.field("l").type.value_type == oracle
    assert_same(produced, duck.sql("SELECT l FROM t"))


def test_nested_list_narrow_float_keeps_its_width_like_duckdb(duck):
    """A narrow float inside a list does **not** widen — and DuckDB is the reason it may not.

    This is the one nested case that is not about wrap. The file's whole argument is the
    `int32` overflow above, and it is an integer argument: a `float32` does not wrap, so the
    only thing widening its child bought was bytes. A list of floats is a tensor — an
    embedding, a decoded image, a feature vector — and widening it doubled the column and
    forced a full cast on a path that is otherwise zero-copy (measured: 1,476 ms against
    0.9 ms on 500,000 x 512 `f32`).

    The previous version of this test asserted `float64` and consulted **no oracle**, which
    is what let it read as a differential guarantee when it was a note about the boundary.
    Asked directly, DuckDB reports `FLOAT[]` for this column, so the widening was the
    divergence and this is the agreement. The two integer cases above keep their oracle and
    their behaviour unchanged.
    """
    t = pa.table({"l": pa.array([[1.5, 2.5]], pa.list_(pa.float32()))})
    d = bt.from_arrow(t)
    # Declared and produced must agree, or `Dataset.schema` lies.
    assert d.schema.field("l").type == pa.list_(pa.field("item", pa.float32()))
    produced = d.collect()
    assert produced.schema.field("l").type.value_type == pa.float32()

    duck.register("t", t)
    relation = duck.sql("SELECT l FROM t")
    # The element *width* is the claim; the element field's NAME is a producer convention
    # (`item` here, `l` from DuckDB) and comparing whole list types would fail on that alone.
    oracle_child = relation.arrow().schema.field("l").type.value_type
    assert produced.schema.field("l").type.value_type == oracle_child
    assert_same(produced, duck.sql("SELECT l FROM t"))


def test_nested_struct_widens_under_shrink_output_dtypes(duck):
    """The Rust boundary recursion is the backstop when Python pre-widening is disabled.

    ``shrink_output_dtypes`` turns off ``InMemorySource``'s pre-widening, so the nested narrow
    column reaches the engine un-widened and only ``normalize_batch`` (recursing) prevents the
    int32 wrap. This case wraps until the Rust side is rebuilt with the recursive normalize.
    """
    cfg = Config().replace(execution=ExecutionConfig(shrink_output_dtypes=True))
    t = pa.table({"s": pa.array([{"a": 2_000_000_000}], pa.struct([("a", pa.int32())]))})
    with config_context(cfg):
        out = (
            bt.from_arrow(t)
            .select(r=bt.col("s").struct.field("a") + bt.col("s").struct.field("a"))
            .collect()
        )
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT CAST(s.a AS BIGINT) + CAST(s.a AS BIGINT) AS r FROM t"))
