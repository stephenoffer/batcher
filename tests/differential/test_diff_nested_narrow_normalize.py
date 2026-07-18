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
from batcher.config import Config, ExecutionConfig, config_context
from conftest import assert_same

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


def test_nested_list_narrow_int_widens(duck):
    """A narrow int inside a list widens, so list-element arithmetic does not wrap."""
    t = pa.table({"l": pa.array([[2_000_000_000, 1]], pa.list_(pa.int32()))})
    d = bt.from_arrow(t)
    assert d.schema.field("l").type == pa.list_(pa.field("item", pa.int64()))
    out = d.select(r=bt.col("l").list.get(0) + bt.col("l").list.get(0)).collect()
    duck.register("t", t)
    # DuckDB list indexing is 1-based; list.get(0) selects the same (first) element.
    assert_same(out, duck.sql("SELECT CAST(l[1] AS BIGINT) + CAST(l[1] AS BIGINT) AS r FROM t"))


def test_nested_list_narrow_float_widens():
    """A narrow float inside a list widens to float64 in the declared schema."""
    t = pa.table({"l": pa.array([[1.5, 2.5]], pa.list_(pa.float32()))})
    assert bt.from_arrow(t).schema.field("l").type == pa.list_(pa.field("item", pa.float64()))


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
