"""Opt-in output re-narrowing — Int32-in stays Int32-out, losslessly.

The FFI widens narrow numerics (Int8/16/32, Float16/32) to Int64/Float64 on input
so kernels stay on two paths. With ``shrink_output_dtypes`` on, a column that passes
a narrow *source* column through unchanged is cast back to its source width on the
way out — halving the footprint of pass-through id/feature columns. It is off by
default so the physical schema matches the statically-inferred ``Dataset.schema``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import Config, ExecutionConfig, config_context

pytestmark = pytest.mark.integration


def _narrow_table():
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int32()),
            "feat": pa.array([1.5, 2.5, 3.5], pa.float32()),
            "big": pa.array([10, 20, 30], pa.int64()),
        }
    )


def _with_shrink(on: bool):
    return config_context(Config().replace(execution=ExecutionConfig(shrink_output_dtypes=on)))


def test_default_widens_output():
    # Off by default: the engine widens narrow numerics and leaves them widened.
    out = bt.from_arrow(_narrow_table()).select("id", "feat").collect()
    assert out.schema.field("id").type == pa.int64()
    assert out.schema.field("feat").type == pa.float64()


def test_shrink_restores_source_width():
    with _with_shrink(True):
        out = bt.from_arrow(_narrow_table()).select("id", "feat", "big").collect()
    # Pass-through narrow columns return at their source width; values are identical.
    assert out.schema.field("id").type == pa.int32()
    assert out.schema.field("feat").type == pa.float32()
    assert out.schema.field("big").type == pa.int64()  # already wide, unchanged
    assert out.to_pydict()["id"] == [1, 2, 3]
    assert out.to_pydict()["feat"] == pytest.approx([1.5, 2.5, 3.5])


def test_shrink_is_lossless_on_overflow():
    # A computed column whose values exceed int32 must NOT be re-narrowed, even if it
    # is aliased to a source column name — the representability check keeps it wide.
    t = pa.table({"id": pa.array([2_000_000_000, 2_000_000_000], pa.int32())})
    with _with_shrink(True):
        # id + id overflows int32 (4e9), aliased back to "id".
        out = bt.from_arrow(t).select(id=bt.col("id") + bt.col("id")).collect()
    assert out.schema.field("id").type == pa.int64()
    assert out.to_pydict()["id"] == [4_000_000_000, 4_000_000_000]


def test_shrink_filtered_passthrough_stays_narrow():
    # Filtering preserves values, so a filtered Int32 column is still re-narrowable.
    with _with_shrink(True):
        out = bt.from_arrow(_narrow_table()).filter(bt.col("id") > 1).select("id").collect()
    assert out.schema.field("id").type == pa.int32()
    assert out.to_pydict()["id"] == [2, 3]
