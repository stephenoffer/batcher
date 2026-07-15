"""Narrowing a wider float to `float32` must match DuckDB on finite overflow.

Arrow's `f64 -> f32` kernel silently rounds an out-of-range *finite* value (e.g.
``1e300``) to ``+inf``. DuckDB treats that as an overflow: strict ``CAST`` errors and
``TRY_CAST`` yields NULL — it never fabricates an infinity from a finite input. A
genuine ``inf`` input still passes through as ``inf`` on both engines. These pin the
`float32` cast path against that contract.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential


def test_try_cast_f64_to_f32_finite_overflow_is_null(duck):
    """`TRY_CAST(1e300 AS REAL)` → NULL (not ``inf``); a real ``inf`` stays ``inf``."""
    t = pa.table(
        {"x": pa.array([1.5, 1e300, -1e300, math.inf, -math.inf, None], pa.float64())}
    )
    duck.register("nf", t)
    out = bt.from_arrow(t).select(w=col("x").try_cast("float32")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(x AS REAL) w FROM nf"))


def test_strict_cast_f64_to_f32_finite_overflow_errors():
    """Strict `CAST(1e300 AS REAL)` errors rather than silently yielding ``+inf``."""
    t = pa.table({"x": pa.array([1e300], pa.float64())})
    with pytest.raises(Exception):  # noqa: B017 — engine raises on the overflow
        bt.from_arrow(t).select(w=col("x").cast("float32")).collect()


def test_strict_cast_f64_to_f32_in_range_ok(duck):
    """An in-range value still narrows cleanly under strict `CAST`."""
    t = pa.table({"x": pa.array([1.5, -2.25, 0.0, None], pa.float64())})
    duck.register("nf2", t)
    out = bt.from_arrow(t).select(w=col("x").cast("float32")).collect()
    assert_same(out, duck.sql("SELECT CAST(x AS REAL) w FROM nf2"))
