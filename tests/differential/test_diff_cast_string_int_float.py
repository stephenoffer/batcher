"""CAST string<->int and float->string edge cases vs DuckDB.

`CAST('1.5' AS BIGINT)` and `CAST('1e3' AS BIGINT)` used to **error** (strict) or return
**NULL** (try) — data loss on the safe-ingest path — because arrow's integer parser rejects a
non-integer string, where DuckDB parses and rounds half-away (`'1.5'->2`, `'1e3'->1000`).
And `CAST(nan AS VARCHAR)` rendered arrow's `NaN` where DuckDB renders `nan`. A `-0.0`
keeps its sign, which is the one place DuckDB's literal path and its column path disagree
with each other; see `test_float_to_string_keeps_the_sign_of_a_negative_zero`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, duck_materialize

pytestmark = pytest.mark.differential


def test_string_to_int_parses_fractional_and_scientific(duck):
    """A fractional / scientific string casts to the half-away-rounded integer, like DuckDB."""
    vals = ["1.5", "2.5", "-2.5", "0.5", "1e3", "12345.678", "42"]
    t = pa.table({"s": pa.array(vals, pa.string())})
    out = bt.from_arrow(t).select(i=bt.col("s").cast("int64")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT CAST(s AS BIGINT) AS i FROM t"))


def test_float_to_string_renders_nan_like_duckdb(duck):
    """A float NaN renders as `nan` (not arrow's `NaN`), matching DuckDB, for ordinary values."""
    t = pa.table({"f": pa.array([float("nan"), 0.5, 1.5, -3.25, 100.0], pa.float64())})
    duck_materialize(duck, "t", t)
    out = bt.from_arrow(t).select(s=bt.col("f").cast("string")).collect()
    assert_same(out, duck.sql("SELECT CAST(f AS VARCHAR) AS s FROM t"))


def test_float_to_string_keeps_the_sign_of_a_negative_zero():
    """A column `-0.0` renders as `-0.0`, the way every other engine renders it.

    Pinned directly rather than through the differential oracle, because DuckDB is
    internally inconsistent: it folds the sign away for a *literal* (`SELECT CAST(-0.0 AS
    VARCHAR)` → `0.0`, done by the parser), but an arrow-scanned `-0.0` keeps its sign — as
    do Polars, Arrow's own formatter, and Python's `str`.

    The engine does fold `-0.0` to `0.0` for *key identity* (grouping, joins, ordering).
    That is about which rows are equal, not about how a value is displayed: `sign(x)` and
    `1 / x` still tell the two apart, so rendering the sign away would lose information the
    value still carries. `crates/bc-expr/src/eval/cast.rs::float_to_string` is the other
    half of this contract.
    """
    t = pa.table({"f": pa.array([-0.0, 0.0], pa.float64())})
    out = bt.from_arrow(t).select(s=bt.col("f").cast("string")).collect()
    assert out.column("s").to_pylist() == ["-0.0", "0.0"]
