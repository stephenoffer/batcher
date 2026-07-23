"""Floor division (``//``) matches Python/Polars, stays Int64, and NULLs on a zero divisor.

**DuckDB is deliberately NOT the oracle for the rounding direction here.** Batcher documents
Polars/Python ``//`` semantics — round toward *negative infinity*, so ``-7 // 3 == -3`` —
whereas DuckDB's ``/`` on integers truncates toward zero and gives ``-2``. The oracle for
direction is therefore Python's own ``//`` (cross-checked against Polars where it is
installed). DuckDB is still used as the oracle for the properties the two systems *do* agree
on: a zero divisor yields NULL, and NULLs propagate.

These pin the two bugs that motivated making ``floor_div`` a real IR op rather than sugar for
``floor(a / b)``: the Float64 round-trip silently lost precision above 2^53 (``i64::MAX // 3``
came back as ``3.07e18``) and turned a zero divisor into ``inf``/``nan`` instead of NULL.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

polars = pytest.importorskip("polars")


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": pa.array([7, -7, 7, -7, 6, -6, 0], type=pa.int64()),
            "b": pa.array([3, 3, -3, -3, 3, 3, 5], type=pa.int64()),
            "f": pa.array([7.0, -7.0, 7.5, -7.5, 6.0, -6.0, 0.0], type=pa.float64()),
        }
    )
    duck.register("t", tbl)
    return bt.from_arrow(tbl)


def test_all_sign_combinations_match_python_floordiv(t):
    """Every sign combination rounds toward negative infinity, as Python's ``//`` does."""
    out = t.select(q=col("a") // col("b")).collect().to_pydict()["q"]
    a, b = [7, -7, 7, -7, 6, -6, 0], [3, 3, -3, -3, 3, 3, 5]
    assert out == [x // y for x, y in zip(a, b, strict=True)]
    # The truncating (DuckDB/SQL) answer differs on exactly the mixed-sign, inexact
    # rows — assert we did NOT produce it, so a regression to truncation is caught.
    assert out[1] == -3 and out[2] == -3


def test_direction_deliberately_diverges_from_duckdb(t, duck):
    """Batcher's ``//`` rounds down; DuckDB's ``//`` truncates. This divergence is intended.

    Pinned so the difference stays a visible, deliberate decision rather than being
    "fixed" into DuckDB's truncating direction by a later change. ``Expr.__floordiv__``
    documents Python/Polars semantics, so Python/Polars — not DuckDB — is the oracle.
    """
    ours = t.select(q=col("a") // col("b")).collect().to_pydict()["q"]
    theirs = [r[0] for r in duck.sql("SELECT a // b AS q FROM t").fetchall()]
    # They agree wherever the division is exact, and differ on the inexact mixed-sign rows.
    assert ours[1] == -3 and theirs[1] == -2
    assert ours[2] == -3 and theirs[2] == -2
    assert ours[4] == theirs[4] == 2


def test_result_dtype_is_int64_for_int_operands(t):
    """Int64 ``//`` Int64 stays Int64 — the float desugaring returned Float64."""
    tbl = t.select(q=col("a") // col("b")).collect()
    assert tbl.schema.field("q").type == pa.int64()


def test_exact_above_2_pow_53(duck):
    """Past 2^53 a Float64 round-trip loses precision; the integer op is exact."""
    tbl = pa.table(
        {
            "a": pa.array(
                [9223372036854775807, 9007199254740993, 9007199254740992, -9223372036854775808],
                type=pa.int64(),
            )
        }
    )
    duck.register("big", tbl)
    ds = bt.from_arrow(tbl)
    out = ds.select(q=col("a") // 3).collect()
    assert out.schema.field("q").type == pa.int64()
    assert out.to_pydict()["q"] == [x // 3 for x in tbl.to_pydict()["a"]]
    # The specific value the old float path got wrong.
    assert out.to_pydict()["q"][0] == 3074457345618258602


def test_matches_polars(t):
    """Polars is the reference implementation for the documented ``//`` direction."""
    tbl = t.select(q=col("a") // col("b")).collect()
    pl_out = (
        polars.from_arrow(pa.table({"a": t.collect()["a"], "b": t.collect()["b"]}))
        .select((polars.col("a") // polars.col("b")).alias("q"))
        .to_series()
        .to_list()
    )
    assert tbl.to_pydict()["q"] == pl_out


def test_zero_divisor_is_null_like_duckdb(duck):
    """A zero divisor yields NULL, not ``inf``/``nan``. DuckDB agrees here, so it is the oracle."""
    tbl = pa.table(
        {
            "a": pa.array([7, -7, 0], type=pa.int64()),
            "z": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    duck.register("z", tbl)
    ds = bt.from_arrow(tbl)
    out = ds.select(q=col("a") // col("z")).collect()
    assert out.to_pydict()["q"] == [None, None, None]
    # DuckDB's `//` agrees that a zero divisor is NULL (its *direction* truncates and so
    # differs from ours, but every row here is NULL, so the two coincide exactly).
    # Note `/` is true division in DuckDB and yields inf/-inf/nan — not the oracle here.
    assert_same(out, duck.sql("SELECT a // z AS q FROM z"))


def test_nulls_propagate(duck):
    """A NULL on either side yields NULL, as for every other arithmetic op."""
    tbl = pa.table(
        {
            "a": pa.array([None, 7, None, 9], type=pa.int64()),
            "b": pa.array([3, None, None, 2], type=pa.int64()),
        }
    )
    duck.register("n", tbl)
    ds = bt.from_arrow(tbl)
    out = ds.select(q=col("a") // col("b")).collect()
    assert out.to_pydict()["q"] == [None, None, None, 4]


def test_float_operands_are_ieee_floor(t):
    """Float ``//`` is ``floor(a / b)``, so ``-7.5 // 2`` is ``-4.0``, matching Python."""
    out = t.select(q=col("f") // 2.0).collect()
    assert out.schema.field("q").type == pa.float64()
    f = [7.0, -7.0, 7.5, -7.5, 6.0, -6.0, 0.0]
    assert out.to_pydict()["q"] == [x // 2.0 for x in f]


def test_reflected_floordiv_scalar_over_column(t):
    """``scalar // expr`` (``__rfloordiv__``) uses the same op and direction."""
    out = t.select(q=-7 // col("b")).collect()
    assert out.schema.field("q").type == pa.int64()
    assert out.to_pydict()["q"] == [-7 // y for y in [3, 3, -3, -3, 3, 3, 5]]


def test_jit_eligible_expression_still_matches_interpreter(duck):
    """A ``//`` inside an otherwise JIT-friendly arithmetic expression falls back correctly.

    The Cranelift JIT does not compile ``floor_div`` (its ``sdiv`` truncates), so the whole
    expression must fall back to the interpreter rather than silently miscompile.
    """
    n = 5000
    a = [i - n // 2 for i in range(n)]
    tbl = pa.table({"a": pa.array(a, type=pa.int64())})
    duck.register("j", tbl)
    ds = bt.from_arrow(tbl)
    out = ds.select(q=(col("a") * 3 + 1) // 7).collect()
    assert out.schema.field("q").type == pa.int64()
    assert out.to_pydict()["q"] == [(x * 3 + 1) // 7 for x in a]


def test_floordiv_survives_filter_and_group_by(t, duck):
    """``//`` composes through the optimizer (filter + group-by), still matching Python."""
    ds = t.filter(col("a") != 0).group_by(g=col("a") // col("b")).agg(n=col("a").count())
    a, b = [7, -7, 7, -7, 6, -6], [3, 3, -3, -3, 3, 3]
    expected: dict[int, int] = {}
    for x, y in zip(a, b, strict=True):
        expected[x // y] = expected.get(x // y, 0) + 1
    got = ds.collect().to_pydict()
    assert dict(zip(got["g"], got["n"], strict=True)) == expected
