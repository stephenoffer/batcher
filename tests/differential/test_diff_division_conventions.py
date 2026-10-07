"""The division and remainder conventions, and the eager CASE, pinned against DuckDB.

AP-187: ``%`` is SQL's truncated remainder and matches DuckDB exactly. ``//`` is Python's
floor division and deliberately does **not** match DuckDB's ``//``, which truncates on
integers; the two agree whenever the operands share a sign. The floor rule is pinned against
DuckDB's ``floor(a / b)``, and the documented explicit spellings of the other convention are
pinned against DuckDB too, so neither operator can drift silently.

AP-188: Batcher evaluates every CASE branch on every row, so a cast that fails on a row the
condition excludes still raises, where DuckDB returns NULL. That divergence is recorded as a
strict xfail: it starts failing loudly (XPASS) the day CASE becomes lazy, which is the cue to
delete the marker and the docs caveat in `type-system.md`. `try_cast` is the documented safe
idiom and is checked to agree with DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

SIGNED = pa.table(
    {
        "i": pa.array(range(9), pa.int64()),
        "a": pa.array([-7, 7, -7, 7, 0, -1, 6, None, 9], pa.int64()),
        "b": pa.array([2, -2, -2, 2, 3, 5, 3, 2, 0], pa.int64()),
    }
)


@pytest.fixture
def signed(duck):
    duck.register("t", SIGNED)
    return bt.from_arrow(SIGNED)


def test_remainder_is_truncated_like_duckdb(duck, signed):
    """``%`` takes the dividend's sign; a zero divisor or a null operand is null."""
    out = signed.select("i", r=bt.col("a") % bt.col("b")).collect()
    assert_same(out, duck.sql("SELECT i, a % b AS r FROM t"))


def test_floor_division_rounds_toward_negative_infinity(duck, signed):
    """``//`` is ``floor(a / b)``, integer-typed, with a zero divisor giving null."""
    out = signed.select("i", r=bt.col("a") // bt.col("b")).collect()
    assert out.schema.field("r").type == pa.int64()
    assert_same(
        out,
        duck.sql("SELECT i, CASE WHEN b <> 0 THEN floor(a / b)::BIGINT END AS r FROM t"),
    )


def test_floor_division_differs_from_duckdb_only_on_mixed_signs(duck, signed):
    """The recorded divergence: DuckDB's integer ``//`` truncates.

    Rows where exactly one operand is negative and the division is inexact are the only ones
    that differ, and the documented spelling ``(a - a % b) // b`` reproduces DuckDB's
    answer on every row.
    """
    ours = signed.select("i", r=bt.col("a") // bt.col("b")).collect().to_pydict()["r"]
    theirs = [row[0] for row in duck.sql("SELECT a // b FROM t ORDER BY i").fetchall()]
    differing = [i for i, (x, y) in enumerate(zip(ours, theirs, strict=True)) if x != y]
    assert differing == [0, 1, 5]
    trunc = (bt.col("a") - bt.col("a") % bt.col("b")) // bt.col("b")
    assert_same(signed.select("i", r=trunc).collect(), duck.sql("SELECT i, a // b AS r FROM t"))


def test_documented_floor_remainder_satisfies_pythons_identity(duck, signed):
    """``a - (a // b) * b`` is Python's ``%``, so ``a == (a // b) * b + that`` holds."""
    floor_rem = bt.col("a") - (bt.col("a") // bt.col("b")) * bt.col("b")
    out = signed.select("i", r=floor_rem).collect()
    assert_same(
        out,
        duck.sql("SELECT i, CASE WHEN b <> 0 THEN a - floor(a / b)::BIGINT * b END AS r FROM t"),
    )


CASTS = pa.table({"i": pa.array([0, 1, 2], pa.int64()), "s": pa.array(["1", "x", None])})


@pytest.fixture
def casts(duck):
    duck.register("c", CASTS)
    return bt.from_arrow(CASTS)


@pytest.mark.xfail(
    strict=True,
    raises=bt.ExecutionError,
    reason=(
        "AP-188: CASE evaluates every branch eagerly, so a cast guarded by `when` still "
        "raises; DuckDB evaluates CASE lazily and returns NULL. Use try_cast."
    ),
)
def test_case_guarded_cast_matches_duckdb(duck, casts):
    guarded = bt.when(bt.col("s") != "x").then(bt.col("s").cast("int64"))
    out = casts.select("i", r=guarded).collect()
    assert_same(out, duck.sql("SELECT i, CASE WHEN s <> 'x' THEN s::BIGINT END AS r FROM c"))


def test_case_with_try_cast_matches_duckdb(duck, casts):
    """The documented safe idiom gives DuckDB's lazy-CASE answer."""
    guarded = bt.when(bt.col("s") != "x").then(bt.col("s").try_cast("int64"))
    out = casts.select("i", r=guarded).collect()
    assert_same(out, duck.sql("SELECT i, CASE WHEN s <> 'x' THEN s::BIGINT END AS r FROM c"))
