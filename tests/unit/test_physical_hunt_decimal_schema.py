"""Schema-inference completeness for decimal arithmetic.

`Dataset.schema` must report the same Arrow type the engine produces. Before this
fix, `infer_type` returned ``None`` for decimal ``+``/``-``/``*`` (two decimal128
operands), so the declared schema fell back to a zero-row probe that reported
``null`` — a schema lie: ``decimal(10,2) + decimal(8,3)`` is ``decimal(12,3)``, not
``null``. These tests pin the precision/scale the engine derives.
"""

from __future__ import annotations

from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _decimal_table(p1: int, s1: int, p2: int, s2: int) -> pa.Table:
    return pa.table(
        {
            "a": pa.array([Decimal(1).scaleb(-s1)], pa.decimal128(p1, s1)),
            "b": pa.array([Decimal(1).scaleb(-s2)], pa.decimal128(p2, s2)),
        }
    )


# (p1, s1, p2, s2, op) -> expected (precision, scale). Verified against the engine.
_CASES = [
    (10, 2, 8, 3, "add", (12, 3)),
    (10, 2, 8, 3, "sub", (12, 3)),
    (10, 2, 8, 3, "mul", (19, 5)),
    (5, 0, 5, 0, "add", (6, 0)),
    (5, 0, 5, 0, "mul", (11, 0)),
    (18, 4, 10, 2, "add", (19, 4)),
    (18, 4, 10, 2, "mul", (29, 6)),
    (10, 2, 3, 0, "add", (11, 2)),
    (38, 10, 38, 10, "add", (38, 10)),  # precision caps at 38
    (38, 10, 38, 10, "mul", (38, 20)),  # precision caps at 38, scale kept
    (20, 5, 20, 10, "mul", (38, 15)),
]


@pytest.mark.parametrize(("p1", "s1", "p2", "s2", "op", "expected"), _CASES)
def test_decimal_arith_declared_matches_actual(
    p1: int, s1: int, p2: int, s2: int, op: str, expected: tuple[int, int]
) -> None:
    ds = bt.from_arrow(_decimal_table(p1, s1, p2, s2))
    a, b = bt.col("a"), bt.col("b")
    expr = {"add": a + b, "sub": a - b, "mul": a * b}[op]
    declared = ds.select(v=expr).schema.field("v").type
    actual = ds.select(v=expr).collect().schema.field("v").type
    # The declared schema must not lie (this failed pre-fix: declared was `null`).
    assert declared.equals(actual)
    assert pa.types.is_decimal128(declared)
    assert (declared.precision, declared.scale) == expected


def test_decimal_div_stays_uncertain() -> None:
    """True division of decimals has an engine-derived scale we do not reproduce, so
    inference must stay silent (fall back) rather than guess a wrong precision.

    `mod` used to be silent for the same stated reason, and the reason was not true of it:
    a remainder is bounded by the smaller operand, which makes its result type exactly
    `min(p1 - s1, p2 - s2) + max(s1, s2)` at scale `max(s1, s2)`. That was measured against
    the engine before it was written down, so `mod` is now declared and `div` alone stays
    uncertain."""
    from batcher.plan.expr_ir import Binary, Col
    from batcher.plan.types.infer import infer_type

    ds = bt.from_arrow(_decimal_table(10, 2, 8, 3))
    inp = ds._plan.available_schema()  # type: ignore[attr-defined]
    assert inp is not None
    assert infer_type(Binary("div", Col("a"), Col("b")), inp) is None
    assert infer_type(Binary("mod", Col("a"), Col("b")), inp) == pa.decimal128(8, 3)
    # And it is the type the engine produces, not merely a rule that agrees with itself.
    produced = ds.select(v=bt.col("a") % bt.col("b")).collect().schema.field("v").type
    assert produced == pa.decimal128(8, 3)


def test_decimal_mixed_with_int_declares_the_decimal_the_engine_produces() -> None:
    """A decimal mixed with an integer used to be left uncertain, and need not be.

    The engine's rule for the pair is exact: `coerce_numeric` brings the integer to the
    *decimal operand's own type* and the two-decimal rule then applies, so
    `decimal(10,2) + int64` is `decimal(11,2)` and `decimal(10,2) * int64` is
    `decimal(21,4)` — scale doubled, because the coerced integer carries the decimal's
    scale too. Leaving it uncertain meant `Dataset.schema` reported `null` for `price *
    qty`, the most ordinary shape a money column appears in.

    A decimal mixed with a **float** is not a decimal at all: DOUBLE dominates DECIMAL, so
    the whole expression is a plain double.
    """
    from batcher.plan.expr_ir import Binary, Col
    from batcher.plan.types.infer import infer_type

    T = pa.table(
        {
            "a": pa.array([Decimal("1.50")], pa.decimal128(10, 2)),
            "i": pa.array([3], pa.int64()),
            "f": pa.array([2.5], pa.float64()),
        }
    )
    ds = bt.from_arrow(T)
    inp = ds._plan.available_schema()  # type: ignore[attr-defined]
    assert inp is not None
    assert infer_type(Binary("add", Col("a"), Col("i")), inp) == pa.decimal128(11, 2)
    assert infer_type(Binary("mul", Col("a"), Col("i")), inp) == pa.decimal128(21, 4)
    assert infer_type(Binary("add", Col("a"), Col("f")), inp) == pa.float64()
    # Each one against what the engine actually returns, which is the only thing that
    # makes the rule above a fact rather than a second guess.
    for expr, want in (
        (bt.col("a") + bt.col("i"), pa.decimal128(11, 2)),
        (bt.col("a") * bt.col("i"), pa.decimal128(21, 4)),
        (bt.col("a") + bt.col("f"), pa.float64()),
    ):
        assert ds.select(v=expr).collect().schema.field("v").type == want
