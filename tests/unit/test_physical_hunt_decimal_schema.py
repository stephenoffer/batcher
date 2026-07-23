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
    inference must stay silent (fall back) rather than guess a wrong precision."""
    from batcher.plan.expr_ir import Binary, Col
    from batcher.plan.types.infer import infer_type

    ds = bt.from_arrow(_decimal_table(10, 2, 8, 3))
    inp = ds._plan.available_schema()  # type: ignore[attr-defined]
    assert inp is not None
    assert infer_type(Binary("div", Col("a"), Col("b")), inp) is None
    # `mod` of two decimals is likewise left uncertain (not add/sub/mul).
    assert infer_type(Binary("mod", Col("a"), Col("b")), inp) is None


def test_decimal_mixed_with_int_stays_uncertain() -> None:
    """A decimal mixed with a non-decimal operand is left uncertain — only the
    two-decimal `add`/`sub`/`mul` result types are reproduced."""
    from batcher.plan.expr_ir import Binary, Col
    from batcher.plan.types.infer import infer_type

    T = pa.table(
        {
            "a": pa.array([Decimal("1.50")], pa.decimal128(10, 2)),
            "i": pa.array([3], pa.int64()),
        }
    )
    inp = bt.from_arrow(T)._plan.available_schema()  # type: ignore[attr-defined]
    assert inp is not None
    assert infer_type(Binary("add", Col("a"), Col("i")), inp) is None
