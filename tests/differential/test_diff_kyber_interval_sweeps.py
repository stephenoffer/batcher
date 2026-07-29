"""Dense sweeps proving the interval rewrites select exactly the rows they replace.

The example-based tests in `test_diff_kyber_*` pin the shape a rewrite produces and check a
handful of boundary rows against DuckDB. These sweeps check the *whole* boundary: every
bucket, every comparison operator, and a value grid fine enough that a half-open interval
closed on the wrong side, or shifted by one, cannot pass.

The oracle here is the engine's own kernel rather than DuckDB, and deliberately so. A bare
`floor(f)` projection is not a comparison, so no interval rule touches it — evaluating it
gives the value the predicate is *about*, and comparing that elementwise in Python is a
direct check of the rewrite rather than a restatement of it. (It also covers `rint`, which
DuckDB has no counterpart for.) Each sweep first asserts its Python model reproduces the
engine's kernel exactly, so a divergence cannot hide behind a wrong model.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt
from batcher import col, lit
from batcher.plan.expr_ir import Binary
from batcher.plan.expr_ir.core import MathExpr

#: The six comparisons, and the Python predicate each one means.
_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}

#: A quarter-unit grid across six units either side of zero, plus both signed zeros. Fine
#: enough to land on and just off every integer and half-integer boundary the rounding
#: buckets are cut at.
_FLOAT_GRID = [round(-6 + 0.25 * i, 4) for i in range(49)] + [0.0, -0.0]

#: Python models of the engine's rounding functions. `round` is half-away-from-zero (the
#: engine's, and *not* Python's built-in `round`, which is half-to-even); `rint` is
#: half-to-even, which Python's built-in `round` does provide.
_ROUNDING_MODELS = {
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "round": lambda v: math.floor(abs(v) + 0.5) * (1 if v >= 0 else -1),
    "rint": round,
}


def _floats():
    return bt.from_pydict({"f": _FLOAT_GRID})


def _direct(dataset, expr):
    return dataset.select(r=expr).to_pydict()["r"]


@pytest.mark.parametrize("fn", sorted(_ROUNDING_MODELS))
def test_rounding_model_matches_the_engine_kernel(fn):
    """The sweep's oracle is only trustworthy if it is the engine's own function."""
    model = _ROUNDING_MODELS[fn]
    got = _direct(_floats(), MathExpr(fn, col("f")))
    assert got == [float(model(v)) for v in _FLOAT_GRID]


@pytest.mark.parametrize("fn", sorted(_ROUNDING_MODELS))
def test_rounding_intervals_select_exactly_the_rounded_rows(fn):
    dataset = _floats()
    direct = _direct(dataset, MathExpr(fn, col("f")))
    for bucket in range(-5, 6):
        for op, predicate in _OPS.items():
            got = _direct(dataset, Binary(op, MathExpr(fn, col("f")), lit(bucket)))
            want = [predicate(value, bucket) for value in direct]
            assert got == want, f"{fn}({op}) at bucket {bucket}"


def test_absolute_value_intervals_select_exactly_the_magnitude_rows():
    dataset = _floats()
    direct = _direct(dataset, MathExpr("abs", col("f")))
    for bound in range(1, 6):
        for op, predicate in _OPS.items():
            got = _direct(dataset, Binary(op, MathExpr("abs", col("f")), lit(bound)))
            want = [predicate(value, bound) for value in direct]
            assert got == want, f"abs({op}) at bound {bound}"


@pytest.mark.parametrize("divisor", [2, 3, 7])
def test_integer_bucket_intervals_select_exactly_the_bucket_rows(divisor):
    values = list(range(-30, 31))
    dataset = bt.from_pydict({"i": values})
    direct = _direct(dataset, col("i") // lit(divisor))
    # Floored division, so a negative dividend rounds *down*: -1 // 3 is -1, not 0. The
    # buckets below span both signs precisely to catch a rule that assumed truncation.
    assert direct == [value // divisor for value in values]
    for bucket in range(-6, 7):
        for op, predicate in _OPS.items():
            got = _direct(dataset, Binary(op, col("i") // lit(divisor), lit(bucket)))
            want = [predicate(value, bucket) for value in direct]
            assert got == want, f"//{divisor} ({op}) at bucket {bucket}"


def _even(value: float) -> float:
    """The engine's `even`: away from zero to the next even integer, and `0` for zero."""
    if value == 0:
        return 0.0
    magnitude = math.ceil(abs(value) / 2) * 2
    return float(magnitude if value > 0 else -magnitude)


def test_even_model_matches_the_engine_kernel():
    got = _direct(_floats(), MathExpr("even", col("f")))
    assert got == [_even(v) for v in _FLOAT_GRID]


def test_even_intervals_select_exactly_the_rounded_rows():
    """`even`'s buckets are two units wide and only the even ones are inhabited.

    Odd buckets are swept too: they name an empty set, the rule declines them, and the
    engine's own answer has to come back unchanged.
    """
    dataset = _floats()
    direct = _direct(dataset, MathExpr("even", col("f")))
    for bucket in range(-6, 7):
        for op, predicate in _OPS.items():
            got = _direct(dataset, Binary(op, MathExpr("even", col("f")), lit(bucket)))
            want = [predicate(value, bucket) for value in direct]
            assert got == want, f"even({op}) at bucket {bucket}"


def test_sign_comparisons_select_exactly_the_signed_rows():
    values = list(range(-8, 9))
    dataset = bt.from_pydict({"i": values})
    direct = _direct(dataset, MathExpr("sign", col("i")))
    assert direct == [float((v > 0) - (v < 0)) for v in values]
    for bound in (-1, 0, 1):
        for op, predicate in _OPS.items():
            got = _direct(dataset, Binary(op, MathExpr("sign", col("i")), lit(bound)))
            want = [predicate(value, bound) for value in direct]
            assert got == want, f"sign({op}) at {bound}"
