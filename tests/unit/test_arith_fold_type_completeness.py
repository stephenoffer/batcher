"""Every member of `_FOLDABLE_MATH` folds to a literal of the type it started with.

`fold_math_of_int_literal` evaluates a unary math function over an integer literal at plan
time. Doing that changes the expression's *type* if the fold picks a different Python number
than the engine's kernel would produce, and a type change is invisible to almost everything
that would normally catch a bad rewrite: the values are equal, `assert_same` is int/float
tolerant, and `to_pydict()` comparisons are keyed by name. It surfaces only in the Arrow
schema.

That is not hypothetical. `sign`, `round` and `trunc` all folded an integer to a **float**
literal and retyped the column double, where the engine and DuckDB both answer an integer
type; `round` additionally lost precision past 2^53. They were fixed in `87d82730`.

The reason all three survived is a coverage gap rather than a hard problem. The existing
suite reaches this rule through `_fire`, whose helper asserts the rewrite preserves the
inferred type -- but only three of the eight foldable functions were ever passed to it
(`abs`, `sign`, `floor`). `ceil`, `rint`, `round`, `trunc` and `sqrt` were never type-checked
at the fold at all, and two of those five were wrong.

So this file does not add more spot checks. It derives its cases from `_FOLDABLE_MATH`
itself, which means a ninth foldable function cannot be added without either satisfying the
invariant or being classified deliberately. The classification table below is the second
half of that: it states which functions keep an integer and which promote, so a *change* in
that partition has to be written down rather than merely observed.

This is a control-plane test on purpose. It compares the fold against `infer_type`, the
engine's static type analysis, and executes nothing -- the value-level agreement with the
real kernels is `tests/differential/test_diff_int_math_type_and_exactness.py`'s job. Two
independent statements of the same rule, which is what makes their agreement evidence.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher.kyber.rules.extra.arith_extra as ax
from batcher.plan.expr_ir import col, lit
from batcher.plan.expr_ir.core import Lit, MathExpr
from batcher.plan.schema import SchemaRef
from batcher.plan.types.infer.dispatch import infer_type

pytestmark = pytest.mark.unit

#: Which foldable functions hand an integer back, and which promote it to Float64. Stated
#: rather than measured, so that flipping one is a deliberate edit to this line.
_KEEPS_INT = frozenset({"abs", "sign", "round", "trunc"})
_PROMOTES = frozenset({"ceil", "floor", "rint", "sqrt"})

#: Positive, so `sqrt` folds (it declines a negative operand) and every function is defined.
#: Its square root is exact, so no case here depends on float formatting.
_VALUE = 9

_SCHEMA = SchemaRef(arrow=pa.schema([("i", pa.int64())]))


def test_every_foldable_function_is_classified():
    """The completeness half: a new entry in `_FOLDABLE_MATH` fails here until it is placed.

    Without this, the parametrized tests below would silently skip a ninth function -- they
    derive their cases from the same set, so an unclassified addition would simply not be
    compared against anything.
    """
    assert _KEEPS_INT | _PROMOTES == ax._FOLDABLE_MATH
    assert not (_KEEPS_INT & _PROMOTES), "a function cannot both keep and promote"


@pytest.mark.parametrize("fn", sorted(ax._FOLDABLE_MATH))
def test_the_fold_preserves_the_inferred_type(fn):
    """The invariant, over every foldable function rather than the three that had a test.

    `_fold_math_lit` is the production folding function the rule applies; calling it
    directly keeps the case to the one thing being asserted.
    """
    original = MathExpr(fn=fn, input=lit(_VALUE))
    folded = ax._fold_math_lit(original)

    assert folded is not original, f"`{fn}` over an integer literal must fold"
    assert isinstance(folded, Lit)
    assert infer_type(folded, _SCHEMA) == infer_type(original, _SCHEMA), (
        f"folding `{fn}({_VALUE})` moved the expression's type"
    )


@pytest.mark.parametrize("fn", sorted(ax._FOLDABLE_MATH))
def test_the_folded_literal_matches_the_stated_classification(fn):
    """And the type it lands on is the one this file says it should, not merely a stable one.

    The test above would pass if the fold and `infer_type` were wrong in the same direction.
    This one holds both to an independently written answer.
    """
    expected = pa.int64() if fn in _KEEPS_INT else pa.float64()
    folded = ax._fold_math_lit(MathExpr(fn=fn, input=lit(_VALUE)))
    assert infer_type(folded, _SCHEMA) == expected


@pytest.mark.parametrize("fn", sorted(ax._FOLDABLE_MATH))
def test_the_classification_is_the_kernels_own_behaviour_on_a_column(fn):
    """The same partition must hold for an integer *column*, where no folding happens.

    This is what ties the table to the engine rather than to the rewrite: if `sign` of an
    Int64 column is Int64, then folding `sign(-7)` to a float is wrong regardless of what
    any rule prefers. It is also the assertion that would have failed first when the
    kernels changed, had it existed.
    """
    expected = pa.int64() if fn in _KEEPS_INT else pa.float64()
    assert infer_type(MathExpr(fn=fn, input=col("i")), _SCHEMA) == expected


def test_a_wrong_fold_is_actually_caught():
    """The negative control, without which the three tests above prove only that they run.

    A fold that promotes an integer-preserving function is exactly the defect that shipped,
    so the check has to be shown rejecting it rather than assumed to.
    """
    original = MathExpr(fn="sign", input=lit(-7))
    correct = ax._fold_math_lit(original)
    assert infer_type(correct, _SCHEMA) == infer_type(original, _SCHEMA)

    regressed = Lit(float((-7 > 0) - (-7 < 0)))  # the pre-87d82730 fold: -1.0, not -1
    assert regressed.value == correct.value, "the defect was a type change, not a wrong value"
    assert infer_type(regressed, _SCHEMA) != infer_type(original, _SCHEMA), (
        "the type comparison must reject the float fold, or it would not have caught this"
    )
