"""`f(f(x))` collapses to `f(x)` for every idempotent math function, and only for those.

`collapse_idempotent_math_fn` drops the outer call of a doubled `abs`/`sign`/`floor`/`ceil`/
`trunc`/`round`/`rint`. It is a small rule with a strong claim in its docstring -- that the
output type "cannot move", because the two calls are the same function -- and it had no test
that named it. `tests/unit/test_kyber_third_wave_families.py` exercises a *different* rule
over an overlapping set of functions (`nan_check_through_rounding`), which is close enough to
read as coverage without being it.

That matters more than an ordinary gap because of what the rule is adjacent to. `sign`,
`round` and `trunc` were promoting an integer to Float64 until `87d82730`, so the set this
rule iterates has just had its type behaviour changed underneath it. A collapse that was
type-preserving before such a change is not automatically type-preserving after.

Both halves are derived from `_IDEMPOTENT_MATH` rather than listed here, so a function added
to or removed from that set changes what is tested without a second edit. The exclusion side
is derived too: every foldable math function *not* in the idempotent set must be left alone,
which is what stops the rule from quietly widening to `sqrt`, where `sqrt(sqrt(x))` is a
fourth root and not a square one.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher.kyber.rules.extra.arith_extra as ax
from batcher.plan.expr_ir import col
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.schema import SchemaRef
from batcher.plan.types.infer.dispatch import infer_type

pytestmark = pytest.mark.unit

_SCHEMA = SchemaRef(arrow=pa.schema([("i", pa.int64()), ("f", pa.float64())]))

#: Math functions the rule must NOT collapse: everything foldable that is not idempotent.
#: `sqrt` is the one that matters -- `sqrt(sqrt(x))` is a fourth root.
_NOT_IDEMPOTENT = sorted(ax._FOLDABLE_MATH - ax._IDEMPOTENT_MATH)


def _nested(fn: str, name: str) -> MathExpr:
    return MathExpr(fn=fn, input=MathExpr(fn=fn, input=col(name)))


@pytest.mark.parametrize("fn", sorted(ax._IDEMPOTENT_MATH))
def test_a_doubled_call_loses_its_outer_one(fn):
    collapsed = ax._collapse_idempotent(_nested(fn, "f"))
    assert collapsed.to_ir() == MathExpr(fn=fn, input=col("f")).to_ir()


@pytest.mark.parametrize("column", ["i", "f"])
@pytest.mark.parametrize("fn", sorted(ax._IDEMPOTENT_MATH))
def test_the_collapse_does_not_move_the_type(fn, column):
    """The docstring's own claim, over an integer column as well as a float one.

    The integer column is the half that was worth adding: four of these seven functions
    answer Int64 for an Int64 operand and three promote to Float64, and which is which
    changed recently. A collapse that dropped the outer call of a promoting function while
    the inner one preserved -- or the reverse -- would retype the column silently.
    """
    nested = _nested(fn, column)
    collapsed = ax._collapse_idempotent(nested)
    assert infer_type(collapsed, _SCHEMA) == infer_type(nested, _SCHEMA)


@pytest.mark.parametrize("fn", _NOT_IDEMPOTENT)
def test_a_function_that_is_not_idempotent_is_left_alone(fn):
    """The exclusion, derived so that moving a function into `_IDEMPOTENT_MATH` is deliberate.

    `sqrt(sqrt(x))` is the case with teeth: it is a fourth root, and collapsing it would be a
    wrong answer rather than a missed optimization.
    """
    nested = _nested(fn, "f")
    assert ax._collapse_idempotent(nested).to_ir() == nested.to_ir()


def test_two_different_functions_do_not_collapse_into_one():
    """The other way the rule could over-fire: `abs(sign(x))` is not `abs` and not `sign`.

    Without this, a rule that collapsed on "outer is idempotent" rather than on "outer and
    inner are the same function" would pass every case above.
    """
    mixed = MathExpr(fn="abs", input=MathExpr(fn="sign", input=col("f")))
    assert ax._collapse_idempotent(mixed).to_ir() == mixed.to_ir()


def test_the_two_sets_do_not_overlap_by_accident():
    """`_NOT_IDEMPOTENT` must be non-empty, or the exclusion test above covers nothing.

    A parametrized test over an empty list passes silently, which would leave the most
    important assertion in this file -- that `sqrt` is excluded -- reporting success without
    running.
    """
    assert _NOT_IDEMPOTENT, "the exclusion cases vanished; the rule may have widened"
    assert "sqrt" in _NOT_IDEMPOTENT
