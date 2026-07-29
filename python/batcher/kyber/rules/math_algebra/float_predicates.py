"""`isnan` / `isinf` see through the rounding functions.

Rounding moves a value to a nearby integer and leaves the two non-finite classes exactly
where they are: `floor(NaN)` is NaN, `abs(-inf)` is `inf`, and none of the six functions can
turn a finite value into a NaN or an infinity, nor a non-finite one into a finite value. So
`isnan(round(x))` asks the same question as `isnan(x)`, and the rounding is pure overhead
inside the test.

The payoff is the same as for the null tests in `nulls/strictness`: a NaN or infinity check
on a *bare column* is a shape the later rules can act on — `nan_check_on_integer_to_false`
folds it away entirely on an integer column, and a check on a computed value is opaque to
that. It also composes, so `isnan(floor(abs(x)))` collapses to `isnan(x)` in one bottom-up
pass.

`sign` is deliberately absent from the vocabulary even though it looks like it belongs. The
engine answers `sign(NaN) = 0.0` — it does *not* propagate the NaN — so `isnan(sign(x))` is
`false` everywhere rather than `isnan(x)`. That single function is why this module carries
an explicit list instead of accepting any unary math call.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.core import IsInf, IsNan, MathExpr
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window

__all__ = ["NON_FINITE_THROUGH_ROUNDING_RULES"]

_NODES = (Filter, Project, Aggregate, Sort, Window)

#: Unary math functions that map NaN to NaN, each infinity to an infinity, and every finite
#: value to a finite one — so they change neither of the two non-finite classes. `sign` is
#: excluded (it reports a NaN as `0.0`), and so is every function that can *introduce* a
#: non-finite value from a finite argument (`ln`, `exp`, `sqrt`, the trigonometric family).
_CLASS_PRESERVING = frozenset({"abs", "ceil", "floor", "rint", "round", "trunc"})


def _through_rounding(check: type) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, check)
            and isinstance(expr.input, MathExpr)
            and expr.input.fn in _CLASS_PRESERVING
        ):
            return check(expr.input.input)
        return expr

    return leaf


def _register(name: str, leaf: Callable[[Expr], Expr], expr_matches: tuple[type, ...]):
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=expr_matches,
        )
    )


#: `isnan(f(x)) -> isnan(x)` and `isinf(f(x)) -> isinf(x)` for the six class-preserving
#: rounding functions. Both directions collapse a chain in one bottom-up pass, since the
#: operand they produce is again a candidate for the same rule.
NON_FINITE_THROUGH_ROUNDING_RULES = [
    _register("nan_check_through_rounding", _through_rounding(IsNan), (IsNan,)),
    _register("inf_check_through_rounding", _through_rounding(IsInf), (IsInf,)),
]
