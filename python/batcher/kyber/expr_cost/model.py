"""Folding the per-node weights into a per-row cost for a whole expression."""

from __future__ import annotations

from batcher.kyber.expr_cost.jit import JIT_SPEEDUP, jit_compilable
from batcher.kyber.expr_cost.weights import BINARY_COST, own_cost, sub_exprs
from batcher.plan.expr_ir import Col, Expr, Lit

__all__ = ["expr_cost", "expr_cost_factor", "raw_expr_cost"]

# The raw cost of the archetypal predicate — an uncompiled `col OP literal` comparison.
_BASELINE_RAW = own_cost(Col("x")) + own_cost(Lit(0)) + BINARY_COST["lt"]

# A single expression must not dominate the whole cost model. The floor keeps a trivial
# projection from costing ~nothing; the ceiling bounds how far a media decode can swamp
# the row-count term that drives join ordering. The ceiling is set above the priciest
# *measured* scalar function (`sha256`, ~930x a comparison) so real costs are not
# truncated — only the unmeasured media-decode estimate is.
_MIN_FACTOR, _MAX_FACTOR = 0.2, 1000.0


def raw_expr_cost(expr: Expr) -> float:
    """Per-row cost of evaluating `expr` through the Tier-0 interpreter.

    Sums each node's own cost over the whole expression tree, ignoring whether the
    expression is JIT-compilable. Use `expr_cost` for the number a cost model consumes.

    Memoized on the (immutable) expression node: it is a pure function of the node's
    structure, and the cost model and the join-order search price the same expressions
    repeatedly. `expr_cost` is *not* memoized, because it also depends on the measured
    JIT speedup, which changes as calibration learns.

    Args:
        expr: The scalar expression to price.

    Returns:
        Cost in work-units, where one numeric comparison over one row is 1.0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.expr_cost import raw_expr_cost
            >>> raw_expr_cost(bt.col("x") > 5) > raw_expr_cost(bt.col("x"))
            True
    """
    cached = expr.__dict__.get("_c_rawcost")
    if cached is not None:
        return cached
    total = own_cost(expr)
    for child in sub_exprs(expr):
        total += raw_expr_cost(child)
    expr.__dict__["_c_rawcost"] = total
    return total


def expr_cost(expr: Expr, jit_speedup: float = JIT_SPEEDUP) -> float:
    """Effective per-row cost of `expr` on the tier that will actually evaluate it.

    A JIT-compilable expression is priced at `raw_expr_cost / jit_speedup`, because the
    Cranelift tier compiles it once per operator and reuses it across every morsel.

    `jit_speedup` defaults to the built-in prior but is supplied by `CostModel` from the
    *measured* speedup once `calibration` has seen the engine run both tiers — so the
    gap between compiled and interpreted expressions is a fact about this build on this
    hardware, not a constant.

    Args:
        expr: The scalar expression to price.
        jit_speedup: How much cheaper the compiled tier is per row.

    Returns:
        Cost in work-units, comparable with `CostCoefficients`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.expr_cost import expr_cost
            >>> cheap = expr_cost(bt.col("x") > 5)
            >>> pricey = expr_cost(bt.col("s").str.regexp_matches("^a.*z$"))
            >>> pricey > 10 * cheap
            True
    """
    raw = raw_expr_cost(expr)
    return raw / jit_speedup if jit_compilable(expr) else raw


def expr_cost_factor(expr: Expr, jit_speedup: float = JIT_SPEEDUP) -> float:
    """`expr`'s per-row cost relative to an ordinary `col OP literal` comparison.

    This is the multiplier an operator's per-row cost coefficient carries: 1.0 for the
    archetypal simple predicate (so calibrated coefficients keep their meaning), well
    below 1 for a bare column reference, and one-to-two orders of magnitude above it for
    a regex, a JSON extraction, or a media decode. Clamped so that no single expression
    can dominate the row-count term that drives join ordering.

    The baseline is measured on the same tier, so the archetypal compiled comparison is
    always exactly 1.0 whatever `jit_speedup` is; raising the measured speedup makes
    *interpreted* expressions relatively more expensive, which is what makes the
    optimizer work harder to keep them off the hot rows.

    Args:
        expr: The scalar expression to price.
        jit_speedup: How much cheaper the compiled tier is per row.

    Returns:
        A multiplier in `[0.2, 200.0]`, equal to 1.0 for a simple compiled comparison.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.expr_cost import expr_cost_factor
            >>> round(expr_cost_factor(bt.col("x") > 5), 2)
            1.0
            >>> expr_cost_factor(bt.col("s").str.regexp_matches("^a")) > 50
            True
    """
    speedup = jit_speedup if jit_speedup > 0.0 else JIT_SPEEDUP
    baseline = _BASELINE_RAW / speedup  # the archetypal predicate, priced on its own tier
    return min(_MAX_FACTOR, max(_MIN_FACTOR, expr_cost(expr, speedup) / baseline))
