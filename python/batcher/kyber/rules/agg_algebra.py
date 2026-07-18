"""Algebraic rewrites over *aggregate* expressions — share a base scan across a
family of linear aggregates.

`SUM` is linear, so a family of sums over the same base column shifted/scaled by
constants collapses onto **one** `SUM(base)` (and one `COUNT(base)`), with each
original output derived by a cheap scalar projection:

    SUM(base + c) = SUM(base) + c*COUNT(base)
    SUM(base - c) = SUM(base) - c*COUNT(base)
    SUM(c - base) = c*COUNT(base) - SUM(base)
    SUM(base * c) = c*SUM(base)                     (no COUNT needed)

The motivating shape is ClickBench Q29 — ``SUM(x + 0), SUM(x + 1), … SUM(x + 89)`` —
where an engine that materializes each ``x + i`` and sums it pays 90 passes over the
column. This rewrite turns those 90 aggregates into **2** (one `SUM(x)`, one
`COUNT(x)`) plus a 90-column scalar projection, which is both far less work than 90
passes and less than DuckDB's 90 accumulators.

Correctness: a non-null constant `c` makes ``base op c`` null exactly when ``base`` is
null, so `COUNT(base)` is the right multiplier for the shift and the null semantics
match (``SUM`` over an all-null base stays null: ``null + c*0`` is ``null``). `SUM` and
`COUNT` are the most mergeable aggregates, so the rewrite is identical single-node,
multi-core, and distributed — the derived projection is stateless. Only fires when it
**strictly reduces** the aggregate count, so a lone ``SUM(x+1)`` is never turned into
``SUM(x)+COUNT(x)`` (which would be more work, not less).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import AggExpr, Binary, Col, Expr, Lit
from batcher.plan.logical import Aggregate, AggregateSpec, LogicalPlan, Project, Projection

__all__ = ["decompose_linear_sum_aggregates"]

# A builder turns the shared (SUM(base), COUNT(base)) column references into the
# expression that reproduces one original aggregate's value.
_Build = Callable[[Col, Col], Expr]


def _numeric_lit(expr: Expr) -> int | float | None:
    """The value of `expr` if it is a non-null numeric literal (not bool), else None."""
    if (
        isinstance(expr, Lit)
        and isinstance(expr.value, (int, float))
        and not isinstance(expr.value, bool)
    ):
        return expr.value
    return None


def _decompose(agg: AggExpr) -> tuple[Expr, bool, _Build] | None:
    """`(base, needs_count, build)` if `agg` is a linear `SUM` over a base x constant.

    `build(sum_col, count_col)` returns the expression reproducing the original value
    from the shared `SUM(base)` / `COUNT(base)` columns. Returns None when the aggregate
    is not a decomposable linear sum (so it is kept verbatim)."""
    if agg.func != "sum" or not isinstance(agg.input, Binary):
        return None
    b = agg.input
    left_c, right_c = _numeric_lit(b.left), _numeric_lit(b.right)
    # Both sides constant → constant folding's job, not ours; skip.
    if left_c is not None and right_c is not None:
        return None
    if b.op == "add":
        # SUM(base + c) = SUM(base) + c*COUNT(base) — commutative, base is the non-lit side.
        if right_c is not None:
            base, c = b.left, right_c
        elif left_c is not None:
            base, c = b.right, left_c
        else:
            return None
        return (base, True, lambda s, n: Binary("add", s, Binary("mul", Lit(c), n)))
    if b.op == "sub":
        if right_c is not None:  # SUM(base - c) = SUM(base) - c*COUNT(base)
            base, c = b.left, right_c
            return (base, True, lambda s, n: Binary("sub", s, Binary("mul", Lit(c), n)))
        if left_c is not None:  # SUM(c - base) = c*COUNT(base) - SUM(base)
            base, c = b.right, left_c
            return (base, True, lambda s, n: Binary("sub", Binary("mul", Lit(c), n), s))
        return None
    if b.op == "mul":
        # SUM(base * c) = c*SUM(base) — needs no COUNT.
        if right_c is not None:
            base, c = b.left, right_c
        elif left_c is not None:
            base, c = b.right, left_c
        else:
            return None
        return (base, False, lambda s, _n: Binary("mul", Lit(c), s))
    return None


def _base_key(base: Expr) -> str:
    """A stable identity for a base expression, so sums over the same base share it."""
    return json.dumps(base.to_ir(), sort_keys=True)


@rule(name="decompose_linear_sum_aggregates", phase=Phase.REWRITE, matches=(Aggregate,))
def decompose_linear_sum_aggregates(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Collapse a family of ``SUM(base ± c)`` / ``SUM(base * c)`` onto one shared
    ``SUM(base)`` (+ ``COUNT(base)``), deriving each original output by projection.

    Fires only when it strictly reduces the number of aggregates — so it always removes
    work and can never re-fire on its own output (the rebuilt aggregate holds no
    decomposable sum), keeping the fixpoint convergent."""
    if node.watermark is not None:
        # A windowed streaming aggregate carries event-time state; leave it untouched
        # rather than reason about the watermark across the rewrite.
        return None

    # Per distinct base: (sum_alias, count_alias_or_None, base_expr). Insertion order is
    # deterministic (dict preserves it), so the rebuilt plan is stable across runs.
    bases: dict[str, tuple[str, str | None, Expr]] = {}
    needs_count: dict[str, bool] = {}
    # Output projections in the aggregate's exact output order (keys, then aggregates).
    projections: list[Projection] = [Projection(k.alias, Col(k.alias)) for k in node.group_keys]
    kept: list[AggregateSpec] = []
    decomposed = 0

    for spec in node.aggregates:
        decomp = _decompose(spec.agg)
        if decomp is None:
            kept.append(spec)
            projections.append(Projection(spec.alias, Col(spec.alias)))
            continue
        base, wants_count, build = decomp
        key = _base_key(base)
        if key not in bases:
            idx = len(bases)
            bases[key] = (f"__lsum_{idx}", f"__lcnt_{idx}", base)
            needs_count[key] = False
        needs_count[key] = needs_count[key] or wants_count
        sum_alias, count_alias, _ = bases[key]
        projections.append(Projection(spec.alias, build(Col(sum_alias), Col(count_alias))))
        decomposed += 1

    if decomposed == 0:
        return None

    # Hidden aggregates introduced: one SUM per base, plus one COUNT per base that a
    # shift needs. Only rewrite when that is strictly fewer than what it replaces.
    hidden = sum(1 + (1 if needs_count[k] else 0) for k in bases)
    if len(kept) + hidden >= len(node.aggregates):
        return None

    new_specs: list[AggregateSpec] = list(kept)
    for key, (sum_alias, count_alias, base) in bases.items():
        new_specs.append(AggregateSpec(sum_alias, AggExpr("sum", base)))
        if needs_count[key]:
            new_specs.append(AggregateSpec(count_alias, AggExpr("count", base)))

    new_agg = Aggregate(node.input, node.group_keys, tuple(new_specs), node.watermark)
    return Project(new_agg, tuple(projections))
