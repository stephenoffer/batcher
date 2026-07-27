"""Self-comparison collapses: `x = x`, `x < x`, and the rest of the reflexive six.

DuckDB performs these as `comparison_simplification` and Spark as
`SimplifyBinaryComparison`. They arise mechanically rather than by hand: a
generated predicate, a templated filter, or a join condition that a rewrite has
substituted both sides of, all end up comparing an expression with itself. Folding
one to a constant then feeds constant propagation and the empty-relation rules,
which can delete an entire branch of the plan.

Three preconditions are checked on every one of them, and each rules out a real
counter-example rather than being defensive.

* **Non-nullable.** `NULL = NULL` is `NULL`, not `TRUE`, and `NULL < NULL` is `NULL`,
  not `FALSE`. On a nullable column the constant would be wrong on exactly the rows
  that are hardest to notice.
* **Provably not floating point.** This engine orders floats *totally*, NaN included,
  which would make `x <= x` true even for NaN -- but that is a property of these
  kernels rather than of IEEE 754, where `NaN <= NaN` is false. Leaning on it would
  bake an engine-local detail into a plan rewrite, so float operands are declined.
* **`safe_expr`.** Both sides stop being evaluated, so collapsing them must not drop
  an error the query would otherwise have raised.

The schema plumbing is `guards.schema_rule`, shared with the sibling `numeric` module.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import is_float, nullable, schema_rule
from batcher.kyber.rules.leaf_rewrite import safe_expr
from batcher.plan.expr_ir import Expr, Lit
from batcher.plan.expr_ir.core import Binary
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

__all__ = [
    "self_comparison_eq_to_true",
    "self_comparison_ge_to_true",
    "self_comparison_gt_to_false",
    "self_comparison_le_to_true",
    "self_comparison_lt_to_false",
    "self_comparison_ne_to_false",
]


# The shared precondition, stated once because all six rules below carry it: the two
# sides must be structurally identical, the value must be non-nullable (`NULL op NULL`
# is `NULL`, not a boolean), it must provably not be floating point (so the engine's
# total NaN ordering is never leaned on), and it must be `safe_expr` (so collapsing two
# evaluations into a constant cannot drop an error the query would have raised).


def _self_cmp(op: str, result: bool):
    """Build the leaf rewrite collapsing `x <op> x` to the constant `result`."""

    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        if (
            isinstance(expr, Binary)
            and expr.op == op
            and safe_expr(expr.left)
            and not nullable(expr.left, schema)
            and not is_float(expr.left, schema)
            and expr_key(expr.left) == expr_key(expr.right)
        ):
            return Lit(result)
        return expr

    return leaf


_LT_SELF = _self_cmp("lt", False)
_GT_SELF = _self_cmp("gt", False)
_NE_SELF = _self_cmp("ne", False)
_EQ_SELF = _self_cmp("eq", True)
_LE_SELF = _self_cmp("le", True)
_GE_SELF = _self_cmp("ge", True)


@rule(name="self_comparison_lt_to_false", phase=Phase.NORMALIZE, matches=(Filter, Project))
def self_comparison_lt_to_false(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x < x -> FALSE`: no value precedes itself under a strict order."""
    return schema_rule(node, _LT_SELF, carries=(Binary,))


@rule(name="self_comparison_gt_to_false", phase=Phase.NORMALIZE, matches=(Filter, Project))
def self_comparison_gt_to_false(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x > x -> FALSE`: no value follows itself under a strict order."""
    return schema_rule(node, _GT_SELF, carries=(Binary,))


@rule(name="self_comparison_ne_to_false", phase=Phase.NORMALIZE, matches=(Filter, Project))
def self_comparison_ne_to_false(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x != x -> FALSE`: a value is equal to itself, so it is never unequal."""
    return schema_rule(node, _NE_SELF, carries=(Binary,))


@rule(name="self_comparison_eq_to_true", phase=Phase.NORMALIZE, matches=(Filter, Project))
def self_comparison_eq_to_true(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x = x -> TRUE`: equality is reflexive."""
    return schema_rule(node, _EQ_SELF, carries=(Binary,))


@rule(name="self_comparison_le_to_true", phase=Phase.NORMALIZE, matches=(Filter, Project))
def self_comparison_le_to_true(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x <= x -> TRUE`: a non-strict order is reflexive."""
    return schema_rule(node, _LE_SELF, carries=(Binary,))


@rule(name="self_comparison_ge_to_true", phase=Phase.NORMALIZE, matches=(Filter, Project))
def self_comparison_ge_to_true(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x >= x -> TRUE`: a non-strict order is reflexive."""
    return schema_rule(node, _GE_SELF, carries=(Binary,))
