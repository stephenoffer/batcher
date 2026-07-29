"""Shared traversal for scalar `Expr` trees and for the expressions inside a node.

Like `plan/visitor.py` but one level down: expression rewrites (constant folding,
simplification, and every future algebraic rule) should say *what* to do at a node
and never re-walk the `Binary`/`Not`/`Case`/… ladder. `referenced_columns` and
`remap_columns` in `expr_ir` predate this; new rules build on `transform_expr_up`.

`map_node_expressions` bridges the two levels: it applies an `Expr -> Expr`
rewrite to every expression a plan node carries (a `Filter`'s predicate, a
`Project`'s items, a `Sort`'s keys, …), so a pass is just
`transform_up(plan, lambda n: map_node_expressions(n, rule))`.

Split into a package along its responsibility seams (the structural ladder, boolean
algebra, subtree identity, the plan-node bridge); the import path is unchanged.
"""

from __future__ import annotations

from batcher.plan.expr_rewrite.algebra import (
    WINDOW_TEMP_PREFIX,
    combine_conjuncts,
    combine_disjuncts,
    hoist_windows,
    is_bare_window,
    split_conjuncts,
    split_disjuncts,
    substitute_columns,
)
from batcher.plan.expr_rewrite.nodes import map_node_expressions
from batcher.plan.expr_rewrite.subtrees import (
    contained_types,
    expr_key,
    replace_subtrees,
    subexpressions,
)
from batcher.plan.expr_rewrite.traverse import ExprRule, transform_expr_up

__all__ = [
    "WINDOW_TEMP_PREFIX",
    "ExprRule",
    "combine_conjuncts",
    "combine_disjuncts",
    "contained_types",
    "expr_key",
    "hoist_windows",
    "is_bare_window",
    "map_node_expressions",
    "replace_subtrees",
    "split_conjuncts",
    "split_disjuncts",
    "subexpressions",
    "substitute_columns",
    "transform_expr_up",
]
