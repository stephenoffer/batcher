"""NORMALIZE-phase whole-tree rewrites, grouped by family.

The first phase Kyber runs. `fold` evaluates constant sub-expressions, `simplify`
drops the algebraic identities that folding (or the query itself) leaves behind,
`ranges` turns opaque predicates — a prefix `LIKE`, a `date_trunc` equality, an `OR`
chain of equalities — into sargable comparisons that zone-map pruning and source
pushdown can act on, and `predicates` normalizes the boolean shapes that hide a usable
predicate inside a self-comparison, a `CASE`, a redundant `IN` pair, or a constant
grouping key.

Importing this package registers the range and predicate rules; `ConstantFolding` and
`ExprSimplification` are instantiated by `kyber.registry.register_builtin_rules`.
The pure functions stay importable for unit tests.
"""

from __future__ import annotations

# Registration order *is* within-phase run order, so these imports are ordered by when
# their rules must register, not alphabetically. `disjunctions` was split out of `ranges`
# and its rules registered last there, so it is imported directly after it — sorting the
# block moves `or_equalities_to_in_list` / `or_to_in_and_range` ahead of nine rules that
# used to precede them, which `just surface-diff` reports as a behavior change.
# isort: off
from batcher.kyber.rules.normalize.fold import ConstantFolding, fold_constants
from batcher.kyber.rules.normalize.predicates import (
    boolean_case_to_predicate,
    constant_group_key_removed,
    self_comparison_to_null_check,
)
from batcher.kyber.rules.normalize.ranges import (
    date_trunc_to_range,
    like_prefix_to_range,
)
from batcher.kyber.rules.normalize.disjunctions import or_to_in_and_range
from batcher.kyber.rules.normalize.simplify import ExprSimplification, simplify_expressions

# isort: on

__all__ = [
    "ConstantFolding",
    "ExprSimplification",
    "boolean_case_to_predicate",
    "constant_group_key_removed",
    "date_trunc_to_range",
    "fold_constants",
    "like_prefix_to_range",
    "or_to_in_and_range",
    "self_comparison_to_null_check",
    "simplify_expressions",
]
