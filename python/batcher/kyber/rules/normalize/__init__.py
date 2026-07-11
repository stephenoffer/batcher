"""NORMALIZE-phase whole-tree rewrites, grouped by family.

The first phase Kyber runs. `fold` evaluates constant sub-expressions, `simplify`
drops the algebraic identities that folding (or the query itself) leaves behind, and
`ranges` turns opaque predicates — a prefix `LIKE`, a `date_trunc` equality, an `OR`
chain of equalities — into sargable comparisons that zone-map pruning and source
pushdown can act on.

Importing this package registers the range rules; `ConstantFolding` and
`ExprSimplification` are instantiated by `kyber.registry.register_builtin_rules`.
The pure functions stay importable for unit tests.
"""

from __future__ import annotations

from batcher.kyber.rules.normalize.fold import ConstantFolding, fold_constants
from batcher.kyber.rules.normalize.ranges import (
    date_trunc_to_range,
    like_prefix_to_range,
    or_to_in_and_range,
)
from batcher.kyber.rules.normalize.simplify import ExprSimplification, simplify_expressions

__all__ = [
    "ConstantFolding",
    "ExprSimplification",
    "date_trunc_to_range",
    "fold_constants",
    "like_prefix_to_range",
    "or_to_in_and_range",
    "simplify_expressions",
]
