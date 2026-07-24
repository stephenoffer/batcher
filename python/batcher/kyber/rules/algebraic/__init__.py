"""Algebraic rewrites: small, local, unconditionally semantics-preserving simplifications.

Split out of the former single `algebraic.py` when it outgrew the module size limit, along
the seam between the two things it did: rewriting a *disjunction* and applying relational
*identities*.

**The import order below is the run order.** Registration order is within-phase rule order,
so `disjunctions` is imported first because that is where its two rules sat in the file this
package replaces — `factor_common_conjuncts` ran before everything here, and folding an `IN`
list before `constant_propagation` is what the plan shapes in `tests/unit` assert. Sorting
these two lines would silently reorder five NORMALIZE rules.
"""

from __future__ import annotations

# isort: off
from batcher.kyber.rules.algebraic.disjunctions import (
    factor_common_conjuncts,
    fold_in_list,
)
from batcher.kyber.rules.algebraic.identities import (
    combine_limits,
    constant_propagation,
    eliminate_sort_before_aggregate,
    merge_adjacent_filters,
    prune_true_filter,
    push_distinct_into_union,
    push_filter_into_union,
    push_limit_into_union,
    push_limit_through_project,
    remove_redundant_distinct,
)

# isort: on

__all__ = [
    "combine_limits",
    "constant_propagation",
    "eliminate_sort_before_aggregate",
    "factor_common_conjuncts",
    "fold_in_list",
    "merge_adjacent_filters",
    "prune_true_filter",
    "push_distinct_into_union",
    "push_filter_into_union",
    "push_limit_into_union",
    "push_limit_through_project",
    "remove_redundant_distinct",
]
