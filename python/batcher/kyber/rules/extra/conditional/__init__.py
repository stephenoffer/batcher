"""NORMALIZE-phase rewrites for the conditional family — CASE / NULLIF / COALESCE / GREATEST /
LEAST.

A SQL front end (and a `when(...).then(...)` builder chain) emits conditional shapes that are
frequently dead on arrival: a branch guarded by a constant, a `CASE` whose arms all return the
same value, a `COALESCE` nested in a `COALESCE`, a `GREATEST` over constants. Each rule here
removes one such shape — shrinking the tree the data plane evaluates and, where a `CASE` collapses
to a `COALESCE` or a bare column, exposing a simpler expression to pushdown and constant folding.

**Branch selection under three-valued logic.** A `CASE` branch fires only where its condition is
*TRUE*: a NULL condition selects nothing, exactly like FALSE. That is what makes it sound to *drop*
a branch whose condition is constant-null, and why `case_first_true_branch_wins` may fire only on a
literal TRUE (a FALSE/NULL condition proves nothing about the branches below it). In this IR a NULL
condition is not even expressible as a literal — `Lit` cannot hold `None`, and the engine's
typed-NULL idiom is `NULLIF(lit(v), lit(v))` — so the constant-condition rules key on
`Lit(False)`/`Lit(True)`, and `_is_null_lit` recognizes that idiom.

**Two guards every dropping rule passes.** *Purity* (`_pure`): a removed sub-expression must be
deterministic and unable to raise, so deleting it changes neither the value nor whether the query
errors. *Type preservation* (`_droppable`): CASE/COALESCE/GREATEST/LEAST take their result type from
the *join* of their arms' types, so dropping an arm can move the output type —
`CASE WHEN FALSE THEN 1 ELSE 2.5 END` is a DOUBLE, and naively deleting the dead branch would make
it an INT. An arm is dropped only when a *kept* arm provably carries the same type.

Float edges are refused, not reasoned about: no rule folds a float extremum or a NaN comparison.

`nullif_same_operands` (`NULLIF(x, x)` → NULL) is deliberately **not** implemented: there is no NULL
literal in this IR to rewrite it to, and `NULLIF(lit(v), lit(v))` *is* the engine's canonical typed
NULL — rewriting it would destroy the type it carries.
"""

from __future__ import annotations

from batcher.kyber.rules.extra.conditional.case import (
    case_all_branches_same_result,
    case_drop_duplicate_conditions,
    case_drop_unreachable_branches,
    case_first_true_branch_wins,
    case_no_branches_to_else,
    case_to_coalesce,
    coalesce_dedup_args,
    coalesce_drop_nulls_after_first_non_null,
    coalesce_flatten_nested,
    coalesce_single_arg,
    nullif_distinct_literals,
)
from batcher.kyber.rules.extra.conditional.minmax import (
    greatest_least_dedup_args,
    greatest_least_flatten_nested,
    greatest_least_fold_literals,
    greatest_least_single_arg,
)
from batcher.kyber.rules.extra.conditional.shared import (
    _pure,  # noqa: F401  (sibling families reuse the purity guard)
)

__all__ = [
    "case_all_branches_same_result",
    "case_drop_duplicate_conditions",
    "case_drop_unreachable_branches",
    "case_first_true_branch_wins",
    "case_no_branches_to_else",
    "case_to_coalesce",
    "coalesce_dedup_args",
    "coalesce_drop_nulls_after_first_non_null",
    "coalesce_flatten_nested",
    "coalesce_single_arg",
    "greatest_least_dedup_args",
    "greatest_least_flatten_nested",
    "greatest_least_fold_literals",
    "greatest_least_single_arg",
    "nullif_distinct_literals",
]
