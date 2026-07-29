"""Predicate families that reason about *disjunctions* of bounds and sets.

Kyber already reasons well about conjunctions: `tighten_comparison_bounds` intersects two
bounds on a column, `filter_range_contradiction` refutes an empty intersection, and
`refine_in_list_by_comparison` narrows a set against a range. The `OR` side is thinner, and
it is the side a generated predicate lands on — an `IN` list split across two clauses, a
date range assembled from two overlapping windows, a UI filter that emits one disjunct per
selected option.

`bounds` closes that gap: it unions same-direction bounds, absorbs an equality into the
bound that already covers it, merges `IN` lists, and collapses two overlapping ranges into
one. Each rewrite makes the predicate strictly smaller and, more importantly, leaves a
*single* bound or set per column — which is the shape `zonemap_prune_filter` and source
pushdown can act on. Two disjuncts on the same column are opaque to both.

Everything here is exact under three-valued logic without a non-null guard, and for one
reason: every rule keeps the same operand and replaces a disjunction of null-strict
comparisons with another null-strict comparison on it. A null row answers `NULL` on both
sides, because `NULL OR NULL` is `NULL`.
"""

from __future__ import annotations

from batcher.kyber.rules.predicate_algebra import bounds as _bounds  # noqa: F401  (registers)

__all__: list[str] = []
