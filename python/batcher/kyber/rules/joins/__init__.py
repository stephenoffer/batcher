"""The join rule family — every rewrite that reshapes a join, in one package.

Three responsibilities, one per module, all speaking about the same operator:

- `rewrites`: change a join's *type* or drop a side (`outer_to_inner_join`,
  `join_to_semijoin`, `eliminate_left_join`, `runtime_join_filter`). It owns the
  engine's uniqueness proof (`_right_unique_on_keys`) and the filterable-side table
  (`_FILTERABLE_SIDES`), which the elimination and runtime-filter families read from
  here rather than restate.
- `order`: cost-based reordering of a connected inner-join subtree (the JOIN_REORDER
  phase — exact DP, DPhyp, greedy fallback).
- `projection`: push a derived projection through a join onto the side it reads.

The public import path `batcher.kyber.rules.joins` is unchanged, and so is what it
means: importing it registers the `rewrites` rules and re-exports the shared join
vocabulary below. `order` and `projection` are imported for registration by
`kyber.rules` at the same point in the sequence they always were — registration order
is within-phase *run* order, so this package deliberately does not pull them in early.
"""

from __future__ import annotations

# The private names are re-exported deliberately (redundant alias = an explicit
# re-export): `_right_unique_on_keys` / `_FILTERABLE_SIDES` are the one uniqueness proof
# and the one filterable-side table the elimination and runtime-filter families read.
from batcher.kyber.rules.joins.rewrites import _FILTERABLE_SIDES as _FILTERABLE_SIDES
from batcher.kyber.rules.joins.rewrites import _null_rejecting_cols as _null_rejecting_cols
from batcher.kyber.rules.joins.rewrites import _right_unique_on_keys as _right_unique_on_keys
from batcher.kyber.rules.joins.rewrites import _strengthened as _strengthened
from batcher.kyber.rules.joins.rewrites import (
    drop_redundant_cross_key,
    drop_redundant_distinct_build,
    eliminate_left_join,
    join_to_semijoin,
    outer_to_inner_join,
    runtime_join_filter,
)

__all__ = [
    "drop_redundant_cross_key",
    "drop_redundant_distinct_build",
    "eliminate_left_join",
    "join_to_semijoin",
    "outer_to_inner_join",
    "runtime_join_filter",
]
