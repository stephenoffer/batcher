"""List-column rule families: order-insensitivity and constant folding.

`exprs/complex_types` already collapses the idempotent list calls and pushes the
order-insensitive reductions through `sort`/`reverse`/`unique`. These modules add the two
shapes it leaves:

* `ordering` — a membership test does not care what order the list is in, so it commutes
  with `sort`, `reverse` and `unique`. Dropping the reordering is usually the whole cost of
  the expression.
* `folds` — a list call over an `ARRAY[...]` of literals is a constant, and the identity
  `transform`/`filter` is not a call at all.

Two families that belong here in spirit and are absent for measured reasons:

* **`list_position(x, v) > 0` is not `list_contains(x, v)`.** The engine (matching DuckDB)
  answers *NULL* for the position of a value in an **empty** list, while `list_contains`
  answers `false`. The two disagree on every empty-list row, and nothing in a plan can tell
  an empty list from a populated one, so there is no guard that would make the rewrite
  sound. The string twin of this rule *is* registered (`text_algebra/predicates`), because
  `strpos` answers `0` for an empty string rather than NULL.
* **The numeric list reductions through a reordering.** `sum`, `mean`, `l2_norm` and the
  rest look order-insensitive and are not: floating-point addition is not associative, so
  summing a sorted list can differ from summing the original in the last ulp. `min`, `max`,
  `n_unique` and `len` are exact under reordering and are the only ones the sibling module
  moves.
"""

from __future__ import annotations

from batcher.kyber.rules.collections_algebra import folds as _folds  # noqa: F401  (registers)
from batcher.kyber.rules.collections_algebra import ordering as _ordering  # noqa: F401  (registers)

__all__: list[str] = []
