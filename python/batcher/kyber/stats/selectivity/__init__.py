"""Predicate selectivity — the fraction of rows a `Filter` keeps.

Selinger-style structural estimation: conjunctions combine with **exponential backoff**
(not a raw independence product, which badly underestimates the kept fraction on correlated
predicates), disjunctions use inclusion-exclusion, negation complements. A leaf
`col = literal` uses `1/ndv` when the distinct count is known; `col < literal` interpolates
the fraction below the literal from per-column quantile boundaries when known, else a
Selinger range constant. These feed the row-count estimator; they are *estimates* and never
carry `EXACT` provenance.

Structured as three layers so it stays under the module-size limit: `scalars` (value and
column-stat primitives) → `leaves` (one estimate per non-composite predicate) → `combine`
(walk the boolean tree). `comparison_col_side` is re-exported because several Kyber rules
reuse it to recognise a `col OP literal` comparison.
"""

from __future__ import annotations

from batcher.kyber.stats.selectivity.combine import predicate_selectivity
from batcher.kyber.stats.selectivity.scalars import comparison_col_side

__all__ = ["comparison_col_side", "predicate_selectivity"]
