"""String rule families: predicate absorption, predicate normalization, and lengths.

`exprs/text` and `exprs/text_folds` already fold literal string calls and turn a `LIKE`
or a plain regular expression into the cheaper `starts_with`/`ends_with`/`contains` form.
These modules pick up where that leaves off, on the three shapes that survive it:

* `absorption` — two string predicates over the same column where one implies the other.
  A `WHERE path LIKE 'a/%' AND path LIKE 'a/b/%'` reaches the pushdown phase as two
  independent `starts_with` calls, and only the longer one carries information.
* `predicates` — a string call compared against a number or a literal, where the
  comparison itself is the predicate: `position(s, 'x') > 0` is `contains(s, 'x')`, and
  `substr(s, 1, 3) = 'abc'` is `starts_with(s, 'abc')`.
* `lengths` — comparisons against `length(s)`, which are emptiness tests in disguise.

Every rule is exact under SQL's three-valued logic rather than merely on non-null rows:
each rewrite keeps the same operand, and each string function here is null-strict, so a
null input yields a null answer on both sides. Where a rewrite would need the operand to
be non-null to hold, the rule is absent rather than guarded — an emptiness test that
answers `false` for a null row instead of `NULL` is wrong inside a `NOT`, and only looks
right under a filter.
"""

from __future__ import annotations

from batcher.kyber.rules.text_algebra import absorption as _absorption  # noqa: F401  (registers)
from batcher.kyber.rules.text_algebra import lengths as _lengths  # noqa: F401  (registers)
from batcher.kyber.rules.text_algebra import predicates as _predicates  # noqa: F401  (registers)

__all__: list[str] = []
