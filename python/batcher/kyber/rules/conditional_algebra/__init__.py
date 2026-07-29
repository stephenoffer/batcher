"""`CASE` rule families: pushing calls into the branches, and merging the branches.

`extra/conditional` already pushes the arithmetic operators and the comparisons into a
`CASE`'s branches, folds a `CASE` over literals, and drops branches that can never be
reached. These modules extend the same two ideas:

* `push_calls` pushes the remaining *function* families — the math functions, the string
  functions, the date extractions, the NaN/infinity predicates, the Kleene connectives and
  the bitwise operators — down onto each branch value.
* `branches` merges branches that produce the same value and collapses a nested `CASE`
  that re-tests a condition its parent has already decided.

Pushing a call into the branches is not about the call itself; it is about what happens
next. Branch values are overwhelmingly literals, and `f(literal)` is a constant. So a
`upper(CASE WHEN c THEN 'a' ELSE 'b' END) = 'A'` becomes `CASE WHEN c THEN true ELSE false
END` after folding, and then just `c` — a whole expression tree collapses into the
predicate that was inside it all along.
"""

from __future__ import annotations

from batcher.kyber.rules.conditional_algebra import branches as _branches  # noqa: F401  (registers)
from batcher.kyber.rules.conditional_algebra import (
    push_calls as _push_calls,  # noqa: F401  (registers)
)

__all__: list[str] = []
