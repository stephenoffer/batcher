"""Per-row cost of evaluating a scalar `Expr` — the dimension the cost model lacked.

Kyber priced a `Filter` at `filter_row x in_rows` regardless of its predicate, so
`x > 5` and `regexp_matches(s, '^a.*z$')` cost the same. They differ by ~two orders of
magnitude, and that difference decides real plans: whether to split a conjunction, how
much a projection costs above a join, whether to push a filter below a media decode.

Two facts about the data plane set the numbers:

* **Tier-1 (`bc-codegen`)** compiles a *subset* of `Expr` to native code and vectorizes
  it — numeric/temporal columns, non-string literals, arithmetic, comparisons,
  `and`/`or`/`not`, exact numeric casts, numeric `case`. An expression in that subset is
  several times cheaper per row than the same expression run through Arrow kernels.
* **Tier-0 (`bc-expr`)** evaluates everything else with Arrow compute kernels, whose
  per-row cost varies enormously by function: a `length` is a buffer read, a
  `regexp_replace` runs a regex engine per row.

`weights` prices one node, `jit` decides which tier runs the expression, and `model`
folds the two into `expr_cost` (absolute) and `expr_cost_factor` (relative to an
ordinary comparison — the multiplier the cost model applies).
"""

from __future__ import annotations

from batcher.kyber.expr_cost.jit import JIT_SPEEDUP, jit_compilable
from batcher.kyber.expr_cost.model import expr_cost, expr_cost_factor, raw_expr_cost

__all__ = [
    "JIT_SPEEDUP",
    "expr_cost",
    "expr_cost_factor",
    "jit_compilable",
    "raw_expr_cost",
]
