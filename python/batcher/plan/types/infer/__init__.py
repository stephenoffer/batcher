"""Per-expression output-type inference — a column's Arrow type before the engine runs.

`infer_type(expr, schema)` computes the Arrow `DataType` an expression produces
given its input schema, mirroring the engine's actual behavior (post FFI
widening). It is **sound, not complete**: any node whose output type is not
certain returns ``None`` so the caller falls back to the proven zero-row execution
rather than ever reporting a wrong type. This is what lets `available_schema()`
answer `Dataset.schema` without scanning, and lets the plan validate types early.

The rules are grouped by the operand information each family needs, which is also the
order they were hardest to get right in:

* `arithmetic` — binary operators and the math functions, whose result depends on the
  *operands'* types (and where `div`, `abs`, `round` and the decimals each break the
  rule their names suggest).
* `collections` — `list`/`struct`/`map`, which answer from one already-resolved type.
* `scalars` — `str`/`dt`, which answer from the function name alone.
* `dispatch` — the node-by-node recursion that routes to the three above.

Neutral layer.
"""

from __future__ import annotations

from batcher.plan.types.infer.dispatch import infer_type

__all__ = ["infer_type"]
