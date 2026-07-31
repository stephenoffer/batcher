"""Ordered-comparison sargable transposition, proved rather than assumed.

`sargable.py` transposes constant arithmetic across `=`/`<>` unconditionally and declines
the ordered comparisons, because the engine's i64 arithmetic wraps and wrapping breaks
monotonicity. This package supplies the missing half: the same transposition for `<`, `<=`,
`>`, `>=`, gated on a proof that the arithmetic cannot wrap over the column's range.

The range comes from the column's **recorded min/max** (`bounds`), the bounds a Parquet
footer, ORC index, lakehouse manifest, or Kyber's learned statistics supply and that zone-map
pruning already trusts. A column's *declared* width would prove the same thing for free, but
it cannot be used here: the plan layer normalizes every narrow numeric to `int64` before a
rule ever sees the schema (`bc-py/src/normalize.rs` does the same at the FFI boundary), so
`Int32` is not observable from inside the optimizer. `shared` holds the decomposition and the
overflow proof.
"""

from __future__ import annotations

from batcher.kyber.rules.extra.sargable_range.bounds import sarg_bounded_ordered

__all__ = ["sarg_bounded_ordered"]
