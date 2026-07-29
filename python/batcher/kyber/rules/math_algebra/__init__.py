"""Numeric rule families that turn a computed comparison back into a sargable one.

`extra/sargable` peels constant *arithmetic* off a column so a predicate reads
`col OP literal`. These modules do the same job for the numeric *functions*: a
predicate over `abs(x)`, `sign(x)`, `floor(x)`, `ceil(x)`, `bit_count(x)` or an integer
`x // k` is opaque to zonemap pruning and to source pushdown, and each one has an exact
restatement in terms of the bare column.

The payoff is not the saved kernel pass. It is that `x BETWEEN -5 AND 5` can skip whole
row groups from a file's min/max statistics, while `abs(x) < 5` can skip none of them.

Every rewrite here is checked against the engine's actual numeric semantics rather than
against the mathematical ideal, because the two differ in ways that decide correctness:
integer `abs` **saturates** at `INT64_MAX` instead of wrapping, float comparison uses
Arrow's **total order** (a NaN sorts above every finite value and equals itself, so
`NaN > 0` is `true`), and `sign` answers `0.0` for that same NaN. The last two together
are why the whole `sign` family is restricted to integer columns. Each module documents
which of its rules a given fact constrains.
"""

from __future__ import annotations

from batcher.kyber.rules.math_algebra import absolute as _absolute  # noqa: F401  (registers)
from batcher.kyber.rules.math_algebra import (
    float_predicates as _float_predicates,  # noqa: F401  (registers)
)
from batcher.kyber.rules.math_algebra import rounding as _rounding  # noqa: F401  (registers)

__all__: list[str] = []
