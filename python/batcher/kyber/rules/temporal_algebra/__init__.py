"""Temporal rule families that put a predicate back onto the raw timestamp column.

`exprs/temporal` and `extra/temporal_sargable` already turn `year(ts) = 2024` and
`date_trunc('month', ts) = …` into instant ranges. This package covers the two other ways
a query hides a timestamp behind a computation:

* `epoch(ts)`, the Unix-second count, which turns an instant comparison into an integer
  one. It is a *floored* second count, so the comparison has an exact half-open instant
  interval — the same argument the integer bucket rules use for `//`.
* `offset_by(ts, …)`, a shift by a fixed number of days and microseconds, which a query
  writes when it compares "the timestamp plus a grace period" against a deadline. Moving
  the shift onto the literal leaves the bare column in the comparison.

Both are worth more than the kernel pass they save: a bare timestamp comparison is what
partition pruning matches against a `dt=` path component, what the zonemap pruner refutes
from a row group's min/max, and what a source pushes into a Parquet or Iceberg scan.

Calendar months are deliberately absent from the `offset_by` rules. A month is not a
fixed duration, so `ts + 1 month > L` is not `ts > L - 1 month` — the two disagree at the
end of a long month — and the rule declines rather than approximating.
"""

from __future__ import annotations

from batcher.kyber.rules.temporal_algebra import epoch as _epoch  # noqa: F401  (registers)
from batcher.kyber.rules.temporal_algebra import offsets as _offsets  # noqa: F401  (registers)

__all__: list[str] = []
