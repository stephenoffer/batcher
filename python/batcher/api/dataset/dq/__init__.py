"""The `Dataset.dq` namespace — data-quality expectations with quarantine.

Breadth on `Dataset` lives on accessors (the Polars/v2 pattern), so quality checks
are reached as ``ds.dq.not_null("id").unique(["id"]).in_range("age", 0, 120)`` and
then a terminal action:

- ``.fail()`` — raise `DataQualityError` (with per-constraint counts) if any row
  violates; otherwise return the dataset unchanged. The data-contract gate.
- ``.drop()`` — return only the rows that satisfy every constraint.
- ``.quarantine()`` — return ``(clean, rejected)`` so bad rows route to a
  dead-letter sink instead of failing the pipeline.
- ``.validate()`` — a `ValidationReport` of per-constraint violation counts.

A constraint is just a boolean `Expr` that is TRUE for a valid row (plus the
group-level uniqueness check, which lowers to a window count). Everything lowers to
existing relational ops (FILTER, a keyless AGGREGATE for the report, ``count() OVER
(PARTITION BY keys)`` for uniqueness) — no new IR, and the valid/invalid split is a
provably total partition (validity is forced to a non-null boolean, so
``valid ⊎ invalid == input``).

**NULL is not a violation.** Value constraints (`in_range`/`matches`/`accepted_values`)
and `foreign_key` all treat NULL as valid, so they compose independently and a column
that is merely *optional* does not fail every check written against it — the
dbt/Great-Expectations convention, and SQL's own for a foreign key. Forbid nulls
explicitly with `not_null`.
"""

from __future__ import annotations

from batcher.api.dataset.dq.accessor import DatasetDQ
from batcher.api.dataset.dq.constraints import Constraint, RowConstraint, UniqueConstraint
from batcher.api.dataset.dq.report import ValidationReport

__all__ = [
    "Constraint",
    "DatasetDQ",
    "RowConstraint",
    "UniqueConstraint",
    "ValidationReport",
]
