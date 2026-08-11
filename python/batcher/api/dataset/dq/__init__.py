"""The `Dataset.dq` namespace — data-quality expectations with quarantine.

Breadth on `Dataset` lives on accessors (the Polars/v2 pattern), so quality checks
are reached as ``ds.dq.not_null("id").unique(["id"]).in_range("age", 0, 120)`` and
then a terminal action:

- ``.fail()`` — raise `DataQualityError` (with per-constraint counts) if any row
  violates; otherwise return the dataset unchanged. The data-contract gate.
- ``.drop()`` — return only the rows that satisfy every constraint.
- ``.quarantine()`` — return ``(clean, rejected)`` so bad rows route to a
  dead-letter sink instead of failing the pipeline.
- ``.annotate()`` — keep every row and add a column naming what it failed.
- ``.validate()`` — a `ValidationReport` of per-constraint results.

A constraint is a boolean `Expr` that is TRUE for a valid row, plus three kinds that need
more than an expression: uniqueness (a window count), referential integrity (a join against
the reference keys), and the relation-level checks (one aggregate over the whole table).
Everything lowers to existing relational ops — FILTER, a keyless AGGREGATE, ``count() OVER
(PARTITION BY keys)``, a LEFT JOIN — so there is no new IR, no separate distributed
semantics, and the valid/invalid split is a provably total partition (validity is forced to
a non-null boolean, so ``valid ⊎ invalid == input``).

**NULL is not a violation.** Value constraints (`in_range`/`matches`/`accepted_values` and
the rest) and the referential checks all treat NULL as valid, so they compose independently
and a column that is merely *optional* does not fail every check written against it — the
dbt/Great-Expectations convention, and SQL's own for a foreign key. Forbid nulls explicitly
with `not_null`.

**Every constraint carries a tolerance and a severity.** ``mostly=0.99`` passes the
constraint while 1% of rows violate it; ``severity="warn"`` reports a violation without
enforcing it anywhere. Both are declared per constraint, so one chain can mix a hard gate
with a rule that is still being trialled.
"""

from __future__ import annotations

from batcher.api.dataset.dq.accessor import DatasetDQ
from batcher.api.dataset.dq.constraints import (
    AggregateConstraint,
    Constraint,
    ReferenceConstraint,
    RowConstraint,
    SchemaConstraint,
    UniqueConstraint,
)
from batcher.api.dataset.dq.report import ConstraintResult, ValidationReport

__all__ = [
    "AggregateConstraint",
    "Constraint",
    "ConstraintResult",
    "DatasetDQ",
    "ReferenceConstraint",
    "RowConstraint",
    "SchemaConstraint",
    "UniqueConstraint",
    "ValidationReport",
]
