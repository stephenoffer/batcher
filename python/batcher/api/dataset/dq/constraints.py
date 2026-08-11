"""The constraint values a `ds.dq` chain accumulates, before any of them is applied.

Separated from the accessor that builds them and the report that counts them because
these are the only part `api.dataset.meta.prove` reads: it discharges a contract from
metadata by inspecting `total` and `keys`, without executing anything. Keeping the values
in their own module is what stops that consumer from importing the whole accessor.

Four kinds, distinguished by *what a violation is*, because that decides which terminal
actions can act on one:

* `RowConstraint` — a boolean per row. Countable, droppable, quarantinable.
* `UniqueConstraint` — a boolean per row too, but one that needs a window count to
  evaluate, so it is carried separately until `_prepared` lowers it.
* `AggregateConstraint` — a single number over the whole relation (row count, mean,
  null rate). There is no violating *row* to drop, so `drop`/`quarantine` refuse it.
* `SchemaConstraint` — answered from the schema before anything executes, so it carries
  its verdict rather than an expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from batcher.plan.expr_ir import Expr

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "AggregateConstraint",
    "Constraint",
    "ReferenceConstraint",
    "RowConstraint",
    "SchemaConstraint",
    "Severity",
    "UniqueConstraint",
]

Severity = Literal["error", "warn"]
"""How a violated constraint is treated: `"error"` enforces, `"warn"` only reports."""


@dataclass(frozen=True, slots=True)
class RowConstraint:
    """A row-level constraint: `valid` is TRUE exactly for rows that satisfy it."""

    name: str
    valid: Expr
    # Whether `valid` is *total* — TRUE or FALSE for every row, and never NULL.
    #
    # Every built-in constraint is (`in_range` and friends are `col IS NULL OR <test>`, which
    # is TRUE when the column is null; `not_null` is a plain `IS NOT NULL`). That matters for
    # one thing only: it is what lets the contract be discharged from metadata, by asking
    # whether the filter `NOT valid` keeps any row (`meta.prove`). With a NULL-valued validity
    # that probe would be wrong — a NULL validity counts as a *violation*, but `NOT NULL` is
    # NULL, so the filter would not see it. A user's `check()` predicate is not assumed total,
    # so it takes the executing path, which treats NULL as a violation exactly as before.
    total: bool = True
    # The fraction of rows that must satisfy `valid` for the constraint to *pass* — Great
    # Expectations' `mostly`. It moves the pass/fail line only; the violating rows are still
    # counted, and still dropped or quarantined, because a tolerated violation is a violation
    # you chose not to fail the run over, not a row that became valid.
    mostly: float = 1.0
    severity: Severity = "error"

    @property
    def enforced(self) -> bool:
        """Whether a violation of this constraint blocks — False for a `warn` constraint."""
        return self.severity == "error"


@dataclass(frozen=True, slots=True)
class UniqueConstraint:
    """A uniqueness constraint over `keys` — a row is valid iff its key combination
    occurs once (lowered to ``count() OVER (PARTITION BY keys) == 1``)."""

    name: str
    keys: tuple[str, ...]
    mostly: float = 1.0
    severity: Severity = "error"

    @property
    def enforced(self) -> bool:
        """Whether a violation of this constraint blocks — False for a `warn` constraint."""
        return self.severity == "error"


@dataclass(frozen=True, slots=True)
class ReferenceConstraint:
    """Referential integrity: every non-null key must resolve in another relation.

    Row-level in its verdict but not expressible as a row expression, because the answer
    depends on a second relation. It lowers to a LEFT JOIN against the distinct reference
    keys plus a marker column, which is what lets `drop` and `quarantine` act on it with no
    new IR and no separate distributed semantics.

    A row whose key is NULL is **not** an orphan: a NULL foreign key means "no reference",
    not "a reference that is broken" — SQL's own ``FOREIGN KEY`` accepts it, dbt's
    ``relationships`` test excludes it, and it is the convention every value constraint here
    already follows.
    """

    name: str
    columns: tuple[str, ...]
    references: Any
    ref_columns: tuple[str, ...]
    mostly: float = 1.0
    severity: Severity = "error"

    @property
    def enforced(self) -> bool:
        """Whether a violation of this constraint blocks — False for a `warn` constraint."""
        return self.severity == "error"

    def reference_keys(self, prefix: str) -> Dataset:
        """The distinct reference keys, renamed to collision-proof helper columns.

        Named for the reference side rather than `keys`, because `api.dataset.meta.prove`
        recognizes a uniqueness constraint by the presence of a `keys` attribute and would
        read a method of that name as a key tuple.

        Args:
            prefix: A per-constraint prefix keeping two reference checks' helpers apart.

        Returns:
            A lazy `Dataset` of one distinct row per key, with columns ``{prefix}k{i}``.
        """
        from batcher.plan.expr_ir import Col

        projected = {f"{prefix}k{i}": Col(c) for i, c in enumerate(self.ref_columns)}
        return self.references.select(**projected).distinct()


@dataclass(frozen=True, slots=True)
class AggregateConstraint:
    """A relation-level constraint: one aggregate value that must fall within bounds.

    `value` is a keyless aggregate expression (a row count, a mean, a null rate). `low`
    and `high` are inclusive and either may be `None` for an open side. A NULL measured
    value — the aggregate of an empty relation — counts as a violation, because a
    contract that cannot be evaluated has not been met.
    """

    name: str
    value: Expr
    low: float | None = None
    high: float | None = None
    severity: Severity = "error"

    @property
    def enforced(self) -> bool:
        """Whether a violation of this constraint blocks — False for a `warn` constraint."""
        return self.severity == "error"

    def holds(self, measured: float | None) -> bool:
        """Whether `measured` satisfies the bounds; a NULL measurement never does.

        Args:
            measured: The aggregate's value, or `None` when it evaluated to NULL.

        Returns:
            True when the value lies inside the inclusive bounds.
        """
        if measured is None:
            return False
        if self.low is not None and measured < self.low:
            return False
        return not (self.high is not None and measured > self.high)


@dataclass(frozen=True, slots=True)
class SchemaConstraint:
    """A schema-level constraint, already decided — the schema is known before execution.

    Carries its verdict rather than an expression: `satisfied` is the answer and `detail`
    says what was wrong, so the report can name the missing column or the wrong type
    without a scan. A schema check that fails is worth catching *first*, because every
    value constraint written against a column that is absent fails for the wrong reason.
    """

    name: str
    satisfied: bool
    detail: str = ""
    severity: Severity = "error"

    @property
    def enforced(self) -> bool:
        """Whether a violation of this constraint blocks — False for a `warn` constraint."""
        return self.severity == "error"


Constraint = (
    RowConstraint | UniqueConstraint | ReferenceConstraint | AggregateConstraint | SchemaConstraint
)
