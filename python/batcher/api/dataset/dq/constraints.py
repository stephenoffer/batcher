"""The constraint values a `ds.dq` chain accumulates, before any of them is applied.

Separated from the accessor that builds them and the report that counts them because
these are the only part `api.dataset.meta.prove` reads: it discharges a contract from
metadata by inspecting `total` and `keys`, without executing anything. Keeping the values
in their own module is what stops that consumer from importing the whole accessor.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.plan.expr_ir import Expr

__all__ = ["Constraint", "RowConstraint", "UniqueConstraint"]


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


@dataclass(frozen=True, slots=True)
class UniqueConstraint:
    """A uniqueness constraint over `keys` — a row is valid iff its key combination
    occurs once (lowered to ``count() OVER (PARTITION BY keys) == 1``)."""

    name: str
    keys: tuple[str, ...]


Constraint = RowConstraint | UniqueConstraint
