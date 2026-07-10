"""The policy objects a `SecurityCatalog` holds: grants, column masks, row filters.

Each is an immutable value describing *what* is allowed or rewritten, never *how* —
resolution against a `Principal` lives in `catalog`, and the plan rewrite in `enforce`.
Keeping them inert makes a catalog a serializable, diffable, reviewable artifact: the
thing a security team signs off on.

A policy's `mask` / `predicate` is a callable that builds an `Expr`, not an `Expr`
itself. A column mask must be applied to whichever column it governs — the same
"redact to the last four" rule serves `card_number` and `ssn` — and a row filter must
see the principal to compare against its attributes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from batcher.governance.principal import Principal
from batcher.plan.expr_ir import Expr

__all__ = ["ColumnMask", "Grant", "RowFilter", "TagMask"]

#: A column mask: given the column's expression, return the expression to read instead.
MaskFn = Callable[[Expr], Expr]

#: A row filter: given the principal, return the predicate rows must satisfy.
PredicateFn = Callable[[Principal], Expr]


def _frozen(roles: Iterable[str]) -> frozenset[str]:
    return frozenset(roles)


@dataclass(frozen=True, slots=True)
class Grant:
    """`role` may `SELECT` `columns` of `table` (all columns when `columns` is None).

    The presence of *any* grant on a table switches that table to deny-by-default: a
    principal then sees exactly the union of the columns its roles are granted. A table
    with no grant at all is ungoverned for access (though it may still carry masks and
    row filters), so installing a catalog does not silently lock out every query.
    """

    role: str
    table: str
    columns: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.columns is not None:
            object.__setattr__(self, "columns", frozenset(self.columns))


@dataclass(frozen=True, slots=True)
class ColumnMask:
    """Read `table`.`column` through `mask` unless the principal holds an exempt role.

    The mask is applied at the scan, so *everything* downstream — filters, joins,
    aggregates, the final projection — sees only the masked value. A principal cannot
    recover the underlying value by filtering on it or grouping by it.
    """

    table: str
    column: str
    mask: MaskFn
    exempt_roles: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exempt_roles", _frozen(self.exempt_roles))


@dataclass(frozen=True, slots=True)
class TagMask:
    """Mask every column tagged `tag`, wherever it appears, unless exempt.

    The reason a catalog scales past a handful of tables: classify a column once
    (``catalog.tag(table, column, "pii")``) and one `TagMask` governs every column so
    classified, in every table, including tables added later. An explicit `ColumnMask`
    on a column overrides the tag-derived mask for that column.
    """

    tag: str
    mask: MaskFn
    exempt_roles: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exempt_roles", _frozen(self.exempt_roles))


@dataclass(frozen=True, slots=True)
class RowFilter:
    """Restrict `table` to the rows satisfying ``predicate(principal)``, unless exempt.

    Applied *below* column pruning, so the predicate may reference columns the
    principal cannot itself select — a row-access policy runs with the catalog's
    authority, not the caller's. That is what lets ``region = principal.attrs["region"]``
    work for an analyst who has no `SELECT` on `region`.

    Multiple row filters on one table are conjoined (``AND``): filters restrict, and
    adding one can never widen what a principal sees.
    """

    table: str
    predicate: PredicateFn
    name: str = "row_filter"
    exempt_roles: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exempt_roles", _frozen(self.exempt_roles))
