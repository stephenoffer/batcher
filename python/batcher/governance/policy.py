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

__all__ = ["PRIVILEGES", "ColumnMask", "Denial", "Grant", "RowFilter", "TagMask"]

#: The privileges a `Grant` or `Denial` can carry, spelled as SQL spells them — the
#: same four names Snowflake and Unity Catalog use, so a policy ported from either
#: reads the same here.
#:
#: ``SELECT`` is the only one that takes columns. The other three act on whole rows,
#: so there is no such thing as inserting half a row, and a column list against one of
#: them is rejected at declaration rather than silently widened to the table.
PRIVILEGES = ("SELECT", "INSERT", "UPDATE", "DELETE")

#: A column mask: given the column's expression, return the expression to read instead.
MaskFn = Callable[[Expr], Expr]

#: A row filter: given the principal, return the predicate rows must satisfy.
PredicateFn = Callable[[Principal], Expr]


def _frozen(roles: Iterable[str]) -> frozenset[str]:
    return frozenset(roles)


@dataclass(frozen=True, slots=True)
class Grant:
    """`role` holds `privilege` on `columns` of `table` (all columns when `columns` is None).

    The presence of *any* grant on a table switches that table to deny-by-default for
    **every** privilege: a principal then holds exactly the union of what its roles are
    granted, privilege by privilege. A table with no grant at all is ungoverned for access
    (though it may still carry masks and row filters), so installing a catalog does not
    silently lock out every query.

    Granting one privilege therefore does not confer another. A role given ``INSERT``
    cannot ``DELETE``, which is the whole reason to grant ``INSERT`` rather than "write":
    a load job can add today's data and cannot drop yesterday's.

    Examples:
        .. doctest::

            >>> from batcher.governance import Grant
            >>> grant = Grant("analyst", "orders", columns={"order_id", "total"})
            >>> sorted(grant.columns)
            ['order_id', 'total']
            >>> Grant("admin", "orders").columns is None  # every column
            True
            >>> Grant("loader", "orders", privilege="INSERT").privilege
            'INSERT'
    """

    role: str
    table: str
    columns: frozenset[str] | None = None
    #: One of `PRIVILEGES`. Defaults to ``"SELECT"`` so every grant written before
    #: write privileges existed keeps meaning exactly what it meant.
    privilege: str = "SELECT"

    def __post_init__(self) -> None:
        if self.columns is not None:
            object.__setattr__(self, "columns", frozenset(self.columns))
        object.__setattr__(self, "privilege", normalize_privilege(self.privilege, self.columns))


@dataclass(frozen=True, slots=True)
class Denial:
    """`role` is refused `privilege` on `columns` of `table`, whatever it was granted.

    The counterpart to `Grant`, and the reason a catalog needs one: grants union across a
    principal's roles, so a principal holding both ``analyst`` and ``auditor`` sees
    everything either role sees. There is no way to express "everything except `salary`"
    by granting, and enumerating the complement breaks the moment a column is added to the
    table. A denial says it directly, and **wins over every grant**, which is the same
    precedence SQL Server's ``DENY`` and Unity Catalog's ``DENY`` have.

    A denial with ``columns=None`` refuses the privilege on the whole table.

    Examples:
        .. doctest::

            >>> from batcher.governance import Denial
            >>> deny = Denial("contractor", "employees", columns={"salary"})
            >>> deny.privilege, sorted(deny.columns)
            ('SELECT', ['salary'])
    """

    role: str
    table: str
    columns: frozenset[str] | None = None
    #: One of `PRIVILEGES`.
    privilege: str = "SELECT"

    def __post_init__(self) -> None:
        if self.columns is not None:
            object.__setattr__(self, "columns", frozenset(self.columns))
        object.__setattr__(self, "privilege", normalize_privilege(self.privilege, self.columns))


def normalize_privilege(privilege: object, columns: frozenset[str] | None) -> str:
    """`privilege` as one of `PRIVILEGES`, uppercased, or a `PlanError` naming the fix.

    Checked at declaration rather than at the read, because every wrong value here fails
    *open*: a policy stored under ``"select"`` or ``"WRITE"`` matches no privilege the
    engine ever asks about, so it governs nothing while looking installed.

    Args:
        privilege: The privilege the policy names.
        columns: The columns it names, used only to reject a column list on a
            row-level privilege.

    Returns:
        The canonical uppercase spelling.

    Raises:
        PlanError: If `privilege` is not one of `PRIVILEGES`, or a column list was given
            for a privilege that acts on whole rows.
    """
    from batcher._internal.errors import PlanError, unknown_value

    if not isinstance(privilege, str):
        raise unknown_value(PlanError, "privilege", privilege, PRIVILEGES, label="Known privileges")
    folded = privilege.strip().upper()
    if folded not in PRIVILEGES:
        raise unknown_value(PlanError, "privilege", privilege, PRIVILEGES, label="Known privileges")
    if columns is not None and folded != "SELECT":
        raise PlanError(
            f"A column list was given for the {folded} privilege, which acts on whole rows.",
            hint=(
                "Column-level policy applies to SELECT only. Drop the column list to "
                f"govern {folded} on the whole table."
            ),
        )
    return folded


@dataclass(frozen=True, slots=True)
class ColumnMask:
    """Read `table`.`column` through `mask` unless the principal holds an exempt role.

    The mask is applied at the scan, so *everything* downstream — filters, joins,
    aggregates, the final projection — sees only the masked value. A principal cannot
    recover the underlying value by filtering on it or grouping by it.

    Examples:
        .. doctest::

            >>> from batcher import col
            >>> from batcher.governance import ColumnMask, Redact
            >>> policy = ColumnMask("customers", "ssn", Redact(show_last=4))
            >>> policy.mask(col("ssn"))  # doctest: +ELLIPSIS
            when(...).otherwise(col('ssn').cast('string').str.mask('X', 0, 4))
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

    Examples:
        .. doctest::

            >>> from batcher.governance import Nullify, TagMask
            >>> policy = TagMask("pii", Nullify(), exempt_roles={"security"})
            >>> policy.tag
            'pii'
            >>> sorted(policy.exempt_roles)  # these roles read the raw value
            ['security']
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

    Examples:
        .. doctest::

            >>> from batcher.governance import MatchesAttribute, Principal, RowFilter
            >>> policy = RowFilter(
            ...     "orders", MatchesAttribute("region", "region"), name="by_region"
            ... )
            >>> analyst = Principal("ana", roles={"analyst"}, attrs={"region": "EU"})
            >>> policy.predicate(analyst)
            (col('region') == lit('EU'))
    """

    table: str
    predicate: PredicateFn
    name: str = "row_filter"
    exempt_roles: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exempt_roles", _frozen(self.exempt_roles))
