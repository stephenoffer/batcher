"""Declarative, picklable row-filter factories for attribute-based row access.

Like `governance.masks`, these exist so a row-access policy can be *persisted*: a lambda
over `principal.attrs` cannot be pickled, so a platform that stores its policy externally
needs filter definitions that survive a round-trip. Each factory is a small frozen
dataclass that is callable (``Principal -> Expr``) and picklable.

They cover attribute-based row access — restrict rows to those matching a principal's
attribute — which is the dominant row-level-security pattern (a regional analyst sees
only their region). A raw callable remains available for arbitrary in-process predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError, suggestion
from batcher.plan.expr_ir import Col, Expr, lit

if TYPE_CHECKING:
    from batcher.governance.principal import Principal

__all__ = ["AttributeIn", "MatchesAttribute"]


def _check_names(kind: str, column: str, attribute: str) -> None:
    """Reject an empty or non-string column or attribute name.

    An empty column name builds ``Col("")``, which survives policy declaration and plan
    construction and only fails at read time as an unknown column — pointing at the
    governed scan rather than at the policy that is actually wrong.

    Raises:
        PlanError: If either name is not a non-empty string.
    """
    for label, value in (("column", column), ("attribute", attribute)):
        if not isinstance(value, str) or not value:
            raise PlanError(
                f"{kind} needs a non-empty {label} name, but got {type(value).__name__} {value!r}.",
                hint=f"The {label} is matched by name, so it cannot be blank.",
            )


@dataclass(frozen=True, slots=True)
class MatchesAttribute:
    """Keep rows where `column` equals the principal's `attribute`.

    ``MatchesAttribute("region", "region")`` restricts a table to rows whose ``region``
    matches the reading principal's ``region`` attribute — one policy for every regional
    analyst. A principal missing the attribute is a policy error, surfaced clearly.

    Examples:
        .. doctest::

            >>> from batcher.governance import MatchesAttribute, Principal
            >>> analyst = Principal("ana", roles={"analyst"}, attrs={"region": "EU"})
            >>> MatchesAttribute("region", "region")(analyst)
            (col('region') == lit('EU'))
    """

    column: str
    attribute: str

    def __post_init__(self) -> None:
        """Reject an unusable column or attribute name. See `_check_names`."""
        _check_names("MatchesAttribute", self.column, self.attribute)

    def __call__(self, principal: Principal) -> Expr:
        return Col(self.column) == lit(_require_attr(principal, self.attribute))


@dataclass(frozen=True, slots=True)
class AttributeIn:
    """Keep rows where `column` is one of the principal's multi-valued `attribute`.

    The attribute value is split on `sep` (default ``,``) into the allowed set — e.g. a
    principal with ``regions="EU,US"`` sees rows in either. Empty attribute → no rows.

    Examples:
        .. doctest::

            >>> from batcher.governance import AttributeIn, Principal
            >>> analyst = Principal("ana", roles={"analyst"}, attrs={"regions": "EU,US"})
            >>> AttributeIn("region", "regions")(analyst)
            ((col('region') == lit('EU')) | (col('region') == lit('US')))
    """

    column: str
    attribute: str
    sep: str = ","

    def __post_init__(self) -> None:
        """Reject unusable names, and a separator `str.split` cannot use.

        ``sep=""`` raises a bare ``ValueError: empty separator`` from inside `split`,
        at read time, from a policy declared somewhere else entirely.

        Raises:
            PlanError: If a name is empty, or `sep` is not a non-empty string.
        """
        _check_names("AttributeIn", self.column, self.attribute)
        if not isinstance(self.sep, str) or not self.sep:
            raise PlanError(
                f"AttributeIn needs a non-empty separator, but got {self.sep!r}.",
                hint="The default ',' splits an attribute like 'EU,US' into two values.",
            )

    def __call__(self, principal: Principal) -> Expr:
        raw = _require_attr(principal, self.attribute)
        values = [v for v in (part.strip() for part in raw.split(self.sep)) if v]
        return Col(self.column).is_in(values)


def _require_attr(principal: Principal, name: str) -> str:
    """The principal's `name` attribute, or a clear `PlanError` if it is absent.

    A row-access policy that references an attribute the principal lacks is a
    misconfiguration; failing closed with an actionable message beats silently admitting
    or excluding every row. The two ways to reach here are a typo in the policy and a
    principal built without the attribute, so the message shows what the principal *does*
    carry and suggests the nearest name — which distinguishes them at a glance.
    """
    try:
        return principal.attrs[name]
    except KeyError:
        held = sorted(principal.attrs)
        near = suggestion(name, held)
        # Point at the likelier of the two fixes. A near-miss against an attribute the
        # principal *does* hold is almost always a typo in the policy; with nothing
        # close, the principal is the thing that is missing something.
        hint = (
            "Correct the attribute name in the policy."
            if near
            else f"Set it on the principal: Principal({principal.name!r}, attrs={{{name!r}: ...}})."
        )
        raise PlanError(
            f"A row-access policy reads attribute {name!r}, which principal "
            f"{principal.name!r} does not have.",
            suggestion=near,
            available=held,
            available_label="Attributes it does have",
            hint=hint,
        ) from None
