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

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr, lit

if TYPE_CHECKING:
    from batcher.governance.principal import Principal

__all__ = ["AttributeIn", "MatchesAttribute"]


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

    def __call__(self, principal: Principal) -> Expr:
        raw = _require_attr(principal, self.attribute)
        values = [v for v in (part.strip() for part in raw.split(self.sep)) if v]
        return Col(self.column).is_in(values)


def _require_attr(principal: Principal, name: str) -> str:
    """The principal's `name` attribute, or a clear `PlanError` if it is absent.

    A row-access policy that references an attribute the principal lacks is a
    misconfiguration; failing closed with an actionable message beats silently admitting
    or excluding every row.
    """
    try:
        return principal.attrs[name]
    except KeyError:
        raise PlanError(
            f"row-access policy references attribute {name!r}, which principal "
            f"{principal.name!r} does not have"
        ) from None
