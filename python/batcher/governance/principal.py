"""`Principal` — who is running the query.

The identity every governance decision is made against: a name, the roles it holds,
and free-form attributes for attribute-based row filters (``region``, ``department``,
``clearance``). A principal is immutable and carries no credentials — authentication
happens outside the engine; Batcher only *authorizes*.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = ["Principal"]


@dataclass(frozen=True, slots=True)
class Principal:
    """An identity a query runs as: a name, its roles, and its attributes.

    Attributes drive attribute-based row filters — ``attrs["region"]`` lets one policy
    (``region = principal.attrs["region"]``) serve every regional analyst, rather than
    one policy per region.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> p = bt.Principal("ana", roles=["analyst"], attrs={"region": "EU"})
            >>> p.has_role("analyst"), p.has_role("admin")
            (True, False)
            >>> p.attrs["region"]
            'EU'
    """

    name: str
    roles: frozenset[str] = field(default_factory=frozenset)
    attrs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze `roles` and `attrs` so a `Principal` handed to the catalog cannot be
        mutated afterwards — a mutable role set would make an authorization decision
        depend on when it was read."""
        object.__setattr__(self, "roles", frozenset(self.roles))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))

    def has_role(self, role: str) -> bool:
        """Whether this principal holds `role`.

        Args:
            role: The role name to test.

        Returns:
            True if `role` is among the principal's roles.
        """
        return role in self.roles

    def has_any_role(self, roles: Iterable[str]) -> bool:
        """Whether this principal holds at least one of `roles`.

        The exemption test: a masking or row-access policy is skipped for a principal
        holding any of the policy's exempt roles.

        Args:
            roles: Role names to test against.

        Returns:
            True if the principal holds any of `roles`. Empty `roles` is False — an
            empty exemption list must exempt nobody, never everybody.
        """
        return not self.roles.isdisjoint(frozenset(roles))
