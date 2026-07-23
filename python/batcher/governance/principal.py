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

from batcher._internal.errors import PlanError
from batcher.governance._validate import reject_bare_string

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
        """Validate, then freeze `roles` and `attrs`.

        Freezing is so a `Principal` handed to the catalog cannot be mutated afterwards
        — a mutable role set would make an authorization decision depend on when it was
        read. Validating is because every mistake here fails *open* rather than loudly:
        ``roles="analyst"`` iterates into the eight single-character roles ``{'a', 'n',
        ...}``, none of which any grant names, so the principal silently sees nothing and
        the catalog looks broken. This is an authorization boundary; it must not guess.

        Raises:
            PlanError: If `name` is not a non-empty string, `roles` is a bare string, or
                any role or attribute is not a string.
        """
        if not isinstance(self.name, str) or not self.name:
            raise PlanError(
                f"A Principal needs a non-empty name, but got "
                f"{type(self.name).__name__} {self.name!r}.",
                hint="The name identifies the principal in audit events.",
            )
        reject_bare_string(
            self.roles,
            what="Principal(roles=...)",
            param="roles",
            reads_as="one role per character",
        )
        roles = frozenset(self.roles)
        bad_roles = sorted(repr(r) for r in roles if not isinstance(r, str))
        if bad_roles:
            raise PlanError(
                f"Principal roles must all be strings, but got {', '.join(bad_roles)}.",
                hint="A role is matched by name against the catalog's grants.",
            )
        attrs = dict(self.attrs)
        bad_attrs = sorted(k for k, v in attrs.items() if not isinstance(v, str))
        if bad_attrs:
            raise PlanError(
                f"Principal attribute values must be strings, but "
                f"{', '.join(repr(k) for k in bad_attrs)} "
                f"{'are' if len(bad_attrs) > 1 else 'is'} not.",
                hint=(
                    "Row filters compare an attribute against a column value, so a "
                    "non-string would never match. Convert it, e.g. attrs={'level': '3'}."
                ),
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "attrs", MappingProxyType(attrs))

    def __repr__(self) -> str:
        """Name, roles, and attributes, in the spelling that reconstructs the principal.

        The generated dataclass repr renders `attrs` as a `mappingproxy(...)`, which is
        neither what the user wrote nor something they can paste back."""
        return f"Principal({self.name!r}, roles={sorted(self.roles)!r}, attrs={dict(self.attrs)!r})"

    def has_role(self, role: str) -> bool:
        """Whether this principal holds `role`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> p = bt.Principal("ana", roles=["analyst"])
                >>> p.has_role("analyst")
                True
                >>> p.has_role("admin")
                False

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> p = bt.Principal("ana", roles=["analyst", "auditor"])
                >>> p.has_any_role(["admin", "auditor"])
                True
                >>> p.has_any_role([])
                False

        Args:
            roles: Role names to test against.

        Returns:
            True if the principal holds any of `roles`. Empty `roles` is False — an
            empty exemption list must exempt nobody, never everybody.
        """
        return not self.roles.isdisjoint(frozenset(roles))
