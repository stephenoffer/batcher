"""Input checks for the governance declaration surface.

A `SecurityCatalog` method takes several bare strings and a callable, and every wrong
value fails *open*: a policy stored under an empty or mistyped name matches no table,
role, or tag, so it governs nothing while looking installed. ``select="id"`` is worse —
it iterates into a grant on the columns ``'i'`` and ``'d'``, so the principal sees
nothing and the catalog looks broken rather than misconfigured.

On an authorization boundary "silently does nothing" is the one outcome that must never
be reachable by a typo, so these run at declaration time and name the fix. They live in
their own module because they are shared by `catalog` and would otherwise push it past
the module size limit.
"""

from __future__ import annotations

from collections.abc import Sequence

from batcher._internal.errors import PlanError

__all__ = ["check_callable", "column_set", "policy_name", "reject_bare_string"]


def reject_bare_string(value: object, *, what: str, param: str, reads_as: str) -> None:
    """Reject a string where a sequence of names belongs.

    The single most damaging mistake on this surface, because a string *is* a sequence:
    ``roles="analyst"``, ``select="id"``, ``tables="people"`` and
    ``columns="id"`` are all accepted and silently become one entry per character. The
    resulting policy matches nothing, so the failure reads as "governance is broken"
    rather than "that argument wants a list".

    Args:
        value: The supplied argument.
        what: How to name it, e.g. ``"enforce(tables=...)"``.
        param: The parameter's name, so the hint is the exact fix to type.
        reads_as: What the string would be misread as, e.g. ``"one table per character"``.

    Raises:
        PlanError: If `value` is a string.
    """
    if isinstance(value, str):
        raise PlanError(
            f"{what} needs a sequence of names, but got the string {value!r}, "
            f"which would be read as {reads_as}.",
            hint=f'Wrap it in a list: {param}=["{value}"].',
        )


def policy_name(value: object, what: str) -> str:
    """`value` as a non-empty policy name, or a `PlanError` naming what was wrong.

    Args:
        value: The supplied name.
        what: What it names, e.g. ``"role name"``, used in the message.

    Returns:
        `value` unchanged.

    Raises:
        PlanError: If `value` is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise PlanError(
            f"A {what} must be a non-empty string, but got {type(value).__name__} {value!r}.",
            hint="A policy stored under an unmatched name silently governs nothing.",
        )
    return value


def column_set(select: Sequence[str] | None) -> frozenset[str] | None:
    """`select` as a column set, rejecting a bare string.

    Args:
        select: The column names, or None for every column.

    Returns:
        The columns as a frozenset, or None.

    Raises:
        PlanError: If `select` is a string, which would become one column per character,
            or if any entry is not a usable column name.
    """
    if select is None:
        return None
    reject_bare_string(
        select, what="grant(select=...)", param="select", reads_as="one column per character"
    )
    return frozenset(policy_name(c, "column name") for c in select)


def check_callable(fn: object, what: str, signature: str) -> None:
    """Reject a non-callable policy body at declaration time, not at read time.

    A string passed where a mask or predicate belongs is stored happily and only fails
    when a query reads the governed table — by which point the traceback points at the
    plan rewrite rather than at the line that declared the policy.

    Args:
        fn: The supplied mask or predicate.
        what: How to name the parameter, e.g. ``"mask_tag(mask=...)"``.
        signature: How it is called, e.g. ``"mask(column_expression) -> expression"``.

    Raises:
        PlanError: If `fn` is not callable.
    """
    if not callable(fn):
        raise PlanError(
            f"{what} must be callable, but got {type(fn).__name__} {fn!r}.",
            hint=f"It is called as {signature}.",
        )
